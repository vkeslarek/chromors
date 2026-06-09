use std::ffi::CString;
use std::path::PathBuf;
use std::ptr::NonNull;

use crate::slang_wrapper_ffi::{
    slangw_compile_to_spirv, slangw_create_global_session, slangw_create_session,
    slangw_free_buffer, slangw_free_string, slangw_global_session_release, slangw_release,
};

// Mirrors SLANGW_OPT_MAXIMAL from slang_wrapper.h (value 3).
const OPT_MAXIMAL: u32 = 3;

/* --------------------------------------------------------------------------
 * RAII wrappers
 * -------------------------------------------------------------------------- */

struct GlobalSession(NonNull<std::ffi::c_void>);
unsafe impl Send for GlobalSession {}
impl Drop for GlobalSession {
    fn drop(&mut self) {
        unsafe { slangw_global_session_release(self.0.as_ptr()) };
    }
}

struct Session(NonNull<std::ffi::c_void>);
unsafe impl Send for Session {}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe { slangw_release(self.0.as_ptr()) };
    }
}

struct SpirvBuf(*const std::ffi::c_void, usize);
impl Drop for SpirvBuf {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { slangw_free_buffer(self.0 as *mut std::ffi::c_void) };
        }
    }
}
impl SpirvBuf {
    fn as_bytes(&self) -> &[u8] {
        if self.0.is_null() || self.1 == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.0 as *const u8, self.1) }
    }
}

struct DiagString(*mut std::os::raw::c_char);
impl Drop for DiagString {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { slangw_free_string(self.0) };
        }
    }
}
impl DiagString {
    fn as_str(&self) -> &str {
        if self.0.is_null() {
            return "(no diagnostic)";
        }
        unsafe { std::ffi::CStr::from_ptr(self.0) }
            .to_str()
            .unwrap_or("(invalid utf-8 in diagnostic)")
    }
}

/* --------------------------------------------------------------------------
 * Global state
 *
 * All Slang API calls (createSession, loadModuleFromSource, link,
 * getTargetCode) are serialised under ONE mutex.  Slang sessions created
 * from the same IGlobalSession share internal state — calling these APIs
 * concurrently from different threads causes Slang::InternalError / SIGSEGV.
 *
 * The persistent Session avoids re-creating the compiler infrastructure
 * (~10-50 ms) on each call.  It is stored here so the session outlives the
 * per-call SlangCompiler instances created by compile.rs.
 *
 * Only spirv-opt runs outside this lock (pure CPU, no Slang state).
 * -------------------------------------------------------------------------- */

struct SlangState {
    _global: GlobalSession, // must outlive session
    session: Session,
}

static SLANG: std::sync::Mutex<Option<SlangState>> = std::sync::Mutex::new(None);

/* --------------------------------------------------------------------------
 * SlangCompiler — public entry point
 * -------------------------------------------------------------------------- */

pub struct SlangCompiler {
    pub shader_dir: PathBuf,
    pub out_dir: PathBuf,
}

impl SlangCompiler {
    pub fn new(shader_dir: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            shader_dir,
            out_dir,
        }
    }

    pub fn compile_ir(&self, source_text: &str, hash_val: u64) -> Result<Vec<u8>, String> {
        let _ = std::fs::create_dir_all(&self.out_dir);
        let opt_path = self.out_dir.join(format!("{hash_val:016x}.opt.spv"));

        // Fast path: disk cache — no lock needed.
        if opt_path.exists()
            && let Ok(b) = std::fs::read(&opt_path)
        {
            return Ok(b);
        }

        let c_paths: Vec<CString> = [self.shader_dir.as_path(), self.out_dir.as_path()]
            .iter()
            .filter_map(|p| CString::new(p.as_os_str().as_encoded_bytes()).ok())
            .collect();
        let c_path_ptrs: Vec<*const std::os::raw::c_char> =
            c_paths.iter().map(|p| p.as_ptr()).collect();

        // Each call gets a unique module name. Slang's session dictionary rejects
        // duplicate names, so using the hash (which repeats across calls for the
        // same shader) would cause "key already exists" after the first load.
        // The disk cache is the true deduplication layer — unique names here are safe.
        static MODULE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let call_id = MODULE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name_c = CString::new(format!("m{call_id:016x}")).unwrap();
        let src_c =
            CString::new(source_text).map_err(|_| "source text contains null byte".to_string())?;

        // Acquire the global Slang lock for ALL Slang operations.
        // Released before spirv-opt so optimisation can run in parallel.
        let spirv_bytes = {
            let mut guard = SLANG.lock().unwrap();

            // Double-check disk cache while holding the lock.
            if opt_path.exists()
                && let Ok(b) = std::fs::read(&opt_path)
            {
                return Ok(b);
            }

            // Lazy-init global session + compilation session.
            if guard.is_none() {
                let gs_raw = unsafe { slangw_create_global_session() };
                let global = NonNull::new(gs_raw)
                    .map(GlobalSession)
                    .ok_or_else(|| "slangw_create_global_session returned null".to_string())?;

                let sess_raw = unsafe {
                    slangw_create_session(
                        global.0.as_ptr(),
                        c_path_ptrs.as_ptr(),
                        c_path_ptrs.len() as i32,
                        OPT_MAXIMAL,
                    )
                };
                let session = NonNull::new(sess_raw)
                    .map(Session)
                    .ok_or_else(|| "slangw_create_session returned null".to_string())?;

                *guard = Some(SlangState {
                    _global: global,
                    session,
                });
            }

            let state = guard.as_ref().unwrap();
            let mut out_code: *const std::ffi::c_void = std::ptr::null();
            let mut out_size: usize = 0;
            let mut diag_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();

            let _sw = crate::utils::Stopwatch::new("gpu.compile.slang");
            let rc = unsafe {
                slangw_compile_to_spirv(
                    state.session.0.as_ptr(),
                    name_c.as_ptr(),
                    src_c.as_ptr(),
                    source_text.len(),
                    0,
                    &mut out_code,
                    &mut out_size,
                    &mut diag_ptr,
                )
            };
            drop(_sw);

            let diag = DiagString(diag_ptr);
            if rc != 0 {
                return Err(format!(
                    "slangw_compile_to_spirv failed (rc={}): {}",
                    rc,
                    diag.as_str()
                ));
            }

            let raw = SpirvBuf(out_code, out_size).as_bytes().to_vec();

            // Write raw (unoptimized) SPIRV to disk BEFORE releasing the lock.
            // Threads that acquire the lock next will find the file in their
            // double-check and return early — preventing "key already exists"
            // errors from re-loading the same module name into the session.
            // spirv-opt will overwrite with the optimized version outside the lock.
            static TMP_COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = self.out_dir.join(format!("{hash_val:016x}_{n}.opt.spv"));
            if std::fs::write(&tmp, &raw).is_ok() {
                let _ = std::fs::rename(&tmp, &opt_path);
            }

            raw
        }; // SLANG lock released here — spirv-opt runs in parallel

        // Optimise and overwrite the raw cache entry.
        let optimized = {
            let _sw = crate::utils::Stopwatch::new("gpu.compile.opt");
            Self::opt_spirv(spirv_bytes, hash_val)
        };

        // Atomic overwrite with optimized version.
        {
            static OPT_COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = OPT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = self.out_dir.join(format!("{hash_val:016x}_opt_{n}.spv"));
            if std::fs::write(&tmp, &optimized).is_ok() {
                let _ = std::fs::rename(&tmp, &opt_path);
            }
        }

        Ok(optimized)
    }

    fn opt_spirv(spirv: Vec<u8>, hash_val: u64) -> Vec<u8> {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        use spirv_tools::opt::Optimizer;
        let mut opt = spirv_tools::opt::create(None);
        opt.register_performance_passes();
        opt.register_pass(spirv_tools::opt::Passes::StripDebugInfo);

        let mut logger = |msg: spirv_tools::error::Message| {
            tracing::warn!(target: "gpu_compile", "spirv-opt warning: {:?}", msg.message);
        };

        match opt.optimize(words, &mut logger, None) {
            Ok(bin) => bin.as_bytes().to_vec(),
            Err(e) => {
                tracing::warn!(
                    target: "gpu_compile",
                    "spirv-opt failed for hash={:016x}: {:?} — using unoptimized",
                    hash_val, e
                );
                spirv
            }
        }
    }
}
