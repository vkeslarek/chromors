#include "slang.h"
#include "slang_wrapper.h"
#include <cstring>
#include <cstdio>

using namespace slang;

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static char* dup_diag(IBlob* blob) {
    if (!blob) return nullptr;
    const char* msg = static_cast<const char*>(blob->getBufferPointer());
    size_t      len = blob->getBufferSize();
    if (!msg || len == 0) return nullptr;
    char* out = new char[len + 1];
    std::memcpy(out, msg, len);
    out[len] = '\0';
    return out;
}

static char* make_error(const char* msg) {
    size_t len = std::strlen(msg);
    char*  out = new char[len + 1];
    std::memcpy(out, msg, len + 1);
    return out;
}

static SlangOptimizationLevel map_opt(SlangwOptLevel lvl) {
    switch (lvl) {
        case SLANGW_OPT_NONE:    return SLANG_OPTIMIZATION_LEVEL_NONE;
        case SLANGW_OPT_DEFAULT: return SLANG_OPTIMIZATION_LEVEL_DEFAULT;
        case SLANGW_OPT_HIGH:    return SLANG_OPTIMIZATION_LEVEL_HIGH;
        default:                 return SLANG_OPTIMIZATION_LEVEL_MAXIMAL;
    }
}

/* --------------------------------------------------------------------------
 * Global session
 * -------------------------------------------------------------------------- */

extern "C" void* slangw_create_global_session(void) {
    try {
        SlangGlobalSessionDesc desc = {};
        IGlobalSession* gs = nullptr;
        SlangResult res = slang_createGlobalSession2(&desc, &gs);
        if (SLANG_FAILED(res) || !gs) return nullptr;
        return gs;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void slangw_global_session_release(void* gs) {
    if (gs) static_cast<IGlobalSession*>(gs)->release();
}

/* --------------------------------------------------------------------------
 * Per-compilation session
 * -------------------------------------------------------------------------- */

extern "C" void* slangw_create_session(
    void*              global_session,
    const char* const* search_paths,
    int                search_path_count,
    SlangwOptLevel     opt_level)
{
    try {
        auto* gs = static_cast<IGlobalSession*>(global_session);
        if (!gs) return nullptr;

        TargetDesc target = {};
        target.structureSize = sizeof(TargetDesc);
        target.format        = SLANG_SPIRV;
        target.flags         = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

        CompilerOptionEntry options[3] = {};
        options[0].name            = CompilerOptionName::VulkanUseEntryPointName;
        options[0].value.intValue0 = 1;
        options[1].name            = CompilerOptionName::GLSLForceScalarLayout;
        options[1].value.intValue0 = 1;
        options[2].name            = CompilerOptionName::Optimization;
        options[2].value.intValue0 = static_cast<int>(map_opt(opt_level));

        SessionDesc desc = {};
        desc.structureSize            = sizeof(SessionDesc);
        desc.targets                  = &target;
        desc.targetCount              = 1;
        desc.searchPaths              = search_paths;
        desc.searchPathCount          = search_path_count;
        desc.compilerOptionEntries    = options;
        desc.compilerOptionEntryCount = 3;

        ISession* session = nullptr;
        SlangResult res = gs->createSession(desc, &session);
        if (SLANG_FAILED(res) || !session) return nullptr;
        return session;
    } catch (...) {
        return nullptr;
    }
}

/* --------------------------------------------------------------------------
 * Combined load + compile  (module never crosses the FFI boundary)
 * --------------------------------------------------------------------------
 *
 * ISession owns all modules it loads.  loadModuleFromSource returns a
 * borrowed pointer into the session's internal cache — it is NOT addref'd
 * for the caller.  We must NOT call release() on it; the session frees it
 * when the session itself is released.  Keeping the module internal to this
 * function ensures no RAII wrapper on the Rust side can accidentally release
 * the borrowed pointer and corrupt the session's heap.
 * -------------------------------------------------------------------------- */

extern "C" int slangw_compile_to_spirv(
    void*        s,
    const char*  name,
    const char*  source,
    size_t       source_len,
    int          target_idx,
    const void** out_code,
    size_t*      out_size,
    char**       out_diag)
{
    *out_code = nullptr;
    *out_size = 0;
    if (out_diag) *out_diag = nullptr;

    try {
        auto* session = static_cast<ISession*>(s);
        if (!session) {
            if (out_diag) *out_diag = make_error("null session");
            return -1;
        }
        if (source_len == 0) source_len = std::strlen(source);

        /* --- Load module (borrowed pointer — do NOT release) --- */
        IBlob* src_blob = slang_createBlob(source, source_len);
        if (!src_blob) {
            if (out_diag) *out_diag = make_error("slang_createBlob returned null");
            return -2;
        }
        IBlob* load_diag = nullptr;
        /* loadModuleFromSource: borrowed ref, owned by session.
           Use `name` as the path too — Slang indexes the dictionary by path,
           so reusing a fixed path (e.g. "pixors.slang") causes "key already
           exists" on the second call even when the module name is unique. */
        IModule* module = session->loadModuleFromSource(name, name, src_blob, &load_diag);
        src_blob->release();

        if (!module) {
            if (out_diag) {
                *out_diag = load_diag
                    ? dup_diag(load_diag)
                    : make_error("loadModuleFromSource returned null");
            }
            if (load_diag) load_diag->release();
            return -3;
        }
        if (load_diag) { load_diag->release(); load_diag = nullptr; }

        /* --- Collect entry points --- */
        SlangInt ep_count = module->getDefinedEntryPointCount();
        if (ep_count == 0) {
            if (out_diag) *out_diag = make_error("module has no defined entry points");
            return -4;
        }

        IComponentType** components = new IComponentType*[ep_count + 1];
        components[0] = module;  /* borrowed — not addref'd for us */
        for (SlangInt i = 0; i < ep_count; ++i) {
            IEntryPoint* ep = nullptr;
            if (SLANG_FAILED(module->getDefinedEntryPoint(i, &ep)) || !ep) {
                for (SlangInt j = 0; j < i; ++j)
                    if (components[j + 1]) components[j + 1]->release();
                delete[] components;
                if (out_diag) *out_diag = make_error("getDefinedEntryPoint failed");
                return -5;
            }
            components[i + 1] = ep;  /* owned — addref'd by getDefinedEntryPoint */
        }

        /* Release our entry-point references after composite is created. */
        auto release_eps = [&]() {
            for (SlangInt i = 0; i < ep_count; ++i)
                if (components[i + 1]) components[i + 1]->release();
            delete[] components;
        };

        /* --- Create composite --- */
        IBlob* diag = nullptr;
        IComponentType* composite = nullptr;
        SlangResult res = session->createCompositeComponentType(
            components, ep_count + 1, &composite, &diag);
        release_eps();  /* release our EP refs; composite holds its own */

        if (SLANG_FAILED(res) || !composite) {
            if (out_diag) *out_diag = diag ? dup_diag(diag) : make_error("createCompositeComponentType failed");
            if (diag) diag->release();
            return -6;
        }
        if (diag) { diag->release(); diag = nullptr; }

        /* --- Link --- */
        IComponentType* linked = nullptr;
        res = composite->link(&linked, &diag);
        composite->release();

        if (SLANG_FAILED(res) || !linked) {
            if (out_diag) *out_diag = diag ? dup_diag(diag) : make_error("link failed");
            if (diag) diag->release();
            return -7;
        }
        if (diag) { diag->release(); diag = nullptr; }

        /* --- Extract SPIR-V --- */
        IBlob* code = nullptr;
        res = linked->getTargetCode(target_idx, &code, &diag);
        linked->release();

        if (SLANG_FAILED(res) || !code) {
            if (out_diag) *out_diag = diag ? dup_diag(diag) : make_error("getTargetCode failed");
            if (diag) diag->release();
            return -8;
        }
        if (diag) diag->release();

        size_t      sz  = code->getBufferSize();
        const void* ptr = code->getBufferPointer();
        char*       buf = new char[sz];
        std::memcpy(buf, ptr, sz);
        code->release();

        *out_code = buf;
        *out_size = sz;
        return 0;
    } catch (...) {
        if (out_diag) *out_diag = make_error("exception in slangw_compile_to_spirv");
        return -9;
    }
}

/* --------------------------------------------------------------------------
 * Memory management
 * -------------------------------------------------------------------------- */

extern "C" void slangw_free_buffer(void* ptr) {
    delete[] static_cast<char*>(ptr);
}

extern "C" void slangw_free_string(char* ptr) {
    delete[] ptr;
}

extern "C" void slangw_release(void* obj) {
    if (obj) static_cast<ISlangUnknown*>(obj)->release();
}

extern "C" void slangw_shutdown(void) {
    slang_shutdown();
}
