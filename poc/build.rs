use std::env;
use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let vips = pkg_config::Config::new()
        .atleast_version("8.6")
        .probe("vips")
        .expect("libvips not found via pkg-config");

    let bindings = bindgen::Builder::default()
        .header("../cpp/wrapper.h")
        .clang_args(
            vips.include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("(vips_.*|g_free|g_strfreev|g_object_ref|g_object_unref|g_object_get_property|g_object_set_property|g_value_init|g_value_unset|g_value_dup_object|g_value_set_object|g_value_set_boxed|g_value_set_double|g_value_set_int|g_value_set_string|g_value_set_boolean|g_value_get_int|g_value_get_double|g_value_get_string|g_value_get_boolean)")
        .allowlist_type("(Vips.*|GValue|GObject|GType|G_TYPE_.*)")
        .allowlist_var("(VIPS_.*|G_TYPE_.*)")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed");

    bindings
        .write_to_file(out_path.join("ffi.rs"))
        .expect("failed to write bindings");

    println!("cargo:rerun-if-changed=../cpp/wrapper.h");

    // ── LibRaw C API bindings ───────────────────────────────────────────────────
    let libraw = pkg_config::Config::new()
        .atleast_version("0.19")
        .probe("libraw")
        .expect("libraw not found via pkg-config; install libraw-dev");
    let libraw_bindings = bindgen::Builder::default()
        .header("../cpp/wrapper_libraw.h")
        .clang_args(
            libraw
                .include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("libraw_.*")
        .allowlist_type("libraw_.*")
        .allowlist_var("LIBRAW_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("libraw bindgen failed");
    libraw_bindings
        .write_to_file(out_path.join("libraw_ffi.rs"))
        .expect("failed to write libraw bindings");
    println!("cargo:rerun-if-changed=../cpp/wrapper_libraw.h");

    if let Some((slang_include, slang_lib)) = find_slang_sdk() {
        println!("cargo:rustc-link-search=native={}", slang_lib.display());
        println!("cargo:rustc-link-lib=dylib=slang-compiler");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", slang_lib.display());
        println!("cargo:rerun-if-changed=../cpp/wrapper_slang.h");

        let slang_bindings = bindgen::Builder::default()
            .header("../cpp/wrapper_slang.h")
            .clang_arg(format!("-I{}", slang_include.display()))
            .clang_arg("-x")
            .clang_arg("c++")
            .clang_arg("-std=c++17")
            .allowlist_function("slang_.*")
            .allowlist_type("(Slang.*|ISlang.*|IBlob)")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("slang bindgen failed");
        slang_bindings
            .write_to_file(out_path.join("slang_ffi.rs"))
            .expect("failed to write slang bindings");

        println!("cargo:rerun-if-changed=../cpp/slang_wrapper.cpp");
        println!("cargo:rerun-if-changed=../cpp/slang_wrapper.h");

        cc::Build::new()
            .file("../cpp/slang_wrapper.cpp")
            .include(slang_include.to_str().unwrap())
            .cpp(true)
            .std("c++17")
            .compile("slang_wrapper");

        let wrapper_bindings = bindgen::Builder::default()
            .header("../cpp/slang_wrapper.h")
            .allowlist_function("slangw_.*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("slang wrapper bindgen failed");
        wrapper_bindings
            .write_to_file(out_path.join("slang_wrapper_ffi.rs"))
            .expect("failed to write slang wrapper bindings");
    }
}

fn find_slang_sdk() -> Option<(PathBuf, PathBuf)> {
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        home.to_string() + "/Downloads/slang-2026.8-linux-x86_64-glibc-2.27",
        home + "/Downloads/slang",
    ];
    for root in &candidates {
        let include = PathBuf::from(root).join("include");
        let lib = PathBuf::from(root).join("lib");
        if include.join("slang.h").exists() && lib.join("libslang-compiler.so").exists() {
            return Some((include, lib));
        }
    }
    if PathBuf::from("/usr/include/slang.h").exists()
        && PathBuf::from("/usr/lib/libslang-compiler.so").exists()
    {
        return Some((PathBuf::from("/usr/include"), PathBuf::from("/usr/lib")));
    }
    if PathBuf::from("/usr/local/include/slang.h").exists()
        && PathBuf::from("/usr/local/lib/libslang-compiler.so").exists()
    {
        return Some((
            PathBuf::from("/usr/local/include"),
            PathBuf::from("/usr/local/lib"),
        ));
    }
    None
}
