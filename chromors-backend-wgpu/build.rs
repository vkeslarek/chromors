use std::env;
use std::path::PathBuf;

fn main() {
    if let Some((slang_include, slang_lib)) = find_slang_sdk() {
        println!("cargo:rustc-link-search=native={}", slang_lib.display());
        println!("cargo:rustc-link-lib=dylib=slang-compiler");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", slang_lib.display());

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

        println!("cargo:rerun-if-changed=../cpp/wrapper_slang.h");
        let slang_bindings = bindgen::Builder::default()
            .header("../cpp/wrapper_slang.h")
            .clang_arg(format!("-I{}", slang_include.display()))
            .clang_arg("-x").clang_arg("c++").clang_arg("-std=c++17")
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
            .cpp(true).std("c++17")
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
    None
}
