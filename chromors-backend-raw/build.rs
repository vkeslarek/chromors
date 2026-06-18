use std::env;
use std::path::PathBuf;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

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
}
