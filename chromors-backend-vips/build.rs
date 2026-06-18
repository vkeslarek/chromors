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
}
