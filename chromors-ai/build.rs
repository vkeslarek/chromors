use std::env;
use std::process::Command;

fn main() {
    // Run the python script to download models if URLs are configured
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let script_path = format!("{}/download_models.py", manifest_dir);

    // Rerun this build script if download_models.py changes
    println!("cargo:rerun-if-changed={}", script_path);

    let mut cmd = Command::new("python3");
    cmd.arg(&script_path);

    let mut has_features = false;
    if env::var("CARGO_FEATURE_SAM2").is_ok() {
        cmd.arg("sam2");
        has_features = true;
    }
    if env::var("CARGO_FEATURE_SAM3").is_ok() {
        cmd.arg("sam3");
        has_features = true;
    }
    if env::var("CARGO_FEATURE_CASCADEPSP").is_ok() {
        cmd.arg("cascadepsp");
        has_features = true;
    }

    if !has_features {
        cmd.arg("none");
    }

    let status = cmd.status();

    if let Ok(exit_status) = status {
        if !exit_status.success() {
            println!("cargo:warning=Failed to run download_models.py successfully.");
        }
    } else {
        println!("cargo:warning=Python3 not found or failed to execute download_models.py");
    }
}
