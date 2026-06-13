use std::fs;
use std::path::{Path, PathBuf};

/// Finds all `.rs` files in the given directory recursively.
fn find_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_rs_files(&path));
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn test_agnostic_half_is_pure() {
    let src_dir = Path::new("src");
    if !src_dir.exists() {
        return; // Skip if run from wrong directory
    }

    let files = find_rs_files(src_dir);

    // The "Agnostic" files that should NEVER know about backends.
    let agnostic_modules = [
        "src/node.rs",
        "src/work_unit.rs",
        "src/kind.rs",
        "src/io.rs",
        "src/buffer.rs",
    ];

    let forbidden_words = [
        "Vips",
        "Gpu",
        "Slang",
        "slang_",
        "View",
        "ParamBlock",
        "libvips",
        "wgpu",
    ];

    for path in files {
        let path_str = path.to_string_lossy().replace('\\', "/");

        // Is this an agnostic file?
        if agnostic_modules.iter().any(|m| path_str.ends_with(m)) {
            let content = fs::read_to_string(&path).unwrap();
            for line in content.lines() {
                // Ignore comments
                if line.trim().starts_with("//") {
                    continue;
                }
                for word in forbidden_words {
                    if line.contains(word) {
                        // View is a bit tricky, might be part of "viewport" or something.
                        // Let's be strict but avoid false positives if needed.
                        if word == "View" && line.contains("Viewport") {
                            continue;
                        }
                        panic!(
                            "ARCH VIOLATION in {}:\nLine contains forbidden backend-specific term '{}':\n{}",
                            path_str, word, line
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_operations_are_agnostic() {
    let src_dir = Path::new("src/operation");
    if !src_dir.exists() {
        return;
    }

    // Operations can have `Lower<GpuBackend>` and `Lower<VipsBackend>` impls,
    // BUT the structural part (`Operation<B>`) must not hardcode backends.
    // However, since we put `Lower` impls in the same file as `Operation` for now,
    // it's acceptable for them to mention Gpu/Vips.
    // The key invariant is: no loose graph traversals.

    let files = find_rs_files(src_dir);
    for path in files {
        let content = fs::read_to_string(&path).unwrap();
        // Operations should not do loose walks.
        if content.contains("demand_walk") || content.contains("lower_walk") {
            panic!(
                "ARCH VIOLATION in {}: Operations must not implement or call loose graph traversals.",
                path.display()
            );
        }
    }
}

#[test]
fn test_no_loose_matches_on_node() {
    let src_dir = Path::new("src/backend");
    if !src_dir.exists() {
        return;
    }

    let files = find_rs_files(src_dir);
    for path in files {
        let content = fs::read_to_string(&path).unwrap();
        if content.contains("match &**node") || content.contains("Node::Op(") {
            panic!(
                "ARCH VIOLATION in {}: Backends must not match on Node::Op or Node::Source! Use delegated methods like node.lower()",
                path.display()
            );
        }
    }
}

#[test]
fn test_backends_are_datatype_agnostic() {
    let src_dir = Path::new("src/backend");
    if !src_dir.exists() {
        return;
    }

    let files = find_rs_files(src_dir);
    for path in files {
        let path_str = path.to_string_lossy().replace('\\', "/");
        // We allow FFI boundary files like custom.rs or region.rs to import Image2D since
        // libvips C-API explicitly requires VipsImage mappings, but pure backend logic must not.
        if path_str.ends_with("custom.rs") || path_str.ends_with("region.rs") {
            continue;
        }

        let content = fs::read_to_string(&path).unwrap();

        // Ensure no datatype-specific Sources or Targets are placed in the generic backend directory.
        // They must reside in `src/data/<datatype>.rs` because they are specific to a Kind!
        if content.contains("struct VipsImageSource")
            || content.contains("struct GpuImageSource")
            || content.contains("ImageSource")
        {
            panic!(
                "ARCH VIOLATION in {}: Datatype-specific Sources (like VipsImageSource) belong in src/data/, NOT in src/backend/!",
                path.display()
            );
        }
    }
}
