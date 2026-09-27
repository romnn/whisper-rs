#![allow(clippy::uninlined_format_args)]

extern crate bindgen;
extern crate semver;

use cmake::Config;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap();
    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }
    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        #[cfg(feature = "coreml")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=CoreML");
        }
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    #[cfg(feature = "coreml")]
    println!("cargo:rustc-link-lib=static=whisper.coreml");

    #[cfg(feature = "openblas")]
    {
        if let Ok(openblas_path) = env::var("OPENBLAS_PATH") {
            println!(
                "cargo::rustc-link-search={}",
                PathBuf::from(openblas_path).join("lib").display()
            );
        }
        if cfg!(windows) {
            println!("cargo:rustc-link-lib=libopenblas");
        } else {
            println!("cargo:rustc-link-lib=openblas");
        }
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
                println!("cargo:rustc-link-search={}", cuda_path.display());
            } else {
                println!("cargo:rustc-link-lib=culibos");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
                println!("cargo:rustc-link-search=/opt/cuda/lib64");
                println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
            }
        }
    }
    #[cfg(feature = "hipblas")]
    {
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/whisper.cpp/issues/2202 for more details.")
            } else {
                println!("cargo:rerun-if-env-changed=HIP_PATH");

                let hip_path = match env::var("HIP_PATH") {
                    Ok(path) =>PathBuf::from(path),
                    Err(_) => PathBuf::from("/opt/rocm"),
                };
                let hip_lib_path = hip_path.join("lib");

                println!("cargo:rustc-link-search={}",hip_lib_path.display());
            }
        }
    }

    #[cfg(feature = "openmp")]
    {
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=gomp");
        } else if target.contains("apple") {
            println!("cargo:rustc-link-lib=omp");
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
        }
    }

    println!("cargo:rerun-if-changed=wrapper.h");
    // The version is declared here, so a submodule bump always changes this file.
    println!("cargo:rerun-if-changed=whisper.cpp/CMakeLists.txt");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = out.join("whisper.cpp");

    // The copy is reused across builds, so it is refreshed whenever the submodule moves;
    // otherwise CMake would keep compiling the previous release while bindgen reads the new
    // headers.
    let source_stamp = whisper_source_stamp(Path::new("./whisper.cpp"));
    let stamp_path = out.join("whisper.cpp.stamp");
    let copy_is_current = whisper_root.exists()
        && std::fs::read_to_string(&stamp_path).is_ok_and(|recorded| recorded == source_stamp);
    if !copy_is_current {
        if whisper_root.exists() {
            std::fs::remove_dir_all(&whisper_root).unwrap_or_else(|e| {
                panic!(
                    "Failed to clear stale whisper sources in {}: {}",
                    whisper_root.display(),
                    e
                )
            });
        }
        std::fs::create_dir_all(&whisper_root).unwrap();
        fs_extra::dir::copy("./whisper.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy whisper sources into {}: {}",
                whisper_root.display(),
                e
            )
        });
        std::fs::write(&stamp_path, &source_stamp)
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", stamp_path.display(), e));
    }

    if env::var("WHISPER_DONT_GENERATE_BINDINGS").is_ok() {
        let _: u64 = std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
            .expect("Failed to copy bindings.rs");
    } else {
        // https://github.com/rust-lang/rust-bindgen/issues/2691
        // https://github.com/rust-lang/rust-bindgen/issues/3264
        // https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/issues/779
        let package_msrv = match option_env!("CARGO_PKG_RUST_VERSION") {
            Some(v) if !v.is_empty() => {
                let v = semver::Version::parse(v).expect("Invalid CARGO_PKG_RUST_VERSION");
                bindgen::RustTarget::stable(v.minor, v.patch)
            }
            // as_chunks in utilities.rs requires 1.88+
            _ => bindgen::RustTarget::stable(88, 0),
        }
        .map_err(|v| v.to_string())
        .unwrap();

        let bindings = bindgen::Builder::default()
            .rust_edition(bindgen::RustEdition::Edition2021)
            .rust_target(package_msrv)
            .header("wrapper.h");

        #[cfg(feature = "metal")]
        let bindings = bindings.header("whisper.cpp/ggml/include/ggml-metal.h");
        #[cfg(feature = "vulkan")]
        let bindings = bindings
            .header("whisper.cpp/ggml/include/ggml-vulkan.h")
            .clang_arg("-DGGML_USE_VULKAN=1");

        let bindings = bindings
            .clang_arg("-I./whisper.cpp/")
            .clang_arg("-I./whisper.cpp/include")
            .clang_arg("-I./whisper.cpp/ggml/include")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate();

        match bindings {
            Ok(b) => {
                let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
                b.write_to_file(out_path.join("bindings.rs"))
                    .expect("Couldn't write bindings!");
            }
            Err(e) => {
                println!("cargo:warning=Unable to generate bindings: {}", e);
                println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
                // copy src/bindings.rs to OUT_DIR
                std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
                    .expect("Unable to copy bindings.rs");
            }
        }
    };

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let mut config = Config::new(&whisper_root);

    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .very_verbose(true)
        .pic(true);

    if cfg!(feature = "system-ggml") {
        // Share a single externally-built ggml with llama-cpp-sys-2 instead of
        // building this crate's own bundled copy. `ggml-sys` (declared as an
        // optional `[dependencies]` entry, activated by this feature) installs
        // ggml under its own OUT_DIR and exports the path via
        // `cargo:cmake_prefix_path=…`. Cargo surfaces that to this build script
        // as `DEP_GGML_CMAKE_PREFIX_PATH`. Pointing CMAKE_PREFIX_PATH at it lets
        // whisper.cpp's `find_package(ggml REQUIRED)` (triggered by
        // WHISPER_USE_SYSTEM_GGML=ON) resolve against the shared build.
        config.define("WHISPER_USE_SYSTEM_GGML", "ON");
        let ggml_prefix = std::env::var("DEP_GGML_CMAKE_PREFIX_PATH").unwrap_or_else(|_| {
            panic!(
                "system-ggml feature is enabled but DEP_GGML_CMAKE_PREFIX_PATH is not set.\n\
                 This means ggml-sys is not in the dependency graph. Did you forget to\n\
                 enable the feature in the consuming workspace (e.g. via [patch.crates-io]\n\
                 redirecting ggml-sys to a workspace-local crate)?"
            )
        });
        config.define("CMAKE_PREFIX_PATH", &ggml_prefix);
        println!("cargo:rerun-if-env-changed=DEP_GGML_INCLUDE");
        println!("cargo:rerun-if-env-changed=DEP_GGML_CMAKE_PREFIX_PATH");
        println!("cargo:rerun-if-env-changed=DEP_GGML_GGML_REV");

        // Soft check for ggml SHA drift between the linked libggml and the headers
        // bindgen reads from. Emits a `cargo:warning=` (not a panic) because
        // ggml's exposed C API is API/ABI-stable across small drift windows —
        // we still want to know if the gap grows.
        let bundled_sync = std::path::Path::new("whisper.cpp/scripts/sync-ggml.last");
        if let Ok(bundled_sha) = std::fs::read_to_string(bundled_sync) {
            let bundled_sha = bundled_sha.trim();
            if let Ok(expected_sha) = std::env::var("DEP_GGML_GGML_REV") {
                if bundled_sha != expected_sha {
                    println!(
                        "cargo:warning=ggml SHA drift: ggml-sys is pinned at {expected_sha} but \
                         this fork's bundled whisper.cpp/ggml synced from {bundled_sha}. The \
                         linked libggml comes from ggml-sys's SHA, while bindgen reads bundled \
                         headers. Safe only if ggml's C API is binary-compatible across this \
                         range. Bump the whisper.cpp submodule to a commit whose \
                         scripts/sync-ggml.last matches ggml-sys's pin to make this exact."
                    );
                }
            }
        }
    }

    if cfg!(target_os = "windows") {
        config.cxxflag("/utf-8");
        println!("cargo:rustc-link-lib=advapi32");
    }

    if cfg!(feature = "coreml") {
        config.define("WHISPER_COREML", "ON");
        config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        config.define("CMAKE_CUDA_FLAGS", "-Xcompiler=-fPIC");
    }

    if cfg!(feature = "hipblas") {
        config.define("GGML_HIP", "ON");
        config.define("CMAKE_C_COMPILER", "hipcc");
        config.define("CMAKE_CXX_COMPILER", "hipcc");
        println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
        if let Ok(gpu_targets) = env::var("AMDGPU_TARGETS") {
            config.define("AMDGPU_TARGETS", gpu_targets);
        }
    }

    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        if cfg!(windows) {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan-1");
            let vulkan_path = match env::var("VULKAN_SDK") {
                Ok(path) => PathBuf::from(path),
                Err(_) => panic!(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                ),
            };
            let vulkan_lib_path = vulkan_path.join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else if cfg!(target_os = "macos") {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan");
            let vulkan_path = match env::var("VULKAN_SDK") {
                Ok(path) => PathBuf::from(path),
                Err(_) => panic!(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                ),
            };
            let vulkan_lib_path = vulkan_path.join("lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "openblas") {
        config.define("GGML_BLAS", "ON");
        config.define("GGML_BLAS_VENDOR", "OpenBLAS");
        if env::var("BLAS_INCLUDE_DIRS").is_err() {
            panic!("BLAS_INCLUDE_DIRS environment variable must be set when using OpenBLAS");
        }
        config.define("BLAS_INCLUDE_DIRS", env::var("BLAS_INCLUDE_DIRS").unwrap());
        println!("cargo:rerun-if-env-changed=BLAS_INCLUDE_DIRS");
    }

    if cfg!(feature = "metal") {
        config.define("GGML_METAL", "ON");
        config.define("GGML_METAL_NDEBUG", "ON");
        config.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        // Metal is enabled by default, so we need to explicitly disable it
        config.define("GGML_METAL", "OFF");
    }

    if cfg!(debug_assertions) || cfg!(feature = "force-debug") {
        // debug builds are too slow to even remotely be usable,
        // so we build with optimizations even in debug mode
        config.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");
        config.cxxflag("-DWHISPER_DEBUG");
    } else {
        // we're in release mode, explicitly set to release mode
        // see also https://codeberg.org/tazz4843/whisper-rs/issues/226
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    // Allow passing any WHISPER or CMAKE compile flags
    for (key, value) in env::vars() {
        let is_whisper_flag =
            key.starts_with("WHISPER_") && key != "WHISPER_DONT_GENERATE_BINDINGS";
        let is_ggml_flag = key.starts_with("GGML_");
        let is_cmake_flag = key.starts_with("CMAKE_");
        if is_whisper_flag || is_ggml_flag || is_cmake_flag {
            config.define(&key, &value);
        }
    }

    if cfg!(not(feature = "openmp")) {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(feature = "intel-sycl") {
        config.define("BUILD_SHARED_LIBS", "ON");
        config.define("GGML_SYCL", "ON");
        config.define("GGML_SYCL_TARGET", "INTEL");
        config.define("CMAKE_C_COMPILER", "icx");
        config.define("CMAKE_CXX_COMPILER", "icpx");
    }

    let destination = config.build();

    add_link_search_path(&out.join("build")).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());

    // When `system-ggml` is on, ggml-sys owns all ggml link directives — emit
    // only the whisper-specific ones here. Doubling up would either fail at
    // link time (duplicate symbols) or, worse, link two ggmls into one binary.
    let link_ggml = !cfg!(feature = "system-ggml");

    if cfg!(feature = "intel-sycl") {
        println!("cargo:rustc-link-lib=whisper");
        if link_ggml {
            println!("cargo:rustc-link-lib=ggml");
            println!("cargo:rustc-link-lib=ggml-base");
            println!("cargo:rustc-link-lib=ggml-cpu");
        }
    } else {
        println!("cargo:rustc-link-lib=static=whisper");
        if link_ggml {
            println!("cargo:rustc-link-lib=static=ggml");
            println!("cargo:rustc-link-lib=static=ggml-base");
            println!("cargo:rustc-link-lib=static=ggml-cpu");
        }
    }
    if link_ggml && (cfg!(target_os = "macos") || cfg!(feature = "openblas")) {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if link_ggml && cfg!(feature = "vulkan") {
        if cfg!(feature = "intel-sycl") {
            println!("cargo:rustc-link-lib=ggml-vulkan");
        } else {
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
        }
    }

    if link_ggml && cfg!(feature = "hipblas") {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }

    if link_ggml && cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }

    if link_ggml && cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }

    if link_ggml && cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }

    if link_ggml && cfg!(feature = "intel-sycl") {
        println!("cargo:rustc-link-lib=ggml-sycl");
    }

    println!(
        "cargo:WHISPER_CPP_VERSION={}",
        get_whisper_cpp_version(&whisper_root)
            .expect("Failed to read whisper.cpp CMake config")
            .expect("Could not find whisper.cpp version declaration"),
    );

    // for whatever reason this file is generated during build and triggers cargo complaining
    _ = std::fs::remove_file("bindings/javascript/package.json");
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn add_link_search_path(dir: &std::path::Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in std::fs::read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}

fn get_whisper_cpp_version(whisper_root: &std::path::Path) -> std::io::Result<Option<String>> {
    let cmake_lists = BufReader::new(File::open(whisper_root.join("CMakeLists.txt"))?);

    // Releases up to v1.8 declare `project("whisper.cpp" VERSION x.y.z)`; later ones set
    // `WHISPER_VERSION_{MAJOR,MINOR,PATCH}` individually.
    let (mut major, mut minor, mut patch) = (None, None, None);
    for line in cmake_lists.lines() {
        let line = line?;

        if let Some(suffix) = line.strip_prefix(r#"project("whisper.cpp" VERSION "#) {
            let whisper_cpp_version = suffix.trim_end_matches(')');
            return Ok(Some(whisper_cpp_version.into()));
        }
        for (name, slot) in [
            ("MAJOR", &mut major),
            ("MINOR", &mut minor),
            ("PATCH", &mut patch),
        ] {
            if let Some(value) = line
                .strip_prefix(&format!("set(WHISPER_VERSION_{name} "))
                .and_then(|rest| rest.strip_suffix(')'))
            {
                *slot = Some(value.trim().to_owned());
            }
        }
    }

    Ok(match (major, minor, patch) {
        (Some(major), Some(minor), Some(patch)) => Some(format!("{major}.{minor}.{patch}")),
        _ => None,
    })
}

/// Identifies the whisper.cpp sources: the submodule commit in a git checkout, otherwise the
/// contents of its `CMakeLists.txt`, which carries the release version.
fn whisper_source_stamp(whisper_src: &Path) -> String {
    // Without its own `.git`, `git rev-parse` would report an enclosing repository's commit.
    if whisper_src.join(".git").exists() {
        let head = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(whisper_src)
            .output()
            .ok()
            .filter(|output| output.status.success());
        if let Some(output) = head {
            return format!("commit {}", String::from_utf8_lossy(&output.stdout).trim());
        }
    }
    let cmake_lists = whisper_src.join("CMakeLists.txt");
    let contents = std::fs::read_to_string(&cmake_lists)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", cmake_lists.display(), e));
    format!("cmake {contents}")
}
