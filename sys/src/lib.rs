#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Force ggml-sys into the linker graph when `system-ggml` is on. Without
// this, downstream test binaries (e.g. `cargo test -p speech-to-text-whisper`)
// fail to link with `undefined symbol: ggml_*` because rustc dead-code-
// eliminates ggml-sys (it has no Rust symbols used from whisper-rs-sys) and
// drops its `cargo:rustc-link-lib=ggml*` directives and `#[link]` attrs.
// The const-reference below USES the sentinel static from ggml-sys,
// anchoring the crate so its link directives survive DCE and the linker
// pulls in libggml.so for any binary that transitively pulls us in.
#[cfg(feature = "system-ggml")]
#[allow(dead_code)]
const __ANCHOR_GGML_SYS: unsafe extern "C" fn() -> i64 = ggml_sys::__FORCE_LINK_LIBGGML;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
