//! A [`whisper_model_loader`] that reads a model file and reports how much of it has been read.
//!
//! whisper.cpp has no load-progress callback of its own, but it loads through whatever loader
//! the caller supplies.
//! This loader counts the bytes it hands over, so the fraction of the file consumed tracks how
//! far the load has got.
//!
//! The loader never forms a Rust slice or reference over the memory whisper.cpp asks it to fill.
//! It reads into its own initialized scratch buffer and copies each chunk out with raw pointer
//! writes, and it zero-fills whatever part of a request it cannot satisfy, so whisper.cpp never
//! reads uninitialized bytes.

use crate::WhisperError;
use std::ffi::c_void;
use std::fs::File;
use std::io::{ErrorKind, Read as _};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr::NonNull;
use whisper_rs_sys::{whisper_context, whisper_context_params, whisper_model_loader};

/// Size of the scratch buffer each read request is copied through, chunk by chunk.
const SCRATCH_LEN: usize = 1 << 20;

/// Loads a whisper context from the file at `path`, calling `on_progress` as it is read.
///
/// `on_progress` receives the fraction of the file's bytes consumed so far, in `0.0..=1.0`,
/// monotonically non-decreasing.
///
/// # Errors
///
/// Returns [`WhisperError::ModelReadFailed`] when the file cannot be opened or read, with
/// [`ErrorKind::UnexpectedEof`] when it ends before whisper.cpp has read all it asks for, and
/// [`WhisperError::ProgressCallbackPanicked`] when `on_progress` panics.
/// Returns [`WhisperError::InitError`] when whisper.cpp rejects the model for any other reason.
pub(crate) fn init_with_progress<F>(
    path: &Path,
    params: whisper_context_params,
    on_progress: F,
) -> Result<NonNull<whisper_context>, WhisperError>
where
    F: FnMut(f32) + Send,
{
    let mut loader = FileLoader::open(path, on_progress)?;
    let mut raw = loader.raw();
    // SAFETY: `raw.context` points at `loader`, which outlives this call and is not touched
    // through any other path until it returns, and its callbacks are the ones `raw` built for
    // exactly this loader type.
    // whisper.cpp calls `close` once on every path and never uses the loader after returning.
    let ctx = unsafe { whisper_rs_sys::whisper_init_with_params_no_state(&mut raw, params) };
    if let Some(error) = loader.error {
        if !ctx.is_null() {
            // SAFETY: `ctx` is a context whisper.cpp just created and nothing else owns it.
            unsafe { whisper_rs_sys::whisper_free(ctx) };
        }
        return Err(error);
    }
    NonNull::new(ctx).ok_or(WhisperError::InitError)
}

/// A model file read through a fixed scratch buffer, counting the bytes handed to whisper.cpp.
struct FileLoader<F> {
    file: File,
    /// File length recorded at open; reaching it marks a clean end of file.
    len: u64,
    position: u64,
    scratch: Box<[u8]>,
    eof: bool,
    /// The first failure; once set, every later read only zero-fills and returns 0.
    error: Option<WhisperError>,
    on_progress: F,
}

impl<F> FileLoader<F>
where
    F: FnMut(f32) + Send,
{
    fn open(path: &Path, on_progress: F) -> Result<Self, WhisperError> {
        let read_failed = |error: std::io::Error| WhisperError::ModelReadFailed(error.kind());
        let file = File::open(path).map_err(read_failed)?;
        let len = file.metadata().map_err(read_failed)?.len();
        Ok(Self {
            file,
            len,
            position: 0,
            scratch: vec![0; SCRATCH_LEN].into_boxed_slice(),
            eof: false,
            error: None,
            on_progress,
        })
    }

    /// Builds the native loader whose context points at `self`.
    ///
    /// The returned value is valid only while `self` stays in place and is not otherwise
    /// accessed.
    fn raw(&mut self) -> whisper_model_loader {
        whisper_model_loader {
            context: (self as *mut Self).cast::<c_void>(),
            read: Some(read::<F>),
            eof: Some(eof::<F>),
            close: Some(close),
        }
    }

    /// Fills `read_size` bytes at `output` from the file and returns how many came from it.
    ///
    /// # Safety
    ///
    /// `output` must be valid for writes of `read_size` bytes.
    unsafe fn read_into(&mut self, output: *mut u8, read_size: usize) -> usize {
        if self.error.is_some() {
            // SAFETY: the caller guarantees `output` is writable for `read_size` bytes.
            unsafe { output.write_bytes(0, read_size) };
            return 0;
        }

        let start = self.position;
        let mut filled = 0;
        while filled < read_size {
            let chunk_len = (read_size - filled).min(self.scratch.len());
            let n = match self.file.read(&mut self.scratch[..chunk_len]) {
                Ok(0) => break,
                Ok(n) => n,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => {
                    self.error = Some(WhisperError::ModelReadFailed(error.kind()));
                    break;
                }
            };
            // SAFETY: `n <= chunk_len <= read_size - filled`, so the destination range lies
            // within the `read_size` writable bytes at `output`, and the scratch buffer is a
            // separate Rust allocation holding at least `n` initialized bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(self.scratch.as_ptr(), output.add(filled), n);
            }
            filled += n;
            self.position += n as u64;
            let fraction = self.fraction();
            (self.on_progress)(fraction);
        }

        if filled < read_size {
            // SAFETY: `filled < read_size`, so the tail lies within the writable bytes at
            // `output`.
            unsafe { output.add(filled).write_bytes(0, read_size - filled) };
            self.eof = true;
            // Only a read that starts exactly at the end and gets nothing is a clean end of file.
            // whisper.cpp relies on it: it reads the next tensor header at the end of the file
            // and only then checks `eof`.
            let clean_eof = filled == 0 && start == self.len;
            if !clean_eof && self.error.is_none() {
                self.error = Some(WhisperError::ModelReadFailed(ErrorKind::UnexpectedEof));
            }
        }
        filled
    }

    fn fraction(&self) -> f32 {
        if self.len == 0 {
            return 1.0;
        }
        (self.position as f64 / self.len as f64).min(1.0) as f32
    }
}

/// The native `read` callback.
///
/// # Safety
///
/// `ctx` must point at a live `FileLoader<F>` with no other active reference to it, and `output`
/// must be valid for writes of `read_size` bytes.
unsafe extern "C" fn read<F>(ctx: *mut c_void, output: *mut c_void, read_size: usize) -> usize
where
    F: FnMut(f32) + Send,
{
    // SAFETY: whisper.cpp passes back the `context` this loader was given, which points at the
    // `FileLoader<F>` that stays alive and otherwise untouched for the whole init call.
    let loader = unsafe { &mut *ctx.cast::<FileLoader<F>>() };
    let output = output.cast::<u8>();
    // SAFETY: whisper.cpp passes an `output` valid for `read_size` writable bytes.
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        loader.read_into(output, read_size)
    }));
    result.unwrap_or_else(|_| {
        if loader.error.is_none() {
            loader.error = Some(WhisperError::ProgressCallbackPanicked);
        }
        loader.eof = true;
        // SAFETY: whisper.cpp passes an `output` valid for `read_size` writable bytes.
        unsafe { output.write_bytes(0, read_size) };
        0
    })
}

/// The native `eof` callback.
///
/// # Safety
///
/// `ctx` must point at a live `FileLoader<F>`.
unsafe extern "C" fn eof<F>(ctx: *mut c_void) -> bool
where
    F: FnMut(f32) + Send,
{
    // SAFETY: whisper.cpp passes back the `context` this loader was given, which points at the
    // `FileLoader<F>` that stays alive for the whole init call.
    let loader = unsafe { &*ctx.cast::<FileLoader<F>>() };
    catch_unwind(AssertUnwindSafe(|| loader.eof)).unwrap_or(true)
}

/// The native `close` callback, which does nothing.
///
/// The loader lives on the Rust stack for the duration of the init call, and Rust drops it,
/// closing the file, once whisper.cpp has returned.
unsafe extern "C" fn close(_ctx: *mut c_void) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A file under the system temporary directory, removed on drop.
    struct TempFile(PathBuf);

    impl TempFile {
        fn with_contents(contents: &[u8]) -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "whisper-rs-file-loader-{}-{id}.bin",
                std::process::id()
            ));
            std::fs::write(&path, contents).expect("write temporary file");
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Deterministic contents with no repeating period that aligns with the scratch buffer.
    fn contents(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    /// Calls the native `read` callback the way whisper.cpp does.
    fn native_read(raw: &whisper_model_loader, output: &mut [u8]) -> usize {
        let read = raw.read.expect("read callback");
        // SAFETY: `raw.context` points at a loader the test keeps alive and does not otherwise
        // access during the call, and `output` is writable for its whole length.
        unsafe { read(raw.context, output.as_mut_ptr().cast(), output.len()) }
    }

    /// Calls the native `eof` callback the way whisper.cpp does.
    fn native_eof(raw: &whisper_model_loader) -> bool {
        let eof = raw.eof.expect("eof callback");
        // SAFETY: `raw.context` points at a loader the test keeps alive.
        unsafe { eof(raw.context) }
    }

    /// A whole file read in several requests, one larger than the scratch buffer, arrives intact.
    #[test]
    fn whole_file_reads_copy_exactly_and_progress_reaches_one() {
        let data = contents(SCRATCH_LEN * 2 + SCRATCH_LEN / 2 + 17);
        let file = TempFile::with_contents(&data);
        let mut progress = Vec::new();
        let mut loader = FileLoader::open(&file.0, |p| progress.push(p)).expect("open");
        let raw = loader.raw();

        // A small header, a request spanning more than two scratch buffers, then the rest
        let request_lens = [
            12,
            SCRATCH_LEN * 2 + 5,
            data.len() - 12 - (SCRATCH_LEN * 2 + 5),
        ];
        let mut read_back = Vec::new();
        for request_len in request_lens {
            let mut output = vec![0xAA; request_len];
            assert_eq!(native_read(&raw, &mut output), request_len);
            read_back.extend_from_slice(&output);
        }
        assert!(
            !native_eof(&raw),
            "a fully satisfied read is not the end of file"
        );

        let error = loader.error;
        drop(loader);
        assert!(error.is_none(), "unexpected error: {error:?}");
        assert!(read_back == data, "the bytes read differ from the file");
        // Progress moves forward chunk by chunk and ends with the whole file consumed.
        assert!(progress.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(progress.last(), Some(&1.0));
    }

    /// A read that starts at the end of the file is the clean end whisper.cpp's tensor loop
    /// expects.
    #[test]
    fn read_at_end_of_file_is_clean_eof() {
        let data = contents(64);
        let file = TempFile::with_contents(&data);
        let mut loader = FileLoader::open(&file.0, |_| {}).expect("open");
        let raw = loader.raw();

        let mut output = vec![0; data.len()];
        assert_eq!(native_read(&raw, &mut output), data.len());
        assert!(!native_eof(&raw));

        // The next tensor header read finds nothing, zero-filled, and only then reports eof.
        let mut header = [0xAA; 12];
        assert_eq!(native_read(&raw, &mut header), 0);
        assert_eq!(header, [0; 12]);
        assert!(native_eof(&raw));
        assert!(
            loader.error.is_none(),
            "clean eof latched {:?}",
            loader.error
        );
    }

    /// A request that runs past the end is zero-filled beyond the file and reported as truncation.
    #[test]
    fn read_past_end_zero_fills_and_latches_unexpected_eof() {
        let data = contents(100);
        let file = TempFile::with_contents(&data);
        let mut loader = FileLoader::open(&file.0, |_| {}).expect("open");
        let raw = loader.raw();

        let mut output = vec![0xAA; 160];
        assert_eq!(native_read(&raw, &mut output), data.len());
        assert_eq!(output[..data.len()], data[..]);
        assert!(output[data.len()..].iter().all(|&byte| byte == 0));
        assert!(native_eof(&raw));

        // Once latched, later reads only zero-fill.
        let mut later = [0xAA; 8];
        assert_eq!(native_read(&raw, &mut later), 0);
        assert_eq!(later, [0; 8]);
        assert!(matches!(
            loader.error,
            Some(WhisperError::ModelReadFailed(ErrorKind::UnexpectedEof))
        ));
    }

    /// A panicking progress callback is caught at the C boundary and fails the load.
    #[test]
    fn panicking_progress_callback_latches_error_without_unwinding() {
        let data = contents(32);
        let file = TempFile::with_contents(&data);
        let mut loader =
            FileLoader::open(&file.0, |_| panic!("progress callback panic")).expect("open");
        let raw = loader.raw();

        // The panic stops at the callback, which zero-fills the whole request instead.
        let mut output = vec![0xAA; data.len()];
        assert_eq!(native_read(&raw, &mut output), 0);
        assert!(output.iter().all(|&byte| byte == 0));
        assert!(native_eof(&raw));
        assert!(matches!(
            loader.error,
            Some(WhisperError::ProgressCallbackPanicked)
        ));
    }
}
