//! Minimal cross-boundary benchmark spans (replacement for the old
//! `pixors_engine::utils::{Stopwatch, start_span, finish_span}`).
//!
//! Scope-based timing should use `tracing::trace_span!(...).entered()`
//! directly. This module only covers the cross-boundary case (start in one
//! function, finish in another) needed by [`crate::renderer::ViewportRenderer`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

fn span_registry() -> &'static Mutex<HashMap<u64, (&'static str, std::time::Instant)>> {
    static R: OnceLock<Mutex<HashMap<u64, (&'static str, std::time::Instant)>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(0);

/// Starts a cross-boundary timing span, returning an id to pass to [`finish_span`].
pub fn start_span(label: &'static str) -> u64 {
    let id = NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed);
    span_registry()
        .lock()
        .unwrap()
        .insert(id, (label, std::time::Instant::now()));
    id
}

/// Finishes a span started with [`start_span`], logging its duration.
pub fn finish_span(id: u64) {
    if let Some((label, start)) = span_registry().lock().unwrap().remove(&id) {
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::trace!(target: "bench", "[{}] {:.3}ms", label, ms);
    }
}
