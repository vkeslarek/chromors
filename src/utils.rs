// ── Bench ────────────────────────────────────────────────────────────────────
//
// RAII scope timer with automatic statistical aggregation.
// Use `let _sw = Stopwatch::new("label");` at the top of a method.
//
// For cross-boundary timing (start in one function, finish in another):
//   let id = start_span("label");  ...  finish_span(id);
//
// On drop / finish, the sample is recorded into a global per-label accumulator.
// Every ~5 s each label prints μ, σ, min, max and resets.
// Memory footprint per label: ~56 bytes.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ── global registries ────────────────────────────────────────────────────────

fn bench_registry() -> &'static Mutex<HashMap<&'static str, Slot>> {
    static R: OnceLock<Mutex<HashMap<&'static str, Slot>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn span_registry() -> &'static Mutex<HashMap<u64, (&'static str, std::time::Instant)>> {
    static R: OnceLock<Mutex<HashMap<u64, (&'static str, std::time::Instant)>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(0);

// ── stats ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Slot {
    stats: Stats,
    last_log: std::time::Instant,
}

#[derive(Debug, Clone, Default)]
struct Stats {
    count: u64,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn record(&mut self, ms: f64) {
        self.count += 1;
        self.sum += ms;
        self.sum_sq += ms * ms;
        if self.count == 1 || ms < self.min {
            self.min = ms;
        }
        if self.count == 1 || ms > self.max {
            self.max = ms;
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let var = (self.sum_sq / self.count as f64) - (mean * mean);
        var.sqrt().max(0.0)
    }
}

fn record_sample(label: &'static str, ms: f64) {
    let mut map = bench_registry().lock().unwrap();
    let slot = map.entry(label).or_insert_with(|| Slot {
        stats: Stats::default(),
        last_log: std::time::Instant::now(),
    });
    slot.stats.record(ms);
    if slot.last_log.elapsed().as_secs() >= 5 {
        let s = &slot.stats;
        tracing::info!(target: "bench",
            "[{}] n={} μ={:.3}ms σ={:.3}ms [{:.3}..{:.3}]ms",
            label, s.count, s.mean(), s.std_dev(), s.min, s.max,
        );
        slot.stats = Stats::default();
        slot.last_log = std::time::Instant::now();
    }
}

// ── Stopwatch (scope-based) ──────────────────────────────────────────────────

pub struct Stopwatch {
    label: &'static str,
    start: std::time::Instant,
}

impl Stopwatch {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for Stopwatch {
    fn drop(&mut self) {
        let ms = self.start.elapsed().as_secs_f64() * 1000.0;
        record_sample(self.label, ms);
    }
}

// ── Span (cross-boundary) ────────────────────────────────────────────────────

pub fn start_span(label: &'static str) -> u64 {
    let id = NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed);
    span_registry()
        .lock()
        .unwrap()
        .insert(id, (label, std::time::Instant::now()));
    id
}

pub fn finish_span(id: u64) {
    if let Some((label, start)) = span_registry().lock().unwrap().remove(&id) {
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        record_sample(label, ms);
    }
}

pub trait ApproximateEq<Rhs = Self> {
    fn approx_eq(&self, other: &Rhs, epsilon: f32) -> bool;
}

impl ApproximateEq for f32 {
    fn approx_eq(&self, other: &f32, epsilon: f32) -> bool {
        (self - other).abs() <= epsilon
    }
}

impl ApproximateEq for f64 {
    fn approx_eq(&self, other: &f64, epsilon: f32) -> bool {
        (self - other).abs() <= epsilon as f64
    }
}

impl<T: ApproximateEq, const N: usize> ApproximateEq for [T; N] {
    fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.approx_eq(b, epsilon))
    }
}

impl<T1: ApproximateEq, T2: ApproximateEq> ApproximateEq for (T1, T2) {
    fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.0.approx_eq(&other.0, epsilon) && self.1.approx_eq(&other.1, epsilon)
    }
}

impl<T1: ApproximateEq, T2: ApproximateEq, T3: ApproximateEq> ApproximateEq for (T1, T2, T3) {
    fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.0.approx_eq(&other.0, epsilon)
            && self.1.approx_eq(&other.1, epsilon)
            && self.2.approx_eq(&other.2, epsilon)
    }
}

impl<T1: ApproximateEq, T2: ApproximateEq, T3: ApproximateEq, T4: ApproximateEq> ApproximateEq
    for (T1, T2, T3, T4)
{
    fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.0.approx_eq(&other.0, epsilon)
            && self.1.approx_eq(&other.1, epsilon)
            && self.2.approx_eq(&other.2, epsilon)
            && self.3.approx_eq(&other.3, epsilon)
    }
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr) => { $crate::assert_approx_eq!($left, $right, 1e-6); };
    ($left:expr, $right:expr, $epsilon:expr) => {{
        use $crate::utils::ApproximateEq;
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !left_val.approx_eq(right_val, $epsilon) {
                    panic!(
                        "assertion failed: `(left ≈ right)`\n  left: `{:?}`\n right: `{:?}`\n epsilon: `{}`",
                        left_val, right_val, $epsilon
                    );
                }
            }
        }
    }};
}
