/// The Level of Detail for operations, used to scale parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lod(pub u32);

impl Lod {
    pub fn scale_factor(&self) -> u32 {
        1 << self.0
    }
}

/// A 2D work unit slice, typical for images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub lod: Lod,
}
impl Region {
    pub fn full(dims: (i32, i32), lod: Lod) -> Self {
        Self { x: 0, y: 0, w: dims.0, h: dims.1, lod }
    }
    /// Expand the region for halo demands (e.g. Blur).
    pub fn expanded(&self, amount: i32) -> Self {
        Self { x: self.x - amount, y: self.y - amount, w: self.w + amount * 2, h: self.h + amount * 2, lod: self.lod }
    }
    /// Smallest rect covering both (bounding box). Used to accumulate demands
    /// reaching one node from several consumers.
    pub fn bounding(&self, other: &Self) -> Self {
        debug_assert_eq!(self.lod, other.lod, "cannot union regions at different LODs");
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w).max(other.x + other.w);
        let y1 = (self.y + self.h).max(other.y + other.h);
        Self { x: x0, y: y0, w: x1 - x0, h: y1 - y0, lod: self.lod }
    }
    /// Snap to a tile grid, growing outward — a source-side fetch optimization
    /// (lives here, the region knows its own geometry; the generic engine just
    /// asks for it).
    pub fn tile_aligned(&self, tile: i32) -> Self {
        let x0 = (self.x.div_euclid(tile)) * tile;
        let y0 = (self.y.div_euclid(tile)) * tile;
        let x1 = ((self.x + self.w + tile - 1).div_euclid(tile)) * tile;
        let y1 = ((self.y + self.h + tile - 1).div_euclid(tile)) * tile;
        Self { x: x0, y: y0, w: x1 - x0, h: y1 - y0, lod: self.lod }
    }
}

/// A 1D work unit slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Range {
    pub start: i32,
    pub end: i32,
}

impl Range {
    pub fn bounding(&self, other: &Self) -> Self {
        Self { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
}

/// A 0D work unit, representing the entirety of a data structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atomic;

/// The erased slice enum, used wherever the graph crosses heterogeneous node boundaries.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkUnit {
    Region(Region),
    Range(Range),
    Atomic,
}

impl WorkUnit {
    /// Accumulate two demands of the same shape into the smallest unit covering
    /// both. This 3-arm match is the **one allowed shape switch** (per-shape
    /// strategy, not per-datatype) — the demand walk calls it generically; the
    /// real geometry math lives on `Region`/`Range`. Mismatched shapes can't
    /// happen for a single node (its output shape is fixed), so we keep `self`.
    pub fn union(&self, other: &WorkUnit) -> WorkUnit {
        match (self, other) {
            (WorkUnit::Region(a), WorkUnit::Region(b)) => WorkUnit::Region(a.bounding(b)),
            (WorkUnit::Range(a), WorkUnit::Range(b)) => WorkUnit::Range(a.bounding(b)),
            (WorkUnit::Atomic, WorkUnit::Atomic) => WorkUnit::Atomic,
            _ => self.clone(),
        }
    }
}

/// The typed counterpart to the erased WorkUnit. Region/Range/Atomic each implement it.
pub trait WorkUnitFor: Clone + Send + Sync + 'static {
    fn erase(&self) -> WorkUnit;
    fn typed(wu: &WorkUnit) -> Option<Self>;
}

impl WorkUnitFor for Region {
    fn erase(&self) -> WorkUnit { WorkUnit::Region(self.clone()) }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Region(r) = wu { Some(r.clone()) } else { None }
    }
}

impl WorkUnitFor for Range {
    fn erase(&self) -> WorkUnit { WorkUnit::Range(self.clone()) }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Range(r) = wu { Some(r.clone()) } else { None }
    }
}

impl WorkUnitFor for Atomic {
    fn erase(&self) -> WorkUnit { WorkUnit::Atomic }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Atomic = wu { Some(Atomic) } else { None }
    }
}
