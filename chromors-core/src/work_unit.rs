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
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
            lod,
        }
    }
    /// Expand the region for halo demands (e.g. Blur).
    pub fn expanded(&self, amount: i32) -> Self {
        Self {
            x: self.x - amount,
            y: self.y - amount,
            w: self.w + amount * 2,
            h: self.h + amount * 2,
            lod: self.lod,
        }
    }
    /// Smallest rect covering both (bounding box). Used to accumulate demands
    /// reaching one node from several consumers.
    pub fn bounding(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.lod, other.lod,
            "cannot union regions at different LODs"
        );
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w).max(other.x + other.w);
        let y1 = (self.y + self.h).max(other.y + other.h);
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
            lod: self.lod,
        }
    }
    /// Snap to a tile grid, growing outward — a source-side fetch optimization
    /// (lives here, the region knows its own geometry; the generic engine just
    /// asks for it).
    pub fn tile_aligned(&self, tile: i32) -> Self {
        let x0 = (self.x.div_euclid(tile)) * tile;
        let y0 = (self.y.div_euclid(tile)) * tile;
        let x1 = ((self.x + self.w + tile - 1).div_euclid(tile)) * tile;
        let y1 = ((self.y + self.h + tile - 1).div_euclid(tile)) * tile;
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
            lod: self.lod,
        }
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
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
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

    /// Split this WorkUnit into smaller tiles whose byte cost (evaluated by
    /// `calc_bytes`) each fits under `max_bytes`.
    ///
    /// - **Region**: bisects along the longest axis repeatedly until every tile
    ///   fits. A 4000×3000 image that exceeds the limit becomes e.g. four
    ///   2000×1500 tiles.
    /// - **Range**: bisects the interval at the midpoint repeatedly.
    /// - **Atomic**: cannot be subdivided — returns `Err`. The caller must
    ///   handle this (e.g. by refusing the operation or falling back to a
    ///   smaller configuration).
    ///
    /// This method is pure geometry — no backend types, no GPU references.
    pub fn split<F>(
        &self,
        max_bytes: u64,
        calc_bytes: F,
    ) -> Result<Vec<WorkUnit>, crate::error::Error>
    where
        F: Fn(&WorkUnit) -> u64,
    {
        if calc_bytes(self) <= max_bytes {
            return Ok(vec![self.clone()]);
        }
        match self {
            WorkUnit::Region(_) => {
                let mut tiles = vec![self.clone()];
                loop {
                    let mut next = Vec::new();
                    let mut all_fit = true;
                    for tile in tiles {
                        if calc_bytes(&tile) <= max_bytes {
                            next.push(tile);
                            continue;
                        }
                        all_fit = false;
                        let WorkUnit::Region(r) = &tile else {
                            unreachable!()
                        };
                        if r.w > r.h {
                            let half = r.w / 2;
                            next.push(WorkUnit::Region(Region {
                                x: r.x,
                                y: r.y,
                                w: half,
                                h: r.h,
                                lod: r.lod,
                            }));
                            next.push(WorkUnit::Region(Region {
                                x: r.x + half,
                                y: r.y,
                                w: r.w - half,
                                h: r.h,
                                lod: r.lod,
                            }));
                        } else {
                            let half = r.h / 2;
                            next.push(WorkUnit::Region(Region {
                                x: r.x,
                                y: r.y,
                                w: r.w,
                                h: half,
                                lod: r.lod,
                            }));
                            next.push(WorkUnit::Region(Region {
                                x: r.x,
                                y: r.y + half,
                                w: r.w,
                                h: r.h - half,
                                lod: r.lod,
                            }));
                        }
                    }
                    tiles = next;
                    if all_fit {
                        break;
                    }
                }
                Ok(tiles)
            }
            WorkUnit::Range(_) => {
                let mut tiles = vec![self.clone()];
                loop {
                    let mut next = Vec::new();
                    let mut all_fit = true;
                    for tile in tiles {
                        if calc_bytes(&tile) <= max_bytes {
                            next.push(tile);
                            continue;
                        }
                        all_fit = false;
                        let WorkUnit::Range(r) = &tile else {
                            unreachable!()
                        };
                        let mid = r.start + (r.end - r.start) / 2;
                        next.push(WorkUnit::Range(Range {
                            start: r.start,
                            end: mid,
                        }));
                        next.push(WorkUnit::Range(Range {
                            start: mid,
                            end: r.end,
                        }));
                    }
                    tiles = next;
                    if all_fit {
                        break;
                    }
                }
                Ok(tiles)
            }
            WorkUnit::Atomic => Err(crate::error::Error::InvalidWorkUnit(
                "cannot split an Atomic WorkUnit: the operation requires more buffer space \
                 than the hardware limit allows, and Atomic work units are indivisible"
                    .into(),
            )),
        }
    }
}

/// The typed counterpart to the erased WorkUnit. Region/Range/Atomic each implement it.
pub trait WorkUnitFor: Clone + Send + Sync + 'static {
    fn erase(&self) -> WorkUnit;
    fn typed(wu: &WorkUnit) -> Option<Self>;
}

impl WorkUnitFor for Region {
    fn erase(&self) -> WorkUnit {
        WorkUnit::Region(self.clone())
    }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Region(r) = wu {
            Some(r.clone())
        } else {
            None
        }
    }
}

impl WorkUnitFor for Range {
    fn erase(&self) -> WorkUnit {
        WorkUnit::Range(self.clone())
    }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Range(r) = wu {
            Some(r.clone())
        } else {
            None
        }
    }
}

impl WorkUnitFor for Atomic {
    fn erase(&self) -> WorkUnit {
        WorkUnit::Atomic
    }
    fn typed(wu: &WorkUnit) -> Option<Self> {
        if let WorkUnit::Atomic = wu {
            Some(Atomic)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_region_fits_returns_single_tile() {
        let wu = WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
            lod: Lod(0),
        });
        let tiles = wu
            .split(1_000_000, |wu| match wu {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
                _ => 0,
            })
            .unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], wu);
    }

    #[test]
    fn split_region_bisects_landscape() {
        let wu = WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: 200,
            h: 100,
            lod: Lod(0),
        });
        let tiles = wu
            .split(50_000, |wu| match wu {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
                _ => 0,
            })
            .unwrap();
        assert!(tiles.len() >= 2);
        for tile in &tiles {
            let bytes = match tile {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
                _ => panic!(),
            };
            assert!(bytes <= 50_000);
        }
        let total_pixels: i64 = tiles
            .iter()
            .map(|t| match t {
                WorkUnit::Region(r) => r.w as i64 * r.h as i64,
                _ => 0,
            })
            .sum();
        assert_eq!(total_pixels, 200 * 100);
    }

    #[test]
    fn split_region_bisects_portrait() {
        let wu = WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: 100,
            h: 200,
            lod: Lod(0),
        });
        let tiles = wu
            .split(50_000, |wu| match wu {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
                _ => 0,
            })
            .unwrap();
        assert!(tiles.len() >= 2);
        let WorkUnit::Region(first) = &tiles[0] else {
            panic!()
        };
        let WorkUnit::Region(second) = &tiles[1] else {
            panic!()
        };
        assert_eq!(first.y, 0);
        assert_eq!(first.h, 100);
        assert_eq!(second.y, 100);
        assert_eq!(second.h, 100);
    }

    #[test]
    fn split_region_recursive_many_tiles() {
        let wu = WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: 1000,
            h: 1000,
            lod: Lod(0),
        });
        let tiles = wu
            .split(1_000_000, |wu| match wu {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 16,
                _ => 0,
            })
            .unwrap();
        assert!(tiles.len() >= 16);
        for tile in &tiles {
            let bytes = match tile {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 16,
                _ => panic!(),
            };
            assert!(bytes <= 1_000_000);
        }
        let total: i64 = tiles
            .iter()
            .map(|t| match t {
                WorkUnit::Region(r) => r.w as i64 * r.h as i64,
                _ => 0,
            })
            .sum();
        assert_eq!(total, 1_000_000);
    }

    #[test]
    fn split_region_preserves_offset_and_lod() {
        let wu = WorkUnit::Region(Region {
            x: 50,
            y: 100,
            w: 200,
            h: 200,
            lod: Lod(2),
        });
        let tiles = wu
            .split(50_000, |wu| match wu {
                WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
                _ => 0,
            })
            .unwrap();
        for tile in &tiles {
            let WorkUnit::Region(r) = tile else { panic!() };
            assert_eq!(r.lod, Lod(2));
            assert!(r.x >= 50);
            assert!(r.y >= 100);
        }
    }

    #[test]
    fn split_range_bisects() {
        let wu = WorkUnit::Range(Range {
            start: 0,
            end: 1000,
        });
        let tiles = wu
            .split(500, |wu| match wu {
                WorkUnit::Range(r) => (r.end - r.start) as u64,
                _ => 0,
            })
            .unwrap();
        assert!(tiles.len() >= 2);
        for tile in &tiles {
            let bytes = match tile {
                WorkUnit::Range(r) => (r.end - r.start) as u64,
                _ => panic!(),
            };
            assert!(bytes <= 500);
        }
    }

    #[test]
    fn split_atomic_returns_error() {
        let wu = WorkUnit::Atomic;
        let result = wu.split(100, |_| 200);
        assert!(result.is_err());
    }

    #[test]
    fn split_atomic_below_limit_returns_single() {
        let wu = WorkUnit::Atomic;
        let tiles = wu.split(100, |_| 50).unwrap();
        assert_eq!(tiles.len(), 1);
    }
}
