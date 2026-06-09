/// Axis-aligned integer rectangle (pixel coordinates, inclusive origin).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    pub fn is_empty(self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    fn x2(self) -> i32 {
        self.x + self.width
    }
    fn y2(self) -> i32 {
        self.y + self.height
    }

    pub fn intersects(self, other: Rect) -> bool {
        self.x < other.x2() && other.x < self.x2() && self.y < other.y2() && other.y < self.y2()
    }

    pub fn intersection(self, other: Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());
        if x < x2 && y < y2 {
            Some(Rect::new(x, y, x2 - x, y2 - y))
        } else {
            None
        }
    }

    pub fn bounding_box(self, other: Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let x2 = self.x2().max(other.x2());
        let y2 = self.y2().max(other.y2());
        Rect::new(x, y, x2 - x, y2 - y)
    }

    pub fn clamp(self, bounds: Rect) -> Rect {
        let x = self.x.max(bounds.x);
        let y = self.y.max(bounds.y);
        let x2 = self.x2().min(bounds.x2());
        let y2 = self.y2().min(bounds.y2());
        Rect::new(x, y, (x2 - x).max(0), (y2 - y).max(0))
    }

    pub fn expand(self, halo: i32, bounds: Rect) -> Rect {
        Rect::new(
            self.x - halo,
            self.y - halo,
            self.width + 2 * halo,
            self.height + 2 * halo,
        )
        .clamp(bounds)
    }
}

/// Merge a list of rects: any two that intersect or touch are merged into their
/// bounding box.
pub fn merge_overlapping(mut rects: Vec<Rect>) -> Vec<Rect> {
    if rects.len() <= 1 {
        return rects;
    }

    let mut changed = true;
    while changed {
        changed = false;
        rects.sort_by_key(|r| (r.x, r.y));

        let mut out: Vec<Rect> = Vec::with_capacity(rects.len());
        for r in rects.drain(..) {
            let merged = if let Some(last) = out.last_mut()
                && (last.intersects(r) || touches(*last, r))
            {
                *last = last.bounding_box(r);
                changed = true;
                true
            } else {
                false
            };
            if !merged {
                out.push(r);
            }
        }
        rects = out;
    }
    rects
}

fn touches(a: Rect, b: Rect) -> bool {
    let (a_x2, a_y2) = (a.x + a.width, a.y + a.height);
    let (b_x2, b_y2) = (b.x + b.width, b.y + b.height);
    a.x <= b_x2 && b.x <= a_x2 && a.y <= b_y2 && b.y <= a_y2
}

/// Subtract every rect in `holes` from `target`, returning the uncovered
/// remainder as a set of disjoint rects (empty if fully covered).
pub fn subtract_all(target: Rect, holes: &[Rect]) -> Vec<Rect> {
    let mut remaining = vec![target];
    for &h in holes {
        let mut next = Vec::new();
        for r in remaining {
            next.extend(subtract_one(r, h));
        }
        remaining = next;
    }
    remaining.retain(|r| !r.is_empty());
    remaining
}

/// `a` minus `b` → up to four rects (the parts of `a` not overlapped by `b`).
pub fn subtract_one(a: Rect, b: Rect) -> Vec<Rect> {
    let Some(i) = a.intersection(b) else {
        return vec![a];
    };
    let mut out = Vec::new();
    let (a_right, a_bottom) = (a.x + a.width, a.y + a.height);
    let (i_right, i_bottom) = (i.x + i.width, i.y + i.height);
    if i.y > a.y {
        out.push(Rect::new(a.x, a.y, a.width, i.y - a.y));
    }
    if i_bottom < a_bottom {
        out.push(Rect::new(a.x, i_bottom, a.width, a_bottom - i_bottom));
    }
    if i.x > a.x {
        out.push(Rect::new(a.x, i.y, i.x - a.x, i.height));
    }
    if i_right < a_right {
        out.push(Rect::new(i_right, i.y, a_right - i_right, i.height));
    }
    out
}
