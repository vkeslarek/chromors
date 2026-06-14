use crate::Camera;
use crate::interactive::{FloatRect, OverlayAction, OverlayElement};
use vello::Scene;
use vello::kurbo::{BezPath, Circle, ParamCurve, Point, Stroke};
use vello::peniko::{Brush, Color};

pub struct BezierElement {
    id: &'static str,
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    hovered: bool,
    active_handle: Option<usize>,
}

impl BezierElement {
    pub fn new(id: &'static str, p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        Self {
            id,
            p0,
            p1,
            p2,
            p3,
            hovered: false,
            active_handle: None,
        }
    }

    // Size of the interactive handle circle in screen pixels (divided by zoom to keep constant screen size)
    fn handle_radius(&self, camera: &Camera) -> f32 {
        10.0 / camera.zoom
    }
}

impl OverlayElement for BezierElement {
    fn id(&self) -> &'static str {
        self.id
    }

    fn bounds(&self, camera: &Camera) -> FloatRect {
        let r = self.handle_radius(camera) + 2.0; // Adding some padding
        let min_x = self.p0.x.min(self.p1.x).min(self.p2.x).min(self.p3.x) as f32 - r;
        let max_x = self.p0.x.max(self.p1.x).max(self.p2.x).max(self.p3.x) as f32 + r;
        let min_y = self.p0.y.min(self.p1.y).min(self.p2.y).min(self.p3.y) as f32 - r;
        let max_y = self.p0.y.max(self.p1.y).max(self.p2.y).max(self.p3.y) as f32 + r;
        FloatRect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }

    fn hit_test(&self, world_x: f32, world_y: f32, camera: &Camera) -> bool {
        let r = self.handle_radius(camera) as f64;
        let r_sq = r * r;
        let p = Point::new(world_x as f64, world_y as f64);

        // Hit handles first
        if (p.x - self.p0.x).powi(2) + (p.y - self.p0.y).powi(2) <= r_sq {
            return true;
        }
        if (p.x - self.p1.x).powi(2) + (p.y - self.p1.y).powi(2) <= r_sq {
            return true;
        }
        if (p.x - self.p2.x).powi(2) + (p.y - self.p2.y).powi(2) <= r_sq {
            return true;
        }
        if (p.x - self.p3.x).powi(2) + (p.y - self.p3.y).powi(2) <= r_sq {
            return true;
        }

        // Approximate distance to curve by flattening it
        let mut min_dist_sq = f64::MAX;
        let mut last_pt = self.p0;

        let path = vello::kurbo::CubicBez::new(self.p0, self.p1, self.p2, self.p3);
        // We evaluate points along the curve (simplified sampling)
        for i in 1..=20 {
            let t = i as f64 / 20.0;
            let pt = path.eval(t);

            // Point to line segment distance squared
            let l2 = (pt.x - last_pt.x).powi(2) + (pt.y - last_pt.y).powi(2);
            let dist_sq = if l2 == 0.0 {
                (p.x - pt.x).powi(2) + (p.y - pt.y).powi(2)
            } else {
                let t = ((p.x - last_pt.x) * (pt.x - last_pt.x)
                    + (p.y - last_pt.y) * (pt.y - last_pt.y))
                    / l2;
                let t = t.clamp(0.0, 1.0);
                let proj_x = last_pt.x + t * (pt.x - last_pt.x);
                let proj_y = last_pt.y + t * (pt.y - last_pt.y);
                (p.x - proj_x).powi(2) + (p.y - proj_y).powi(2)
            };

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
            }
            last_pt = pt;
        }

        let hit_radius = 5.0 / camera.zoom as f64; // 5px visual hit tolerance for the line
        min_dist_sq <= hit_radius * hit_radius
    }

    fn render(&self, scene: &mut Scene, camera: &Camera) {
        let mut path = BezPath::new();
        path.move_to(self.p0);
        path.curve_to(self.p1, self.p2, self.p3);

        let stroke_width = 4.0 / camera.zoom as f64;
        let handle_r = 6.0 / camera.zoom as f64;
        let line_stroke = 2.0 / camera.zoom as f64;

        let color = if self.hovered {
            Color::from_rgb8(100, 200, 255)
        } else {
            Color::WHITE
        };

        let cam_affine = vello::kurbo::Affine::new([
            camera.zoom as f64,
            0.0,
            0.0,
            camera.zoom as f64,
            -(camera.pan_x * camera.zoom) as f64,
            -(camera.pan_y * camera.zoom) as f64,
        ]);

        // Draw handles guides
        if self.hovered || self.active_handle.is_some() {
            let mut guides = BezPath::new();
            guides.move_to(self.p0);
            guides.line_to(self.p1);
            guides.move_to(self.p3);
            guides.line_to(self.p2);
            scene.stroke(
                &Stroke::new(line_stroke),
                cam_affine,
                &Brush::Solid(Color::from_rgba8(255, 255, 255, 128)),
                None,
                &guides,
            );
        }

        // Draw curve
        scene.stroke(
            &Stroke::new(stroke_width),
            cam_affine,
            &Brush::Solid(color),
            None,
            &path,
        );

        // Draw handles
        if self.hovered || self.active_handle.is_some() {
            for (i, p) in [self.p0, self.p1, self.p2, self.p3].iter().enumerate() {
                let is_active = self.active_handle == Some(i);
                let handle_color = if is_active {
                    vello::peniko::color::palette::css::RED
                } else {
                    Color::WHITE
                };
                scene.fill(
                    vello::peniko::Fill::NonZero,
                    cam_affine,
                    &Brush::Solid(handle_color),
                    None,
                    &Circle::new(*p, handle_r),
                );
            }
        }
    }

    fn on_hover(&mut self) -> bool {
        self.hovered = true;
        true
    }

    fn on_unhover(&mut self) -> bool {
        self.hovered = false;
        true
    }

    fn on_press(&mut self, world_x: f32, world_y: f32, camera: &Camera) -> bool {
        let p = Point::new(world_x as f64, world_y as f64);
        let r = self.handle_radius(camera) as f64;
        let r_sq = r * r;

        if (p.x - self.p0.x).powi(2) + (p.y - self.p0.y).powi(2) <= r_sq {
            self.active_handle = Some(0);
        } else if (p.x - self.p1.x).powi(2) + (p.y - self.p1.y).powi(2) <= r_sq {
            self.active_handle = Some(1);
        } else if (p.x - self.p2.x).powi(2) + (p.y - self.p2.y).powi(2) <= r_sq {
            self.active_handle = Some(2);
        } else if (p.x - self.p3.x).powi(2) + (p.y - self.p3.y).powi(2) <= r_sq {
            self.active_handle = Some(3);
        } else {
            self.active_handle = None;
        }

        true
    }

    fn on_release(&mut self) -> bool {
        self.active_handle = None;
        true
    }

    fn on_drag(&mut self, dx: f32, dy: f32) -> Option<OverlayAction> {
        if let Some(h) = self.active_handle {
            let dp = vello::kurbo::Vec2::new(dx as f64, dy as f64);
            match h {
                0 => self.p0 += dp,
                1 => self.p1 += dp,
                2 => self.p2 += dp,
                3 => self.p3 += dp,
                _ => {}
            }
            Some(OverlayAction::Drag {
                id: self.id,
                dx,
                dy,
            })
        } else {
            // Drag the whole curve
            let dp = vello::kurbo::Vec2::new(dx as f64, dy as f64);
            self.p0 += dp;
            self.p1 += dp;
            self.p2 += dp;
            self.p3 += dp;
            Some(OverlayAction::Drag {
                id: self.id,
                dx,
                dy,
            })
        }
    }
}
