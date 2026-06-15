//! Vector-graphics overlay primitives.
//!
//! A [`VectorGraphics`] draws itself into a `vello::Scene` in **image-space**
//! coordinates (the same space as the pixels of the layer it annotates). The
//! [`crate::vello_overlay::VelloOverlay`] composites every attached graphic over
//! the viewport each frame, applying the camera pan/zoom — so a graphic authored
//! at image coordinates tracks the image as the user pans and zooms.
//!
//! This is the interactive overlay path: immediate, re-drawn every frame, no DAG
//! / tile fetch. Build editor handles, bezier curves, node wires, etc. on it.

use vello::Scene;
use vello::kurbo::{Affine, BezPath, Point, Stroke};
use vello::peniko::{Brush, Color};

use crate::camera::CameraState;

/// Anything the viewport can draw as a vector overlay. Implement `draw` to emit
/// paths/shapes into the scene in image-space coordinates.
pub trait VectorGraphics: Send {
    fn draw(&self, scene: &mut Scene, camera: &CameraState, vp_w: u32, vp_h: u32);
    fn is_screen_space(&self) -> bool {
        false
    }
}

/// A cubic Bézier curve (`p0 → p3`, control points `p1`,`p2`), stroked.
#[derive(Clone, Debug)]
pub struct BezierGraphic {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    pub stroke_width: f64,
    pub color: Color,
}

impl BezierGraphic {
    pub fn new(p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        Self {
            p0,
            p1,
            p2,
            p3,
            stroke_width: 2.0,
            color: Color::from_rgb8(255, 200, 0),
        }
    }

    pub fn with_stroke(mut self, width: f64, color: Color) -> Self {
        self.stroke_width = width;
        self.color = color;
        self
    }
}

impl VectorGraphics for BezierGraphic {
    fn draw(&self, scene: &mut Scene, _camera: &CameraState, _vp_w: u32, _vp_h: u32) {
        let mut path = BezPath::new();
        path.move_to(self.p0);
        path.curve_to(self.p1, self.p2, self.p3);
        scene.stroke(
            &Stroke::new(self.stroke_width),
            Affine::IDENTITY,
            &Brush::Solid(self.color),
            None,
            &path,
        );
    }
}
