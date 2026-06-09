use super::VectorGraphics;
use vello::Scene;
use vello::kurbo::{Affine, BezPath, Point, Stroke};
use vello::peniko::{Brush, Color};

/// Um objeto que implementa o trait (BezierCurve)
pub struct BezierCurveGraphic {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    pub stroke_width: f64,
    pub color: Color,
}

impl VectorGraphics for BezierCurveGraphic {
    fn draw(&self, scene: &mut Scene) {
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
