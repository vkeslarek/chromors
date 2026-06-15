use crate::editor::registry::Category;
use crate::editor::types::DataType;
use vello::peniko::Color;

pub const NODE_WIDTH: f64 = 180.0;
pub const NODE_TITLE_H: f64 = 26.0;
pub const NODE_ROW_H: f64 = 22.0;
pub const NODE_CORNER: f64 = 6.0;
pub const SOCKET_R: f64 = 5.0;
pub const SOCKET_HIT_R: f64 = 10.0;
pub const WIRE_WIDTH: f64 = 2.0;
pub const GRID_STEP: f64 = 32.0;

pub const COL_BG: Color = Color::from_rgb8(30, 30, 34);
pub const COL_GRID: Color = Color::from_rgb8(40, 40, 46);
pub const COL_NODE_BODY: Color = Color::from_rgb8(52, 52, 60);
pub const COL_NODE_SEL: Color = Color::from_rgb8(255, 180, 40);
pub const COL_WIRE: Color = Color::from_rgb8(200, 200, 210);
pub const COL_TEXT: Color = Color::from_rgb8(230, 230, 235);

pub fn socket_color(t: DataType) -> Color {
    match t {
        DataType::Image => Color::from_rgb8(100, 150, 255),
        DataType::Mask => Color::from_rgb8(180, 180, 180),
        DataType::Scalar => Color::from_rgb8(100, 255, 150),
        DataType::Color => Color::from_rgb8(255, 100, 200),
    }
}

pub fn category_color(c: Category) -> Color {
    match c {
        Category::Source => Color::from_rgb8(40, 160, 140),
        Category::Color => Color::from_rgb8(140, 80, 200),
        Category::Filter => Color::from_rgb8(200, 80, 80),
        Category::Geometry => Color::from_rgb8(200, 140, 40),
        Category::Composite => Color::from_rgb8(180, 180, 60),
        Category::Sink => Color::from_rgb8(60, 60, 60),
    }
}
