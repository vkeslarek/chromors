//! GPU tiled viewport for poc's `Image2D<GpuBackend>` — camera/pan/zoom,
//! tile atlas, fetcher, and overlay rendering on top of the lazy DAG.

pub mod atlas;
pub mod bench;
pub mod camera;
pub mod controller;
pub mod fetcher;
pub mod interactive;
pub mod layer;
pub mod overlay;
pub mod pipeline;
pub mod rect;
pub mod renderer;
pub mod source;
pub mod vector;
pub mod vello_overlay;

pub use camera::{Camera, CameraState, CameraUniform, compute_floor_mip, compute_max_mip};
pub use controller::ViewportController;
pub use interactive::{FloatRect, OverlayAction, OverlayElement, OverlayScene};
pub use layer::{ImageLayer, LayerTransform};
pub use overlay::OverlayVertex;
pub use rect::Rect;
pub use renderer::{ViewportBounds, ViewportRenderer};
pub use source::{
    ImageViewportSource, MippedViewportSource, VectorGraphicsViewportSource, ViewportLayerSource,
};
pub use vector::{BezierGraphic, VectorGraphics};
pub use vello_overlay::VelloOverlay;
