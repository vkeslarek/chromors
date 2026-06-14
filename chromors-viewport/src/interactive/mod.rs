use crate::Camera;
use vello::Scene;

pub mod bezier;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl FloatRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
    pub fn intersects(&self, other: &FloatRect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

pub enum OverlayAction {
    Drag { id: &'static str, dx: f32, dy: f32 },
    HoverChanged { id: &'static str, is_hovered: bool },
    Clicked { id: &'static str },
}

/// Interface for interactive viewport objects (handles, bezier curves, overlays).
pub trait OverlayElement {
    fn id(&self) -> &'static str;

    /// Returns the world-space bounding box. Used for spatial invalidation.
    fn bounds(&self, camera: &Camera) -> FloatRect;

    /// Tests if the world-space coordinate hits this element.
    fn hit_test(&self, world_x: f32, world_y: f32, camera: &Camera) -> bool {
        let b = self.bounds(camera);
        world_x >= b.x && world_x <= b.x + b.width && world_y >= b.y && world_y <= b.y + b.height
    }

    /// Draw the element into the provided Vello scene.
    fn render(&self, scene: &mut Scene, camera: &Camera);

    // Lifecycle & Interaction hooks
    // Return true to request a redraw of this element.
    fn on_hover(&mut self) -> bool {
        false
    }
    fn on_unhover(&mut self) -> bool {
        false
    }
    fn on_press(&mut self, _world_x: f32, _world_y: f32, _camera: &Camera) -> bool {
        false
    }
    fn on_release(&mut self) -> bool {
        false
    }

    /// Return an action if the drag should be propagated to the application business logic.
    fn on_drag(&mut self, _dx: f32, _dy: f32) -> Option<OverlayAction> {
        None
    }
}

pub struct ElementNode {
    pub element: Box<dyn OverlayElement>,
    pub cached_scene: Option<Scene>,
    pub cached_bounds: Option<FloatRect>,
}

pub struct OverlayScene {
    pub nodes: Vec<ElementNode>,
    pub active_id: Option<&'static str>,
    pub hovered_id: Option<&'static str>,
    pub dirty_regions: Vec<FloatRect>,
    pub needs_render: bool,
    pub last_world_pos: Option<(f32, f32)>,
    pub last_camera_affine: Option<[f64; 6]>,
}

impl Default for OverlayScene {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayScene {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            active_id: None,
            hovered_id: None,
            dirty_regions: Vec::new(),
            needs_render: true,
            last_world_pos: None,
            last_camera_affine: None,
        }
    }

    pub fn take_needs_render(&mut self) -> bool {
        let n = self.needs_render;
        self.needs_render = false;
        n
    }

    pub fn add(&mut self, element: Box<dyn OverlayElement>) {
        self.nodes.push(ElementNode {
            element,
            cached_scene: None,
            cached_bounds: None,
        });
        // We don't have a camera to compute bounds right now, so we just
        // let the first build_vello_scene pass generate it.
    }

    pub fn clear(&mut self) {
        for node in &self.nodes {
            if let Some(b) = node.cached_bounds {
                self.dirty_regions.push(b);
            }
        }
        self.nodes.clear();
        self.active_id = None;
        self.hovered_id = None;
        self.needs_render = true;
    }

    pub fn invalidate(&mut self, rect: FloatRect) {
        self.dirty_regions.push(rect);
        self.needs_render = true;
    }

    pub fn invalidate_all(&mut self) {
        for node in &mut self.nodes {
            node.cached_scene = None;
        }
        self.needs_render = true;
    }

    /// Compiles all elements into a single Vello Scene, only re-rendering elements
    /// that intersect with dirty regions (spatial invalidation).
    pub fn build_vello_scene(&mut self, camera: &Camera) -> Scene {
        let cam_affine = [
            camera.zoom as f64,
            0.0,
            0.0,
            camera.zoom as f64,
            -(camera.pan_x * camera.zoom) as f64,
            -(camera.pan_y * camera.zoom) as f64,
        ];

        if self.last_camera_affine != Some(cam_affine) {
            self.invalidate_all();
            self.last_camera_affine = Some(cam_affine);
        }

        let mut master_scene = Scene::new();

        // 1. Identify which nodes need to be redrawn
        for node in &mut self.nodes {
            let current_bounds = node.element.bounds(camera);

            // Re-render if:
            // - It has no cached scene
            // - Its current bounds intersect a dirty region
            // - Its old bounds intersect a dirty region (meaning something under/over it changed)
            let mut needs_redraw = node.cached_scene.is_none();

            if !needs_redraw {
                for dirty in &self.dirty_regions {
                    if current_bounds.intersects(dirty)
                        || node
                            .cached_bounds
                            .map(|b| b.intersects(dirty))
                            .unwrap_or(false)
                    {
                        needs_redraw = true;
                        break;
                    }
                }
            }

            if needs_redraw {
                // Invalidate the old bounds
                if let Some(old_bounds) = node.cached_bounds {
                    self.dirty_regions.push(old_bounds);
                }

                let mut sub_scene = Scene::new();
                node.element.render(&mut sub_scene, camera);
                node.cached_scene = Some(sub_scene);
                node.cached_bounds = Some(current_bounds);
            }
        }

        // 2. Clear dirty regions for the next frame
        self.dirty_regions.clear();

        // 3. Append all sub-scenes in painter's order
        for node in &self.nodes {
            if let Some(sub) = &node.cached_scene {
                master_scene.append(sub, None);
            }
        }

        master_scene
    }

    pub fn on_cursor_moved(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        camera: &Camera,
    ) -> Vec<OverlayAction> {
        let world_x = camera.screen_to_world_x(screen_x);
        let world_y = camera.screen_to_world_y(screen_y);
        let mut actions = Vec::new();

        if let Some(active) = self.active_id {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.element.id() == active)
                && let Some((lx, ly)) = self.last_world_pos
            {
                let dx = world_x - lx;
                let dy = world_y - ly;
                if let Some(action) = node.element.on_drag(dx, dy) {
                    actions.push(action);
                }
                if let Some(b) = node.cached_bounds {
                    self.dirty_regions.push(b);
                }
                node.cached_scene = None;
                self.needs_render = true;
            }
        } else {
            // Hit test for hover
            let mut hit_id = None;
            for node in self.nodes.iter_mut().rev() {
                if node.element.hit_test(world_x, world_y, camera) {
                    hit_id = Some(node.element.id());
                    break;
                }
            }

            if hit_id != self.hovered_id {
                if let Some(old) = self.hovered_id
                    && let Some(node) = self.nodes.iter_mut().find(|n| n.element.id() == old)
                    && node.element.on_unhover()
                {
                    if let Some(b) = node.cached_bounds {
                        self.dirty_regions.push(b);
                    }
                    node.cached_scene = None;
                }
                if let Some(new) = hit_id
                    && let Some(node) = self.nodes.iter_mut().find(|n| n.element.id() == new)
                    && node.element.on_hover()
                {
                    if let Some(b) = node.cached_bounds {
                        self.dirty_regions.push(b);
                    }
                    node.cached_scene = None;
                }

                if hit_id != self.hovered_id {
                    self.hovered_id = hit_id;
                    self.needs_render = true;
                }
            }
        }

        self.last_world_pos = Some((world_x, world_y));
        actions
    }

    pub fn on_mouse_press(
        &mut self,
        world_x: f32,
        world_y: f32,
        camera: &Camera,
    ) -> Vec<OverlayAction> {
        let mut actions = Vec::new();
        if let Some(hovered) = self.hovered_id
            && let Some(node) = self.nodes.iter_mut().find(|n| n.element.id() == hovered)
        {
            self.active_id = Some(node.element.id());
            if node.element.on_press(world_x, world_y, camera) {
                if let Some(b) = node.cached_bounds {
                    self.dirty_regions.push(b);
                }
                node.cached_scene = None;
            }
            actions.push(OverlayAction::Clicked { id: hovered });
        }
        actions
    }

    pub fn on_mouse_release(&mut self) -> Vec<OverlayAction> {
        let actions = Vec::new();
        if let Some(active) = self.active_id {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.element.id() == active)
                && node.element.on_release()
            {
                if let Some(b) = node.cached_bounds {
                    self.dirty_regions.push(b);
                }
                node.cached_scene = None;
            }
            self.active_id = None;
        }
        actions
    }
}
