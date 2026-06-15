use crate::interactive::{OverlayAction, OverlayScene};
use crate::renderer::ViewportRenderer;

pub struct ViewportController {
    pub dragging_canvas: bool,
    pub pan_velocity: (f32, f32),
    pub target_zoom: Option<f32>,
    pub zoom_anchor: Option<(f32, f32)>,
    pub last_cursor: Option<(f32, f32)>,
    last_pan_time: Option<std::time::Instant>,
}

impl Default for ViewportController {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportController {
    pub fn new() -> Self {
        Self {
            dragging_canvas: false,
            pan_velocity: (0.0, 0.0),
            target_zoom: None,
            zoom_anchor: None,
            last_cursor: None,
            last_pan_time: None,
        }
    }

    pub fn on_mouse_down(
        &mut self,
        x: f32,
        y: f32,
        scene: &mut OverlayScene,
        vp: &ViewportRenderer,
    ) -> Vec<OverlayAction> {
        let world_x = vp.camera.screen_to_world_x(x);
        let world_y = vp.camera.screen_to_world_y(y);
        let actions = scene.on_mouse_press(world_x, world_y, &vp.camera);
        if scene.active_id.is_none() {
            // Start panning canvas if no overlay element is active
            self.dragging_canvas = true;
            self.pan_velocity = (0.0, 0.0);
            self.last_cursor = Some((x, y));
            self.last_pan_time = Some(std::time::Instant::now());
        }
        actions
    }

    pub fn on_mouse_up(&mut self, scene: &mut OverlayScene) -> Vec<OverlayAction> {
        self.dragging_canvas = false;
        self.last_cursor = None;
        self.last_pan_time = None;
        scene.on_mouse_release()
    }

    pub fn on_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        scene: &mut OverlayScene,
        vp: &mut ViewportRenderer,
    ) -> Vec<OverlayAction> {
        if self.dragging_canvas {
            if let Some((last_x, last_y)) = self.last_cursor {
                let dx = x - last_x;
                let dy = y - last_y;
                vp.pan(dx, dy);

                let now = std::time::Instant::now();
                if let Some(lpt) = self.last_pan_time {
                    let dt = now.duration_since(lpt).as_secs_f32().max(0.001);
                    let inst_vx = dx / dt;
                    let inst_vy = dy / dt;
                    // Exponential moving average for smooth velocity
                    self.pan_velocity.0 = self.pan_velocity.0 * 0.5 + inst_vx * 0.5;
                    self.pan_velocity.1 = self.pan_velocity.1 * 0.5 + inst_vy * 0.5;
                }
                self.last_pan_time = Some(now);
                vp.camera.velocity_x = self.pan_velocity.0;
                vp.camera.velocity_y = self.pan_velocity.1;
            }
            self.last_cursor = Some((x, y));
            Vec::new() // No actions when panning
        } else {
            self.last_cursor = Some((x, y));
            scene.on_cursor_moved(x, y, &vp.camera)
        }
    }

    pub fn on_scroll(&mut self, dy: f32, x: f32, y: f32, vp: &mut ViewportRenderer) {
        let current = self.target_zoom.unwrap_or(vp.camera.zoom);
        let f = 1.0 + dy.abs() * 0.1;
        let new_z = if dy > 0.0 { current * f } else { current / f };
        let (bw, bh) = vp
            .layers
            .first()
            .map(|l| (l.base_w as f32, l.base_h as f32))
            .unwrap_or((1.0, 1.0));
        self.target_zoom = Some(new_z.clamp(vp.camera.min_zoom(bw, bh), 64.0));
        self.zoom_anchor = Some((x, y));
    }

    pub fn update_physics(&mut self, vp: &mut ViewportRenderer, dt: f32) -> bool {
        let mut physics_active = false;
        vp.camera.target_zoom = self.target_zoom;

        if !self.dragging_canvas {
            let (mut vx, mut vy) = self.pan_velocity;
            if vx.abs() > 1.0 || vy.abs() > 1.0 {
                vp.pan(vx * dt, vy * dt);
                let friction = (-dt * 5.0).exp();
                vx *= friction;
                vy *= friction;
                self.pan_velocity = (vx, vy);
                physics_active = true;
            } else {
                self.pan_velocity = (0.0, 0.0);
            }
            vp.camera.velocity_x = self.pan_velocity.0;
            vp.camera.velocity_y = self.pan_velocity.1;
        }

        if let Some(tz) = self.target_zoom {
            let cz = vp.camera.zoom;
            let diff = tz - cz;
            if diff.abs() > 0.001 * cz {
                let spring = 15.0;
                let new_z = cz + diff * (1.0 - (-dt * spring).exp());

                if let Some((ax, ay)) = self.zoom_anchor {
                    let ix = ax / cz + vp.camera.pan_x;
                    let iy = ay / cz + vp.camera.pan_y;
                    vp.camera.zoom = new_z;
                    vp.camera.pan_x = ix - ax / new_z;
                    vp.camera.pan_y = iy - ay / new_z;
                } else {
                    vp.camera.zoom = new_z;
                }

                vp.clamp_camera();

                vp.stale = true;
                physics_active = true;
            } else {
                vp.camera.zoom = tz;
                self.target_zoom = None;
                vp.stale = true;
            }
        }

        physics_active
    }
}
