//! Opaque handle for Vello backend.

use std::sync::Arc;

/// A placeholder for a Vello scene.
#[derive(Clone, Debug)]
pub struct VelloScene;

/// The payload for a VelloBackend node.
#[derive(Clone, Debug)]
pub struct VelloHandle {
    pub scene: Arc<VelloScene>,
}

impl VelloHandle {
    pub fn new(scene: Arc<VelloScene>) -> Self {
        Self { scene }
    }
}
