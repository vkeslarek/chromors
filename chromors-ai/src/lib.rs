pub mod prelude;

#[cfg(feature = "sam2")]
pub mod sam2;

#[cfg(feature = "sam3")]
pub mod sam3;

#[cfg(feature = "cascadepsp")]
pub mod cascadepsp;

#[cfg(feature = "modnet")]
pub mod modnet;

#[cfg(feature = "vitmatte")]
pub mod vitmatte;

#[cfg(feature = "realesrgan")]
pub mod realesrgan;

#[cfg(feature = "swinir")]
pub mod swinir;

#[cfg(feature = "depth_anything")]
pub mod depth_anything;

#[cfg(feature = "lama")]
pub mod lama;

