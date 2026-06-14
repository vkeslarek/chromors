# Chromors Roadmap

This document outlines the long-term vision and next steps for **Chromors**. The ultimate goal is to integrate the engine as the core of the **Pixors** photo editor, ensuring the core is robust, performant, and highly extensible before moving on to evolutionary and AI features.

The roadmap is divided into two main phases: **Core Stabilization** and **Evolutionary Backlog**.

---

## Phase 1: Core Stabilization
*Goal: Make the engine's foundation flawless, self-sufficient, and well-documented for stable consumption.*

### 1. GPU Operations Parity (Always Fast Preview)
For fluid editing, all CPU operations need a GPU counterpart.
* [ ] Map and implement missing `libvips` operations within the WGPU/Slang backend.
* [ ] Ensure the real-time preview pipeline never unexpectedly falls back to the CPU.
* [ ] Ensure mathematical parity (within tolerance limits) between `libvips` output and Slang JIT shaders.

### 2. Tiered Caching Algorithm (VRAM -> RAM -> Disk)
The caching algorithm from the old `pixors-engine` proved to be outstanding and fail-safe. We need to port it to the new Chromors architecture.
* [ ] Implement the structure based on cryptographic identifiers (DAG Hashes).
* [ ] Structure the eviction hierarchy and memory limits for VRAM, system RAM, and disk persistence.
* [ ] Seamlessly integrate with asynchronous requests and the materializer.

### 3. Native Color Management (In-House)
Currently, ICC profile communication and transformations rely heavily on `libvips`, causing friction with our own color models.
* [ ] Unify the color handling interface (transfer functions, primary spaces, etc.).
* [ ] Integrate parsing and application of ICC profiles **directly into the core**, without depending on complex external crates (write from scratch or port the logic to JIT shaders and native CPU fallback).
* [ ] Ensure the JIT pipeline applies color corrections in a fused manner during *WorkingDecode/Encode*.

### 4. Documentation and Specification
A powerful engine is useless if it's unreadable.
* [ ] Write documentation for public functions, with basic and advanced usage guides.
* [ ] Document the JIT shader compilation flow and the DAG system.
* [ ] Specify the main algorithms in detail (color math, limits, blending logic).

---

## Phase 2: Evolutionary Backlog (Pixors Integration & Expansion)
*Goal: Turn Chromors into an image manipulation monster, embracing AI, Computer Vision, and high extensibility.*

### 1. Enhanced Support and Caching for Vectors (Vello)
The Vello backend has proven excellent but can be optimized for complex scenarios.
* [ ] Improve vector rendering wrappers.
* [ ] Implement caching support for primitives and pre-rendered scenes, avoiding unnecessary rasterization during non-destructive changes to the vector tree.

### 2. High-Level Utilities (Engine Helpers)
Useful abstractions that live "above" the engine, easing the life of the UI developer.
* [ ] **`LayerStack`**: A structure encompassing an image array that automatically builds the DAG portion using the `Composite2` operation under the hood, abstracting Photoshop-like layers.
* [ ] Implement fluent builders aimed purely at manipulation UX/UI.

### 3. Artificial Intelligence and Models (via Burn)
Integration with machine learning running on VRAM is the future of image editing.
* [ ] Integrate the `burn` crate using the WGPU backend, allowing "zero-copy" buffer sharing.
* [ ] **Semantic Segmentation (YOLO/Segment Anything):** Implement native operations to select and mask isolated objects directly in the editor.
* [ ] Structure the input/output pipeline of ML tensors back into Chromors' `Image` or `Mask2D` Kinds.

### 4. Native Computer Vision Algorithms
Bring advanced tools used by professional photographers directly into Chromors' mathematical scope.
* [ ] **Feature Detection & Image Alignment**: Keypoint detection (SIFT/ORB) to align burst photos (HDR background).
* [ ] **Focus Stacking**: Combine shallow depth-of-field images into one infinite-focus image.
* [ ] **Panorama Blending**: Smooth stitching of multiple perspective images.

### 5. External Shader Extensibility
The engine must allow behavior injection without requiring core library recompilation.
* [ ] Allow consuming applications (like Pixors) to pass their own `.slang` code (custom operations).
* [ ] Create an interface for the Slang compiler within Chromors to inject arbitrary logic (e.g., complex *Color Grading* nodes built visually in the user's UI) into the fused pipeline.
* [ ] Validate inputs and outputs of external shaders against Chromors' type system (Kinds).


## INTEGRATE INTO ROADMAP:
GPU-Native Generators (Procedural Sources)

libvips exposes a create family of operations that synthesize image data from parameters alone (constants, ramps, coordinate fields, noise, test patterns, kernel/LUT/frequency-mask builders). A generator has zero image inputs — its value at (x, y) for a given region and LOD is a pure function of the coordinate and a parameter set.

Because there are no input pixels, generating these on the CPU and uploading the buffer to VRAM is pure waste. Generators must instead be implemented as Slang compute kernels that write their region directly in VRAM, participating fully in JIT fusion. This is the single biggest "free win" available to the engine.

Why GPU-native is more correct, not just faster


Region/LOD native. Because the output is a pure function of coordinates, a generator is evaluated lazily per requested Region at any Lod — no full-canvas materialization ever. A zone plate or gradient pulled at Lod(3) computes only the downsampled region. Generators become first-class citizens of the tile/preview pipeline, effectively resolution-independent, infinite-canvas sources that cost only what you pull.
Fusion. A generator is the first stage of a fused pipeline. gaussnoise → add → blur collapses into far fewer dispatches because the noise is produced in-register at the point of use rather than read back from a materialized buffer.
Tiling-correct noise. Sequential CPU RNGs are fundamentally incompatible with a tiled engine: pull(A) then pull(B) would not reconstruct pull(A ∪ B). GPU generators use a counter-based, coordinate-keyed RNG (PCG / Philox style, keyed by (x, y, seed)), making noise deterministic and independent of how the canvas is tiled. This is a property libvips cannot offer for free.