# Chromors Roadmap

This document outlines the long-term vision and next steps for **Chromors**. The ultimate goal is to integrate the engine as the core of the **Pixors** photo editor, ensuring the core is robust, performant, and highly extensible before moving on to evolutionary and AI features.

The roadmap is divided into two main phases: **Core Stabilization** and **Evolutionary Backlog**.

---

## Phase 1: Core Stabilization
*Goal: Make the engine's foundation flawless, self-sufficient, and well-documented for stable consumption.*

### 1. Tiered Caching Algorithm (VRAM -> RAM -> Disk)
The caching algorithm from the old `pixors-engine` proved to be outstanding and fail-safe. We need to port it to the new Chromors architecture.
* [ ] Implement the structure based on cryptographic identifiers (DAG Hashes). *(Note: Currently using `SipHash` which is not cryptographically safe against collisions for disk caching. Needs upgrade to BLAKE3/SHA256).*
* [ ] Structure the eviction hierarchy and memory limits for VRAM, system RAM, and disk persistence. *(Note: Basic flat RAM/VRAM caching via `RegionCache` is implemented, but the cascading eviction to System RAM and local Disk is still missing).*
* [x] Seamlessly integrate with asynchronous requests and the materializer.

### 2. Native Color Management (In-House)
Currently, ICC profile communication and transformations rely heavily on `libvips`, causing friction with our own color models.
* [x] Unify the color handling interface (transfer functions, primary spaces, etc.). *(Note: Excellent architecture in place using `ColorSpace` as the single source of truth).*
* [ ] Integrate parsing and application of ICC profiles **directly into the core**, without depending on complex external crates (write from scratch or port the logic to JIT shaders and native CPU fallback). *(Note: Currently only classifies known profiles via string matching. We need `TryFrom<&[u8]> for ColorSpace` to parse arbitrary tags and `From<ColorSpace> for IccProfile` to generate ICC binaries from our models).*
* [x] Ensure the JIT pipeline applies color corrections in a fused manner during *WorkingDecode/Encode*.

### 3. Documentation and Specification
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

### 3. Native Computer Vision Algorithms
Bring advanced tools used by professional photographers directly into Chromors' mathematical scope.
* [ ] **Feature Detection & Image Alignment**: Keypoint detection (SIFT/ORB) to align burst photos (HDR background).
* [ ] **Focus Stacking**: Combine shallow depth-of-field images into one infinite-focus image.
* [ ] **Panorama Blending**: Smooth stitching of multiple perspective images.

### 4. External Shader Extensibility
The engine must allow behavior injection without requiring core library recompilation.
* [ ] Allow consuming applications (like Pixors) to pass their own `.slang` code (custom operations).
* [ ] Create an interface for the Slang compiler within Chromors to inject arbitrary logic (e.g., complex *Color Grading* nodes built visually in the user's UI) into the fused pipeline.
* [ ] Validate inputs and outputs of external shaders against Chromors' type system (Kinds).
