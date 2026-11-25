# FacetGlass — project brief for Claude Code

FacetGlass is a **free, open-source (MIT) OpenFX plugin** for DaVinci Resolve (and other OFX hosts) that simulates real glass held in front of the lens: prisms, ghost repeats, split and round diopters, and crystal facets. Each pixel is ray-traced: rays start from points across a virtual lens aperture, pass through a glass element (Snell refraction, spectral dispersion, Fresnel reflection) and land on the footage. It's the owner's own clean-room alternative to commercial glass plugins such as Video Village's Mirage.

Owner: Saoud. He wants **top-notch quality** and will share it publicly. **Windows (NVIDIA) first**; macOS (Apple Silicon, Metal) follows on his Mac.

## Ground rules

- **Clean room only.** Never decompile, disassemble or inspect Mirage or any other commercial plugin's binaries, and never use Craft Anything's Deep mode on them. Public product pages, demo videos and stills are fine (Craft Anything **Clean room** mode). Our algorithm is our own (see `src/core/fg_core.h`).
- Don't copy anyone's code, presets, icons or names. Don't use "Mirage" in anything that ships.
- Keep `src/core/fg_core.h` the **single source of truth** for the shading math. Every backend (CPU, CUDA, OpenCL, Metal) includes it through a small prelude. Never fork the math per backend.

## Current status (as of 2026-11-25)

Done and verified:
1. **Fuse v1** (`reference/fuse/FacetGlass.fuse`): a working Fusion Fuse with the same model. It's useful as a reference and quick preview, but untested inside Resolve.
2. **Portable core** (`src/core/`): `fg_params.h` (a plain-C parameter block, all 4-byte fields), `fg_core.h` (the shader, written against the macros FG_DEVICE / FG_GLOBAL / FG_PRIV / FG_SQRT …), and `fg_backend_cpu.h` (the CPU prelude). It reads images from **float RGBA buffers** with (x1, y1, w, h, strideInFloats), because OFX GPU render passes buffers, not textures. It handles render tiles and frame offsets.
3. **Golden test**: `tests/compare_core.py` renders 9 presets through the Fuse kernel (`reference/kernel-harness`) and through the portable core, and requires a max abs difference under 1e-3. It currently passes at about 1e-7 (8e-5 on the mix=0 preset, from sampling). **Run it after any change to the core.**
4. Vendored **OpenFX SDK** (`third_party/openfx`, BSD-3): headers, the C++ Support library, HostSupport (a host library, for a test host), and the `Examples/GPUGain` (CUDA/OpenCL/Metal render pattern) and `Examples/Basic` example plugins.

Not started: the OFX plugin itself (`src/plugin/` is empty). That's the next step.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
python tests/compare_core.py build/<Release/>fg_fuse_harness[.exe] build/<Release/>fg_core_equiv[.exe]
```
The Python test needs `numpy pillow scikit-image`.

On Windows, use Visual Studio 2022 (MSVC) and the CUDA Toolkit for the GPU backend. Install the built plugin to:
`C:\Program Files\Common Files\OFX\Plugins\FacetGlass.ofx.bundle\Contents\Win64\FacetGlass.ofx`
(This needs admin rights. Restart Resolve after installing; Resolve rescans OFX plugins on launch.)

## Next steps (in order)

### 1. OFX plugin, CPU render (`src/plugin/FacetGlassPlugin.cpp`)
Follow `third_party/openfx/Examples/GPUGain/GPUGain.cpp` for structure (an `OFX::ImageProcessor` subclass, `setGPURenderArgs`, a factory, `getPluginIDs`). The CMake target `FacetGlass` is already wired up and activates once that file exists.

- Plugin ID `com.facetglass.FacetGlass`, version 1.0, grouping "FacetGlass". Contexts: Filter and General. Float RGBA only. Supports tiles and multi-resolution.
- Clips: `Source` (main) and optional `Refract Source` (General context). When it isn't connected, pass the source buffer as `gls` and set `useSource=0`.
- **getRegionsOfInterest**: request the **full source RoD**, because refracted rays read anywhere in the frame.
- Fill FGParams: the frame is the source RoD in pixels at the current render scale; src/gls come from `getBounds()` and `getRowBytes()/4` (rowBytes may be negative), with `getPixelAddress(x1, y1)` as the base pointer.
- Parameters (match the Fuse; see `reference/fuse/FacetGlass.fuse` → Create() for every range and default):
  - **Glass:** Shape (choice: Linear Prism, Radial Prism, Split Diopter, Round Diopter, Crystal), Center (Double2D, normalised 0..1, default 0.5, 0.5), Angle (degrees), Size 0.35, Facets 1, Both Sides (bool), Clear Centre 0.35, Prism Angle 8°, Diopter Power 2, Strength (IOR) 1.5, Chromatic Aberration 0.3, Seed 1.
  - **Optics:** F-Stop 4, Focus 1, Diopter Focus 1, Glass Distance 0.05, Outside Frame (Clamp / Mirror / Repeat / Black, default Mirror), Quality (Draft 8 / Good 24 / Final 64 / Custom) + Samples 96.
  - **Look:** Highlight Boost 0.5, Highlight Threshold 0.8, Reflection 0, Light Position (0.3, 0.75), Light Size 0.15, Light Intensity 0, Hand-held Wobble 0, Wobble Speed 1, Mix 1.
- Host-side conversions (same as the Fuse's Lua):
  - `aperture = glassDist * 0.3 / fstop`
  - `focusInv(f) = 1 + (1/glassDist - 1) * (1 - f)^2`, used for both Focus and Diopter Focus
  - angles in radians
  - wobble: see `wobble()` in the Fuse, with time in frames and fps 24 as the base
- CPU render: `multiThreadProcessImages` loops over the render window and calls `fg_shade`.

### 2. OFX test host (`tests/ofxhost/`)
Build a small CLI host on HostSupport (adapt `third_party/openfx/HostSupport/examples/hostDemo*`; it needs expat). It loads the bundle, sets params, renders the test image and compares the result with the golden renders. This lets the plugin be tested without Resolve.

### 3. GPU backends (Resolve passes device buffers)
- **CUDA** (`src/plugin/CudaKernel.cu`): a prelude with `__device__` functions, then a kernel that copies params into a local and calls `fg_shade`. Use the host's CUDA stream (`setSupportsCudaStream`).
- **OpenCL** (`src/plugin/OpenCLKernel.cpp`): build the program at runtime from the prelude plus `fg_core.h` text, embedded as a string at build time. Cache the program per context.
- **Metal** (macOS, `src/plugin/MetalKernel.mm`): an MSL prelude with `FG_GLOBAL = device`, `FG_PRIV = thread`.
- Every GPU backend must match the CPU output within about 1e-4. Add a GPU-vs-CPU test.

### 4. Top-notch features
- An on-screen overlay interact for dragging Center and Angle in the viewer (see `Examples/Basic` BasicInteract).
- Presets.
- Custom glass shapes from an image (a height or normal map input).
- HDRI or image reflections.
- A blurred out-of-frame source (mip chain).
- Recorded hand-held motion import.
- Performance: adaptive samples, and blue-noise sampling instead of hash jitter.

### 5. Release
- GitHub Actions: windows-latest (MSVC + CUDA via Jimver/cuda-toolkit) and macos-14 (universal, Metal).
- An installer (Inno Setup for Windows, a pkg for macOS), docs, a demo reel and a changelog.

## Model notes (why things are the way they are)

- Units: the footage plane sits at z = 1, the glass at z = glassDist (default 0.05), screen focal length 1, and screen y goes from -0.5 to 0.5 with x scaled by aspect.
- **Diopter Focus** is a deliberate non-physical control. With flat footage, a physically correct diopter can only refocus by also shrinking that half of the frame a lot. Rays through a diopter therefore get their own focus distance, and the lens power adds magnification and dispersion on top.
- Diopter slope = `-power * 0.02 * r_screen`, with positive power convex. Prism facets tilt by `tan(prismAngle * k)`.
- Dispersion: 6 wavelengths from 430 to 680 nm, a Cauchy-like IOR offset, and per-channel normalised weights so white stays white.
- Kernel coding rules (for GPU portability): no doubles (always use the `f` suffix), no vector operators (use scalar math and the fg3/fg4 structs), no recursion, no global/constant arrays, no textures, and no variable names that are keywords in some shading languages (`in`, `sample`, `half`).

## Craft Anything

Installed on Saoud's PC. Connect it with `craft-anything-cli setup --client claude_code --yes`, then check `/mcp`. Use it in **Clean room** mode to study public Mirage material (capture the product page and demo images, record claimed features as evidence) and to compare FacetGlass renders against public demo stills. Deep mode is allowed only on our own code and files.
