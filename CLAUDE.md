# FacetGlass — project brief for Claude Code

FacetGlass is a **free, open-source (MIT) OpenFX plugin** for DaVinci Resolve (and other OFX hosts) that simulates real glass held in front of the lens: prisms, ghost repeats, split and round diopters, gems and crystals. Each pixel is path-traced: rays start from points across a virtual lens aperture, pass through a **solid 3D glass object** (Snell refraction, spectral dispersion, Fresnel reflection, total internal reflection, many bounces) and land on the footage, the off-screen world or a virtual light. It's the owner's own clean-room alternative to commercial glass plugins such as Video Village's Mirage.

Owner: Saoud. He wants **top-notch quality** and will share it publicly. **Windows (NVIDIA) first**; macOS (Apple Silicon, Metal) follows on his Mac. Keep explanations to him short and plain; ask one simple question at a time.

## Ground rules

- **Clean room only.** Never decompile, disassemble or inspect Mirage or any other commercial plugin's binaries, and never use Craft Anything's Deep mode on them. Public product pages, demo videos and stills are fine (Craft Anything **Clean room** mode; the `watch` video skill for public videos). What the public demo shows is in `docs/research/public-demo-notes.md`. Our algorithm is our own.
- Don't copy anyone's code, presets, icons, names or UI wording. Don't use "Mirage" in anything that ships.
- Keep `src/core/fg3d_core.h` the **single source of truth** for the shading maths. Every backend (CPU, CUDA, later OpenCL/Metal) includes it through a small prelude (`fg3d_backend_*.h`). Never fork the maths per backend.

## Current status (as of 2026-11-26)

Done and verified:
1. **3D core** (`src/core/fg3d_params.h`, `fg3d_core.h`): path tracer for a solid triangle-mesh glass object with a BVH. Aperture depth of field, continuous spectral dispersion, Fresnel + TIR with a bounce limit, bevelled-edge normals, rough reflections, up to 4 camera-facing area lights, off-screen world (spherical / mirror / repeat / clamp / black / transparent) from a mip pyramid with blur and feather, input colour-space decode/encode, adaptive sampling. Its own exp/log/sin/cos make **CPU and CUDA bit-identical**.
2. **Host library** (`src/host/`): 43 procedural shapes plus OBJ import in `fg3d_shapes.cpp` (building blocks: convex hull, polar/rect height-field plates, ring cuts, parametric surfaces), grouped Prisms & Filters / Lenses / Gems / Solids / Organic / Custom; smooth groups with an angle limit, bevel edge data, orientation fix, BVH (`fg3d_scene.cpp`), UI → params (`makeParams`, the only place for host-side conversions), CPU passes (`fg3d_cpu*.h`), CUDA passes (`fg3d_cuda.cu`). **Shape ids (`enum Shape`) and choice orders are stored in projects: from the first public release on, only append.**
3. **OFX plugin** (`src/plugin/FacetGlassPlugin.cpp`, v1.2): groups Shape / Placement / Glass / Optics / Outside Frame / Lights / Motion / Look / Technical; optional Refract Source (General context); full-source RoI; scene cached per instance; **CUDA render on the host's stream** when offered, multithreaded CPU otherwise. Static MSVC + CUDA runtimes.
4. **Tests**:
   - `fg3d_gpu_test`: CUDA vs CPU on 8 looks + negative stride + 2x refract source; must be exact (currently 0 difference). RTX 2080 Ti: 12–20x faster than CPU.
   - `tests/compare_ofx.py`: the built bundle in the HostSupport test host (`tests/ofxhost/fg_ofxhost.cpp`) vs the standalone renderer (`fg3d_render`): 23 checks (10 looks incl. 2 denoised, tiles plain and denoised, negative rowBytes, render scale 0.5, quality presets, wobble + lights, Refract Source, custom OBJ, a host without the message suite, identity). All exact. Text params are passed to the host as `name:=text`.
   - `fg3d_math_test`: accuracy of the portable maths.
   - `tests/fg3d_preview.py`: contact sheet of looks for eyeballing (`tests/out/fg3d/contact_sheet.png`).
5. **macOS / Metal** (2026-11-26, on Saoud's M4 Pro, macOS 26): `src/core/fg3d_backend_metal.h` (MSL prelude: `device`/`thread`, `math_mode(safe)`, `fp contract(off)`), `src/host/fg3d_metal_kernels.metal` (same passes as the CUDA kernels), `src/host/fg3d_metal.mm` (host side, encodes on the host's command queue, pipelines cached per device). CMake inlines core + kernels into `build/generated/fg3d_metal_src.h` (`cmake/embed_msl.cmake`); the source is compiled at **run time**, so only the Command Line Tools are needed (no Xcode). Plugin uses Metal when the host enables it. Bundle is universal (arm64 + x86_64, macOS 11+), has an Info.plist and is ad-hoc signed. `fg3d_metal_test`: 0 difference vs CPU on all 12 runs (also the x86_64 slice under Rosetta); 3-6x faster than the 14-core CPU. `compare_ofx.py --metal`: 23/23 exact. Installed and running in Resolve on the Mac (2026-11-26, Metal mode, Saoud: "looks so good"). Install note: root cannot read ~/Desktop (macOS privacy), so the install stages the bundle in a temp folder first.
6. Legacy v1 (thin-glass) core `src/core/fg_core.h` + `tests/compare_core.py` + the Fuse in `reference/` remain as a reference only; the plugin no longer uses them.

Not yet verified inside Resolve (v1 loaded and showed up; v1.1 not yet installed/tested).

## Build and test (macOS)

Command Line Tools + `brew install cmake`; Anaconda `python3` has numpy/pillow/scikit-image.
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
build/fg3d_math_test
python3 tests/fg3d_preview.py build/fg3d_render 640 360        (writes tests/out/fg3d/in.raw)
build/fg3d_metal_test tests/out/fg3d/in.raw 640 360
python3 tests/compare_ofx.py build/fg_ofxhost build build/fg3d_render
python3 tests/compare_ofx.py build/fg_ofxhost build build/fg3d_render --metal
```
Install (admin, ask Saoud first): `sh tools/install_macos.sh` copies the bundle to `/Library/OFX/Plugins/FacetGlass.ofx.bundle`. Restart Resolve after installing.
Non-MSVC builds use `-ffp-contract=off` (clang fuses multiply-add on arm64 by default, which breaks bit-exactness).

## Build and test (Windows)

CMake ships with VS Build Tools: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`. The CUDA Toolkit is 13.5. A **fresh** build folder must be configured with the CUDA toolset or CUDA is not found:
```bash
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T cuda="%CUDA_PATH%"
cmake --build build --config Release --parallel
build\Release\fg3d_math_test.exe
build\Release\fg3d_gpu_test.exe tests\out\fg3d\in.raw 640 360      (in.raw is written by fg3d_preview.py)
python tests/compare_ofx.py build/Release/fg_ofxhost.exe build build/Release/fg3d_render.exe
python tests/fg3d_preview.py build/Release/fg3d_render.exe 640 360
```
Python tests use the Anaconda Python (`numpy pillow scikit-image`); in Git Bash `python` is a different 3.14 install without numpy. Pipe long builds to a log file (`cmd /c "... > log 2>&1"`): expat's configure output on stderr makes PowerShell report failure.

Install (admin, ask Saoud first): `powershell -ExecutionPolicy Bypass -File tools\install_windows.ps1` copies the bundle to `C:\Program Files\Common Files\OFX\Plugins\FacetGlass.ofx.bundle`. Restart Resolve after installing.

**Run the GPU test (CUDA or Metal) and compare_ofx after any change to the core, the host library or the plugin.**

Host quirks learned in Resolve: never call `setPersistentMessage` / `clearPersistentMessage` (or any optional suite) unguarded: the Support library throws when the host lacks the suite, and a throw inside render makes Resolve show the untouched source. Use `showMessage()`. The test host provides every suite, so it cannot catch this.

## Looks (presets)

`src/host/fg3d_presets.cpp`: 16 looks as renderer `key=value` strings applied over a neutral base (every creative control; motion, lights and technical settings untouched). Plugin: **Look** menu at the top (`look`, "Custom" + presets; append only). Picking one (user edit) applies base + preset through `kPresetKeys` (renderer key -> OFX param, type); a manual edit of any mapped param sets Look back to Custom. Renderer/tests: `look=N`; test host `--look N` sends a real instance-changed; compare_ofx checks looks 1, 3, 5, 11. Gallery `docs/looks.jpg` (render with look=N on a dark, practically lit plate). Tune presets on dark footage with a lit subject: that is what the demo uses and what reads well. Lenslet shapes (fluted, honeycomb) stay soft at any f-stop, so they have no preset yet.
UX lessons from the public demo: presets give a striking composed shot immediately; the in-scene halo ring framing the subject is the "circular ghost" of its car opening; rows of overlapping copies come from facets smaller than the lens opening (On Lens).

## Speed notes (2026-11-26, M4 Pro, 1080p)

The Metal kernel is compute-bound: thread-group shape, BVH stack size and fast maths (about 13%, and it breaks exactness) barely matter. Done: parameters in constant memory (`FG_PCONST`), nearest-first BVH traversal (both child boxes tested, misses skipped), and a per-pixel lens-cone vs bounding-sphere test that passes footage through untouched when Focus = 1 and the glass cannot be reached (`boundC`/`boundR`). Time is dominated by glass pixels running to maxSamples; the practical lever is **Preview Quality** (Technical, `previewQuality`: Same / Draft / Good, applied when the host sets interactiveRenderStatus; Draft is about 7.6x faster than Good). `fg3d_metal_test <in.raw> W H --bench` times Metal per look. Ideas not yet tried: wavefront/sorted rendering to cut divergence, fewer samples where the denoiser is on.

## Release

`.github/workflows/build.yml`: macOS (macos-latest, universal, Metal tests when the runner has a device) and Windows (windows-2022, CUDA 12.8 via Jimver/cuda-toolkit, CPU tests only: no GPU on hosted runners). Pushing a tag `vX.Y.Z` builds both and publishes a GitHub release with `FacetGlass-macOS.zip` and `FacetGlass-Windows.zip`, using `docs/release-notes/vX.Y.Z.md` as the notes. Bump `kPluginVersionMinor` and the CMake project version first. The macOS bundle is ad-hoc signed, not notarized: users must clear the quarantine flag.

## Next steps (in order)

1. **Resolve test of v1.1** (CUDA mode): Edit, Color and Fusion pages; check every shape, the Refract Source input, proxy/half resolution, Deliver.
2. **On-screen gizmo**: done (colour-coded handles, works in Resolve with Open FX Overlay on the Color/Edit page).
3. **Look polish** against the public demo: bevel quality, split diopter behaviour, the "turned back" environment. (Grain and denoiser: done, see fg3d_noise.py.)
4. **Presets** (our own names) and saved defaults; a global Final-quality switch.
5. **HDRI/EXR world**, more than 2 lights in the UI. (OBJ import is done; try it in Resolve with a Blender export.)
6. **Recorded hand-held motion** (CSV pose import, loop/hold/ease).
7. **OpenCL** (AMD/Intel on Windows) through the same core; must match the CPU exactly. (Metal: done and working in Resolve; still check every shape, Refract Source, proxy and Deliver on the Mac.)
8. **Release**: GitHub Actions (windows-latest with CUDA via Jimver/cuda-toolkit; macos-14 with Metal), installers (Inno Setup / pkg), docs, demo reel, changelog.

## Model notes (why things are the way they are)

- Units: camera at the origin looking down +z, footage plane at z = 1 filling the frame (screen y in [-0.5, 0.5], x scaled by aspect). The object sits at `depth`; its world scale is `size * 0.5 * depth`, so Size is the screen fraction regardless of depth.
- Object transform: `world = pos + scale * (R p)`, R = Rz(roll) · Ry(turn) · Rx(tilt). Meshes are authored around unit radius.
- Aperture radius `0.3 * depth / fstop` (blur scales with the object, as in v1). `focusInv(f) = 1 + (1/depth - 1)(1 - f)^2` for Focus and Glass Focus.
- **Glass Focus** is a deliberate non-physical control: paths that went through glass land shifted by the difference between the two focus distances for their aperture point, so a diopter region can hold its own focus.
- **Diopter Power = magnification** through the lens centre (+P → 1 + 0.1P, −P → 1/(1 + 0.1P)); the lens curvature is solved for the current depth and scale, so lens meshes depend on Size and Depth (the plugin's scene cache keys on them only for lenses).
- Reflections: at outer (entering) surfaces the reflection is a weighted one-bounce environment lookup (`F * reflection`) and the path continues by refraction with throughput `1 - F`; TIR reflects deterministically. **Inner reflections** (Glass > Inner Reflections, `innerReflect`, default 1 = physical, 0 = off, up to 10 for creative ghosts): where a path leaves the glass, the part `F * innerReflect` reflects back inside as a deterministic branch (stack of `FG_BRANCH_STACK` = 4, branches below weight `FG_BRANCH_MIN` = 0.01 are dropped, all branches share a budget of 2 x Bounce Limit segments); the transmitted path keeps `1 - F * innerReflect`. With 0 the result is bit-identical to before. Costs about 2x render time at 1 on gems (2026-11-26). No random reflect/refract choice (it produced sparkle noise).
- **Mount** (Placement, `mount`/`lensMount`): In Scene (default) or **On Lens**. On Lens the core recentres the object per pixel on that pixel's central ray at depth objPos.z and scales it to the pixel's lens cone there (`lensFill` = Size; 1 = covers the lens), so every pixel sees the whole filter, as a real multi-image filter on a lens does: overlapping copies across the frame. Center/wobble are ignored and the pixel skip is off. Multi-Image Row (shape 42, appended after Custom OBJ) on the lens = the "replicator" row of copies; Radial Prism on the lens = circular ghost (ring of copies around the centre image); Halo Ring on the lens turns highlights into rings. Reference look: the public demo video at 4:21 (row of 5-7 copies).
- **Glass Type** (`glassType`, enum `GlassType` in fg3d_scene.h, append only): Custom / Crown (BK7) / Flint (F2) / Dense Flint (SF11) / Fused Quartz / Acrylic / Sapphire / Cubic Zirconia / Diamond / Water. makeParams converts n_d + Abbe V_d to the core's ior/dispersion (exact n_d at 587.6 nm); Chromatic Aberration then scales the real dispersion (0.4 = real). IOR slider is disabled unless Custom.
- **Tint** (RGB, default white = clear, bit-identical to before) + **Tint Density**: Beer-Lambert absorption, `absorb_c = -ln(tint_c) * density` per object unit (mesh radius), so the tint is the colour seen through one radius and does not change with Size/Depth. Core keeps the scalar Fresnel weight `thr` plus per-channel transmittance tR/tG/tB (branches carry their own); alpha is not tinted.
- Dispersion: wavelength per sample (golden-ratio sequence over 400–700 nm), Cauchy-like IOR offset, per-channel normalised weights so white stays white.
- Sampling: one 5-D Kronecker sequence (generalised golden ratio, alpha_k = phi5^-k) for pixel jitter, aperture (concentric disc mapping) and wavelength; per-pixel offsets from interleaved gradient noise so residual error is blue-ish. Adaptive stopping uses per-channel Welford error relative to sqrt(luminance), so dispersion colour noise keeps sampling.
- **Denoiser** (`fg3d_atrous`, Technical > Denoise, default 0.5 -> sigma 2.75): 4 edge-aware a-trous passes (steps 1-8) over a region padded by 30 px, so tiles match full frames exactly. Guides from `fg3d_trace` aux: variance of the weighted mean (prefiltered 3x3, SVGF-style), glass coverage, first-hit normal, landing position, and **landing spread** in pixels; pixels whose rays land within 1.5 px are sharp (anti-aliasing, not grain) and pass through untouched. Colour agreement is judged on compressed luminance l/(1+l) so bokeh highlights smooth. 2026-11-26: Good 0.0245 -> 0.0208 average (gem 0.044 -> 0.031), Draft 0.064 -> 0.049.
- Grain metric: `python tests/fg3d_noise.py build/Release/fg3d_render.exe <tag>` (RMS vs a 512-spp reference, crops in tests/out/noise). 2026-11-26: Good 0.042 -> 0.0245, Final 0.0105 -> 0.0050. Remaining grain is mostly depth of field through facets plus dispersion.
- Kernel coding rules (GPU portability and bit-exactness): float only (`f` suffix), scalar maths and the fgv3/fgv4 structs, no recursion, no global/constant arrays (private arrays like the BVH stack are fine), no textures, no identifiers that are keywords somewhere (`in`, `sample`, `half`), no library exp/log/sin/cos (use the core's), and compile without FMA contraction (`--fmad=false`, MSVC `/fp:precise`).

## Craft Anything

Installed on Saoud's PC. Connect it with `craft-anything-cli setup --client claude_code --yes`, then check `/mcp`. Use it in **Clean room** mode to study public Mirage material (capture the product page and demo images, record claimed features as evidence) and to compare FacetGlass renders against public demo stills. Deep mode is allowed only on our own code and files.
