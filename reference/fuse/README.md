# FacetGlass for DaVinci Resolve

Ray-traced virtual glass: prisms, ghost repeats, split and round diopters, and crystal facets. It works in **DaVinci Resolve free and Studio** (17 or later) and in Fusion Studio, and runs on the GPU through Metal, CUDA or OpenCL.

Each pixel traces rays from points across a virtual lens aperture, through a glass element held in front of the lens, to your footage. Snell's law handles the refraction, spectral dispersion gives real rainbow fringes, and the Fresnel equations set the reflections. Facet edges go soft and bokeh forms the way they do with real filters.

## Install

1. Quit Resolve.
2. Copy **FacetGlass.fuse** into your Fuses folder:
   - **macOS:** `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses/`
   - **Windows:** `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses\`
   - **Linux:** `~/.local/share/DaVinciResolve/Fusion/Fuses/`

   Create the `Fuses` folder if it doesn't exist.
3. (Optional, for the Edit page) Copy **FacetGlass.setting** into `.../Fusion/Templates/Edit/Effects/`, the same parent folder with `Templates/Edit/Effects` in place of `Fuses`.
4. Open Resolve.
   - **Fusion page:** press Shift+Space, type *FacetGlass*, and add it after MediaIn.
   - **Edit page:** go to Effects → Toolbox → Effects, find **FacetGlass**, and drag it onto a clip. Its controls appear in the Inspector.

## Controls

**Glass tab**

| Control | What it does |
|---|---|
| Shape | **Linear Prism** (straight wedge or stepped ghost repeats), **Radial Prism** (multi-image filter with a clear centre), **Split Diopter** (half-moon close-up lens), **Round Diopter** (magnifying disc), **Crystal** (random faceted glass) |
| Center, Angle, Size | Where the glass sits, how it's rotated, and how big it is (as a fraction of frame height) |
| Facets | Number of steps for Linear, faces for Radial, and cell density for Crystal |
| Both Sides (Linear) | Repeats the wedge steps in both directions |
| Clear Centre | Untouched middle area for Radial and Crystal |
| Prism Angle | Wedge angle of each facet. Bigger angles push the ghosts further out |
| Diopter Power | Lens curvature. Positive is convex and magnifies, negative is concave |
| Strength (IOR) | Index of refraction. How hard the glass bends light (1.0 = no glass, 1.5 = crown glass) |
| Chromatic Aberration | Spectral dispersion, giving rainbow fringes on edges and highlights |
| Seed | Varies the Crystal pattern and the noise |

**Optics tab**

| Control | What it does |
|---|---|
| F-Stop | Virtual aperture. Low values give soft, dreamy facet edges and big bokeh; high values give crisp edges |
| Focus | 1 = footage sharp, 0 = focused on the glass (footage melts, facets go sharp) |
| Diopter Focus | Focus for rays passing through a diopter. Use it with Focus to get the split-diopter look |
| Glass Distance | How far in front of the lens the glass sits. Closer gives softer edges |
| Outside Frame | What the glass shows when it bends past the frame edge: Clamp, Mirror, Repeat or Black |
| Quality | Draft (8 rays), Good (24) or Final (64). Use Draft while working and Final for rendering. Custom unlocks a Samples slider |

**Look tab**

| Control | What it does |
|---|---|
| Highlight Boost / Threshold | Brightens bright areas seen through the glass, so highlights and bokeh glow |
| Reflection | Fresnel reflection on the glass surface (a mirrored image of the scene) |
| Light Position / Size / Intensity | A virtual area light that puts a glint on the facets |
| Hand-held Wobble / Speed | Automatic organic drift and rotation, like a prism held by hand |
| Mix | Blends with the original |

**Refract Source** (second input, Fusion page): connect another clip and the glass shows that clip instead of the main one.

## Recipes

- **Prism ghost:** Linear Prism, Facets 1, Prism Angle 6–10, F-Stop 2.8, Wobble 0.3.
- **Ghost repeats:** Linear Prism, Facets 3, Both Sides on, Prism Angle 4, Size 0.25.
- **Multi-image filter:** Radial Prism, Facets 5–6, Clear Centre 0.5, Chromatic Aberration 0.4.
- **Split diopter (one half sharp):** Split Diopter, Focus 0.45, Diopter Focus 1.0, F-Stop 2, Diopter Power 2. Swap the two focus values to flip which half is sharp. Rotate with Angle.
- **Magnifier bubble:** Round Diopter, Size 0.25, Diopter Power 12–16, F-Stop 8.
- **Shattered crystal:** Crystal, Facets 4, Clear Centre 0.5, Chromatic Aberration 0.8, Highlight Boost 1.
- **Dreamy, focused on the glass:** any shape, Focus 0, F-Stop 2, Quality Final.

## Tips

- Noise or grain in very soft areas means more rays are needed. Switch Quality to Final, or to Custom with 128+ samples.
- It works on log footage. For the most physically correct highlight behaviour, put it before your output transform.
- If the node shows nothing, open Workspace → Console in Fusion. Any kernel error is printed there.

## How it differs from Mirage

FacetGlass is an independent clean-room tool, written from scratch and inspired by commercial glass plugins' public feature lists. It doesn't yet have custom imported geometry, HDRI or image-based reflections, recorded motion files, or a blurred out-of-frame source.

License: MIT.
