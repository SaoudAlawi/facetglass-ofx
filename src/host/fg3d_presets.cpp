// FacetGlass 3D - Look presets (tuned on dark, practically lit footage).
// SPDX-License-Identifier: MIT
#include "fg3d_presets.h"

namespace fg3d {

namespace {

const char* const kBase =
    "shape=15 facets=3 bothSides=0 prismAngle=6 clearCentre=0.4 power=4 thickness=0.3 detail=1 seed=1 "
    "center=0.5,0.5 depth=0.15 rotX=0 rotY=0 rotZ=0 size=0.7 ior=1.55 dispersion=0.4 reflection=1 roughness=0 "
    "innerReflect=1 glassType=0 tint=1,1,1 tintDensity=1 lensMount=0 bevelWidth=0.02 bevelAngle=20 "
    "maxBounces=12 fstop=4 focus=1 glassFocus=1 world=0 worldBlur=4 feather=0.05 boost=0.5 threshold=1 mix=1";

// Shape ids: see enum Shape in fg3d_scene.h.
const LookPreset kPresets[] = {
    { "Halo Frame", "shape=5 size=2.2 clearCentre=0.55 prismAngle=12 thickness=0.3 depth=0.15 fstop=4" },
    { "Rainbow Halo", "shape=5 size=2.0 clearCentre=0.6 prismAngle=30 thickness=0.3 depth=0.2 fstop=2 dispersion=0.9" },
    { "Echo Row", "lensMount=1 shape=42 facets=3 prismAngle=14 size=1.2 thickness=0.2 fstop=2.8 dispersion=0.5" },
    { "Ring Echo", "lensMount=1 shape=2 facets=5 prismAngle=12 clearCentre=0.65 size=1.2 fstop=2.8 dispersion=0.5" },
    { "Split Diopter", "shape=11 power=3 size=2.6 rotZ=200 depth=0.15 fstop=4 dispersion=0.5" },
    { "Round Diopter", "shape=10 power=3 size=0.9 center=0.75,0.35 depth=0.15 fstop=5.6 dispersion=0.6" },
    { "Prism Edge", "shape=1 size=0.9 center=0.04,0.5 depth=0.12 fstop=4 dispersion=1.0 thickness=0.3" },
    { "Faceted Ghosts", "shape=0 facets=3 bothSides=1 prismAngle=6 size=2.4 depth=0.12 rotZ=12 fstop=2.8" },
    { "Radial Bloom", "shape=2 facets=6 prismAngle=7 clearCentre=0.5 size=2.2 depth=0.15 fstop=2.8 dispersion=0.6" },
    { "Kaleidoscope", "shape=4 facets=6 size=2.5 depth=0.15 fstop=4 thickness=1.0" },
    { "Diamond", "shape=15 glassType=8 size=0.8 depth=0.15 rotX=-35 rotY=15 focus=0.05 fstop=4" },
    { "Amber Jewel",
      "shape=15 glassType=8 tint=1,0.72,0.35 tintDensity=1.5 size=0.8 depth=0.15 rotX=-35 rotY=15 focus=0.05 fstop=4" },
    { "Ice Block", "shape=36 seed=3 size=1.1 center=0.8,0.3 depth=0.15 fstop=4 rotX=20 rotY=30 ior=1.31 innerReflect=2" },
    { "Rain on Glass", "shape=35 facets=14 seed=2 size=3 depth=0.12 fstop=2.8 focus=0" },
    { "Ripple Pool", "shape=6 facets=6 prismAngle=3 size=2.6 depth=0.12 fstop=11" },
    { "Bubble Glass", "shape=34 facets=12 prismAngle=4 size=3 depth=0.12 fstop=11 seed=2" },
};

}  // namespace

int lookPresetCount() { return (int)(sizeof(kPresets) / sizeof(kPresets[0])); }
const LookPreset& lookPreset(int i) { return kPresets[i < 0 ? 0 : (i >= lookPresetCount() ? lookPresetCount() - 1 : i)]; }
const char* lookPresetBase() { return kBase; }

}  // namespace fg3d
