// FacetGlass 3D - Look presets: ready-made starting points.
// Each preset is a list of "key=value" tokens in the renderer's keys (see
// tests/fg3d_args.h), applied on top of a neutral base that covers every
// creative control. Motion, lights and technical settings are left alone.
// The plugin's Look menu lists them after "Custom", in this order: the menu
// index is stored in projects, so only append.
// SPDX-License-Identifier: MIT
#pragma once

namespace fg3d {

struct LookPreset {
    const char* name;
    const char* args;  // space-separated key=value tokens
};

int lookPresetCount();
const LookPreset& lookPreset(int i);  // 0-based; menu index = i + 1
const char* lookPresetBase();         // applied before every preset

}  // namespace fg3d
