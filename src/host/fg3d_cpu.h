// FacetGlass 3D - CPU render of a whole frame (the three passes of fg3d_core.h),
// shared by the plugin's CPU path and the test tools.
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>

#include "fg3d_scene.h"

namespace fg3d {

// Decodes `img` (RGBA float, pixel (x1, y1) at `base`, stride in floats, may be
// negative) into a linear pyramid covering the frame `fx, fy, fw, fh` (pixels).
// Pixels outside the image bounds clamp to the edge.
void buildPyramidCPU(const float* base, int x1, int y1, int w, int h, int stride, int fx, int fy, int fw, int fh,
                     int colorSpace, std::vector<float>& pyr, int& levels);

// Shades rows [y0, y1) x columns [x0, x1) (absolute pixels) into dst rows.
// dstRow(y) must return a pointer to pixel x0 of row y.
template <class RowFn>
void shadeCPU(const FG3DParams& p, const float* src, const float* gls, const SceneBuffers& s, int x0, int x1, int y0,
              int y1, RowFn dstRow);

}  // namespace fg3d

#include "fg3d_cpu_impl.h"
