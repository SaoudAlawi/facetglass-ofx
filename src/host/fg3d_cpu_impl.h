// FacetGlass 3D - CPU passes (included by fg3d_cpu.h).
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>

extern "C++" {
#include "fg3d_backend_cpu.h"
}

namespace fg3d {

inline void buildPyramidCPU(const float* base, int x1, int y1, int w, int h, int stride, int fx, int fy, int fw,
                            int fh, int colorSpace, std::vector<float>& pyr, int& levels)
{
    levels = mipLevels(fw, fh);
    pyr.resize(pyramidFloats(fw, fh, levels));
    // Level 0: decode, clamped to the image bounds.
    for (int j = 0; j < fh; ++j) {
        const int sy = std::min(std::max(fy + j - y1, 0), h - 1);
        const float* row = base + (ptrdiff_t)sy * stride;
        float* out = &pyr[(size_t)j * fw * 4];
        for (int i = 0; i < fw; ++i) {
            const int sx = std::min(std::max(fx + i - x1, 0), w - 1);
            fgv4 c = fg3d_decode_pixel(colorSpace, row + (size_t)sx * 4);
            out[4 * i] = c.x; out[4 * i + 1] = c.y; out[4 * i + 2] = c.z; out[4 * i + 3] = c.w;
        }
    }
    // Mip levels.
    size_t off = 0;
    for (int l = 0; l + 1 < levels; ++l) {
        const int lw = std::max(1, fw >> l), lh = std::max(1, fh >> l);
        const int nw = std::max(1, fw >> (l + 1)), nh = std::max(1, fh >> (l + 1));
        const float* prev = &pyr[off];
        float* next = &pyr[off + (size_t)lw * lh * 4];
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x) {
                fgv4 c = fg3d_mip_pixel(prev, lw, lh, x, y);
                float* o = next + ((size_t)y * nw + x) * 4;
                o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
            }
        off += (size_t)lw * lh * 4;
    }
}

template <class RowFn>
void shadeCPU(const FG3DParams& p, const float* src, const float* gls, const SceneBuffers& s, int x0, int x1, int y0,
              int y1, RowFn dstRow)
{
    const FG3DParams pp = p;  // private copy, as the GPU kernels do
    const float* tris = s.tris.empty() ? nullptr : s.tris.data();
    const float* nf = s.nodesF.empty() ? nullptr : s.nodesF.data();
    const int* ni = s.nodesI.empty() ? nullptr : s.nodesI.data();
    for (int y = y0; y < y1; ++y) {
        float* d = dstRow(y);
        for (int x = x0; x < x1; ++x, d += 4) {
            fgv4 c = fg3d_shade(&pp, src, gls, tris, nf, ni, x, y);
            d[0] = c.x; d[1] = c.y; d[2] = c.z; d[3] = c.w;
        }
    }
}

}  // namespace fg3d
