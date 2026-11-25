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

template <class ParFor, class RowFn>
void renderCPU(const FG3DParams& p, const float* src, const float* gls, const SceneBuffers& s, int x0, int x1, int y0,
               int y1, ParFor parFor, RowFn dstRow)
{
    if (p.denoise <= 0.0f) {
        parFor(y0, y1, [&](int y) { shadeCPU(p, src, gls, s, x0, x1, y, y + 1, dstRow); });
        return;
    }
    const FG3DParams pp = p;
    const float* tris = s.tris.empty() ? nullptr : s.tris.data();
    const float* nf = s.nodesF.empty() ? nullptr : s.nodesF.data();
    const int* ni = s.nodesI.empty() ? nullptr : s.nodesI.data();
    const int fx0 = (int)p.frameX, fy0 = (int)p.frameY, fx1 = fx0 + (int)p.frameW, fy1 = fy0 + (int)p.frameH;
    const int rx0 = std::max(x0 - FG_DENOISE_PAD, fx0), rx1 = std::min(x1 + FG_DENOISE_PAD, fx1);
    const int ry0 = std::max(y0 - FG_DENOISE_PAD, fy0), ry1 = std::min(y1 + FG_DENOISE_PAD, fy1);
    const int w = rx1 - rx0, h = ry1 - ry0;
    if (w <= 0 || h <= 0) return;
    std::vector<float> aux((size_t)w * h * FG_AUX_FLOATS), cvA((size_t)w * h * FG_CV_FLOATS), cvB(cvA.size());
    parFor(0, h, [&](int j) {
        for (int i = 0; i < w; ++i) {
            float* a = &aux[((size_t)j * w + i) * FG_AUX_FLOATS];
            fg3d_trace(&pp, src, gls, tris, nf, ni, rx0 + i, ry0 + j, a);
            float* c = &cvA[((size_t)j * w + i) * FG_CV_FLOATS];
            for (int k = 0; k < FG_CV_FLOATS; ++k) c[k] = a[k];
        }
    });
    float* cur = cvA.data();
    float* nxt = cvB.data();
    for (int it = 0; it < FG_DENOISE_ITERS; ++it) {
        parFor(0, h, [&](int j) {
            for (int i = 0; i < w; ++i) fg3d_atrous(aux.data(), cur, nxt, w, h, i, j, 1 << it, pp.denoise, pp.frameH);
        });
        std::swap(cur, nxt);
    }
    parFor(y0, y1, [&](int y) {
        float* d = dstRow(y);
        for (int x = x0; x < x1; ++x, d += 4) {
            const float* c = cur + ((size_t)(y - ry0) * w + (x - rx0)) * FG_CV_FLOATS;
            fgv4 o = fg3d_finish(&pp, src, x, y, fgv4m(c[0], c[1], c[2], c[3]));
            d[0] = o.x; d[1] = o.y; d[2] = o.z; d[3] = o.w;
        }
    });
}

}  // namespace fg3d
