// FacetGlass 3D - Metal backend: the passes of fg3d_core.h as compute kernels.
// Mirrors fg3d_cuda.cu. At build time CMake inlines the includes into one string
// (fg3d_metal_src.h) that the host compiles at runtime, so no Xcode is needed.
// SPDX-License-Identifier: MIT
#include "fg3d_backend_metal.h"

// Image view: pixel (x1, y1) is at element `base` of the buffer; stride in floats (may be negative).
typedef struct FGImgArgs {
    int base, x1, y1, w, h, stride;
    int fx, fy, fw, fh, cs, pad;
} FGImgArgs;

typedef struct FGMipArgs { int lw, lh, nw, nh; } FGMipArgs;

// Output window [x0, x1) x [y0, y1); dst pixel (dstX1, dstY1) at element dstBase.
// Region (rx0, ry0, rw, rh) is the denoiser's padded region.
typedef struct FGWinArgs {
    int dstBase, dstX1, dstY1, dstStride;
    int x0, y0, x1, y1;
    int rx0, ry0, rw, rh;
    int step;
    float sigma, frameH;
    int pad;
    // Soft areas: half-resolution block (hx0, hy0, hw, hh); soft != 0 to use it;
    // (fx0, fy0) = frame origin in pixels.
    int hx0, hy0, hw, hh;
    int soft, fx0, fy0, pad2;
} FGWinArgs;

kernel void fg_decode(device const float* src [[buffer(0)]], device float* lvl0 [[buffer(1)]],
                      constant FGImgArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.fw || j >= a.fh) return;
    int sx = a.fx + i - a.x1, sy = a.fy + j - a.y1;
    sx = sx < 0 ? 0 : (sx > a.w - 1 ? a.w - 1 : sx);
    sy = sy < 0 ? 0 : (sy > a.h - 1 ? a.h - 1 : sy);
    fgv4 c = fg3d_decode_pixel(a.cs, src + a.base + sy * a.stride + sx * 4);
    device float* o = lvl0 + (j * a.fw + i) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

kernel void fg_mip(device const float* prev [[buffer(0)]], device float* next [[buffer(1)]],
                   constant FGMipArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= a.nw || y >= a.nh) return;
    fgv4 c = fg3d_mip_pixel(prev, a.lw, a.lh, x, y);
    device float* o = next + (y * a.nw + x) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

kernel void fg_shade(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                     device const float* glsPyr [[buffer(2)]], device const float* tris [[buffer(3)]],
                     device const float* nodesF [[buffer(4)]], device const int* nodesI [[buffer(5)]],
                     device float* dst [[buffer(6)]], constant FGWinArgs& a [[buffer(7)]],
                     device const float* halfRes [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = a.x0 + (int)gid.x, y = a.y0 + (int)gid.y;
    if (x >= a.x1 || y >= a.y1) return;
    fgv4 c;
    float t[FG_AUX_FLOATS];
    if (a.soft != 0 && fg3d_soft_up(halfRes, a.hw, a.hh, a.hx0, a.hy0, x - a.fx0, y - a.fy0, FG_SOFT_SPREAD, t))
        c = fg3d_finish(&p, srcPyr, x, y, fgv4m(t[0], t[1], t[2], t[3]));
    else
        c = fg3d_shade(&p, srcPyr, glsPyr, tris, nodesF, nodesI, x, y);
    device float* o = dst + a.dstBase + (y - a.dstY1) * a.dstStride + (x - a.dstX1) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

kernel void fg_trace(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                     device const float* glsPyr [[buffer(2)]], device const float* tris [[buffer(3)]],
                     device const float* nodesF [[buffer(4)]], device const int* nodesI [[buffer(5)]],
                     device float* aux [[buffer(6)]], device float* cv [[buffer(7)]],
                     constant FGWinArgs& a [[buffer(8)]], device const float* halfRes [[buffer(9)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.rw || j >= a.rh) return;
    device float* ax = aux + (j * a.rw + i) * FG_AUX_FLOATS;
    float t[FG_AUX_FLOATS];
    if (a.soft != 0 && fg3d_soft_up(halfRes, a.hw, a.hh, a.hx0, a.hy0, a.rx0 + i - a.fx0, a.ry0 + j - a.fy0,
                                    FG_SOFT_SPREAD, t)) {
        for (int k = 0; k < FG_AUX_FLOATS; ++k) ax[k] = t[k];
    } else {
        fg3d_trace(&p, srcPyr, glsPyr, tris, nodesF, nodesI, a.rx0 + i, a.ry0 + j, ax);
    }
    device float* c = cv + (j * a.rw + i) * FG_CV_FLOATS;
    for (int k = 0; k < FG_CV_FLOATS; ++k) c[k] = ax[k];
}

kernel void fg_atrous(device const float* aux [[buffer(0)]], device const float* cvIn [[buffer(1)]],
                      device float* cvOut [[buffer(2)]], constant FGWinArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.rw || j >= a.rh) return;
    fg3d_atrous(aux, cvIn, cvOut, a.rw, a.rh, i, j, a.step, a.sigma, a.frameH);
}

kernel void fg_finish(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                      device const float* cv [[buffer(2)]], device float* dst [[buffer(3)]],
                      constant FGWinArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = a.x0 + (int)gid.x, y = a.y0 + (int)gid.y;
    if (x >= a.x1 || y >= a.y1) return;
    device const float* c = cv + ((y - a.ry0) * a.rw + (x - a.rx0)) * FG_CV_FLOATS;
    fgv4 o = fg3d_finish(&p, srcPyr, x, y, fgv4m(c[0], c[1], c[2], c[3]));
    device float* d = dst + a.dstBase + (y - a.dstY1) * a.dstStride + (x - a.dstX1) * 4;
    d[0] = o.x; d[1] = o.y; d[2] = o.z; d[3] = o.w;
}

// Playback cache. Region (rx0, ry0, rw, rh) is the window, or the denoiser's
// padded region; a.pad != 0: also write the denoiser guides (aux).
kernel void fg_trace_rec(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                         device const float* glsPyr [[buffer(2)]], device const float* tris [[buffer(3)]],
                         device const float* nodesF [[buffer(4)]], device const int* nodesI [[buffer(5)]],
                         device float* cache [[buffer(6)]], device float* aux [[buffer(7)]],
                         constant FGWinArgs& a [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.rw || j >= a.rh) return;
    device float* ax = a.pad != 0 ? aux + (j * a.rw + i) * FG_AUX_FLOATS : nullptr;
    fg3d_trace_cache(&p, srcPyr, glsPyr, tris, nodesF, nodesI, a.rx0 + i, a.ry0 + j, ax,
                     cache + (j * a.rw + i) * FG_CACHE_FLOATS);
}

kernel void fg_replay_out(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                          device const float* glsPyr [[buffer(2)]], device const float* cache [[buffer(3)]],
                          device float* dst [[buffer(4)]], constant FGWinArgs& a [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]])
{
    const int x = a.x0 + (int)gid.x, y = a.y0 + (int)gid.y;
    if (x >= a.x1 || y >= a.y1) return;
    fgv4 c = fg3d_replay(&p, srcPyr, glsPyr, cache + ((y - a.ry0) * a.rw + (x - a.rx0)) * FG_CACHE_FLOATS);
    c = fg3d_finish(&p, srcPyr, x, y, c);
    device float* o = dst + a.dstBase + (y - a.dstY1) * a.dstStride + (x - a.dstX1) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

kernel void fg_replay_cv(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                         device const float* glsPyr [[buffer(2)]], device const float* cache [[buffer(3)]],
                         device const float* aux [[buffer(4)]], device float* cv [[buffer(5)]],
                         constant FGWinArgs& a [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.rw || j >= a.rh) return;
    fgv4 c = fg3d_replay(&p, srcPyr, glsPyr, cache + (j * a.rw + i) * FG_CACHE_FLOATS);
    device float* o = cv + (j * a.rw + i) * FG_CV_FLOATS;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w; o[4] = aux[(j * a.rw + i) * FG_AUX_FLOATS + 4];
}

// Soft areas: the half-resolution pre-pass (p = half-size frame at origin 0).
kernel void fg_trace_half(constant FG3DParams& p [[buffer(0)]], device const float* srcPyr [[buffer(1)]],
                          device const float* glsPyr [[buffer(2)]], device const float* tris [[buffer(3)]],
                          device const float* nodesF [[buffer(4)]], device const int* nodesI [[buffer(5)]],
                          device float* halfRes [[buffer(6)]], constant FGWinArgs& a [[buffer(7)]],
                          uint2 gid [[thread_position_in_grid]])
{
    const int i = (int)gid.x, j = (int)gid.y;
    if (i >= a.hw || j >= a.hh) return;
    fg3d_trace(&p, srcPyr, glsPyr, tris, nodesF, nodesI, a.hx0 + i, a.hy0 + j, halfRes + (j * a.hw + i) * FG_AUX_FLOATS);
}
