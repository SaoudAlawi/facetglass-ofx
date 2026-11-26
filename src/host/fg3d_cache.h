// FacetGlass 3D - playback cache, host side (see "playback cache" in fg3d_core.h).
// A render either traces normally, traces and records (output identical to a
// normal trace), or replays a recording against new footage. The buffer covers
// the render region: the window, or the denoiser's padded region when Denoise
// is on (then the denoiser guides are stored after the records).
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "fg3d_params.h"

namespace fg3d {

enum CacheMode { kCacheNone = 0, kCacheRecord = 1, kCacheReplay = 2 };

struct CacheRegion {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// Region a render of window [x0, x1) x [y0, y1) traces.
inline CacheRegion cacheRegion(const FG3DParams& p, int x0, int x1, int y0, int y1)
{
    CacheRegion r;
    if (p.denoise > 0.0f) {
        const int fx0 = (int)p.frameX, fy0 = (int)p.frameY, fx1 = fx0 + (int)p.frameW, fy1 = fy0 + (int)p.frameH;
        r.x0 = std::max(x0 - FG_DENOISE_PAD, fx0);
        r.y0 = std::max(y0 - FG_DENOISE_PAD, fy0);
        r.w = std::max(0, std::min(x1 + FG_DENOISE_PAD, fx1) - r.x0);
        r.h = std::max(0, std::min(y1 + FG_DENOISE_PAD, fy1) - r.y0);
    } else {
        r.x0 = x0; r.y0 = y0; r.w = std::max(0, x1 - x0); r.h = std::max(0, y1 - y0);
    }
    return r;
}

// Floats a cache for this region needs (records, then denoiser guides).
inline size_t cacheFloats(const FG3DParams& p, const CacheRegion& r)
{
    const size_t n = (size_t)r.w * r.h;
    return n * FG_CACHE_FLOATS + (p.denoise > 0.0f ? n * FG_AUX_FLOATS : 0);
}

// Identity of everything a recording depends on: the parameters (which include
// the frame, transform, optics and look), the scene, the refract source's size
// and the render window. Footage pixels are deliberately not part of it.
inline std::vector<unsigned char> cacheKey(const FG3DParams& p, const void* scene, int glsW, int glsH, int x0, int x1,
                                           int y0, int y1, int backend)
{
    std::vector<unsigned char> k(sizeof(FG3DParams) + sizeof(void*) + 7 * sizeof(int));
    unsigned char* o = k.data();
    std::memcpy(o, &p, sizeof p); o += sizeof p;
    std::memcpy(o, &scene, sizeof scene); o += sizeof scene;
    const int v[7] = { glsW, glsH, x0, x1, y0, y1, backend };
    std::memcpy(o, v, sizeof v);
    return k;
}

// Per-instance cache state. Decide() is called with the lock held for the
// whole render.
struct CacheState {
    std::vector<unsigned char> lastKey, cacheKey;
    bool valid = false;
    // Seen twice in a row: record; recorded with this key: replay.
    CacheMode decide(const std::vector<unsigned char>& key)
    {
        if (valid && key == cacheKey) return kCacheReplay;
        if (key == lastKey) return kCacheRecord;
        lastKey = key;
        valid = false;
        return kCacheNone;
    }
    void recorded(const std::vector<unsigned char>& key)
    {
        cacheKey = key;
        lastKey = key;
        valid = true;
    }
};

}  // namespace fg3d

namespace fg3d {

// Soft areas (see fg3d_soft_up in fg3d_core.h): the half-resolution pre-pass.
struct SoftRegion {
    int hx0 = 0, hy0 = 0, hw = 0, hh = 0;  // half pixels, frame-relative
};

// Parameters of the half-resolution frame (origin 0, half size).
inline FG3DParams softHalfParams(const FG3DParams& p)
{
    FG3DParams h = p;
    h.frameX = 0.0f; h.frameY = 0.0f;
    h.frameW = p.frameW * 0.5f; h.frameH = p.frameH * 0.5f;
    return h;
}

// Half pixels covering full-resolution region (x0, y0, w, h), with a margin of one.
inline SoftRegion softRegion(const FG3DParams& p, int x0, int y0, int w, int h)
{
    SoftRegion r;
    const int fx0 = x0 - (int)p.frameX, fy0 = y0 - (int)p.frameY;
    const int maxW = ((int)p.frameW + 1) / 2, maxH = ((int)p.frameH + 1) / 2;
    r.hx0 = std::max(0, fx0 / 2 - 1);
    r.hy0 = std::max(0, fy0 / 2 - 1);
    r.hw = std::max(0, std::min(maxW, (fx0 + w) / 2 + 2) - r.hx0);
    r.hh = std::max(0, std::min(maxH, (fy0 + h) / 2 + 2) - r.hy0);
    return r;
}

}  // namespace fg3d
