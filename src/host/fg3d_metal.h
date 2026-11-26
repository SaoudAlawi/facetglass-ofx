// FacetGlass 3D - Metal render of one render window. Images are MTLBuffers (as
// passed by an OFX host in Metal render mode); work is encoded on the host's
// command queue. Scene buffers are host memory and are uploaded per call.
// SPDX-License-Identifier: MIT
#pragma once

#include "fg3d_scene.h"

namespace fg3d {

struct MetalImage {
    void* buffer = nullptr;       // id<MTLBuffer>
    long long base = 0;           // float index of pixel (x1, y1) in the buffer
    int x1 = 0, y1 = 0, w = 0, h = 0;
    int stride = 0;               // floats per row (may be negative)
};

struct MetalFrame {
    void* queue = nullptr;        // id<MTLCommandQueue> (host's queue); null = own queue on the default device
    MetalImage src;               // main source (full RoD)
    MetalImage gls;               // refract source; buffer == nullptr when not connected
    int frameX = 0, frameY = 0, frameW = 0, frameH = 0;      // source RoD, pixels
    int glsFrameX = 0, glsFrameY = 0, glsFrameW = 0, glsFrameH = 0;
    void* dst = nullptr;          // id<MTLBuffer>
    long long dstBase = 0;        // float index of pixel (dstX1, dstY1)
    int dstX1 = 0, dstY1 = 0, dstStride = 0;
    int winX1 = 0, winY1 = 0, winX2 = 0, winY2 = 0;          // render window
    bool wait = false;            // block until the GPU has finished (tests)
    int cacheMode = 0;            // fg3d::CacheMode (fg3d_cache.h)
    bool soft = false;            // soft areas (half-resolution pre-pass); ignored with the cache
    void* cache = nullptr;        // id<MTLBuffer> from metalCacheAlloc, cacheFloats(...) floats
};

// Playback cache buffer on the queue's device (private storage); free with metalCacheFree.
void* metalCacheAlloc(void* queue, size_t bytes);
void metalCacheFree(void* buffer);

// Renders the window. `p` comes from makeParams; image fields are filled here.
// Returns nullptr on success, else an error text.
const char* renderMetal(const MetalFrame& f, FG3DParams p, const SceneBuffers& scene);

}  // namespace fg3d
