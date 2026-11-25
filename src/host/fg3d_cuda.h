// FacetGlass 3D - CUDA render of one render window. Image pointers are device
// memory (as passed by an OFX host in CUDA render mode); scene buffers are host
// memory and are uploaded per call (they are small).
// SPDX-License-Identifier: MIT
#pragma once

#include "fg3d_scene.h"

namespace fg3d {

struct CudaImage {
    const float* data = nullptr;  // device pointer to pixel (x1, y1)
    int x1 = 0, y1 = 0, w = 0, h = 0;
    int stride = 0;               // floats per row (may be negative)
};

struct CudaFrame {
    void* stream = nullptr;       // cudaStream_t (host's stream), may be null
    CudaImage src;                // main source (full RoD)
    CudaImage gls;                // refract source; data == nullptr when not connected
    int frameX = 0, frameY = 0, frameW = 0, frameH = 0;      // source RoD, pixels
    int glsFrameX = 0, glsFrameY = 0, glsFrameW = 0, glsFrameH = 0;
    float* dst = nullptr;         // device pointer to pixel (dstX1, dstY1)
    int dstX1 = 0, dstY1 = 0, dstStride = 0;
    int winX1 = 0, winY1 = 0, winX2 = 0, winY2 = 0;          // render window
};

// Renders the window. `p` comes from makeParams (frame, scene, look fields);
// image fields are filled here. Returns nullptr on success, else an error text.
const char* renderCUDA(const CudaFrame& f, FG3DParams p, const SceneBuffers& scene);

}  // namespace fg3d
