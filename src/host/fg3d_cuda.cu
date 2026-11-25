// FacetGlass 3D - CUDA backend: the three passes of fg3d_core.h as kernels.
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <cstdio>

#include "fg3d_backend_cuda.cuh"
#include "fg3d_cuda.h"

namespace {

__global__ void decodeKernel(const float* src, int x1, int y1, int w, int h, int stride, int fx, int fy, int fw,
                             int fh, int cs, float* lvl0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= fw || j >= fh) return;
    int sx = fx + i - x1, sy = fy + j - y1;
    sx = sx < 0 ? 0 : (sx > w - 1 ? w - 1 : sx);
    sy = sy < 0 ? 0 : (sy > h - 1 ? h - 1 : sy);
    fgv4 c = fg3d_decode_pixel(cs, src + (long long)sy * stride + (long long)sx * 4);
    float* o = lvl0 + ((long long)j * fw + i) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

__global__ void mipKernel(const float* prev, int lw, int lh, float* next, int nw, int nh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nw || y >= nh) return;
    fgv4 c = fg3d_mip_pixel(prev, lw, lh, x, y);
    float* o = next + ((long long)y * nw + x) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

__global__ void shadeKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* tris,
                            const float* nodesF, const int* nodesI, float* dst, int dstX1, int dstY1, int dstStride,
                            int x0, int y0, int x1, int y1)
{
    const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x1 || y >= y1) return;
    FG3DParams pp = p;
    fgv4 c = fg3d_shade(&pp, srcPyr, glsPyr, tris, nodesF, nodesI, x, y);
    float* o = dst + (long long)(y - dstY1) * dstStride + (long long)(x - dstX1) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

dim3 grid2d(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

// Builds a linear pyramid for one image in device memory `pyr`.
void buildPyramid(cudaStream_t st, const fg3d::CudaImage& img, int fx, int fy, int fw, int fh, int cs, int levels,
                  float* pyr)
{
    const dim3 b(16, 16);
    decodeKernel<<<grid2d(fw, fh, b), b, 0, st>>>(img.data, img.x1, img.y1, img.w, img.h, img.stride, fx, fy, fw, fh,
                                                  cs, pyr);
    size_t off = 0;
    for (int l = 0; l + 1 < levels; ++l) {
        const int lw = fw >> l > 0 ? fw >> l : 1, lh = fh >> l > 0 ? fh >> l : 1;
        const int nw = fw >> (l + 1) > 0 ? fw >> (l + 1) : 1, nh = fh >> (l + 1) > 0 ? fh >> (l + 1) : 1;
        float* prev = pyr + off;
        float* next = prev + (size_t)lw * lh * 4;
        mipKernel<<<grid2d(nw, nh, b), b, 0, st>>>(prev, lw, lh, next, nw, nh);
        off += (size_t)lw * lh * 4;
    }
}

}  // namespace

namespace fg3d {

const char* renderCUDA(const CudaFrame& f, FG3DParams p, const SceneBuffers& scene)
{
    static thread_local char err[256];
    cudaStream_t st = static_cast<cudaStream_t>(f.stream);
    const bool hasGls = f.gls.data != nullptr;

    const int srcLevels = mipLevels(f.frameW, f.frameH);
    const int glsLevels = hasGls ? mipLevels(f.glsFrameW, f.glsFrameH) : srcLevels;
    const size_t srcFloats = pyramidFloats(f.frameW, f.frameH, srcLevels);
    const size_t glsFloats = hasGls ? pyramidFloats(f.glsFrameW, f.glsFrameH, glsLevels) : 0;
    const size_t triBytes = scene.tris.size() * sizeof(float);
    const size_t nfBytes = scene.nodesF.size() * sizeof(float);
    const size_t niBytes = scene.nodesI.size() * sizeof(int);

    // One allocation for everything this render needs.
    auto align = [](size_t n) { return (n + 255) & ~(size_t)255; };
    const size_t oSrc = 0, oGls = align(oSrc + srcFloats * 4), oTri = align(oGls + glsFloats * 4),
                 oNF = align(oTri + triBytes + 16), oNI = align(oNF + nfBytes + 16), total = align(oNI + niBytes + 16);
    char* mem = nullptr;
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&mem), total, st);
    if (e != cudaSuccess) {
        std::snprintf(err, sizeof err, "cudaMallocAsync(%zu): %s", total, cudaGetErrorString(e));
        return err;
    }
    float* srcPyr = reinterpret_cast<float*>(mem + oSrc);
    float* glsPyr = hasGls ? reinterpret_cast<float*>(mem + oGls) : srcPyr;
    float* tris = reinterpret_cast<float*>(mem + oTri);
    float* nodesF = reinterpret_cast<float*>(mem + oNF);
    int* nodesI = reinterpret_cast<int*>(mem + oNI);
    if (triBytes) cudaMemcpyAsync(tris, scene.tris.data(), triBytes, cudaMemcpyHostToDevice, st);
    if (nfBytes) cudaMemcpyAsync(nodesF, scene.nodesF.data(), nfBytes, cudaMemcpyHostToDevice, st);
    if (niBytes) cudaMemcpyAsync(nodesI, scene.nodesI.data(), niBytes, cudaMemcpyHostToDevice, st);

    buildPyramid(st, f.src, f.frameX, f.frameY, f.frameW, f.frameH, p.colorSpace, srcLevels, srcPyr);
    if (hasGls) buildPyramid(st, f.gls, f.glsFrameX, f.glsFrameY, f.glsFrameW, f.glsFrameH, p.colorSpace, glsLevels, glsPyr);

    p.srcW = f.frameW; p.srcH = f.frameH; p.srcLevels = srcLevels;
    p.glsW = hasGls ? f.glsFrameW : f.frameW;
    p.glsH = hasGls ? f.glsFrameH : f.frameH;
    p.glsLevels = glsLevels;
    p.useSource = hasGls ? 1 : 0;

    const int ww = f.winX2 - f.winX1, wh = f.winY2 - f.winY1;
    if (ww > 0 && wh > 0) {
        const dim3 b(16, 8);
        shadeKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, glsPyr, tris, nodesF, nodesI, f.dst, f.dstX1, f.dstY1,
                                                     f.dstStride, f.winX1, f.winY1, f.winX2, f.winY2);
    }
    e = cudaGetLastError();
    cudaFreeAsync(mem, st);
    if (e != cudaSuccess) {
        std::snprintf(err, sizeof err, "kernel launch: %s", cudaGetErrorString(e));
        return err;
    }
    return nullptr;
}

}  // namespace fg3d
