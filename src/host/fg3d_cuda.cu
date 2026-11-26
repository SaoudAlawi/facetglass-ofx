// FacetGlass 3D - CUDA backend: the three passes of fg3d_core.h as kernels.
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <utility>

#include "fg3d_backend_cuda.cuh"
#include "fg3d_cache.h"
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

// Soft areas: half-resolution block and frame origin (soft = 0: unused).
struct SoftArgs {
    const float* half;
    int hx0, hy0, hw, hh, soft, fx0, fy0;
};

__global__ void halfKernel(FG3DParams ph, const float* srcPyr, const float* glsPyr, const float* tris,
                           const float* nodesF, const int* nodesI, SoftArgs sa, float* half)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sa.hw || j >= sa.hh) return;
    FG3DParams pp = ph;
    fg3d_trace(&pp, srcPyr, glsPyr, tris, nodesF, nodesI, sa.hx0 + i, sa.hy0 + j,
               half + ((long long)j * sa.hw + i) * FG_AUX_FLOATS);
}

__global__ void shadeKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* tris,
                            const float* nodesF, const int* nodesI, float* dst, int dstX1, int dstY1, int dstStride,
                            int x0, int y0, int x1, int y1, SoftArgs sa)
{
    const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x1 || y >= y1) return;
    FG3DParams pp = p;
    fgv4 c;
    float t[FG_AUX_FLOATS];
    if (sa.soft && fg3d_soft_up(sa.half, sa.hw, sa.hh, sa.hx0, sa.hy0, x - sa.fx0, y - sa.fy0, FG_SOFT_SPREAD, t))
        c = fg3d_finish(&pp, srcPyr, x, y, fgv4m(t[0], t[1], t[2], t[3]));
    else
        c = fg3d_shade(&pp, srcPyr, glsPyr, tris, nodesF, nodesI, x, y);
    float* o = dst + (long long)(y - dstY1) * dstStride + (long long)(x - dstX1) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

__global__ void traceKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* tris,
                            const float* nodesF, const int* nodesI, float* aux, float* cv, int rx0, int ry0, int w, int h,
                            SoftArgs sa)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    FG3DParams pp = p;
    float* a = aux + ((long long)j * w + i) * FG_AUX_FLOATS;
    float t[FG_AUX_FLOATS];
    if (sa.soft && fg3d_soft_up(sa.half, sa.hw, sa.hh, sa.hx0, sa.hy0, rx0 + i - sa.fx0, ry0 + j - sa.fy0,
                                FG_SOFT_SPREAD, t)) {
        for (int k = 0; k < FG_AUX_FLOATS; ++k) a[k] = t[k];
    } else {
        fg3d_trace(&pp, srcPyr, glsPyr, tris, nodesF, nodesI, rx0 + i, ry0 + j, a);
    }
    float* c = cv + ((long long)j * w + i) * FG_CV_FLOATS;
    for (int k = 0; k < FG_CV_FLOATS; ++k) c[k] = a[k];
}

__global__ void atrousKernel(const float* aux, const float* in, float* out, int w, int h, int step, float sigma,
                             float frameH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    fg3d_atrous(aux, in, out, w, h, i, j, step, sigma, frameH);
}

__global__ void finishKernel(FG3DParams p, const float* srcPyr, const float* cv, int rx0, int ry0, int w, float* dst,
                             int dstX1, int dstY1, int dstStride, int x0, int y0, int x1, int y1)
{
    const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x1 || y >= y1) return;
    FG3DParams pp = p;
    const float* c = cv + ((long long)(y - ry0) * w + (x - rx0)) * FG_CV_FLOATS;
    fgv4 o = fg3d_finish(&pp, srcPyr, x, y, fgv4m(c[0], c[1], c[2], c[3]));
    float* d = dst + (long long)(y - dstY1) * dstStride + (long long)(x - dstX1) * 4;
    d[0] = o.x; d[1] = o.y; d[2] = o.z; d[3] = o.w;
}

// Playback cache (see fg3d_core.h). Region (rx0, ry0, w, h).
__global__ void recordKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* tris,
                             const float* nodesF, const int* nodesI, float* cache, float* aux, int rx0, int ry0, int w,
                             int h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    FG3DParams pp = p;
    float* a = aux ? aux + ((long long)j * w + i) * FG_AUX_FLOATS : nullptr;
    fg3d_trace_cache(&pp, srcPyr, glsPyr, tris, nodesF, nodesI, rx0 + i, ry0 + j, a,
                     cache + ((long long)j * w + i) * FG_CACHE_FLOATS);
}

__global__ void replayOutKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* cache, int rx0,
                                int ry0, int rw, float* dst, int dstX1, int dstY1, int dstStride, int x0, int y0, int x1,
                                int y1)
{
    const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x1 || y >= y1) return;
    FG3DParams pp = p;
    fgv4 c = fg3d_replay(&pp, srcPyr, glsPyr, cache + ((long long)(y - ry0) * rw + (x - rx0)) * FG_CACHE_FLOATS);
    c = fg3d_finish(&pp, srcPyr, x, y, c);
    float* o = dst + (long long)(y - dstY1) * dstStride + (long long)(x - dstX1) * 4;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
}

__global__ void replayCvKernel(FG3DParams p, const float* srcPyr, const float* glsPyr, const float* cache,
                               const float* aux, float* cv, int w, int h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    FG3DParams pp = p;
    const long long k = (long long)j * w + i;
    fgv4 c = fg3d_replay(&pp, srcPyr, glsPyr, cache + k * FG_CACHE_FLOATS);
    float* o = cv + k * FG_CV_FLOATS;
    o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w; o[4] = aux[k * FG_AUX_FLOATS + 4];
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

    // Denoiser region: the window padded by the filter's reach, clamped to the frame.
    const bool dn = p.denoise > 0.0f;
    const int rx0 = std::max(f.winX1 - FG_DENOISE_PAD, f.frameX), rx1 = std::min(f.winX2 + FG_DENOISE_PAD, f.frameX + f.frameW);
    const int ry0 = std::max(f.winY1 - FG_DENOISE_PAD, f.frameY), ry1 = std::min(f.winY2 + FG_DENOISE_PAD, f.frameY + f.frameH);
    const int rw = dn ? std::max(0, rx1 - rx0) : 0, rh = dn ? std::max(0, ry1 - ry0) : 0;
    const size_t auxBytes = (size_t)rw * rh * FG_AUX_FLOATS * 4, cvBytes = (size_t)rw * rh * FG_CV_FLOATS * 4;

    // One allocation for everything this render needs.
    auto align = [](size_t n) { return (n + 255) & ~(size_t)255; };
    const size_t oSrc = 0, oGls = align(oSrc + srcFloats * 4), oTri = align(oGls + glsFloats * 4),
                 oNF = align(oTri + triBytes + 16), oNI = align(oNF + nfBytes + 16), oAux = align(oNI + niBytes + 16),
                 oCvA = align(oAux + auxBytes), oCvB = align(oCvA + cvBytes), total = align(oCvB + cvBytes);
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
    const int cmode = f.cache ? f.cacheMode : fg3d::kCacheNone;
    // Soft areas: half-resolution pre-pass over the traced region.
    SoftArgs sa = {};
    float* halfBuf = nullptr;
    if (f.soft && cmode == fg3d::kCacheNone) {
        const fg3d::CacheRegion reg = fg3d::cacheRegion(p, f.winX1, f.winX2, f.winY1, f.winY2);
        const fg3d::SoftRegion sr = fg3d::softRegion(p, reg.x0, reg.y0, reg.w, reg.h);
        if (sr.hw > 1 && sr.hh > 1 &&
            cudaMallocAsync(reinterpret_cast<void**>(&halfBuf), (size_t)sr.hw * sr.hh * FG_AUX_FLOATS * 4, st) == cudaSuccess) {
            sa.half = halfBuf; sa.hx0 = sr.hx0; sa.hy0 = sr.hy0; sa.hw = sr.hw; sa.hh = sr.hh; sa.soft = 1;
            sa.fx0 = (int)p.frameX; sa.fy0 = (int)p.frameY;
            const dim3 b(16, 8);
            halfKernel<<<grid2d(sr.hw, sr.hh, b), b, 0, st>>>(fg3d::softHalfParams(p), srcPyr, glsPyr, tris, nodesF,
                                                              nodesI, sa, halfBuf);
        } else {
            halfBuf = nullptr;
        }
    }
    if (cmode != fg3d::kCacheNone && !dn && ww > 0 && wh > 0) {
        // Region = window. Recording also outputs the replay (no change in look after frame 2).
        const dim3 b(16, 8);
        if (cmode == fg3d::kCacheRecord)
            recordKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, glsPyr, tris, nodesF, nodesI, f.cache, nullptr,
                                                          f.winX1, f.winY1, ww, wh);
        replayOutKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, glsPyr, f.cache, f.winX1, f.winY1, ww, f.dst,
                                                         f.dstX1, f.dstY1, f.dstStride, f.winX1, f.winY1, f.winX2,
                                                         f.winY2);
    } else if (cmode != fg3d::kCacheNone && dn && ww > 0 && wh > 0 && rw > 0 && rh > 0) {
        float* caux = f.cache + (size_t)rw * rh * FG_CACHE_FLOATS;
        float* cur = reinterpret_cast<float*>(mem + oCvA);
        float* nxt = reinterpret_cast<float*>(mem + oCvB);
        const dim3 b(16, 8);
        if (cmode == fg3d::kCacheRecord)
            recordKernel<<<grid2d(rw, rh, b), b, 0, st>>>(p, srcPyr, glsPyr, tris, nodesF, nodesI, f.cache, caux, rx0,
                                                          ry0, rw, rh);
        replayCvKernel<<<grid2d(rw, rh, b), b, 0, st>>>(p, srcPyr, glsPyr, f.cache, caux, cur, rw, rh);
        for (int it = 0; it < FG_DENOISE_ITERS; ++it) {
            atrousKernel<<<grid2d(rw, rh, b), b, 0, st>>>(caux, cur, nxt, rw, rh, 1 << it, p.denoise, p.frameH);
            std::swap(cur, nxt);
        }
        finishKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, cur, rx0, ry0, rw, f.dst, f.dstX1, f.dstY1, f.dstStride,
                                                      f.winX1, f.winY1, f.winX2, f.winY2);
    } else if (dn && ww > 0 && wh > 0 && rw > 0 && rh > 0) {
        float* aux = reinterpret_cast<float*>(mem + oAux);
        float* cur = reinterpret_cast<float*>(mem + oCvA);
        float* nxt = reinterpret_cast<float*>(mem + oCvB);
        const dim3 b(16, 8);
        traceKernel<<<grid2d(rw, rh, b), b, 0, st>>>(p, srcPyr, glsPyr, tris, nodesF, nodesI, aux, cur, rx0, ry0, rw, rh, sa);
        for (int it = 0; it < FG_DENOISE_ITERS; ++it) {
            atrousKernel<<<grid2d(rw, rh, b), b, 0, st>>>(aux, cur, nxt, rw, rh, 1 << it, p.denoise, p.frameH);
            std::swap(cur, nxt);
        }
        finishKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, cur, rx0, ry0, rw, f.dst, f.dstX1, f.dstY1, f.dstStride,
                                                      f.winX1, f.winY1, f.winX2, f.winY2);
    } else if (ww > 0 && wh > 0) {
        const dim3 b(16, 8);
        shadeKernel<<<grid2d(ww, wh, b), b, 0, st>>>(p, srcPyr, glsPyr, tris, nodesF, nodesI, f.dst, f.dstX1, f.dstY1,
                                                     f.dstStride, f.winX1, f.winY1, f.winX2, f.winY2, sa);
    }
    e = cudaGetLastError();
    if (halfBuf) cudaFreeAsync(halfBuf, st);
    cudaFreeAsync(mem, st);
    if (e != cudaSuccess) {
        std::snprintf(err, sizeof err, "kernel launch: %s", cudaGetErrorString(e));
        return err;
    }
    return nullptr;
}

float* cudaCacheAlloc(size_t bytes)
{
    void* m = nullptr;
    if (bytes == 0 || cudaMalloc(&m, bytes) != cudaSuccess) return nullptr;
    return static_cast<float*>(m);
}

void cudaCacheFree(float* buffer)
{
    if (buffer) cudaFree(buffer);
}

}  // namespace fg3d
