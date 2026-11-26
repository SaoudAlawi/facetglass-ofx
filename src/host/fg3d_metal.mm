// FacetGlass 3D - Metal backend host side (Objective-C++, ARC).
// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>

#include "fg3d_cache.h"
#include "fg3d_metal.h"
#include "fg3d_metal_src.h"  // generated: const char* kFG3DMetalSource

namespace {

// Must match fg3d_metal_kernels.metal.
struct FGImgArgs {
    int base, x1, y1, w, h, stride;
    int fx, fy, fw, fh, cs, pad;
};
struct FGMipArgs { int lw, lh, nw, nh; };
struct FGWinArgs {
    int dstBase, dstX1, dstY1, dstStride;
    int x0, y0, x1, y1;
    int rx0, ry0, rw, rh;
    int step;
    float sigma, frameH;
    int pad;
    int hx0, hy0, hw, hh;
    int soft, fx0, fy0, pad2;
};

struct Pipelines {
    id<MTLComputePipelineState> decode, mip, shade, trace, atrous, finish, rec, replayOut, replayCv, half;
};

// Compiled once per device (the source compiles in a fraction of a second).
const Pipelines* pipelinesFor(id<MTLDevice> dev, const char** err)
{
    static std::mutex m;
    static std::map<uint64_t, Pipelines*> cache;
    static char msg[512];
    std::lock_guard<std::mutex> lock(m);
    const uint64_t key = dev.registryID;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    MTLCompileOptions* opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kFG3DMetalSource] options:opt error:&e];
    if (!lib) {
        std::snprintf(msg, sizeof msg, "Metal compile: %s", e ? e.localizedDescription.UTF8String : "unknown error");
        *err = msg;
        return nullptr;
    }
    auto make = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) return nil;
        NSError* pe = nil;
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!ps) std::snprintf(msg, sizeof msg, "Metal pipeline %s: %s", name, pe ? pe.localizedDescription.UTF8String : "?");
        return ps;
    };
    auto* p = new Pipelines;  // lives for the process
    p->decode = make("fg_decode");
    p->mip = make("fg_mip");
    p->shade = make("fg_shade");
    p->trace = make("fg_trace");
    p->atrous = make("fg_atrous");
    p->finish = make("fg_finish");
    p->rec = make("fg_trace_rec");
    p->replayOut = make("fg_replay_out");
    p->replayCv = make("fg_replay_cv");
    p->half = make("fg_trace_half");
    if (!p->decode || !p->mip || !p->shade || !p->trace || !p->atrous || !p->finish || !p->rec || !p->replayOut ||
        !p->replayCv || !p->half) {
        delete p;
        *err = msg;
        return nullptr;
    }
    cache[key] = p;
    return p;
}

void dispatch2d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int w, int h)
{
    [enc setComputePipelineState:ps];
    const NSUInteger tw = std::min<NSUInteger>(16, ps.threadExecutionWidth);
    const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(8, ps.maxTotalThreadsPerThreadgroup / tw));
    [enc dispatchThreads:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

// Encodes decode + mip passes for one image into `pyr` (starting at byte 0).
void encodePyramid(id<MTLComputeCommandEncoder> enc, const Pipelines& ps, const fg3d::MetalImage& img, int fx, int fy,
                   int fw, int fh, int cs, int levels, id<MTLBuffer> pyr)
{
    FGImgArgs a = { (int)img.base, img.x1, img.y1, img.w, img.h, img.stride, fx, fy, fw, fh, cs, 0 };
    [enc setBuffer:(__bridge id<MTLBuffer>)img.buffer offset:0 atIndex:0];
    [enc setBuffer:pyr offset:0 atIndex:1];
    [enc setBytes:&a length:sizeof a atIndex:2];
    dispatch2d(enc, ps.decode, fw, fh);
    size_t off = 0;
    for (int l = 0; l + 1 < levels; ++l) {
        const int lw = std::max(1, fw >> l), lh = std::max(1, fh >> l);
        const int nw = std::max(1, fw >> (l + 1)), nh = std::max(1, fh >> (l + 1));
        const size_t next = off + (size_t)lw * lh * 4;
        FGMipArgs m = { lw, lh, nw, nh };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setBuffer:pyr offset:off * 4 atIndex:0];
        [enc setBuffer:pyr offset:next * 4 atIndex:1];
        [enc setBytes:&m length:sizeof m atIndex:2];
        dispatch2d(enc, ps.mip, nw, nh);
        off = next;
    }
}

id<MTLBuffer> upload(id<MTLDevice> dev, const void* data, size_t bytes)
{
    if (bytes == 0) return [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    return [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}

}  // namespace

namespace fg3d {

const char* renderMetal(const MetalFrame& f, FG3DParams p, const SceneBuffers& scene)
{
    static thread_local char err[256];
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)f.queue;
        id<MTLDevice> dev = queue ? queue.device : MTLCreateSystemDefaultDevice();
        if (!dev) return "no Metal device";
        if (!queue) {
            static std::mutex qm;
            static id<MTLCommandQueue> own = nil;
            std::lock_guard<std::mutex> lock(qm);
            if (!own) own = [dev newCommandQueue];
            queue = own;
        }
        const char* perr = nullptr;
        const Pipelines* ps = pipelinesFor(dev, &perr);
        if (!ps) return perr;

        const bool hasGls = f.gls.buffer != nullptr;
        const int srcLevels = mipLevels(f.frameW, f.frameH);
        const int glsLevels = hasGls ? mipLevels(f.glsFrameW, f.glsFrameH) : srcLevels;
        const size_t srcFloats = pyramidFloats(f.frameW, f.frameH, srcLevels);
        const size_t glsFloats = hasGls ? pyramidFloats(f.glsFrameW, f.glsFrameH, glsLevels) : 0;
        if (srcFloats >= 0x7fffffff || glsFloats >= 0x7fffffff) return "frame too large";

        // Denoiser region: the window padded by the filter's reach, clamped to the frame.
        const bool dn = p.denoise > 0.0f;
        const int rx0 = std::max(f.winX1 - FG_DENOISE_PAD, f.frameX), rx1 = std::min(f.winX2 + FG_DENOISE_PAD, f.frameX + f.frameW);
        const int ry0 = std::max(f.winY1 - FG_DENOISE_PAD, f.frameY), ry1 = std::min(f.winY2 + FG_DENOISE_PAD, f.frameY + f.frameH);
        const int rw = dn ? std::max(0, rx1 - rx0) : 0, rh = dn ? std::max(0, ry1 - ry0) : 0;
        const int ww = f.winX2 - f.winX1, wh = f.winY2 - f.winY1;
        if (ww <= 0 || wh <= 0) return nullptr;

        id<MTLBuffer> srcPyr = [dev newBufferWithLength:srcFloats * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> glsPyr = hasGls ? [dev newBufferWithLength:glsFloats * 4 options:MTLResourceStorageModePrivate] : srcPyr;
        id<MTLBuffer> tris = upload(dev, scene.tris.data(), scene.tris.size() * sizeof(float));
        id<MTLBuffer> nodesF = upload(dev, scene.nodesF.data(), scene.nodesF.size() * sizeof(float));
        id<MTLBuffer> nodesI = upload(dev, scene.nodesI.data(), scene.nodesI.size() * sizeof(int));
        if (!srcPyr || !glsPyr || !tris || !nodesF || !nodesI) return "Metal buffer allocation failed";

        p.srcW = f.frameW; p.srcH = f.frameH; p.srcLevels = srcLevels;
        p.glsW = hasGls ? f.glsFrameW : f.frameW;
        p.glsH = hasGls ? f.glsFrameH : f.frameH;
        p.glsLevels = glsLevels;
        p.useSource = hasGls ? 1 : 0;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encodePyramid(enc, *ps, f.src, f.frameX, f.frameY, f.frameW, f.frameH, p.colorSpace, srcLevels, srcPyr);
        if (hasGls)
            encodePyramid(enc, *ps, f.gls, f.glsFrameX, f.glsFrameY, f.glsFrameW, f.glsFrameH, p.colorSpace, glsLevels, glsPyr);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)f.dst;
        FGWinArgs a = {};
        a.dstBase = (int)f.dstBase; a.dstX1 = f.dstX1; a.dstY1 = f.dstY1; a.dstStride = f.dstStride;
        a.x0 = f.winX1; a.y0 = f.winY1; a.x1 = f.winX2; a.y1 = f.winY2;
        a.rx0 = rx0; a.ry0 = ry0; a.rw = rw; a.rh = rh;
        a.sigma = p.denoise; a.frameH = p.frameH;

        auto bindScene = [&]() {
            [enc setBytes:&p length:sizeof p atIndex:0];
            [enc setBuffer:srcPyr offset:0 atIndex:1];
            [enc setBuffer:glsPyr offset:0 atIndex:2];
            [enc setBuffer:tris offset:0 atIndex:3];
            [enc setBuffer:nodesF offset:0 atIndex:4];
            [enc setBuffer:nodesI offset:0 atIndex:5];
        };
        id<MTLBuffer> cacheBuf = (__bridge id<MTLBuffer>)f.cache;
        const int cmode = cacheBuf ? f.cacheMode : kCacheNone;
        // Soft areas: half-resolution pre-pass over the traced region.
        id<MTLBuffer> half = srcPyr;  // placeholder binding when unused
        if (f.soft && cmode == kCacheNone) {
            const CacheRegion reg = cacheRegion(p, f.winX1, f.winX2, f.winY1, f.winY2);
            const SoftRegion sr = softRegion(p, reg.x0, reg.y0, reg.w, reg.h);
            if (sr.hw > 1 && sr.hh > 1) {
                id<MTLBuffer> hb = [dev newBufferWithLength:(size_t)sr.hw * sr.hh * FG_AUX_FLOATS * 4
                                                    options:MTLResourceStorageModePrivate];
                if (hb) {
                    half = hb;
                    a.soft = 1; a.hx0 = sr.hx0; a.hy0 = sr.hy0; a.hw = sr.hw; a.hh = sr.hh;
                    a.fx0 = (int)p.frameX; a.fy0 = (int)p.frameY;
                    FG3DParams ph = softHalfParams(p);
                    [enc setBytes:&ph length:sizeof ph atIndex:0];
                    [enc setBuffer:srcPyr offset:0 atIndex:1];
                    [enc setBuffer:glsPyr offset:0 atIndex:2];
                    [enc setBuffer:tris offset:0 atIndex:3];
                    [enc setBuffer:nodesF offset:0 atIndex:4];
                    [enc setBuffer:nodesI offset:0 atIndex:5];
                    [enc setBuffer:half offset:0 atIndex:6];
                    [enc setBytes:&a length:sizeof a atIndex:7];
                    dispatch2d(enc, ps->half, sr.hw, sr.hh);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }
        }
        if (cmode != kCacheNone && !dn) {
            // Region = window. Recording also outputs the replay (no change in look after frame 2).
            a.rx0 = f.winX1; a.ry0 = f.winY1; a.rw = ww; a.rh = wh; a.pad = 0;
            if (cmode == kCacheRecord) {
                bindScene();
                [enc setBuffer:cacheBuf offset:0 atIndex:6];
                [enc setBuffer:cacheBuf offset:0 atIndex:7];
                [enc setBytes:&a length:sizeof a atIndex:8];
                dispatch2d(enc, ps->rec, ww, wh);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc setBytes:&p length:sizeof p atIndex:0];
            [enc setBuffer:srcPyr offset:0 atIndex:1];
            [enc setBuffer:glsPyr offset:0 atIndex:2];
            [enc setBuffer:cacheBuf offset:0 atIndex:3];
            [enc setBuffer:dst offset:0 atIndex:4];
            [enc setBytes:&a length:sizeof a atIndex:5];
            dispatch2d(enc, ps->replayOut, ww, wh);
        } else if (cmode != kCacheNone && rw > 0 && rh > 0) {
            const size_t auxOff = (size_t)rw * rh * FG_CACHE_FLOATS * 4;
            id<MTLBuffer> cur = [dev newBufferWithLength:(size_t)rw * rh * FG_CV_FLOATS * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> nxt = [dev newBufferWithLength:(size_t)rw * rh * FG_CV_FLOATS * 4 options:MTLResourceStorageModePrivate];
            if (!cur || !nxt) { [enc endEncoding]; return "Metal buffer allocation failed"; }
            a.pad = 1;
            if (cmode == kCacheRecord) {
                bindScene();
                [enc setBuffer:cacheBuf offset:0 atIndex:6];
                [enc setBuffer:cacheBuf offset:auxOff atIndex:7];
                [enc setBytes:&a length:sizeof a atIndex:8];
                dispatch2d(enc, ps->rec, rw, rh);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc setBytes:&p length:sizeof p atIndex:0];
            [enc setBuffer:srcPyr offset:0 atIndex:1];
            [enc setBuffer:glsPyr offset:0 atIndex:2];
            [enc setBuffer:cacheBuf offset:0 atIndex:3];
            [enc setBuffer:cacheBuf offset:auxOff atIndex:4];
            [enc setBuffer:cur offset:0 atIndex:5];
            [enc setBytes:&a length:sizeof a atIndex:6];
            dispatch2d(enc, ps->replayCv, rw, rh);
            for (int it = 0; it < FG_DENOISE_ITERS; ++it) {
                a.step = 1 << it;
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setBuffer:cacheBuf offset:auxOff atIndex:0];
                [enc setBuffer:cur offset:0 atIndex:1];
                [enc setBuffer:nxt offset:0 atIndex:2];
                [enc setBytes:&a length:sizeof a atIndex:3];
                dispatch2d(enc, ps->atrous, rw, rh);
                std::swap(cur, nxt);
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setBytes:&p length:sizeof p atIndex:0];
            [enc setBuffer:srcPyr offset:0 atIndex:1];
            [enc setBuffer:cur offset:0 atIndex:2];
            [enc setBuffer:dst offset:0 atIndex:3];
            [enc setBytes:&a length:sizeof a atIndex:4];
            dispatch2d(enc, ps->finish, ww, wh);
        } else if (dn && rw > 0 && rh > 0) {
            id<MTLBuffer> aux = [dev newBufferWithLength:(size_t)rw * rh * FG_AUX_FLOATS * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> cur = [dev newBufferWithLength:(size_t)rw * rh * FG_CV_FLOATS * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> nxt = [dev newBufferWithLength:(size_t)rw * rh * FG_CV_FLOATS * 4 options:MTLResourceStorageModePrivate];
            if (!aux || !cur || !nxt) { [enc endEncoding]; return "Metal buffer allocation failed"; }
            bindScene();
            [enc setBuffer:aux offset:0 atIndex:6];
            [enc setBuffer:cur offset:0 atIndex:7];
            [enc setBytes:&a length:sizeof a atIndex:8];
            [enc setBuffer:half offset:0 atIndex:9];
            dispatch2d(enc, ps->trace, rw, rh);
            for (int it = 0; it < FG_DENOISE_ITERS; ++it) {
                a.step = 1 << it;
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setBuffer:aux offset:0 atIndex:0];
                [enc setBuffer:cur offset:0 atIndex:1];
                [enc setBuffer:nxt offset:0 atIndex:2];
                [enc setBytes:&a length:sizeof a atIndex:3];
                dispatch2d(enc, ps->atrous, rw, rh);
                std::swap(cur, nxt);
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setBytes:&p length:sizeof p atIndex:0];
            [enc setBuffer:srcPyr offset:0 atIndex:1];
            [enc setBuffer:cur offset:0 atIndex:2];
            [enc setBuffer:dst offset:0 atIndex:3];
            [enc setBytes:&a length:sizeof a atIndex:4];
            dispatch2d(enc, ps->finish, ww, wh);
        } else {
            bindScene();
            [enc setBuffer:dst offset:0 atIndex:6];
            [enc setBytes:&a length:sizeof a atIndex:7];
            [enc setBuffer:half offset:0 atIndex:8];
            dispatch2d(enc, ps->shade, ww, wh);
        }
        [enc endEncoding];
        // The command buffer retains every buffer it uses, so the locals can go.
        [cb commit];
        if (f.wait) {
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                std::snprintf(err, sizeof err, "Metal: %s", cb.error ? cb.error.localizedDescription.UTF8String : "error");
                return err;
            }
        }
    }
    return nullptr;
}

void* metalCacheAlloc(void* queue, size_t bytes)
{
    @autoreleasepool {
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLDevice> dev = q ? q.device : MTLCreateSystemDefaultDevice();
        if (!dev || bytes == 0) return nullptr;
        id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        return b ? (__bridge_retained void*)b : nullptr;
    }
}

void metalCacheFree(void* buffer)
{
    if (buffer) CFRelease(buffer);
}

}  // namespace fg3d
