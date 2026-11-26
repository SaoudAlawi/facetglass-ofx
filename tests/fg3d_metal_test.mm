// FacetGlass 3D - Metal vs CPU test (the macOS twin of fg3d_gpu_test).
//   fg3d_metal_test <in.raw> <W> <H> [--bench]
// --bench: Metal only, full frame, best of 3 (GPU time per look).
// Renders the same looks on the CPU and with Metal (in 2x2 render windows, one
// look with a negative-stride source, one with a 2x refract source, all images at
// a non-zero buffer offset) and compares. The core uses its own exp/log/sin/cos and
// no fused multiply-add, so the results must be bit-identical.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "fg3d_args.h"
#include "fg3d_cpu.h"
#include "fg3d_metal.h"

using namespace fg3d;

static const char* kLooks[][8] = {
    { "shape=0", "facets=3", "prismAngle=6", "size=0.9", "depth=0.12", "center=0.42,0.5", "rotZ=8", "innerReflect=3" },
    { "shape=2", "facets=6", "prismAngle=7", "clearCentre=0.5", "size=1.6", "depth=0.15", nullptr },
    { "shape=10", "power=6", "size=0.7", "depth=0.2", "center=0.5,0.6", "dispersion=0.6", nullptr },
    { "shape=15", "size=0.7", "depth=0.15", "rotX=-35", "rotY=15", "ior=2.0", "focus=0.05", "light=0.2,0.8,0.3,0.1,6500,3" },
    { "shape=21", "facets=7", "size=1.2", "depth=0.15", "center=0.3,0.4", "ior=1.7", "seed=3", "roughness=0.2" },
    { "shape=15", "size=0.7", "depth=0.15", "rotX=-35", "rotY=15", "ior=2.0", "focus=0.05", "innerReflect=0" },
    { "shape=15", "size=0.7", "depth=0.15", "rotX=-35", "glassType=8", "tint=1,0.7,0.35", "tintDensity=1.5", "innerReflect=2" },
    { "lensMount=1", "shape=42", "facets=7", "prismAngle=1.5", "size=1.1", "fstop=2.8", "rotZ=10", nullptr },
    { "lensMount=1", "shape=2", "facets=6", "prismAngle=3", "clearCentre=0.4", "size=1.1", "denoise=0.5", nullptr },
    { "shape=23", "size=0.6", "depth=0.25", "rotX=25", "rotY=35", "focus=0", "fstop=8", "colorSpace=3" },
    { "shape=27", "detail=2", "size=0.8", "depth=0.2", "world=5", "wobble=0.7", "time=31", nullptr },
    { "shape=11", "power=4", "size=2.5", "depth=0.15", "rotZ=180", "focus=0.5", "glassFocus=1", "fstop=2" },
    { "shape=15", "size=0.7", "depth=0.15", "rotX=-35", "rotY=15", "ior=2.0", "focus=0.05", "denoise=0.5" },
    { "shape=21", "facets=7", "size=1.2", "depth=0.15", "fstop=2", "dispersion=0.7", "seed=3", "denoise=1" },
};

static const int kPad = 1024;  // floats before each image in its buffer (tests the base offset)

int main(int argc, char** argv)
{
    @autoreleasepool {
        if (argc < 4) { std::fprintf(stderr, "usage: fg3d_metal_test in.raw W H\n"); return 2; }
        const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
        const bool bench = argc > 4 && std::strcmp(argv[4], "--bench") == 0;
        std::vector<float> img((size_t)W * H * 4);
        FILE* f = std::fopen(argv[1], "rb");
        if (!f || std::fread(img.data(), 4, img.size(), f) != img.size()) { std::fprintf(stderr, "bad input\n"); return 2; }
        std::fclose(f);
        const int GW = 2 * W, GH = 2 * H;
        std::vector<float> gimg((size_t)GW * GH * 4);
        for (int y = 0; y < GH; ++y)
            for (int x = 0; x < GW; ++x)
                for (int c = 0; c < 4; ++c)
                    gimg[((size_t)y * GW + x) * 4 + c] = img[((size_t)(y / 2) * W + x / 2) * 4 + (c < 3 ? 2 - c : 3)];

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 3; }
        std::printf("GPU: %s\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        const size_t n = img.size();
        id<MTLBuffer> bSrc = [dev newBufferWithLength:(kPad + n) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bDst = [dev newBufferWithLength:(kPad + n) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bGls = [dev newBufferWithLength:(kPad + gimg.size()) * 4 options:MTLResourceStorageModeShared];
        std::memcpy((float*)bGls.contents + kPad, gimg.data(), gimg.size() * 4);

        const int nLooks = (int)(sizeof(kLooks) / sizeof(kLooks[0]));
        bool allOk = true;
        for (int li = 0; li < nLooks + 2; ++li) {
            const int look = li < nLooks ? li : (li == nLooks ? 4 : 2);
            const bool negStride = li == nLooks;
            const bool refract = li == nLooks + 1;
            UIParams ui;
            double time = 0;
            for (int k = 0; k < 8 && kLooks[look][k]; ++k) applyArg(ui, time, kLooks[look][k]);
            ui.shape.worldScale = (float)objectScale(ui);
            ui.shape.depth = (float)ui.depth;
            SceneBuffers scene;
            buildScene(makeShape(ui.shape), (float)ui.bevelAngle, scene);
            FG3DParams p;
            makeParams(ui, 0, 0, (float)W, (float)H, time, scene, p);

            if (bench) {
                if (negStride || refract) continue;
                std::memcpy((float*)bSrc.contents + kPad, img.data(), n * 4);
                MetalFrame fr;
                fr.queue = (__bridge void*)queue;
                fr.src.buffer = (__bridge void*)bSrc; fr.src.base = kPad;
                fr.src.w = W; fr.src.h = H; fr.src.stride = W * 4;
                fr.frameW = W; fr.frameH = H;
                fr.dst = (__bridge void*)bDst; fr.dstBase = kPad; fr.dstStride = W * 4;
                fr.winX2 = W; fr.winY2 = H; fr.wait = true;
                double best = 1e30;
                for (int rep = 0; rep < 3; ++rep) {
                    auto g0 = std::chrono::steady_clock::now();
                    if (const char* err = renderMetal(fr, p, scene)) { std::fprintf(stderr, "renderMetal: %s\n", err); return 3; }
                    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count());
                }
                std::printf("bench look %d: %.1f ms\n", look, best);
                continue;
            }
            auto c0 = std::chrono::steady_clock::now();
            std::vector<float> pyr, gpyr, cpu(n);
            int levels = 0, glevels = 0;
            buildPyramidCPU(img.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyr, levels);
            FG3DParams pc = p;
            pc.srcW = W; pc.srcH = H; pc.srcLevels = levels;
            pc.glsW = W; pc.glsH = H; pc.glsLevels = levels; pc.useSource = 0;
            if (refract) {
                buildPyramidCPU(gimg.data(), 0, 0, GW, GH, GW * 4, 0, 0, GW, GH, ui.colorSpace, gpyr, glevels);
                pc.glsW = GW; pc.glsH = GH; pc.glsLevels = glevels; pc.useSource = 1;
            }
            renderCPU(pc, pyr.data(), refract ? gpyr.data() : pyr.data(), scene, 0, W, 0, H,
                      [](int b, int e, auto fn) {
                          const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
                          std::vector<std::thread> pool;
                          for (unsigned t = 0; t < nt; ++t)
                              pool.emplace_back([&, t] { for (int i = b + (int)t; i < e; i += (int)nt) fn(i); });
                          for (auto& th : pool) th.join();
                      },
                      [&](int yy) { return &cpu[(size_t)yy * W * 4]; });
            const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

            float* s = (float*)bSrc.contents + kPad;
            if (negStride)
                for (int y = 0; y < H; ++y) std::memcpy(s + (size_t)(H - 1 - y) * W * 4, &img[(size_t)y * W * 4], (size_t)W * 16);
            else
                std::memcpy(s, img.data(), n * 4);
            std::memset(bDst.contents, 0, (kPad + n) * 4);

            MetalFrame fr;
            fr.queue = (__bridge void*)queue;
            fr.src.buffer = (__bridge void*)bSrc;
            fr.src.base = negStride ? kPad + (long long)(H - 1) * W * 4 : kPad;
            fr.src.w = W; fr.src.h = H; fr.src.stride = negStride ? -W * 4 : W * 4;
            if (refract) {
                fr.gls.buffer = (__bridge void*)bGls;
                fr.gls.base = kPad; fr.gls.w = GW; fr.gls.h = GH; fr.gls.stride = GW * 4;
                fr.glsFrameW = GW; fr.glsFrameH = GH;
            }
            fr.frameW = W; fr.frameH = H;
            fr.dst = (__bridge void*)bDst;
            fr.dstBase = kPad; fr.dstStride = W * 4;
            auto g0 = std::chrono::steady_clock::now();
            for (int ty = 0; ty < 2; ++ty)
                for (int tx = 0; tx < 2; ++tx) {
                    fr.winX1 = W * tx / 2; fr.winX2 = W * (tx + 1) / 2;
                    fr.winY1 = H * ty / 2; fr.winY2 = H * (ty + 1) / 2;
                    fr.wait = tx == 1 && ty == 1;  // same queue: the last wait covers all four
                    if (const char* err = renderMetal(fr, p, scene)) { std::fprintf(stderr, "renderMetal: %s\n", err); return 3; }
                }
            const double gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
            const float* gpu = (const float*)bDst.contents + kPad;

            double maxd = 0, sum = 0;
            size_t bad = 0, diff = 0;
            for (size_t i = 0; i < n; i += 4) {
                double pd = 0;
                for (int c = 0; c < 4; ++c) pd = std::max(pd, (double)std::fabs(cpu[i + c] - gpu[i + c]));
                maxd = std::max(maxd, pd);
                sum += pd;
                bad += pd > 1e-3;
                diff += pd > 0;
            }
            const double mean = sum / (n / 4), badPct = 100.0 * bad / (n / 4);
            const bool ok = maxd <= 1e-5;
            allOk = allOk && ok;
            std::printf("%s look %d%s%s: max %.2e mean %.2e >1e-3 %.3f%% differing px %zu  cpu %.0f ms metal %.1f ms\n",
                        ok ? "ok  " : "FAIL", look, negStride ? " (neg stride)" : "", refract ? " (2x refract)" : "", maxd,
                        mean, badPct, diff, cpuMs, gpuMs);
        }
        // Playback cache: record on frame A, replay on a different frame B; CPU and
        // Metal replays must match exactly (record frames output the replay too).
        if (!bench) {
            std::vector<float> img2(n);
            for (size_t i = 0; i < n; i += 4) {
                img2[i] = img[i + 1] * 1.2f; img2[i + 1] = img[i + 2]; img2[i + 2] = img[i] * 0.9f; img2[i + 3] = img[i + 3];
            }
            const int cacheLooks[] = { 0, 3, 6, 8, 12 };
            for (int look : cacheLooks) {
                if (look >= nLooks) continue;
                UIParams ui;
                double time = 0;
                for (int k = 0; k < 8 && kLooks[look][k]; ++k) applyArg(ui, time, kLooks[look][k]);
                ui.shape.worldScale = (float)objectScale(ui);
                ui.shape.depth = (float)ui.depth;
                SceneBuffers scene;
                buildScene(makeShape(ui.shape), (float)ui.bevelAngle, scene);
                FG3DParams p;
                makeParams(ui, 0, 0, (float)W, (float)H, time, scene, p);
                const size_t cf = cacheFloats(p, cacheRegion(p, 0, W, 0, H));
                std::vector<float> pyrA, pyrB, cpuRec(n), cpuRep(n), cache(cf);
                int lv = 0;
                buildPyramidCPU(img.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyrA, lv);
                buildPyramidCPU(img2.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyrB, lv);
                FG3DParams pc = p;
                pc.srcW = pc.glsW = W; pc.srcH = pc.glsH = H; pc.srcLevels = pc.glsLevels = lv; pc.useSource = 0;
                auto par = [](int b, int e, auto fn) {
                    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
                    std::vector<std::thread> pool;
                    for (unsigned t = 0; t < nt; ++t)
                        pool.emplace_back([&, t] { for (int i = b + (int)t; i < e; i += (int)nt) fn(i); });
                    for (auto& th : pool) th.join();
                };
                renderCPU(pc, pyrA.data(), pyrA.data(), scene, 0, W, 0, H, par, [&](int y) { return &cpuRec[(size_t)y * W * 4]; },
                          kCacheRecord, cache.data());
                auto c0 = std::chrono::steady_clock::now();
                renderCPU(pc, pyrB.data(), pyrB.data(), scene, 0, W, 0, H, par, [&](int y) { return &cpuRep[(size_t)y * W * 4]; },
                          kCacheReplay, cache.data());
                const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

                void* mcache = metalCacheAlloc((__bridge void*)queue, cf * 4);
                MetalFrame fr;
                fr.queue = (__bridge void*)queue;
                fr.src.buffer = (__bridge void*)bSrc; fr.src.base = kPad;
                fr.src.w = W; fr.src.h = H; fr.src.stride = W * 4;
                fr.frameW = W; fr.frameH = H;
                fr.dst = (__bridge void*)bDst; fr.dstBase = kPad; fr.dstStride = W * 4;
                fr.winX2 = W; fr.winY2 = H; fr.wait = true; fr.cache = mcache;
                std::memcpy((float*)bSrc.contents + kPad, img.data(), n * 4);
                fr.cacheMode = kCacheRecord;
                if (const char* err = renderMetal(fr, p, scene)) { std::fprintf(stderr, "renderMetal: %s\n", err); return 3; }
                std::vector<float> mRec((const float*)bDst.contents + kPad, (const float*)bDst.contents + kPad + n);
                std::memcpy((float*)bSrc.contents + kPad, img2.data(), n * 4);
                fr.cacheMode = kCacheReplay;
                auto g0 = std::chrono::steady_clock::now();
                if (const char* err = renderMetal(fr, p, scene)) { std::fprintf(stderr, "renderMetal: %s\n", err); return 3; }
                const double gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
                const float* mRep = (const float*)bDst.contents + kPad;
                metalCacheFree(mcache);
                double dRec = 0, dRep = 0;
                for (size_t i = 0; i < n; ++i) {
                    dRec = std::max(dRec, (double)std::fabs(cpuRec[i] - mRec[i]));
                    dRep = std::max(dRep, (double)std::fabs(cpuRep[i] - mRep[i]));
                }
                const bool ok = dRec <= 1e-5 && dRep <= 1e-5;
                allOk = allOk && ok;
                std::printf("%s cache look %d: record max %.2e replay max %.2e  replay cpu %.1f ms metal %.1f ms\n",
                            ok ? "ok  " : "FAIL", look, dRec, dRep, cpuMs, gpuMs);
            }
        }
        // Soft areas: CPU and Metal must match exactly (plain and denoised).
        if (!bench) {
            const int softLooks[] = { 2, 6, 8, 9 };
            for (int look : softLooks) {
                UIParams ui;
                double time = 0;
                for (int k = 0; k < 8 && kLooks[look][k]; ++k) applyArg(ui, time, kLooks[look][k]);
                ui.shape.worldScale = (float)objectScale(ui);
                ui.shape.depth = (float)ui.depth;
                SceneBuffers scene;
                buildScene(makeShape(ui.shape), (float)ui.bevelAngle, scene);
                FG3DParams p;
                makeParams(ui, 0, 0, (float)W, (float)H, time, scene, p);
                std::vector<float> pyr, cpu(n);
                int lv = 0;
                buildPyramidCPU(img.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyr, lv);
                FG3DParams pc = p;
                pc.srcW = pc.glsW = W; pc.srcH = pc.glsH = H; pc.srcLevels = pc.glsLevels = lv; pc.useSource = 0;
                auto par = [](int b, int e, auto fn) {
                    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
                    std::vector<std::thread> pool;
                    for (unsigned t = 0; t < nt; ++t)
                        pool.emplace_back([&, t] { for (int i = b + (int)t; i < e; i += (int)nt) fn(i); });
                    for (auto& th : pool) th.join();
                };
                renderCPU(pc, pyr.data(), pyr.data(), scene, 0, W, 0, H, par, [&](int y) { return &cpu[(size_t)y * W * 4]; },
                          kCacheNone, nullptr, true);
                MetalFrame fr;
                fr.queue = (__bridge void*)queue;
                fr.src.buffer = (__bridge void*)bSrc; fr.src.base = kPad;
                fr.src.w = W; fr.src.h = H; fr.src.stride = W * 4;
                fr.frameW = W; fr.frameH = H;
                fr.dst = (__bridge void*)bDst; fr.dstBase = kPad; fr.dstStride = W * 4;
                fr.wait = true; fr.soft = true;
                std::memcpy((float*)bSrc.contents + kPad, img.data(), n * 4);
                // In 2x2 windows, like the main test.
                for (int ty = 0; ty < 2; ++ty)
                    for (int tx = 0; tx < 2; ++tx) {
                        fr.winX1 = W * tx / 2; fr.winX2 = W * (tx + 1) / 2;
                        fr.winY1 = H * ty / 2; fr.winY2 = H * (ty + 1) / 2;
                        if (const char* err = renderMetal(fr, p, scene)) { std::fprintf(stderr, "renderMetal: %s\n", err); return 3; }
                    }
                const float* g = (const float*)bDst.contents + kPad;
                double d = 0;
                for (size_t i = 0; i < n; ++i) d = std::max(d, (double)std::fabs(cpu[i] - g[i]));
                // Tiles trace their own half-resolution pass, so compare against a CPU render in the same windows.
                std::vector<float> cpuT(n);
                for (int ty = 0; ty < 2; ++ty)
                    for (int tx = 0; tx < 2; ++tx)
                        renderCPU(pc, pyr.data(), pyr.data(), scene, W * tx / 2, W * (tx + 1) / 2, H * ty / 2, H * (ty + 1) / 2,
                                  par, [&](int y) { return &cpuT[(size_t)y * W * 4 + (size_t)(W * tx / 2) * 4]; },
                                  kCacheNone, nullptr, true);
                double dT = 0, dFull = d;
                for (size_t i = 0; i < n; ++i) dT = std::max(dT, (double)std::fabs(cpuT[i] - g[i]));
                const bool ok = dT <= 1e-5;
                allOk = allOk && ok;
                std::printf("%s soft look %d: metal vs cpu (same windows) max %.2e  (vs cpu full frame %.2e)\n",
                            ok ? "ok  " : "FAIL", look, dT, dFull);
            }
        }
        std::printf("%s\n", allOk ? "PASS" : "FAIL");
        return allOk ? 0 : 1;
    }
}
