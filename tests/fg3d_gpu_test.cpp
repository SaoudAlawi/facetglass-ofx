// FacetGlass 3D - CUDA vs CPU test.
//   fg3d_gpu_test <in.raw> <W> <H>
// Renders a set of looks on the CPU and with CUDA (in 2x2 render windows, one
// look with a negative-stride source, one with a 2x refract source) and compares.
// The core uses its own exp/log/sin/cos and no fused multiply-add, so the CPU
// (MSVC /fp:precise) and CUDA (--fmad=false) results must be bit-identical: any
// difference means a backend diverged. (Library maths differs in the last bits,
// which in a path tracer flips random reflect/refract choices.)
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "fg3d_args.h"
#include "fg3d_cpu.h"
#include "fg3d_cuda.h"

using namespace fg3d;

static const char* kLooks[][8] = {
    { "shape=0", "facets=3", "prismAngle=6", "size=0.9", "depth=0.12", "center=0.42,0.5", "rotZ=8", nullptr },
    { "shape=2", "facets=6", "prismAngle=7", "clearCentre=0.5", "size=1.6", "depth=0.15", nullptr },
    { "shape=10", "power=6", "size=0.7", "depth=0.2", "center=0.5,0.6", "dispersion=0.6", nullptr },
    { "shape=15", "size=0.7", "depth=0.15", "rotX=-35", "rotY=15", "ior=2.0", "focus=0.05", "light=0.2,0.8,0.3,0.1,6500,3" },
    { "shape=21", "facets=7", "size=1.2", "depth=0.15", "center=0.3,0.4", "ior=1.7", "seed=3", "roughness=0.2" },
    { "shape=23", "size=0.6", "depth=0.25", "rotX=25", "rotY=35", "focus=0", "fstop=8", "colorSpace=3" },
    { "shape=27", "detail=2", "size=0.8", "depth=0.2", "world=5", "wobble=0.7", "time=31", nullptr },
    { "shape=11", "power=4", "size=2.5", "depth=0.15", "rotZ=180", "focus=0.5", "glassFocus=1", "fstop=2" },
};

static void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(3); }
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: fg3d_gpu_test in.raw W H\n"); return 2; }
    const int W = std::atoi(argv[2]), H = std::atoi(argv[3]);
    std::vector<float> img((size_t)W * H * 4);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(img.data(), 4, img.size(), f) != img.size()) { std::fprintf(stderr, "bad input\n"); return 2; }
    std::fclose(f);
    // 2x refract source: nearest upsample with a colour twist so it is distinguishable.
    const int GW = 2 * W, GH = 2 * H;
    std::vector<float> gimg((size_t)GW * GH * 4);
    for (int y = 0; y < GH; ++y)
        for (int x = 0; x < GW; ++x)
            for (int c = 0; c < 4; ++c)
                gimg[((size_t)y * GW + x) * 4 + c] = img[((size_t)(y / 2) * W + x / 2) * 4 + (c < 3 ? 2 - c : 3)];

    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    float *dSrc = nullptr, *dDst = nullptr, *dGls = nullptr;
    check(cudaMalloc(&dSrc, img.size() * 4), "malloc src");
    check(cudaMalloc(&dDst, img.size() * 4), "malloc dst");
    check(cudaMalloc(&dGls, gimg.size() * 4), "malloc gls");
    check(cudaMemcpy(dGls, gimg.data(), gimg.size() * 4, cudaMemcpyHostToDevice), "upload gls");

    const int nLooks = (int)(sizeof(kLooks) / sizeof(kLooks[0]));
    bool allOk = true;
    for (int li = 0; li < nLooks + 2; ++li) {
        const int look = li < nLooks ? li : (li == nLooks ? 4 : 2);
        const bool negStride = li == nLooks;      // extra run: look 4 with a negative-stride source
        const bool refract = li == nLooks + 1;    // extra run: look 2 with a 2x refract source
        UIParams ui;
        double time = 0;
        for (int k = 0; k < 8 && kLooks[look][k]; ++k) applyArg(ui, time, kLooks[look][k]);
        ui.shape.worldScale = (float)objectScale(ui);
        ui.shape.depth = (float)ui.depth;
        SceneBuffers scene;
        buildScene(makeShape(ui.shape), (float)ui.bevelAngle, scene);
        FG3DParams p;
        makeParams(ui, 0, 0, (float)W, (float)H, time, scene, p);

        // CPU reference.
        auto c0 = std::chrono::steady_clock::now();
        std::vector<float> pyr, gpyr, cpu((size_t)W * H * 4), gpu((size_t)W * H * 4);
        int levels = 0, glevels = 0;
        buildPyramidCPU(img.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyr, levels);
        FG3DParams pc = p;
        pc.srcW = W; pc.srcH = H; pc.srcLevels = levels;
        pc.glsW = W; pc.glsH = H; pc.glsLevels = levels; pc.useSource = 0;
        if (refract) {
            buildPyramidCPU(gimg.data(), 0, 0, GW, GH, GW * 4, 0, 0, GW, GH, ui.colorSpace, gpyr, glevels);
            pc.glsW = GW; pc.glsH = GH; pc.glsLevels = glevels; pc.useSource = 1;
        }
        const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (int y = (int)t; y < H; y += (int)nt)
                    shadeCPU(pc, pyr.data(), refract ? gpyr.data() : pyr.data(), scene, 0, W, y, y + 1,
                             [&](int yy) { return &cpu[(size_t)yy * W * 4]; });
            });
        for (auto& th : pool) th.join();
        const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

        // GPU, in 2x2 render windows.
        if (negStride) {
            for (int y = 0; y < H; ++y)
                check(cudaMemcpy(dSrc + (size_t)(H - 1 - y) * W * 4, &img[(size_t)y * W * 4], (size_t)W * 16,
                                 cudaMemcpyHostToDevice), "upload src rows");
        } else {
            check(cudaMemcpy(dSrc, img.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload src");
        }
        check(cudaMemset(dDst, 0, img.size() * 4), "clear dst");
        CudaFrame fr;
        fr.src.data = negStride ? dSrc + (size_t)(H - 1) * W * 4 : dSrc;
        fr.src.x1 = 0; fr.src.y1 = 0; fr.src.w = W; fr.src.h = H; fr.src.stride = negStride ? -W * 4 : W * 4;
        if (refract) {
            fr.gls.data = dGls; fr.gls.w = GW; fr.gls.h = GH; fr.gls.stride = GW * 4;
            fr.glsFrameW = GW; fr.glsFrameH = GH;
        }
        fr.frameW = W; fr.frameH = H;
        fr.dst = dDst; fr.dstStride = W * 4;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int ty = 0; ty < 2; ++ty)
            for (int tx = 0; tx < 2; ++tx) {
                fr.winX1 = W * tx / 2; fr.winX2 = W * (tx + 1) / 2;
                fr.winY1 = H * ty / 2; fr.winY2 = H * (ty + 1) / 2;
                const char* err = renderCUDA(fr, p, scene);
                if (err) { std::fprintf(stderr, "renderCUDA: %s\n", err); return 3; }
            }
        cudaEventRecord(e1);
        check(cudaEventSynchronize(e1), "sync");
        float gpuMs = 0;
        cudaEventElapsedTime(&gpuMs, e0, e1);
        check(cudaMemcpy(gpu.data(), dDst, gpu.size() * 4, cudaMemcpyDeviceToHost), "download");

        double maxd = 0, sum = 0;
        size_t bad = 0;
        for (size_t i = 0; i < cpu.size(); i += 4) {
            double pd = 0;
            for (int c = 0; c < 4; ++c) pd = std::max(pd, (double)std::fabs(cpu[i + c] - gpu[i + c]));
            maxd = std::max(maxd, pd);
            sum += pd;
            bad += pd > 1e-3;
        }
        const double mean = sum / (cpu.size() / 4), badPct = 100.0 * bad / (cpu.size() / 4);
        const bool ok = maxd <= 1e-5;  // the core is bit-identical across backends
        allOk = allOk && ok;
        std::printf("%s look %d%s%s: max %.2e mean %.2e >1e-3 %.3f%%  cpu %.0f ms gpu %.1f ms\n", ok ? "ok  " : "FAIL",
                    look, negStride ? " (neg stride)" : "", refract ? " (2x refract)" : "", maxd, mean, badPct, cpuMs,
                    gpuMs);
    }
    cudaFree(dSrc); cudaFree(dDst); cudaFree(dGls);
    std::printf("%s\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
