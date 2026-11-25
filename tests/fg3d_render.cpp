// FacetGlass 3D - command-line CPU renderer for previews and reference renders.
//   fg3d_render <in.raw> <out.raw> <W> <H> [key=value ...]
// Raw images are float32 RGBA, row 0 = bottom. Keys are the UIParams field names
// (shape, facets, bothSides, prismAngle, clearCentre, power, thickness, detail,
// center=x,y, depth, rotX, rotY, rotZ, size, ior, dispersion, fstop, focus,
// glassFocus, reflection, roughness, bevelWidth, bevelAngle, maxBounces,
// minSamples, maxSamples, noise, world, worldBlur, feather, colorSpace, boost,
// threshold, mix, seed, wobble, wobbleSpeed, time, light=x,y,depth,size,kelvin,intensity).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "fg3d_args.h"
#include "fg3d_cpu.h"

using namespace fg3d;

// Calls fn(i) for i in [b, e) on all cores (interleaved rows balance the load).
struct ParallelRows {
    template <class Fn>
    void operator()(int b, int e, Fn fn) const
    {
        const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (int i = b + (int)t; i < e; i += (int)nt) fn(i);
            });
        for (auto& th : pool) th.join();
    }
};

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: fg3d_render in.raw out.raw W H [key=value ...]\n");
        return 2;
    }
    const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
    std::vector<float> img((size_t)W * H * 4), out((size_t)W * H * 4);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(img.data(), 4, img.size(), f) != img.size()) { std::fprintf(stderr, "bad input\n"); return 2; }
    std::fclose(f);

    UIParams ui;
    double time = 0;
    for (int a = 5; a < argc; ++a)
        if (!applyArg(ui, time, argv[a])) std::fprintf(stderr, "unknown argument %s\n", argv[a]);

    auto t0 = std::chrono::steady_clock::now();
    SceneBuffers scene;
    ui.shape.worldScale = (float)objectScale(ui);
    ui.shape.depth = (float)ui.depth;
    buildScene(makeShape(ui.shape), (float)ui.bevelAngle, scene);
    auto t1 = std::chrono::steady_clock::now();

    std::vector<float> pyr;
    int levels = 0;
    buildPyramidCPU(img.data(), 0, 0, W, H, W * 4, 0, 0, W, H, ui.colorSpace, pyr, levels);

    FG3DParams p;
    makeParams(ui, 0, 0, (float)W, (float)H, time, scene, p);
    p.srcW = p.glsW = W; p.srcH = p.glsH = H; p.srcLevels = p.glsLevels = levels; p.useSource = 0;

    renderCPU(p, pyr.data(), pyr.data(), scene, 0, W, 0, H, ParallelRows{},
              [&](int yy) { return &out[(size_t)yy * W * 4]; });
    auto t2 = std::chrono::steady_clock::now();

    f = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("tris %d nodes %d scene %.1f ms render %.1f ms\n", scene.numTris(), scene.numNodes(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    return 0;
}
