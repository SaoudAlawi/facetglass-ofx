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

#include "fg3d_cpu.h"

using namespace fg3d;

static std::vector<double> nums(const char* s)
{
    std::vector<double> v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtod(s, &e));
        if (e == s) break;
        s = (*e == ',') ? e + 1 : e;
    }
    return v;
}

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
    for (int a = 5; a < argc; ++a) {
        const char* eq = std::strchr(argv[a], '=');
        if (!eq) continue;
        std::string k(argv[a], eq - argv[a]);
        std::vector<double> v = nums(eq + 1);
        if (v.empty()) continue;
        const double x = v[0];
        if (k == "shape") ui.shape.shape = (int)x;
        else if (k == "facets") ui.shape.facets = (int)x;
        else if (k == "bothSides") ui.shape.bothSides = x != 0;
        else if (k == "prismAngle") ui.shape.prismAngleDeg = (float)x;
        else if (k == "clearCentre") ui.shape.clearCentre = (float)x;
        else if (k == "power") ui.shape.power = (float)x;
        else if (k == "thickness") ui.shape.thickness = (float)x;
        else if (k == "detail") ui.shape.detail = (int)x;
        else if (k == "center" && v.size() >= 2) { ui.centerX = v[0]; ui.centerY = v[1]; }
        else if (k == "depth") ui.depth = x;
        else if (k == "rotX") ui.rotX = x;
        else if (k == "rotY") ui.rotY = x;
        else if (k == "rotZ") ui.rotZ = x;
        else if (k == "size") ui.size = x;
        else if (k == "ior") ui.ior = x;
        else if (k == "dispersion") ui.dispersion = x;
        else if (k == "fstop") ui.fstop = x;
        else if (k == "focus") ui.focus = x;
        else if (k == "glassFocus") ui.glassFocus = x;
        else if (k == "reflection") ui.reflection = x;
        else if (k == "roughness") ui.roughness = x;
        else if (k == "bevelWidth") ui.bevelWidth = x;
        else if (k == "bevelAngle") ui.bevelAngle = x;
        else if (k == "maxBounces") ui.maxBounces = (int)x;
        else if (k == "minSamples") ui.minSamples = (int)x;
        else if (k == "maxSamples") ui.maxSamples = (int)x;
        else if (k == "noise") ui.noiseThreshold = x;
        else if (k == "world") ui.worldMode = (int)x;
        else if (k == "worldBlur") ui.worldBlur = x;
        else if (k == "feather") ui.feather = x;
        else if (k == "colorSpace") ui.colorSpace = (int)x;
        else if (k == "boost") ui.boost = x;
        else if (k == "threshold") ui.threshold = x;
        else if (k == "mix") ui.mix = x;
        else if (k == "seed") { ui.seed = (int)x; ui.shape.seed = (int)x; }
        else if (k == "wobble") ui.wobble = x;
        else if (k == "wobbleSpeed") ui.wobbleSpeed = x;
        else if (k == "time") time = x;
        else if (k == "light" && v.size() >= 6 && ui.numLights < FG_MAX_LIGHTS) {
            UIParams::Light& L = ui.lights[ui.numLights++];
            L.x = v[0]; L.y = v[1]; L.depth = v[2]; L.size = v[3]; L.kelvin = v[4]; L.intensity = v[5];
        }
        else std::fprintf(stderr, "unknown key %s\n", k.c_str());
    }

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

    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (int y = (int)t; y < H; y += (int)nt)   // interleaved rows balance the load
                shadeCPU(p, pyr.data(), pyr.data(), scene, 0, W, y, y + 1,
                         [&](int yy) { return &out[(size_t)yy * W * 4]; });
        });
    for (auto& th : pool) th.join();
    auto t2 = std::chrono::steady_clock::now();

    f = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("tris %d nodes %d scene %.1f ms render %.1f ms\n", scene.numTris(), scene.numNodes(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    return 0;
}
