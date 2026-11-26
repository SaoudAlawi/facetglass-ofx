// FacetGlass 3D - host side scene building. See fg3d_scene.h.
// SPDX-License-Identifier: MIT
#include "fg3d_scene.h"
#include "fg3d_geom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <unordered_map>

namespace fg3d {

using namespace geom;

namespace {
V3 vert(const Mesh& m, int i) { return { m.v[3 * i], m.v[3 * i + 1], m.v[3 * i + 2] }; }
}  // namespace

// ---------------------------------------------------------------- scene build

namespace {

struct BuildTri {
    V3 v[3];
    V3 n[3];      // shading normals
    V3 fn;        // face normal
    V3 ne[3];     // bevel normals (zero = none)
    double h[3];  // heights
    V3 c;         // centroid
    V3 lo, hi;
};

void bvhBuild(std::vector<BuildTri>& t, std::vector<int>& idx, int begin, int end, std::vector<float>& nf,
              std::vector<int>& ni, int node)
{
    V3 lo = { 1e30, 1e30, 1e30 }, hi = { -1e30, -1e30, -1e30 };
    V3 clo = lo, chi = hi;
    for (int i = begin; i < end; ++i) {
        const BuildTri& b = t[idx[i]];
        lo = { std::min(lo.x, b.lo.x), std::min(lo.y, b.lo.y), std::min(lo.z, b.lo.z) };
        hi = { std::max(hi.x, b.hi.x), std::max(hi.y, b.hi.y), std::max(hi.z, b.hi.z) };
        clo = { std::min(clo.x, b.c.x), std::min(clo.y, b.c.y), std::min(clo.z, b.c.z) };
        chi = { std::max(chi.x, b.c.x), std::max(chi.y, b.c.y), std::max(chi.z, b.c.z) };
    }
    const double pad = 1e-5;
    float* f = &nf[(size_t)node * FG_NODE_FLOATS];
    f[0] = (float)(lo.x - pad); f[1] = (float)(lo.y - pad); f[2] = (float)(lo.z - pad);
    f[3] = (float)(hi.x + pad); f[4] = (float)(hi.y + pad); f[5] = (float)(hi.z + pad);
    const int count = end - begin;
    V3 ext = chi - clo;
    int axis = ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
    double extent = axis == 0 ? ext.x : axis == 1 ? ext.y : ext.z;
    if (count <= 4 || extent < 1e-9) {
        ni[(size_t)node * 2] = begin;
        ni[(size_t)node * 2 + 1] = count;
        return;
    }
    const int mid = begin + count / 2;
    auto key = [&](int i) { const V3& c = t[i].c; return axis == 0 ? c.x : axis == 1 ? c.y : c.z; };
    std::nth_element(idx.begin() + begin, idx.begin() + mid, idx.begin() + end,
                     [&](int a, int b) { return key(a) < key(b); });
    const int left = (int)(ni.size() / 2);
    ni.resize(ni.size() + 4, 0);
    nf.resize(nf.size() + 2 * FG_NODE_FLOATS, 0.0f);
    ni[(size_t)node * 2] = left;
    ni[(size_t)node * 2 + 1] = 0;
    bvhBuild(t, idx, begin, mid, nf, ni, left);
    bvhBuild(t, idx, mid, end, nf, ni, left + 1);
}

}  // namespace

void buildScene(const Mesh& mesh, float bevelAngleDeg, SceneBuffers& out)
{
    const size_t nt = mesh.f.size() / 3;
    out.tris.clear(); out.nodesF.clear(); out.nodesI.clear();
    if (nt == 0) return;

    // Weld vertices by position.
    std::vector<int> weld(mesh.v.size() / 3);
    {
        std::map<std::tuple<long long, long long, long long>, int> seen;
        for (size_t i = 0; i < weld.size(); ++i) {
            auto q = [&](float x) { return (long long)std::llround(x * 1e5); };
            auto k = std::make_tuple(q(mesh.v[3 * i]), q(mesh.v[3 * i + 1]), q(mesh.v[3 * i + 2]));
            auto it = seen.find(k);
            weld[i] = it == seen.end() ? (seen[k] = (int)i) : it->second;
        }
    }

    std::vector<BuildTri> T(nt);
    std::vector<V3> fnArea(nt);
    for (size_t t = 0; t < nt; ++t) {
        for (int k = 0; k < 3; ++k) T[t].v[k] = vert(mesh, mesh.f[3 * t + k]);
        V3 c = cross(T[t].v[1] - T[t].v[0], T[t].v[2] - T[t].v[0]);
        fnArea[t] = c;
        T[t].fn = norm(c);
    }

    // Smooth normals: per (welded vertex, group), area-weighted over the faces of
    // that group around the vertex whose normal is within the smoothing angle.
    const double cosSmooth = std::cos(std::min(180.0, (double)mesh.smoothAngleDeg) * kPi / 180.0) - 1e-9;
    std::map<std::pair<int, int>, std::vector<int>> around;
    for (size_t t = 0; t < nt; ++t) {
        if (mesh.group[t] == 0) continue;
        for (int k = 0; k < 3; ++k) around[std::make_pair(weld[mesh.f[3 * t + k]], mesh.group[t])].push_back((int)t);
    }
    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) {
            if (mesh.group[t] == 0) { T[t].n[k] = T[t].fn; continue; }
            V3 sum = { 0, 0, 0 };
            for (int o : around[std::make_pair(weld[mesh.f[3 * t + k]], mesh.group[t])])
                if (dot(T[o].fn, T[t].fn) >= cosSmooth) sum = sum + fnArea[o];
            T[t].n[k] = norm(sum);
        }

    // Edge adjacency for bevels.
    std::map<std::pair<int, int>, std::vector<int>> edges;  // welded edge -> triangles
    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) {
            int a = weld[mesh.f[3 * t + (k + 1) % 3]], b = weld[mesh.f[3 * t + (k + 2) % 3]];
            edges[std::make_pair(std::min(a, b), std::max(a, b))].push_back((int)t);
        }
    const double cosLimit = std::cos(bevelAngleDeg * kPi / 180.0);
    for (size_t t = 0; t < nt; ++t) {
        double area2 = len(fnArea[t]);
        for (int k = 0; k < 3; ++k) {
            V3 a = T[t].v[(k + 1) % 3], b = T[t].v[(k + 2) % 3];
            T[t].h[k] = area2 / std::max(len(b - a), 1e-12);
            T[t].ne[k] = { 0, 0, 0 };
            int wa = weld[mesh.f[3 * t + (k + 1) % 3]], wb = weld[mesh.f[3 * t + (k + 2) % 3]];
            const auto& adj = edges[std::make_pair(std::min(wa, wb), std::max(wa, wb))];
            if (adj.size() != 2) continue;
            int o = adj[0] == (int)t ? adj[1] : adj[0];
            if (mesh.group[t] != 0 && mesh.group[t] == mesh.group[o] && dot(T[t].fn, T[o].fn) >= cosSmooth)
                continue;                                                         // smooth edge
            if (dot(T[t].fn, T[o].fn) > cosLimit) continue;                       // too flat to bevel
            T[t].ne[k] = norm(T[t].fn + T[o].fn);
        }
        V3 lo = T[t].v[0], hi = T[t].v[0];
        for (int k = 1; k < 3; ++k) {
            lo = { std::min(lo.x, T[t].v[k].x), std::min(lo.y, T[t].v[k].y), std::min(lo.z, T[t].v[k].z) };
            hi = { std::max(hi.x, T[t].v[k].x), std::max(hi.y, T[t].v[k].y), std::max(hi.z, T[t].v[k].z) };
        }
        T[t].lo = lo; T[t].hi = hi;
        T[t].c = (T[t].v[0] + T[t].v[1] + T[t].v[2]) * (1.0 / 3.0);
    }

    // BVH.
    std::vector<int> idx(nt);
    std::iota(idx.begin(), idx.end(), 0);
    out.nodesF.assign(FG_NODE_FLOATS, 0.0f);
    out.nodesI.assign(FG_NODE_INTS, 0);
    bvhBuild(T, idx, 0, (int)nt, out.nodesF, out.nodesI, 0);

    out.tris.assign(nt * FG_TRI_FLOATS, 0.0f);
    for (size_t i = 0; i < nt; ++i) {
        const BuildTri& b = T[idx[i]];
        float* r = &out.tris[i * FG_TRI_FLOATS];
        auto put = [&](int o, V3 a) { r[o] = (float)a.x; r[o + 1] = (float)a.y; r[o + 2] = (float)a.z; };
        put(0, b.v[0]); put(3, b.v[1] - b.v[0]); put(6, b.v[2] - b.v[0]);
        put(9, b.n[0]); put(12, b.n[1]); put(15, b.n[2]);
        put(18, b.fn);
        put(21, b.ne[0]); put(24, b.ne[1]); put(27, b.ne[2]);
        r[30] = (float)b.h[0]; r[31] = (float)b.h[1]; r[32] = (float)b.h[2];
    }
}

// ---------------------------------------------------------------- params

int mipLevels(int w, int h)
{
    int l = 1;
    while (l < 14 && std::min(w >> l, h >> l) >= 8) ++l;
    return l;
}

size_t pyramidFloats(int w, int h, int levels)
{
    size_t n = 0;
    for (int l = 0; l < levels; ++l) n += (size_t)std::max(1, w >> l) * std::max(1, h >> l) * 4;
    return n;
}

void kelvinToRGB(double kelvin, float rgb[3])
{
    // Fit of black-body colour (display-referred), then to linear, normalised.
    double t = std::min(std::max(kelvin, 1000.0), 40000.0) / 100.0;
    double r, g, b;
    if (t <= 66) { r = 255; g = 99.4708025861 * std::log(t) - 161.1195681661; }
    else { r = 329.698727446 * std::pow(t - 60, -0.1332047592); g = 288.1221695283 * std::pow(t - 60, -0.0755148492); }
    if (t >= 66) b = 255;
    else if (t <= 19) b = 0;
    else b = 138.5177312231 * std::log(t - 10) - 305.0447927307;
    double c[3] = { r, g, b }, mx = 0;
    for (double& v : c) {
        v = std::min(std::max(v, 0.0), 255.0) / 255.0;
        v = v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
        mx = std::max(mx, v);
    }
    for (int i = 0; i < 3; ++i) rgb[i] = (float)(c[i] / std::max(mx, 1e-6));
}

double objectScale(const UIParams& ui)
{
    return std::max(ui.size, 0.005) * 0.5 * std::min(std::max(ui.depth, 0.01), 0.95);
}

void makeParams(const UIParams& ui, float frameX, float frameY, float frameW, float frameH, double time,
                const SceneBuffers& scene, FG3DParams& p)
{
    std::memset(&p, 0, sizeof p);
    p.frameX = frameX; p.frameY = frameY; p.frameW = frameW; p.frameH = frameH;
    const double aspect = frameW / std::max(frameH, 1.0f);

    // Hand-held wobble from time in frames, 24 fps base (drift, roll and tilt).
    double ox = 0, oy = 0, oa = 0, otx = 0, oty = 0;
    if (ui.wobble > 0) {
        const double s = ui.seed * 1.7319, tt = time * ui.wobbleSpeed / 24.0, a = ui.wobble;
        ox = a * 0.03 * (std::sin(1.31 * tt + s) + 0.5 * std::sin(2.97 * tt + 2.1 * s));
        oy = a * 0.03 * (std::sin(1.07 * tt + 3.3 * s) + 0.5 * std::sin(2.41 * tt + 0.7 * s));
        oa = a * 6.0 * (std::sin(0.83 * tt + 5.1 * s) + 0.4 * std::sin(2.13 * tt + 1.3 * s));
        otx = a * 8.0 * (std::sin(0.71 * tt + 2.3 * s) + 0.4 * std::sin(1.93 * tt + 0.4 * s));
        oty = a * 8.0 * (std::sin(0.61 * tt + 4.2 * s) + 0.4 * std::sin(1.71 * tt + 2.9 * s));
    }

    const double zd = std::min(std::max(ui.depth, 0.01), 0.95);
    p.objPos[0] = (float)((ui.centerX + ox - 0.5) * aspect * zd);
    p.objPos[1] = (float)((ui.centerY + oy - 0.5) * zd);
    p.objPos[2] = (float)zd;
    p.objScale = (float)(std::max(ui.size, 0.005) * 0.5 * zd);

    const double rx = (ui.rotX + otx) * kPi / 180.0, ry = (ui.rotY + oty) * kPi / 180.0,
                 rz = (ui.rotZ + oa) * kPi / 180.0;
    const double cx = std::cos(rx), sx = std::sin(rx), cy = std::cos(ry), sy = std::sin(ry), cz = std::cos(rz),
                 sz = std::sin(rz);
    // R = Rz * Ry * Rx
    const double R[9] = { cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                          sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
                          -sy,     cy * sx,                cy * cx };
    for (int i = 0; i < 9; ++i) p.objRot[i] = (float)R[i];

    const double fstop = std::max(ui.fstop, 0.5);
    p.aperture = (float)(zd * 0.3 / fstop);
    auto focusInv = [&](double f) {
        f = std::min(std::max(f, 0.0), 1.0);
        return 1.0 + (1.0 / zd - 1.0) * (1.0 - f) * (1.0 - f);
    };
    p.focusInv = (float)focusInv(ui.focus);
    p.glassFocusInv = (float)focusInv(ui.glassFocus);

    p.ior = (float)std::max(ui.ior, 1.0);
    p.dispersion = (float)std::max(ui.dispersion, 0.0);
    if (ui.glassType > kGlassCustom && ui.glassType < kGlassTypeCount) {
        // Real glass: refractive index n_d (587.6 nm) and Abbe number V_d. The core's
        // index is ior + 0.05 * dispersion * (1/lambda^2 - 1/0.55^2), lambda in um, so
        // n_F - n_C = 0.05 * dispersion * (1/0.4861^2 - 1/0.6563^2) = (n_d - 1) / V_d.
        // Chromatic Aberration then scales the real dispersion (0.4 = real).
        static const double kNd[kGlassTypeCount] = { 0, 1.5168, 1.6200, 1.7847, 1.4585, 1.4917, 1.7682, 2.1500, 2.4175, 1.3330 };
        static const double kVd[kGlassTypeCount] = { 1, 64.17, 36.37, 25.68, 67.82, 57.40, 72.20, 33.80, 55.30, 55.60 };
        const double nd = kNd[ui.glassType], vd = kVd[ui.glassType];
        const double k = 0.05 * (1.0 / (0.4861 * 0.4861) - 1.0 / (0.6563 * 0.6563));
        const double disp = (nd - 1.0) / (vd * k) * std::max(ui.dispersion, 0.0) / 0.4;
        // Shift so the index at 587.6 nm is exactly n_d.
        const double ior = nd + 0.05 * disp * (1.0 / (0.55 * 0.55) - 1.0 / (0.5876 * 0.5876));
        p.ior = (float)ior;
        p.dispersion = (float)disp;
    }
    p.spectral = p.dispersion > 0 ? 1 : 0;
    p.reflect = (float)std::max(ui.reflection, 0.0);
    p.roughness = (float)std::max(ui.roughness, 0.0);
    p.innerReflect = (float)std::max(ui.innerReflect, 0.0);
    // Tint: transmittance `tint` through one object unit (the mesh radius) at density 1.
    {
        const double tint[3] = { ui.tintR, ui.tintG, ui.tintB };
        const double dens = std::max(ui.tintDensity, 0.0);
        p.absorbOn = 0;
        for (int c = 0; c < 3; ++c) {
            const double t = std::min(std::max(tint[c], 1e-3), 1.0);
            p.absorb[c] = (float)(-std::log(t) * dens);
            if (p.absorb[c] > 0.0f) p.absorbOn = 1;
        }
    }
    p.bevelWidth = (float)(std::max(ui.bevelWidth, 0.0) * p.objScale);
    p.maxBounces = std::max(1, std::min(ui.maxBounces, 256));
    p.minSamples = std::max(1, ui.minSamples);
    p.maxSamples = std::max(p.minSamples, ui.maxSamples);
    p.noiseThreshold = (float)std::max(ui.noiseThreshold, 0.0);
    p.seed = ui.seed;
    p.worldMode = ui.worldMode;
    p.worldBlur = (float)std::max(ui.worldBlur, 0.0);
    p.feather = (float)std::max(ui.feather, 0.0);
    p.colorSpace = ui.colorSpace;
    p.boost = (float)ui.boost;
    p.threshold = (float)ui.threshold;
    p.mix = (float)ui.mix;
    // Denoiser edge-stopping strength: 0 = off; the default amount 0.5 gives sigma
    // 2.75, the best all-round setting in tests/fg3d_noise.py (1.0 -> 4.5, 2.0 -> 8).
    p.denoise = ui.denoise > 0 ? (float)(1.0 + 3.5 * std::min(ui.denoise, 2.0)) : 0.0f;

    int nl = 0;
    for (int i = 0; i < ui.numLights && i < FG_MAX_LIGHTS; ++i) {
        const UIParams::Light& L = ui.lights[i];
        if (L.intensity <= 0) continue;
        const double lz = std::min(std::max(L.depth, 0.01), 2.0);
        float rgb[3];
        kelvinToRGB(L.kelvin, rgb);
        float* d = &p.lights[nl * FG_LIGHT_FLOATS];
        d[0] = (float)((L.x - 0.5) * aspect * lz);
        d[1] = (float)((L.y - 0.5) * lz);
        d[2] = (float)lz;
        d[3] = (float)(std::max(L.size, 0.001) * 0.5 * lz);
        for (int c = 0; c < 3; ++c) d[4 + c] = (float)(rgb[c] * L.intensity);
        ++nl;
    }
    p.numLights = nl;
    p.numTris = scene.numTris();
    p.numNodes = scene.numNodes();
    // Bounding sphere of the root BVH box, in world space.
    p.boundC[0] = p.boundC[1] = p.boundC[2] = 0.0f;
    p.boundR = 0.0f;
    if (p.numNodes > 0) {
        const float* b = scene.nodesF.data();
        const double c[3] = { 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4]), 0.5 * (b[2] + b[5]) };
        const double h[3] = { 0.5 * (b[3] - b[0]), 0.5 * (b[4] - b[1]), 0.5 * (b[5] - b[2]) };
        for (int i = 0; i < 3; ++i)
            p.boundC[i] = (float)(p.objPos[i] + p.objScale * (p.objRot[3 * i] * c[0] + p.objRot[3 * i + 1] * c[1] +
                                                              p.objRot[3 * i + 2] * c[2]));
        p.boundR = (float)(p.objScale * std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]));
    }
}

}  // namespace fg3d
