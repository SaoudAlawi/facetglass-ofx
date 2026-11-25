// FacetGlass 3D - procedural glass shapes and OBJ import.
// SPDX-License-Identifier: MIT
//
// All shapes are authored around a unit radius in object space; the camera looks
// down +z, so z < 0 faces the camera. Building blocks: convex hulls (solids,
// shards), polar and rectangular height-field plates (lenses, ripples, honeycomb,
// Fresnel, organic panes), ring cuts (gems) and parametric surfaces (torus, bar).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

#include "fg3d_geom.h"
#include "fg3d_scene.h"

namespace fg3d {

using namespace geom;

int Mesh::addV(float x, float y, float z)
{
    v.push_back(x); v.push_back(y); v.push_back(z);
    return (int)(v.size() / 3) - 1;
}
void Mesh::tri(int a, int b, int c, int g)
{
    f.push_back(a); f.push_back(b); f.push_back(c);
    group.push_back(g);
}
void Mesh::quad(int a, int b, int c, int d, int g)
{
    tri(a, b, c, g);
    tri(a, c, d, g);
}

bool operator==(const ShapeParams& a, const ShapeParams& b)
{
    return a.shape == b.shape && a.facets == b.facets && a.bothSides == b.bothSides &&
           a.prismAngleDeg == b.prismAngleDeg && a.clearCentre == b.clearCentre && a.power == b.power &&
           a.thickness == b.thickness && a.detail == b.detail && a.seed == b.seed &&
           a.smoothAngleDeg == b.smoothAngleDeg && a.objPath == b.objPath && a.worldScale == b.worldScale &&
           a.depth == b.depth;
}

namespace {

V3 vert(const Mesh& m, int i) { return { m.v[3 * i], m.v[3 * i + 1], m.v[3 * i + 2] }; }
int addV(Mesh& m, V3 p) { return m.addV((float)p.x, (float)p.y, (float)p.z); }

// ---------------------------------------------------------------- randomness

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 747796405u + 2891336453u) {}
    double next()
    {
        s = s * 747796405u + 2891336453u;
        uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        w = (w >> 22u) ^ w;
        return (w >> 8) * (1.0 / 16777216.0);
    }
    double range(double a, double b) { return a + (b - a) * next(); }
};

double hash3(int x, int y, int z, int seed)
{
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u +
                 (uint32_t)seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xffffff) / 16777215.0;
}

// Smooth value noise in [-1, 1].
double noise3(double x, double y, double z, int seed)
{
    const int xi = (int)std::floor(x), yi = (int)std::floor(y), zi = (int)std::floor(z);
    double fx = x - xi, fy = y - yi, fz = z - zi;
    auto s = [](double t) { return t * t * t * (t * (t * 6 - 15) + 10); };
    fx = s(fx); fy = s(fy); fz = s(fz);
    double r = 0;
    for (int c = 0; c < 8; ++c) {
        const int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        const double w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
        r += w * hash3(xi + dx, yi + dy, zi + dz, seed);
    }
    return r * 2 - 1;
}

double fbm3(double x, double y, double z, int seed)
{
    return 0.6 * noise3(x, y, z, seed) + 0.3 * noise3(2.03 * x, 2.03 * y, 2.03 * z, seed + 17) +
           0.1 * noise3(4.1 * x, 4.1 * y, 4.1 * z, seed + 31);
}

// ---------------------------------------------------------------- orientation

void orientConvex(Mesh& m, size_t firstTri, V3 center)
{
    for (size_t t = firstTri; t < m.f.size() / 3; ++t) {
        int* f = &m.f[3 * t];
        V3 a = vert(m, f[0]), b = vert(m, f[1]), c = vert(m, f[2]);
        if (dot(cross(b - a, c - a), (a + b + c) * (1.0 / 3.0) - center) < 0) std::swap(f[1], f[2]);
    }
}

// A ray from a face along its normal must leave the solid (cross an even number of faces).
void orientByParity(Mesh& m)
{
    const size_t nt = m.f.size() / 3;
    std::vector<int> flip(nt, 0);
    for (size_t t = 0; t < nt; ++t) {
        const int* f = &m.f[3 * t];
        V3 a = vert(m, f[0]), b = vert(m, f[1]), c = vert(m, f[2]);
        V3 n = cross(b - a, c - a);
        if (len(n) < 1e-14) continue;
        n = norm(n);
        V3 o = a * 0.31 + b * 0.37 + c * 0.32 + n * 1e-6;
        int hits = 0;
        for (size_t k = 0; k < nt; ++k) {
            if (k == t) continue;
            const int* g = &m.f[3 * k];
            V3 v0 = vert(m, g[0]), e1 = vert(m, g[1]) - v0, e2 = vert(m, g[2]) - v0;
            V3 pv = cross(n, e2);
            double det = dot(e1, pv);
            if (std::fabs(det) < 1e-18) continue;
            double inv = 1.0 / det;
            V3 tv = o - v0;
            double u = dot(tv, pv) * inv;
            if (u < 0 || u > 1) continue;
            V3 qv = cross(tv, e1);
            double v = dot(n, qv) * inv;
            if (v < 0 || u + v > 1) continue;
            if (dot(e2, qv) * inv > 1e-9) ++hits;
        }
        flip[t] = hits & 1;
    }
    for (size_t t = 0; t < nt; ++t)
        if (flip[t]) std::swap(m.f[3 * t + 1], m.f[3 * t + 2]);
}

// ---------------------------------------------------------------- convex hull

// Incremental 3D convex hull of a small point set; appends outward triangles.
void convexHull(Mesh& m, const std::vector<V3>& pts, int group = 0)
{
    const int n = (int)pts.size();
    if (n < 4) return;
    // Initial tetrahedron from extreme, non-degenerate points.
    int i0 = 0, i1 = -1, i2 = -1, i3 = -1;
    double best = 0;
    for (int i = 1; i < n; ++i) if (len(pts[i] - pts[i0]) > best) { best = len(pts[i] - pts[i0]); i1 = i; }
    best = 0;
    for (int i = 0; i < n; ++i) {
        double a = len(cross(pts[i1] - pts[i0], pts[i] - pts[i0]));
        if (a > best) { best = a; i2 = i; }
    }
    best = 0;
    V3 nrm = cross(pts[i1] - pts[i0], pts[i2] - pts[i0]);
    for (int i = 0; i < n; ++i) {
        double d = std::fabs(dot(nrm, pts[i] - pts[i0]));
        if (d > best) { best = d; i3 = i; }
    }
    if (i1 < 0 || i2 < 0 || i3 < 0 || best < 1e-12) return;
    struct F { int a, b, c; bool alive; };
    std::vector<F> faces;
    auto add = [&](int a, int b, int c) { faces.push_back({ a, b, c, true }); };
    V3 cen = (pts[i0] + pts[i1] + pts[i2] + pts[i3]) * 0.25;
    auto addOriented = [&](int a, int b, int c) {
        if (dot(cross(pts[b] - pts[a], pts[c] - pts[a]), pts[a] - cen) < 0) std::swap(b, c);
        add(a, b, c);
    };
    addOriented(i0, i1, i2); addOriented(i0, i1, i3); addOriented(i0, i2, i3); addOriented(i1, i2, i3);
    const double eps = 1e-9;
    for (int p = 0; p < n; ++p) {
        if (p == i0 || p == i1 || p == i2 || p == i3) continue;
        std::vector<int> vis;
        for (int f = 0; f < (int)faces.size(); ++f) {
            if (!faces[f].alive) continue;
            const F& q = faces[f];
            V3 fn = cross(pts[q.b] - pts[q.a], pts[q.c] - pts[q.a]);
            if (dot(fn, pts[p] - pts[q.a]) > eps * len(fn)) vis.push_back(f);
        }
        if (vis.empty()) continue;
        std::map<std::pair<int, int>, int> edgeCount;
        for (int f : vis) {
            const F& q = faces[f];
            int e[3][2] = { { q.a, q.b }, { q.b, q.c }, { q.c, q.a } };
            for (auto& ed : e) edgeCount[{ ed[0], ed[1] }]++;
        }
        std::vector<std::pair<int, int>> horizon;
        for (auto& kv : edgeCount)
            if (!edgeCount.count({ kv.first.second, kv.first.first })) horizon.push_back(kv.first);
        for (int f : vis) faces[f].alive = false;
        for (auto& ed : horizon) add(ed.first, ed.second, p);
    }
    std::map<int, int> idx;
    for (const F& q : faces) {
        if (!q.alive) continue;
        int t[3] = { q.a, q.b, q.c };
        for (int k = 0; k < 3; ++k)
            if (!idx.count(t[k])) idx[t[k]] = addV(m, pts[t[k]]);
        m.tri(idx[q.a], idx[q.b], idx[q.c], group);
    }
}

// ---------------------------------------------------------------- plates

// Polar height-field plate over angles [th0, th1] (full disc if closed).
// front(x, y) and back(x, y) give the two surfaces' z; back must stay behind front.
void polarPlate(Mesh& m, int R, int S, double th0, double th1, bool closed,
                const std::function<double(double, double)>& front, const std::function<double(double, double)>& back,
                bool frontSmooth, bool backSmooth)
{
    const int cols = closed ? S : S + 1;
    const int gf = frontSmooth ? 1 : 0, gb = backSmooth ? 2 : 0;
    int cf = m.addV(0, 0, (float)front(0, 0)), cb = m.addV(0, 0, (float)back(0, 0));
    std::vector<int> F((size_t)R * cols), B((size_t)R * cols);
    for (int j = 0; j < cols; ++j) {
        const double th = th0 + (th1 - th0) * j / S, c = std::cos(th), s = std::sin(th);
        for (int i = 1; i <= R; ++i) {
            const double r = (double)i / R, x = r * c, y = r * s;
            F[(size_t)(i - 1) * cols + j] = m.addV((float)x, (float)y, (float)front(x, y));
            B[(size_t)(i - 1) * cols + j] = m.addV((float)x, (float)y, (float)back(x, y));
        }
    }
    auto at = [&](std::vector<int>& A, int i, int j) { return A[(size_t)(i - 1) * cols + j]; };
    for (int j = 0; j < S; ++j) {
        const int jn = closed ? (j + 1) % S : j + 1;
        m.tri(cf, at(F, 1, jn), at(F, 1, j), gf);
        m.tri(cb, at(B, 1, j), at(B, 1, jn), gb);
        for (int i = 1; i < R; ++i) {
            m.quad(at(F, i, j), at(F, i, jn), at(F, i + 1, jn), at(F, i + 1, j), gf);
            m.quad(at(B, i, j), at(B, i + 1, j), at(B, i + 1, jn), at(B, i, jn), gb);
        }
        m.quad(at(F, R, j), at(F, R, jn), at(B, R, jn), at(B, R, j));   // rim
    }
    if (!closed) {
        // Cut walls along th0 (facing -tangent) and th1 (facing +tangent), meeting at the centre.
        for (int side = 0; side < 2; ++side) {
            const int j = side == 0 ? 0 : S;
            int pf = cf, pb = cb;
            for (int i = 1; i <= R; ++i) {
                const int f = at(F, i, j), b = at(B, i, j);
                if (side == 0) m.quad(pf, pb, b, f); else m.quad(pf, f, b, pb);
                pf = f; pb = b;
            }
        }
    }
    m.trustWinding = true;
}

// Rectangular height-field plate over x in [-1, 1], y in [-hy, hy].
void rectPlate(Mesh& m, int nx, int ny, double hy, const std::function<double(double, double)>& front,
               const std::function<double(double, double)>& back, bool frontSmooth, bool backSmooth)
{
    const int gf = frontSmooth ? 1 : 0, gb = backSmooth ? 2 : 0;
    std::vector<int> F((size_t)(nx + 1) * (ny + 1)), B(F.size());
    auto id = [&](int i, int j) { return (size_t)j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            const double x = -1.0 + 2.0 * i / nx, y = -hy + 2.0 * hy * j / ny;
            F[id(i, j)] = m.addV((float)x, (float)y, (float)front(x, y));
            B[id(i, j)] = m.addV((float)x, (float)y, (float)back(x, y));
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            m.quad(F[id(i, j)], F[id(i, j + 1)], F[id(i + 1, j + 1)], F[id(i + 1, j)], gf);
            m.quad(B[id(i, j)], B[id(i + 1, j)], B[id(i + 1, j + 1)], B[id(i, j + 1)], gb);
        }
    for (int j = 0; j < ny; ++j) {
        m.quad(F[id(0, j)], B[id(0, j)], B[id(0, j + 1)], F[id(0, j + 1)]);
        m.quad(F[id(nx, j)], F[id(nx, j + 1)], B[id(nx, j + 1)], B[id(nx, j)]);
    }
    for (int i = 0; i < nx; ++i) {
        m.quad(F[id(i, 0)], F[id(i + 1, 0)], B[id(i + 1, 0)], B[id(i, 0)]);
        m.quad(F[id(i, ny)], B[id(i, ny)], B[id(i + 1, ny)], F[id(i + 1, ny)]);
    }
    m.trustWinding = true;
}

// Lens curvature (object units) for a magnification, at the object's depth/scale.
double lensCurvature(const ShapeParams& sp, double powerScale = 1.0)
{
    const double z = std::min(std::max((double)sp.depth, 0.01), 0.95);
    const double P = sp.power * powerScale;
    const double mag = P >= 0 ? 1.0 + 0.1 * P : 1.0 / (1.0 - 0.1 * P);
    const double cWorld = (1.0 - 1.0 / mag) / (z * (1.0 - z)) / 0.5;
    return std::max(-1.6, std::min(1.6, cWorld * std::max(1e-4, (double)sp.worldScale)));
}

double tanDeg(double d) { return std::tan(std::min(std::max(d, 0.0), 80.0) * kPi / 180.0); }

// ---------------------------------------------------------------- prisms & filters

// Slab extruded along y: flat front at zf, back profile (x, z) with x non-decreasing.
void slabProfile(Mesh& m, const std::vector<std::pair<double, double>>& back, double zf, double halfLen)
{
    const double L = halfLen;
    std::vector<int> fb, bb, ft, bt;
    for (auto& pt : back) {
        fb.push_back(m.addV((float)pt.first, (float)-L, (float)zf));
        bb.push_back(m.addV((float)pt.first, (float)-L, (float)pt.second));
        ft.push_back(m.addV((float)pt.first, (float)L, (float)zf));
        bt.push_back(m.addV((float)pt.first, (float)L, (float)pt.second));
    }
    const size_t n = back.size();
    for (size_t i = 0; i + 1 < n; ++i) {
        m.quad(bb[i], bb[i + 1], bt[i + 1], bt[i]);
        if (back[i + 1].first > back[i].first) {
            m.quad(fb[i], ft[i], ft[i + 1], fb[i + 1]);
            m.quad(fb[i], fb[i + 1], bb[i + 1], bb[i]);
            m.quad(ft[i], bt[i], bt[i + 1], ft[i + 1]);
        }
    }
    m.quad(fb[0], bb[0], bt[0], ft[0]);
    m.quad(fb[n - 1], ft[n - 1], bt[n - 1], bb[n - 1]);
}

void facetPlate(Mesh& m, const ShapeParams& sp)
{
    const int N = std::max(1, std::min(sp.facets, 32));
    const double t = std::max(0.02, (double)sp.thickness), zf = -t * 0.5, z0 = t * 0.5;
    const double a = sp.prismAngleDeg * kPi / 180.0;
    auto slope = [&](int k) { return std::tan(std::min(a * (k + 1), 1.3)); };
    std::vector<std::pair<double, double>> back;
    if (!sp.bothSides) {
        for (int k = 0; k < N; ++k) {
            double xa = -1.0 + 2.0 * k / N, xb = -1.0 + 2.0 * (k + 1) / N;
            back.push_back({ xa, z0 });
            back.push_back({ xb, z0 + slope(k) * (xb - xa) });
        }
    } else {
        const double w = 1.0 / N;
        for (int k = N - 1; k >= 0; --k) {
            back.push_back({ -(k + 1) * w, z0 + slope(k) * w });
            back.push_back({ -k * w, z0 });
        }
        for (int k = 0; k < N; ++k) {
            back.push_back({ k * w, z0 });
            back.push_back({ (k + 1) * w, z0 + slope(k) * w });
        }
    }
    slabProfile(m, back, zf, 4.0);
}

void extrudeY(Mesh& m, const std::vector<std::pair<double, double>>& poly, double halfLen, int sideGroup)
{
    const size_t n = poly.size();
    std::vector<int> b, t;
    for (auto& p : poly) {
        b.push_back(m.addV((float)p.first, (float)-halfLen, (float)p.second));
        t.push_back(m.addV((float)p.first, (float)halfLen, (float)p.second));
    }
    for (size_t i = 0; i < n; ++i) m.quad(b[i], b[(i + 1) % n], t[(i + 1) % n], t[i], sideGroup);
    for (size_t i = 1; i + 1 < n; ++i) {
        m.tri(b[0], b[i + 1], b[i]);
        m.tri(t[0], t[i], t[i + 1]);
    }
}

void radialPrism(Mesh& m, const ShapeParams& sp, bool half)
{
    const int N = std::max(3, std::min(sp.facets < 3 ? 6 : sp.facets, 64));
    const double t = std::max(0.02, (double)sp.thickness);
    const double r0 = std::min(std::max((double)sp.clearCentre, 0.01), 0.95);
    const double a = std::min(sp.prismAngleDeg * kPi / 180.0, 1.3);
    const double zf = -t * 0.5, zi = t * 0.5, zo = zi + std::tan(a) * (1.0 - r0);
    int cf = m.addV(0, 0, (float)zf), cb = m.addV(0, 0, (float)zi);
    std::vector<int> fo(N), bi(N), bo(N);
    for (int k = 0; k < N; ++k) {
        const double th = 2 * kPi * (k + 0.5) / N, c = std::cos(th), s = std::sin(th);
        const double z = (!half || c > 0.01) ? zo : zi + 0.02 * (1.0 - r0);
        fo[k] = m.addV((float)c, (float)s, (float)zf);
        bi[k] = m.addV((float)(r0 * c), (float)(r0 * s), (float)zi);
        bo[k] = m.addV((float)c, (float)s, (float)z);
    }
    for (int k = 0; k < N; ++k) {
        const int j = (k + 1) % N;
        m.tri(cf, fo[j], fo[k]);
        m.tri(cb, bi[k], bi[j]);
        m.quad(bi[k], bo[k], bo[j], bi[j]);
        m.quad(fo[k], fo[j], bo[j], bo[k]);
    }
    m.trustWinding = true;
}

void kaleidoscope(Mesh& m, const ShapeParams& sp)
{
    // Polygonal glass rod along the view axis: light reflects off its inner walls.
    const int n = std::max(3, std::min(sp.facets, 12));
    std::vector<V3> pts;
    const double len = 1.0 + 2.0 * std::max(0.05, (double)sp.thickness);
    for (int k = 0; k < n; ++k) {
        const double a = 2 * kPi * k / n + kPi / 2;
        pts.push_back({ 0.8 * std::cos(a), 0.8 * std::sin(a), -0.6 });
        pts.push_back({ 0.8 * std::cos(a), 0.8 * std::sin(a), -0.6 + len });
    }
    convexHull(m, pts);
}

void polarHeight(Mesh& m, const ShapeParams& sp, int R, int S, bool half, const std::function<double(double, double)>& h,
                 double minThick)
{
    const double t = std::max(minThick, (double)sp.thickness * 0.5);
    polarPlate(m, R, S, half ? -kPi / 2 : 0.0, half ? kPi / 2 : 2 * kPi, !half,
               [&](double, double) { return -t * 0.5; },
               [&](double x, double y) { return t * 0.5 + h(x, y); }, false, true);
}

void rippleRings(Mesh& m, const ShapeParams& sp, bool half)
{
    const double f = std::max(1, sp.facets) + 2.0;                 // ripples across the radius
    const double A = 0.5 * tanDeg(sp.prismAngleDeg * 1.5) / (2 * kPi * f);
    m.smoothAngleDeg = 80.0f;
    polarHeight(m, sp, 120, half ? 96 : 160, half,
                [&](double x, double y) {
                    const double r = std::sqrt(x * x + y * y);
                    return A * (1.0 + std::sin(2 * kPi * f * r - kPi / 2));
                }, 0.05);
}

void haloRing(Mesh& m, const ShapeParams& sp)
{
    const double r0 = std::min(std::max((double)sp.clearCentre, 0.0), 0.95);
    const double s = tanDeg(sp.prismAngleDeg * 2.0);
    m.smoothAngleDeg = 40.0f;
    polarHeight(m, sp, 64, 160, false,
                [&](double x, double y) {
                    const double r = std::sqrt(x * x + y * y);
                    return r <= r0 ? 0.0 : s * (r - r0);
                }, 0.05);
}

void honeycomb(Mesh& m, const ShapeParams& sp)
{
    // Hexagonal lenslets: Facets sets the cells across the disc (2 * facets + 1).
    const double cells = 2.0 * std::max(1, sp.facets) + 1.0;
    const double rc = 2.0 / cells / std::sqrt(3.0) * 1.15;          // cell circumradius
    const double slope = tanDeg(sp.prismAngleDeg * 1.5);
    m.smoothAngleDeg = 25.0f;
    polarHeight(m, sp, 150, 360, false,
                [&](double x, double y) {
                    // Nearest centre on a hex lattice (pointy-top), via cube rounding.
                    const double q = (std::sqrt(3.0) / 3.0 * x - y / 3.0) / rc, r = (2.0 / 3.0 * y) / rc;
                    double cx = q, cz = r, cy = -q - r;
                    double rx = std::round(cx), ry = std::round(cy), rz = std::round(cz);
                    const double dx = std::fabs(rx - cx), dy = std::fabs(ry - cy), dz = std::fabs(rz - cz);
                    if (dx > dy && dx > dz) rx = -ry - rz; else if (dy > dz) ry = -rx - rz; else rz = -rx - ry;
                    const double px = rc * std::sqrt(3.0) * (rx + rz / 2.0), py = rc * 1.5 * rz;
                    const double d2 = (x - px) * (x - px) + (y - py) * (y - py);
                    return slope / (2.0 * rc) * std::max(0.0, rc * rc - d2);
                }, 0.05);
}

void fresnelLens(Mesh& m, const ShapeParams& sp)
{
    const double c = lensCurvature(sp);
    const int zones = 3 * std::max(1, sp.facets);
    const double H = std::fabs(0.5 * c);                           // full lens sag
    const double step = std::max(H / zones, 1e-4);
    m.smoothAngleDeg = 30.0f;
    polarHeight(m, sp, 240, 128, false,
                [&](double x, double y) {
                    const double r2 = x * x + y * y;
                    const double h = 0.5 * std::fabs(c) * (1.0 - r2);
                    const double v = std::fmod(h, step);
                    return c >= 0 ? v : step - v;
                }, 0.04);
}

// ---------------------------------------------------------------- lenses

void lens(Mesh& m, const ShapeParams& sp, bool half, bool both)
{
    const double c = lensCurvature(sp, both ? 0.5 : 1.0);
    const double sag = 0.5 * c;
    const double base = 0.06 + 0.1 * std::max(0.0, (double)sp.thickness - 0.3);
    const double minC = both ? 2 * std::max(0.0, -sag) : std::max(0.0, -sag);
    const double zf = -0.05 - (both ? std::max(0.0, sag) : 0.0);
    auto bulge = [&](double x, double y) { return sag * (1.0 - (x * x + y * y)); };
    m.smoothAngleDeg = 60.0f;
    polarPlate(m, 32, half ? 64 : 128, half ? -kPi / 2 : 0.0, half ? kPi / 2 : 2 * kPi, !half,
               [&](double x, double y) { return both ? zf + base * 0.5 - bulge(x, y) + minC * 0.5 : zf; },
               [&](double x, double y) { return zf + base + minC + bulge(x, y) + (both ? minC * 0.5 : 0.0); },
               both, true);
}

void cylinderLens(Mesh& m, const ShapeParams& sp)
{
    const double c = lensCurvature(sp), sag = 0.5 * c;
    const double base = 0.06 + std::max(0.0, -sag) + 0.1 * std::max(0.0, (double)sp.thickness - 0.3);
    m.smoothAngleDeg = 60.0f;
    rectPlate(m, 96, 2, 1.6, [&](double, double) { return -0.05; },
              [&](double x, double) { return -0.05 + base + sag * (1.0 - x * x); }, false, true);
}

void rod(Mesh& m)
{
    std::vector<std::pair<double, double>> poly;
    for (int k = 0; k < 64; ++k) poly.push_back({ 0.5 * std::cos(2 * kPi * k / 64), 0.5 * std::sin(2 * kPi * k / 64) });
    extrudeY(m, poly, 4.0, 1);
}

// ---------------------------------------------------------------- gems

// Brilliant-style cut over a girdle outline given by unit-scale points (CCW).
void brilliant(Mesh& m, const std::vector<std::pair<double, double>>& outline)
{
    const int n = (int)outline.size();
    const double zt = -0.38, zc = -0.22, zg0 = -0.06, zg1 = 0.04, zp = 0.9;
    double cx = 0, cy = 0;
    for (auto& p : outline) { cx += p.first; cy += p.second; }
    cx /= n; cy /= n;
    auto ring = [&](int k, double s, double zz, bool halfStep) {
        auto p = outline[k];
        if (halfStep) {
            auto q = outline[(k + 1) % n];
            p = { 0.5 * (p.first + q.first), 0.5 * (p.second + q.second) };
        }
        return m.addV((float)(cx + (p.first - cx) * s), (float)(cy + (p.second - cy) * s), (float)zz);
    };
    int ct = m.addV((float)cx, (float)cy, (float)zt), cul = m.addV((float)cx, (float)cy, (float)zp);
    std::vector<int> tab(n), crown(n), g0(n), g1(n);
    for (int k = 0; k < n; ++k) {
        tab[k] = ring(k, 0.52, zt, false);
        crown[k] = ring(k, 0.82, zc, true);
        g0[k] = ring(k, 1.0, zg0, false);
        g1[k] = ring(k, 1.0, zg1, false);
    }
    for (int k = 0; k < n; ++k) {
        const int j = (k + 1) % n;
        m.tri(ct, tab[k], tab[j]);
        m.tri(tab[k], crown[k], tab[j]);
        m.tri(tab[j], crown[k], crown[j]);
        m.tri(crown[k], g0[j], crown[j]);
        m.tri(crown[k], g0[k], g0[j]);
        m.quad(g0[k], g1[k], g1[j], g0[j]);
        m.tri(g1[k], cul, g1[j]);
    }
}

// Step cut: rings of the outline scaled at given depths, flat table and keel.
void stepCut(Mesh& m, const std::vector<std::pair<double, double>>& outline,
             const std::vector<std::pair<double, double>>& rings /* (scale, z) top to bottom */)
{
    const int n = (int)outline.size();
    std::vector<std::vector<int>> R;
    for (auto& r : rings) {
        std::vector<int> ids(n);
        for (int k = 0; k < n; ++k)
            ids[k] = m.addV((float)(outline[k].first * r.first), (float)(outline[k].second * r.first), (float)r.second);
        R.push_back(ids);
    }
    int top = m.addV(0, 0, (float)rings.front().second), bot = m.addV(0, 0, (float)rings.back().second);
    for (int k = 0; k < n; ++k) {
        const int j = (k + 1) % n;
        m.tri(top, R.front()[j], R.front()[k]);
        for (size_t r = 0; r + 1 < R.size(); ++r) m.quad(R[r][k], R[r][j], R[r + 1][j], R[r + 1][k]);
        m.tri(bot, R.back()[k], R.back()[j]);
    }
}

std::vector<std::pair<double, double>> polygonOutline(int n, double rx, double ry, double phase)
{
    std::vector<std::pair<double, double>> o;
    for (int k = 0; k < n; ++k) {
        const double a = phase + 2 * kPi * k / n;
        o.push_back({ rx * std::cos(a), ry * std::sin(a) });
    }
    return o;
}

void gemCut(Mesh& m, int shape)
{
    const int n = 24;
    std::vector<std::pair<double, double>> o;
    if (shape == kGem) o = polygonOutline(16, 1, 1, 0);
    else if (shape == kMarquiseCut) {
        for (int k = 0; k < n; ++k) {  // pointed ellipse: two circular arcs
            const double t = 2 * kPi * k / n, s = std::sin(t);
            o.push_back({ std::cos(t), 0.55 * s * (0.6 + 0.4 * std::fabs(s)) });
        }
    } else if (shape == kPearCut) {
        for (int k = 0; k < n; ++k) {
            const double t = 2 * kPi * k / n;
            const double w = 0.62 * (1.0 + 0.35 * std::cos(t)) * std::sin(t);
            o.push_back({ std::cos(t), w });
        }
    } else {  // heart
        for (int k = 0; k < n; ++k) {
            const double t = 2 * kPi * k / n;
            const double x = 16 * std::pow(std::sin(t), 3);
            const double y = 13 * std::cos(t) - 5 * std::cos(2 * t) - 2 * std::cos(3 * t) - std::cos(4 * t);
            o.push_back({ x / 17.0, (y + 2.0) / 17.0 });
        }
    }
    brilliant(m, o);
}

// ---------------------------------------------------------------- solids

std::vector<V3> icoPoints()
{
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<V3> v = { { -1, t, 0 }, { 1, t, 0 }, { -1, -t, 0 }, { 1, -t, 0 }, { 0, -1, t }, { 0, 1, t },
                          { 0, -1, -t }, { 0, 1, -t }, { t, 0, -1 }, { t, 0, 1 }, { -t, 0, -1 }, { -t, 0, 1 } };
    for (auto& p : v) p = norm(p);
    return v;
}

// Geodesic sphere vertices/faces (unit radius).
void icosphere(int detail, std::vector<V3>& v, std::vector<int>& f)
{
    v = icoPoints();
    f = { 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
          3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 };
    for (int d = 0; d < std::min(std::max(detail, 0), 5); ++d) {
        std::map<std::pair<int, int>, int> mid;
        auto midpoint = [&](int a, int b) {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            v.push_back(norm((v[a] + v[b]) * 0.5));
            return mid[key] = (int)v.size() - 1;
        };
        std::vector<int> nf;
        for (size_t i = 0; i < f.size(); i += 3) {
            int a = f[i], b = f[i + 1], c = f[i + 2];
            int ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            int t[12] = { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca };
            nf.insert(nf.end(), t, t + 12);
        }
        f.swap(nf);
    }
}

// Sphere-like mesh with a radius function of direction; outward winding.
void radialSolid(Mesh& m, int detail, const std::function<V3(V3)>& shapeFn, int group)
{
    std::vector<V3> v;
    std::vector<int> f;
    icosphere(detail, v, f);
    const int base = (int)(m.v.size() / 3);
    for (auto& d : v) addV(m, shapeFn(d));
    for (size_t i = 0; i < f.size(); i += 3) {
        // icosphere faces are CCW from outside in this table for the +x half; fix per face
        V3 a = v[f[i]], b = v[f[i + 1]], c = v[f[i + 2]];
        if (dot(cross(b - a, c - a), a + b + c) >= 0) m.tri(base + f[i], base + f[i + 1], base + f[i + 2], group);
        else m.tri(base + f[i], base + f[i + 2], base + f[i + 1], group);
    }
    m.trustWinding = true;
}

void dodecahedron(Mesh& m)
{
    // Dual of the icosahedron: face centres become vertices.
    std::vector<V3> v;
    std::vector<int> f;
    icosphere(0, v, f);
    std::vector<V3> pts;
    for (size_t i = 0; i < f.size(); i += 3) pts.push_back(norm(v[f[i]] + v[f[i + 1]] + v[f[i + 2]]));
    convexHull(m, pts);
}

void torus(Mesh& m)
{
    const int U = 96, Vn = 48;
    const double R = 0.68, r = 0.3;
    std::vector<int> id((size_t)U * Vn);
    for (int i = 0; i < U; ++i)
        for (int j = 0; j < Vn; ++j) {
            const double u = 2 * kPi * i / U, w = 2 * kPi * j / Vn;
            id[(size_t)i * Vn + j] = addV(m, { (R + r * std::cos(w)) * std::cos(u), (R + r * std::cos(w)) * std::sin(u),
                                               r * std::sin(w) });
        }
    for (int i = 0; i < U; ++i)
        for (int j = 0; j < Vn; ++j) {
            const int in = (i + 1) % U, jn = (j + 1) % Vn;
            m.quad(id[(size_t)i * Vn + j], id[(size_t)in * Vn + j], id[(size_t)in * Vn + jn], id[(size_t)i * Vn + jn], 1);
        }
    m.trustWinding = true;
}

void twistedBar(Mesh& m, const ShapeParams& sp)
{
    const int N = 96;
    const double twist = sp.prismAngleDeg * 20.0 * kPi / 180.0, hw = 0.3, L = 3.0;
    std::vector<std::vector<int>> ring;
    for (int i = 0; i <= N; ++i) {
        const double y = -L + 2 * L * i / N, a = twist * (double)i / N;
        std::vector<int> r;
        for (int k = 0; k < 4; ++k) {
            const double ang = a + kPi / 4 + kPi / 2 * k;
            r.push_back(addV(m, { hw * std::sqrt(2.0) * std::cos(ang), y, hw * std::sqrt(2.0) * std::sin(ang) }));
        }
        ring.push_back(r);
    }
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < 4; ++k) m.quad(ring[i][k], ring[i + 1][k], ring[i + 1][(k + 1) % 4], ring[i][(k + 1) % 4], 1 + k);
    m.quad(ring[0][0], ring[0][1], ring[0][2], ring[0][3]);
    m.quad(ring[N][3], ring[N][2], ring[N][1], ring[N][0]);
}

void shards(Mesh& m, const ShapeParams& sp)
{
    Rng rng((uint32_t)sp.seed * 104729u + 7u);
    const int K = std::max(1, std::min(sp.facets * 2, 24));
    for (int s = 0; s < K; ++s) {
        // Thin random convex piece, randomly placed and turned.
        const double ax = rng.range(0, 2 * kPi), ay = rng.range(0, kPi), az = rng.range(0, 2 * kPi);
        const double sc = rng.range(0.2, 0.42);
        V3 c = { rng.range(-0.75, 0.75), rng.range(-0.75, 0.75), rng.range(-0.15, 0.15) };
        std::vector<V3> pts;
        for (int p = 0; p < 9; ++p) {
            V3 q = { rng.range(-1, 1), rng.range(-1, 1), rng.range(-0.18, 0.18) };
            // rotate
            V3 r1 = { q.x, q.y * std::cos(ax) - q.z * std::sin(ax), q.y * std::sin(ax) + q.z * std::cos(ax) };
            V3 r2 = { r1.x * std::cos(ay) + r1.z * std::sin(ay), r1.y, -r1.x * std::sin(ay) + r1.z * std::cos(ay) };
            V3 r3 = { r2.x * std::cos(az) - r2.y * std::sin(az), r2.x * std::sin(az) + r2.y * std::cos(az), r2.z };
            pts.push_back(c + r3 * sc);
        }
        convexHull(m, pts);
    }
}

void crystal(Mesh& m, V3 base, V3 axis, double radius, double length, double twist)
{
    axis = norm(axis);
    V3 ref = std::fabs(axis.z) < 0.9 ? V3{ 0, 0, 1 } : V3{ 1, 0, 0 };
    V3 u = norm(cross(axis, ref)), w = cross(axis, u);
    std::vector<V3> pts;
    for (int k = 0; k < 6; ++k) {
        const double a = twist + kPi * k / 3.0;
        V3 d = u * std::cos(a) + w * std::sin(a);
        pts.push_back(base + d * radius);
        pts.push_back(base + d * radius + axis * (length * 0.75));
    }
    pts.push_back(base + axis * length);
    pts.push_back(base - axis * (radius * 0.5));
    convexHull(m, pts);
}

void crystalCluster(Mesh& m, const ShapeParams& sp)
{
    Rng rng((uint32_t)sp.seed * 7919u + 13u);
    const int K = std::max(1, std::min(sp.facets, 12));
    for (int i = 0; i < K; ++i) {
        const double th = rng.range(0, 2 * kPi), ph = rng.range(0.15, 1.25);
        V3 axis = { std::cos(th) * std::sin(ph), std::sin(th) * std::sin(ph), -std::cos(ph) * 0.6 };
        const double len = rng.range(0.55, 1.0), rad = rng.range(0.12, 0.22);
        V3 base = { rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), rng.range(-0.05, 0.1) };
        crystal(m, base, axis, rad, len, rng.range(0, kPi));
    }
}

// ---------------------------------------------------------------- organic panes

void wavyGlass(Mesh& m, const ShapeParams& sp)
{
    const double k = 1.2 * std::max(1, sp.facets), A = 0.25 * tanDeg(sp.prismAngleDeg) / k;
    const double t = std::max(0.05, (double)sp.thickness * 0.5);
    const int seed = sp.seed;
    m.smoothAngleDeg = 90.0f;
    rectPlate(m, 140, 140, 1.0,
              [&](double x, double y) { return -t * 0.5 + 0.5 * A * fbm3(x * k, y * k, 0.5, seed + 101); },
              [&](double x, double y) { return t * 0.5 + A * (1.0 + fbm3(x * k, y * k, 3.5, seed)); }, true, true);
}

void flutedGlass(Mesh& m, const ShapeParams& sp)
{
    const double f = 2.0 * std::max(1, sp.facets) + 2.0;           // flutes across
    const double w = 2.0 / f, depth = std::min(1.0, tanDeg(sp.prismAngleDeg) * 0.8) * w * 0.5;
    const double t = std::max(0.05, (double)sp.thickness * 0.5);
    m.smoothAngleDeg = 35.0f;
    rectPlate(m, (int)(f * 16), 2, 1.6, [&](double, double) { return -t * 0.5; },
              [&](double x, double) {
                  const double u = std::fmod(x + 1.0, w) / w * 2.0 - 1.0;
                  return t * 0.5 + depth * std::sqrt(std::max(0.0, 1.0 - u * u));
              }, false, true);
}

void bumpyPane(Mesh& m, const ShapeParams& sp, bool drops)
{
    Rng rng((uint32_t)sp.seed * 6151u + (drops ? 3u : 5u));
    const int n = std::max(1, sp.facets) * (drops ? 15 : 8);
    struct Bump { double x, y, r, h; };
    std::vector<Bump> b;
    const double steep = std::min(1.0, tanDeg(sp.prismAngleDeg) * 5.0);
    for (int i = 0; i < n; ++i) {
        const double r = drops ? rng.range(0.03, 0.13) : rng.range(0.08, 0.22);
        b.push_back({ rng.range(-1, 1), rng.range(-1, 1), r, (drops ? 1.0 : -0.6) * r * steep });
    }
    const double t = std::max(0.05, (double)sp.thickness * 0.5) + (drops ? 0.0 : 0.2 * steep);
    m.smoothAngleDeg = drops ? 50.0f : 70.0f;
    rectPlate(m, 200, 200, 1.0, [&](double, double) { return -t * 0.5; },
              [&](double x, double y) {
                  double h = 0;
                  for (const Bump& q : b) {
                      const double d2 = ((x - q.x) * (x - q.x) + (y - q.y) * (y - q.y)) / (q.r * q.r);
                      if (d2 < 1.0) {
                          const double v = q.h * std::sqrt(1.0 - d2);
                          h = q.h > 0 ? std::max(h, v) : std::min(h, v);
                      }
                  }
                  return t * 0.5 + h;
              }, false, true);
}

void bentSlab(Mesh& m, const ShapeParams& sp)
{
    const double A = tanDeg(sp.prismAngleDeg * 5.0) * 0.6, t = std::max(0.05, (double)sp.thickness * 0.5);
    auto bend = [&](double x, double y) { return A * std::sin(kPi * y / 1.6) + 0.25 * A * x * x; };
    m.smoothAngleDeg = 60.0f;
    // A bent sheet of even thickness barely refracts: let the thickness swell and
    // thin along the bend so it warps the picture like a stretched lens.
    rectPlate(m, 48, 96, 1.6, [&](double x, double y) { return bend(x, y) - t * 0.5; },
              [&](double x, double y) {
                  return bend(x, y) + t * 0.5 + 0.35 * A * (1.0 + std::sin(kPi * y / 1.6 + 0.8)) * (1.0 - 0.3 * x * x);
              }, true, true);
}

// ---------------------------------------------------------------- OBJ

bool loadOBJ(const std::string& path, Mesh& m, std::string* err)
{
    std::ifstream in(path);
    if (!in) { if (err) *err = "cannot open " + path; return false; }
    std::vector<V3> v;
    std::vector<std::vector<int>> faces;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            V3 p;
            ss >> p.x >> p.y >> p.z;
            v.push_back(p);
        } else if (tag == "f") {
            std::vector<int> f;
            std::string tok;
            while (ss >> tok) {
                int i = std::atoi(tok.c_str());  // "i", "i/t", "i//n", "i/t/n"
                if (i < 0) i = (int)v.size() + i + 1;
                if (i >= 1 && i <= (int)v.size()) f.push_back(i - 1);
            }
            if (f.size() >= 3) faces.push_back(f);
        }
    }
    if (v.empty() || faces.empty()) { if (err) *err = "no geometry in " + path; return false; }
    // Centre and scale to unit radius; OBJ is y-up facing +z, ours looks down +z: flip z.
    V3 lo = v[0], hi = v[0];
    for (auto& p : v) {
        lo = { std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z) };
        hi = { std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z) };
    }
    const V3 c = (lo + hi) * 0.5;
    double r = 0;
    for (auto& p : v) r = std::max(r, len(p - c));
    const double s = r > 0 ? 1.0 / r : 1.0;
    for (auto& p : v) m.addV((float)((p.x - c.x) * s), (float)((p.y - c.y) * s), (float)(-(p.z - c.z) * s));
    for (auto& f : faces)
        for (size_t k = 1; k + 1 < f.size(); ++k) m.tri(f[0], f[k + 1], f[k], 1);  // z flip reverses winding
    return true;
}

}  // namespace

const char* shapeName(int shape)
{
    static const char* names[kShapeCount] = {
        "Faceted Plate", "Wedge Prism", "Radial Prism", "Half Radial Prism", "Kaleidoscope", "Halo Ring",
        "Ripple Rings", "Half Ripple", "Honeycomb", "Fresnel Lens",
        "Round Diopter", "Split Diopter", "Double Convex Lens", "Cylinder Lens", "Rod",
        "Round Brilliant", "Emerald Cut", "Princess Cut", "Pear Cut", "Heart Cut", "Marquise Cut", "Crystal Cluster",
        "Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron", "Geodesic Sphere", "Glass Ball",
        "Pyramid", "Hexagonal Prism", "Ring",
        "Wavy Glass", "Fluted Glass", "Bubble Glass", "Water Drops", "Ice Cube", "Melted Blob", "Shards",
        "Twisted Bar", "Bent Slab",
        "Custom OBJ" };
    return shape >= 0 && shape < kShapeCount ? names[shape] : "?";
}

const char* shapeCategory(int shape)
{
    if (shape <= kFresnelLens) return "Prisms & Filters";
    if (shape <= kRod) return "Lenses";
    if (shape <= kCrystalCluster) return "Gems";
    if (shape <= kTorus) return "Solids";
    if (shape <= kBentSlab) return "Organic";
    return "Custom";
}

Mesh makeShape(const ShapeParams& sp, std::string* err)
{
    Mesh m;
    bool parity = true;
    switch (sp.shape) {
    case kFacetPlate: facetPlate(m, sp); break;
    case kWedgePrism: {
        const double h = 0.577 * std::max(0.3, sp.thickness / 0.3);
        extrudeY(m, { { -1.0, -h }, { 1.0, -h }, { 0.0, 2 * h } }, 4.0, 0);
        break;
    }
    case kRadialPrism: radialPrism(m, sp, false); break;
    case kHalfRadial: radialPrism(m, sp, true); break;
    case kKaleidoscope: kaleidoscope(m, sp); break;
    case kHaloRing: haloRing(m, sp); break;
    case kRippleRings: rippleRings(m, sp, false); break;
    case kHalfRipple: rippleRings(m, sp, true); break;
    case kHoneycomb: honeycomb(m, sp); break;
    case kFresnelLens: fresnelLens(m, sp); break;
    case kRoundDiopter: lens(m, sp, false, false); break;
    case kSplitDiopter: lens(m, sp, true, false); break;
    case kDoubleConvex: lens(m, sp, false, true); break;
    case kCylinderLens: cylinderLens(m, sp); break;
    case kRod: rod(m); break;
    case kGem: case kMarquiseCut: case kPearCut: case kHeartCut: gemCut(m, sp.shape); break;
    case kEmeraldCut: {
        // Rectangle with cut corners, stepped crown and pavilion.
        std::vector<std::pair<double, double>> o = { { 0.78, -0.62 }, { 1.0, -0.4 }, { 1.0, 0.4 }, { 0.78, 0.62 },
                                                     { -0.78, 0.62 }, { -1.0, 0.4 }, { -1.0, -0.4 }, { -0.78, -0.62 } };
        stepCut(m, o, { { 0.62, -0.3 }, { 0.82, -0.2 }, { 0.94, -0.1 }, { 1.0, -0.04 }, { 1.0, 0.04 },
                        { 0.74, 0.28 }, { 0.46, 0.5 }, { 0.2, 0.68 } });
        break;
    }
    case kPrincessCut: {
        auto o = polygonOutline(4, 1.0, 1.0, kPi / 4);
        stepCut(m, o, { { 0.66, -0.3 }, { 0.9, -0.12 }, { 1.0, -0.04 }, { 1.0, 0.04 }, { 0.62, 0.4 }, { 0.25, 0.7 } });
        break;
    }
    case kCrystalCluster: crystalCluster(m, sp); parity = false; break;
    case kTetrahedron: convexHull(m, { { 1, 1, 1 }, { 1, -1, -1 }, { -1, 1, -1 }, { -1, -1, 1 } }); break;
    case kBox: {
        const double hz = 0.6 * std::max(0.1, sp.thickness / 0.3);
        std::vector<V3> p;
        for (int i = 0; i < 8; ++i) p.push_back({ (i & 1) ? 0.6 : -0.6, (i & 2) ? 0.6 : -0.6, (i & 4) ? hz : -hz });
        convexHull(m, p);
        break;
    }
    case kOctahedron: convexHull(m, { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } }); break;
    case kDodecahedron: dodecahedron(m); break;
    case kIcosahedron: convexHull(m, icoPoints()); break;
    case kGeoSphere: radialSolid(m, sp.detail, [](V3 d) { return d; }, 0); break;
    case kBall: m.smoothAngleDeg = 180.0f; radialSolid(m, 4, [](V3 d) { return d; }, 1); break;
    case kPyramid:
        convexHull(m, { { 0, 0, -0.6 }, { -0.75, -0.75, 0.4 }, { 0.75, -0.75, 0.4 }, { 0.75, 0.75, 0.4 }, { -0.75, 0.75, 0.4 } });
        break;
    case kHexPrism: {
        const double hz = 0.5 * std::max(0.1, sp.thickness / 0.3);
        std::vector<V3> p;
        for (int k = 0; k < 6; ++k) {
            const double a = kPi / 3 * k;
            p.push_back({ std::cos(a), std::sin(a), -hz });
            p.push_back({ std::cos(a), std::sin(a), hz });
        }
        convexHull(m, p);
        break;
    }
    case kTorus: m.smoothAngleDeg = 180.0f; torus(m); break;
    case kWavyGlass: wavyGlass(m, sp); break;
    case kFlutedGlass: flutedGlass(m, sp); break;
    case kBubbleGlass: bumpyPane(m, sp, false); break;
    case kWaterDrops: bumpyPane(m, sp, true); break;
    case kIceCube: {
        const int seed = sp.seed;
        m.smoothAngleDeg = 180.0f;
        radialSolid(m, 4, [seed](V3 d) {
            const double n = 8.0;
            const double k = std::pow(std::pow(std::fabs(d.x), n) + std::pow(std::fabs(d.y), n) + std::pow(std::fabs(d.z), n), 1.0 / n);
            return d * ((0.72 / k) * (1.0 + 0.035 * fbm3(d.x * 3, d.y * 3, d.z * 3, seed)));
        }, 1);
        break;
    }
    case kMeltedBlob: {
        const int seed = sp.seed;
        m.smoothAngleDeg = 180.0f;
        radialSolid(m, 4, [seed](V3 d) { return d * (0.85 * (1.0 + 0.28 * fbm3(d.x * 1.7, d.y * 1.7, d.z * 1.7, seed))); }, 1);
        break;
    }
    case kShards: shards(m, sp); parity = false; break;
    case kTwistedBar: m.smoothAngleDeg = 50.0f; twistedBar(m, sp); break;
    case kBentSlab: bentSlab(m, sp); break;
    case kCustomOBJ:
        m.smoothAngleDeg = sp.smoothAngleDeg;
        if (sp.objPath.empty()) { if (err) *err = "no OBJ file chosen"; }
        else loadOBJ(sp.objPath, m, err);
        parity = false;
        break;
    default: gemCut(m, kGem); break;
    }
    // Drop degenerate triangles.
    Mesh out;
    out.v = m.v;
    out.smoothAngleDeg = m.smoothAngleDeg;
    out.trustWinding = m.trustWinding;
    for (size_t t = 0; t < m.f.size() / 3; ++t) {
        V3 a = vert(m, m.f[3 * t]), b = vert(m, m.f[3 * t + 1]), c = vert(m, m.f[3 * t + 2]);
        if (len(cross(b - a, c - a)) > 1e-12) out.tri(m.f[3 * t], m.f[3 * t + 1], m.f[3 * t + 2], m.group[t]);
    }
    if (parity && !out.trustWinding && out.f.size() / 3 <= 6000) orientByParity(out);
    return out;
}

}  // namespace fg3d
