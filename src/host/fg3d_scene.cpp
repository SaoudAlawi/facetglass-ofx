// FacetGlass 3D - host side scene building. See fg3d_scene.h.
// SPDX-License-Identifier: MIT
#include "fg3d_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <unordered_map>

namespace fg3d {

namespace {

const double kPi = 3.14159265358979323846;

struct V3 {
    double x, y, z;
};
V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
V3 operator*(V3 a, double s) { return { a.x * s, a.y * s, a.z * s }; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
double len(V3 a) { return std::sqrt(dot(a, a)); }
V3 norm(V3 a)
{
    double l = len(a);
    return l > 1e-30 ? a * (1.0 / l) : V3{ 0, 0, 1 };
}

V3 vert(const Mesh& m, int i) { return { m.v[3 * i], m.v[3 * i + 1], m.v[3 * i + 2] }; }

// Deterministic RNG for shape variations.
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

// Flip triangles [first, end) so they face away from `center` (convex pieces).
void orientConvex(Mesh& m, size_t firstTri, V3 center)
{
    for (size_t t = firstTri; t < m.f.size() / 3; ++t) {
        int* f = &m.f[3 * t];
        V3 a = vert(m, f[0]), b = vert(m, f[1]), c = vert(m, f[2]);
        V3 n = cross(b - a, c - a);
        V3 mid = (a + b + c) * (1.0 / 3.0);
        if (dot(n, mid - center) < 0) std::swap(f[1], f[2]);
    }
}

// Ray-parity orientation for a closed (possibly non-convex) mesh: a ray from a
// face along its normal must leave the solid, i.e. cross the surface an even
// number of times.
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
        // Slightly off-centre origin avoids hitting shared edges exactly.
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

// ---------------------------------------------------------------- shapes

// Slab extruded along y: flat front at z = zf, back profile (x, z) with x
// non-decreasing (equal x = a vertical riser), from x0 to x1.
void slabProfile(Mesh& m, const std::vector<std::pair<double, double>>& back, double zf, double halfLen)
{
    const double L = halfLen;
    std::vector<int> fb, bb, ft, bt;  // front/back vertices at y = -L (b) and y = +L (t)
    for (auto& pt : back) {
        fb.push_back(m.addV((float)pt.first, (float)-L, (float)zf));
        bb.push_back(m.addV((float)pt.first, (float)-L, (float)pt.second));
        ft.push_back(m.addV((float)pt.first, (float)L, (float)zf));
        bt.push_back(m.addV((float)pt.first, (float)L, (float)pt.second));
    }
    const size_t n = back.size();
    for (size_t i = 0; i + 1 < n; ++i) {
        // Back face strip (also covers risers).
        m.quad(bb[i], bb[i + 1], bt[i + 1], bt[i]);
        if (back[i + 1].first > back[i].first) {
            m.quad(fb[i], ft[i], ft[i + 1], fb[i + 1]);                       // front
            m.quad(fb[i], fb[i + 1], bb[i + 1], bb[i]);                       // cap y = -L
            m.quad(ft[i], bt[i], bt[i + 1], ft[i + 1]);                       // cap y = +L
        }
    }
    m.quad(fb[0], bb[0], bt[0], ft[0]);                                       // side x0
    m.quad(fb[n - 1], ft[n - 1], bt[n - 1], bb[n - 1]);                       // side x1
}

void facetPlate(Mesh& m, const ShapeParams& sp)
{
    const int N = std::max(1, std::min(sp.facets, 32));
    const double t = std::max(0.02, (double)sp.thickness);
    const double zf = -t * 0.5, z0 = t * 0.5;
    const double a = sp.prismAngleDeg * kPi / 180.0;
    std::vector<std::pair<double, double>> back;
    auto slope = [&](int k) { return std::tan(std::min(a * (k + 1), 1.3)); };
    if (!sp.bothSides) {
        for (int k = 0; k < N; ++k) {
            double xa = -1.0 + 2.0 * k / N, xb = -1.0 + 2.0 * (k + 1) / N;
            back.push_back({ xa, z0 });
            back.push_back({ xb, z0 + slope(k) * (xb - xa) });
        }
    } else {
        // N facets on each side of the centre line, steeper outward.
        const double w = 1.0 / N;
        for (int k = N - 1; k >= 0; --k) {   // left side, outermost first
            double xa = -(k + 1) * w, xb = -k * w;
            back.push_back({ xa, z0 + slope(k) * w });
            back.push_back({ xb, z0 });
        }
        for (int k = 0; k < N; ++k) {
            double xa = k * w, xb = (k + 1) * w;
            back.push_back({ xa, z0 });
            back.push_back({ xb, z0 + slope(k) * w });
        }
    }
    slabProfile(m, back, zf, 4.0);
}

void extrudeY(Mesh& m, const std::vector<std::pair<double, double>>& poly, double halfLen, int sideGroup)
{
    // Convex polygon in x-z, extruded along y; caps as fans.
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

void radialPrism(Mesh& m, const ShapeParams& sp)
{
    const int N = std::max(3, std::min(sp.facets, 64));
    const double t = std::max(0.02, (double)sp.thickness);
    const double r0 = std::min(std::max((double)sp.clearCentre, 0.01), 0.95);
    const double a = std::min(sp.prismAngleDeg * kPi / 180.0, 1.3);
    const double zf = -t * 0.5, zi = t * 0.5, zo = zi + std::tan(a) * (1.0 - r0);
    int cf = m.addV(0, 0, (float)zf), cb = m.addV(0, 0, (float)zi);
    std::vector<int> fo(N), bi(N), bo(N);
    for (int k = 0; k < N; ++k) {
        double th = 2 * kPi * (k + 0.5) / N;
        double c = std::cos(th), s = std::sin(th);
        fo[k] = m.addV((float)c, (float)s, (float)zf);
        bi[k] = m.addV((float)(r0 * c), (float)(r0 * s), (float)zi);
        bo[k] = m.addV((float)c, (float)s, (float)zo);
    }
    for (int k = 0; k < N; ++k) {
        int j = (k + 1) % N;
        m.tri(cf, fo[j], fo[k]);              // front
        m.tri(cb, bi[k], bi[j]);              // clear centre (back)
        m.quad(bi[k], bo[k], bo[j], bi[j]);   // conical facet
        m.quad(fo[k], fo[j], bo[j], bo[k]);   // side wall
    }
}

// Plano-convex/concave lens over an angular range; back surface smooth (group 1).
void lens(Mesh& m, const ShapeParams& sp, double th0, double th1, bool cutWall)
{
    const int R = 24, S = cutWall ? 48 : 96;
    // Diopter Power is the magnification of the footage seen through the lens
    // centre: P >= 0 gives 1 + 0.1 P, P < 0 gives 1 / (1 - 0.1 P). For a thin lens
    // at depth z between the camera and the footage (at 1) the magnification is
    // 1 / (1 - z (1 - z) / f); then 1/f = (n - 1) c with n = 1.5 nominal.
    const double z = std::min(std::max((double)sp.depth, 0.01), 0.95);
    const double mag = sp.power >= 0 ? 1.0 + 0.1 * sp.power : 1.0 / (1.0 - 0.1 * sp.power);
    const double cWorld = (1.0 - 1.0 / mag) / (z * (1.0 - z)) / 0.5;
    const double c = std::max(-1.6, std::min(1.6, cWorld * std::max(1e-4, (double)sp.worldScale)));
    const double sag = 0.5 * c;                         // centre minus edge thickness
    const double base = 0.06 + std::max(0.0, -sag) + 0.1 * std::max(0.0, (double)sp.thickness - 0.3);
    const double zf = -0.05;
    auto zb = [&](double r) { return zf + base + sag * (1.0 - r * r); };
    const bool full = !cutWall;
    const int cols = full ? S : S + 1;
    int cf = m.addV(0, 0, (float)zf), cb = m.addV(0, 0, (float)zb(0));
    std::vector<int> back((size_t)R * cols), frontRim(cols);
    for (int j = 0; j < cols; ++j) {
        double th = th0 + (th1 - th0) * j / S;
        double cs = std::cos(th), sn = std::sin(th);
        for (int i = 1; i <= R; ++i) {
            double r = (double)i / R;
            back[(size_t)(i - 1) * cols + j] = m.addV((float)(r * cs), (float)(r * sn), (float)zb(r));
        }
        frontRim[j] = m.addV((float)cs, (float)sn, (float)zf);
    }
    for (int j = 0; j < S; ++j) {
        int jn = full ? (j + 1) % S : j + 1;
        m.tri(cf, frontRim[jn], frontRim[j]);                                            // front
        m.tri(cb, back[jn], back[j], 1);                                                 // back centre
        for (int i = 1; i < R; ++i)
            m.quad(back[(size_t)(i - 1) * cols + j], back[(size_t)i * cols + j], back[(size_t)i * cols + jn],
                   back[(size_t)(i - 1) * cols + jn], 1);
        m.quad(frontRim[j], frontRim[jn], back[(size_t)(R - 1) * cols + jn], back[(size_t)(R - 1) * cols + j]);  // rim
    }
    if (cutWall) {
        // Flat wall along the cut (both radial edges meet at the centre line).
        for (int side = 0; side < 2; ++side) {
            int j = side == 0 ? 0 : S;
            int prevB = cb, prevF = cf;
            for (int i = 1; i <= R; ++i) {
                double th = th0 + (th1 - th0) * j / S;
                double r = (double)i / R;
                int f = m.addV((float)(r * std::cos(th)), (float)(r * std::sin(th)), (float)zf);
                int b = back[(size_t)(i - 1) * cols + j];
                m.quad(prevF, f, b, prevB);
                prevF = f;
                prevB = b;
            }
        }
    }
}

void geoSphere(Mesh& m, int detail)
{
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<V3> v = { { -1, t, 0 }, { 1, t, 0 }, { -1, -t, 0 }, { 1, -t, 0 }, { 0, -1, t }, { 0, 1, t },
                          { 0, -1, -t }, { 0, 1, -t }, { t, 0, -1 }, { t, 0, 1 }, { -t, 0, -1 }, { -t, 0, 1 } };
    std::vector<int> f = { 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6,
                           7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 };
    for (auto& p : v) p = norm(p);
    for (int d = 0; d < std::min(std::max(detail, 0), 4); ++d) {
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
            int tris[12] = { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca };
            nf.insert(nf.end(), tris, tris + 12);
        }
        f.swap(nf);
    }
    int base = (int)(m.v.size() / 3);
    for (auto& p : v) m.addV((float)p.x, (float)p.y, (float)p.z);
    for (size_t i = 0; i < f.size(); i += 3) m.tri(base + f[i], base + f[i + 1], base + f[i + 2]);
}

void gem(Mesh& m)
{
    const int n = 16;
    const double zt = -0.38, zc = -0.22, zg0 = -0.06, zg1 = 0.04, zp = 0.9;
    int ct = m.addV(0, 0, (float)zt), cul = m.addV(0, 0, (float)zp);
    std::vector<int> tab(n), crown(n), g0(n), g1(n);
    for (int k = 0; k < n; ++k) {
        double a = 2 * kPi * k / n, ah = 2 * kPi * (k + 0.5) / n;
        tab[k] = m.addV((float)(0.52 * std::cos(a)), (float)(0.52 * std::sin(a)), (float)zt);
        crown[k] = m.addV((float)(0.8 * std::cos(ah)), (float)(0.8 * std::sin(ah)), (float)zc);
        g0[k] = m.addV((float)std::cos(a), (float)std::sin(a), (float)zg0);
        g1[k] = m.addV((float)std::cos(a), (float)std::sin(a), (float)zg1);
    }
    for (int k = 0; k < n; ++k) {
        int j = (k + 1) % n;
        m.tri(ct, tab[k], tab[j]);
        m.tri(tab[k], crown[k], tab[j]);          // star facets
        m.tri(tab[j], crown[k], crown[j]);
        m.tri(crown[k], g0[j], crown[j]);         // upper girdle facets
        m.tri(crown[k], g0[k], g0[j]);
        m.quad(g0[k], g1[k], g1[j], g0[j]);       // girdle
        m.tri(g1[k], cul, g1[j]);                 // pavilion
    }
}

void crystal(Mesh& m, V3 base, V3 axis, double radius, double length, double twist)
{
    // Hexagonal prism with a pointed tip, from `base` along `axis`.
    axis = norm(axis);
    V3 ref = std::fabs(axis.z) < 0.9 ? V3{ 0, 0, 1 } : V3{ 1, 0, 0 };
    V3 u = norm(cross(axis, ref)), w = cross(axis, u);
    size_t first = m.f.size() / 3;
    const double body = length * 0.75;
    std::vector<int> lo(6), hi(6);
    for (int k = 0; k < 6; ++k) {
        double a = twist + kPi * k / 3.0;
        V3 d = u * std::cos(a) + w * std::sin(a);
        V3 p0 = base + d * radius, p1 = base + d * radius + axis * body;
        lo[k] = m.addV((float)p0.x, (float)p0.y, (float)p0.z);
        hi[k] = m.addV((float)p1.x, (float)p1.y, (float)p1.z);
    }
    V3 tip = base + axis * length, bot = base - axis * (radius * 0.5);
    int ti = m.addV((float)tip.x, (float)tip.y, (float)tip.z), bi = m.addV((float)bot.x, (float)bot.y, (float)bot.z);
    for (int k = 0; k < 6; ++k) {
        int j = (k + 1) % 6;
        m.quad(lo[k], lo[j], hi[j], hi[k]);
        m.tri(hi[k], hi[j], ti);
        m.tri(lo[j], lo[k], bi);
    }
    orientConvex(m, first, base + axis * (length * 0.45));
}

void crystalCluster(Mesh& m, const ShapeParams& sp)
{
    Rng rng((uint32_t)sp.seed * 7919u + 13u);
    const int K = std::max(1, std::min(sp.facets, 12));
    for (int i = 0; i < K; ++i) {
        double th = rng.range(0, 2 * kPi), ph = rng.range(0.15, 1.25);
        V3 axis = { std::cos(th) * std::sin(ph), std::sin(th) * std::sin(ph), -std::cos(ph) * 0.6 };
        double len = rng.range(0.55, 1.0);
        double rad = rng.range(0.12, 0.22);
        V3 base = { rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), rng.range(-0.05, 0.1) };
        crystal(m, base, axis, rad, len, rng.range(0, kPi));
    }
}

void box(Mesh& m, double hx, double hy, double hz)
{
    int v[8];
    for (int i = 0; i < 8; ++i)
        v[i] = m.addV((float)((i & 1) ? hx : -hx), (float)((i & 2) ? hy : -hy), (float)((i & 4) ? hz : -hz));
    size_t first = m.f.size() / 3;
    m.quad(v[0], v[2], v[3], v[1]); m.quad(v[4], v[5], v[7], v[6]);
    m.quad(v[0], v[1], v[5], v[4]); m.quad(v[2], v[6], v[7], v[3]);
    m.quad(v[0], v[4], v[6], v[2]); m.quad(v[1], v[3], v[7], v[5]);
    orientConvex(m, first, { 0, 0, 0 });
}

void pyramid(Mesh& m)
{
    // Square pyramid, apex toward the camera.
    int apex = m.addV(0, 0, -0.6f);
    int b[4] = { m.addV(-0.75f, -0.75f, 0.4f), m.addV(0.75f, -0.75f, 0.4f), m.addV(0.75f, 0.75f, 0.4f),
                 m.addV(-0.75f, 0.75f, 0.4f) };
    size_t first = m.f.size() / 3;
    for (int k = 0; k < 4; ++k) m.tri(apex, b[k], b[(k + 1) % 4]);
    m.quad(b[0], b[3], b[2], b[1]);
    orientConvex(m, first, { 0, 0, 0.15 });
}

void rod(Mesh& m)
{
    std::vector<std::pair<double, double>> poly;
    const int S = 64;
    for (int k = 0; k < S; ++k) {
        double a = 2 * kPi * k / S;
        poly.push_back({ 0.5 * std::cos(a), 0.5 * std::sin(a) });
    }
    extrudeY(m, poly, 4.0, 1);
}

}  // namespace

const char* shapeName(int shape)
{
    static const char* names[kShapeCount] = { "Faceted Plate", "Wedge Prism", "Radial Prism", "Split Diopter",
                                              "Round Diopter", "Geodesic Sphere", "Gem", "Crystal Cluster",
                                              "Box", "Pyramid", "Rod" };
    return shape >= 0 && shape < kShapeCount ? names[shape] : "?";
}

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

Mesh makeShape(const ShapeParams& sp)
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
    case kRadialPrism: radialPrism(m, sp); break;
    case kSplitDiopter: lens(m, sp, -kPi / 2, kPi / 2, true); break;
    case kRoundDiopter: lens(m, sp, 0, 2 * kPi, false); break;
    case kGeoSphere: geoSphere(m, sp.detail); break;
    case kGem: gem(m); break;
    case kCrystalCluster: crystalCluster(m, sp); parity = false; break;
    case kBox: box(m, 0.7, 0.7, 0.7 * std::max(0.1, sp.thickness / 0.3)); break;
    case kPyramid: pyramid(m); break;
    case kRod: rod(m); break;
    default: facetPlate(m, sp); break;
    }
    // Drop degenerate triangles.
    Mesh out;
    out.v = m.v;
    for (size_t t = 0; t < m.f.size() / 3; ++t) {
        V3 a = vert(m, m.f[3 * t]), b = vert(m, m.f[3 * t + 1]), c = vert(m, m.f[3 * t + 2]);
        if (len(cross(b - a, c - a)) > 1e-12) out.tri(m.f[3 * t], m.f[3 * t + 1], m.f[3 * t + 2], m.group[t]);
    }
    if (parity) orientByParity(out);
    return out;
}

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

    // Smooth normals within groups: per (welded vertex, group) area-weighted sum.
    std::map<std::pair<int, int>, V3> vsum;
    for (size_t t = 0; t < nt; ++t) {
        if (mesh.group[t] == 0) continue;
        for (int k = 0; k < 3; ++k) {
            auto key = std::make_pair(weld[mesh.f[3 * t + k]], mesh.group[t]);
            auto it = vsum.find(key);
            vsum[key] = (it == vsum.end() ? V3{ 0, 0, 0 } : it->second) + fnArea[t];
        }
    }
    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k)
            T[t].n[k] = mesh.group[t] == 0 ? T[t].fn
                                           : norm(vsum[std::make_pair(weld[mesh.f[3 * t + k]], mesh.group[t])]);

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
            if (mesh.group[t] != 0 && mesh.group[t] == mesh.group[o]) continue;   // smooth edge
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
    p.spectral = ui.dispersion > 0 ? 1 : 0;
    p.reflect = (float)std::max(ui.reflection, 0.0);
    p.roughness = (float)std::max(ui.roughness, 0.0);
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
}

}  // namespace fg3d
