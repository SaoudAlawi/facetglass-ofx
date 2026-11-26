/* FacetGlass 3D core - single-source path tracer for solid glass objects.
 *
 * Compiled unchanged as C++ (CPU), CUDA, OpenCL C and Metal. A backend prelude
 * defines, before including this file:
 *   FG_DEVICE  function qualifier        FG_GLOBAL  buffer address space
 *   FG_PRIV    private pointer space     FG_PCONST  parameter block space
 *                                         FG_SQRT FG_SIN FG_COS FG_ATAN2 FG_FLOOR
 *   FG_FABS FG_FMAX FG_FMIN               float math (exp/log/sin/cos are portable
 *                                         implementations below, for bit-identical backends)
 *
 * Model. A pinhole-plus-aperture camera at the origin looks down +z. The footage
 * is a plane at z = 1 that exactly fills the frame (screen y in [-0.5, 0.5], x
 * scaled by aspect). A solid glass object (triangle mesh with a BVH) sits between
 * them. Every sample is a ray from a point on the lens aperture through the focus
 * plane, with its own wavelength. It refracts into the glass (Snell), bounces
 * inside with Fresnel reflection and total internal reflection, refracts out, and
 * finally lands on the footage, the off-screen world, or a virtual light.
 *
 * Pipeline per frame (each backend runs the three passes):
 *   1. decode: source pixels -> linear light, level 0 of a mip pyramid
 *   2. mip:    box-filter each level from the previous one (world blur)
 *   3. shade:  fg3d_shade per output pixel (returns encoded RGBA)
 *
 * Kernel rules: float only (f suffix), scalar math and structs, no recursion, no
 * global/constant arrays, no textures, no identifiers that are keywords in some
 * shading language (in, sample, half, ...).
 */
#ifndef FG3D_CORE_H
#define FG3D_CORE_H

#include "fg3d_params.h"

typedef struct fgv3 { float x, y, z; } fgv3;
typedef struct fgv4 { float x, y, z, w; } fgv4;

FG_DEVICE fgv3 fgv(float x, float y, float z) { fgv3 r; r.x = x; r.y = y; r.z = z; return r; }
FG_DEVICE fgv4 fgv4m(float x, float y, float z, float w) { fgv4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
FG_DEVICE fgv3 fgv_add(fgv3 a, fgv3 b) { return fgv(a.x + b.x, a.y + b.y, a.z + b.z); }
FG_DEVICE fgv3 fgv_sub(fgv3 a, fgv3 b) { return fgv(a.x - b.x, a.y - b.y, a.z - b.z); }
FG_DEVICE fgv3 fgv_mul(fgv3 a, float s) { return fgv(a.x * s, a.y * s, a.z * s); }
FG_DEVICE fgv3 fgv_madd(fgv3 a, fgv3 b, float s) { return fgv(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s); }
FG_DEVICE float fgv_dot(fgv3 a, fgv3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
FG_DEVICE fgv3 fgv_cross(fgv3 a, fgv3 b)
{
  return fgv(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
FG_DEVICE fgv3 fgv_norm(fgv3 a)
{
  float l = FG_SQRT(fgv_dot(a, a));
  return l > 1e-20f ? fgv_mul(a, 1.0f / l) : fgv(0.0f, 0.0f, 1.0f);
}
FG_DEVICE fgv3 fgv_load(FG_GLOBAL const float* p) { return fgv(p[0], p[1], p[2]); }
FG_DEVICE float fg3d_clampf(float v, float lo, float hi) { return FG_FMAX(lo, FG_FMIN(hi, v)); }
FG_DEVICE int fg3d_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
FG_DEVICE float fg3d_fract(float v) { return v - FG_FLOOR(v); }
/* Portable exp / log / sin / cos built only from +, -, *, / and floor, so every
 * backend (CPU, CUDA, OpenCL, Metal) computes bit-identical results. Library
 * versions differ in the last bits between platforms, which in a path tracer can
 * flip a random reflect/refract choice and change a whole sample. */
FG_DEVICE float fg3d_exp2i(int n)
{
  /* 2^n by binary powering (exact for normal results). */
  float b = n < 0 ? 0.5f : 2.0f;
  int k = n < 0 ? -n : n;
  float r = 1.0f;
  if (k > 140) k = 140;
  while (k > 0) {
    if ((k & 1) != 0) r *= b;
    b *= b;
    k >>= 1;
  }
  return r;
}

FG_DEVICE float fg3d_exp(float x)
{
  if (x > 88.0f) x = 88.0f;
  if (x < -87.0f) return 0.0f;
  float n = FG_FLOOR(x * 1.44269504f + 0.5f);
  float r = (x - n * 0.693145752f) - n * 1.42860677e-6f;  /* ln2 split hi/lo */
  float pr = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f + r * (0.00833333377f +
             r * 0.00138888892f)))));
  return pr * fg3d_exp2i((int)n);
}

FG_DEVICE float fg3d_log(float x)
{
  if (x <= 0.0f) return -87.0f;
  int e = 0;
  float m = x;
  while (m >= 65536.0f) { m *= (1.0f / 65536.0f); e += 16; }
  while (m < (1.0f / 65536.0f)) { m *= 65536.0f; e -= 16; }
  while (m >= 1.41421356f) { m *= 0.5f; e += 1; }
  while (m < 0.707106781f) { m *= 2.0f; e -= 1; }
  float t = (m - 1.0f) / (m + 1.0f);
  float t2 = t * t;
  float sr = t * (2.0f + t2 * (0.666666667f + t2 * (0.4f + t2 * (0.285714286f + t2 * (0.222222222f +
             t2 * 0.181818182f)))));
  return (float)e * 0.693147181f + sr;
}

/* sin and cos for any x (range-reduced to [-pi/4, pi/4] around a quadrant). */
FG_DEVICE float fg3d_sin_poly(float x)
{
  float x2 = x * x;
  return x * (1.0f + x2 * (-0.166666667f + x2 * (0.00833333333f + x2 * (-0.000198412698f + x2 * 2.75573192e-6f))));
}
FG_DEVICE float fg3d_cos_poly(float x)
{
  float x2 = x * x;
  return 1.0f + x2 * (-0.5f + x2 * (0.0416666667f + x2 * (-0.00138888889f + x2 * (2.48015873e-5f +
         x2 * -2.75573192e-7f))));
}
FG_DEVICE void fg3d_sincos(float x, FG_PRIV float* sn, FG_PRIV float* cs)
{
  float q = FG_FLOOR(x * 0.636619772f + 0.5f);          /* nearest multiple of pi/2 */
  float r = (x - q * 1.57079601f) - q * 3.13916473e-7f;  /* pi/2 split hi/lo */
  int k = ((int)q) & 3;
  float s0 = fg3d_sin_poly(r), c0 = fg3d_cos_poly(r);
  if (k == 0) { *sn = s0; *cs = c0; }
  else if (k == 1) { *sn = c0; *cs = -s0; }
  else if (k == 2) { *sn = -s0; *cs = -c0; }
  else { *sn = -c0; *cs = s0; }
}

FG_DEVICE float fg3d_pow(float b, float e) { return b > 0.0f ? fg3d_exp(e * fg3d_log(b)) : 0.0f; }

/* ------------------------------------------------------------------ random */

FG_DEVICE unsigned int fg3d_hash(unsigned int x)
{
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
FG_DEVICE float fg3d_rand(unsigned int x) { return (float)(fg3d_hash(x) >> 8) * (1.0f / 16777216.0f); }

/* ------------------------------------------------------------------ colour */

FG_DEVICE float fg3d_decode(int cs, float x)
{
  if (cs == FG_CS_REC709) return x > 0.0f ? fg3d_pow(x, 2.4f) : x;
  if (cs == FG_CS_SRGB) return x <= 0.04045f ? x / 12.92f : fg3d_pow((x + 0.055f) / 1.055f, 2.4f);
  if (cs == FG_CS_SLOG3) {
    if (x >= 171.2102946929f / 1023.0f) return fg3d_exp(((x * 1023.0f - 420.0f) / 261.5f) * 2.302585093f) * 0.19f - 0.01f;
    return (x * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f);
  }
  if (cs == FG_CS_LOGC3) {
    if (x > 0.149658f) return (fg3d_exp(((x - 0.385537f) / 0.247190f) * 2.302585093f) - 0.052272f) / 5.555556f;
    return (x - 0.092809f) / 5.367655f;
  }
  if (cs == FG_CS_LOGC4) {
    /* a = (2^18 - 16) / 117.45, b = (1023 - 95) / 1023, c = 95 / 1023 */
    float a = 2231.826309f, b = 0.9071358749f, c = 0.0928641251f;
    float s = 0.1135972086f, t = -0.0180569961f;
    if (x >= 0.0f) return (fg3d_exp((14.0f * (x - c) / b + 6.0f) * 0.6931471806f) - 64.0f) / a;
    return x * s + t;
  }
  if (cs == FG_CS_VLOG) {
    if (x < 0.181f) return (x - 0.125f) / 5.6f;
    return fg3d_exp(((x - 0.598206f) / 0.241514f) * 2.302585093f) - 0.00873f;
  }
  if (cs == FG_CS_DAVINCI) {
    if (x <= 0.02740668f) return x / 10.44426855f;
    return fg3d_exp((x / 0.07329248f - 7.0f) * 0.6931471806f) - 0.0075f;
  }
  return x;
}

FG_DEVICE float fg3d_encode(int cs, float y)
{
  if (cs == FG_CS_REC709) return y > 0.0f ? fg3d_pow(y, 1.0f / 2.4f) : y;
  if (cs == FG_CS_SRGB) return y <= 0.0031308f ? y * 12.92f : 1.055f * fg3d_pow(y, 1.0f / 2.4f) - 0.055f;
  if (cs == FG_CS_SLOG3) {
    if (y >= 0.01125000f) return (420.0f + fg3d_log((y + 0.01f) / 0.19f) / 2.302585093f * 261.5f) / 1023.0f;
    return (y * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f;
  }
  if (cs == FG_CS_LOGC3) {
    if (y > 0.010591f) return 0.247190f * fg3d_log(5.555556f * y + 0.052272f) / 2.302585093f + 0.385537f;
    return 5.367655f * y + 0.092809f;
  }
  if (cs == FG_CS_LOGC4) {
    float a = 2231.826309f, b = 0.9071358749f, c = 0.0928641251f;
    float s = 0.1135972086f, t = -0.0180569961f;
    if (y >= t) {
      float v = a * y + 64.0f;
      return v > 0.0f ? (fg3d_log(v) / 0.6931471806f - 6.0f) / 14.0f * b + c : c;
    }
    return (y - t) / s;
  }
  if (cs == FG_CS_VLOG) {
    if (y < 0.01f) return 5.6f * y + 0.125f;
    return 0.241514f * fg3d_log(y + 0.00873f) / 2.302585093f + 0.598206f;
  }
  if (cs == FG_CS_DAVINCI) {
    if (y <= 0.00262409f) return y * 10.44426855f;
    return (fg3d_log(y + 0.0075f) / 0.6931471806f + 7.0f) * 0.07329248f;
  }
  return y;
}

/* Pass 1: decode one source pixel into level 0 (alpha passes through). */
FG_DEVICE fgv4 fg3d_decode_pixel(int cs, FG_GLOBAL const float* px)
{
  return fgv4m(fg3d_decode(cs, px[0]), fg3d_decode(cs, px[1]), fg3d_decode(cs, px[2]), px[3]);
}

/* Pass 2: one pixel of mip level l + 1 from level l (w, h = size of level l). */
FG_DEVICE fgv4 fg3d_mip_pixel(FG_GLOBAL const float* lvl, int w, int h, int x, int y)
{
  int x0 = fg3d_clampi(2 * x, 0, w - 1), x1 = fg3d_clampi(2 * x + 1, 0, w - 1);
  int y0 = fg3d_clampi(2 * y, 0, h - 1), y1 = fg3d_clampi(2 * y + 1, 0, h - 1);
  FG_GLOBAL const float* a = lvl + (y0 * w + x0) * 4;
  FG_GLOBAL const float* b = lvl + (y0 * w + x1) * 4;
  FG_GLOBAL const float* c = lvl + (y1 * w + x0) * 4;
  FG_GLOBAL const float* d = lvl + (y1 * w + x1) * 4;
  return fgv4m(0.25f * (a[0] + b[0] + c[0] + d[0]), 0.25f * (a[1] + b[1] + c[1] + d[1]),
               0.25f * (a[2] + b[2] + c[2] + d[2]), 0.25f * (a[3] + b[3] + c[3] + d[3]));
}

FG_DEVICE int fg3d_level_w(int w0, int l) { int w = w0 >> l; return w < 1 ? 1 : w; }
FG_DEVICE int fg3d_level_offset(int w0, int h0, int l)
{
  int off = 0;
  for (int i = 0; i < l; i++) off += fg3d_level_w(w0, i) * fg3d_level_w(h0, i) * 4;
  return off;
}

/* Bilinear fetch from one pyramid level at normalised (u, v), clamped. */
FG_DEVICE fgv4 fg3d_fetch(FG_GLOBAL const float* pyr, int w0, int h0, int l, float u, float v)
{
  int w = fg3d_level_w(w0, l), h = fg3d_level_w(h0, l);
  FG_GLOBAL const float* img = pyr + fg3d_level_offset(w0, h0, l);
  float fx = u * (float)w - 0.5f, fy = v * (float)h - 0.5f;
  float flx = FG_FLOOR(fx), fly = FG_FLOOR(fy);
  float ax = fx - flx, ay = fy - fly;
  int x0 = fg3d_clampi((int)flx, 0, w - 1), x1 = fg3d_clampi((int)flx + 1, 0, w - 1);
  int y0 = fg3d_clampi((int)fly, 0, h - 1), y1 = fg3d_clampi((int)fly + 1, 0, h - 1);
  FG_GLOBAL const float* p00 = img + (y0 * w + x0) * 4;
  FG_GLOBAL const float* p10 = img + (y0 * w + x1) * 4;
  FG_GLOBAL const float* p01 = img + (y1 * w + x0) * 4;
  FG_GLOBAL const float* p11 = img + (y1 * w + x1) * 4;
  float w00 = (1.0f - ax) * (1.0f - ay), w10 = ax * (1.0f - ay), w01 = (1.0f - ax) * ay, w11 = ax * ay;
  return fgv4m(p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
               p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
               p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
               p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11);
}

/* Trilinear: lod in mip levels (0 = sharp). */
FG_DEVICE fgv4 fg3d_lookup(FG_GLOBAL const float* pyr, int w0, int h0, int levels, float u, float v, float lod)
{
  lod = fg3d_clampf(lod, 0.0f, (float)(levels - 1));
  int l0 = (int)FG_FLOOR(lod);
  float t = lod - (float)l0;
  fgv4 a = fg3d_fetch(pyr, w0, h0, l0, u, v);
  if (t <= 0.0f || l0 + 1 >= levels) return a;
  fgv4 b = fg3d_fetch(pyr, w0, h0, l0 + 1, u, v);
  return fgv4m(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

FG_DEVICE float fg3d_mirror(float u) { float m = u - 2.0f * FG_FLOOR(u * 0.5f); return m > 1.0f ? 2.0f - m : m; }

/* ------------------------------------------------------------------ geometry */

/* Slab test of one BVH node box; returns the entry distance, or 1e30 on a miss. */
FG_DEVICE float fg3d_box(FG_GLOBAL const float* nb, fgv3 ro, float ix, float iy, float iz, float tmin, float tBest)
{
  float t0x = (nb[0] - ro.x) * ix, t1x = (nb[3] - ro.x) * ix;
  float t0y = (nb[1] - ro.y) * iy, t1y = (nb[4] - ro.y) * iy;
  float t0z = (nb[2] - ro.z) * iz, t1z = (nb[5] - ro.z) * iz;
  float tn = FG_FMAX(FG_FMAX(FG_FMIN(t0x, t1x), FG_FMIN(t0y, t1y)), FG_FMIN(t0z, t1z));
  float tf = FG_FMIN(FG_FMIN(FG_FMAX(t0x, t1x), FG_FMAX(t0y, t1y)), FG_FMAX(t0z, t1z));
  if (tf < FG_FMAX(tn, tmin) || tn > tBest) return 1e30f;
  return tn;
}

/* Iterative BVH traversal in object space. Returns triangle index or -1.
   Both child boxes are tested before descending: misses are skipped and the
   nearer child is visited first, so the closest hit is found early and prunes
   the rest (entries farther than the best hit so far are dropped on pop). */
FG_DEVICE int fg3d_intersect(FG_GLOBAL const float* tris, FG_GLOBAL const float* nodesF, FG_GLOBAL const int* nodesI,
                             int numNodes, fgv3 ro, fgv3 rd, float tmin, FG_PRIV float* tHit, FG_PRIV float* uHit,
                             FG_PRIV float* vHit)
{
  int stack[FG_BVH_STACK];
  float stackT[FG_BVH_STACK];
  int sp = 0;
  int best = -1;
  float tBest = 1e30f;
  float ix = 1.0f / (FG_FABS(rd.x) > 1e-12f ? rd.x : 1e-12f);
  float iy = 1.0f / (FG_FABS(rd.y) > 1e-12f ? rd.y : 1e-12f);
  float iz = 1.0f / (FG_FABS(rd.z) > 1e-12f ? rd.z : 1e-12f);
  if (numNodes <= 0) return -1;
  float tr0 = fg3d_box(nodesF, ro, ix, iy, iz, tmin, tBest);
  if (tr0 >= 1e30f) { *tHit = tBest; return -1; }
  stack[sp] = 0; stackT[sp] = tr0; sp++;
  while (sp > 0) {
    --sp;
    int ni = stack[sp];
    if (stackT[sp] > tBest) continue;
    int first = nodesI[ni * FG_NODE_INTS];
    int count = nodesI[ni * FG_NODE_INTS + 1];
    if (count > 0) {
      for (int k = first; k < first + count; k++) {
        FG_GLOBAL const float* tr = tris + k * FG_TRI_FLOATS;
        fgv3 v0 = fgv_load(tr), e1 = fgv_load(tr + 3), e2 = fgv_load(tr + 6);
        fgv3 pv = fgv_cross(rd, e2);
        float det = fgv_dot(e1, pv);
        if (FG_FABS(det) < 1e-14f) continue;
        float inv = 1.0f / det;
        fgv3 tv = fgv_sub(ro, v0);
        float u = fgv_dot(tv, pv) * inv;
        if (u < 0.0f || u > 1.0f) continue;
        fgv3 qv = fgv_cross(tv, e1);
        float v = fgv_dot(rd, qv) * inv;
        if (v < 0.0f || u + v > 1.0f) continue;
        float t = fgv_dot(e2, qv) * inv;
        if (t > tmin && t < tBest) { tBest = t; best = k; *uHit = u; *vHit = v; }
      }
    } else {
      float ta = fg3d_box(nodesF + first * FG_NODE_FLOATS, ro, ix, iy, iz, tmin, tBest);
      float tb = fg3d_box(nodesF + (first + 1) * FG_NODE_FLOATS, ro, ix, iy, iz, tmin, tBest);
      int na = first, nb2 = first + 1;
      if (tb < ta) { float tt = ta; ta = tb; tb = tt; na = first + 1; nb2 = first; }
      /* Push the farther child first so the nearer one is popped next. */
      if (tb < 1e30f && sp < FG_BVH_STACK) { stack[sp] = nb2; stackT[sp] = tb; sp++; }
      if (ta < 1e30f && sp < FG_BVH_STACK) { stack[sp] = na; stackT[sp] = ta; sp++; }
    }
  }
  *tHit = tBest;
  return best;
}

/* Object <-> world (uniform scale): world = pos + scale * (R p). */
FG_DEVICE fgv3 fg3d_rot(FG_PCONST const FG3DParams* p, fgv3 a)
{
  return fgv(p->objRot[0] * a.x + p->objRot[1] * a.y + p->objRot[2] * a.z,
             p->objRot[3] * a.x + p->objRot[4] * a.y + p->objRot[5] * a.z,
             p->objRot[6] * a.x + p->objRot[7] * a.y + p->objRot[8] * a.z);
}
FG_DEVICE fgv3 fg3d_rotT(FG_PCONST const FG3DParams* p, fgv3 a)
{
  return fgv(p->objRot[0] * a.x + p->objRot[3] * a.y + p->objRot[6] * a.z,
             p->objRot[1] * a.x + p->objRot[4] * a.y + p->objRot[7] * a.z,
             p->objRot[2] * a.x + p->objRot[5] * a.y + p->objRot[8] * a.z);
}

/* Shading normal (object space, outward) with bevelled edges. */
FG_DEVICE fgv3 fg3d_shading_normal(FG_GLOBAL const float* tr, float u, float v, float bevelWidth)
{
  float w = 1.0f - u - v;
  fgv3 n = fgv_norm(fgv_add(fgv_add(fgv_mul(fgv_load(tr + 9), w), fgv_mul(fgv_load(tr + 12), u)),
                            fgv_mul(fgv_load(tr + 15), v)));
  if (bevelWidth > 0.0f) {
    /* Closest bevelled edge: edge i is opposite vertex i, barycentric b_i. */
    float best = bevelWidth;
    int edge = -1;
    float b0 = w * tr[30], b1 = u * tr[31], b2 = v * tr[32];
    if (b0 < best && fgv_dot(fgv_load(tr + 21), fgv_load(tr + 21)) > 0.0f) { best = b0; edge = 0; }
    if (b1 < best && fgv_dot(fgv_load(tr + 24), fgv_load(tr + 24)) > 0.0f) { best = b1; edge = 1; }
    if (b2 < best && fgv_dot(fgv_load(tr + 27), fgv_load(tr + 27)) > 0.0f) { best = b2; edge = 2; }
    if (edge >= 0) {
      /* Rounded profile: the normal turns from the face normal to the edge normal. */
      float t = best / bevelWidth;
      t = t * t * (3.0f - 2.0f * t);
      fgv3 ne = fgv_load(tr + 21 + 3 * edge);
      n = fgv_norm(fgv_add(fgv_mul(ne, 1.0f - t), fgv_mul(n, t)));
    }
  }
  return n;
}

FG_DEVICE float fg3d_fresnel(float cosi, float n1, float n2, FG_PRIV float* cost)
{
  float eta = n1 / n2;
  float s2 = eta * eta * (1.0f - cosi * cosi);
  if (s2 >= 1.0f) { *cost = 0.0f; return 1.0f; }
  float ct = FG_SQRT(1.0f - s2);
  *cost = ct;
  float rs = (n1 * cosi - n2 * ct) / (n1 * cosi + n2 * ct);
  float rp = (n1 * ct - n2 * cosi) / (n1 * ct + n2 * cosi);
  return 0.5f * (rs * rs + rp * rp);
}

/* ------------------------------------------------------------------ lights / world */

FG_DEVICE fgv3 fg3d_spectral_weight(float lam)
{
  float r = (lam - 0.600f) / 0.050f, g = (lam - 0.540f) / 0.045f, b = (lam - 0.450f) / 0.040f;
  return fgv(fg3d_exp(-0.5f * r * r), fg3d_exp(-0.5f * g * g), fg3d_exp(-0.5f * b * b));
}

/* Source or refract-source image at normalised frame position. */
FG_DEVICE fgv4 fg3d_image(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                          int useGls, float u, float v, float lod)
{
  if (useGls != 0) return fg3d_lookup(gls, p->glsW, p->glsH, p->glsLevels, u, v, lod);
  return fg3d_lookup(src, p->srcW, p->srcH, p->srcLevels, u, v, lod);
}

/* Where a ray lands once it has left the glass: tap[0] = 1 when it sees an
   image (tap[1] = 1 for the refract source, tap[2..4] = u, v, lod), 0 when it
   sees black; tap[5..7] = virtual light seen along it (glass paths only). */
FG_DEVICE void fg3d_escape_tap(FG_PCONST const FG3DParams* p, fgv3 o, fgv3 d, int glassPath, float aspect, float pixU,
                               float pixV, FG_PRIV float* tap)
{
  int useGls = (glassPath != 0 && p->useSource != 0) ? 1 : 0;
  int mode = p->worldMode;
  tap[0] = 1.0f; tap[1] = (float)useGls; tap[2] = 0.0f; tap[3] = 0.0f; tap[4] = 0.0f;
  tap[5] = 0.0f; tap[6] = 0.0f; tap[7] = 0.0f;

  if (d.z > 1e-6f) {
    float t = (1.0f - o.z) / d.z;
    float u = (o.x + d.x * t) / aspect + 0.5f;
    float v = o.y + d.y * t + 0.5f;
    float out = FG_FMAX(FG_FMAX(-u, u - 1.0f) * aspect, FG_FMAX(-v, v - 1.0f));
    if (out <= 0.0f) {
      tap[2] = u; tap[3] = v;
    } else if (mode == FG_WORLD_BLACK) {
      tap[0] = 0.0f;
    } else if (mode == FG_WORLD_TRANSPARENT) {
      tap[1] = 0.0f; tap[2] = pixU; tap[3] = pixV;
    } else {
      float wu = u, wv = v, lod = 0.0f;
      if (mode == FG_WORLD_REPEAT) { wu = fg3d_fract(u); wv = fg3d_fract(v); }
      else if (mode == FG_WORLD_CLAMP) { wu = fg3d_clampf(u, 0.0f, 1.0f); wv = fg3d_clampf(v, 0.0f, 1.0f); }
      else { wu = fg3d_mirror(u); wv = fg3d_mirror(v); }
      if (mode == FG_WORLD_SPHERICAL) {
        float f = p->feather > 0.0f ? fg3d_clampf(out / p->feather, 0.0f, 1.0f) : 1.0f;
        lod = p->worldBlur * f;
      }
      tap[2] = wu; tap[3] = wv; tap[4] = lod;
    }
  } else if (mode == FG_WORLD_BLACK) {
    tap[0] = 0.0f;
  } else if (mode == FG_WORLD_TRANSPARENT) {
    tap[1] = 0.0f; tap[2] = pixU; tap[3] = pixV;
  } else {
    /* Turned back toward the camera: a soft mirrored environment built from the source. */
    tap[2] = fg3d_mirror(0.5f - 0.75f * d.x / aspect);
    tap[3] = fg3d_mirror(0.5f + 0.75f * d.y);
    tap[4] = p->worldBlur + 1.0f;
  }

  /* Virtual lights: camera-facing discs, seen only via the glass. */
  if (glassPath != 0) {
    for (int li = 0; li < p->numLights && li < FG_MAX_LIGHTS; li++) {
      int b = li * FG_LIGHT_FLOATS;
      float lz = p->lights[b + 2];
      if (FG_FABS(d.z) < 1e-6f) continue;
      float t = (lz - o.z) / d.z;
      if (t <= 0.0f) continue;
      float dx = o.x + d.x * t - p->lights[b + 0];
      float dy = o.y + d.y * t - p->lights[b + 1];
      float r = p->lights[b + 3];
      float e = fg3d_clampf((r - FG_SQRT(dx * dx + dy * dy)) / (0.15f * r + 1e-6f), 0.0f, 1.0f);
      tap[5] += p->lights[b + 4] * e; tap[6] += p->lights[b + 5] * e; tap[7] += p->lights[b + 6] * e;
    }
  }
}

/* Colour of a landing (see fg3d_escape_tap). */
FG_DEVICE fgv4 fg3d_tap_colour(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                               FG_PRIV const float* tap)
{
  fgv4 c = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
  if (tap[0] != 0.0f) c = fg3d_image(p, src, gls, tap[1] != 0.0f ? 1 : 0, tap[2], tap[3], tap[4]);
  c.x += tap[5]; c.y += tap[6]; c.z += tap[7];
  return c;
}

/* What a ray sees once it has left the glass. */
FG_DEVICE fgv4 fg3d_escape(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                           fgv3 o, fgv3 d, int glassPath, float aspect, float pixU, float pixV)
{
  float tap[8];
  fg3d_escape_tap(p, o, d, glassPath, aspect, pixU, pixV, tap);
  return fg3d_tap_colour(p, src, gls, tap);
}

/* Interleaved gradient noise: a per-pixel value whose error pattern between
   neighbouring pixels is high-frequency (blue-ish), so leftover sampling noise
   reads as fine, even grain rather than blotches. */
FG_DEVICE float fg3d_ign(float x, float y)
{
  return fg3d_fract(52.9829189f * fg3d_fract(0.06711056f * x + 0.00583715f * y));
}

/* Highlight boost for light seen via the glass. */
FG_DEVICE fgv4 fg3d_boost(FG_PCONST const FG3DParams* p, fgv4 c)
{
  float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
  float e = fg3d_clampf((lum - p->threshold) / FG_FMAX(0.25f * p->threshold, 0.05f), 0.0f, 1.0f);
  float g = 1.0f + p->boost * e * e * (3.0f - 2.0f * e);
  c.x *= g; c.y *= g; c.z *= g;
  return c;
}

/* ------------------------------------------------------------------ shade */

/* ------------------------------------------------------------------ playback cache
   When nothing but the footage changes between frames, every ray takes the same
   path. fg3d_trace_cache records, per pixel, where its light lands as up to
   FG_CACHE_SLOTS clusters (image, mean position, blur level from the cluster's
   spread, per-channel weight, highlight boost flag) plus the virtual light it
   saw; fg3d_replay rebuilds the pixel from the new footage with a few lookups.
   Accumulator per slot (FG_SLOT_ACC): 0 sum weight, 1-2 sum w*u, w*v,
   3-4 sum w*u^2, w*v^2, 5 sum w*lod, 6-9 weights R, G, B, A, 10 flags,
   11 sum w*log2 brightness (rays that saw very different brightness are kept
   apart, so a highlight does not smear into its surroundings). */

/* Merge cost of two clusters (weights wa, wb at squared pixel distance d2): the
   spread the merge would add. Different flags (image / boost) cost extra. */
FG_DEVICE float fg3d_merge_cost(float wa, float wb, float d2, int sameFlags)
{
  return (wa * wb / (wa + wb)) * (d2 + (sameFlags != 0 ? 0.0f : 1e6f));
}

FG_DEVICE void fg3d_cache_add(FG_PRIV float* cs, FG_PRIV int* ncs, FG_PRIV float* cL, FG_PRIV const float* tap,
                              int boost, fgv4 col, float wr, float wg, float wb, float wa, float W, float H)
{
  cL[0] += tap[5] * wr; cL[1] += tap[6] * wg; cL[2] += tap[7] * wb;
  if (tap[0] == 0.0f) return;                       /* black: nothing to look up */
  float flags = tap[1] + 2.0f * (float)boost;
  float u = tap[2], v = tap[3], lod = tap[4];
  float w = wr + wg + wb + wa + 1e-12f;
  float reach = FG_CACHE_REACH + fg3d_exp(lod * 0.6931472f) - 1.0f;  /* merge radius in pixels */
  /* Brightness in stops; one stop of difference counts as 2 pixels of distance. */
  float bl = fg3d_log(FG_FMAX(0.2126f * col.x + 0.7152f * col.y + 0.0722f * col.z, 0.0f) + 0.01f) * 1.4426950f;
  /* Nearest cluster with the same flags. */
  int best = -1;
  float bd = 1e30f;
  for (int k = 0; k < *ncs; k++) {
    FG_PRIV float* q = cs + k * FG_SLOT_ACC;
    float du = (q[1] / q[0] - u) * W, dv = (q[2] / q[0] - v) * H, dl = 2.0f * (q[11] / q[0] - bl);
    float d2 = du * du + dv * dv + dl * dl;
    if (q[10] == flags && d2 < bd) { bd = d2; best = k; }
  }
  int k = best;
  if (k < 0 || bd > reach * reach) {
    if (*ncs < FG_CACHE_SLOTS) {
      k = *ncs;
      *ncs = k + 1;
    } else {
      /* Full: merge whichever pair (two clusters, or the new landing and a
         cluster) adds the least spread, then use the freed slot if any. */
      float bc = best >= 0 ? fg3d_merge_cost(cs[best * FG_SLOT_ACC], w, bd, 1) : 1e30f;
      int ma = -1, mb = -1;
      for (int i = 0; i < FG_CACHE_SLOTS; i++) {
        FG_PRIV float* qi = cs + i * FG_SLOT_ACC;
        if (best < 0) {
          float du = (qi[1] / qi[0] - u) * W, dv = (qi[2] / qi[0] - v) * H, dl = 2.0f * (qi[11] / qi[0] - bl);
          float c = fg3d_merge_cost(qi[0], w, du * du + dv * dv + dl * dl, qi[10] == flags ? 1 : 0);
          if (c < bc) { bc = c; k = i; }
        }
        for (int j = i + 1; j < FG_CACHE_SLOTS; j++) {
          FG_PRIV float* qj = cs + j * FG_SLOT_ACC;
          float du = (qi[1] / qi[0] - qj[1] / qj[0]) * W, dv = (qi[2] / qi[0] - qj[2] / qj[0]) * H;
          float dl = 2.0f * (qi[11] / qi[0] - qj[11] / qj[0]);
          float c = fg3d_merge_cost(qi[0], qj[0], du * du + dv * dv + dl * dl, qi[10] == qj[10] ? 1 : 0);
          if (c < bc) { bc = c; ma = i; mb = j; }
        }
      }
      if (ma >= 0) {
        FG_PRIV float* qa = cs + ma * FG_SLOT_ACC;
        FG_PRIV float* qb = cs + mb * FG_SLOT_ACC;
        for (int j = 0; j < 10; j++) qa[j] += qb[j];
        qa[11] += qb[11];
        k = mb;
      } else {
        FG_PRIV float* q = cs + k * FG_SLOT_ACC;
        q[0] += w; q[1] += w * u; q[2] += w * v; q[3] += w * u * u; q[4] += w * v * v; q[5] += w * lod;
        q[6] += wr; q[7] += wg; q[8] += wb; q[9] += wa; q[11] += w * bl;
        return;
      }
    }
    FG_PRIV float* q = cs + k * FG_SLOT_ACC;
    for (int j = 0; j < FG_SLOT_ACC; j++) q[j] = 0.0f;
    q[10] = flags;
  }
  FG_PRIV float* q = cs + k * FG_SLOT_ACC;
  q[0] += w; q[1] += w * u; q[2] += w * v; q[3] += w * u * u; q[4] += w * v * v; q[5] += w * lod;
  q[6] += wr; q[7] += wg; q[8] += wb; q[9] += wa; q[11] += w * bl;
}

FG_DEVICE void fg3d_cache_write(FG_GLOBAL float* out, FG_PRIV const float* cs, int ncs, FG_PRIV const float* cL,
                                float wR, float wG, float wB, float n, float W, float H)
{
  float iR = 1.0f / FG_FMAX(wR, 1e-6f), iG = 1.0f / FG_FMAX(wG, 1e-6f), iB = 1.0f / FG_FMAX(wB, 1e-6f);
  out[0] = cL[0] * iR; out[1] = cL[1] * iG; out[2] = cL[2] * iB; out[3] = (float)ncs;
  for (int k = 0; k < FG_CACHE_SLOTS; k++) {
    FG_GLOBAL float* o = out + 4 + k * FG_SLOT_FLOATS;
    if (k >= ncs) { for (int j = 0; j < FG_SLOT_FLOATS; j++) o[j] = 0.0f; continue; }
    FG_PRIV const float* q = cs + k * FG_SLOT_ACC;
    float iw = 1.0f / q[0];
    float u = q[1] * iw, v = q[2] * iw;
    float su = FG_SQRT(FG_FMAX(0.0f, q[3] * iw - u * u)), sv = FG_SQRT(FG_FMAX(0.0f, q[4] * iw - v * v));
    o[0] = u; o[1] = v; o[2] = su; o[3] = sv; o[4] = q[5] * iw + 32.0f * q[10];
    o[5] = q[6] * iR; o[6] = q[7] * iG; o[7] = q[8] * iB; o[8] = q[9] / n;
  }
}

/* A pixel the glass cannot touch: the footage at the pixel, unchanged. */
FG_DEVICE void fg3d_cache_passthrough(FG_GLOBAL float* out, float pixU, float pixV)
{
  for (int j = 0; j < FG_CACHE_FLOATS; j++) out[j] = 0.0f;
  out[3] = 1.0f;
  out[4] = pixU; out[5] = pixV;
  out[9] = 1.0f; out[10] = 1.0f; out[11] = 1.0f; out[12] = 1.0f;
}

/* Rebuilds a pixel's linear colour from its cache record and the current footage.
   Each cluster is read with 4 taps at its mean +- its spread in u and v (which
   approximates the footprint's average), blurred by the cluster's spread.
   Tuned against 256-sample references (tests: fg3d_render cacheFrom=). */
FG_DEVICE fgv4 fg3d_replay(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                           FG_GLOBAL const float* rec)
{
  fgv4 c = fgv4m(rec[0], rec[1], rec[2], 0.0f);
  int ns = (int)rec[3];
  for (int k = 0; k < ns && k < FG_CACHE_SLOTS; k++) {
    FG_GLOBAL const float* o = rec + 4 + k * FG_SLOT_FLOATS;
    float fl = FG_FLOOR(o[4] / 32.0f);
    float lod = o[4] - 32.0f * fl;
    int flags = (int)fl;
    int img = (flags & 1) != 0 ? 1 : 0;
    float spx = FG_SQRT(o[2] * o[2] * p->frameW * p->frameW + o[3] * o[3] * p->frameH * p->frameH);
    lod = FG_FMAX(lod, fg3d_log(FG_FMAX(1.0f, spx)) * 1.4426950f);
    fgv4 t = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
    for (int q = 0; q < 4; q++) {
      float du = (q & 1) != 0 ? o[2] : -o[2], dv = (q & 2) != 0 ? o[3] : -o[3];
      fgv4 a = fg3d_image(p, src, gls, img, o[0] + du, o[1] + dv, lod);
      if ((flags & 2) != 0) a = fg3d_boost(p, a);
      t.x += a.x; t.y += a.y; t.z += a.z; t.w += a.w;
    }
    c.x += t.x * 0.25f * o[5]; c.y += t.y * 0.25f * o[6]; c.z += t.z * 0.25f * o[7]; c.w += t.w * 0.25f * o[8];
  }
  return c;
}

/* ------------------------------------------------------------------ trace */

#define FG3D_TRACE_FN fg3d_trace
#define FG3D_TRACE_CACHE 0
#define FG3D_TRACE_EXTRA
#include "fg3d_trace.h"
#undef FG3D_TRACE_FN
#undef FG3D_TRACE_CACHE
#undef FG3D_TRACE_EXTRA
#define FG3D_TRACE_FN fg3d_trace_cache
#define FG3D_TRACE_CACHE 1
#define FG3D_TRACE_EXTRA , FG_GLOBAL float* cache
#include "fg3d_trace.h"
#undef FG3D_TRACE_FN
#undef FG3D_TRACE_CACHE
#undef FG3D_TRACE_EXTRA

/* ------------------------------------------------------------------ soft areas
   Speed-up for soft regions: the region is first traced at half resolution
   (fg3d_trace with a half-size frame at origin 0, so half pixel i covers full
   frame-relative pixels 2i and 2i + 1). A full-resolution pixel whose four
   surrounding half-res pixels are all soft (same-path landing spread, aux[12],
   >= minSpread half pixels) and agree on glass coverage takes their bilinear interpolation of
   colour and denoiser guides instead of being traced. `ha` holds FG_AUX_FLOATS
   per half pixel of the hw x hh block starting at half pixel (hx0, hy0);
   (fx, fy) are frame-relative full pixel indices. Returns 1 and fills `out`. */
FG_DEVICE int fg3d_soft_up(FG_GLOBAL const float* ha, int hw, int hh, int hx0, int hy0, int fx, int fy,
                           float minSpread, FG_PRIV float* out)
{
  float gx = ((float)fx + 0.5f) * 0.5f - 0.5f - (float)hx0;
  float gy = ((float)fy + 0.5f) * 0.5f - 0.5f - (float)hy0;
  float x0f = FG_FLOOR(gx), y0f = FG_FLOOR(gy);
  float tx = gx - x0f, ty = gy - y0f;
  int x0 = (int)x0f, y0 = (int)y0f;
  if (x0 < 0 || y0 < 0 || x0 + 1 >= hw || y0 + 1 >= hh) return 0;
  FG_GLOBAL const float* a00 = ha + (y0 * hw + x0) * FG_AUX_FLOATS;
  FG_GLOBAL const float* a10 = a00 + FG_AUX_FLOATS;
  FG_GLOBAL const float* a01 = a00 + hw * FG_AUX_FLOATS;
  FG_GLOBAL const float* a11 = a01 + FG_AUX_FLOATS;
  if (a00[12] < minSpread || a10[12] < minSpread || a01[12] < minSpread || a11[12] < minSpread) return 0;
  float cmin = FG_FMIN(FG_FMIN(a00[5], a10[5]), FG_FMIN(a01[5], a11[5]));
  float cmax = FG_FMAX(FG_FMAX(a00[5], a10[5]), FG_FMAX(a01[5], a11[5]));
  if (cmax - cmin > 0.01f) return 0;
  float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty), w01 = (1.0f - tx) * ty, w11 = tx * ty;
  for (int k = 0; k < FG_AUX_FLOATS; k++) out[k] = a00[k] * w00 + a10[k] * w10 + a01[k] * w01 + a11[k] * w11;
  out[11] *= 2.0f; out[12] *= 2.0f;  /* spreads back in full-resolution pixels */
  return 1;
}

/* Blends a traced (or denoised) linear colour with the original by Mix and
   encodes it back to the footage's colour space. */
FG_DEVICE fgv4 fg3d_finish(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, int x, int y, fgv4 c)
{
  float pixU = ((float)x - p->frameX + 0.5f) / p->frameW, pixV = ((float)y - p->frameY + 0.5f) / p->frameH;
  fgv4 orig = fg3d_lookup(src, p->srcW, p->srcH, p->srcLevels, pixU, pixV, 0.0f);
  float m = p->mix;
  int cs = p->colorSpace;
  return fgv4m(fg3d_encode(cs, orig.x + (c.x - orig.x) * m), fg3d_encode(cs, orig.y + (c.y - orig.y) * m),
               fg3d_encode(cs, orig.z + (c.z - orig.z) * m), orig.w + (c.w - orig.w) * m);
}

/* Trace + finish (no denoiser). */
FG_DEVICE fgv4 fg3d_shade(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                          FG_GLOBAL const float* tris, FG_GLOBAL const float* nodesF, FG_GLOBAL const int* nodesI,
                          int x, int y)
{
  return fg3d_finish(p, src, x, y, fg3d_trace(p, src, gls, tris, nodesF, nodesI, x, y, 0));
}

/* ------------------------------------------------------------------ denoiser */

/* One edge-aware a-trous wavelet pass (5x5 B3-spline taps spaced `step` apart)
   over a w x h region. Neighbours are weighted by how well their colour agrees
   relative to the estimated noise, and by the glass coverage, first-hit normal
   and landing position guides, so facet edges and refracted detail survive.
   The variance is propagated so later passes adapt (as in SVGF). */
FG_DEVICE void fg3d_atrous(FG_GLOBAL const float* aux, FG_GLOBAL const float* cvIn, FG_GLOBAL float* cvOut, int w, int h,
                           int x, int y, int step, float sigma, float frameH)
{
  FG_GLOBAL const float* ap = aux + (y * w + x) * FG_AUX_FLOATS;
  FG_GLOBAL const float* cp = cvIn + (y * w + x) * FG_CV_FLOATS;
  FG_GLOBAL float* op = cvOut + (y * w + x) * FG_CV_FLOATS;
  /* Pixels whose rays land within ~1.5 px of each other show sharp detail (the
     footage in focus, or seen through flat glass): their variation is
     anti-aliasing, not grain, so they pass through. Blurred regions, inside the
     glass or not (out-of-focus footage), are denoised. */
  if (ap[11] < 1.5f) {
    op[0] = cp[0]; op[1] = cp[1]; op[2] = cp[2]; op[3] = cp[3]; op[4] = cp[4];
    return;
  }
  /* Colour agreement is judged on compressed luminance T(l) = l / (1 + l), so
     bright out-of-focus highlights (bokeh) can be smoothed; blending itself
     stays in linear light. The noise scale maps through T'(l) = 1 / (1 + l)^2. */
  float lin = FG_FMAX(0.0f, 0.2126f * cp[0] + 0.7152f * cp[1] + 0.0722f * cp[2]);
  float lp = lin / (1.0f + lin);
  float dT = 1.0f / ((1.0f + lin) * (1.0f + lin));
  /* Noise scale from a 3x3 Gaussian of the neighbours' variance (as in SVGF): a
     pixel whose few rays all missed a small bright light looks noise-free on its
     own, which would stop it blending with its bright neighbours. */
  float vsum = 0.0f;
  for (int vy = -1; vy <= 1; vy++)
    for (int vx = -1; vx <= 1; vx++) {
      int sx = fg3d_clampi(x + vx, 0, w - 1), sy = fg3d_clampi(y + vy, 0, h - 1);
      float kw = ((vx == 0) ? 0.5f : 0.25f) * ((vy == 0) ? 0.5f : 0.25f);
      vsum += kw * FG_FMAX(cvIn[(sy * w + sx) * FG_CV_FLOATS + 4], 0.0f);
    }
  float denom = sigma * FG_SQRT(vsum) * dT + 1e-4f;
  float sr = 0.0f, sg = 0.0f, sb = 0.0f, sa = 0.0f, sv = 0.0f, sw = 0.0f;
  for (int dy = -2; dy <= 2; dy++) {
    float ky = (dy == 0) ? 0.375f : ((dy == 1 || dy == -1) ? 0.25f : 0.0625f);
    int qy = fg3d_clampi(y + dy * step, 0, h - 1);
    for (int dx = -2; dx <= 2; dx++) {
      float kx = (dx == 0) ? 0.375f : ((dx == 1 || dx == -1) ? 0.25f : 0.0625f);
      int qx = fg3d_clampi(x + dx * step, 0, w - 1);
      FG_GLOBAL const float* aq = aux + (qy * w + qx) * FG_AUX_FLOATS;
      FG_GLOBAL const float* cq = cvIn + (qy * w + qx) * FG_CV_FLOATS;
      float wt = kx * ky;
      if (aq[11] < 1.5f) wt = 0.0f;   /* never borrow from sharp pixels */
      else if (dx != 0 || dy != 0) {
        float lq = FG_FMAX(0.0f, 0.2126f * cq[0] + 0.7152f * cq[1] + 0.0722f * cq[2]);
        lq = lq / (1.0f + lq);
        wt *= fg3d_exp(-FG_FABS(lp - lq) / denom);
        wt *= fg3d_exp(-6.0f * FG_FABS(ap[5] - aq[5]));
        if (ap[5] > 0.01f && aq[5] > 0.01f) {
          float nd = FG_FMAX(0.0f, ap[6] * aq[6] + ap[7] * aq[7] + ap[8] * aq[8]);
          nd *= nd; nd *= nd; nd *= nd; nd *= nd; nd *= nd;   /* ^32 */
          wt *= nd;
        }
        float du = ap[9] - aq[9], dv = ap[10] - aq[10];
        float duv = FG_SQRT(du * du + dv * dv) * frameH / (float)step;
        wt *= fg3d_exp(-0.5f * FG_FMAX(0.0f, duv - 4.0f));
      }
      sr += cq[0] * wt; sg += cq[1] * wt; sb += cq[2] * wt; sa += cq[3] * wt;
      sv += cq[4] * wt * wt;
      sw += wt;
    }
  }
  float inv = 1.0f / sw;  /* sw > 0: the centre always counts */
  op[0] = sr * inv; op[1] = sg * inv; op[2] = sb * inv; op[3] = sa * inv; op[4] = sv * inv * inv;
}

#endif
