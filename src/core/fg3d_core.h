/* FacetGlass 3D core - single-source path tracer for solid glass objects.
 *
 * Compiled unchanged as C++ (CPU), CUDA, OpenCL C and Metal. A backend prelude
 * defines, before including this file:
 *   FG_DEVICE  function qualifier        FG_GLOBAL  buffer address space
 *   FG_PRIV    private pointer space     FG_SQRT FG_SIN FG_COS FG_ATAN2 FG_FLOOR
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

/* Iterative BVH traversal in object space. Returns triangle index or -1. */
FG_DEVICE int fg3d_intersect(FG_GLOBAL const float* tris, FG_GLOBAL const float* nodesF, FG_GLOBAL const int* nodesI,
                             int numNodes, fgv3 ro, fgv3 rd, float tmin, FG_PRIV float* tHit, FG_PRIV float* uHit,
                             FG_PRIV float* vHit)
{
  int stack[FG_BVH_STACK];
  int sp = 0;
  int best = -1;
  float tBest = 1e30f;
  float ix = 1.0f / (FG_FABS(rd.x) > 1e-12f ? rd.x : 1e-12f);
  float iy = 1.0f / (FG_FABS(rd.y) > 1e-12f ? rd.y : 1e-12f);
  float iz = 1.0f / (FG_FABS(rd.z) > 1e-12f ? rd.z : 1e-12f);
  if (numNodes <= 0) return -1;
  stack[sp++] = 0;
  while (sp > 0) {
    int ni = stack[--sp];
    FG_GLOBAL const float* nb = nodesF + ni * FG_NODE_FLOATS;
    float t0x = (nb[0] - ro.x) * ix, t1x = (nb[3] - ro.x) * ix;
    float t0y = (nb[1] - ro.y) * iy, t1y = (nb[4] - ro.y) * iy;
    float t0z = (nb[2] - ro.z) * iz, t1z = (nb[5] - ro.z) * iz;
    float tn = FG_FMAX(FG_FMAX(FG_FMIN(t0x, t1x), FG_FMIN(t0y, t1y)), FG_FMIN(t0z, t1z));
    float tf = FG_FMIN(FG_FMIN(FG_FMAX(t0x, t1x), FG_FMAX(t0y, t1y)), FG_FMAX(t0z, t1z));
    if (tf < FG_FMAX(tn, tmin) || tn > tBest) continue;
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
    } else if (sp + 2 <= FG_BVH_STACK) {
      stack[sp++] = first + 1;
      stack[sp++] = first;
    }
  }
  *tHit = tBest;
  return best;
}

/* Object <-> world (uniform scale): world = pos + scale * (R p). */
FG_DEVICE fgv3 fg3d_rot(FG_PRIV const FG3DParams* p, fgv3 a)
{
  return fgv(p->objRot[0] * a.x + p->objRot[1] * a.y + p->objRot[2] * a.z,
             p->objRot[3] * a.x + p->objRot[4] * a.y + p->objRot[5] * a.z,
             p->objRot[6] * a.x + p->objRot[7] * a.y + p->objRot[8] * a.z);
}
FG_DEVICE fgv3 fg3d_rotT(FG_PRIV const FG3DParams* p, fgv3 a)
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
FG_DEVICE fgv4 fg3d_image(FG_PRIV const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                          int useGls, float u, float v, float lod)
{
  if (useGls != 0) return fg3d_lookup(gls, p->glsW, p->glsH, p->glsLevels, u, v, lod);
  return fg3d_lookup(src, p->srcW, p->srcH, p->srcLevels, u, v, lod);
}

/* What a ray sees once it has left the glass. */
FG_DEVICE fgv4 fg3d_escape(FG_PRIV const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                           fgv3 o, fgv3 d, int glassPath, float aspect, float pixU, float pixV)
{
  fgv4 c = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
  int useGls = (glassPath != 0 && p->useSource != 0) ? 1 : 0;
  int mode = p->worldMode;

  if (d.z > 1e-6f) {
    float t = (1.0f - o.z) / d.z;
    float u = (o.x + d.x * t) / aspect + 0.5f;
    float v = o.y + d.y * t + 0.5f;
    float out = FG_FMAX(FG_FMAX(-u, u - 1.0f) * aspect, FG_FMAX(-v, v - 1.0f));
    if (out <= 0.0f) {
      c = fg3d_image(p, src, gls, useGls, u, v, 0.0f);
    } else if (mode == FG_WORLD_BLACK) {
      c = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (mode == FG_WORLD_TRANSPARENT) {
      c = fg3d_image(p, src, gls, 0, pixU, pixV, 0.0f);
    } else {
      float wu = u, wv = v, lod = 0.0f;
      if (mode == FG_WORLD_REPEAT) { wu = fg3d_fract(u); wv = fg3d_fract(v); }
      else if (mode == FG_WORLD_CLAMP) { wu = fg3d_clampf(u, 0.0f, 1.0f); wv = fg3d_clampf(v, 0.0f, 1.0f); }
      else { wu = fg3d_mirror(u); wv = fg3d_mirror(v); }
      if (mode == FG_WORLD_SPHERICAL) {
        float f = p->feather > 0.0f ? fg3d_clampf(out / p->feather, 0.0f, 1.0f) : 1.0f;
        lod = p->worldBlur * f;
      }
      c = fg3d_image(p, src, gls, useGls, wu, wv, lod);
    }
  } else if (mode == FG_WORLD_BLACK) {
    c = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
  } else if (mode == FG_WORLD_TRANSPARENT) {
    c = fg3d_image(p, src, gls, 0, pixU, pixV, 0.0f);
  } else {
    /* Turned back toward the camera: a soft mirrored environment built from the source. */
    float u = fg3d_mirror(0.5f - 0.75f * d.x / aspect);
    float v = fg3d_mirror(0.5f + 0.75f * d.y);
    c = fg3d_image(p, src, gls, useGls, u, v, p->worldBlur + 1.0f);
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
      c.x += p->lights[b + 4] * e; c.y += p->lights[b + 5] * e; c.z += p->lights[b + 6] * e;
    }
  }
  return c;
}

/* Interleaved gradient noise: a per-pixel value whose error pattern between
   neighbouring pixels is high-frequency (blue-ish), so leftover sampling noise
   reads as fine, even grain rather than blotches. */
FG_DEVICE float fg3d_ign(float x, float y)
{
  return fg3d_fract(52.9829189f * fg3d_fract(0.06711056f * x + 0.00583715f * y));
}

/* Highlight boost for light seen via the glass. */
FG_DEVICE fgv4 fg3d_boost(FG_PRIV const FG3DParams* p, fgv4 c)
{
  float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
  float e = fg3d_clampf((lum - p->threshold) / FG_FMAX(0.25f * p->threshold, 0.05f), 0.0f, 1.0f);
  float g = 1.0f + p->boost * e * e * (3.0f - 2.0f * e);
  c.x *= g; c.y *= g; c.z *= g;
  return c;
}

/* ------------------------------------------------------------------ shade */

/* Traces one pixel: returns its linear RGBA (before Mix and encoding). When
   `aux` is not null it also writes the denoiser guides (FG_AUX_FLOATS). */
FG_DEVICE fgv4 fg3d_trace(FG_PRIV const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                          FG_GLOBAL const float* tris, FG_GLOBAL const float* nodesF, FG_GLOBAL const int* nodesI,
                          int x, int y, FG_GLOBAL float* aux)
{
  float W = p->frameW, H = p->frameH;
  float aspect = W / H;
  float fx = (float)x - p->frameX, fy = (float)y - p->frameY;
  float pixU = (fx + 0.5f) / W, pixV = (fy + 0.5f) / H;
  float covN = 0.0f, gnx = 0.0f, gny = 0.0f, gnz = 0.0f, luSum = 0.0f, lvSum = 0.0f, luSq = 0.0f, lvSq = 0.0f;
  /* Sums for the variance of the weighted (spectral) mean, per channel:
     sum w^2 c^2, sum w^2 c, sum w^2. */
  float qR2 = 0.0f, qR1 = 0.0f, qR0 = 0.0f, qG2 = 0.0f, qG1 = 0.0f, qG0 = 0.0f, qB2 = 0.0f, qB1 = 0.0f, qB0 = 0.0f;

  float zf = 1.0f / p->focusInv;
  float zg = 1.0f / p->glassFocusInv;
  float s = p->objScale;
  float eps = 2e-4f;               /* object-space ray offset */

  unsigned int pix = fg3d_hash((unsigned int)x * 1973u + (unsigned int)y * 9277u + (unsigned int)p->seed * 26699u);
  /* Per-pixel offsets for the progressive (R2 / golden) sequences, from
     interleaved gradient noise so neighbouring pixels' errors cancel visually. */
  float bx = (float)x + (float)(p->seed % 1000) * 7.0f, by = (float)y + (float)(p->seed % 1000) * 3.0f;
  float o1 = fg3d_ign(bx, by), o2 = fg3d_ign(bx + 5.588238f, by + 5.588238f);
  float o3 = fg3d_ign(bx + 11.176476f, by + 11.176476f), o4 = fg3d_ign(bx + 16.764714f, by + 16.764714f);
  float o5 = fg3d_ign(bx + 22.352952f, by + 22.352952f);

  float accR = 0.0f, accG = 0.0f, accB = 0.0f, accA = 0.0f;
  float wR = 0.0f, wG = 0.0f, wB = 0.0f;
  /* Adaptive sampling statistics per channel (Welford), so colour noise from
     dispersion keeps sampling even when the brightness has settled. */
  float meanR = 0.0f, meanG = 0.0f, meanB = 0.0f, m2R = 0.0f, m2G = 0.0f, m2B = 0.0f;
  int minN = p->minSamples < 1 ? 1 : p->minSamples;
  int maxN = p->maxSamples < minN ? minN : p->maxSamples;
  int n = 0;

  for (int i = 0; i < maxN; i++) {
    float fi = (float)i;
    /* One 5-D low-discrepancy (Kronecker / generalised golden ratio) sequence for
       pixel jitter, aperture point and wavelength, so the dimensions are not
       correlated with each other and every prefix is evenly spread (adaptive
       stopping keeps good coverage). alpha_k = phi5^-k, phi5^6 = phi5 + 1. */
    float jx = fg3d_fract(o1 + fi * 0.8812714616f) - 0.5f;
    float jy = fg3d_fract(o2 + fi * 0.7766393385f) - 0.5f;
    /* Concentric square-to-disc mapping keeps the aperture samples stratified. */
    float su = 2.0f * fg3d_fract(o3 + fi * 0.6844320127f) - 1.0f;
    float sv = 2.0f * fg3d_fract(o4 + fi * 0.6031704985f) - 1.0f;
    float ar = 0.0f, at = 0.0f;
    if (su * su > sv * sv) { ar = su; at = 0.7853981634f * (sv / su); }
    else if (sv != 0.0f) { ar = sv; at = 1.5707963268f - 0.7853981634f * (su / sv); }
    ar *= p->aperture;
    float sa = 0.0f, ca = 1.0f;
    fg3d_sincos(at, &sa, &ca);
    float ax = ar * ca, ay = ar * sa;
    float px = ((fx + 0.5f + jx) / W - 0.5f) * aspect;
    float py = (fy + 0.5f + jy) / H - 0.5f;

    float lam = 0.55f;
    float ior = p->ior;
    fgv3 sw = fgv(1.0f, 1.0f, 1.0f);
    if (p->spectral != 0) {
      lam = 0.40f + 0.30f * fg3d_fract(o5 + fi * 0.5315564363f);
      ior += p->dispersion * 0.05f * (1.0f / (lam * lam) - 3.306f);
      sw = fg3d_spectral_weight(lam);
    }

    unsigned int rs = fg3d_hash(pix + (unsigned int)i * 2654435761u);

    /* Camera ray through the focus plane. */
    fgv3 o = fgv(ax, ay, 0.0f);
    fgv3 d = fgv_norm(fgv(px * zf - ax, py * zf - ay, zf));
    int glass = 0;   /* entered the glass at least once */
    /* Outer reflections are added as weighted environment lookups instead of a
       random reflect/refract choice (which made sparkle noise); the path itself
       always continues by refraction, or total internal reflection. */
    float thr = 1.0f;
    /* Colour left after absorption inside tinted glass (1 = clear). */
    float tR = 1.0f, tG = 1.0f, tB = 1.0f;
    int inside = 0;
    fgv4 add = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);
    /* Inner reflections: where light leaves the glass, the Fresnel part that
       reflects back inside is traced too, as a deterministic branch with its own
       weight (no random choice, so no sparkle noise). Pending branches wait on a
       small stack; the main path is traced first, then each branch. */
    float stO[FG_BRANCH_STACK * 3], stD[FG_BRANCH_STACK * 3], stW[FG_BRANCH_STACK], stT[FG_BRANCH_STACK * 3];
    int stB[FG_BRANCH_STACK];
    int nst = 0, isMain = 1, b = 0, seg = 0;
    int segMax = 2 * p->maxBounces;  /* all branches of one sample together */
    fgv4 c = fgv4m(0.0f, 0.0f, 0.0f, 0.0f);

    for (;;) {
      for (; b < p->maxBounces && seg < segMax; b++, seg++) {
        /* Into object space. */
        fgv3 ro = fg3d_rotT(p, fgv_mul(fgv_sub(o, fgv(p->objPos[0], p->objPos[1], p->objPos[2])), 1.0f / s));
        fgv3 rd = fg3d_rotT(p, d);
        float th = 0.0f, uh = 0.0f, vh = 0.0f;
        int k = fg3d_intersect(tris, nodesF, nodesI, p->numNodes, ro, rd, eps, &th, &uh, &vh);
        if (k < 0) break;
        FG_GLOBAL const float* tr = tris + k * FG_TRI_FLOATS;
        fgv3 hitO = fgv_madd(ro, rd, th);
        if (inside != 0 && p->absorbOn != 0) {
          /* Travelled th object units through tinted glass. */
          tR *= fg3d_exp(-p->absorb[0] * th);
          tG *= fg3d_exp(-p->absorb[1] * th);
          tB *= fg3d_exp(-p->absorb[2] * th);
        }
        fgv3 ng = fgv_load(tr + 18);
        fgv3 ns = fg3d_shading_normal(tr, uh, vh, p->bevelWidth / s);
        if (b == 0 && isMain != 0) {
          fgv3 nw = fg3d_rot(p, ns);
          covN += 1.0f; gnx += nw.x; gny += nw.y; gnz += nw.z;
        }
        int entering = fgv_dot(rd, ng) < 0.0f ? 1 : 0;
        fgv3 nn = entering != 0 ? ns : fgv_mul(ns, -1.0f);      /* faces the incoming ray */
        fgv3 gn = entering != 0 ? ng : fgv_mul(ng, -1.0f);
        if (fgv_dot(nn, rd) >= 0.0f) nn = gn;
        float n1 = entering != 0 ? 1.0f : ior, n2 = entering != 0 ? ior : 1.0f;
        float cosi = -fgv_dot(rd, nn);
        float cost = 0.0f;
        float F = fg3d_fresnel(cosi, n1, n2, &cost);
        int tir = F >= 1.0f ? 1 : 0;
        fgv3 nd;
        fgv3 hitW = fgv_add(fgv(p->objPos[0], p->objPos[1], p->objPos[2]), fgv_mul(fg3d_rot(p, hitO), s));
        if (tir == 0 && entering != 0) {
          float fr = fg3d_clampf(F * p->reflect, 0.0f, 1.0f);
          if (fr > 0.0f) {
            fgv3 rr = fgv_madd(rd, nn, 2.0f * cosi);
            if (p->roughness > 0.0f) {
              float r1 = fg3d_rand(rs ^ (unsigned int)(b * 0x85EBCA6Bu + 3u)) * 2.0f - 1.0f;
              float r2 = fg3d_rand(rs ^ (unsigned int)(b * 0xC2B2AE35u + 5u)) * 2.0f - 1.0f;
              float r3 = fg3d_rand(rs ^ (unsigned int)(b * 0x27D4EB2Fu + 7u)) * 2.0f - 1.0f;
              fgv3 pd = fgv_norm(fgv_add(rr, fgv_mul(fgv(r1, r2, r3), p->roughness)));
              if (fgv_dot(pd, gn) > 0.0f) rr = pd;
            }
            if (fgv_dot(rr, gn) <= 0.0f) rr = fgv_madd(rd, gn, -2.0f * fgv_dot(rd, gn));
            fgv4 rc = fg3d_boost(p, fg3d_escape(p, src, gls, hitW, fg3d_rot(p, fgv_norm(rr)), 1, aspect, pixU, pixV));
            float wgt = thr * fr;
            float wr = wgt, wg = wgt, wbl = wgt;
            if (p->absorbOn != 0) { wr *= tR; wg *= tG; wbl *= tB; }
            add.x += rc.x * wr; add.y += rc.y * wg; add.z += rc.z * wbl; add.w += rc.w * wgt;
            thr *= 1.0f - fr;
          }
        }
        if (tir == 0 && entering == 0 && p->innerReflect > 0.0f) {
          /* Leaving the glass: the reflected part stays inside as a branch. */
          float fq = fg3d_clampf(F * p->innerReflect, 0.0f, 1.0f);
          float wb = thr * fq;
          float wbc = p->absorbOn != 0 ? wb * FG_FMAX(tR, FG_FMAX(tG, tB)) : wb;
          if (wbc >= FG_BRANCH_MIN && nst < FG_BRANCH_STACK) {
            fgv3 rr = fgv_madd(rd, nn, 2.0f * cosi);
            if (fgv_dot(rr, gn) <= 0.0f) rr = fgv_madd(rd, gn, -2.0f * fgv_dot(rd, gn));
            fgv3 rw = fg3d_rot(p, fgv_norm(rr));
            stO[nst * 3] = hitW.x; stO[nst * 3 + 1] = hitW.y; stO[nst * 3 + 2] = hitW.z;
            stD[nst * 3] = rw.x; stD[nst * 3 + 1] = rw.y; stD[nst * 3 + 2] = rw.z;
            stW[nst] = wb; stB[nst] = b + 1;
            stT[nst * 3] = tR; stT[nst * 3 + 1] = tG; stT[nst * 3 + 2] = tB;
            nst++;
          }
          thr *= 1.0f - fq;
        }
        if (tir != 0) {
          nd = fgv_madd(rd, nn, 2.0f * cosi);
          if (fgv_dot(nd, gn) <= 0.0f) nd = fgv_madd(rd, gn, -2.0f * fgv_dot(rd, gn));
        } else {
          float eta = n1 / n2;
          nd = fgv_add(fgv_mul(rd, eta), fgv_mul(nn, eta * cosi - cost));
          if (fgv_dot(nd, gn) >= 0.0f) {
            /* The shading normal sent it the wrong way: refract about the geometric normal. */
            float cg = -fgv_dot(rd, gn);
            float s2 = eta * eta * (1.0f - cg * cg);
            if (s2 < 1.0f) nd = fgv_add(fgv_mul(rd, eta), fgv_mul(gn, eta * cg - FG_SQRT(1.0f - s2)));
            else nd = fgv_madd(rd, gn, 2.0f * cg);
          }
          if (entering != 0) { glass = 1; inside = 1; } else inside = 0;
        }
        nd = fgv_norm(nd);
        /* Back to world space. */
        o = hitW;
        d = fg3d_rot(p, nd);
      }

      fgv4 e;
      if (isMain != 0) {
        if (glass != 0 && p->glassFocusInv != p->focusInv && d.z > 1e-6f) {
          /* Glass focus: shift the landing point by the difference between the two
             focus distances for this aperture point (a non-physical diopter control). */
          fgv3 dF = fgv_norm(fgv(px * zf - ax, py * zf - ay, zf));
          fgv3 dG = fgv_norm(fgv(px * zg - ax, py * zg - ay, zg));
          float t = (1.0f - o.z) / d.z;
          float lx = o.x + d.x * t + (ax + dG.x / dG.z) - (ax + dF.x / dF.z);
          float ly = o.y + d.y * t + (ay + dG.y / dG.z) - (ay + dF.y / dF.z);
          fgv3 lp = fgv(lx, ly, 1.0f);
          e = fg3d_escape(p, src, gls, fgv_madd(lp, d, -1e-3f), d, glass, aspect, pixU, pixV);
        } else {
          e = fg3d_escape(p, src, gls, o, d, glass, aspect, pixU, pixV);
        }
        /* Landing position on the footage plane (denoiser guide; -1 when turned back). */
        if (d.z > 1e-6f) {
          float tl = (1.0f - o.z) / d.z;
          float lu = (o.x + d.x * tl) / aspect + 0.5f, lv = o.y + d.y * tl + 0.5f;
          luSum += lu; lvSum += lv; luSq += lu * lu; lvSq += lv * lv;
        } else {
          luSum -= 1.0f; lvSum -= 1.0f; luSq += 1.0f; lvSq += 1.0f;
        }
      } else {
        e = fg3d_escape(p, src, gls, o, d, 1, aspect, pixU, pixV);
      }
      if (glass != 0) e = fg3d_boost(p, e);
      {
        float kr = thr, kg = thr, kb = thr;
        if (p->absorbOn != 0) { kr *= tR; kg *= tG; kb *= tB; }
        c.x += e.x * kr; c.y += e.y * kg; c.z += e.z * kb; c.w += e.w * thr;
      }

      if (nst == 0) break;
      nst--;
      o = fgv(stO[nst * 3], stO[nst * 3 + 1], stO[nst * 3 + 2]);
      d = fgv(stD[nst * 3], stD[nst * 3 + 1], stD[nst * 3 + 2]);
      thr = stW[nst];
      tR = stT[nst * 3]; tG = stT[nst * 3 + 1]; tB = stT[nst * 3 + 2];
      b = stB[nst];
      glass = 1;
      inside = 1;
      isMain = 0;
    }
    c.x += add.x; c.y += add.y; c.z += add.z; c.w += add.w;

    accR += c.x * sw.x; wR += sw.x;
    accG += c.y * sw.y; wG += sw.y;
    accB += c.z * sw.z; wB += sw.z;
    accA += c.w;
    n = i + 1;
    {
      float a2 = sw.x * sw.x, b2 = sw.y * sw.y, d2 = sw.z * sw.z;
      qR2 += a2 * c.x * c.x; qR1 += a2 * c.x; qR0 += a2;
      qG2 += b2 * c.y * c.y; qG1 += b2 * c.y; qG0 += b2;
      qB2 += d2 * c.z * c.z; qB1 += d2 * c.z; qB0 += d2;
    }

    /* Per-sample channel estimates: spectral weights divided by their average
       over 400-700 nm, so each is an unbiased sample of the final channel. */
    float vR = c.x, vG = c.y, vB = c.z;
    if (p->spectral != 0) { vR *= sw.x * (1.0f / 0.4178f); vG *= sw.y * (1.0f / 0.3760f); vB *= sw.z * (1.0f / 0.2988f); }
    float inv = 1.0f / (float)n;
    float dR = vR - meanR; meanR += dR * inv; m2R += dR * (vR - meanR);
    float dG = vG - meanG; meanG += dG * inv; m2G += dG * (vG - meanG);
    float dB = vB - meanB; meanB += dB * inv; m2B += dB * (vB - meanB);
    if (n >= minN && n < maxN && (n & 3) == 0 && p->noiseThreshold > 0.0f) {
      float k = 1.0f / ((float)n * (float)(n - 1));
      float lum = FG_FMAX(0.0f, 0.2126f * meanR + 0.7152f * meanG + 0.0722f * meanB);
      /* Error relative to the pixel's brightness, in a perceptual-ish (square-root) sense. */
      float se = FG_SQRT(FG_FMAX(FG_FMAX(m2R, m2G), m2B) * k);
      if (se / (FG_SQRT(lum) * 0.5f + 0.05f) < p->noiseThreshold * 2.0f) break;
    }
  }

  float rR = accR / FG_FMAX(wR, 1e-6f), rG = accG / FG_FMAX(wG, 1e-6f), rB = accB / FG_FMAX(wB, 1e-6f);
  float rA = accA / (float)(n > 0 ? n : 1);
  if (aux != 0) {
    float fn = (float)(n > 0 ? n : 1);
    aux[0] = rR; aux[1] = rG; aux[2] = rB; aux[3] = rA;
    /* Variance of each weighted mean: sum w^2 (c - mean)^2 / (sum w)^2, with the
       n / (n - 1) small-sample correction; combined as luminance. */
    float vr = FG_FMAX(0.0f, qR2 - 2.0f * rR * qR1 + rR * rR * qR0) / FG_FMAX(wR * wR, 1e-12f);
    float vg = FG_FMAX(0.0f, qG2 - 2.0f * rG * qG1 + rG * rG * qG0) / FG_FMAX(wG * wG, 1e-12f);
    float vb = FG_FMAX(0.0f, qB2 - 2.0f * rB * qB1 + rB * rB * qB0) / FG_FMAX(wB * wB, 1e-12f);
    aux[4] = n > 1 ? (0.0452f * vr + 0.5115f * vg + 0.0052f * vb) * fn / (fn - 1.0f) : 1.0f;
    aux[5] = covN / fn;
    float nl = FG_SQRT(gnx * gnx + gny * gny + gnz * gnz);
    float ni = nl > 1e-12f ? 1.0f / nl : 0.0f;
    aux[6] = gnx * ni; aux[7] = gny * ni; aux[8] = gnz * ni;
    aux[9] = luSum / fn; aux[10] = lvSum / fn;
    /* Spread of the landing points in pixels: rays that all land within about a
       pixel see sharp detail (their variation is anti-aliasing, not grain). */
    float vu = FG_FMAX(0.0f, luSq / fn - aux[9] * aux[9]), vv = FG_FMAX(0.0f, lvSq / fn - aux[10] * aux[10]);
    aux[11] = FG_SQRT(vu * W * W + vv * H * H);
  }
  return fgv4m(rR, rG, rB, rA);
}

/* Blends a traced (or denoised) linear colour with the original by Mix and
   encodes it back to the footage's colour space. */
FG_DEVICE fgv4 fg3d_finish(FG_PRIV const FG3DParams* p, FG_GLOBAL const float* src, int x, int y, fgv4 c)
{
  float pixU = ((float)x - p->frameX + 0.5f) / p->frameW, pixV = ((float)y - p->frameY + 0.5f) / p->frameH;
  fgv4 orig = fg3d_lookup(src, p->srcW, p->srcH, p->srcLevels, pixU, pixV, 0.0f);
  float m = p->mix;
  int cs = p->colorSpace;
  return fgv4m(fg3d_encode(cs, orig.x + (c.x - orig.x) * m), fg3d_encode(cs, orig.y + (c.y - orig.y) * m),
               fg3d_encode(cs, orig.z + (c.z - orig.z) * m), orig.w + (c.w - orig.w) * m);
}

/* Trace + finish (no denoiser). */
FG_DEVICE fgv4 fg3d_shade(FG_PRIV const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
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
