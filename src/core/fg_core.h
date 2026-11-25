/* FacetGlass core - single-source ray-traced glass shader.
 *
 * Compiled unchanged as C++ (CPU), CUDA, OpenCL C and Metal. Before including,
 * a backend prelude defines:
 *   FG_DEVICE   function qualifier         FG_GLOBAL  image buffer address space
 *   FG_PRIV     private pointer space       FG_SQRT FG_SIN FG_COS FG_ATAN2 FG_FLOOR
 *   FG_FABS FG_FMAX FG_FMIN FG_EXP           math functions (float)
 *
 * Model: pinhole camera at the origin looking down +z; a thin glass element at
 * z = glassDist; the footage on a plane at z = 1. Each sample is a ray from a
 * point on the lens aperture aimed at the focus plane, refracted by Snell's law
 * through the glass (flat front face, shaped back face), then traced to the
 * footage plane. Aperture sampling gives physically soft facet edges and bokeh;
 * spectral sampling gives dispersion.
 */
#ifndef FG_CORE_H
#define FG_CORE_H

#include "fg_params.h"

typedef struct fg3 { float x, y, z; } fg3;
typedef struct fg4 { float x, y, z, w; } fg4;

FG_DEVICE fg3 fg3_make(float x, float y, float z) { fg3 r; r.x = x; r.y = y; r.z = z; return r; }
FG_DEVICE fg4 fg4_make(float x, float y, float z, float w) { fg4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

FG_DEVICE unsigned int fg_hash(unsigned int x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

FG_DEVICE float fg_rand(unsigned int x)
{
  return (float)(fg_hash(x) & 0x00ffffffU) * (1.0f / 16777216.0f);
}

FG_DEVICE fg3 fg_norm(fg3 a)
{
  float l = FG_SQRT(a.x * a.x + a.y * a.y + a.z * a.z);
  return fg3_make(a.x / l, a.y / l, a.z / l);
}

FG_DEVICE float fg_dot(fg3 a, fg3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/* Snell refraction; N faces against I. Returns z = 0 vector on total internal reflection. */
FG_DEVICE fg3 fg_refract(fg3 I, fg3 N, float eta)
{
  float c = -fg_dot(I, N);
  float k = 1.0f - eta * eta * (1.0f - c * c);
  if (k < 0.0f) return fg3_make(0.0f, 0.0f, 0.0f);
  float s = eta * c - FG_SQRT(k);
  return fg3_make(eta * I.x + s * N.x, eta * I.y + s * N.y, eta * I.z + s * N.z);
}

FG_DEVICE float fg_wrap(float u, int mode)
{
  if (mode == FG_EDGE_MIRROR) {
    float m = u - 2.0f * FG_FLOOR(u * 0.5f);
    return m > 1.0f ? 2.0f - m : m;
  }
  if (mode == FG_EDGE_REPEAT) return u - FG_FLOOR(u);
  return FG_FMAX(0.0f, FG_FMIN(1.0f, u));
}

FG_DEVICE float fg_tan(float a) { return FG_SIN(a) / FG_COS(a); }

FG_DEVICE int fg_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Bilinear fetch at absolute pixel position (X, Y), pixel centres at +0.5. */
FG_DEVICE fg4 fg_sample(FG_GLOBAL const float* img, int x1, int y1, int w, int h, int stride, float X, float Y)
{
  float fx = X - 0.5f - (float)x1;
  float fy = Y - 0.5f - (float)y1;
  float flx = FG_FLOOR(fx);
  float fly = FG_FLOOR(fy);
  float ax = fx - flx;
  float ay = fy - fly;
  int ix0 = fg_clampi((int)flx, 0, w - 1);
  int iy0 = fg_clampi((int)fly, 0, h - 1);
  int ix1 = fg_clampi((int)flx + 1, 0, w - 1);
  int iy1 = fg_clampi((int)fly + 1, 0, h - 1);
  FG_GLOBAL const float* p00 = img + iy0 * stride + ix0 * 4;
  FG_GLOBAL const float* p10 = img + iy0 * stride + ix1 * 4;
  FG_GLOBAL const float* p01 = img + iy1 * stride + ix0 * 4;
  FG_GLOBAL const float* p11 = img + iy1 * stride + ix1 * 4;
  float w00 = (1.0f - ax) * (1.0f - ay), w10 = ax * (1.0f - ay);
  float w01 = (1.0f - ax) * ay, w11 = ax * ay;
  return fg4_make(p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
                  p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
                  p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
                  p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11);
}

/* Back-face slope (x, y) at screen position (sx, sy); z = 1 on glass, 0 clear. */
FG_DEVICE fg3 fg_tilt(FG_PRIV const FGParams* p, float sx, float sy, float cx, float cy, float ca, float sa)
{
  float dx = sx - cx;
  float dy = sy - cy;
  float lx = (dx * ca + dy * sa) / p->size;
  float ly = (-dx * sa + dy * ca) / p->size;
  float r = FG_SQRT(lx * lx + ly * ly);
  float N = (float)(p->facets < 1 ? 1 : p->facets);
  float tlx = 0.0f, tly = 0.0f, inside = 0.0f;
  int shape = p->shape;

  if (shape == FG_SHAPE_LINEAR) {
    /* Stepped wedges, each one more step of tilt than the last. */
    if (lx > 0.0f || p->symmetric != 0) {
      float k = FG_FMIN(FG_FLOOR(FG_FABS(lx) * N) + 1.0f, N);
      float sgn = lx < 0.0f ? -1.0f : 1.0f;
      tlx = sgn * fg_tan(p->prism * k);
      inside = 1.0f;
    }
  } else if (shape == FG_SHAPE_RADIAL) {
    /* N facets around a clear centre, each tilted outward. */
    if (r > p->inner) {
      float a = FG_ATAN2(ly, lx);
      float j = FG_FLOOR(a / 6.2831853f * N + 0.5f);
      float phi = j * 6.2831853f / N;
      float t = fg_tan(p->prism);
      tlx = t * FG_COS(phi);
      tly = t * FG_SIN(phi);
      inside = 1.0f;
    }
  } else if (shape == FG_SHAPE_SPLIT || shape == FG_SHAPE_ROUND) {
    /* Curved back face. Positive power = convex (magnifies). */
    int inr = (shape == FG_SHAPE_SPLIT) ? (lx > 0.0f ? 1 : 0) : (r < 1.0f ? 1 : 0);
    if (inr != 0) {
      float k = -p->power * 0.02f * p->size;
      tlx = k * lx;
      tly = k * ly;
      inside = 1.0f;
    }
  } else {
    /* Crystal: Voronoi facets, each with a random tilt. */
    if (r > p->inner) {
      float gx = lx * N;
      float gy = ly * N;
      float bx = FG_FLOOR(gx);
      float by = FG_FLOOR(gy);
      float best = 1.0e9f;
      unsigned int hb = 0u;
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          float qx = bx + (float)i;
          float qy = by + (float)j;
          unsigned int h = ((unsigned int)((int)qx) * 73856093U) ^ ((unsigned int)((int)qy) * 19349663U) ^ ((unsigned int)p->seed * 83492791U);
          float fx = qx + 0.1f + 0.8f * fg_rand(h);
          float fy = qy + 0.1f + 0.8f * fg_rand(h + 1u);
          float d = (fx - gx) * (fx - gx) + (fy - gy) * (fy - gy);
          if (d < best) { best = d; hb = h; }
        }
      }
      float ang = fg_rand(hb + 2u) * 6.2831853f;
      float mag = fg_tan(p->prism) * (0.35f + 0.65f * fg_rand(hb + 3u));
      tlx = mag * FG_COS(ang);
      tly = mag * FG_SIN(ang);
      inside = 1.0f;
    }
  }
  return fg3_make(tlx * ca - tly * sa, tlx * sa + tly * ca, inside);
}

/* Spectral sample weights (6 wavelengths, 430-680 nm); each channel sums to 1. */
FG_DEVICE fg3 fg_spectral(int li)
{
  if (li == 0) return fg3_make(0.05f, 0.00f, 0.50f);
  if (li == 1) return fg3_make(0.00f, 0.16f, 0.42f);
  if (li == 2) return fg3_make(0.00f, 0.53f, 0.08f);
  if (li == 3) return fg3_make(0.28f, 0.31f, 0.00f);
  if (li == 4) return fg3_make(0.47f, 0.00f, 0.00f);
  if (li == 5) return fg3_make(0.20f, 0.00f, 0.00f);
  return fg3_make(1.0f, 1.0f, 1.0f);
}

/* Shade output pixel (x, y), absolute pixel coordinates. */
FG_DEVICE fg4 fg_shade(FG_PRIV const FGParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls, int x, int y)
{
  float W = p->frameW;
  float H = p->frameH;
  float aspect = W / H;
  float zs = 1.0f;
  float zg = p->glassDist;
  float zf = 1.0f / p->focusInv;
  float cx = (p->centerX - 0.5f) * aspect;
  float cy = p->centerY - 0.5f;
  float ca = FG_COS(p->angle);
  float sa = FG_SIN(p->angle);
  int N = p->samples < 1 ? 1 : p->samples;
  int mode = p->edgeMode;
  float fxp = (float)x - p->frameX;
  float fyp = (float)y - p->frameY;

  unsigned int pix = fg_hash((unsigned int)x * 1973u + (unsigned int)y * 9277u + (unsigned int)p->seed * 26699u);
  float rot = fg_rand(pix) * 6.2831853f;
  fg3 L = fg_norm(fg3_make((p->lightX - 0.5f) * aspect, p->lightY - 0.5f, -1.0f));
  float ls = FG_FMAX(p->lightSize, 0.001f);

  float accR = 0.0f, accG = 0.0f, accB = 0.0f, accA = 0.0f;
  float wR = 0.0f, wG = 0.0f, wB = 0.0f;

  for (int i = 0; i < N; i++) {
    unsigned int si = pix + (unsigned int)i * 2654435761u;
    float jx = (N > 1) ? fg_rand(si ^ 0x68bc21ebu) - 0.5f : 0.0f;
    float jy = (N > 1) ? fg_rand(si ^ 0x02e5be93u) - 0.5f : 0.0f;
    float px = ((fxp + 0.5f + jx) / W - 0.5f) * aspect;
    float py = (fyp + 0.5f + jy) / H - 0.5f;

    /* Point on the lens aperture (Vogel disc, rotated per pixel). */
    float rr = p->aperture * FG_SQRT(((float)i + 0.5f) / (float)N);
    float th = (float)i * 2.39996323f + rot;
    float ax = rr * FG_COS(th);
    float ay = rr * FG_SIN(th);

    /* Ray from the aperture point toward this pixel's point on the focus plane. */
    fg3 d = fg_norm(fg3_make(px * zf - ax, py * zf - ay, zf));
    float tg = zg / d.z;
    float gx = ax + d.x * tg;
    float gy = ay + d.y * tg;

    fg3 tl = fg_tilt(p, gx / zg, gy / zg, cx, cy, ca, sa);
    int inG = tl.z > 0.5f ? 1 : 0;

    int li = (p->dispersion > 0.0f) ? (i % 6) : -1;
    float n = p->ior;
    if (li >= 0) {
      float lam = 0.43f + 0.05f * (float)li;
      n += p->dispersion * 0.05f * (1.0f / (lam * lam) - 3.306f);
    }

    fg3 o = d;
    float R = 0.0f;
    float ru = 0.0f, rv = 0.0f;
    float spec = 0.0f;
    int tir = 0;
    if (inG != 0) {
      fg3 d1 = fg_refract(d, fg3_make(0.0f, 0.0f, -1.0f), 1.0f / n);
      fg3 on = fg_norm(fg3_make(tl.x, tl.y, 1.0f));
      fg3 d2 = fg_refract(d1, fg3_make(-on.x, -on.y, -on.z), n);
      if (d2.z <= 0.0001f) tir = 1; else o = d2;

      float cosi = FG_FABS(fg_dot(d, on));
      float r0 = (n - 1.0f) / (n + 1.0f);
      r0 = r0 * r0;
      float q = 1.0f - cosi;
      float fr = r0 + (1.0f - r0) * q * q * q * q * q;
      R = FG_FMIN(1.0f, p->refl * fr * 6.0f);
      if (tir != 0) R = 1.0f;

      /* Reflection off the shaped face, back toward the camera side. */
      float dn = fg_dot(d, on);
      fg3 rd = fg3_make(d.x - 2.0f * dn * on.x, d.y - 2.0f * dn * on.y, d.z - 2.0f * dn * on.z);
      float rz = FG_FMAX(-rd.z, 0.05f);
      ru = 0.5f - (rd.x / rz) / aspect;
      rv = 0.5f + rd.y / rz;
      float cl = fg_dot(rd, L);
      spec = p->lightInt * FG_EXP(-(1.0f - cl) / (ls * ls * 0.5f));
    }

    float t2 = (zs - zg) / o.z;
    float Sx = gx + o.x * t2;
    float Sy = gy + o.y * t2;
    if (inG != 0 && (p->shape == FG_SHAPE_SPLIT || p->shape == FG_SHAPE_ROUND)) {
      /* Diopter: rays through it get their own focus distance; the lens power
         still adds its magnification and colour on top. */
      float t0 = (zs - zg) / d.z;
      float zfd = 1.0f / p->focusInvD;
      fg3 dd = fg_norm(fg3_make(px * zfd - ax, py * zfd - ay, zfd));
      Sx = ax + dd.x * (zs / dd.z) + (Sx - (gx + d.x * t0));
      Sy = ay + dd.y * (zs / dd.z) + (Sy - (gy + d.y * t0));
    }
    float u = Sx / (zs * aspect) + 0.5f;
    float v = Sy / zs + 0.5f;

    fg4 c = fg4_make(0.0f, 0.0f, 0.0f, 0.0f);
    int valid = 1;
    if (mode == FG_EDGE_BLACK && (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)) valid = 0;
    if (tir != 0) valid = 0;
    if (valid != 0) {
      float SX = p->frameX + fg_wrap(u, mode) * W;
      float SY = p->frameY + fg_wrap(v, mode) * H;
      if (inG != 0 && p->useSource != 0)
        c = fg_sample(gls, p->glsX1, p->glsY1, p->glsW, p->glsH, p->glsStride, SX, SY);
      else
        c = fg_sample(src, p->srcX1, p->srcY1, p->srcW, p->srcH, p->srcStride, SX, SY);
    }

    if (inG != 0) {
      /* Highlight boost on what is seen through the glass. */
      float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
      float e = (lum - p->threshold) / 0.25f;
      e = FG_FMAX(0.0f, FG_FMIN(1.0f, e));
      float g = 1.0f + p->boost * e * e * (3.0f - 2.0f * e);
      c.x *= g; c.y *= g; c.z *= g;

      if (R > 0.0f) {
        fg4 rc = fg_sample(src, p->srcX1, p->srcY1, p->srcW, p->srcH, p->srcStride,
                           p->frameX + fg_wrap(ru, FG_EDGE_MIRROR) * W, p->frameY + fg_wrap(rv, FG_EDGE_MIRROR) * H);
        c.x = c.x * (1.0f - R) + (rc.x + spec) * R;
        c.y = c.y * (1.0f - R) + (rc.y + spec) * R;
        c.z = c.z * (1.0f - R) + (rc.z + spec) * R;
        c.w = c.w * (1.0f - R) + R;
      } else {
        c.x += spec; c.y += spec; c.z += spec;
      }
    }

    fg3 w = fg_spectral(li);
    accR += c.x * w.x; wR += w.x;
    accG += c.y * w.y; wG += w.y;
    accB += c.z * w.z; wB += w.z;
    accA += c.w;
  }

  fg4 orig = fg_sample(src, p->srcX1, p->srcY1, p->srcW, p->srcH, p->srcStride, (float)x + 0.5f, (float)y + 0.5f);
  float m = p->mix;
  float rR = accR / FG_FMAX(wR, 1e-6f);
  float rG = accG / FG_FMAX(wG, 1e-6f);
  float rB = accB / FG_FMAX(wB, 1e-6f);
  float rA = accA / (float)N;
  return fg4_make(orig.x + (rR - orig.x) * m, orig.y + (rG - orig.y) * m,
                  orig.z + (rB - orig.z) * m, orig.w + (rA - orig.w) * m);
}

#endif
