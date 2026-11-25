// FacetGlass kernel - ray-traced virtual glass in front of the lens.
// Model: pinhole camera at the origin looking down +z, a thin glass element at
// z = glassDist, the scene (your footage) on a plane at z = 1. Each sample is a
// ray from a point on the lens aperture, aimed at the focus plane, refracted by
// Snell's law through the glass (flat front face, shaped back face), then traced
// to the scene plane where the footage is sampled.

__DEVICE__ unsigned int fg_hash(unsigned int x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__DEVICE__ float fg_rand(unsigned int x)
{
  return (float)(fg_hash(x) & 0x00ffffffU) * (1.0f / 16777216.0f);
}

__DEVICE__ float3 fg_norm(float3 a)
{
  float l = _sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  return to_float3(a.x / l, a.y / l, a.z / l);
}

// Snell refraction. N must face against I. Returns (0,0,0) on total internal reflection.
__DEVICE__ float3 fg_refract(float3 I, float3 N, float eta)
{
  float c = -(I.x * N.x + I.y * N.y + I.z * N.z);
  float k = 1.0f - eta * eta * (1.0f - c * c);
  if (k < 0.0f) return to_float3(0.0f, 0.0f, 0.0f);
  float s = eta * c - _sqrtf(k);
  return to_float3(eta * I.x + s * N.x, eta * I.y + s * N.y, eta * I.z + s * N.z);
}

// 0 clamp, 1 mirror, 2 repeat (3 = black is handled by the caller, then clamped)
__DEVICE__ float fg_wrap(float u, int mode)
{
  if (mode == 1) {
    float m = u - 2.0f * _floor(u * 0.5f);
    return m > 1.0f ? 2.0f - m : m;
  }
  if (mode == 2) return u - _floor(u);
  return _fmaxf(0.0f, _fminf(1.0f, u));
}

__DEVICE__ float fg_tan(float a)
{
  return _sinf(a) / _cosf(a);
}

// Back-face slope (tx, ty) of the glass at screen position (sx, sy), z = 1 if the
// point is on glass, 0 if the ray passes clear.
__DEVICE__ float3 fg_tilt(float sx, float sy, int shape, int facets, int symmetric,
                          float cx, float cy, float ca, float sa, float size, float inner,
                          float prism, float power, int seed)
{
  float dx = sx - cx;
  float dy = sy - cy;
  float lx = (dx * ca + dy * sa) / size;
  float ly = (-dx * sa + dy * ca) / size;
  float r = _sqrtf(lx * lx + ly * ly);
  float N = (float)(facets < 1 ? 1 : facets);
  float tlx = 0.0f, tly = 0.0f, inside = 0.0f;

  if (shape == 0) {
    // Linear prism: stepped wedges, each one more step of tilt than the last.
    if (lx > 0.0f || symmetric != 0) {
      float k = _fminf(_floor(_fabs(lx) * N) + 1.0f, N);
      float sgn = lx < 0.0f ? -1.0f : 1.0f;
      tlx = sgn * fg_tan(prism * k);
      inside = 1.0f;
    }
  } else if (shape == 1) {
    // Radial prism: N facets around a clear centre, each tilted outward.
    if (r > inner) {
      float a = _atan2f(ly, lx);
      float j = _floor(a / 6.2831853f * N + 0.5f);
      float phi = j * 6.2831853f / N;
      float t = fg_tan(prism);
      tlx = t * _cosf(phi);
      tly = t * _sinf(phi);
      inside = 1.0f;
    }
  } else if (shape == 2 || shape == 3) {
    // Split diopter (half disc, straight edge) or round diopter: a curved back face.
    int inr = (shape == 2) ? (lx > 0.0f ? 1 : 0) : (r < 1.0f ? 1 : 0);
    if (inr != 0) {
      // Positive power = convex (magnifies), negative = concave.
      float k = -power * 0.02f * size;
      tlx = k * lx;
      tly = k * ly;
      inside = 1.0f;
    }
  } else {
    // Crystal: Voronoi facets, each with a random tilt.
    if (r > inner) {
      float gx = lx * N;
      float gy = ly * N;
      float bx = _floor(gx);
      float by = _floor(gy);
      float best = 1.0e9f;
      unsigned int hb = 0u;
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          float qx = bx + (float)i;
          float qy = by + (float)j;
          unsigned int h = ((unsigned int)((int)qx) * 73856093U) ^ ((unsigned int)((int)qy) * 19349663U) ^ ((unsigned int)seed * 83492791U);
          float fx = qx + 0.1f + 0.8f * fg_rand(h);
          float fy = qy + 0.1f + 0.8f * fg_rand(h + 1u);
          float d = (fx - gx) * (fx - gx) + (fy - gy) * (fy - gy);
          if (d < best) { best = d; hb = h; }
        }
      }
      float ang = fg_rand(hb + 2u) * 6.2831853f;
      float mag = fg_tan(prism) * (0.35f + 0.65f * fg_rand(hb + 3u));
      tlx = mag * _cosf(ang);
      tly = mag * _sinf(ang);
      inside = 1.0f;
    }
  }
  return to_float3(tlx * ca - tly * sa, tlx * sa + tly * ca, inside);
}

// Spectral sample weights (6 wavelengths, 430-680 nm) so white stays white.
__DEVICE__ float3 fg_spectral(int li)
{
  if (li == 0) return to_float3(0.05f, 0.00f, 0.50f);
  if (li == 1) return to_float3(0.00f, 0.16f, 0.42f);
  if (li == 2) return to_float3(0.00f, 0.53f, 0.08f);
  if (li == 3) return to_float3(0.28f, 0.31f, 0.00f);
  if (li == 4) return to_float3(0.47f, 0.00f, 0.00f);
  if (li == 5) return to_float3(0.20f, 0.00f, 0.00f);
  return to_float3(1.0f, 1.0f, 1.0f);
}

__KERNEL__ void FacetGlassKernel(__CONSTANTREF__ FacetGlassParams *params, __TEXTURE2D__ src, __TEXTURE2D__ glass, __TEXTURE2D_WRITE__ dst)
{
  DEFINE_KERNEL_ITERATORS_XY(x, y);
  if (x >= params->width || y >= params->height) return;

  float W = (float)params->width;
  float H = (float)params->height;
  float aspect = W / H;
  float zs = 1.0f;
  float zg = params->glassDist;
  float zf = 1.0f / params->focusInv;
  float cx = (params->center[0] - 0.5f) * aspect;
  float cy = params->center[1] - 0.5f;
  float ca = _cosf(params->angle);
  float sa = _sinf(params->angle);
  int N = params->samples < 1 ? 1 : params->samples;
  int mode = params->edgeMode;

  unsigned int pix = fg_hash((unsigned int)x * 1973u + (unsigned int)y * 9277u + (unsigned int)params->seed * 26699u);
  float rot = fg_rand(pix) * 6.2831853f;
  float3 L = fg_norm(to_float3((params->light[0] - 0.5f) * aspect, params->light[1] - 0.5f, -1.0f));
  float ls = _fmaxf(params->lightSize, 0.001f);

  float accR = 0.0f, accG = 0.0f, accB = 0.0f, accA = 0.0f;
  float wR = 0.0f, wG = 0.0f, wB = 0.0f;

  for (int i = 0; i < N; i++) {
    unsigned int si = pix + (unsigned int)i * 2654435761u;
    float jx = (N > 1) ? fg_rand(si ^ 0x68bc21ebu) - 0.5f : 0.0f;
    float jy = (N > 1) ? fg_rand(si ^ 0x02e5be93u) - 0.5f : 0.0f;
    float px = (((float)x + 0.5f + jx) / W - 0.5f) * aspect;
    float py = ((float)y + 0.5f + jy) / H - 0.5f;

    // Point on the lens aperture (Vogel disc, rotated per pixel).
    float rr = params->aperture * _sqrtf(((float)i + 0.5f) / (float)N);
    float th = (float)i * 2.39996323f + rot;
    float ax = rr * _cosf(th);
    float ay = rr * _sinf(th);

    // Ray from the aperture point toward this pixel's point on the focus plane.
    float3 d = fg_norm(to_float3(px * zf - ax, py * zf - ay, zf));
    float tg = zg / d.z;
    float gx = ax + d.x * tg;
    float gy = ay + d.y * tg;

    float3 tl = fg_tilt(gx / zg, gy / zg, params->shape, params->facets, params->symmetric,
                        cx, cy, ca, sa, params->size, params->inner,
                        params->prism, params->power, params->seed);
    int inG = tl.z > 0.5f ? 1 : 0;

    int li = (params->dispersion > 0.0f) ? (i % 6) : -1;
    float n = params->ior;
    if (li >= 0) {
      float lam = 0.43f + 0.05f * (float)li;
      n += params->dispersion * 0.05f * (1.0f / (lam * lam) - 3.306f);
    }

    float3 o = d;
    float R = 0.0f;
    float rcR = 0.0f, rcG = 0.0f, rcB = 0.0f;
    float ru = 0.0f, rv = 0.0f;
    float spec = 0.0f;
    int tir = 0;
    if (inG != 0) {
      float3 d1 = fg_refract(d, to_float3(0.0f, 0.0f, -1.0f), 1.0f / n);
      float3 on = fg_norm(to_float3(tl.x, tl.y, 1.0f));
      float3 d2 = fg_refract(d1, to_float3(-on.x, -on.y, -on.z), n);
      if (d2.z <= 0.0001f) tir = 1; else o = d2;

      float cosi = _fabs(d.x * on.x + d.y * on.y + d.z * on.z);
      float r0 = (n - 1.0f) / (n + 1.0f);
      r0 = r0 * r0;
      float q = 1.0f - cosi;
      float fr = r0 + (1.0f - r0) * q * q * q * q * q;
      R = _fminf(1.0f, params->refl * fr * 6.0f);
      if (tir != 0) R = 1.0f;

      // Reflection off the shaped face, back toward the camera side.
      float dn = d.x * on.x + d.y * on.y + d.z * on.z;
      float3 rd = to_float3(d.x - 2.0f * dn * on.x, d.y - 2.0f * dn * on.y, d.z - 2.0f * dn * on.z);
      float rz = _fmaxf(-rd.z, 0.05f);
      ru = 0.5f - (rd.x / rz) / aspect;
      rv = 0.5f + rd.y / rz;
      float cl = rd.x * L.x + rd.y * L.y + rd.z * L.z;
      spec = params->lightInt * _expf(-(1.0f - cl) / (ls * ls * 0.5f));
    }

    float t2 = (zs - zg) / o.z;
    float Sx = gx + o.x * t2;
    float Sy = gy + o.y * t2;
    if (inG != 0 && (params->shape == 2 || params->shape == 3)) {
      // Diopter: rays through it get their own focus distance; the lens power
      // still adds its magnification and colour on top.
      float t0 = (zs - zg) / d.z;
      float zfd = 1.0f / params->focusInvD;
      float3 dd = fg_norm(to_float3(px * zfd - ax, py * zfd - ay, zfd));
      Sx = ax + dd.x * (zs / dd.z) + (Sx - (gx + d.x * t0));
      Sy = ay + dd.y * (zs / dd.z) + (Sy - (gy + d.y * t0));
    }
    float u = Sx / (zs * aspect) + 0.5f;
    float v = Sy / zs + 0.5f;

    float4 c = to_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int valid = 1;
    if (mode == 3 && (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)) valid = 0;
    if (tir != 0) valid = 0;
    float uu = fg_wrap(u, mode);
    float vv = fg_wrap(v, mode);
    if (valid != 0) {
      if (inG != 0 && params->useSource != 0) c = _tex2DVecN(glass, uu, vv, 15);
      else c = _tex2DVecN(src, uu, vv, 15);
    }

    if (inG != 0) {
      // Highlight boost on what is seen through the glass.
      float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
      float e = (lum - params->threshold) / 0.25f;
      e = _fmaxf(0.0f, _fminf(1.0f, e));
      float g = 1.0f + params->boost * e * e * (3.0f - 2.0f * e);
      c.x *= g; c.y *= g; c.z *= g;

      if (R > 0.0f) {
        float4 rc = _tex2DVecN(src, fg_wrap(ru, 1), fg_wrap(rv, 1), 15);
        rcR = rc.x + spec; rcG = rc.y + spec; rcB = rc.z + spec;
        c.x = c.x * (1.0f - R) + rcR * R;
        c.y = c.y * (1.0f - R) + rcG * R;
        c.z = c.z * (1.0f - R) + rcB * R;
        c.w = c.w * (1.0f - R) + R;
      } else {
        c.x += spec; c.y += spec; c.z += spec;
      }
    }

    float3 w = fg_spectral(li);
    accR += c.x * w.x; wR += w.x;
    accG += c.y * w.y; wG += w.y;
    accB += c.z * w.z; wB += w.z;
    accA += c.w;
  }

  float4 orig = _tex2DVecN(src, ((float)x + 0.5f) / W, ((float)y + 0.5f) / H, 15);
  float m = params->mix;
  float4 res = to_float4(accR / _fmaxf(wR, 1e-6f), accG / _fmaxf(wG, 1e-6f), accB / _fmaxf(wB, 1e-6f), accA / (float)N);
  res = to_float4(orig.x + (res.x - orig.x) * m, orig.y + (res.y - orig.y) * m,
                  orig.z + (res.z - orig.z) * m, orig.w + (res.w - orig.w) * m);
  _tex2DVec4Write(dst, x, y, res);
}
