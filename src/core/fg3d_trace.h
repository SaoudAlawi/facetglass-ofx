/* FacetGlass 3D core - the per-pixel path tracer, included twice by fg3d_core.h:
 * as fg3d_trace (FG3D_TRACE_CACHE 0) and as fg3d_trace_cache (1), which also
 * records where the pixel's light lands (see "playback cache" in fg3d_core.h).
 * One source, so the two cannot drift; separate functions, so the plain trace
 * pays nothing for the recording. Not include-guarded on purpose. */
/* Traces one pixel: returns its linear RGBA (before Mix and encoding). When
   `aux` is not null it also writes the denoiser guides (FG_AUX_FLOATS). */
FG_DEVICE fgv4 FG3D_TRACE_FN(FG_PCONST const FG3DParams* p, FG_GLOBAL const float* src, FG_GLOBAL const float* gls,
                          FG_GLOBAL const float* tris, FG_GLOBAL const float* nodesF, FG_GLOBAL const int* nodesI,
                          int x, int y, FG_GLOBAL float* aux FG3D_TRACE_EXTRA)
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
  fgv3 opos = fgv(p->objPos[0], p->objPos[1], p->objPos[2]);
  if (p->lensMount != 0) {
    /* Filter on the lens: centred on this pixel's central ray, sized to its cone. */
    float d = p->objPos[2];
    float pcx = ((fx + 0.5f) / W - 0.5f) * aspect, pcy = (fy + 0.5f) / H - 0.5f;
    opos = fgv(pcx * d, pcy * d, d);
    float cone = p->aperture * FG_FABS(1.0f - d / zf);
    s = p->lensFill * FG_FMAX(cone, 0.25f * p->aperture) + 1e-7f;
  }
  float eps = 2e-4f;               /* object-space ray offset */

  unsigned int pix = fg3d_hash((unsigned int)x * 1973u + (unsigned int)y * 9277u + (unsigned int)p->seed * 26699u);
  /* Per-pixel offsets for the progressive (R2 / golden) sequences, from
     interleaved gradient noise so neighbouring pixels' errors cancel visually. */
  float bx = (float)x + (float)(p->seed % 1000) * 7.0f, by = (float)y + (float)(p->seed % 1000) * 3.0f;
  float o1 = fg3d_ign(bx, by), o2 = fg3d_ign(bx + 5.588238f, by + 5.588238f);
  float o3 = fg3d_ign(bx + 11.176476f, by + 11.176476f), o4 = fg3d_ign(bx + 16.764714f, by + 16.764714f);
  float o5 = fg3d_ign(bx + 22.352952f, by + 22.352952f);

  /* Pixels the glass cannot touch: with the footage in focus, if no ray of the
     pixel's lens cone (any aperture point, any jitter) can reach the glass's
     bounding sphere, every sample lands on the footage at this pixel, so it
     passes through untouched. Conservative: a sample ray at depth z deviates
     from the pixel's central ray by at most A |1 - z / zf| + 0.7072 z / H. */
  if (p->boundR > 0.0f && p->focusInv == 1.0f) {
    float cx = ((fx + 0.5f) / W - 0.5f) * aspect * zf, cy = ((fy + 0.5f) / H - 0.5f) * zf;
    fgv3 dc = fgv_norm(fgv(cx, cy, zf));
    fgv3 C = fgv(p->boundC[0], p->boundC[1], p->boundC[2]);
    float R = p->boundR * 1.01f + 1e-4f;
    fgv3 perp = fgv_madd(C, dc, -fgv_dot(C, dc));
    float z0 = FG_FMAX(C.z - R, 0.0f), z1 = C.z + R;
    float dev0 = p->aperture * FG_FABS(1.0f - z0 / zf) + 0.7072f * z0 / H;
    float dev1 = p->aperture * FG_FABS(1.0f - z1 / zf) + 0.7072f * z1 / H;
    float lim = R + FG_FMAX(dev0, dev1);
    if (fgv_dot(perp, perp) > lim * lim) {
      fgv4 c0 = fg3d_lookup(src, p->srcW, p->srcH, p->srcLevels, pixU, pixV, 0.0f);
      if (aux != 0) {
        aux[0] = c0.x; aux[1] = c0.y; aux[2] = c0.z; aux[3] = c0.w;
        aux[4] = 0.0f; aux[5] = 0.0f; aux[6] = 0.0f; aux[7] = 0.0f; aux[8] = 0.0f;
        aux[9] = pixU; aux[10] = pixV; aux[11] = 0.0f;
      }
#if FG3D_TRACE_CACHE
      fg3d_cache_passthrough(cache, pixU, pixV);
#endif
      return c0;
    }
  }

#if FG3D_TRACE_CACHE
  float cs[FG_CACHE_SLOTS * FG_SLOT_ACC];
  int ncs = 0;
  float cL[3];
  cL[0] = 0.0f; cL[1] = 0.0f; cL[2] = 0.0f;
#endif
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
        fgv3 ro = fg3d_rotT(p, fgv_mul(fgv_sub(o, opos), 1.0f / s));
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
        fgv3 hitW = fgv_add(opos, fgv_mul(fg3d_rot(p, hitO), s));
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
            float tq[8];
            fg3d_escape_tap(p, hitW, fg3d_rot(p, fgv_norm(rr)), 1, aspect, pixU, pixV, tq);
            fgv4 rc = fg3d_boost(p, fg3d_tap_colour(p, src, gls, tq));
            float wgt = thr * fr;
            float wr = wgt, wg = wgt, wbl = wgt;
            if (p->absorbOn != 0) { wr *= tR; wg *= tG; wbl *= tB; }
#if FG3D_TRACE_CACHE
            fg3d_cache_add(cs, &ncs, cL, tq, 1, rc, wr * sw.x, wg * sw.y, wbl * sw.z, wgt, W, H);
#endif
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
      float te[8];
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
          fg3d_escape_tap(p, fgv_madd(lp, d, -1e-3f), d, glass, aspect, pixU, pixV, te);
          e = fg3d_tap_colour(p, src, gls, te);
        } else {
          fg3d_escape_tap(p, o, d, glass, aspect, pixU, pixV, te);
          e = fg3d_tap_colour(p, src, gls, te);
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
        fg3d_escape_tap(p, o, d, 1, aspect, pixU, pixV, te);
        e = fg3d_tap_colour(p, src, gls, te);
      }
      if (glass != 0) e = fg3d_boost(p, e);
      {
        float kr = thr, kg = thr, kb = thr;
        if (p->absorbOn != 0) { kr *= tR; kg *= tG; kb *= tB; }
        c.x += e.x * kr; c.y += e.y * kg; c.z += e.z * kb; c.w += e.w * thr;
#if FG3D_TRACE_CACHE
        fg3d_cache_add(cs, &ncs, cL, te, glass != 0 ? 1 : 0, e, kr * sw.x, kg * sw.y, kb * sw.z, thr, W, H);
#endif
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
#if FG3D_TRACE_CACHE
  fg3d_cache_write(cache, cs, ncs, cL, wR, wG, wB, (float)(n > 0 ? n : 1), W, H);
#endif
  return fgv4m(rR, rG, rB, rA);
}

