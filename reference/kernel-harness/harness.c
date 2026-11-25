// CPU harness: defines the Fusion DCTL macros and runs the kernel per pixel.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float x, y; } float2;
typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;
static inline float3 to_float3(float x, float y, float z) { float3 r = {x, y, z}; return r; }
static inline float4 to_float4(float x, float y, float z, float w) { float4 r = {x, y, z, w}; return r; }

typedef struct { int w, h; float *d; } Tex;

#define __KERNEL__
#define __DEVICE__ static inline
#define __CONSTANTREF__ const
#define __TEXTURE2D__ Tex *
#define __TEXTURE2D_WRITE__ Tex *
static int g_x, g_y;
#define DEFINE_KERNEL_ITERATORS_XY(a, b) int a = g_x; int b = g_y

#define _sqrtf sqrtf
#define _sinf sinf
#define _cosf cosf
#define _atan2f atan2f
#define _floor floorf
#define _fabs fabsf
#define _fmaxf fmaxf
#define _fminf fminf
#define _expf expf

static float4 tex_sample(Tex *t, float u, float v) {
  float fx = u * t->w - 0.5f, fy = v * t->h - 0.5f;
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float ax = fx - x0, ay = fy - y0;
  float out[4] = {0, 0, 0, 0};
  for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
    int xx = x0 + i, yy = y0 + j;
    if (xx < 0) xx = 0; if (yy < 0) yy = 0;
    if (xx >= t->w) xx = t->w - 1; if (yy >= t->h) yy = t->h - 1;
    float wgt = (i ? ax : 1 - ax) * (j ? ay : 1 - ay);
    for (int c = 0; c < 4; c++) out[c] += wgt * t->d[(yy * t->w + xx) * 4 + c];
  }
  return to_float4(out[0], out[1], out[2], out[3]);
}
static void tex_write(Tex *t, int x, int y, float4 c) {
  float *p = &t->d[(y * t->w + x) * 4];
  p[0] = c.x; p[1] = c.y; p[2] = c.z; p[3] = c.w;
}
#define _tex2DVecN(t, u, v, m) tex_sample(t, u, v)
#define _tex2DVec4Write(t, x, y, c) tex_write(t, x, y, c)

typedef struct {
#include "params.h"
} FacetGlassParams;

#include "kernel.c"

int main(int argc, char **argv) {
  // usage: harness in.raw out.raw W H key=value...
  int W = atoi(argv[3]), H = atoi(argv[4]);
  Tex src = {W, H, malloc(sizeof(float) * W * H * 4)};
  Tex dst = {W, H, malloc(sizeof(float) * W * H * 4)};
  FILE *f = fopen(argv[1], "rb"); fread(src.d, sizeof(float), W * H * 4, f); fclose(f);
  FacetGlassParams p; memset(&p, 0, sizeof p);
  p.width = W; p.height = H;
  for (int a = 5; a < argc; a++) {
    char k[64]; double v, v2 = 0;
    int n = sscanf(argv[a], "%63[^=]=%lf,%lf", k, &v, &v2);
    if (n < 2) continue;
#define I(name) if (!strcmp(k, #name)) p.name = (int)v;
#define F(name) if (!strcmp(k, #name)) p.name = (float)v;
#define P(name) if (!strcmp(k, #name)) { p.name[0] = (float)v; p.name[1] = (float)v2; }
    I(shape) I(facets) I(symmetric) I(samples) I(edgeMode) I(useSource) I(seed)
    P(center) F(angle) F(size) F(inner) F(prism) F(power) F(ior) F(dispersion)
    F(aperture) F(focusInv) F(focusInvD) F(glassDist) F(boost) F(threshold) F(refl) P(light)
    F(lightSize) F(lightInt) F(mix)
  }
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    g_x = x; g_y = y;
    FacetGlassKernel(&p, &src, &src, &dst);
  }
  f = fopen(argv[2], "wb"); fwrite(dst.d, sizeof(float), W * H * 4, f); fclose(f);
  return 0;
}
