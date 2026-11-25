// Renders through fg_core.h (CPU backend) with the same params as the Fuse harness,
// and compares to the Fuse kernel's output to prove the port is exact.
#include "../src/core/fg_backend_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
int main(int argc, char** argv) {
  int W = atoi(argv[3]), H = atoi(argv[4]);
  std::vector<float> src(W * H * 4), dst(W * H * 4);
  FILE* f = fopen(argv[1], "rb"); if (fread(src.data(), 4, src.size(), f) != src.size()) return 2; fclose(f);
  FGParams p; memset(&p, 0, sizeof p);
  p.frameX = 0; p.frameY = 0; p.frameW = (float)W; p.frameH = (float)H;
  p.srcX1 = p.glsX1 = 0; p.srcY1 = p.glsY1 = 0; p.srcW = p.glsW = W; p.srcH = p.glsH = H; p.srcStride = p.glsStride = W * 4;
  for (int a = 5; a < argc; a++) {
    char k[64]; double v = 0, v2 = 0;
    if (sscanf(argv[a], "%63[^=]=%lf,%lf", k, &v, &v2) < 2) continue;
    std::string s(k);
#define I(n, m) if (s == #n) p.m = (int)v;
#define F(n, m) if (s == #n) p.m = (float)v;
    I(shape, shape) I(facets, facets) I(symmetric, symmetric) I(samples, samples) I(edgeMode, edgeMode)
    I(useSource, useSource) I(seed, seed) F(angle, angle) F(size, size) F(inner, inner) F(prism, prism)
    F(power, power) F(ior, ior) F(dispersion, dispersion) F(aperture, aperture) F(focusInv, focusInv)
    F(focusInvD, focusInvD) F(glassDist, glassDist) F(boost, boost) F(threshold, threshold) F(refl, refl)
    F(lightSize, lightSize) F(lightInt, lightInt) F(mix, mix)
    if (s == "center") { p.centerX = (float)v; p.centerY = (float)v2; }
    if (s == "light") { p.lightX = (float)v; p.lightY = (float)v2; }
  }
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    fg4 c = fg_shade(&p, src.data(), src.data(), x, y);
    float* o = &dst[(y * W + x) * 4]; o[0] = c.x; o[1] = c.y; o[2] = c.z; o[3] = c.w;
  }
  f = fopen(argv[2], "wb"); fwrite(dst.data(), 4, dst.size(), f); fclose(f);
}
