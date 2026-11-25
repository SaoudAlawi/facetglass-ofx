/* FacetGlass 3D - shared parameter block and scene buffer layout.
 * Plain C, all 4-byte fields (float/int), no padding: the same bytes are filled on
 * the host and read by every backend (CPU, CUDA, OpenCL, Metal). */
#ifndef FG3D_PARAMS_H
#define FG3D_PARAMS_H

/* Colour spaces (input decode / output encode). */
#define FG_CS_LINEAR   0
#define FG_CS_REC709   1  /* BT.1886 gamma 2.4 */
#define FG_CS_SRGB     2
#define FG_CS_SLOG3    3
#define FG_CS_LOGC3    4  /* ARRI LogC3, EI 800 */
#define FG_CS_LOGC4    5
#define FG_CS_VLOG     6
#define FG_CS_DAVINCI  7  /* DaVinci Intermediate */

/* What rays see once they leave the frame. */
#define FG_WORLD_SPHERICAL   0  /* blurred, mirrored wrap of the source all around */
#define FG_WORLD_MIRROR      1
#define FG_WORLD_REPEAT      2
#define FG_WORLD_CLAMP       3
#define FG_WORLD_BLACK       4
#define FG_WORLD_TRANSPARENT 5  /* rays turned back see the source at the pixel itself */

#define FG_MAX_LIGHTS 4
#define FG_LIGHT_FLOATS 8      /* x, y, z, radius, r, g, b, unused */

/* Triangle record in the scene buffer (floats per triangle). Triangles are
 * stored in BVH leaf order.
 *   0..2   v0            3..5   e1 = v1 - v0      6..8   e2 = v2 - v0
 *   9..11  n0 (shading)  12..14 n1              15..17 n2
 *   18..20 face normal (geometric, outward)
 *   21..29 bevel normals ne0, ne1, ne2 of the edges opposite v0, v1, v2
 *          (average of the two adjacent face normals; zero = not bevelled)
 *   30..32 heights h0, h1, h2 (distance from vertex i to the opposite edge)
 *   33..35 unused */
#define FG_TRI_FLOATS 36
/* BVH node: floats (bmin xyz, bmax xyz) and ints (leftOrFirst, count).
 * count > 0: leaf with triangles [leftOrFirst, leftOrFirst + count).
 * count == 0: interior, children are leftOrFirst and leftOrFirst + 1. */
#define FG_NODE_FLOATS 6
#define FG_NODE_INTS 2
#define FG_BVH_STACK 48

/* Denoiser buffers (per pixel of the padded render region, tightly packed).
 * Aux: 0..3 linear RGBA, 4 luminance variance of the pixel mean, 5 glass
 * coverage, 6..8 mean first-hit normal, 9..10 mean landing position, 11 landing
 * spread in pixels.
 * CV (colour + variance, ping-ponged between iterations): RGBA, variance. */
#define FG_AUX_FLOATS 12
#define FG_CV_FLOATS 5
#define FG_DENOISE_ITERS 4
#define FG_DENOISE_PAD 30   /* 2 * (1 + 2 + 4 + 8): reach of 4 a-trous passes */

typedef struct FG3DParams {
  /* Output frame (source RoD) in pixels at the current render scale. */
  float frameX, frameY, frameW, frameH;

  /* Linear-light mip pyramids, packed level after level, RGBA float, each level
   * tightly packed (stride = width * 4). Level l is max(1, w >> l) wide. */
  int srcW, srcH, srcLevels, glsW;
  int glsH, glsLevels, useSource, pad0;

  /* Scene */
  int numTris, numNodes, maxBounces, minSamples;
  int maxSamples, seed, worldMode, colorSpace;
  int numLights, spectral, pad1, pad2;

  /* Object transform: world = objPos + objScale * (objRot * p). Row-major 3x3. */
  float objRot[9];
  float objPos[3];
  float objScale;

  /* Optics */
  float ior, dispersion;      /* dispersion: Cauchy-like strength */
  float aperture;             /* lens aperture radius (world units) */
  float focusInv;             /* 1 / focus distance */
  float glassFocusInv;        /* 1 / focus for rays through the glass (diopter focus) */
  float reflect;              /* scales Fresnel reflection (1 = physical) */
  float roughness;            /* reflection roughness, 0 = mirror */
  float bevelWidth;           /* object units */

  /* World */
  float worldBlur;            /* blur of the off-screen world, in mip levels */
  float feather;              /* soft edge of the frame, fraction of frame height */
  float noiseThreshold;       /* adaptive sampling: stop when rel. error below this */

  /* Look */
  float boost, threshold, mix;
  float lightExposure;

  float lights[FG_MAX_LIGHTS * FG_LIGHT_FLOATS];

  /* Denoiser: edge-stopping strength (0 = off). */
  float denoise;
  int pad3;
} FG3DParams;

#endif
