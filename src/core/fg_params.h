/* FacetGlass - shared parameter block (plain C, all 4-byte fields, no padding).
 * The same struct is filled on the host and passed to every backend. */
#ifndef FG_PARAMS_H
#define FG_PARAMS_H

#define FG_SHAPE_LINEAR  0
#define FG_SHAPE_RADIAL  1
#define FG_SHAPE_SPLIT   2
#define FG_SHAPE_ROUND   3
#define FG_SHAPE_CRYSTAL 4

#define FG_EDGE_CLAMP  0
#define FG_EDGE_MIRROR 1
#define FG_EDGE_REPEAT 2
#define FG_EDGE_BLACK  3

typedef struct FGParams {
  /* frame (source region of definition) in pixels at the current render scale */
  float frameX, frameY, frameW, frameH;

  /* source image layout: pixel (x1,y1) is element 0; stride in floats per row */
  int srcX1, srcY1, srcW, srcH, srcStride;
  /* refract-source image layout (same as src when not connected) */
  int glsX1, glsY1, glsW, glsH, glsStride;

  int shape, facets, symmetric, samples, edgeMode, useSource, seed, pad0;

  float centerX, centerY;   /* normalised 0..1, y up */
  float angle;              /* radians */
  float size, inner, prism, power;
  float ior, dispersion;
  float aperture, focusInv, focusInvD, glassDist;
  float boost, threshold;
  float refl, lightX, lightY, lightSize, lightInt;
  float mix;
} FGParams;

#endif
