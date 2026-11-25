// FacetGlass 3D - host side: procedural glass shapes, mesh processing (smooth
// groups, bevel edge data), BVH build, and the UI -> FG3DParams conversion shared
// by the OFX plugin and the tests. Plain C++17, no dependencies.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

extern "C" {
#include "fg3d_params.h"
}

namespace fg3d {

enum Shape {
    kFacetPlate = 0,   // flat plate with stepped facets on the back (ghost repeats)
    kWedgePrism,       // classic triangular glass prism bar
    kRadialPrism,      // N-sided disc, conical facets around a clear centre
    kSplitDiopter,     // half of a plano-convex lens with a straight cut edge
    kRoundDiopter,     // plano-convex (or concave) round lens
    kGeoSphere,        // faceted geodesic sphere
    kGem,              // round cut gem: table, crown, girdle, pavilion
    kCrystalCluster,   // several pointed hexagonal crystals
    kBox,
    kPyramid,
    kRod,              // smooth glass cylinder (cylindrical lens)
    kShapeCount
};

const char* shapeName(int shape);

struct Mesh {
    std::vector<float> v;    // xyz per vertex
    std::vector<int> f;      // 3 vertex indices per triangle
    std::vector<int> group;  // per triangle: 0 = flat face, >0 = smooth group
    int addV(float x, float y, float z);
    void tri(int a, int b, int c, int g = 0);
    void quad(int a, int b, int c, int d, int g = 0);  // a-b-c-d counter-clockwise seen from outside
};

struct ShapeParams {
    int shape = kFacetPlate;
    int facets = 3;
    bool bothSides = false;
    float prismAngleDeg = 8.0f;
    float clearCentre = 0.35f;
    float power = 2.0f;
    float thickness = 0.3f;   // relative
    int detail = 1;
    int seed = 1;
    // World units per object unit (see objectScale). Lens curvature is defined in
    // world terms so Diopter Power means the same thing at any size or depth.
    float worldScale = 0.05f;
    float depth = 0.15f;      // object distance from the lens (footage at 1)
};

Mesh makeShape(const ShapeParams& sp);

struct SceneBuffers {
    std::vector<float> tris;    // FG_TRI_FLOATS per triangle, BVH leaf order
    std::vector<float> nodesF;  // FG_NODE_FLOATS per node
    std::vector<int> nodesI;    // FG_NODE_INTS per node
    int numTris() const { return (int)(tris.size() / FG_TRI_FLOATS); }
    int numNodes() const { return (int)(nodesI.size() / FG_NODE_INTS); }
};

// Welds vertices, computes normals (smooth within groups), bevel edge data for
// edges sharper than bevelAngleDeg, and the BVH.
void buildScene(const Mesh& mesh, float bevelAngleDeg, SceneBuffers& out);

// Everything the user controls, in UI units (degrees, f-stops, 0..1 positions).
struct UIParams {
    ShapeParams shape;
    double centerX = 0.5, centerY = 0.5;  // normalised frame position, y up
    double depth = 0.15;                  // distance from the lens; footage at 1
    double rotX = 0, rotY = 0, rotZ = 0;  // degrees: tilt up/down, turn left/right, roll
    double size = 0.6;                    // object diameter as a fraction of frame height
    double ior = 1.5, dispersion = 0.3;
    double fstop = 4.0, focus = 1.0, glassFocus = 1.0;
    double reflection = 1.0, roughness = 0.0;
    double bevelWidth = 0.02, bevelAngle = 20.0;  // width relative to object size
    int maxBounces = 12;
    int minSamples = 16, maxSamples = 64;
    double noiseThreshold = 0.02;
    int worldMode = FG_WORLD_SPHERICAL;
    double worldBlur = 4.0, feather = 0.05;
    int colorSpace = FG_CS_LINEAR;
    double boost = 0.5, threshold = 1.0, mix = 1.0;
    int seed = 1;
    // Virtual lights: position (normalised x, y), depth, size (fraction of frame height),
    // colour temperature (K), intensity (stops: 0 = off).
    struct Light { double x = 0.3, y = 0.75, depth = 0.5, size = 0.1, kelvin = 5600, intensity = 0; };
    Light lights[FG_MAX_LIGHTS];
    int numLights = 0;
    // Hand-held wobble (procedural, from time).
    double wobble = 0.0, wobbleSpeed = 1.0;
};

// World units per object unit for these settings (object radius 1 -> world).
double objectScale(const UIParams& ui);

// Fills the FG3DParams for a frame (in pixels at render scale) and time (frames).
// Image layout fields (srcW/srcH/levels, gls*, useSource) are left to the caller.
void makeParams(const UIParams& ui, float frameX, float frameY, float frameW, float frameH, double time,
                const SceneBuffers& scene, FG3DParams& p);

// Mip levels to build for a w x h image (down to about 8 px).
int mipLevels(int w, int h);
// Total floats for a packed RGBA pyramid.
size_t pyramidFloats(int w, int h, int levels);

// Approximate linear RGB of a black body (normalised to max 1).
void kelvinToRGB(double kelvin, float rgb[3]);

}  // namespace fg3d
