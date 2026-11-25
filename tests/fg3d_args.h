// Shared key=value parsing for the FacetGlass 3D test tools.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fg3d_scene.h"

namespace fg3d {

inline std::vector<double> nums(const char* s)
{
    std::vector<double> v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtod(s, &e));
        if (e == s) break;
        s = (*e == ',') ? e + 1 : e;
    }
    return v;
}

// Applies one "key=value" argument. Returns false for an unknown key.
inline bool applyArg(UIParams& ui, double& time, const char* arg)
{
    const char* eq = std::strchr(arg, '=');
    if (!eq) return false;
    std::string k(arg, eq - arg);
    std::vector<double> v = nums(eq + 1);
    if (v.empty()) return false;
    const double x = v[0];
    if (k == "shape") ui.shape.shape = (int)x;
    else if (k == "facets") ui.shape.facets = (int)x;
    else if (k == "bothSides") ui.shape.bothSides = x != 0;
    else if (k == "prismAngle") ui.shape.prismAngleDeg = (float)x;
    else if (k == "clearCentre") ui.shape.clearCentre = (float)x;
    else if (k == "power") ui.shape.power = (float)x;
    else if (k == "thickness") ui.shape.thickness = (float)x;
    else if (k == "detail") ui.shape.detail = (int)x;
    else if (k == "center" && v.size() >= 2) { ui.centerX = v[0]; ui.centerY = v[1]; }
    else if (k == "depth") ui.depth = x;
    else if (k == "rotX") ui.rotX = x;
    else if (k == "rotY") ui.rotY = x;
    else if (k == "rotZ") ui.rotZ = x;
    else if (k == "size") ui.size = x;
    else if (k == "ior") ui.ior = x;
    else if (k == "dispersion") ui.dispersion = x;
    else if (k == "fstop") ui.fstop = x;
    else if (k == "focus") ui.focus = x;
    else if (k == "glassFocus") ui.glassFocus = x;
    else if (k == "reflection") ui.reflection = x;
    else if (k == "roughness") ui.roughness = x;
    else if (k == "bevelWidth") ui.bevelWidth = x;
    else if (k == "bevelAngle") ui.bevelAngle = x;
    else if (k == "maxBounces") ui.maxBounces = (int)x;
    else if (k == "minSamples") ui.minSamples = (int)x;
    else if (k == "maxSamples") ui.maxSamples = (int)x;
    else if (k == "noise") ui.noiseThreshold = x;
    else if (k == "world") ui.worldMode = (int)x;
    else if (k == "worldBlur") ui.worldBlur = x;
    else if (k == "feather") ui.feather = x;
    else if (k == "colorSpace") ui.colorSpace = (int)x;
    else if (k == "boost") ui.boost = x;
    else if (k == "threshold") ui.threshold = x;
    else if (k == "mix") ui.mix = x;
    else if (k == "seed") { ui.seed = (int)x; ui.shape.seed = (int)x; }
    else if (k == "wobble") ui.wobble = x;
    else if (k == "wobbleSpeed") ui.wobbleSpeed = x;
    else if (k == "time") time = x;
    else if (k == "light" && v.size() >= 6 && ui.numLights < FG_MAX_LIGHTS) {
        UIParams::Light& L = ui.lights[ui.numLights++];
        L.x = v[0]; L.y = v[1]; L.depth = v[2]; L.size = v[3]; L.kelvin = v[4]; L.intensity = v[5];
    }
    else return false;
    return true;
}

}  // namespace fg3d
