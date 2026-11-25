// FacetGlass 3D - small double-precision vector helpers for host-side geometry.
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>

namespace fg3d {
namespace geom {

const double kPi = 3.14159265358979323846;

struct V3 {
    double x, y, z;
};
inline V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline V3 operator*(V3 a, double s) { return { a.x * s, a.y * s, a.z * s }; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline double len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a)
{
    double l = len(a);
    return l > 1e-30 ? a * (1.0 / l) : V3{ 0, 0, 1 };
}

}  // namespace geom
}  // namespace fg3d
