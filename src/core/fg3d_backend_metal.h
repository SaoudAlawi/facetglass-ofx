/* Metal (MSL) backend prelude for fg3d_core.h.
 * Safe maths and no fused multiply-add keep Metal bit-identical to the CPU. */
#ifndef FG3D_BACKEND_METAL_H
#define FG3D_BACKEND_METAL_H
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#define FG_DEVICE static inline
#define FG_GLOBAL device
#define FG_PRIV thread
#define FG_SQRT precise::sqrt
#define FG_SIN precise::sin
#define FG_COS precise::cos
#define FG_ATAN2 precise::atan2
#define FG_FLOOR floor
#define FG_FABS fabs
#define FG_FMAX fmax
#define FG_FMIN fmin
#define FG_EXP precise::exp
#define FG_LOG precise::log
#include "fg3d_core.h"
#endif
