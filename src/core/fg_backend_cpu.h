/* CPU backend prelude for fg_core.h */
#ifndef FG_BACKEND_CPU_H
#define FG_BACKEND_CPU_H
#include <math.h>
#define FG_DEVICE static inline
#define FG_GLOBAL
#define FG_PRIV
#define FG_SQRT sqrtf
#define FG_SIN sinf
#define FG_COS cosf
#define FG_ATAN2 atan2f
#define FG_FLOOR floorf
#define FG_FABS fabsf
#define FG_FMAX fmaxf
#define FG_FMIN fminf
#define FG_EXP expf
#include "fg_core.h"
#endif
