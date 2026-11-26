/* CUDA backend prelude for fg3d_core.h */
#ifndef FG3D_BACKEND_CUDA_CUH
#define FG3D_BACKEND_CUDA_CUH
#define FG_DEVICE static __device__ __forceinline__
#define FG_GLOBAL
#define FG_PRIV
#define FG_PCONST
#define FG_SQRT sqrtf
#define FG_SIN sinf
#define FG_COS cosf
#define FG_ATAN2 atan2f
#define FG_FLOOR floorf
#define FG_FABS fabsf
#define FG_FMAX fmaxf
#define FG_FMIN fminf
#define FG_EXP expf
#define FG_LOG logf
#include "fg3d_core.h"
#endif
