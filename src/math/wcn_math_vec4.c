#include "wcn_math_internal.h"
#include <string.h>

// BEGIN Vec4

WMATH_TYPE(Vec4) WMATH_CREATE(Vec4)(WMATH_CREATE_TYPE(Vec4) vec4_c) {
  WMATH_TYPE(Vec4) vec4;
  vec4.v[0] = WMATH_OR_ELSE_ZERO(vec4_c.v_x);
  vec4.v[1] = WMATH_OR_ELSE_ZERO(vec4_c.v_y);
  vec4.v[2] = WMATH_OR_ELSE_ZERO(vec4_c.v_z);
  vec4.v[3] = WMATH_OR_ELSE_ZERO(vec4_c.v_w);
  return vec4;
}

WMATH_TYPE(Vec4)
WMATH_SET(Vec4)(WMATH_TYPE(Vec4) vec4, float x, float y, float z, float w) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = x;
  result.v[1] = y;
  result.v[2] = z;
  result.v[3] = w;
  return result;
}

WMATH_TYPE(Vec4) WMATH_COPY(Vec4)(WMATH_TYPE(Vec4) vec4) {
  WMATH_TYPE(Vec4) result;
  memcpy(&result, &vec4, sizeof(WMATH_TYPE(Vec4)));
  return result;
}

// 0
WMATH_TYPE(Vec4) WMATH_ZERO(Vec4)() {
  return (WMATH_TYPE(Vec4)){.v = {0.0f, 0.0f, 0.0f, 0.0f}};
}

// 1
WMATH_TYPE(Vec4) WMATH_IDENTITY(Vec4)() {
  return (WMATH_TYPE(Vec4)){.v = {0.0f, 0.0f, 0.0f, 1.0f}};
}

WMATH_TYPE(Vec4) WMATH_CEIL(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
// SSE implementation using SSE4.1 _mm_ceil_ps if available
#ifdef __SSE4_1__
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_res = _mm_ceil_ps(vec_a);
  _mm_storeu_ps(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = ceilf(a.v[0]);
  result.v[1] = ceilf(a.v[1]);
  result.v[2] = ceilf(a.v[2]);
  result.v[3] = ceilf(a.v[3]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndpq_f32 (round towards positive infinity)
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vrndpq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 0);  // Round towards +inf
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_ceil(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrp_s(vec_a);  // Round to +inf (ceil)
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = ceilf(a.v[0]);
  result.v[1] = ceilf(a.v[1]);
  result.v[2] = ceilf(a.v[2]);
  result.v[3] = ceilf(a.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_FLOOR(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
// SSE implementation using SSE4.1 _mm_floor_ps if available
#ifdef __SSE4_1__
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_res = _mm_floor_ps(vec_a);
  _mm_storeu_ps(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = floorf(a.v[0]);
  result.v[1] = floorf(a.v[1]);
  result.v[2] = floorf(a.v[2]);
  result.v[3] = floorf(a.v[3]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndmq_f32 (round towards negative infinity)
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vrndmq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 1);  // Round towards -inf (floor)
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_floor(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrm_s(vec_a);  // Round to -inf (floor)
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = floorf(a.v[0]);
  result.v[1] = floorf(a.v[1]);
  result.v[2] = floorf(a.v[2]);
  result.v[3] = floorf(a.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_ROUND(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation using SSE4.1 _mm_round_ps if available, otherwise manual
#ifdef __SSE4_1__
  __m128 vec_a = _mm_loadu_ps(a.v);
  // Round to nearest integer (banker's rounding)
  __m128 vec_res = _mm_round_ps(vec_a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  _mm_storeu_ps(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
  result.v[3] = roundf(a.v[3]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndnq_f32 (round to nearest)
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vrndnq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 2);  // Round to nearest
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_nearest(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrne_s(vec_a);  // Round to nearest even
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
  result.v[3] = roundf(a.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4)
WMATH_CLAMP(Vec4)(WMATH_TYPE(Vec4) a, float min_val, float max_val) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);
  __m128 vec_res = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);
  float32x4_t vec_res = vminq_f32(vmaxq_f32(vec_a, vec_min), vec_max);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_min = __riscv_vfmv_v_f_f32m1(min_val, 4);
  vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(max_val, 4);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(__riscv_vfmax_vv_f32m1(vec_a, vec_min, 4), vec_max, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_min = wasm_f32x4_splat(min_val);
  v128_t vec_max = wasm_f32x4_splat(max_val);
  v128_t vec_res = wasm_f32x4_min(wasm_f32x4_max(vec_a, vec_min), vec_max);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_min = __lsx_vldrepl_w(&min_val, 0);
  __m128 vec_max = __lsx_vldrepl_w(&max_val, 0);
  __m128 vec_res = __lsx_vfmin_s(__lsx_vfmax_s(vec_a, vec_min), vec_max);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = fminf(fmaxf(a.v[0], min_val), max_val);
  result.v[1] = fminf(fmaxf(a.v[1], min_val), max_val);
  result.v[2] = fminf(fmaxf(a.v[2], min_val), max_val);
  result.v[3] = fminf(fmaxf(a.v[3], min_val), max_val);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_ADD(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfadd_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + b.v[0];
  result.v[1] = a.v[1] + b.v[1];
  result.v[2] = a.v[2] + b.v[2];
  result.v[3] = a.v[3] + b.v[3];
#endif

  return result;
}

WMATH_TYPE(Vec4)
WMATH_ADD_SCALED(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, float scale) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_scale = _mm_set1_ps(scale);

#if defined(WCN_HAS_FMA)
  // Use FMA if available for better performance
  __m128 vec_res = wcn_fma_mul_add_ps(vec_b, vec_scale, vec_a);
#else
  __m128 vec_scaled = _mm_mul_ps(vec_b, vec_scale);
  __m128 vec_res = _mm_add_ps(vec_a, vec_scaled);
#endif
  _mm_storeu_ps(result.v, vec_res);


#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_scale = vdupq_n_f32(scale);
  float32x4_t vec_res = vmlaq_f32(vec_a, vec_b, vec_scale);  // Multiply and add in one operation
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_scale = __riscv_vfmv_v_f_f32m1(scale, 4);
  vfloat32m1_t vec_scaled = __riscv_vfmul_vv_f32m1(vec_b, vec_scale, 4);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_scaled, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_scale = wasm_f32x4_splat(scale);
  v128_t vec_scaled = wasm_f32x4_mul(vec_b, vec_scale);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_scaled);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_scale = __lsx_vldrepl_w(&scale, 0);
  __m128 vec_scaled = __lsx_vfmul_s(vec_b, vec_scale);
  __m128 vec_res = __lsx_vfadd_s(vec_a, vec_scaled);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + b.v[0] * scale;
  result.v[1] = a.v[1] + b.v[1] * scale;
  result.v[2] = a.v[2] + b.v[2] * scale;
  result.v[3] = a.v[3] + b.v[3] * scale;
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_SUB(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfsub_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] - b.v[0];
  result.v[1] = a.v[1] - b.v[1];
  result.v[2] = a.v[2] - b.v[2];
  result.v[3] = a.v[3] - b.v[3];
#endif

  return result;
}

bool WMATH_EQUALS_APPROXIMATELY(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  float ep = WCN_GET_EPSILON();
  return (fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep &&
          fabsf(a.v[2] - b.v[2]) < ep && fabsf(a.v[3] - b.v[3]) < ep);
}

bool WMATH_EQUALS(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] &&
          a.v[3] == b.v[3]);
}

WMATH_TYPE(Vec4)
WMATH_LERP(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, float t) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = _mm_loadu_ps(a.v);
  __m128 vb = _mm_loadu_ps(b.v);
  __m128 vt = _mm_set1_ps(t);
  __m128 vdiff = _mm_sub_ps(vb, va);
#if defined(WCN_HAS_FMA)
  __m128 vres = wcn_fma_mul_add_ps(vdiff, vt, va);
#else
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
#endif
  _mm_storeu_ps(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vt = __riscv_vfmv_v_f_f32m1(t, 4);
  vfloat32m1_t vdiff = __riscv_vfsub_vv_f32m1(vb, va, 4);
  vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vdiff, vt, 4);
  vres = __riscv_vfadd_vv_f32m1(va, vres, 4);
  __riscv_vse32_v_f32m1(result.v, vres, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t vt = wasm_f32x4_splat(t);
  v128_t vdiff = wasm_f32x4_sub(vb, va);
  v128_t vres = wasm_f32x4_add(va, wasm_f32x4_mul(vdiff, vt));
  wasm_v128_store(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 vt = __lsx_vldrepl_w(&t, 0);
  __m128 vdiff = __lsx_vfsub_s(vb, va);
  __m128 vres = __lsx_vfadd_s(va, __lsx_vfmul_s(vdiff, vt));
  __lsx_vst(vres, result.v, 0);

#else
  result.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t);
  result.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t);
  result.v[2] = WMATH_LERP(float)(a.v[2], b.v[2], t);
  result.v[3] = WMATH_LERP(float)(a.v[3], b.v[3], t);
#endif

  return result;
}

WMATH_TYPE(Vec4)
WMATH_LERP_V(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, WMATH_TYPE(Vec4) t) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = _mm_loadu_ps(a.v);
  __m128 vb = _mm_loadu_ps(b.v);
  __m128 vt = _mm_loadu_ps(t.v);
  __m128 vdiff = _mm_sub_ps(vb, va);
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
  _mm_storeu_ps(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vld1q_f32(t.v);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vt = __riscv_vle32_v_f32m1(t.v, 4);
  vfloat32m1_t vdiff = __riscv_vfsub_vv_f32m1(vb, va, 4);
  vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vdiff, vt, 4);
  vres = __riscv_vfadd_vv_f32m1(va, vres, 4);
  __riscv_vse32_v_f32m1(result.v, vres, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t vt = wasm_v128_load(t.v);
  v128_t vdiff = wasm_f32x4_sub(vb, va);
  v128_t vres = wasm_f32x4_add(va, wasm_f32x4_mul(vdiff, vt));
  wasm_v128_store(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 vt = __lsx_vld(t.v, 0);
  __m128 vdiff = __lsx_vfsub_s(vb, va);
  __m128 vres = __lsx_vfadd_s(va, __lsx_vfmul_s(vdiff, vt));
  __lsx_vst(vres, result.v, 0);

#else
  result.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t.v[0]);
  result.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t.v[1]);
  result.v[2] = WMATH_LERP(float)(a.v[2], b.v[2], t.v[2]);
  result.v[3] = WMATH_LERP(float)(a.v[3], b.v[3], t.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_FMAX(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfmax_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_max(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfmax_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = fmaxf(a.v[0], b.v[0]);
  result.v[1] = fmaxf(a.v[1], b.v[1]);
  result.v[2] = fmaxf(a.v[2], b.v[2]);
  result.v[3] = fmaxf(a.v[3], b.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_FMIN(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_min(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfmin_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = fminf(a.v[0], b.v[0]);
  result.v[1] = fminf(a.v[1], b.v[1]);
  result.v[2] = fminf(a.v[2], b.v[2]);
  result.v[3] = fminf(a.v[3], b.v[3]);
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_MULTIPLY(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * b.v[0];
  result.v[1] = a.v[1] * b.v[1];
  result.v[2] = a.v[2] * b.v[2];
  result.v[3] = a.v[3] * b.v[3];
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_MULTIPLY_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_scalar, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * scalar;
  result.v[1] = a.v[1] * scalar;
  result.v[2] = a.v[2] * scalar;
  result.v[3] = a.v[3] * scalar;
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_DIV(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_div(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_b);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] / b.v[0];
  result.v[1] = a.v[1] / b.v[1];
  result.v[2] = a.v[2] / b.v[2];
  result.v[3] = a.v[3] / b.v[3];
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_DIV_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar) {
  if (scalar == 0) {
    return WMATH_ZERO(Vec4)();
  }

  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_div_ps(vec_a, vec_scalar);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_scalar);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_a, vec_scalar, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_res = wasm_f32x4_div(vec_a, vec_scalar);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_scalar);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] / scalar;
  result.v[1] = a.v[1] / scalar;
  result.v[2] = a.v[2] / scalar;
  result.v[3] = a.v[3] / scalar;
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_INVERSE(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, 4);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_one, vec_a, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_one = wasm_f32x4_splat(1.0f);
  v128_t vec_res = wasm_f32x4_div(vec_one, vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_one = __lsx_vldrepl_w(&1.0f, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_one, vec_a);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = 1.0f / a.v[0];
  result.v[1] = 1.0f / a.v[1];
  result.v[2] = 1.0f / a.v[2];
  result.v[3] = 1.0f / a.v[3];
#endif

  return result;
}

float WMATH_DOT(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - multiply and horizontal add all 4 elements
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_mul = _mm_mul_ps(vec_a, vec_b);

  // Horizontally add to get dot product
  __m128 temp = _mm_hadd_ps(vec_mul, vec_mul);
  temp = _mm_hadd_ps(temp, temp);
  return _mm_cvtss_f32(temp);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - multiply and horizontal add all 4 elements
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);

  // Horizontally add to get dot product
  float32x2_t low = vget_low_f32(vec_mul);
  float32x2_t high = vget_high_f32(vec_mul);
  float32x2_t sum = vadd_f32(low, high);
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - multiply and reduce add all 4 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 4);
  vfloat32m1_t vec_mul = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  return __riscv_vfredusum_vs_f32m1_f32(vec_mul, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - multiply and horizontal add all 4 elements
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_mul = wasm_f32x4_mul(vec_a, vec_b);

  // Extract and add each component to get dot product
  v128_t v = wasm_i32x4_shuffle(vec_mul, vec_mul, 0, 1, 2, 3);
  v128_t v2 = wasm_i32x4_shuffle(vec_mul, vec_mul, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  v = wasm_i32x4_shuffle(sum, sum, 0, 1, 0, 1);
  sum = wasm_f32x4_add(v, sum);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - multiply and horizontal add all 4 elements
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_mul = __lsx_vfmul_s(vec_a, vec_b);

  // Horizontally add to get dot product
  __m128 high = __lsx_vshuf4i_w(vec_mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_mul, high);
  high = __lsx_vshuf4i_w(sum, 0x05);  // 0x05 = 0b00000101 = (0, 0, 1, 1)
  sum = __lsx_vfadd_s(sum, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
#endif
}

float WMATH_LENGTH_SQ(Vec4)(WMATH_TYPE(Vec4) v) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 vec_v = _mm_load_ps(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  __m128 hadd1 = _mm_hadd_ps(vec_squared, vec_squared);
  __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
  return _mm_cvtss_f32(hadd2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t vec_v = vld1q_f32(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float32x2_t sum = vpadd_f32(vget_low_f32(vec_squared), vget_high_f32(vec_squared));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 4);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 4);
  return __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);

  // Extract and add each component to get sum of squares
  v128_t v_temp = wasm_i32x4_shuffle(vec_squared, vec_squared, 0, 1, 2, 3);
  v128_t v2 = wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v_temp, v2);
  v_temp = wasm_i32x4_shuffle(sum, sum, 0, 1, 0, 1);
  sum = wasm_f32x4_add(v_temp, sum);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  high = __lsx_vshuf4i_w(sum, 0x05);  // 0x05 = 0b00000101 = (0, 0, 1, 1)
  sum = __lsx_vfadd_s(sum, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  // Scalar fallback
  return v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2] + v.v[3] * v.v[3];
#endif
}

float WMATH_LENGTH(Vec4)(WMATH_TYPE(Vec4) v) {
    return sqrtf(WMATH_LENGTH_SQ(Vec4)(v));
}

float WMATH_DISTANCE_SQ(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  float dw = a.v[3] - b.v[3];
  return dx * dx + dy * dy + dz * dz + dw * dw;
}

float WMATH_DISTANCE(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  return sqrtf(WMATH_DISTANCE_SQ(Vec4)(a, b));
}

WMATH_TYPE(Vec4) WMATH_NORMALIZE(Vec4)(WMATH_TYPE(Vec4) v) {
  WMATH_TYPE(Vec4) result;
  const float epsilon = wcn_math_get_epsilon();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Optimized SSE implementation using fast inverse square root
  __m128 vec_v = _mm_loadu_ps(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);

  // Horizontally add to get length squared
  __m128 temp = _mm_hadd_ps(vec_squared, vec_squared);
  temp = _mm_hadd_ps(temp, temp);
  float len_sq = _mm_cvtss_f32(temp);

  if (len_sq > epsilon * epsilon) {
    // Use fast inverse square root for better performance
    __m128 len_sq_vec = _mm_set_ss(len_sq);
    __m128 inv_len = wcn_fast_inv_sqrt_ps(len_sq_vec);
    __m128 inv_len_broadcast = _mm_shuffle_ps(inv_len, inv_len, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 vec_res = _mm_mul_ps(vec_v, inv_len_broadcast);
    _mm_storeu_ps(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec4)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // Optimized NEON implementation using fast reciprocal square root
  float32x4_t vec_v = vld1q_f32(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);

  // Horizontally add to get length squared
  float32x2_t low = vget_low_f32(vec_squared);
  float32x2_t high = vget_high_f32(vec_squared);
  float32x2_t sum = vadd_f32(low, high);
  sum = vpadd_f32(sum, sum);
  float len_sq = vget_lane_f32(sum, 0);

  if (len_sq > epsilon * epsilon) {
    // Use NEON reciprocal square root estimate with Newton-Raphson refinement
    float32x2_t len_sq_vec = vdup_n_f32(len_sq);
    float32x2_t inv_len_est = vrsqrte_f32(len_sq_vec);
    // One Newton-Raphson iteration for better accuracy
    float32x2_t inv_len = vmul_f32(inv_len_est, vrsqrts_f32(vmul_f32(len_sq_vec, inv_len_est), inv_len_est));
    float32x4_t inv_len_broadcast = vcombine_f32(inv_len, inv_len);
    float32x4_t vec_res = vmulq_f32(vec_v, inv_len_broadcast);
    vst1q_f32(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec4)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 4);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 4);
  float len_sq = __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    vfloat32m1_t inv_len_vec = __riscv_vfmv_v_f_f32m1(inv_len, 4);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_v, inv_len_vec, 4);
    __riscv_vse32_v_f32m1(result.v, vec_res, 4);
  } else {
    result = WMATH_ZERO(Vec4)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);

  // Extract and add each component to get sum of squares
  v128_t v_temp = wasm_i32x4_shuffle(vec_squared, vec_squared, 0, 1, 2, 3);
  v128_t v2 = wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v_temp, v2);
  v_temp = wasm_i32x4_shuffle(sum, sum, 0, 1, 0, 1);
  sum = wasm_f32x4_add(v_temp, sum);
  float len_sq = wasm_f32x4_extract_lane(sum, 0);

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    v128_t inv_len_vec = wasm_f32x4_splat(inv_len);
    v128_t vec_res = wasm_f32x4_mul(vec_v, inv_len_vec);
    wasm_v128_store(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec4)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get length squared
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  high = __lsx_vshuf4i_w(sum, 0x05);  // 0x05 = 0b00000101 = (0, 0, 1, 1)
  sum = __lsx_vfadd_s(sum, high);
  float len_sq = __lsx_vfrep2vr_w(sum)[0];

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    __m128 inv_len_vec = __lsx_vldrepl_w(&inv_len, 0);
    __m128 vec_res = __lsx_vfmul_s(vec_v, inv_len_vec);
    __lsx_vst(vec_res, result.v, 0);
  } else {
    result = WMATH_ZERO(Vec4)();
  }

#else
  // Optimized scalar fallback using fast inverse square root
  float len_sq = v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2] + v.v[3] * v.v[3];

  if (len_sq > epsilon * epsilon) {
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    result.v[0] = v.v[0] * inv_len;
    result.v[1] = v.v[1] * inv_len;
    result.v[2] = v.v[2] * inv_len;
    result.v[3] = v.v[3] * inv_len;
  } else {
    result = WMATH_ZERO(Vec4)();
  }
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_NEGATE(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - negate using XOR with sign bit mask
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - negate using fused multiply-add with -1.0f
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 4);
  vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, 4);
  __riscv_vse32_v_f32m1(result.v, vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - negate using multiply with -1.0f
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t neg_one = wasm_f32x4_splat(-1.0f);
  v128_t vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - negate using XOR with sign bit mask
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 sign_mask = __lsx_vldrepl_w(&(-0.0f), 0); // Load -0.0f into all lanes
  __m128 vec_res = __lsx_vxor_v(vec_a, sign_mask);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = -a.v[0];
  result.v[1] = -a.v[1];
  result.v[2] = -a.v[2];
  result.v[3] = -a.v[3];
#endif

  return result;
}

WMATH_TYPE(Vec4) WMATH_SET_LENGTH(Vec4)(WMATH_TYPE(Vec4) v, float length) {
  return WMATH_MULTIPLY_SCALAR(Vec4)(WMATH_NORMALIZE(Vec4)(v), length);
}

WMATH_TYPE(Vec4) WMATH_TRUNCATE(Vec4)(WMATH_TYPE(Vec4) v, float max_length) {
  if (WMATH_LENGTH(Vec4)(v) > max_length) {
    return WMATH_SET_LENGTH(Vec4)(v, max_length);
  }
  return WMATH_COPY(Vec4)(v);
}

WMATH_TYPE(Vec4) WMATH_MIDPOINT(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  return WMATH_LERP(Vec4)(a, b, 0.5f);
}

// END Vec4
