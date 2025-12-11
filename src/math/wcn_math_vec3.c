#include "WCN/WCN_Math.h"
#include "wcn_math_internal.h"

// BEGIN Vec3

WMATH_TYPE(Vec3) WMATH_CREATE(Vec3)(WMATH_CREATE_TYPE(Vec3) vec3_c) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = WMATH_OR_ELSE_ZERO(vec3_c.v_x);
  vec3.v[1] = WMATH_OR_ELSE_ZERO(vec3_c.v_y);
  vec3.v[2] = WMATH_OR_ELSE_ZERO(vec3_c.v_z);
  return vec3;
}

// copy
WMATH_TYPE(Vec3)
WMATH_COPY(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0];
  vec3.v[1] = a.v[1];
  vec3.v[2] = a.v[2];
  return vec3;
}

// set
WMATH_TYPE(Vec3)
WMATH_SET(Vec3)(WMATH_TYPE(Vec3) a, float x, float y, float z) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = x;
  vec3.v[1] = y;
  vec3.v[2] = z;
  return vec3;
}

// 0
WMATH_TYPE(Vec3)
WMATH_ZERO(Vec3)() { return (WMATH_TYPE(Vec3)){.v = {0.0f, 0.0f, 0.0f}}; }

// ceil
WMATH_TYPE(Vec3)
WMATH_CEIL(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
// SSE implementation using SSE4.1 _mm_ceil_ps if available, otherwise manual
#ifdef __SSE4_1__
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_res = _mm_ceil_ps(vec_a);

  wcn_store_vec3_partial(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = ceilf(a.v[0]);
  result.v[1] = ceilf(a.v[1]);
  result.v[2] = ceilf(a.v[2]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndpq_f32 (round towards positive infinity)
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_res = vrndpq_f32(vec_a);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float tmp_res_arr_2[4];
  vst1q_f32(tmp_res_arr_2, vec_res);
  result.v[0] = tmp_res_arr_2[0];
  result.v[1] = tmp_res_arr_2[1];
  result.v[2] = tmp_res_arr_2[2];
#else
  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation using wasm_f32x4_ceil
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_ceil(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 0);  // Round towards +inf
  __riscv_vse32_v_f32m1(result.v, vec_res, 3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation using __lsx_vfrintrp_s (Round to Plus Infinity)
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrp_s(vec_a);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = ceilf(a.v[0]);
  result.v[1] = ceilf(a.v[1]);
  result.v[2] = ceilf(a.v[2]);
#endif

  return result;
}

// floor
WMATH_TYPE(Vec3)
WMATH_FLOOR(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
// SSE implementation using SSE4.1 _mm_floor_ps if available, otherwise manual
#ifdef __SSE4_1__
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_res = _mm_floor_ps(vec_a);

  wcn_store_vec3_partial(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = floorf(a.v[0]);
  result.v[1] = floorf(a.v[1]);
  result.v[2] = floorf(a.v[2]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndmq_f32 (round towards negative infinity)
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_res = vrndmq_f32(vec_a);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float tmp_arr_local[4];
  vst1q_f32(tmp_arr_local, vec_res);
  result.v[0] = tmp_arr_local[0];
  result.v[1] = tmp_arr_local[1];
  result.v[2] = tmp_arr_local[2];
#else
  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation using wasm_f32x4_floor
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_floor(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 1);  // Round towards -inf (floor)
  __riscv_vse32_v_f32m1(result.v, vec_res, 3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation using __lsx_vfrintrm_s (Round to Minus Infinity)
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrm_s(vec_a);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = floorf(a.v[0]);
  result.v[1] = floorf(a.v[1]);
  result.v[2] = floorf(a.v[2]);
#endif

  return result;
}

// round
WMATH_TYPE(Vec3)
WMATH_ROUND(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
// SSE implementation using SSE4.1 _mm_round_ps if available, otherwise manual
#ifdef __SSE4_1__
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  // Round to nearest integer (banker's rounding)
  __m128 vec_res =
      _mm_round_ps(vec_a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  wcn_store_vec3_partial(result.v, vec_res);
#else
  // Fallback for older SSE versions
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndnq_f32 (round to nearest)
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_res = vrndnq_f32(vec_a);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation using wasm_f32x4_nearest (round to nearest, ties to even)
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_res = wasm_f32x4_nearest(vec_a);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 2);  // Round to nearest
  __riscv_vse32_v_f32m1(result.v, vec_res, 3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation using __lsx_vfrintrne_s (Round to Nearest Even)
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_res = __lsx_vfrintrne_s(vec_a);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
#endif

  return result;
}

// dot
float WMATH_DOT(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
#if defined(WCN_HAS_FMA)
  // Use FMA (128-bit) to compute products then horizontal add
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 prod = _mm_fmadd_ps(vec_a, vec_b, _mm_setzero_ps());
  return _mm_cvtss_f32(wcn_hadd_ps(prod));
#else
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_mul = _mm_mul_ps(vec_a, vec_b);
  return _mm_cvtss_f32(wcn_hadd_ps(vec_mul));
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper function
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);
  return wcn_hadd_f32(vec_mul);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - multiply and horizontal add
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_mul = wasm_f32x4_mul(vec_a, vec_b);

  // Extract and add each component to get sum of products
  v128_t v = wasm_i32x4_shuffle(vec_mul, vec_mul, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(vec_mul, vec_mul, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - multiply and reduce add for 3 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 3);
  vfloat32m1_t vec_mul = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 3);
  return __riscv_vfredusum_vs_f32m1_f32(vec_mul, __riscv_vfmv_v_f_f32m1(0.0f, 3), 3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - multiply and horizontal add
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_mul = __lsx_vfmul_s(vec_a, vec_b);

  // Horizontally add to get sum of products
  float result_arr[4];
  __lsx_vst(vec_mul, result_arr, 0);
  return result_arr[0] + result_arr[1] + result_arr[2];

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
#endif
}

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  // Use helper cross to avoid manual shuffle mistakes
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = wcn_cross_platform(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = wcn_cross_platform(vec_a, vec_b);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float tmp_res_arr[4];
  vst1q_f32(tmp_res_arr, vec_res);
  result.v[0] = tmp_res_arr[0];
  result.v[1] = tmp_res_arr[1];
  result.v[2] = tmp_res_arr[2];
#else
  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation for cross product
  // Cross product: (a1, a2, a3) × (b1, b2, b3) = (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);

  // Create shuffled vectors: (a2, a3, a1) and (b3, b1, b2)
  v128_t a2_a3_a1_ = wasm_i32x4_shuffle(vec_a, vec_a, 1, 2, 0, 3);
  v128_t b3_b1_b2_ = wasm_i32x4_shuffle(vec_b, vec_b, 2, 0, 1, 3);

  // Create shuffled vectors: (a3, a1, a2) and (b2, b3, b1)
  v128_t a3_a1_a2_ = wasm_i32x4_shuffle(vec_a, vec_a, 2, 0, 1, 3);
  v128_t b2_b3_b1_ = wasm_i32x4_shuffle(vec_b, vec_b, 1, 2, 0, 3);

  // Calculate cross product: (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
  v128_t prod1 = wasm_f32x4_mul(a2_a3_a1_, b3_b1_b2_);
  v128_t prod2 = wasm_f32x4_mul(a3_a1_a2_, b2_b3_b1_);
  v128_t vec_res = wasm_f32x4_sub(prod1, prod2);

  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 3);

  // Cross product: (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
  // Extract components
  float a0 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(va, 0));
  float a1 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(va, 1));
  float a2 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(va, 2));
  float b0 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(vb, 0));
  float b1 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(vb, 1));
  float b2 = __riscv_vfmv_f_s_f32m1(__riscv_vget_v_f32m1x4(vb, 2));

  result.v[0] = a1 * b2 - a2 * b1;
  result.v[1] = a2 * b0 - a0 * b2;
  result.v[2] = a0 * b1 - a1 * b0;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation for cross product
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);

  // Extract components
  float a0 = __lsx_vfrep2vr_w(vec_a)[0];
  float a1 = __lsx_vfrep2vr_w(vec_a)[1];
  float a2 = __lsx_vfrep2vr_w(vec_a)[2];
  float b0 = __lsx_vfrep2vr_w(vec_b)[0];
  float b1 = __lsx_vfrep2vr_w(vec_b)[1];
  float b2 = __lsx_vfrep2vr_w(vec_b)[2];

  result.v[0] = a1 * b2 - a2 * b1;
  result.v[1] = a2 * b0 - a0 * b2;
  result.v[2] = a0 * b1 - a1 * b0;

#else
  // Scalar fallback
  result.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
  result.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
  result.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
#endif

  return result;
}

// length
float WMATH_LENGTH(Vec3)(const WMATH_TYPE(Vec3) v) {
    return sqrtf(WMATH_LENGTH_SQ(Vec3)(v));
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec3)(const WMATH_TYPE(Vec3) v) {
  return WMATH_DOT(Vec3)(v, v);
}

// normalize
WMATH_TYPE(Vec3)
WMATH_NORMALIZE(Vec3)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Vec3) result;
  const float epsilon = wcn_math_get_epsilon();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Optimized SSE implementation using standard inverse square root for better precision
  __m128 vec_v = wcn_load_vec3_partial(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));

  if (len_sq > epsilon * epsilon) {
    const __m128 vec_len_sq = _mm_set1_ps(len_sq);
    // Use standard sqrt for better precision
    const __m128 inv_len = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(vec_len_sq));
    __m128 vec_res = _mm_mul_ps(vec_v, inv_len);
    wcn_store_vec3_partial(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec3)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // Optimized NEON implementation using standard reciprocal square root
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float len_sq = wcn_hadd_f32(vec_squared);

  if (len_sq > epsilon * epsilon) {
    // Use standard sqrt for better precision
    float32x4_t vec_len = vdupq_n_f32(sqrtf(len_sq));
    float32x4_t vec_res = vdivq_f32(vec_v, vec_len);
    wcn_store_vec3_partial(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec3)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);

  // Extract and add each component to get sum of squares
  v128_t v_shuffle1 = wasm_i32x4_shuffle(vec_squared, vec_squared, 0, 1, 0, 1);
  v128_t v_shuffle2 = wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v_shuffle1, v_shuffle2);
  float len_sq = wasm_f32x4_extract_lane(sum, 0);

  if (len_sq > epsilon * epsilon) {
    float inv_len = 1.0f / sqrtf(len_sq);
    v128_t vec_inv_len = wasm_f32x4_splat(inv_len);
    v128_t vec_res = wasm_f32x4_mul(vec_v, vec_inv_len);
    wasm_v128_store(result.v, vec_res);
  } else {
    result = WMATH_ZERO(Vec3)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 3);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 3);
  float len_sq = __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 3), 3);

  if (len_sq > epsilon * epsilon) {
    float inv_len = 1.0f / sqrtf(len_sq);
    vfloat32m1_t vec_inv_len = __riscv_vfmv_v_f_f32m1(inv_len, 3);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_v, vec_inv_len, 3);
    __riscv_vse32_v_f32m1(result.v, vec_res, 3);
  } else {
    result = WMATH_ZERO(Vec3)();
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get sum of squares
  float squared_arr[4];
  __lsx_vst(vec_squared, squared_arr, 0);
  float len_sq = squared_arr[0] + squared_arr[1] + squared_arr[2];

  if (len_sq > epsilon * epsilon) {
    float inv_len = 1.0f / sqrtf(len_sq);
    __m128 vec_inv_len = __lsx_vldrepl_w(&inv_len, 0);
    __m128 vec_res = __lsx_vfmul_s(vec_v, vec_inv_len);
    __lsx_vst(vec_res, result.v, 0);
  } else {
    result = WMATH_ZERO(Vec3)();
  }

#else
  // Optimized scalar fallback using standard sqrt for better precision
  float len_sq = v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2];
  if (len_sq > epsilon * epsilon) {
    float len = sqrtf(len_sq);
    result.v[0] = v.v[0] / len;
    result.v[1] = v.v[1] / len;
    result.v[2] = v.v[2] / len;
  } else {
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
  }
#endif

  return result;
}

// clamp
WMATH_TYPE(Vec3)
WMATH_CLAMP(Vec3)(WMATH_TYPE(Vec3) a, float min_val, float max_val) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);
  __m128 vec_res = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);

  // Store using partial helper
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);
  float32x4_t vec_res = vminq_f32(vmaxq_f32(vec_a, vec_min), vec_max);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_min = wasm_f32x4_splat(min_val);
  v128_t vec_max = wasm_f32x4_splat(max_val);

  // Branchless clamp: min(max(a, min_val), max_val)
  v128_t vec_temp = wasm_f32x4_max(vec_a, vec_min);
  v128_t vec_res = wasm_f32x4_min(vec_temp, vec_max);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_min = __riscv_vfmv_v_f_f32m1(min_val, 3);
  vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(max_val, 3);

  // Clamp: min(max(a, min_val), max_val)
  vfloat32m1_t vec_temp = __riscv_vfmax_vv_f32m1(vec_a, vec_min, 3);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(vec_temp, vec_max, 3);
  __riscv_vse32_v_f32m1(result.v, vec_res, 3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_min = __lsx_vldrepl_w(&min_val, 0);
  __m128 vec_max = __lsx_vldrepl_w(&max_val, 0);

  // Clamp using min/max operations
  __m128 vec_temp = __lsx_vfmax_s(vec_a, vec_min);
  __m128 vec_res = __lsx_vfmin_s(vec_temp, vec_max);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = fminf(fmaxf(a.v[0], min_val), max_val);
  result.v[1] = fminf(fmaxf(a.v[1], min_val), max_val);
  result.v[2] = fminf(fmaxf(a.v[2], min_val), max_val);
#endif

  return result;
}

// +
WMATH_TYPE(Vec3) WMATH_ADD(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 3);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, 3);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, 3);
  __riscv_vse32_v_f32m1(result.v, vec_res, 3);

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
#endif

  return result;
}

WMATH_TYPE(Vec3)
WMATH_ADD_SCALED(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float scalar) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);

#if defined(WCN_HAS_FMA)
  __m128 vec_res = wcn_fma_mul_add_ps(vec_b, vec_scalar, vec_a);
#else
  __m128 vec_scaled = _mm_mul_ps(vec_b, vec_scalar);
  __m128 vec_res = _mm_add_ps(vec_a, vec_scaled);
#endif

  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_scaled = vmulq_f32(vec_b, vec_scalar);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_scaled);

  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_scaled = wasm_f32x4_mul(vec_b, vec_scalar);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_scaled);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4);
  vfloat32m1_t vec_scaled = __riscv_vfmul_vv_f32m1(vec_b, vec_scalar, 4);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_scaled, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0);
  __m128 vec_scaled = __lsx_vfmul_s(vec_b, vec_scalar);
  __m128 vec_res = __lsx_vfadd_s(vec_a, vec_scaled);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + b.v[0] * scalar;
  result.v[1] = a.v[1] + b.v[1] * scalar;
  result.v[2] = a.v[2] + b.v[2] * scalar;
#endif

  return result;
}

// -
WMATH_TYPE(Vec3) WMATH_SUB(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = __lsx_vfsub_s(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] - b.v[0];
  result.v[1] = a.v[1] - b.v[1];
  result.v[2] = a.v[2] - b.v[2];
#endif

  return result;
}

// angle
float WMATH_ANGLE(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b) {
  const float mag_1 = WMATH_LENGTH(Vec3)(a);
  const float mag_2 = WMATH_LENGTH(Vec3)(b);
  const float mag = mag_1 * mag_2;
  const float cosine = mag && WMATH_DOT(Vec3)(a, b) / mag;
  return acosf(cosine);
}

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b) {
  const float ep = WCN_GET_EPSILON();
  return (fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep &&
          fabsf(a.v[2] - b.v[2]) < ep);
}

// =
bool WMATH_EQUALS(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2]);
}

// lerp
WMATH_TYPE(Vec3)
WMATH_LERP(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float t) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(&a.v[0]);
  __m128 vec_b = _mm_loadu_ps(&b.v[0]);
  __m128 vec_t = _mm_set1_ps(t);
  __m128 vec_diff = _mm_sub_ps(vec_b, vec_a);

  // Use FMA if available: a + (b - a) * t
#if defined(WCN_HAS_FMA)
  __m128 vec_res = wcn_fma_mul_add_ps(vec_diff, vec_t, vec_a);
#else
  __m128 vec_res = _mm_add_ps(vec_a, _mm_mul_ps(vec_diff, vec_t));
#endif

  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_b = {b.v[0], b.v[1], b.v[2], 0.0f};
  float32x4_t vec_t = vdupq_n_f32(t);
  float32x4_t vec_diff = vsubq_f32(vec_b, vec_a);
  float32x4_t vec_res = vaddq_f32(vec_a, vmulq_f32(vec_diff, vec_t));

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_t = wasm_f32x4_splat(t);
  v128_t vec_diff = wasm_f32x4_sub(vec_b, vec_a);
  v128_t vec_res = wasm_f32x4_add(vec_a, wasm_f32x4_mul(vec_diff, vec_t));
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_t = __riscv_vfmv_v_f_f32m1(t, 4);
  vfloat32m1_t vec_diff = __riscv_vfsub_vv_f32m1(vec_b, vec_a, 4);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, __riscv_vfmul_vv_f32m1(vec_diff, vec_t, 4), 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_t = __lsx_vldrepl_w(&t, 0);
  __m128 vec_diff = __lsx_vfsub_s(vec_b, vec_a);
  __m128 vec_res = __lsx_vfadd_s(vec_a, __lsx_vfmul_s(vec_diff, vec_t));
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t;
  result.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t;
  result.v[2] = a.v[2] + (b.v[2] - a.v[2]) * t;
#endif

  return result;
}

// lerpV
WMATH_TYPE(Vec3)
WMATH_LERP_V(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, WMATH_TYPE(Vec3) t) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(&a.v[0]);
  __m128 vec_b = _mm_loadu_ps(&b.v[0]);
  __m128 vec_t = _mm_loadu_ps(&t.v[0]);
  __m128 vec_diff = _mm_sub_ps(vec_b, vec_a);
  __m128 vec_res = _mm_add_ps(vec_a, _mm_mul_ps(vec_diff, vec_t));

  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_b = {b.v[0], b.v[1], b.v[2], 0.0f};
  float32x4_t vec_t = {t.v[0], t.v[1], t.v[2], 0.0f};
  float32x4_t vec_diff = vsubq_f32(vec_b, vec_a);
  float32x4_t vec_res = vaddq_f32(vec_a, vmulq_f32(vec_diff, vec_t));

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_t = wcn_load_vec3_partial(t.v);
  v128_t vec_diff = wasm_f32x4_sub(vec_b, vec_a);
  v128_t vec_res = wasm_f32x4_add(vec_a, wasm_f32x4_mul(vec_diff, vec_t));
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_t = wcn_load_vec3_partial(t.v);
  vfloat32m1_t vec_diff = __riscv_vfsub_vv_f32m1(vec_b, vec_a, 4);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, __riscv_vfmul_vv_f32m1(vec_diff, vec_t, 4), 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_t = wcn_load_vec3_partial(t.v);
  __m128 vec_diff = __lsx_vfsub_s(vec_b, vec_a);
  __m128 vec_res = __lsx_vfadd_s(vec_a, __lsx_vfmul_s(vec_diff, vec_t));
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t.v[0];
  result.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t.v[1];
  result.v[2] = a.v[2] + (b.v[2] - a.v[2]) * t.v[2];
#endif

  return result;
}

// fmax
WMATH_TYPE(Vec3)
WMATH_FMAX(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_res = wasm_f32x4_max(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmax_vv_f32m1(vec_a, vec_b, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = __lsx_vfmax_s(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = fmaxf(a.v[0], b.v[0]);
  result.v[1] = fmaxf(a.v[1], b.v[1]);
  result.v[2] = fmaxf(a.v[2], b.v[2]);
#endif

  return result;
}

// fmin
WMATH_TYPE(Vec3)
WMATH_FMIN(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_res = wasm_f32x4_min(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(vec_a, vec_b, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = __lsx_vfmin_s(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = fminf(a.v[0], b.v[0]);
  result.v[1] = fminf(a.v[1], b.v[1]);
  result.v[2] = fminf(a.v[2], b.v[2]);
#endif

  return result;
}

// *
WMATH_TYPE(Vec3) WMATH_MULTIPLY(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * b.v[0];
  result.v[1] = a.v[1] * b.v[1];
  result.v[2] = a.v[2] * b.v[2];
#endif

  return result;
}

// .*
WMATH_TYPE(Vec3) WMATH_MULTIPLY_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_scalar, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * scalar;
  result.v[1] = a.v[1] * scalar;
  result.v[2] = a.v[2] * scalar;
#endif

  return result;
}

// div
WMATH_TYPE(Vec3)
WMATH_DIV(Vec3)
(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_b = wcn_load_vec3_partial(b.v);
  v128_t vec_res = wasm_f32x4_div(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec3_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_a, vec_b, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] / b.v[0];
  result.v[1] = a.v[1] / b.v[1];
  result.v[2] = a.v[2] / b.v[2];
#endif

  return result;
}

// .div
WMATH_TYPE(Vec3)
WMATH_DIV_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar) {
  if (scalar == 0) {
    return WMATH_ZERO(Vec3)();
  }

  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(&a.v[0]);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_div_ps(vec_a, vec_scalar);

  // Extract results using array access
  float temp[4];
  _mm_storeu_ps(temp, vec_res);
  result.v[0] = temp[0];
  result.v[1] = temp[1];
  result.v[2] = temp[2];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_scalar);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_res = wasm_f32x4_div(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_a, vec_scalar, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] / scalar;
  result.v[1] = a.v[1] / scalar;
  result.v[2] = a.v[2] / scalar;
#endif

  return result;
}

// inverse
WMATH_TYPE(Vec3)
WMATH_INVERSE(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t vec_one = wasm_f32x4_splat(1.0f);
  v128_t vec_res = wasm_f32x4_div(vec_one, vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, 4);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_one, vec_a, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_one = __lsx_vldrepl_w(&(const float){1.0f}, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_one, vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = 1.0f / a.v[0];
  result.v[1] = 1.0f / a.v[1];
  result.v[2] = 1.0f / a.v[2];
#endif

  return result;
}

// distance
float WMATH_DISTANCE(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  float sum = _mm_cvtss_f32(wcn_hadd_ps(mul));
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec3_partial(a.v);
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  float sum = wcn_hadd_f32(mul);
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec3_partial(a.v);
  v128_t vb = wcn_load_vec3_partial(b.v);
  v128_t diff = wasm_f32x4_sub(va, vb);
  v128_t mul = wasm_f32x4_mul(diff, diff);

  // Horizontal add to get sum
  v128_t shuf = wasm_i32x4_shuffle(mul, mul, 2, 3, 0, 1);
  v128_t sums = wasm_f32x4_add(mul, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  float sum = wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec3_partial(b.v);
  vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(va, vb, 4);
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(diff, diff, 4);
  float sum = __riscv_vfredusum_vs_f32m1_f32(mul, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = __lsx_vfsub_s(va, vb);
  __m128 mul = __lsx_vfmul_s(diff, diff);

  // Horizontal add to get sum
  __m128 high = __lsx_vshuf4i_w(mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(mul, high);
  return sqrtf(__lsx_vfrep2vr_w(sum)[0]);

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  return sqrtf(dx * dx + dy * dy + dz * dz);
#endif
}

// distanceSquared
float WMATH_DISTANCE_SQ(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec3_partial(a.v);
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  return wcn_hadd_f32(mul);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec3_partial(a.v);
  v128_t vb = wcn_load_vec3_partial(b.v);
  v128_t diff = wasm_f32x4_sub(va, vb);
  v128_t mul = wasm_f32x4_mul(diff, diff);

  // Horizontal add to get sum
  v128_t shuf = wasm_i32x4_shuffle(mul, mul, 2, 3, 0, 1);
  v128_t sums = wasm_f32x4_add(mul, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  return wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec3_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec3_partial(b.v);
  vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(va, vb, 4);
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(diff, diff, 4);
  return __riscv_vfredusum_vs_f32m1_f32(mul, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = __lsx_vfsub_s(va, vb);
  __m128 mul = __lsx_vfmul_s(diff, diff);

  // Horizontal add to get sum
  __m128 high = __lsx_vshuf4i_w(mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(mul, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  return dx * dx + dy * dy + dz * dz;
#endif
}

// negate
WMATH_TYPE(Vec3)
WMATH_NEGATE(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - negate using XOR with sign bit mask and partial
  // helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - negate using multiply with -1.0f
  v128_t vec_a = wcn_load_vec3_partial(a.v);
  v128_t neg_one = wasm_f32x4_splat(-1.0f);
  v128_t vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - negate using multiply with -1.0f
  vfloat32m1_t vec_a = wcn_load_vec3_partial(a.v);
  vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, 4);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - negate using XOR with sign bit mask
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 sign_mask = __lsx_vldrepl_w(&(-0.0f), 0); // Load -0.0f into all lanes
  __m128 vec_res = __lsx_vxor_v(vec_a, sign_mask);
  wcn_store_vec3_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = -a.v[0];
  result.v[1] = -a.v[1];
  result.v[2] = -a.v[2];
#endif

  return result;
}

// random
WMATH_TYPE(Vec3)
WMATH_RANDOM(Vec3)(float scale) {
  WMATH_TYPE(Vec3) vec3;
  float angle = WMATH_RANDOM(float)() * WMATH_2PI;
  float z = WMATH_RANDOM(float)() * 2.0f - 1.0f;
  float z_scale = sqrtf(1.0f - z * z) * scale;
  vec3.v[0] = cosf(angle) * z_scale;
  vec3.v[1] = sinf(angle) * z_scale;
  vec3.v[2] = z * scale;
  return vec3;
}

// setLength
WMATH_TYPE(Vec3)
WMATH_SET_LENGTH(Vec3)(WMATH_TYPE(Vec3) v, float length) {
  return WMATH_MULTIPLY_SCALAR(Vec3)(WMATH_NORMALIZE(Vec3)(v), length);
}

// truncate
WMATH_TYPE(Vec3)
WMATH_TRUNCATE(Vec3)(WMATH_TYPE(Vec3) v, float max_length) {
  if (WMATH_LENGTH(Vec3)(v) > max_length) {
    return WMATH_SET_LENGTH(Vec3)(v, max_length);
  }
  return WMATH_COPY(Vec3)(v);
}

// midpoint
WMATH_TYPE(Vec3)
WMATH_MIDPOINT(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  return WMATH_LERP(Vec3)(a, b, 0.5f);
}

// END Vec3