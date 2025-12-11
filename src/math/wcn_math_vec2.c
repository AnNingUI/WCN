#include "WCN/WCN_Math.h"
#include "wcn_math_internal.h"
// BEGIN Vec2

// create
WMATH_TYPE(Vec2)
WMATH_CREATE(Vec2)(WMATH_CREATE_TYPE(Vec2) vec2_c) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = WMATH_OR_ELSE_ZERO(vec2_c.v_x);
  vec2.v[1] = WMATH_OR_ELSE_ZERO(vec2_c.v_y);
  return vec2;
}

// set
WMATH_TYPE(Vec2)
WMATH_SET(Vec2)(WMATH_TYPE(Vec2) vec2, float x, float y) {
  vec2.v[0] = x;
  vec2.v[1] = y;
  return vec2;
}

// copy
WMATH_TYPE(Vec2)
WMATH_COPY(Vec2)(WMATH_TYPE(Vec2) vec2) {
  WMATH_TYPE(Vec2) result;
  result.v[0] = vec2.v[0];
  result.v[1] = vec2.v[1];
  return result;
}

// 0
WMATH_TYPE(Vec2)
WMATH_ZERO(Vec2)() {
  return (WMATH_TYPE(Vec2)){
      .v = {0.0f, 0.0f},
  };
}

// 1
WMATH_TYPE(Vec2)
WMATH_IDENTITY(Vec2)() {
  return (WMATH_TYPE(Vec2)){
      .v = {1.0f, 1.0f},
  };
}

// ceil
WMATH_TYPE(Vec2)
WMATH_CEIL(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation using SSE4.1 _mm_ceil_ps if available
#ifdef __SSE4_1__
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_res = _mm_ceil_ps(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);
#else
  // Fallback for older SSE versions
  vec2.v[0] = ceilf(a.v[0]);
  vec2.v[1] = ceilf(a.v[1]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndpq_f32 (round towards positive infinity)
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_res = vrndpq_f32(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation for 2 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 0);  // Round towards +inf
  __riscv_vse32_v_f32m1(vec2.v, vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM
  v128_t vec_a = wasm_v128_load64_zero(a.v);
  v128_t vec_res = wasm_f32x4_ceil(vec_a);
  wasm_v128_store64_lane(vec2.v, vec_res, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Round to Plus Infinity
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_res = __lsx_vfrintrp_s(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#else
  // Scalar fallback
  vec2.v[0] = ceilf(a.v[0]);
  vec2.v[1] = ceilf(a.v[1]);
#endif

  return vec2;
}

// floor
WMATH_TYPE(Vec2)
WMATH_FLOOR(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation using SSE4.1 _mm_floor_ps if available
  #ifdef __SSE4_1__
    __m128 vec_a = wcn_load_vec2_partial(a.v);
    __m128 vec_res = _mm_floor_ps(vec_a);
    wcn_store_vec2_partial(vec2.v, vec_res);
  #else
    // Fallback for older SSE versions
    vec2.v[0] = floorf(a.v[0]);
    vec2.v[1] = floorf(a.v[1]);
  #endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndmq_f32 (round towards negative infinity)
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_res = vrndmq_f32(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation for 2 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 1);  // Round towards -inf (floor)
  __riscv_vse32_v_f32m1(vec2.v, vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM: Direct instruction available
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_res = wasm_f32x4_floor(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX: Round to Minus Infinity
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_res = __lsx_vfrintrm_s(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#else
  // Scalar fallback
  vec2.v[0] = floorf(a.v[0]);
  vec2.v[1] = floorf(a.v[1]);
#endif

  return vec2;
}

// round
WMATH_TYPE(Vec2)
WMATH_ROUND(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation using SSE4.1 _mm_round_ps if available, otherwise manual
#ifdef __SSE4_1__
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  // Round to nearest integer (banker's rounding)
  __m128 vec_res = _mm_round_ps(vec_a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  wcn_store_vec2_partial(vec2.v, vec_res);
#else
  // Fallback for older SSE versions
  vec2.v[0] = roundf(a.v[0]);
  vec2.v[1] = roundf(a.v[1]);
#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using vrndnq_f32 (round to nearest)
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_res = vrndnq_f32(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation for 2 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vec_res = __riscv_vfroundto_f_v_f32m1(vec_a, 2);  // Round to nearest
  __riscv_vse32_v_f32m1(vec2.v, vec_res, 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM Direct instruction available
  // Uses "Round to Nearest, ties to even" which matches SIMD behavior of x86/Neon
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_res = wasm_f32x4_nearest(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX Round to Nearest Even
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_res = __lsx_vfrintrne_s(vec_a);
  wcn_store_vec2_partial(vec2.v, vec_res);

#else
  // Scalar fallback
  vec2.v[0] = roundf(a.v[0]);
  vec2.v[1] = roundf(a.v[1]);
#endif

  return vec2;
}

// clamp
WMATH_TYPE(Vec2)
WMATH_CLAMP(Vec2)(WMATH_TYPE(Vec2) a, float min_val, float max_val) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation with branchless operations
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);

  // Branchless clamp using min/max
  __m128 vec_clamped = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);
  wcn_store_vec2_partial(vec2.v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ARM NEON implementation
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);

  // Branchless clamp: max(min(a, max_val), min_val)
  float32x4_t vec_clamped = vmaxq_f32(vminq_f32(vec_a, vec_max), vec_min);
  wcn_store_vec2_partial(vec2.v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = vsetvl_e32m1(2);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_min = __riscv_vfmv_v_f_f32m1(min_val, vl);
  vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(max_val, vl);

  // Clamp: min(max(a, min_val), max_val)
  vfloat32m1_t vec_temp = __riscv_vfmax_vv_f32m1(vec_a, vec_min, vl);
  vfloat32m1_t vec_clamped = __riscv_vfmin_vv_f32m1(vec_temp, vec_max, vl);
  __riscv_vse32_v_f32m1(vec2.v, vec_clamped, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_min = wasm_f32x4_splat(min_val);
  v128_t vec_max = wasm_f32x4_splat(max_val);

  // Branchless clamp
  v128_t vec_temp = wasm_f32x4_max(vec_a, vec_min);
  v128_t vec_clamped = wasm_f32x4_min(vec_temp, vec_max);
  wcn_store_vec2_partial(vec2.v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_min = __lsx_vreplfr2vr_s(min_val);
  __m128 vec_max = __lsx_vreplfr2vr_s(max_val);

  // Clamp using min/max operations
  __m128 vec_temp = __lsx_vfmax_s(vec_a, vec_min);
  __m128 vec_clamped = __lsx_vfmin_s(vec_temp, vec_max);
  wcn_store_vec2_partial(vec2.v, vec_clamped);
#else
  // Scalar fallback
  vec2.v[0] = WMATH_CLAMP(float)(a.v[0], min_val, max_val);
  vec2.v[1] = WMATH_CLAMP(float)(a.v[1], min_val, max_val);
#endif

  return vec2;
}

// dot
float WMATH_DOT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 mul = _mm_mul_ps(va, vb);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t mul = vmulq_f32(va, vb);
  return wcn_hadd_f32(mul);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - multiply and reduce add for 2 elements
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 2);
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(va, vb, 2);
  return __riscv_vfredusum_vs_f32m1_f32(mul, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - multiply and horizontal add
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t mul = wasm_f32x4_mul(va, vb);

  // Extract and add each component to get dot product
  v128_t v = wasm_i32x4_shuffle(mul, mul, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(mul, mul, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - multiply and horizontal add
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 mul = __lsx_vfmul_s(va, vb);

  // Horizontally add to get dot product
  __m128 high = __lsx_vshuf4i_w(mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(mul, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  return a.v[0] * b.v[0] + a.v[1] * b.v[1];
#endif
}

// add
WMATH_TYPE(Vec2)
WMATH_ADD(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfadd_s(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + b.v[0];
  result.v[1] = a.v[1] + b.v[1];
#endif

  return result;
}

// addScaled
WMATH_TYPE(Vec2)
WMATH_ADD_SCALED(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float scale) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  const __m128 v_scale = _mm_set1_ps(scale);
#if defined(WCN_HAS_FMA)
  __m128 v_res = wcn_fma_mul_add_ps(vb, v_scale, va);
#else
  __m128 v_res = _mm_add_ps(va, _mm_mul_ps(vb, v_scale));
#endif
  wcn_store_vec2_partial(result.v, v_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vres = vmlaq_f32(va, vb, vscale);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t v_scale = __riscv_vfmv_v_f_f32m1(scale, 4); // Broadcast scale to all lanes
  vfloat32m1_t scaled_b = __riscv_vfmul_vv_f32m1(vb, v_scale, 4);
  vfloat32m1_t vres = __riscv_vfadd_vv_f32m1(va, scaled_b, 4);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t v_scale = wasm_f32x4_splat(scale);
  v128_t scaled_b = wasm_f32x4_mul(vb, v_scale);
  v128_t vres = wasm_f32x4_add(va, scaled_b);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 v_scale = __lsx_vldrepl_w(&scale, 0); // Load scale into all lanes
  __m128 scaled_b = __lsx_vfmul_s(vb, v_scale);
  __m128 vres = __lsx_vfadd_s(va, scaled_b);
  wcn_store_vec2_partial(result.v, vres);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + b.v[0] * scale;
  result.v[1] = a.v[1] + b.v[1] * scale;
#endif

  return result;
}

// sub
WMATH_TYPE(Vec2)
WMATH_SUB(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfsub_s(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] - b.v[0];
  result.v[1] = a.v[1] - b.v[1];
#endif

  return result;
}

// angle
float WMATH_ANGLE(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE optimized implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  __m128 va_sq = _mm_mul_ps(va, va);
  __m128 vb_sq = _mm_mul_ps(vb, vb);
  float mag_1_sq = _mm_cvtss_f32(wcn_hadd_ps(va_sq));
  float mag_2_sq = _mm_cvtss_f32(wcn_hadd_ps(vb_sq));

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON optimized implementation
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);

  float32x4_t va_sq = vmulq_f32(va, va);
  float32x4_t vb_sq = vmulq_f32(vb, vb);
  float mag_1_sq = wcn_hadd_f32(va_sq);
  float mag_2_sq = wcn_hadd_f32(vb_sq);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  vfloat32m1_t va_sq = __riscv_vfmul_vv_f32m1(va, va, 4);
  vfloat32m1_t vb_sq = __riscv_vfmul_vv_f32m1(vb, vb, 4);
  float mag_1_sq = __riscv_vfredusum_vs_f32m1_f32(va_sq, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  float mag_2_sq = __riscv_vfredusum_vs_f32m1_f32(vb_sq, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  v128_t va_sq = wasm_f32x4_mul(va, va);
  v128_t vb_sq = wasm_f32x4_mul(vb, vb);
  // Extract and add each component to get sum of squares
  v128_t v = wasm_i32x4_shuffle(va_sq, va_sq, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(va_sq, va_sq, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  float mag_1_sq = wasm_f32x4_extract_lane(sum, 0);

  v = wasm_i32x4_shuffle(vb_sq, vb_sq, 0, 1, 0, 1);
  v2 = wasm_i32x4_shuffle(vb_sq, vb_sq, 2, 3, 2, 3);
  sum = wasm_f32x4_add(v, v2);
  float mag_2_sq = wasm_f32x4_extract_lane(sum, 0);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  __m128 va_sq = __lsx_vfmul_s(va, va);
  __m128 vb_sq = __lsx_vfmul_s(vb, vb);
  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(va_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(va_sq, high);
  float mag_1_sq = __lsx_vfrep2vr_w(sum)[0];

  high = __lsx_vshuf4i_w(vb_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(vb_sq, high);
  float mag_2_sq = __lsx_vfrep2vr_w(sum)[0];

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#else
  // Scalar fallback with safety checks
  const float mag_1_sq = a.v[0] * a.v[0] + a.v[1] * a.v[1];
  const float mag_2_sq = b.v[0] * b.v[0] + b.v[1] * b.v[1];

  if (mag_1_sq < wcn_math_get_epsilon() * wcn_math_get_epsilon() ||
      mag_2_sq < wcn_math_get_epsilon() * wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  const float mag_1 = sqrtf(mag_1_sq);
  const float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);
#endif
}

// equalsApproximately
bool WMATH_EQUALS_APPROXIMATELY(Vec2)(const WMATH_TYPE(Vec2) a,
                                      const WMATH_TYPE(Vec2) b) {
  float ep = wcn_math_get_epsilon();
  return fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep;
}

// equals
bool WMATH_EQUALS(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1]);
}

// Linear Interpolation
WMATH_TYPE(Vec2)
WMATH_LERP(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                 const float t) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vt = _mm_set1_ps(t);
  __m128 vdiff = _mm_sub_ps(vb, va);
#if defined(WCN_HAS_FMA)
  __m128 vres = wcn_fma_mul_add_ps(vdiff, vt, va);
#else
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
#endif
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vt = __riscv_vfmv_v_f_f32m1(t, 4); // Broadcast t to all lanes
  vfloat32m1_t vdiff = __riscv_vfsub_vv_f32m1(vb, va, 4);
  vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vdiff, vt, 4);
  vres = __riscv_vfadd_vv_f32m1(va, vres, 4);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t vt = wasm_f32x4_splat(t);
  v128_t vdiff = wasm_f32x4_sub(vb, va);
  v128_t vres = wasm_f32x4_add(va, wasm_f32x4_mul(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vt = __lsx_vldrepl_w(&t, 0); // Load t into all lanes
  __m128 vdiff = __lsx_vfsub_s(vb, va);
  __m128 vres = __lsx_vfadd_s(va, __lsx_vfmul_s(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#else
  result.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t;
  result.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t;
#endif

  return result;
}

// Linear Interpolation V
WMATH_TYPE(Vec2)
WMATH_LERP_V(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                   const WMATH_TYPE(Vec2) t) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vt = wcn_load_vec2_partial(t.v);
  __m128 vdiff = _mm_sub_ps(vb, va);
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vt = wcn_load_vec2_partial(t.v);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vt = wcn_load_vec2_partial(t.v);
  vfloat32m1_t vdiff = __riscv_vfsub_vv_f32m1(vb, va, 4);
  vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vdiff, vt, 4);
  vres = __riscv_vfadd_vv_f32m1(va, vres, 4);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t vt = wcn_load_vec2_partial(t.v);
  v128_t vdiff = wasm_f32x4_sub(vb, va);
  v128_t vres = wasm_f32x4_add(va, wasm_f32x4_mul(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vt = wcn_load_vec2_partial(t.v);
  __m128 vdiff = __lsx_vfsub_s(vb, va);
  __m128 vres = __lsx_vfadd_s(va, __lsx_vfmul_s(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#else
  result.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t.v[0];
  result.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t.v[1];
#endif

  return result;
}

// f max
WMATH_TYPE(Vec2)
WMATH_FMAX(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmax_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_max(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfmax_s(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = fmaxf(a.v[0], b.v[0]);
  result.v[1] = fmaxf(a.v[1], b.v[1]);
#endif

  return result;
}

// f min
WMATH_TYPE(Vec2)
WMATH_FMIN(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_min(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfmin_s(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = fminf(a.v[0], b.v[0]);
  result.v[1] = fminf(a.v[1], b.v[1]);
#endif

  return result;
}

// multiplyScalar
WMATH_TYPE(Vec2)
WMATH_MULTIPLY_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4); // Broadcast scalar to all lanes
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_scalar, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0); // Load scalar into all lanes
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * scalar;
  result.v[1] = a.v[1] * scalar;
#endif

  return result;
}

// multiply
WMATH_TYPE(Vec2)
WMATH_MULTIPLY(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = a.v[0] * b.v[0];
  result.v[1] = a.v[1] * b.v[1];
#endif

  return result;
}

// divScalar
/**
 * (divScalar) if scalar is 0, returns a zero vector
 */
WMATH_TYPE(Vec2)
WMATH_DIV_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation with branchless operations and unified epsilon/abs logic
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);

  // Use explicit abs mask and wcn_math_get_epsilon()
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_epsilon = _mm_set1_ps(wcn_math_get_epsilon());
  __m128 vec_abs_scalar = _mm_and_ps(vec_scalar, abs_mask);
  __m128 cmp_mask = _mm_cmplt_ps(vec_abs_scalar, vec_epsilon);

  __m128 vec_div = _mm_div_ps(vec_a, vec_scalar);
  __m128 vec_res = wcn_select_ps(cmp_mask, vec_zero, vec_div);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ARM NEON implementation with safety checks
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float epsilon = wcn_math_get_epsilon();
  float32x4_t vec_epsilon = vdupq_n_f32(epsilon);

  // Create absolute value of scalar using bitwise AND with sign bit mask
  int32x4_t int_scalar = vreinterpretq_s32_f32(vec_scalar);
  int32x4_t sign_mask = vdupq_n_s32(0x7FFFFFFF); // Mask to clear sign bit
  int32x4_t abs_int = vandq_s32(int_scalar, sign_mask);
  float32x4_t vec_abs_scalar = vreinterpretq_f32_s32(abs_int);

  // Compare with epsilon
  uint32x4_t cmp_mask = vcltq_f32(vec_abs_scalar, vec_epsilon);

  // Perform division
  float32x4_t vec_div = vdivq_f32(vec_a, vec_scalar);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);

  // Select result based on comparison (branchless)
  float32x4_t vec_res = vbslq_f32(cmp_mask, vec_zero, vec_div);
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation with safety checks
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_scalar = __riscv_vfmv_v_f_f32m1(scalar, 4); // Broadcast scalar to all lanes
  float epsilon = wcn_math_get_epsilon();
  vfloat32m1_t vec_epsilon = __riscv_vfmv_v_f_f32m1(epsilon, 4);

  // Create absolute value of scalar
  vint32m1_t int_scalar = __riscv_vreinterpret_v_f32m1_i32m1(vec_scalar);
  vint32m1_t sign_mask = __riscv_vmv_v_x_i32m1(0x7FFFFFFF, 4); // Mask to clear sign bit
  vint32m1_t abs_int = __riscv_vand_vv_i32m1(int_scalar, sign_mask, 4);
  vfloat32m1_t vec_abs_scalar = __riscv_vreinterpret_v_i32m1_f32m1(abs_int);

  // Compare with epsilon
  vbool32_t cmp_mask = __riscv_vmflt_vv_f32m1_b32(vec_abs_scalar, vec_epsilon, 4);

  // Perform division
  vfloat32m1_t vec_div = __riscv_vfdiv_vv_f32m1(vec_a, vec_scalar, 4);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, 4);

  // Select result based on comparison
  vfloat32m1_t vec_res = __riscv_vmerge_vvm_f32m1(cmp_mask, vec_zero, vec_div, 4); // If cmp_mask is true, use vec_zero; else vec_div
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation with safety checks
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_scalar = wasm_f32x4_splat(scalar);
  float epsilon = wcn_math_get_epsilon();
  v128_t vec_epsilon = wasm_f32x4_splat(epsilon);

  // Create absolute value of scalar using the proper WASM SIMD function
  v128_t vec_abs_scalar = wasm_f32x4_abs(vec_scalar);

  // Compare with epsilon
  v128_t cmp_mask = wasm_f32x4_lt(vec_abs_scalar, vec_epsilon);

  // Perform division
  v128_t vec_div = wasm_f32x4_div(vec_a, vec_scalar);
  v128_t vec_zero = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);

  // Select result based on comparison
  v128_t vec_res = wasm_v128_bitselect(vec_div, vec_zero, cmp_mask); // If mask is 1, select from vec_zero; if mask is 0, select from vec_div
  wcn_store_vec2_partial(vec2.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation with safety checks
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_scalar = __lsx_vldrepl_w(&scalar, 0); // Load scalar into all lanes
  float epsilon = wcn_math_get_epsilon();
  __m128 vec_epsilon = __lsx_vldrepl_w(&epsilon, 0);

  // Create absolute value of scalar
  __m128i int_scalar = __lsx_vreinterpret_v_f32_s32(vec_scalar);
  __m128i sign_mask = __lsx_vreplgr2vr_w(0x7FFFFFFF); // Mask to clear sign bit
  __m128i abs_int = __lsx_vand_v_s32(int_scalar, sign_mask);
  __m128 vec_abs_scalar = __lsx_vreinterpret_v_s32_f32(abs_int);

  // Compare with epsilon
  __m128 cmp_mask = __lsx_vslt_s(vec_abs_scalar, vec_epsilon);

  // Perform division
  __m128 vec_div = __lsx_vfdiv_s(vec_a, vec_scalar);
  __m128 vec_zero = __lsx_vldi(0x0); // Load zero

  // Select result based on comparison (using bitwise operations)
  __m128 masked_zero = __lsx_vand_v(vec_zero, cmp_mask);
  __m128 masked_div = __lsx_vandn_v(vec_div, cmp_mask); // vec_div AND NOT cmp_mask
  __m128 vec_res = __lsx_vor_v(masked_zero, masked_div);
  wcn_store_vec2_partial(vec2.v, vec_res);

#else
  // Scalar implementation with branchless operations
  vec2.v[0] = wcn_safe_div_float(a.v[0], scalar);
  vec2.v[1] = wcn_safe_div_float(a.v[1], scalar);
#endif

  return vec2;
}

// div
WMATH_TYPE(Vec2)
WMATH_DIV(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vres = _mm_div_ps(va, vb);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vres = vdivq_f32(va, vb);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vres = __riscv_vfdiv_vv_f32m1(va, vb, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t vres = wasm_f32x4_div(va, vb);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vres = __lsx_vfdiv_s(va, vb);
  wcn_store_vec2_partial(result.v, vres);

#else
  result.v[0] = a.v[0] / b.v[0];
  result.v[1] = a.v[1] / b.v[1];
#endif

  return result;
}

// inverse
WMATH_TYPE(Vec2)
WMATH_INVERSE(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, 4); // Broadcast 1.0f to all lanes
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_one, vec_a, 4); // Use full vector length for helper
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_one = wasm_f32x4_splat(1.0f);
  v128_t vec_res = wasm_f32x4_div(vec_one, vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_one = __lsx_vldrepl_w(&(const float){1.0f}, 0); // Load 1.0f into all lanes
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_one);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback with zero division check
  result.v[0] = (a.v[0] != 0.0f) ? 1.0f / a.v[0] : 0.0f;
  result.v[1] = (a.v[1] != 0.0f) ? 1.0f / a.v[1] : 0.0f;
#endif

  return result;
}

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = 0;
  vec3.v[1] = 0;
  // z
  vec3.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
  return vec3;
}

// length
float WMATH_LENGTH(Vec2)(WMATH_TYPE(Vec2) v) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper function
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper function
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float len_sq = wcn_hadd_f32(vec_squared);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - multiply and reduce add for 2 elements
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 2);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 2);
  float len_sq = __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - multiply and horizontal add
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);

  // Extract and add each component to get sum of squares
  v128_t v_shuffle1 = wasm_i32x4_shuffle(vec_squared, vec_squared, 0, 1, 0, 1);
  v128_t v_shuffle2 = wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v_shuffle1, v_shuffle2);
  float len_sq = wasm_f32x4_extract_lane(sum, 0);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - multiply and horizontal add
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  return sqrtf(__lsx_vfrep2vr_w(sum)[0]);

#else
  // Scalar fallback
  return sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1]);
#endif
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(WMATH_TYPE(Vec2) v) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper function
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  return _mm_cvtss_f32(wcn_hadd_ps(vec_squared));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper function
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  return wcn_hadd_f32(vec_squared);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - multiply and reduce add for 2 elements
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 2);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 2);
  return __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - multiply and horizontal add
  v128_t vec_v = wasm_v128_load(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);

  // Extract and add each component to get sum of squares
  v128_t v_shuffle1 = wasm_i32x4_shuffle(vec_squared, vec_squared, 0, 1, 0, 1);
  v128_t v_shuffle2 = wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v_shuffle1, v_shuffle2);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - multiply and horizontal add
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  return v.v[0] * v.v[0] + v.v[1] * v.v[1];
#endif
}

// distance
float WMATH_DISTANCE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  float sum = _mm_cvtss_f32(wcn_hadd_ps(mul));
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  float sum = wcn_hadd_f32(mul);
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 2);
  vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(va, vb, 2);
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(diff, diff, 2);
  float sum = __riscv_vfredusum_vs_f32m1_f32(mul, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t diff = wasm_f32x4_sub(va, vb);
  v128_t mul = wasm_f32x4_mul(diff, diff);

  // Extract and add each component to get sum of squares
  v128_t v = wasm_i32x4_shuffle(mul, mul, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(mul, mul, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  return sqrtf(wasm_f32x4_extract_lane(sum, 0));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 diff = __lsx_vfsub_s(va, vb);
  __m128 mul = __lsx_vfmul_s(diff, diff);

  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(mul, high);
  return sqrtf(__lsx_vfrep2vr_w(sum)[0]);

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return sqrtf(dx * dx + dy * dy);
#endif
}

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  return wcn_hadd_f32(mul);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, 2);
  vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(va, vb, 2);
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(diff, diff, 2);
  return __riscv_vfredusum_vs_f32m1_f32(mul, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t diff = wasm_f32x4_sub(va, vb);
  v128_t mul = wasm_f32x4_mul(diff, diff);

  // Extract and add each component to get sum of squares
  v128_t v = wasm_i32x4_shuffle(mul, mul, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(mul, mul, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  return wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 diff = __lsx_vfsub_s(va, vb);
  __m128 mul = __lsx_vfmul_s(diff, diff);

  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(mul, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(mul, high);
  return __lsx_vfrep2vr_w(sum)[0];

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return dx * dx + dy * dy;
#endif
}

// negate
WMATH_TYPE(Vec2)
WMATH_NEGATE(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - negate using XOR with sign bit mask
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - negate using "v n e g q _ f32"
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - negate using fused multiply-add with -1.0f
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, 2);
  vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, 2);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, 2);
  __riscv_vse32_v_f32m1(result.v, vec_res, 2);

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
#endif

  return result;
}

// random
WMATH_TYPE(Vec2)
WMATH_RANDOM(Vec2)(float scale) {
  WMATH_TYPE(Vec2) vec2;
  float angle = WMATH_RANDOM(float)() * WMATH_2PI;
  vec2.v[0] = cosf(angle) * scale;
  vec2.v[1] = sinf(angle) * scale;
  return vec2;
}

// normalize
WMATH_TYPE(Vec2)
WMATH_NORMALIZE(Vec2)(WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Vec2) vec2;
  const float epsilon = wcn_math_get_epsilon();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Optimized SSE implementation using fast inverse square root
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));

  if (len_sq > epsilon * epsilon) {
    // Use fast inverse square root for better performance
    __m128 len_sq_vec = _mm_set_ss(len_sq);
    __m128 inv_len = wcn_fast_inv_sqrt_ps(len_sq_vec);
    __m128 inv_len_broadcast = _mm_shuffle_ps(inv_len, inv_len, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 vec_res = _mm_mul_ps(vec_v, inv_len_broadcast);
    wcn_store_vec2_partial(vec2.v, vec_res);
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // Optimized NEON implementation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float len_sq = wcn_hadd_f32(vec_squared);

  if (len_sq > epsilon * epsilon) {
    // Use NEON reciprocal square root estimate with Newton-Raphson refinement
    float32x2_t len_sq_vec = vdup_n_f32(len_sq);
    float32x2_t inv_len_est = vrsqrte_f32(len_sq_vec);
    // One Newton-Raphson iteration for better accuracy
    float32x2_t inv_len = vmul_f32(inv_len_est, vrsqrts_f32(vmul_f32(len_sq_vec, inv_len_est), inv_len_est));
    float32x4_t inv_len_broadcast = vcombine_f32(inv_len, inv_len);
    float32x4_t vec_res = vmulq_f32(vec_v, inv_len_broadcast);
    wcn_store_vec2_partial(vec2.v, vec_res);
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 2);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 2);
  float len_sq = __riscv_vfredusum_vs_f32m1_f32(vec_squared, __riscv_vfmv_v_f_f32m1(0.0f, 2), 2);

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    vfloat32m1_t inv_len_vec = __riscv_vfmv_v_f_f32m1(inv_len, 2);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_v, inv_len_vec, 2);
    __riscv_vse32_v_f32m1(vec2.v, vec_res, 2);
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
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
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    v128_t inv_len_vec = wasm_f32x4_splat(inv_len);
    v128_t vec_res = wasm_f32x4_mul(vec_v, inv_len_vec);
    wasm_v128_store(vec2.v, vec_res);
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = __lsx_vld(v.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get length squared
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  float len_sq = __lsx_vfrep2vr_w(sum)[0];

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    __m128 inv_len_vec = __lsx_vldrepl_w(&inv_len, 0);
    __m128 vec_res = __lsx_vfmul_s(vec_v, inv_len_vec);
    __lsx_vst(vec_res, vec2.v, 0);
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
  }

#else
  // Optimized scalar fallback using fast inverse square root
  float len_sq = v.v[0] * v.v[0] + v.v[1] * v.v[1];
  if (len_sq > epsilon * epsilon) {
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    vec2.v[0] = v.v[0] * inv_len;
    vec2.v[1] = v.v[1] * inv_len;
  } else {
    vec2.v[0] = 0.0f;
    vec2.v[1] = 0.0f;
  }
#endif

  return vec2;
}

// rotate
WMATH_TYPE(Vec2)
WMATH_ROTATE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float rad) {
  WMATH_TYPE(Vec2) vec2;
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float s = sinf(rad);
  float c = cosf(rad);
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 v = _mm_sub_ps(va, vb); // [p0, p1, 0, 0]

  __m128 px = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 py = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 sc = _mm_set1_ps(s);
  __m128 cc = _mm_set1_ps(c);

  __m128 rx = _mm_sub_ps(_mm_mul_ps(px, cc), _mm_mul_ps(py, sc));
  __m128 ry = _mm_add_ps(_mm_mul_ps(px, sc), _mm_mul_ps(py, cc));

  float rx_s = _mm_cvtss_f32(rx);
  float ry_s = _mm_cvtss_f32(ry);
  __m128 res = _mm_set_ps(0.0f, 0.0f, ry_s, rx_s);
  // add back center
  res = _mm_add_ps(res, vb);
  wcn_store_vec2_partial(vec2.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float s = sinf(rad);
  float c = cosf(rad);
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t v = vsubq_f32(va, vb);
  float32x4_t px = vdupq_n_f32(vgetq_lane_f32(v, 0));
  float32x4_t py = vdupq_n_f32(vgetq_lane_f32(v, 1));
  float32x4_t sc = vdupq_n_f32(s);
  float32x4_t cc = vdupq_n_f32(c);
  float32x4_t rx = vsubq_f32(vmulq_f32(px, cc), vmulq_f32(py, sc));
  float32x4_t ry = vaddq_f32(vmulq_f32(px, sc), vmulq_f32(py, cc));
  float rx_s = vgetq_lane_f32(rx, 0);
  float ry_s = vgetq_lane_f32(ry, 0);
  float32x4_t res = {rx_s + vgetq_lane_f32(vb, 0), ry_s + vgetq_lane_f32(vb, 1),
                     0.0f, 0.0f};
  wcn_store_vec2_partial(vec2.v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float s = sinf(rad);
  float c = cosf(rad);
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t v = __riscv_vfsub_vv_f32m1(va, vb, 4); // [p0, p1, 0, 0] (using helper function)

  // Extract the x and y components - use first element and second element
  float p0 = __riscv_vfmv_f_s_f32m1(v);  // Extract x component (first lane)
  float temp[4];
  __riscv_vse32_v_f32m1(temp, v, 4);  // Store vector to temp array
  float p1 = temp[1];  // Extract y component (second lane)

  // Create vectors with the components broadcasted
  vfloat32m1_t px = __riscv_vfmv_v_f_f32m1(p0, 4);
  vfloat32m1_t py = __riscv_vfmv_v_f_f32m1(p1, 4);
  vfloat32m1_t sc = __riscv_vfmv_v_f_f32m1(s, 4);
  vfloat32m1_t cc = __riscv_vfmv_v_f_f32m1(c, 4);
  vfloat32m1_t rx = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(px, cc, 4), __riscv_vfmul_vv_f32m1(py, sc, 4), 4);
  vfloat32m1_t ry = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(px, sc, 4), __riscv_vfmul_vv_f32m1(py, cc, 4), 4);

  // Extract the results
  float rx_s = __riscv_vfmv_f_s_f32m1(rx);
  float temp2[4];
  __riscv_vse32_v_f32m1(temp2, ry, 4);  // Store ry vector to temp array
  float ry_s = temp2[1];  // Extract result from first lane

  // Add back center point
  vec2.v[0] = rx_s + b.v[0];
  vec2.v[1] = ry_s + b.v[1];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  float s = sinf(rad);
  float c = cosf(rad);
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t v = wasm_f32x4_sub(va, vb);

  // Extract the x and y components
  float p0 = wasm_f32x4_extract_lane(v, 0);
  float p1 = wasm_f32x4_extract_lane(v, 1);

  // Perform rotation: rx = p0*c - p1*s, ry = p0*s + p1*c
  float rx_s = p0 * c - p1 * s;
  float ry_s = p0 * s + p1 * c;

  // Add back center point
  vec2.v[0] = rx_s + b.v[0];
  vec2.v[1] = ry_s + b.v[1];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float s = sinf(rad);
  float c = cosf(rad);
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 v = __lsx_vfsub_s(va, vb); // [p0, p1, 0, 0]

  // Extract the x and y components using shuffle
  __m128 px = __lsx_vshuf4i_w(v, 0x00); // duplicate x component (lane 0)
  __m128 py = __lsx_vshuf4i_w(v, 0x55); // duplicate y component (lane 1 with pattern 0x55=0b01010101)

  __m128 sc = __lsx_vldrepl_w(&s, 0);
  __m128 cc = __lsx_vldrepl_w(&c, 0);
  __m128 rx = __lsx_vfsub_s(__lsx_vfmul_s(px, cc), __lsx_vfmul_s(py, sc));
  __m128 ry = __lsx_vfadd_s(__lsx_vfmul_s(px, sc), __lsx_vfmul_s(py, cc));

  float rx_s = __lsx_vfrep2vr_w(rx)[0];
  float ry_s = __lsx_vfrep2vr_w(ry)[0];  // Get from first element as both lanes are the same

  // Add back center
  vec2.v[0] = rx_s + b.v[0];
  vec2.v[1] = ry_s + b.v[1];

#else
  // Scalar fallback
  float p0 = a.v[0] - b.v[0];
  float p1 = a.v[1] - b.v[1];
  float s = sinf(rad);
  float c = cosf(rad);
  vec2.v[0] = p0 * c - p1 * s + b.v[0];
  vec2.v[1] = p0 * s + p1 * c + b.v[1];
#endif
  return vec2;
}

// set length
WMATH_TYPE(Vec2)
WMATH_SET_LENGTH(Vec2)(WMATH_TYPE(Vec2) a, float length) {
  return WMATH_MULTIPLY_SCALAR(Vec2)(WMATH_NORMALIZE(Vec2)(a), length);
}

// truncate
WMATH_TYPE(Vec2)
WMATH_TRUNCATE(Vec2)(WMATH_TYPE(Vec2) a, float length) {
  if (WMATH_LENGTH(Vec2)(a) > length) {
    return WMATH_SET_LENGTH(Vec2)(a, length);
  }
  return WMATH_COPY(Vec2)(a);
}

// midpoint
WMATH_TYPE(Vec2)
WMATH_MIDPOINT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  return WMATH_LERP(Vec2)(a, b, 0.5f);
}

// END Vec2
