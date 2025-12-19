#include "WCN/WCN_Math.h"
#include "wcn_math_internal.h"
#include "WCN/WCN_PLATFORM_MACROS.h"
#include <math.h>
// BEGIN Quat

// 0
WMATH_TYPE(Quat)
WMATH_ZERO(Quat)(void) {
  return (WMATH_TYPE(Quat)){
      .v = {0.0f, 0.0f, 0.0f, 0.0f},
  };
}

// 1
WMATH_TYPE(Quat)
WMATH_IDENTITY(Quat)(void) {
  return (WMATH_TYPE(Quat)){
      .v = {0.0f, 0.0f, 0.0f, 1.0f},
  };
}

WMATH_TYPE(Quat)
WMATH_CREATE(Quat)(WMATH_CREATE_TYPE(Quat) c) {
  WMATH_TYPE(Quat) result;
  result.v[0] = WMATH_OR_ELSE_ZERO(c.v_x);
  result.v[1] = WMATH_OR_ELSE_ZERO(c.v_y);
  result.v[2] = WMATH_OR_ELSE_ZERO(c.v_z);
  result.v[3] = WMATH_OR_ELSE_ZERO(c.v_w);
  return result;
}

WMATH_TYPE(Quat)
WMATH_SET(Quat)(WMATH_TYPE(Quat) a, float x, float y, float z, float w) {
  a.v[0] = x;
  a.v[1] = y;
  a.v[2] = z;
  a.v[3] = w;
  return a;
}

WMATH_TYPE(Quat)
WMATH_COPY(Quat)(WMATH_TYPE(Quat) a) {
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = a.v[0],
      .v_y = a.v[1],
      .v_z = a.v[2],
      .v_w = a.v[3],
  });
}

float WMATH_DOT(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper function
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_mul = _mm_mul_ps(vec_a, vec_b);
  return _mm_cvtss_f32(wcn_hadd_ps(vec_mul));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper function
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);
  return wcn_hadd_f32(vec_mul);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_mul = wasm_f32x4_mul(vec_a, vec_b);
  return wasm_f32x4_extract_lane(wasm_f32x4_add(wasm_f32x4_add(vec_mul, wasm_i32x4_shuffle(vec_mul, vec_mul, 1, 0, 3, 2)), wasm_i32x4_shuffle(vec_mul, vec_mul, 2, 3, 0, 1)), 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, vl);
  vfloat32m1_t vec_mul = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
  vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_mul, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  return __riscv_vfmv_f_s_f32m1_f32(vec_sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vld(b.v, 0);
  __m128 vec_mul = __lsx_vfmul_s(vec_a, vec_b);
  __m128 sum1 = __lsx_vfadd_s(__lsx_vpickve2gr_w(vec_mul, 0), __lsx_vpickve2gr_w(vec_mul, 1));
  __m128 sum2 = __lsx_vfadd_s(__lsx_vpickve2gr_w(vec_mul, 2), __lsx_vpickve2gr_w(vec_mul, 3));
  return __lsx_vfadd_s(sum1, sum2);

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
#endif
}

WMATH_TYPE(Quat)
WMATH_LERP(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
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
  // NEON implementation
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);
  v128_t vt = wasm_f32x4_splat(t);
  v128_t vdiff = wasm_f32x4_sub(vb, va);
  v128_t vres = wasm_f32x4_add(va, wasm_f32x4_mul(vdiff, vt));
  wasm_v128_store(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, vl);
  vfloat32m1_t vt = __riscv_vfmv_v_f_f32m1(t, vl);
  vfloat32m1_t vdiff = __riscv_vfsub_vv_f32m1(vb, va, vl);
  vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(vdiff, vt, vl);
  vres = __riscv_vfadd_vv_f32m1(va, vres, vl);
  __riscv_vse32_v_f32m1(result.v, vres, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);
  __m128 vt = __lsx_vldrepl_w(&t, 0);
  __m128 vdiff = __lsx_vfsub_s(vb, va);
  __m128 vres = __lsx_vfmul_s(vdiff, vt);
  vres = __lsx_vfadd_s(va, vres);
  __lsx_vst(vres, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t;
  result.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t;
  result.v[2] = a.v[2] + (b.v[2] - a.v[2]) * t;
  result.v[3] = a.v[3] + (b.v[3] - a.v[3]) * t;
#endif

  return result;
}

// slerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, slerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 va = _mm_loadu_ps(a.v);
  __m128 vb = _mm_loadu_ps(b.v);

  // Calculate dot product: a_x * b_x + a_y * b_y + a_z * b_z + a_w * b_w
  __m128 mul = _mm_mul_ps(va, vb);
  __m128 dot = _mm_hadd_ps(mul, mul);
  dot = _mm_hadd_ps(dot, dot);
  float cosOmega = _mm_cvtss_f32(dot);

  __m128 vb_orig = vb;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    // Negate vb: vb = -vb
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    vb = _mm_xor_ps(vb, sign_mask);
  }

  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    float scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    float scale_1 = sinf(t * omega) / sinOmega;

    // Perform scaled addition: result = scale_0 * va + scale_1 * vb
    __m128 vscale_0 = _mm_set1_ps(scale_0);
    __m128 vscale_1 = _mm_set1_ps(scale_1);
#if defined(WCN_HAS_FMA)
    __m128 vres = wcn_fma_mul_add_ps(vb, vscale_1, _mm_mul_ps(va, vscale_0));
#else
    __m128 vres =
        _mm_add_ps(_mm_mul_ps(va, vscale_0), _mm_mul_ps(vb, vscale_1));
#endif
    _mm_storeu_ps(result.v, vres);
  } else {
    // Linear interpolation fallback
    float scale_0 = 1.0f - t;
    float scale_1 = t;

    // Perform scaled addition: result = scale_0 * va + scale_1 * vb
    __m128 vscale_0 = _mm_set1_ps(scale_0);
    __m128 vscale_1 = _mm_set1_ps(scale_1);
#if defined(WCN_HAS_FMA)
    __m128 vres =
        wcn_fma_mul_add_ps(vb_orig, vscale_1, _mm_mul_ps(va, vscale_0));
#else
    __m128 vres =
        _mm_add_ps(_mm_mul_ps(va, vscale_0), _mm_mul_ps(vb_orig, vscale_1));
#endif
    _mm_storeu_ps(result.v, vres);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);

  // Calculate dot product
  float32x4_t mul = vmulq_f32(va, vb);
  float32x2_t low = vget_low_f32(mul);
  float32x2_t high = vget_high_f32(mul);
  float32x2_t sum = vadd_f32(low, high);
  sum = vpadd_f32(sum, sum);
  float cosOmega = vget_lane_f32(sum, 0);

  float32x4_t vb_orig = vb;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    // Negate vb
    float32x4_t sign_mask = vdupq_n_f32(-0.0f);
    vb = vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(vb), vreinterpretq_u32_f32(sign_mask)));
  }

  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    float scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    float scale_1 = sinf(t * omega) / sinOmega;

    // Perform scaled addition
    float32x4_t vscale_0 = vdupq_n_f32(scale_0);
    float32x4_t vscale_1 = vdupq_n_f32(scale_1);
    float32x4_t vres =
        vaddq_f32(vmulq_f32(va, vscale_0), vmulq_f32(vb, vscale_1));
    vst1q_f32(result.v, vres);
  } else {
    // Linear interpolation fallback
    float scale_0 = 1.0f - t;
    float scale_1 = t;

    // Perform scaled addition
    float32x4_t vscale_0 = vdupq_n_f32(scale_0);
    float32x4_t vscale_1 = vdupq_n_f32(scale_1);
    float32x4_t vres =
        vaddq_f32(vmulq_f32(va, vscale_0), vmulq_f32(vb_orig, vscale_1));
    vst1q_f32(result.v, vres);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t va = wasm_v128_load(a.v);
  v128_t vb = wasm_v128_load(b.v);

  // Calculate dot product
  v128_t mul = wasm_f32x4_mul(va, vb);
  v128_t dot = wasm_f32x4_add(wasm_f32x4_add(mul, wasm_i32x4_shuffle(mul, mul, 1, 0, 3, 2)), wasm_i32x4_shuffle(mul, mul, 2, 3, 0, 1));
  float cosOmega = wasm_f32x4_extract_lane(dot, 0);

  v128_t vb_orig = vb;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    // Negate vb
    v128_t sign_mask = wasm_f32x4_splat(-0.0f);
    vb = wasm_v128_xor(vb, sign_mask);
  }

  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    float scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    float scale_1 = sinf(t * omega) / sinOmega;

    // Perform scaled addition
    v128_t vscale_0 = wasm_f32x4_splat(scale_0);
    v128_t vscale_1 = wasm_f32x4_splat(scale_1);
    v128_t vres = wasm_f32x4_add(wasm_f32x4_mul(va, vscale_0), wasm_f32x4_mul(vb, vscale_1));
    wasm_v128_store(result.v, vres);
  } else {
    // Linear interpolation fallback
    float scale_0 = 1.0f - t;
    float scale_1 = t;

    // Perform scaled addition
    v128_t vscale_0 = wasm_f32x4_splat(scale_0);
    v128_t vscale_1 = wasm_f32x4_splat(scale_1);
    v128_t vres = wasm_f32x4_add(wasm_f32x4_mul(va, vscale_0), wasm_f32x4_mul(vb_orig, vscale_1));
    wasm_v128_store(result.v, vres);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t va = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vb = __riscv_vle32_v_f32m1(b.v, vl);

  // Calculate dot product
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(va, vb, vl);
  vfloat32m1_t dot = __riscv_vfredusum_vs_f32m1_f32m1(mul, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  float cosOmega = __riscv_vfmv_f_s_f32m1_f32(dot);

  vfloat32m1_t vb_orig = vb;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    // Negate vb
    vfloat32m1_t sign_mask = __riscv_vfmv_v_f_f32m1(-0.0f, vl);
    vb = __riscv_vfsgnjn_vv_f32m1(vb, sign_mask, vl);
  }

  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    float scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    float scale_1 = sinf(t * omega) / sinOmega;

    // Perform scaled addition
    vfloat32m1_t vscale_0 = __riscv_vfmv_v_f_f32m1(scale_0, vl);
    vfloat32m1_t vscale_1 = __riscv_vfmv_v_f_f32m1(scale_1, vl);
    vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(va, vscale_0, vl);
    vfloat32m1_t vb_scaled = __riscv_vfmul_vv_f32m1(vb, vscale_1, vl);
    vres = __riscv_vfadd_vv_f32m1(vres, vb_scaled, vl);
    __riscv_vse32_v_f32m1(result.v, vres, vl);
  } else {
    // Linear interpolation fallback
    float scale_0 = 1.0f - t;
    float scale_1 = t;

    // Perform scaled addition
    vfloat32m1_t vscale_0 = __riscv_vfmv_v_f_f32m1(scale_0, vl);
    vfloat32m1_t vscale_1 = __riscv_vfmv_v_f_f32m1(scale_1, vl);
    vfloat32m1_t vres = __riscv_vfmul_vv_f32m1(va, vscale_0, vl);
    vfloat32m1_t vb_scaled = __riscv_vfmul_vv_f32m1(vb_orig, vscale_1, vl);
    vres = __riscv_vfadd_vv_f32m1(vres, vb_scaled, vl);
    __riscv_vse32_v_f32m1(result.v, vres, vl);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = __lsx_vld(a.v, 0);
  __m128 vb = __lsx_vld(b.v, 0);

  // Calculate dot product
  __m128 mul = __lsx_vfmul_s(va, vb);
  __m128 dot = __lsx_vfadd_s(__lsx_vfadd_s(mul, __lsx_vshuf4i_w(mul, 0xB1)), __lsx_vshuf4i_w(mul, 0x1B));
  float cosOmega = __lsx_vpickve2gr_w_s(dot, 0);

  __m128 vb_orig = vb;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    // Negate vb
    __m128 sign_mask = __lsx_vldrepl_w(&(float){-0.0f}, 0);
    vb = __lsx_vxor_v(vb, sign_mask);
  }

  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    float scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    float scale_1 = sinf(t * omega) / sinOmega;

    // Perform scaled addition
    __m128 vscale_0 = __lsx_vldrepl_w(&scale_0, 0);
    __m128 vscale_1 = __lsx_vldrepl_w(&scale_1, 0);
    __m128 vres = __lsx_vfmul_s(va, vscale_0);
    __m128 vb_scaled = __lsx_vfmul_s(vb, vscale_1);
    vres = __lsx_vfadd_s(vres, vb_scaled);
    __lsx_vst(vres, result.v, 0);
  } else {
    // Linear interpolation fallback
    float scale_0 = 1.0f - t;
    float scale_1 = t;

    // Perform scaled addition
    __m128 vscale_0 = __lsx_vldrepl_w(&scale_0, 0);
    __m128 vscale_1 = __lsx_vldrepl_w(&scale_1, 0);
    __m128 vres = __lsx_vfmul_s(va, vscale_0);
    __m128 vb_scaled = __lsx_vfmul_s(vb_orig, vscale_1);
    vres = __lsx_vfadd_s(vres, vb_scaled);
    __lsx_vst(vres, result.v, 0);
  }

#else
  // Scalar fallback (original implementation)
  const float a_x = a.v[0];
  const float a_y = a.v[1];
  const float a_z = a.v[2];
  const float a_w = a.v[3];
  float b_x = b.v[0];
  float b_y = b.v[1];
  float b_z = b.v[2];
  float b_w = b.v[3];

  float cosOmega = a_x * b_x + a_y * b_y + a_z * b_z + a_w * b_w;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    b_x = -b_x;
    b_y = -b_y;
    b_z = -b_z;
    b_w = -b_w;
  }

  float scale_0;
  float scale_1;
  float ep = WCN_GET_EPSILON();
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    scale_0 = sinf((1.0f - t) * omega) / sinOmega;
    scale_1 = sinf(t * omega) / sinOmega;
  } else {
    scale_0 = 1.0f - t;
    scale_1 = t;
  }

  result.v[0] = scale_0 * a_x + scale_1 * b_x;
  result.v[1] = scale_0 * a_y + scale_1 * b_y;
  result.v[2] = scale_0 * a_z + scale_1 * b_z;
  result.v[3] = scale_0 * a_w + scale_1 * b_w;
#endif

  return result;
}

// sqlerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, sqlerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b,
                         WMATH_TYPE(Quat) c, WMATH_TYPE(Quat) d, float t) {

  WMATH_TYPE(Quat) temp_quat_1 = WMATH_CALL(Quat, slerp)(a, b, t);
  WMATH_TYPE(Quat) temp_quat_2 = WMATH_CALL(Quat, slerp)(b, c, t);
  float vt = 2 * t * (1 - t);
  return WMATH_CALL(Quat, slerp)(temp_quat_1, temp_quat_2, vt);
}

float WMATH_LENGTH(Quat)(WMATH_TYPE(Quat) a) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper function
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_squared = _mm_mul_ps(vec_a, vec_a);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper function
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_squared = vmulq_f32(vec_a, vec_a);
  float len_sq = wcn_hadd_f32(vec_squared);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_a, vec_a);
  v128_t sum = wasm_f32x4_add(wasm_f32x4_add(vec_squared, wasm_i32x4_shuffle(vec_squared, vec_squared, 1, 0, 3, 2)), wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 0, 1));
  float len_sq = wasm_f32x4_extract_lane(sum, 0);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_a, vec_a, vl);
  vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_squared, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  float len_sq = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_a, vec_a);
  __m128 sum = __lsx_vfadd_s(__lsx_vfadd_s(vec_squared, __lsx_vshuf4i_w(vec_squared, 0xB1)), __lsx_vshuf4i_w(vec_squared, 0x1B));
  float len_sq = __lsx_vpickve2gr_w_s(sum, 0);
  return sqrtf(len_sq);

#else
  // Scalar fallback
  return sqrtf(WMATH_DOT(Quat)(a, a));
#endif
}

float WMATH_LENGTH_SQ(Quat)(WMATH_TYPE(Quat) a) {
  return WMATH_DOT(Quat)(a, a);
}

WMATH_TYPE(Quat)
WMATH_NORMALIZE(Quat)(WMATH_TYPE(Quat) a) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation with fast inverse square root
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_squared = _mm_mul_ps(vec_a, vec_a);

  // Horizontally add to get length squared
  __m128 temp = _mm_hadd_ps(vec_squared, vec_squared);
  temp = _mm_hadd_ps(temp, temp);
  float len_sq = _mm_cvtss_f32(temp);

  if (len_sq > 0.00001f) {
    // Use fast inverse square root for better performance
    __m128 vec_len_sq = _mm_set1_ps(len_sq);
    __m128 inv_len = wcn_fast_inv_sqrt_ps(vec_len_sq);
    __m128 vec_res = _mm_mul_ps(vec_a, inv_len);
    _mm_storeu_ps(result.v, vec_res);
  } else {
    // Return identity quaternion
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation with fast inverse square root
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_squared = vmulq_f32(vec_a, vec_a);

  // Horizontally add to get length squared
  float32x2_t low = vget_low_f32(vec_squared);
  float32x2_t high = vget_high_f32(vec_squared);
  float32x2_t sum = vadd_f32(low, high);
  sum = vpadd_f32(sum, sum);
  float len_sq = vget_lane_f32(sum, 0);

  if (len_sq > 0.00001f) {
    // Use fast inverse square root for better performance
    float32x4_t vec_len_sq = vdupq_n_f32(len_sq);
    float32x4_t inv_len = wcn_fast_inv_sqrt_platform(vec_len_sq);
    float32x4_t vec_res = vmulq_f32(vec_a, inv_len);
    vst1q_f32(result.v, vec_res);
  } else {
    // Return identity quaternion
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation with fast inverse square root
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_a, vec_a);

  // Horizontally add to get length squared
  v128_t sum = wasm_f32x4_add(wasm_f32x4_add(vec_squared, wasm_i32x4_shuffle(vec_squared, vec_squared, 1, 0, 3, 2)), wasm_i32x4_shuffle(vec_squared, vec_squared, 2, 3, 0, 1));
  float len_sq = wasm_f32x4_extract_lane(sum, 0);

  if (len_sq > 0.00001f) {
    // Use fast inverse square root for better performance
    v128_t vec_len_sq = wasm_f32x4_splat(len_sq);
    v128_t inv_len = wcn_fast_inv_sqrt_wasm(vec_len_sq);
    v128_t vec_res = wasm_f32x4_mul(vec_a, inv_len);
    wasm_v128_store(result.v, vec_res);
  } else {
    // Return identity quaternion
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation with fast inverse square root
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_a, vec_a, vl);

  // Horizontally add to get length squared
  vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_squared, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  float len_sq = __riscv_vfmv_f_s_f32m1_f32(vec_sum);

  if (len_sq > 0.00001f) {
    // Use fast inverse square root for better performance
    vfloat32m1_t vec_len_sq = __riscv_vfmv_v_f_f32m1(len_sq, vl);
    vfloat32m1_t inv_len = wcn_fast_inv_sqrt_platform(vec_len_sq);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, inv_len, vl);
    __riscv_vse32_v_f32m1(result.v, vec_res, vl);
  } else {
    // Return identity quaternion
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation with fast inverse square root
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_squared = __lsx_vfmul_s(vec_a, vec_a);

  // Horizontally add to get length squared
  __m128 sum = __lsx_vfadd_s(__lsx_vfadd_s(vec_squared, __lsx_vshuf4i_w(vec_squared, 0xB1)), __lsx_vshuf4i_w(vec_squared, 0x1B));
  float len_sq = __lsx_vpickve2gr_w_s(sum, 0);

  if (len_sq > 0.00001f) {
    // Use fast inverse square root for better performance
    __m128 vec_len_sq = __lsx_vldrepl_w(&len_sq, 0);
    __m128 inv_len = wcn_fast_inv_sqrt_lsx(vec_len_sq);
    __m128 vec_res = __lsx_vfmul_s(vec_a, inv_len);
    __lsx_vst(vec_res, result.v, 0);
  } else {
    // Return identity quaternion
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }

#else
  // Scalar fallback with fast inverse square root
  float len_sq =
      a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2] + a.v[3] * a.v[3];
  if (len_sq > 0.00001f) {
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    result.v[0] = a.v[0] * inv_len;
    result.v[1] = a.v[1] * inv_len;
    result.v[2] = a.v[2] * inv_len;
    result.v[3] = a.v[3] * inv_len;
  } else {
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
    result.v[3] = 1.0f;
  }
#endif

  return result;
}

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  const float ep = wcn_math_get_epsilon();
  return fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep &&
         fabsf(a.v[2] - b.v[2]) < ep && fabsf(a.v[3] - b.v[3]) < ep;
}

// ==
bool WMATH_EQUALS(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] &&
         a.v[3] == b.v[3];
}

float WMATH_ANGLE(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  const float cosOmega = WMATH_DOT(Quat)(a, b);
  return acosf(2.0f * cosOmega - 1.0f);
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, rotation_to)(const WMATH_TYPE(Vec3) a_unit,
                              const WMATH_TYPE(Vec3) b_unit) {
  WMATH_TYPE(Quat) result = WMATH_ZERO(Quat)();
  WMATH_TYPE(Vec3) tempVec3 = wcn_math_Vec3_zero();
  const WMATH_TYPE(Vec3) xUnitVec3 = { .v = {1.0f, 0.0f, 0.0f} };
  const WMATH_TYPE(Vec3) yUnitVec3 = { .v = {0.0f, 1.0f, 0.0f} };
  const float dot = WMATH_DOT(Vec3)(a_unit, b_unit);
  if (dot < -0.999999f) {
    tempVec3 = WMATH_CALL(Vec3, cross)(xUnitVec3, a_unit);
    if (WMATH_LENGTH(Vec3)(tempVec3) < 0.000001f) {
      tempVec3 = WMATH_CALL(Vec3, cross)(yUnitVec3, a_unit);
    }

    tempVec3 = WMATH_NORMALIZE(Vec3)(tempVec3);
    result = WMATH_CALL(Quat, from_axis_angle)(tempVec3, WMATH_PI);
    return result;
  } else if (dot > 0.999999f) {
    result.v[0] = 0;
    result.v[1] = 0;
    result.v[2] = 0;
    result.v[3] = 1;

    return result;
  } else {
    tempVec3 = WMATH_CALL(Vec3, cross)(a_unit, b_unit);
    result.v[0] = tempVec3.v[0];
    result.v[1] = tempVec3.v[1];
    result.v[2] = tempVec3.v[2];
    result.v[3] = 1 + dot;
    return WMATH_NORMALIZE(Quat)(result);
  }
}

// *
WMATH_TYPE(Quat)
WMATH_MULTIPLY(Quat)
(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) r;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // ------------------------ SSE (dot+cross style) ------------------------
  // r.xyz = aw*b.xyz + bw*a.xyz + cross(a.xyz, b.xyz)
  // r.w   = aw*bw - dot(a.xyz, b.xyz)

  __m128 a_vec = _mm_loadu_ps(a.v); // [ax, ay, az, aw]
  __m128 b_vec = _mm_loadu_ps(b.v); // [bx, by, bz, bw]

  // Extract scalar aw, bw into vectors for FMA (broadcast)
  __m128 aw_vec = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3,3,3,3));
  __m128 bw_vec = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3,3,3,3));

  // b.xyz as vector with w slot zeroed (so fma won't touch w)
  // We'll compute with full 4-lane vectors but ignore final lane in packing.
  // Create mask to zero w if needed (not strictly necessary if we only read xyz)
  // Cross product using shuffles:
  // cross = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

  // Shuffle helpers
  __m128 a_yzx = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3,0,2,1)); // [ay, az, ax, aw]
  __m128 a_zxy = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3,1,0,2)); // [az, ax, ay, aw]
  __m128 b_yzx = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3,0,2,1)); // [by, bz, bx, bw]
  __m128 b_zxy = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3,1,0,2)); // [bz, bx, by, bw]

  // cross = a_yzx * b_zxy - a_zxy * b_yzx  (lanewise)
  __m128 cross = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
  // cross now has [cx, cy, cz, ?] where ? may be garbage in w lane.

  // aw * b.xyz + bw * a.xyz
  __m128 t1 = _mm_mul_ps(aw_vec, b_vec);
  __m128 t2 = _mm_mul_ps(bw_vec, a_vec);
  // add them and then add cross
  __m128 xyz_full = _mm_add_ps(_mm_add_ps(t1, t2), cross);

  // compute w = aw*bw - dot(a.xyz, b.xyz)
  float w_scalar;
  #if defined(__SSE4_1__)
    // use dot product intrinsic (mask 0x7 for xyz, 0xF for all lanes if needed)
    __m128 dp = _mm_dp_ps(a_vec, b_vec, 0x71); // bits: lower 3 lanes, store to low lane
    // dp contains dot(a.xyz, b.xyz) in lowest lane
    float dot_xyz = _mm_cvtss_f32(dp);
    const float aw_s = a.v[3];
    const float bw_s = b.v[3];
    w_scalar = aw_s * bw_s - dot_xyz;
  #else
    // fallback: mul + horizontal add
    __m128 mul = _mm_mul_ps(a_vec, b_vec); // [ax*bx, ay*by, az*bz, aw*bw]
    __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2,1,0,3)); // rotate
    __m128 sums = _mm_add_ps(mul, shuf);
    sums = _mm_add_ss(sums, _mm_movehl_ps(sums, sums)); // sums[0] = ax*bx + ay*by + az*bz + aw*bw
    float sum_all = _mm_cvtss_f32(sums);
    // subtract aw*bw
    float awbw = a.v[3] * b.v[3];
    float dot_xyz = sum_all - awbw;
    w_scalar = a.v[3] * b.v[3] - dot_xyz;
  #endif

  // store xyz from xyz_full (we want lanes 0..2), and w from w_scalar
  _mm_storeu_ps(r.v, xyz_full); // temporarily writes w with garbage; we'll overwrite
  r.v[3] = w_scalar;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ------------------------ NEON (dot+cross style) ------------------------
  // Use vector cross + fmla to implement:
  // r.xyz = aw*b.xyz + bw*a.xyz + cross(a.xyz, b.xyz)
  // r.w   = aw*bw - dot(a.xyz, b.xyz)

  float32x4_t a_vec = vld1q_f32(a.v); // [ax, ay, az, aw]
  float32x4_t b_vec = vld1q_f32(b.v); // [bx, by, bz, bw]

  // Extract xyz as vectors where w slot can be anything (we'll ignore it)
  // Using vext to rotate for cross product
  float32x4_t a_yzx = vextq_f32(a_vec, a_vec, 1); // [ay, az, aw, ax]  (we'll ignore aw)
  float32x4_t a_zxy = vextq_f32(a_vec, a_vec, 2); // [az, aw, ax, ay]
  float32x4_t b_yzx = vextq_f32(b_vec, b_vec, 1);
  float32x4_t b_zxy = vextq_f32(b_vec, b_vec, 2);

  // cross = a_yzx * b_zxy - a_zxy * b_yzx
  float32x4_t cross = vsubq_f32(vmulq_f32(a_yzx, b_zxy), vmulq_f32(a_zxy, b_yzx));
  // cross lanes correspond to [cx, cy, cz, ?] but note positions due to vext shifts:
  // We need to realign to [cx, cy, cz, ...]. For canonical rotation using vext above,
  // the low 3 lanes hold cx, cy, cz but might be offset by one; to be safe, we can
  // rotate cross back by 3 (or 1) — below we'll extract needed lanes via vget_low/vcombine.

  // Compute aw*b + bw*a with fmla style (broadcast aw/bw)
  float32x4_t aw_vec = vdupq_n_f32(a.v[3]);
  float32x4_t bw_vec = vdupq_n_f32(b.v[3]);

  float32x4_t t1 = vmulq_f32(aw_vec, b_vec); // aw * b
  float32x4_t t2 = vmulq_f32(bw_vec, a_vec); // bw * a

  float32x4_t xyz_full = vaddq_f32(vaddq_f32(t1, t2), cross);

  // compute dot(a.xyz, b.xyz)
  #if defined(__ARM_FEATURE_DOTPROD)
    // vdotq_u32 family exists on some targets; in float domain we can use vdotq_f32 if available
    float32x4_t dot_v = vdotq_f32(vdupq_n_f32(0.0f), a_vec, b_vec); // not standard in all toolchains
    // The above is illustrative; if vdotq_f32 not available, use fallback below.
    float dot_xyz = vgetq_lane_f32(dot_v, 0);
  #else
    // fallback: multiply then horizontal add
    float32x4_t mul = vmulq_f32(a_vec, b_vec); // [ax*bx, ay*by, az*bz, aw*bw]
    // sum lower three lanes: use vget_low & vaddv for portability
    float32x2_t low = vget_low_f32(mul);       // [ax*bx, ay*by]
    float32x2_t high = vget_high_f32(mul);     // [az*bz, aw*bw]
    float sum0 = vget_lane_f32(low, 0) + vget_lane_f32(low, 1);
    float sum1 = vget_lane_f32(high, 0); // az*bz
    float dot_xyz = sum0 + sum1;
  #endif

  float w_scalar = a.v[3] * b.v[3] - dot_xyz;

  vst1q_f32(r.v, xyz_full);
  r.v[3] = w_scalar;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // ------------------------ WASM SIMD (dot+cross style) ------------------------
  // r.xyz = aw*b.xyz + bw*a.xyz + cross(a.xyz, b.xyz)
  // r.w   = aw*bw - dot(a.xyz, b.xyz)

  v128_t a_vec = wasm_v128_load(a.v); // [ax, ay, az, aw]
  v128_t b_vec = wasm_v128_load(b.v); // [bx, by, bz, bw]

  // Extract scalar aw, bw into vectors for broadcast
  v128_t aw_vec = wasm_i32x4_shuffle(a_vec, a_vec, 3, 3, 3, 3);
  v128_t bw_vec = wasm_i32x4_shuffle(b_vec, b_vec, 3, 3, 3, 3);

  // Shuffle helpers for cross product
  v128_t a_yzx = wasm_i32x4_shuffle(a_vec, a_vec, 1, 2, 0, 3); // [ay, az, ax, aw]
  v128_t a_zxy = wasm_i32x4_shuffle(a_vec, a_vec, 2, 0, 1, 3); // [az, ax, ay, aw]
  v128_t b_yzx = wasm_i32x4_shuffle(b_vec, b_vec, 1, 2, 0, 3); // [by, bz, bx, bw]
  v128_t b_zxy = wasm_i32x4_shuffle(b_vec, b_vec, 2, 0, 1, 3); // [bz, bx, by, bw]

  // cross = a_yzx * b_zxy - a_zxy * b_yzx
  v128_t cross = wasm_f32x4_sub(wasm_f32x4_mul(a_yzx, b_zxy), wasm_f32x4_mul(a_zxy, b_yzx));

  // aw * b.xyz + bw * a.xyz
  v128_t t1 = wasm_f32x4_mul(aw_vec, b_vec);
  v128_t t2 = wasm_f32x4_mul(bw_vec, a_vec);
  // add them and then add cross
  v128_t xyz_full = wasm_f32x4_add(wasm_f32x4_add(t1, t2), cross);

  // compute dot(a.xyz, b.xyz)
  v128_t mul = wasm_f32x4_mul(a_vec, b_vec); // [ax*bx, ay*by, az*bz, aw*bw]
  v128_t sum1 = wasm_f32x4_add(mul, wasm_i32x4_shuffle(mul, mul, 1, 0, 3, 2));
  v128_t sum2 = wasm_f32x4_add(sum1, wasm_i32x4_shuffle(sum1, sum1, 2, 3, 0, 1));
  float dot_xyz = wasm_f32x4_extract_lane(sum2, 0) - a.v[3] * b.v[3]; // subtract aw*bw

  float w_scalar = a.v[3] * b.v[3] - dot_xyz;

  wasm_v128_store(r.v, xyz_full);
  r.v[3] = w_scalar;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // ------------------------ RISC-V Vector Extension (dot+cross style) ------------------------
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t a_vec = __riscv_vle32_v_f32m1(a.v, vl); // [ax, ay, az, aw]
  vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b.v, vl); // [bx, by, bz, bw]

  // Extract scalar aw, bw into vectors for broadcast
  float aw_s = a.v[3];
  float bw_s = b.v[3];
  vfloat32m1_t aw_vec = __riscv_vfmv_v_f_f32m1(aw_s, vl);
  vfloat32m1_t bw_vec = __riscv_vfmv_v_f_f32m1(bw_s, vl);

  // Shuffle helpers for cross product (manual implementation)
  vfloat32m1_t a_yzx = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(a_vec, 1, vl), __riscv_vslidedown_vx_f32m1(a_vec, 0, vl), 2, vl);
  vfloat32m1_t a_zxy = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(a_vec, 2, vl), __riscv_vslidedown_vx_f32m1(a_vec, 0, vl), 1, vl);
  vfloat32m1_t b_yzx = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(b_vec, 1, vl), __riscv_vslidedown_vx_f32m1(b_vec, 0, vl), 2, vl);
  vfloat32m1_t b_zxy = __riscv_vslideup_vx_f32m1(__riscv_vslidedown_vx_f32m1(b_vec, 2, vl), __riscv_vslidedown_vx_f32m1(b_vec, 0, vl), 1, vl);

  // cross = a_yzx * b_zxy - a_zxy * b_yzx
  vfloat32m1_t cross = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(a_yzx, b_zxy, vl), __riscv_vfmul_vv_f32m1(a_zxy, b_yzx, vl), vl);

  // aw * b.xyz + bw * a.xyz
  vfloat32m1_t t1 = __riscv_vfmul_vv_f32m1(aw_vec, b_vec, vl);
  vfloat32m1_t t2 = __riscv_vfmul_vv_f32m1(bw_vec, a_vec, vl);
  // add them and then add cross
  vfloat32m1_t xyz_full = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(t1, t2, vl), cross, vl);

  // compute dot(a.xyz, b.xyz)
  vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(a_vec, b_vec, vl);
  vfloat32m1_t sum_vec = __riscv_vfredusum_vs_f32m1_f32m1(mul, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  float sum_all = __riscv_vfmv_f_s_f32m1_f32(sum_vec);
  // subtract aw*bw
  float awbw = aw_s * bw_s;
  float dot_xyz = sum_all - awbw;

  float w_scalar = awbw - dot_xyz;

  __riscv_vse32_v_f32m1(r.v, xyz_full, vl);
  r.v[3] = w_scalar;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // ------------------------ LoongArch LSX (dot+cross style) ------------------------
  // r.xyz = aw*b.xyz + bw*a.xyz + cross(a.xyz, b.xyz)
  // r.w   = aw*bw - dot(a.xyz, b.xyz)

  __m128 a_vec = __lsx_vld(a.v, 0); // [ax, ay, az, aw]
  __m128 b_vec = __lsx_vld(b.v, 0); // [bx, by, bz, bw]

  // Extract scalar aw, bw into vectors for broadcast
  __m128 aw_vec = __lsx_vreplvei_w(a_vec, 3);
  __m128 bw_vec = __lsx_vreplvei_w(b_vec, 3);

  // Shuffle helpers for cross product
  __m128 a_yzx = __lsx_vshuf4i_w(a_vec, 0x39); // [ay, az, ax, aw]
  __m128 a_zxy = __lsx_vshuf4i_w(a_vec, 0x93); // [az, ax, ay, aw]
  __m128 b_yzx = __lsx_vshuf4i_w(b_vec, 0x39); // [by, bz, bx, bw]
  __m128 b_zxy = __lsx_vshuf4i_w(b_vec, 0x93); // [bz, bx, by, bw]

  // cross = a_yzx * b_zxy - a_zxy * b_yzx
  __m128 cross = __lsx_vfsub_s(__lsx_vfmul_s(a_yzx, b_zxy), __lsx_vfmul_s(a_zxy, b_yzx));

  // aw * b.xyz + bw * a.xyz
  __m128 t1 = __lsx_vfmul_s(aw_vec, b_vec);
  __m128 t2 = __lsx_vfmul_s(bw_vec, a_vec);
  // add them and then add cross
  __m128 xyz_full = __lsx_vfadd_s(__lsx_vfadd_s(t1, t2), cross);

  // compute dot(a.xyz, b.xyz)
  __m128 mul = __lsx_vfmul_s(a_vec, b_vec);
  __m128 sum1 = __lsx_vfadd_s(mul, __lsx_vshuf4i_w(mul, 0xB1));
  __m128 sum2 = __lsx_vfadd_s(sum1, __lsx_vshuf4i_w(sum1, 0x1B));
  float dot_xyz = __lsx_vpickve2gr_w_s(sum2, 0) - a.v[3] * b.v[3]; // subtract aw*bw

  float w_scalar = a.v[3] * b.v[3] - dot_xyz;

  __lsx_vst(xyz_full, r.v, 0);
  r.v[3] = w_scalar;

#else
  // ------------------------ 标量回退 ------------------------
  float ax = a.v[0], ay = a.v[1], az = a.v[2], aw = a.v[3];
  float bx = b.v[0], by = b.v[1], bz = b.v[2], bw = b.v[3];
  // vector part
  float cx = ay*bz - az*by;
  float cy = az*bx - ax*bz;
  float cz = ax*by - ay*bx;
  r.v[0] = aw*bx + bw*ax + cx;
  r.v[1] = aw*by + bw*ay + cy;
  r.v[2] = aw*bz + bw*az + cz;
  r.v[3] = aw*bw - (ax*bx + ay*by + az*bz);
#endif

  return r;
}


// .*
WMATH_TYPE(Quat)
WMATH_MULTIPLY_SCALAR(Quat)(WMATH_TYPE(Quat) a, float b) {
  WMATH_TYPE(Quat) r;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_set1_ps(b);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vdupq_n_f32(b);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - process all 4 elements at once
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_f32x4_splat(b);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wasm_v128_store(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - process all 4 elements at once
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_b = __riscv_vfmv_v_f_f32m1(b, vl);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(r.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process all 4 elements at once
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_b = __lsx_vldrepl_w(&b, 0);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_b);
  __lsx_vst(vec_res, r.v, 0);

#else
  // Scalar fallback
  r.v[0] = a.v[0] * b;
  r.v[1] = a.v[1] * b;
  r.v[2] = a.v[2] * b;
  r.v[3] = a.v[3] * b;
#endif

  return r;
}

// -
WMATH_TYPE(Quat)
WMATH_SUB(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) result;

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

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - process all 4 elements at once
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - process all 4 elements at once
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, vl);
  vfloat32m1_t vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(result.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process all 4 elements at once
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

// +
WMATH_TYPE(Quat)
WMATH_ADD(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) result;

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

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - process all 4 elements at once
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_b = wasm_v128_load(b.v);
  v128_t vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - process all 4 elements at once
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b.v, vl);
  vfloat32m1_t vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(result.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - process all 4 elements at once
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

// inverse
WMATH_TYPE(Quat)
WMATH_INVERSE(Quat)(const WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Quat) r;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_q = _mm_loadu_ps(q.v);
  __m128 vec_sq = _mm_mul_ps(vec_q, vec_q);

  // Horizontally add to get dot product
  __m128 temp = _mm_hadd_ps(vec_sq, vec_sq);
  temp = _mm_hadd_ps(temp, temp);
  float dot = _mm_cvtss_f32(temp);

  const float invDot = dot ? 1.0f / dot : 0.0f;
  __m128 vec_invDot = _mm_set1_ps(invDot);

  // Conjugate: negate x, y, z components
  __m128 sign_mask = _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f);
  __m128 vec_conj = _mm_xor_ps(vec_q, sign_mask);

  // Multiply conjugate by invDot
  __m128 vec_res = _mm_mul_ps(vec_conj, vec_invDot);
  _mm_storeu_ps(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_q = vld1q_f32(q.v);
  float32x4_t vec_sq = vmulq_f32(vec_q, vec_q);

  // Horizontally add to get dot product
  float32x2_t low = vget_low_f32(vec_sq);
  float32x2_t high = vget_high_f32(vec_sq);
  float32x2_t sum = vadd_f32(low, high);
  sum = vpadd_f32(sum, sum);
  float dot = vget_lane_f32(sum, 0);

  const float invDot = dot ? 1.0f / dot : 0.0f;
  float32x4_t vec_invDot = vdupq_n_f32(invDot);

  // Conjugate: negate x, y, z components
  float32x4_t sign_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x80000000));
  sign_mask = vsetq_lane_f32(0.0f, sign_mask, 3); // w component is positive
  float32x4_t vec_conj = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec_q), vreinterpretq_u32_f32(sign_mask)));

  // Multiply conjugate by invDot
  float32x4_t vec_res = vmulq_f32(vec_conj, vec_invDot);
  vst1q_f32(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_q = wasm_v128_load(q.v);
  v128_t vec_sq = wasm_f32x4_mul(vec_q, vec_q);

  // Horizontally add to get dot product
  v128_t sum1 = wasm_f32x4_add(vec_sq, wasm_i32x4_shuffle(vec_sq, vec_sq, 1, 0, 3, 2));
  v128_t sum2 = wasm_f32x4_add(sum1, wasm_i32x4_shuffle(sum1, sum1, 2, 3, 0, 1));
  float dot = wasm_f32x4_extract_lane(sum2, 0);

  const float invDot = dot ? 1.0f / dot : 0.0f;
  v128_t vec_invDot = wasm_f32x4_splat(invDot);

  // Conjugate: negate x, y, z components
  v128_t sign_mask = wasm_f32x4_make(-0.0f, -0.0f, -0.0f, 0.0f);
  v128_t vec_conj = wasm_v128_xor(vec_q, sign_mask);

  // Multiply conjugate by invDot
  v128_t vec_res = wasm_f32x4_mul(vec_conj, vec_invDot);
  wasm_v128_store(r.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_q = __riscv_vle32_v_f32m1(q.v, vl);
  vfloat32m1_t vec_sq = __riscv_vfmul_vv_f32m1(vec_q, vec_q, vl);

  // Horizontally add to get dot product
  vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_sq, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl);
  float dot = __riscv_vfmv_f_s_f32m1_f32(vec_sum);

  const float invDot = dot ? 1.0f / dot : 0.0f;
  vfloat32m1_t vec_invDot = __riscv_vfmv_v_f_f32m1(invDot, vl);

  // Conjugate: negate x, y, z components
  vfloat32m1_t sign_mask = __riscv_vfmv_v_f_f32m1(-0.0f, vl);
  sign_mask = __riscv_vslide1down_vx_f32m1(sign_mask, 0.0f, 3, vl); // w component is positive
  vfloat32m1_t vec_conj = __riscv_vfsgnjn_vv_f32m1(vec_q, sign_mask, vl);

  // Multiply conjugate by invDot
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_conj, vec_invDot, vl);
  __riscv_vse32_v_f32m1(r.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_q = __lsx_vld(q.v, 0);
  __m128 vec_sq = __lsx_vfmul_s(vec_q, vec_q);

  // Horizontally add to get dot product
  __m128 sum = __lsx_vfadd_s(__lsx_vfadd_s(vec_sq, __lsx_vshuf4i_w(vec_sq, 0xB1)), __lsx_vshuf4i_w(vec_sq, 0x1B));
  float dot = __lsx_vpickve2gr_w_s(sum, 0);

  const float invDot = dot ? 1.0f / dot : 0.0f;
  __m128 vec_invDot = __lsx_vldrepl_w(&invDot, 0);

  // Conjugate: negate x, y, z components
  __m128 sign_mask = __lsx_vldrepl_w(&(float){-0.0f}, 0);
  sign_mask = __lsx_vinsgr2vr_w(sign_mask, 0, 3); // w component is positive
  __m128 vec_conj = __lsx_vxor_v(vec_q, sign_mask);

  // Multiply conjugate by invDot
  __m128 vec_res = __lsx_vfmul_s(vec_conj, vec_invDot);
  __lsx_vst(vec_res, r.v, 0);

#else
  // Scalar fallback
  const float dot = q.v[0] * q.v[0] + q.v[1] * q.v[1] + q.v[2] * q.v[2] + q.v[3] * q.v[3];
  const float invDot = dot ? 1.0f / dot : 0.0f;

  // Conjugate divided by squared length
  r.v[0] = -q.v[0] * invDot;
  r.v[1] = -q.v[1] * invDot;
  r.v[2] = -q.v[2] * invDot;
  r.v[3] = q.v[3] * invDot;
#endif

  return r;
}

// conjugate
WMATH_TYPE(Quat)
WMATH_CALL(Quat, conjugate)(WMATH_TYPE(Quat) q) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  WMATH_TYPE(Quat) result;
  __m128 vec_q = _mm_loadu_ps(q.v);

  // Create sign mask to negate x, y, z components
  __m128 sign_mask = _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f);
  __m128 vec_res = _mm_xor_ps(vec_q, sign_mask);

  _mm_storeu_ps(result.v, vec_res);
  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  WMATH_TYPE(Quat) result;
  float32x4_t vec_q = vld1q_f32(q.v);

  // Create sign mask to negate x, y, z components
  float32x4_t sign_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x80000000));
  sign_mask = vsetq_lane_f32(0.0f, sign_mask, 3); // w component is positive
  float32x4_t vec_res = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec_q), vreinterpretq_u32_f32(sign_mask)));

  vst1q_f32(result.v, vec_res);
  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  WMATH_TYPE(Quat) result;
  v128_t vec_q = wasm_v128_load(q.v);

  // Create sign mask to negate x, y, z components
  v128_t sign_mask = wasm_f32x4_make(-0.0f, -0.0f, -0.0f, 0.0f);
  v128_t vec_res = wasm_v128_xor(vec_q, sign_mask);

  wasm_v128_store(result.v, vec_res);
  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  WMATH_TYPE(Quat) result;
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_q = __riscv_vle32_v_f32m1(q.v, vl);

  // Create sign mask to negate x, y, z components
  vfloat32m1_t sign_mask = __riscv_vfmv_v_f_f32m1(-0.0f, vl);
  sign_mask = __riscv_vslide1down_vx_f32m1(sign_mask, 0.0f, 3, vl); // w component is positive
  vfloat32m1_t vec_res = __riscv_vfsgnjn_vv_f32m1(vec_q, sign_mask, vl);

  __riscv_vse32_v_f32m1(result.v, vec_res, vl);
  return result;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  WMATH_TYPE(Quat) result;
  __m128 vec_q = __lsx_vld(q.v, 0);

  // Create sign mask to negate x, y, z components
  __m128 sign_mask = __lsx_vldrepl_w(&(float){-0.0f}, 0);
  sign_mask = __lsx_vinsgr2vr_w(sign_mask, 0, 3); // w component is positive
  __m128 vec_res = __lsx_vxor_v(vec_q, sign_mask);

  __lsx_vst(vec_res, result.v, 0);
  return result;

#else
  // Scalar fallback
  return (WMATH_TYPE(Quat)){
      .v = {-q.v[0], -q.v[1], -q.v[2], q.v[3]},
  };
#endif
}

// divScalar
WMATH_TYPE(Quat)
WMATH_DIV_SCALAR(Quat)(WMATH_TYPE(Quat) a, float v) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_v = _mm_set1_ps(v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_v);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_v = vdupq_n_f32(v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_v);
  vst1q_f32(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wasm_v128_load(a.v);
  v128_t vec_v = wasm_f32x4_splat(v);
  v128_t vec_res = wasm_f32x4_div(vec_a, vec_v);
  wasm_v128_store(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(4);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_v = __riscv_vfmv_v_f_f32m1(v, vl);
  vfloat32m1_t vec_res = __riscv_vfdiv_vv_f32m1(vec_a, vec_v, vl);
  __riscv_vse32_v_f32m1(result.v, vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = __lsx_vld(a.v, 0);
  __m128 vec_v = __lsx_vldrepl_w(&v, 0);
  __m128 vec_res = __lsx_vfdiv_s(vec_a, vec_v);
  __lsx_vst(vec_res, result.v, 0);

#else
  // Scalar fallback
  result.v[0] = a.v[0] / v;
  result.v[1] = a.v[1] / v;
  result.v[2] = a.v[2] / v;
  result.v[3] = a.v[3] / v;
#endif

  return result;
}

// END Quat
