#include "wcn_math_internal.h"
#include <string.h>

// BEGIN Mat4

// 0 add 1 Mat4

WMATH_TYPE(Mat4) WMATH_IDENTITY(Mat4)() {
  WMATH_TYPE(Mat4) result = {0};
  result.m[0] = 1.0f;
  result.m[5] = 1.0f;
  result.m[10] = 1.0f;
  result.m[15] = 1.0f;
  return result;
}

WMATH_TYPE(Mat4) WMATH_ZERO(Mat4)() {
  WMATH_TYPE(Mat4) result = {0};
  return result;
}

// Init Mat4

WMATH_TYPE(Mat4) WMATH_CREATE(Mat4)(WMATH_CREATE_TYPE(Mat4) mat4_c) {
  WMATH_TYPE(Mat4) mat = {0};
  mat.m[0] = WMATH_OR_ELSE_ZERO(mat4_c.m_00);
  mat.m[1] = WMATH_OR_ELSE_ZERO(mat4_c.m_01);
  mat.m[2] = WMATH_OR_ELSE_ZERO(mat4_c.m_02);
  mat.m[3] = WMATH_OR_ELSE_ZERO(mat4_c.m_03);
  mat.m[4] = WMATH_OR_ELSE_ZERO(mat4_c.m_10);
  mat.m[5] = WMATH_OR_ELSE_ZERO(mat4_c.m_11);
  mat.m[6] = WMATH_OR_ELSE_ZERO(mat4_c.m_12);
  mat.m[7] = WMATH_OR_ELSE_ZERO(mat4_c.m_13);
  mat.m[8] = WMATH_OR_ELSE_ZERO(mat4_c.m_20);
  mat.m[9] = WMATH_OR_ELSE_ZERO(mat4_c.m_21);
  mat.m[10] = WMATH_OR_ELSE_ZERO(mat4_c.m_22);
  mat.m[11] = WMATH_OR_ELSE_ZERO(mat4_c.m_23);
  mat.m[12] = WMATH_OR_ELSE_ZERO(mat4_c.m_30);
  mat.m[13] = WMATH_OR_ELSE_ZERO(mat4_c.m_31);
  mat.m[14] = WMATH_OR_ELSE_ZERO(mat4_c.m_32);
  mat.m[15] = WMATH_OR_ELSE_ZERO(mat4_c.m_33);
  return mat;
}

WMATH_TYPE(Mat4) WMATH_COPY(Mat4)(WMATH_TYPE(Mat4) mat) {
  WMATH_TYPE(Mat4) mat_copy;
  memcpy(&mat_copy, &mat, sizeof(WMATH_TYPE(Mat4)));
  return mat_copy;
}

WMATH_TYPE(Mat4)
WMATH_SET(Mat4)(WMATH_TYPE(Mat4) mat, float m00, float m01, float m02,
                float m03, float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23, float m30,
                float m31, float m32, float m33) {
  mat.m[0] = m00;
  mat.m[1] = m01;
  mat.m[2] = m02;
  mat.m[3] = m03;
  mat.m[4] = m10;
  mat.m[5] = m11;
  mat.m[6] = m12;
  mat.m[7] = m13;
  mat.m[8] = m20;
  mat.m[9] = m21;
  mat.m[10] = m22;
  mat.m[11] = m23;
  mat.m[12] = m30;
  mat.m[13] = m31;
  mat.m[14] = m32;
  mat.m[15] = m33;
  return mat;
}

WMATH_TYPE(Mat4)
WMATH_NEGATE(Mat4)(WMATH_TYPE(Mat4) mat) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - negate using XOR with sign bit mask
  const __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements

  // Process all 16 elements in groups of 4
  __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[0], vec_res);

  vec_a = _mm_loadu_ps(&mat.m[4]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[4], vec_res);

  vec_a = _mm_loadu_ps(&mat.m[8]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[8], vec_res);

  vec_a = _mm_loadu_ps(&mat.m[12]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a, vec_res;

  // Process all 16 elements in groups of 4
  vec_a = vld1q_f32(&mat.m[0]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[0], vec_res);

  vec_a = vld1q_f32(&mat.m[4]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[4], vec_res);

  vec_a = vld1q_f32(&mat.m[8]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[8], vec_res);

  vec_a = vld1q_f32(&mat.m[12]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - negate using multiply with -1.0f
  v128_t vec_a, vec_res;
  const v128_t neg_one = wasm_f32x4_splat(-1.0f);

  // Process all 16 elements in groups of 4
  vec_a = wasm_v128_load(&mat.m[0]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[0], vec_res);

  vec_a = wasm_v128_load(&mat.m[4]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[4], vec_res);

  vec_a = wasm_v128_load(&mat.m[8]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[8], vec_res);

  vec_a = wasm_v128_load(&mat.m[12]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - negate using vector operations
  size_t vl;
  vfloat32m1_t vec_a, vec_res;
  float *a_ptr = (float*)&mat;
  float *res_ptr = (float*)&result;

  // Process elements in chunks of 4 for 16 elements total
  vl = __riscv_vsetvli(4, RV32, e32, m1);
  vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
  vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
  __riscv_vse32_v_f32m1(&res_ptr[0], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
  vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
  __riscv_vse32_v_f32m1(&res_ptr[4], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
  vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
  __riscv_vse32_v_f32m1(&res_ptr[8], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
  vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
  __riscv_vse32_v_f32m1(&res_ptr[12], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - negate using XOR with sign bit mask
  __m128 vec_a, vec_res;
  const __m128 sign_mask = (__m128)__lsx_vldrepl_w(&wcn_minus_zero_bits, 0); // Load -0.0f as bit pattern

  // Process all 16 elements in groups of 4
  vec_a = (__m128)__lsx_vld(&mat.m[0], 0);
  vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
  __lsx_vst((__m128i)vec_res, &result.m[0], 0);

  vec_a = (__m128)__lsx_vld(&mat.m[4], 0);
  vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
  __lsx_vst((__m128i)vec_res, &result.m[4], 0);

  vec_a = (__m128)__lsx_vld(&mat.m[8], 0);
  vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
  __lsx_vst((__m128i)vec_res, &result.m[8], 0);

  vec_a = (__m128)__lsx_vld(&mat.m[12], 0);
  vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
  __lsx_vst((__m128i)vec_res, &result.m[12], 0);

#else
  // Scalar fallback
  return WMATH_SET(Mat4)(
      WMATH_COPY(Mat4)(mat),                           // Self(Mat4)
      -mat.m[0], -mat.m[1], -mat.m[2], -mat.m[3],      // 00 ~ 03
      -mat.m[4], -mat.m[5], -mat.m[6], -mat.m[7],      // 10 ~ 13
      -mat.m[8], -mat.m[9], -mat.m[10], -mat.m[11],    // 20 ~ 23
      -mat.m[12], -mat.m[13], -mat.m[14], -mat.m[15]); // 30 ~ 33
#endif

  return result;
}

bool WMATH_EQUALS(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] &&
          a.m[3] == b.m[3] && //
          a.m[4] == b.m[4] && a.m[5] == b.m[5] && a.m[6] == b.m[6] &&
          a.m[7] == b.m[7] && //
          a.m[8] == b.m[8] && a.m[9] == b.m[9] && a.m[10] == b.m[10] &&
          a.m[11] == b.m[11] && //
          a.m[12] == b.m[12] && a.m[13] == b.m[13] && a.m[14] == b.m[14] &&
          a.m[15] == b.m[15] //
  );
}

bool WMATH_EQUALS_APPROXIMATELY(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  return (fabsf(a.m[0] - b.m[0]) < WCN_GET_EPSILON() &&
          fabsf(a.m[1] - b.m[1]) < WCN_GET_EPSILON() &&
          fabsf(a.m[2] - b.m[2]) < WCN_GET_EPSILON() &&
          fabsf(a.m[3] - b.m[3]) < WCN_GET_EPSILON() && //
          fabsf(a.m[4] - b.m[4]) < WCN_GET_EPSILON() &&
          fabsf(a.m[5] - b.m[5]) < WCN_GET_EPSILON() &&
          fabsf(a.m[6] - b.m[6]) < WCN_GET_EPSILON() &&
          fabsf(a.m[7] - b.m[7]) < WCN_GET_EPSILON() && //
          fabsf(a.m[8] - b.m[8]) < WCN_GET_EPSILON() &&
          fabsf(a.m[9] - b.m[9]) < WCN_GET_EPSILON() &&
          fabsf(a.m[10] - b.m[10]) < WCN_GET_EPSILON() &&
          fabsf(a.m[11] - b.m[11]) < WCN_GET_EPSILON() && //
          fabsf(a.m[12] - b.m[12]) < WCN_GET_EPSILON() &&
          fabsf(a.m[13] - b.m[13]) < WCN_GET_EPSILON() &&
          fabsf(a.m[14] - b.m[14]) < WCN_GET_EPSILON() &&
          fabsf(a.m[15] - b.m[15]) < WCN_GET_EPSILON() //
  );
}

// + add - Mat4
WMATH_TYPE(Mat4) WMATH_ADD(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  WMATH_TYPE(Mat4) result;
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 vec_a = _mm_loadu_ps(&a.m[0]);
  __m128 vec_b = _mm_loadu_ps(&b.m[0]);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[0], vec_res);
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_b = _mm_loadu_ps(&b.m[4]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[4], vec_res);
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_b = _mm_loadu_ps(&b.m[8]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[8], vec_res);
  vec_a = _mm_loadu_ps(&a.m[12]);
  vec_b = _mm_loadu_ps(&b.m[12]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t vec_a, vec_b, vec_res;
  vec_a = vld1q_f32(&a.m[0]);
  vec_b = vld1q_f32(&b.m[0]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[0], vec_res);
  vec_a = vld1q_f32(&a.m[4]);
  vec_b = vld1q_f32(&b.m[4]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[4], vec_res);
  vec_a = vld1q_f32(&a.m[8]);
  vec_b = vld1q_f32(&b.m[8]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[8], vec_res);
  vec_a = vld1q_f32(&a.m[12]);
  vec_b = vld1q_f32(&b.m[12]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  v128_t vec_a, vec_b, vec_res;
  vec_a = wasm_v128_load(&a.m[0]);
  vec_b = wasm_v128_load(&b.m[0]);
  vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(&result.m[0], vec_res);
  vec_a = wasm_v128_load(&a.m[4]);
  vec_b = wasm_v128_load(&b.m[4]);
  vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(&result.m[4], vec_res);
  vec_a = wasm_v128_load(&a.m[8]);
  vec_b = wasm_v128_load(&b.m[8]);
  vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(&result.m[8], vec_res);
  vec_a = wasm_v128_load(&a.m[12]);
  vec_b = wasm_v128_load(&b.m[12]);
  vec_res = wasm_f32x4_add(vec_a, vec_b);
  wasm_v128_store(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  size_t vl;
  vfloat32m1_t vec_a, vec_b, vec_res;
  float *a_ptr = (float*)&a;
  float *b_ptr = (float*)&b;
  float *res_ptr = (float*)&result;

  // Process elements in chunks of 4 for 16 elements total
  vl = __riscv_vsetvli(4, RV32, e32, m1);
  vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
  vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[0], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
  vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[4], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
  vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[8], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
  vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  __m128 vec_a, vec_b, vec_res;
  vec_a = (__m128)__lsx_vld(&a.m[0], 0);
  vec_b = (__m128)__lsx_vld(&b.m[0], 0);
  vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[0], 0);
  vec_a = (__m128)__lsx_vld(&a.m[4], 0);
  vec_b = (__m128)__lsx_vld(&b.m[4], 0);
  vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[4], 0);
  vec_a = (__m128)__lsx_vld(&a.m[8], 0);
  vec_b = (__m128)__lsx_vld(&b.m[8], 0);
  vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[8], 0);
  vec_a = (__m128)__lsx_vld(&a.m[12], 0);
  vec_b = (__m128)__lsx_vld(&b.m[12], 0);
  vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[12], 0);
#else
  for (int i = 0; i < 16; ++i) {
    result.m[i] = a.m[i] + b.m[i];
  }
#endif
  return result;
}

WMATH_TYPE(Mat4) WMATH_SUB(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  WMATH_TYPE(Mat4) result;
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 vec_a = _mm_loadu_ps(&a.m[0]);
  __m128 vec_b = _mm_loadu_ps(&b.m[0]);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[0], vec_res);
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_b = _mm_loadu_ps(&b.m[4]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[4], vec_res);
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_b = _mm_loadu_ps(&b.m[8]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[8], vec_res);
  vec_a = _mm_loadu_ps(&a.m[12]);
  vec_b = _mm_loadu_ps(&b.m[12]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t vec_a, vec_b, vec_res;
  vec_a = vld1q_f32(&a.m[0]);
  vec_b = vld1q_f32(&b.m[0]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[0], vec_res);
  vec_a = vld1q_f32(&a.m[4]);
  vec_b = vld1q_f32(&b.m[4]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[4], vec_res);
  vec_a = vld1q_f32(&a.m[8]);
  vec_b = vld1q_f32(&b.m[8]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[8], vec_res);
  vec_a = vld1q_f32(&a.m[12]);
  vec_b = vld1q_f32(&b.m[12]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  v128_t vec_a, vec_b, vec_res;
  vec_a = wasm_v128_load(&a.m[0]);
  vec_b = wasm_v128_load(&b.m[0]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[0], vec_res);
  vec_a = wasm_v128_load(&a.m[4]);
  vec_b = wasm_v128_load(&b.m[4]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[4], vec_res);
  vec_a = wasm_v128_load(&a.m[8]);
  vec_b = wasm_v128_load(&b.m[8]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[8], vec_res);
  vec_a = wasm_v128_load(&a.m[12]);
  vec_b = wasm_v128_load(&b.m[12]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  size_t vl;
  vfloat32m1_t vec_a, vec_b, vec_res;
  float *a_ptr = (float*)&a;
  float *b_ptr = (float*)&b;
  float *res_ptr = (float*)&result;

  // Process elements in chunks of 4 for 16 elements total
  vl = __riscv_vsetvli(4, RV32, e32, m1);
  vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[0], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[4], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[8], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
  vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
  __riscv_vse32_v_f32m1(&res_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  __m128 vec_a, vec_b, vec_res;
  vec_a = (__m128)__lsx_vld(&a.m[0], 0);
  vec_b = (__m128)__lsx_vld(&b.m[0], 0);
  vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[0], 0);
  vec_a = (__m128)__lsx_vld(&a.m[4], 0);
  vec_b = (__m128)__lsx_vld(&b.m[4], 0);
  vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[4], 0);
  vec_a = (__m128)__lsx_vld(&a.m[8], 0);
  vec_b = (__m128)__lsx_vld(&b.m[8], 0);
  vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[8], 0);
  vec_a = (__m128)__lsx_vld(&a.m[12], 0);
  vec_b = (__m128)__lsx_vld(&b.m[12], 0);
  vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
  __lsx_vst((__m128i)vec_res, &result.m[12], 0);
#else
  for (int i = 0; i < 16; ++i) {
    result.m[i] = a.m[i] - b.m[i];
  }
#endif
  return result;
}

// .* Mat4

WMATH_TYPE(Mat4) WMATH_MULTIPLY_SCALAR(Mat4)(WMATH_TYPE(Mat4) a, float b) {
  WMATH_TYPE(Mat4) result;
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  __m128 vec_b_scalar = _mm_set1_ps(b);
  __m128 vec_a = _mm_loadu_ps(&a.m[0]);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
  _mm_storeu_ps(&result.m[0], vec_res);
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
  _mm_storeu_ps(&result.m[4], vec_res);
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
  _mm_storeu_ps(&result.m[8], vec_res);
  vec_a = _mm_loadu_ps(&a.m[12]);
  vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
  _mm_storeu_ps(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  float32x4_t vec_a, vec_b_scalar, vec_res;
  vec_b_scalar = vdupq_n_f32(b);
  vec_a = vld1q_f32(&a.m[0]);
  vec_res = vmulq_f32(vec_a, vec_b_scalar);
  vst1q_f32(&result.m[0], vec_res);
  vec_a = vld1q_f32(&a.m[4]);
  vec_res = vmulq_f32(vec_a, vec_b_scalar);
  vst1q_f32(&result.m[4], vec_res);
  vec_a = vld1q_f32(&a.m[8]);
  vec_res = vmulq_f32(vec_a, vec_b_scalar);
  vst1q_f32(&result.m[8], vec_res);
  vec_a = vld1q_f32(&a.m[12]);
  vec_res = vmulq_f32(vec_a, vec_b_scalar);
  vst1q_f32(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  v128_t vec_b_scalar = wasm_f32x4_splat(b);
  v128_t vec_a = wasm_v128_load(&a.m[0]);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
  wasm_v128_store(&result.m[0], vec_res);
  vec_a = wasm_v128_load(&a.m[4]);
  vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
  wasm_v128_store(&result.m[4], vec_res);
  vec_a = wasm_v128_load(&a.m[8]);
  vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
  wasm_v128_store(&result.m[8], vec_res);
  vec_a = wasm_v128_load(&a.m[12]);
  vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
  wasm_v128_store(&result.m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  size_t vl;
  vfloat32m1_t vec_a, vec_b_scalar, vec_res;
  float *a_ptr = (float*)&a;
  float *res_ptr = (float*)&result;

  // Process elements in chunks of 4 for 16 elements total
  vl = __riscv_vsetvli(4, RV32, e32, m1);
  vec_b_scalar = __riscv_vfmv_v_f_f32m1(b, vl);
  vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
  __riscv_vse32_v_f32m1(&res_ptr[0], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
  __riscv_vse32_v_f32m1(&res_ptr[4], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
  __riscv_vse32_v_f32m1(&res_ptr[8], vec_res, vl);

  vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
  __riscv_vse32_v_f32m1(&res_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  __m128 vec_b_scalar = __lsx_vldrepl_w(&b, 0);
  __m128 vec_a = (__m128)__lsx_vld(&a.m[0], 0);
  __m128 vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
  __lsx_vst((__m128i)vec_res, &result.m[0], 0);
  vec_a = (__m128)__lsx_vld(&a.m[4], 0);
  vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
  __lsx_vst((__m128i)vec_res, &result.m[4], 0);
  vec_a = (__m128)__lsx_vld(&a.m[8], 0);
  vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
  __lsx_vst((__m128i)vec_res, &result.m[8], 0);
  vec_a = (__m128)__lsx_vld(&a.m[12], 0);
  vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
  __lsx_vst((__m128i)vec_res, &result.m[12], 0);
#else
  for (int i = 0; i < 16; ++i) {
    result.m[i] = a.m[i] * b;
  }
#endif
  return result;
}

// * Mat4

WMATH_TYPE(Mat4)
WMATH_MULTIPLY(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

  // SSE implementation for proper matrix multiplication
  // Load matrix B columns for efficient access
  __m128 b_col0 = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
  __m128 b_col1 = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
  __m128 b_col2 = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
  __m128 b_col3 = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);

  // Calculate each row of the result matrix
  for (int i = 0; i < 4; i++) {
    __m128 a_row = _mm_loadu_ps(&a.m[i * 4]);

    // Calculate result.m[i][0] = dot(a_row, b_col0)
    __m128 temp = _mm_mul_ps(a_row, b_col0);
    temp = _mm_hadd_ps(temp, temp);
    temp = _mm_hadd_ps(temp, temp);
    result.m[i * 4 + 0] = _mm_cvtss_f32(temp);

    // Calculate result.m[i][1] = dot(a_row, b_col1)
    temp = _mm_mul_ps(a_row, b_col1);
    temp = _mm_hadd_ps(temp, temp);
    temp = _mm_hadd_ps(temp, temp);
    result.m[i * 4 + 1] = _mm_cvtss_f32(temp);

    // Calculate result.m[i][2] = dot(a_row, b_col2)
    temp = _mm_mul_ps(a_row, b_col2);
    temp = _mm_hadd_ps(temp, temp);
    temp = _mm_hadd_ps(temp, temp);
    result.m[i * 4 + 2] = _mm_cvtss_f32(temp);

    // Calculate result.m[i][3] = dot(a_row, b_col3)
    temp = _mm_mul_ps(a_row, b_col3);
    temp = _mm_hadd_ps(temp, temp);
    temp = _mm_hadd_ps(temp, temp);
    result.m[i * 4 + 3] = _mm_cvtss_f32(temp);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON optimized matrix multiplication
  float32x4_t row, col, prod, sum;

  // Calculate the first row of a result
  row = vld1q_f32(&a.m[0]); // Load the first row of matrix a

  // result.m[0] = a.m[0]*b.m[0] + a.m[1]*b.m[4] + a.m[2]*b.m[8] +
  // a.m[3]*b.m[12]
  float32x4_t col0 = {b.m[0], b.m[4], b.m[8], b.m[12]};
  prod = vmulq_f32(row, col0);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[0] = vgetq_lane_f32(sum, 0);

  // result.m[1] = a.m[0]*b.m[1] + a.m[1]*b.m[5] + a.m[2]*b.m[9] +
  // a.m[3]*b.m[13]
  float32x4_t col1 = {b.m[1], b.m[5], b.m[9], b.m[13]};
  prod = vmulq_f32(row, col1);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[1] = vgetq_lane_f32(sum, 0);

  // result.m[2] = a.m[0]*b.m[2] + a.m[1]*b.m[6] + a.m[2]*b.m[10] +
  // a.m[3]*b.m[14]
  float32x4_t col2 = {b.m[2], b.m[6], b.m[10], b.m[14]};
  prod = vmulq_f32(row, col2);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[2] = vgetq_lane_f32(sum, 0);

  // result.m[3] = a.m[0]*b.m[3] + a.m[1]*b.m[7] + a.m[2]*b.m[11] +
  // a.m[3]*b.m[15]
  float32x4_t col3 = {b.m[3], b.m[7], b.m[11], b.m[15]};
  prod = vmulq_f32(row, col3);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[3] = vgetq_lane_f32(sum, 0);

  // Calculate the second row of a result
  row = vld1q_f32(&a.m[4]); // Load the second row of matrix a

  // result.m[4] = a.m[4]*b.m[0] + a.m[5]*b.m[4] + a.m[6]*b.m[8] +
  // a.m[7]*b.m[12]
  prod = vmulq_f32(row, col0);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[4] = vgetq_lane_f32(sum, 0);

  // result.m[5] = a.m[4]*b.m[1] + a.m[5]*b.m[5] + a.m[6]*b.m[9] +
  // a.m[7]*b.m[13]
  prod = vmulq_f32(row, col1);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[5] = vgetq_lane_f32(sum, 0);

  // result.m[6] = a.m[4]*b.m[2] + a.m[5]*b.m[6] + a.m[6]*b.m[10] +
  // a.m[7]*b.m[14]
  prod = vmulq_f32(row, col2);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[6] = vgetq_lane_f32(sum, 0);

  // result.m[7] = a.m[4]*b.m[3] + a.m[5]*b.m[7] + a.m[6]*b.m[11] +
  // a.m[7]*b.m[15]
  prod = vmulq_f32(row, col3);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[7] = vgetq_lane_f32(sum, 0);

  // Calculate the third row of a result
  row = vld1q_f32(&a.m[8]); // Load the third row of matrix a

  // result.m[8] = a.m[8]*b.m[0] + a.m[9]*b.m[4] + a.m[10]*b.m[8] +
  // a.m[11]*b.m[12]
  prod = vmulq_f32(row, col0);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[8] = vgetq_lane_f32(sum, 0);

  // result.m[9] = a.m[8]*b.m[1] + a.m[9]*b.m[5] + a.m[10]*b.m[9] +
  // a.m[11]*b.m[13]
  prod = vmulq_f32(row, col1);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[9] = vgetq_lane_f32(sum, 0);

  // result.m[10] = a.m[8]*b.m[2] + a.m[9]*b.m[6] + a.m[10]*b.m[10] +
  // a.m[11]*b.m[14]
  prod = vmulq_f32(row, col2);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[10] = vgetq_lane_f32(sum, 0);

  // result.m[11] = a.m[8]*b.m[3] + a.m[9]*b.m[7] + a.m[10]*b.m[11] +
  // a.m[11]*b.m[15]
  prod = vmulq_f32(row, col3);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[11] = vgetq_lane_f32(sum, 0);

  // Calculate the fourth row of a result
  row = vld1q_f32(&a.m[12]); // Load the fourth row of matrix a

  // result.m[12] = a.m[12]*b.m[0] + a.m[13]*b.m[4] + a.m[14]*b.m[8] +
  // a.m[15]*b.m[12]
  prod = vmulq_f32(row, col0);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[12] = vgetq_lane_f32(sum, 0);

  // result.m[13] = a.m[12]*b.m[1] + a.m[13]*b.m[5] + a.m[14]*b.m[9] +
  // a.m[15]*b.m[13]
  prod = vmulq_f32(row, col1);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[13] = vgetq_lane_f32(sum, 0);

  // result.m[14] = a.m[12]*b.m[2] + a.m[13]*b.m[6] + a.m[14]*b.m[10] +
  // a.m[15]*b.m[14]
  prod = vmulq_f32(row, col2);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[14] = vgetq_lane_f32(sum, 0);

  // result.m[15] = a.m[12]*b.m[3] + a.m[13]*b.m[7] + a.m[14]*b.m[11] +
  // a.m[15]*b.m[15]
  prod = vmulq_f32(row, col3);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[15] = vgetq_lane_f32(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation for matrix multiplication
  // Load matrix B columns for efficient access
  v128_t b_col0 = wasm_f32x4_make(b.m[0], b.m[4], b.m[8], b.m[12]);
  v128_t b_col1 = wasm_f32x4_make(b.m[1], b.m[5], b.m[9], b.m[13]);
  v128_t b_col2 = wasm_f32x4_make(b.m[2], b.m[6], b.m[10], b.m[14]);
  v128_t b_col3 = wasm_f32x4_make(b.m[3], b.m[7], b.m[11], b.m[15]);

  // Calculate each row of the result matrix
  for (int i = 0; i < 4; i++) {
    v128_t a_row = wasm_v128_load(&a.m[i * 4]);

    // Calculate result.m[i][0] = dot(a_row, b_col0)
    v128_t temp = wasm_f32x4_mul(a_row, b_col0);
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
    result.m[i * 4 + 0] = wasm_f32x4_extract_lane(temp, 0);

    // Calculate result.m[i][1] = dot(a_row, b_col1)
    temp = wasm_f32x4_mul(a_row, b_col1);
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
    result.m[i * 4 + 1] = wasm_f32x4_extract_lane(temp, 0);

    // Calculate result.m[i][2] = dot(a_row, b_col2)
    temp = wasm_f32x4_mul(a_row, b_col2);
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
    result.m[i * 4 + 2] = wasm_f32x4_extract_lane(temp, 0);

    // Calculate result.m[i][3] = dot(a_row, b_col3)
    temp = wasm_f32x4_mul(a_row, b_col3);
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
    temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
    result.m[i * 4 + 3] = wasm_f32x4_extract_lane(temp, 0);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V implementation for matrix multiplication
  // Since matrix multiplication is more complex, we'll use scalar for now but implement the basic SIMD operations
  // This is a simplified implementation for demonstration
  size_t vl = __riscv_vsetvli(4, RV32, e32, m1);
  vfloat32m1_t a_row, b_col, temp_vec, sum_vec;
  float dot_result;

  // Calculate each row of the result matrix
  for (int i = 0; i < 4; i++) {
    a_row = __riscv_vle32_v_f32m1(&a.m[i * 4], vl);

    // Calculate result.m[i][0] = dot(a_row, column 0 of b)
    b_col = __riscv_vcreate_v_f32m1(b.m[0 + 4*0], b.m[1 + 4*0], b.m[2 + 4*0], b.m[3 + 4*0]);
    temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
    sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    result.m[i * 4 + 0] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

    // Calculate result.m[i][1] = dot(a_row, column 1 of b)
    b_col = __riscv_vcreate_v_f32m1(b.m[0 + 4*1], b.m[1 + 4*1], b.m[2 + 4*1], b.m[3 + 4*1]);
    temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
    sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    result.m[i * 4 + 1] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

    // Calculate result.m[i][2] = dot(a_row, column 2 of b)
    b_col = __riscv_vcreate_v_f32m1(b.m[0 + 4*2], b.m[1 + 4*2], b.m[2 + 4*2], b.m[3 + 4*2]);
    temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
    sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    result.m[i * 4 + 2] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

    // Calculate result.m[i][3] = dot(a_row, column 3 of b)
    b_col = __riscv_vcreate_v_f32m1(b.m[0 + 4*3], b.m[1 + 4*3], b.m[2 + 4*3], b.m[3 + 4*3]);
    temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
    sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    result.m[i * 4 + 3] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation for matrix multiplication
  __m128 a_row, b_col, temp_vec;
  float dot_result;

  // Calculate each row of the result matrix
  for (int i = 0; i < 4; i++) {
    a_row = (__m128)__lsx_vld(&a.m[i * 4], 0);

    // Calculate result.m[i][0] = dot(a_row, column 0 of b)
    b_col = __lsx_vldrepl_w(&b.m[0 + 4*0], 0);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4*0], 1);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4*0], 2);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4*0], 3);
    temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
    dot_result = __lsx_vfmov_s_f(temp_vec);
    dot_result += __lsx_vextract_float(temp_vec, 1);
    dot_result += __lsx_vextract_float(temp_vec, 2);
    dot_result += __lsx_vextract_float(temp_vec, 3);
    result.m[i * 4 + 0] = dot_result;

    // Calculate result.m[i][1] = dot(a_row, column 1 of b)
    b_col = __lsx_vldrepl_w(&b.m[0 + 4*1], 0);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4*1], 1);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4*1], 2);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4*1], 3);
    temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
    dot_result = __lsx_vfmov_s_f(temp_vec);
    dot_result += __lsx_vextract_float(temp_vec, 1);
    dot_result += __lsx_vextract_float(temp_vec, 2);
    dot_result += __lsx_vextract_float(temp_vec, 3);
    result.m[i * 4 + 1] = dot_result;

    // Calculate result.m[i][2] = dot(a_row, column 2 of b)
    b_col = __lsx_vldrepl_w(&b.m[0 + 4*2], 0);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4*2], 1);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4*2], 2);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4*2], 3);
    temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
    dot_result = __lsx_vfmov_s_f(temp_vec);
    dot_result += __lsx_vextract_float(temp_vec, 1);
    dot_result += __lsx_vextract_float(temp_vec, 2);
    dot_result += __lsx_vextract_float(temp_vec, 3);
    result.m[i * 4 + 2] = dot_result;

    // Calculate result.m[i][3] = dot(a_row, column 3 of b)
    b_col = __lsx_vldrepl_w(&b.m[0 + 4*3], 0);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4*3], 1);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4*3], 2);
    b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4*3], 3);
    temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
    dot_result = __lsx_vfmov_s_f(temp_vec);
    dot_result += __lsx_vextract_float(temp_vec, 1);
    dot_result += __lsx_vextract_float(temp_vec, 2);
    dot_result += __lsx_vextract_float(temp_vec, 3);
    result.m[i * 4 + 3] = dot_result;
  }

#else
  // Scalar fallback implementation
  result.m[0] =
      a.m[0] * b.m[0] + a.m[1] * b.m[4] + a.m[2] * b.m[8] + a.m[3] * b.m[12];
  result.m[1] =
      a.m[0] * b.m[1] + a.m[1] * b.m[5] + a.m[2] * b.m[9] + a.m[3] * b.m[13];
  result.m[2] =
      a.m[0] * b.m[2] + a.m[1] * b.m[6] + a.m[2] * b.m[10] + a.m[3] * b.m[14];
  result.m[3] =
      a.m[0] * b.m[3] + a.m[1] * b.m[7] + a.m[2] * b.m[11] + a.m[3] * b.m[15];

  result.m[4] =
      a.m[4] * b.m[0] + a.m[5] * b.m[4] + a.m[6] * b.m[8] + a.m[7] * b.m[12];
  result.m[5] =
      a.m[4] * b.m[1] + a.m[5] * b.m[5] + a.m[6] * b.m[9] + a.m[7] * b.m[13];
  result.m[6] =
      a.m[4] * b.m[2] + a.m[5] * b.m[6] + a.m[6] * b.m[10] + a.m[7] * b.m[14];
  result.m[7] =
      a.m[4] * b.m[3] + a.m[5] * b.m[7] + a.m[6] * b.m[11] + a.m[7] * b.m[15];

  result.m[8] =
      a.m[8] * b.m[0] + a.m[9] * b.m[4] + a.m[10] * b.m[8] + a.m[11] * b.m[12];
  result.m[9] =
      a.m[8] * b.m[1] + a.m[9] * b.m[5] + a.m[10] * b.m[9] + a.m[11] * b.m[13];
  result.m[10] =
      a.m[8] * b.m[2] + a.m[9] * b.m[6] + a.m[10] * b.m[10] + a.m[11] * b.m[14];
  result.m[11] =
      a.m[8] * b.m[3] + a.m[9] * b.m[7] + a.m[10] * b.m[11] + a.m[11] * b.m[15];

  result.m[12] = a.m[12] * b.m[0] + a.m[13] * b.m[4] + a.m[14] * b.m[8] +
                 a.m[15] * b.m[12];
  result.m[13] = a.m[12] * b.m[1] + a.m[13] * b.m[5] + a.m[14] * b.m[9] +
                 a.m[15] * b.m[13];
  result.m[14] = a.m[12] * b.m[2] + a.m[13] * b.m[6] + a.m[14] * b.m[10] +
                 a.m[15] * b.m[14];
  result.m[15] = a.m[12] * b.m[3] + a.m[13] * b.m[7] + a.m[14] * b.m[11] +
                 a.m[15] * b.m[15];
#endif

  return result;
}

WMATH_TYPE(Mat4)
WMATH_INVERSE(Mat4)(WMATH_TYPE(Mat4) a) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized version using SIMD for parallel computation
  // Load matrix rows
  _mm_loadu_ps(&a.m[0]);
  _mm_loadu_ps(&a.m[4]);
  _mm_loadu_ps(&a.m[8]);
  _mm_loadu_ps(&a.m[12]);

  // Extract elements for computation
  float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
  float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
  float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
  float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

  // Calculate temporary values using SIMD where possible
  __m128 vec_m22 = _mm_set1_ps(m_22);
  __m128 vec_m33 = _mm_set1_ps(m_33);
  __m128 vec_m32 = _mm_set1_ps(m_32);
  __m128 vec_m23 = _mm_set1_ps(m_23);
  __m128 vec_m12 = _mm_set1_ps(m_12);
  __m128 vec_m13 = _mm_set1_ps(m_13);

  // Calculate tmp values using SIMD
  __m128 tmp_0 = _mm_mul_ps(vec_m22, vec_m33);
  __m128 tmp_1 = _mm_mul_ps(vec_m32, vec_m23);
  __m128 tmp_2 = _mm_mul_ps(vec_m12, vec_m33);
  __m128 tmp_3 = _mm_mul_ps(vec_m32, vec_m13);

  // Extract scalar values
  float tmp_0_val = _mm_cvtss_f32(tmp_0);
  float tmp_1_val = _mm_cvtss_f32(tmp_1);
  float tmp_2_val = _mm_cvtss_f32(tmp_2);
  float tmp_3_val = _mm_cvtss_f32(tmp_3);

  // Continue with remaining scalar calculations (complex cross terms)
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  // Calculate t values
  float t_0 = (tmp_0_val * m_11 + tmp_3_val * m_21 + tmp_4 * m_31) -
              (tmp_1_val * m_11 + tmp_2_val * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1_val * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0_val * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2_val * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3_val * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  // Calculate determinant using SIMD
  __m128 vec_m00 = _mm_set1_ps(m_00);
  __m128 vec_m10 = _mm_set1_ps(m_10);
  __m128 vec_m20 = _mm_set1_ps(m_20);
  __m128 vec_m30 = _mm_set1_ps(m_30);
  __m128 vec_t0 = _mm_set1_ps(t_0);
  __m128 vec_t1 = _mm_set1_ps(t_1);
  __m128 vec_t2 = _mm_set1_ps(t_2);
  __m128 vec_t3 = _mm_set1_ps(t_3);

  __m128 det_part0 = _mm_mul_ps(vec_m00, vec_t0);
  __m128 det_part1 = _mm_mul_ps(vec_m10, vec_t1);
  __m128 det_part2 = _mm_mul_ps(vec_m20, vec_t2);
  __m128 det_part3 = _mm_mul_ps(vec_m30, vec_t3);

  __m128 det_sum01 = _mm_add_ps(det_part0, det_part1);
  __m128 det_sum23 = _mm_add_ps(det_part2, det_part3);
  __m128 det_total = _mm_add_ps(det_sum01, det_sum23);

  float det = _mm_cvtss_f32(det_total);
  float d = 1.0f / det;

  // SIMD vector for the determinant reciprocal
  __m128 vec_d = _mm_set1_ps(d);

  // Calculate inverse matrix elements using SIMD where possible
  __m128 m_00_inv = _mm_mul_ps(vec_d, vec_t0);
  __m128 m_01_inv = _mm_mul_ps(vec_d, vec_t1);
  __m128 m_02_inv = _mm_mul_ps(vec_d, vec_t2);
  __m128 m_03_inv = _mm_mul_ps(vec_d, vec_t3);

  // Extract the first row values
  float m_00_val = _mm_cvtss_f32(m_00_inv);
  float m_01_val = _mm_cvtss_f32(m_01_inv);
  float m_02_val = _mm_cvtss_f32(m_02_inv);
  float m_03_val = _mm_cvtss_f32(m_03_inv);

  // Continue with scalar calculations for complex elements
  float m_10_val = d * ((tmp_1_val * m_10 + tmp_2_val * m_20 + tmp_5 * m_30) -
                        (tmp_0_val * m_10 + tmp_3_val * m_20 + tmp_4 * m_30));
  float m_11_val = d * ((tmp_0_val * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                        (tmp_1_val * m_00 + tmp_6 * m_20 + tmp_9 * m_30));
  float m_12_val = d * ((tmp_3_val * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                        (tmp_2_val * m_00 + tmp_7 * m_10 + tmp_10 * m_30));
  float m_13_val = d * ((tmp_2_val * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                        (tmp_3_val * m_10 + tmp_4 * m_20 + tmp_9 * m_30));
  float m_20_val = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                        (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33));
  float m_21_val = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                        (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33));
  float m_22_val = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                        (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33));
  float m_23_val = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                        (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23));
  float m_30_val = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                        (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22));
  float m_31_val = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                        (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02));
  float m_32_val = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                        (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12));
  float m_33_val = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                        (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02));

  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){.m_00 = m_00_val,
                                                      .m_01 = m_01_val,
                                                      .m_02 = m_02_val,
                                                      .m_03 = m_03_val,
                                                      .m_10 = m_10_val,
                                                      .m_11 = m_11_val,
                                                      .m_12 = m_12_val,
                                                      .m_13 = m_13_val,
                                                      .m_20 = m_20_val,
                                                      .m_21 = m_21_val,
                                                      .m_22 = m_22_val,
                                                      .m_23 = m_23_val,
                                                      .m_30 = m_30_val,
                                                      .m_31 = m_31_val,
                                                      .m_32 = m_32_val,
                                                      .m_33 = m_33_val});

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized version using SIMD for parallel computation
  // Load matrix rows
  float32x4_t row0 = vld1q_f32(&a.m[0]);
  float32x4_t row1 = vld1q_f32(&a.m[4]);
  float32x4_t row2 = vld1q_f32(&a.m[8]);
  float32x4_t row3 = vld1q_f32(&a.m[12]);

  // Extract elements for computation
  float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
  float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
  float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
  float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

  // Calculate temporary values using SIMD where possible
  float32x4_t vec_m22 = vdupq_n_f32(m_22);
  float32x4_t vec_m33 = vdupq_n_f32(m_33);
  float32x4_t vec_m32 = vdupq_n_f32(m_32);
  float32x4_t vec_m23 = vdupq_n_f32(m_23);
  float32x4_t vec_m12 = vdupq_n_f32(m_12);
  float32x4_t vec_m13 = vdupq_n_f32(m_13);

  // Calculate tmp values using SIMD
  float32x4_t tmp_0 = vmulq_f32(vec_m22, vec_m33);
  float32x4_t tmp_1 = vmulq_f32(vec_m32, vec_m23);
  float32x4_t tmp_2 = vmulq_f32(vec_m12, vec_m33);
  float32x4_t tmp_3 = vmulq_f32(vec_m32, vec_m13);

  // Extract scalar values
  float tmp_0_val = vgetq_lane_f32(tmp_0, 0);
  float tmp_1_val = vgetq_lane_f32(tmp_1, 0);
  float tmp_2_val = vgetq_lane_f32(tmp_2, 0);
  float tmp_3_val = vgetq_lane_f32(tmp_3, 0);

  // Continue with remaining scalar calculations (complex cross terms)
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  // Calculate t values
  float t_0 = (tmp_0_val * m_11 + tmp_3_val * m_21 + tmp_4 * m_31) -
              (tmp_1_val * m_11 + tmp_2_val * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1_val * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0_val * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2_val * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3_val * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  // Calculate determinant using SIMD
  float32x4_t vec_m00 = vdupq_n_f32(m_00);
  float32x4_t vec_m10 = vdupq_n_f32(m_10);
  float32x4_t vec_m20 = vdupq_n_f32(m_20);
  float32x4_t vec_m30 = vdupq_n_f32(m_30);
  float32x4_t vec_t0 = vdupq_n_f32(t_0);
  float32x4_t vec_t1 = vdupq_n_f32(t_1);
  float32x4_t vec_t2 = vdupq_n_f32(t_2);
  float32x4_t vec_t3 = vdupq_n_f32(t_3);

  float32x4_t det_part0 = vmulq_f32(vec_m00, vec_t0);
  float32x4_t det_part1 = vmulq_f32(vec_m10, vec_t1);
  float32x4_t det_part2 = vmulq_f32(vec_m20, vec_t2);
  float32x4_t det_part3 = vmulq_f32(vec_m30, vec_t3);

  float32x4_t det_sum01 = vaddq_f32(det_part0, det_part1);
  float32x4_t det_sum23 = vaddq_f32(det_part2, det_part3);
  float32x4_t det_total = vaddq_f32(det_sum01, det_sum23);

  float det = vgetq_lane_f32(det_total, 0);
  float d = 1.0f / det;

  // Calculate inverse matrix elements
  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = d * t_0,
      .m_01 = d * t_1,
      .m_02 = d * t_2,
      .m_03 = d * t_3,
      .m_10 = d * ((tmp_1_val * m_10 + tmp_2_val * m_20 + tmp_5 * m_30) -
                   (tmp_0_val * m_10 + tmp_3_val * m_20 + tmp_4 * m_30)),
      .m_11 = d * ((tmp_0_val * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                   (tmp_1_val * m_00 + tmp_6 * m_20 + tmp_9 * m_30)),
      .m_12 = d * ((tmp_3_val * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                   (tmp_2_val * m_00 + tmp_7 * m_10 + tmp_10 * m_30)),
      .m_13 = d * ((tmp_2_val * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                   (tmp_3_val * m_10 + tmp_4 * m_20 + tmp_9 * m_30)),
      .m_20 = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                   (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33)),
      .m_21 = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                   (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33)),
      .m_22 = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                   (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33)),
      .m_23 = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                   (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23)),
      .m_30 = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                   (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22)),
      .m_31 = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                   (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02)),
      .m_32 = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                   (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12)),
      .m_33 = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                   (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02))});

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - using scalar calculations with vector operations where possible
  float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
  float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
  float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
  float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

  // Calculate temporary values
  float tmp_0 = m_22 * m_33;
  float tmp_1 = m_32 * m_23;
  float tmp_2 = m_12 * m_33;
  float tmp_3 = m_32 * m_13;
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  float t_0 = (tmp_0 * m_11 + tmp_3 * m_21 + tmp_4 * m_31) -
              (tmp_1 * m_11 + tmp_2 * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1 * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0 * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2 * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3 * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  float det = m_00 * t_0 + m_10 * t_1 + m_20 * t_2 + m_30 * t_3;
  float d = 1.0f / det;

  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = d * t_0,
      .m_01 = d * t_1,
      .m_02 = d * t_2,
      .m_03 = d * t_3,
      .m_10 = d * ((tmp_1 * m_10 + tmp_2 * m_20 + tmp_5 * m_30) -
                   (tmp_0 * m_10 + tmp_3 * m_20 + tmp_4 * m_30)),
      .m_11 = d * ((tmp_0 * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                   (tmp_1 * m_00 + tmp_6 * m_20 + tmp_9 * m_30)),
      .m_12 = d * ((tmp_3 * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                   (tmp_2 * m_00 + tmp_7 * m_10 + tmp_10 * m_30)),
      .m_13 = d * ((tmp_2 * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                   (tmp_3 * m_10 + tmp_4 * m_20 + tmp_9 * m_30)),
      .m_20 = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                   (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33)),
      .m_21 = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                   (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33)),
      .m_22 = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                   (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33)),
      .m_23 = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                   (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23)),
      .m_30 = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                   (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22)),
      .m_31 = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                   (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02)),
      .m_32 = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                   (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12)),
      .m_33 = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                   (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02))});

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V implementation - using scalar calculations with vector operations where possible
  float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
  float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
  float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
  float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

  // Calculate temporary values
  float tmp_0 = m_22 * m_33;
  float tmp_1 = m_32 * m_23;
  float tmp_2 = m_12 * m_33;
  float tmp_3 = m_32 * m_13;
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  float t_0 = (tmp_0 * m_11 + tmp_3 * m_21 + tmp_4 * m_31) -
              (tmp_1 * m_11 + tmp_2 * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1 * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0 * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2 * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3 * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  float det = m_00 * t_0 + m_10 * t_1 + m_20 * t_2 + m_30 * t_3;
  float d = 1.0f / det;

  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = d * t_0,
      .m_01 = d * t_1,
      .m_02 = d * t_2,
      .m_03 = d * t_3,
      .m_10 = d * ((tmp_1 * m_10 + tmp_2 * m_20 + tmp_5 * m_30) -
                   (tmp_0 * m_10 + tmp_3 * m_20 + tmp_4 * m_30)),
      .m_11 = d * ((tmp_0 * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                   (tmp_1 * m_00 + tmp_6 * m_20 + tmp_9 * m_30)),
      .m_12 = d * ((tmp_3 * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                   (tmp_2 * m_00 + tmp_7 * m_10 + tmp_10 * m_30)),
      .m_13 = d * ((tmp_2 * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                   (tmp_3 * m_10 + tmp_4 * m_20 + tmp_9 * m_30)),
      .m_20 = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                   (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33)),
      .m_21 = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                   (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33)),
      .m_22 = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                   (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33)),
      .m_23 = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                   (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23)),
      .m_30 = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                   (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22)),
      .m_31 = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                   (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02)),
      .m_32 = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                   (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12)),
      .m_33 = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                   (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02))});

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - using scalar calculations with vector operations where possible
  float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
  float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
  float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
  float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

  // Calculate temporary values
  float tmp_0 = m_22 * m_33;
  float tmp_1 = m_32 * m_23;
  float tmp_2 = m_12 * m_33;
  float tmp_3 = m_32 * m_13;
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  float t_0 = (tmp_0 * m_11 + tmp_3 * m_21 + tmp_4 * m_31) -
              (tmp_1 * m_11 + tmp_2 * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1 * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0 * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2 * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3 * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  float det = m_00 * t_0 + m_10 * t_1 + m_20 * t_2 + m_30 * t_3;
  float d = 1.0f / det;

  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = d * t_0,
      .m_01 = d * t_1,
      .m_02 = d * t_2,
      .m_03 = d * t_3,
      .m_10 = d * ((tmp_1 * m_10 + tmp_2 * m_20 + tmp_5 * m_30) -
                   (tmp_0 * m_10 + tmp_3 * m_20 + tmp_4 * m_30)),
      .m_11 = d * ((tmp_0 * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                   (tmp_1 * m_00 + tmp_6 * m_20 + tmp_9 * m_30)),
      .m_12 = d * ((tmp_3 * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                   (tmp_2 * m_00 + tmp_7 * m_10 + tmp_10 * m_30)),
      .m_13 = d * ((tmp_2 * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                   (tmp_3 * m_10 + tmp_4 * m_20 + tmp_9 * m_30)),
      .m_20 = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                   (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33)),
      .m_21 = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                   (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33)),
      .m_22 = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                   (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33)),
      .m_23 = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                   (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23)),
      .m_30 = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                   (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22)),
      .m_31 = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                   (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02)),
      .m_32 = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                   (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12)),
      .m_33 = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                   (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02))});

#else
  // Scalar fallback implementation
  float m_00 = a.m[0 * 4 + 0];
  float m_01 = a.m[0 * 4 + 1];
  float m_02 = a.m[0 * 4 + 2];
  float m_03 = a.m[0 * 4 + 3];
  float m_10 = a.m[1 * 4 + 0];
  float m_11 = a.m[1 * 4 + 1];
  float m_12 = a.m[1 * 4 + 2];
  float m_13 = a.m[1 * 4 + 3];
  float m_20 = a.m[2 * 4 + 0];
  float m_21 = a.m[2 * 4 + 1];
  float m_22 = a.m[2 * 4 + 2];
  float m_23 = a.m[2 * 4 + 3];
  float m_30 = a.m[3 * 4 + 0];
  float m_31 = a.m[3 * 4 + 1];
  float m_32 = a.m[3 * 4 + 2];
  float m_33 = a.m[3 * 4 + 3];

  float tmp_0 = m_22 * m_33;
  float tmp_1 = m_32 * m_23;
  float tmp_2 = m_12 * m_33;
  float tmp_3 = m_32 * m_13;
  float tmp_4 = m_12 * m_23;
  float tmp_5 = m_22 * m_13;
  float tmp_6 = m_02 * m_33;
  float tmp_7 = m_32 * m_03;
  float tmp_8 = m_02 * m_23;
  float tmp_9 = m_22 * m_03;
  float tmp_10 = m_02 * m_13;
  float tmp_11 = m_12 * m_03;
  float tmp_12 = m_20 * m_31;
  float tmp_13 = m_30 * m_21;
  float tmp_14 = m_10 * m_31;
  float tmp_15 = m_30 * m_11;
  float tmp_16 = m_10 * m_21;
  float tmp_17 = m_20 * m_11;
  float tmp_18 = m_00 * m_31;
  float tmp_19 = m_30 * m_01;
  float tmp_20 = m_00 * m_21;
  float tmp_21 = m_20 * m_01;
  float tmp_22 = m_00 * m_11;
  float tmp_23 = m_10 * m_01;

  float t_0 = (tmp_0 * m_11 + tmp_3 * m_21 + tmp_4 * m_31) -
              (tmp_1 * m_11 + tmp_2 * m_21 + tmp_5 * m_31);
  float t_1 = (tmp_1 * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
              (tmp_0 * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
  float t_2 = (tmp_2 * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
              (tmp_3 * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
  float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
              (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

  float d = 1.0f / (m_00 * t_0 + m_10 * t_1 + m_20 * t_2 + m_30 * t_3);

  result = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = d * t_0,
      .m_01 = d * t_1,
      .m_02 = d * t_2,
      .m_03 = d * t_3,
      .m_10 = d * ((tmp_1 * m_10 + tmp_2 * m_20 + tmp_5 * m_30) -
                   (tmp_0 * m_10 + tmp_3 * m_20 + tmp_4 * m_30)),
      .m_11 = d * ((tmp_0 * m_00 + tmp_7 * m_20 + tmp_8 * m_30) -
                   (tmp_1 * m_00 + tmp_6 * m_20 + tmp_9 * m_30)),
      .m_12 = d * ((tmp_3 * m_00 + tmp_6 * m_10 + tmp_11 * m_30) -
                   (tmp_2 * m_00 + tmp_7 * m_10 + tmp_10 * m_30)),
      .m_13 = d * ((tmp_2 * m_10 + tmp_5 * m_20 + tmp_10 * m_30) -
                   (tmp_3 * m_10 + tmp_4 * m_20 + tmp_9 * m_30)),
      .m_20 = d * ((tmp_12 * m_13 + tmp_15 * m_23 + tmp_16 * m_33) -
                   (tmp_13 * m_13 + tmp_14 * m_23 + tmp_17 * m_33)),
      .m_21 = d * ((tmp_13 * m_03 + tmp_18 * m_23 + tmp_21 * m_33) -
                   (tmp_12 * m_03 + tmp_19 * m_23 + tmp_20 * m_33)),
      .m_22 = d * ((tmp_14 * m_03 + tmp_19 * m_13 + tmp_22 * m_33) -
                   (tmp_15 * m_03 + tmp_18 * m_13 + tmp_23 * m_33)),
      .m_23 = d * ((tmp_17 * m_03 + tmp_20 * m_13 + tmp_23 * m_23) -
                   (tmp_16 * m_03 + tmp_21 * m_13 + tmp_22 * m_23)),
      .m_30 = d * ((tmp_14 * m_22 + tmp_17 * m_32 + tmp_13 * m_12) -
                   (tmp_16 * m_32 + tmp_12 * m_12 + tmp_15 * m_22)),
      .m_31 = d * ((tmp_20 * m_32 + tmp_12 * m_02 + tmp_19 * m_22) -
                   (tmp_18 * m_22 + tmp_21 * m_32 + tmp_13 * m_02)),
      .m_32 = d * ((tmp_18 * m_12 + tmp_23 * m_32 + tmp_15 * m_02) -
                   (tmp_22 * m_32 + tmp_14 * m_02 + tmp_19 * m_12)),
      .m_33 = d * ((tmp_22 * m_22 + tmp_16 * m_02 + tmp_21 * m_12) -
                   (tmp_20 * m_12 + tmp_23 * m_22 + tmp_17 * m_02))});
#endif

  return result;
}

WMATH_TYPE(Mat4)
WMATH_TRANSPOSE(Mat4)(WMATH_TYPE(Mat4) a) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation using efficient transpose algorithm
  __m128 row0 = _mm_loadu_ps(&a.m[0]);  // a00 a01 a02 a03
  __m128 row1 = _mm_loadu_ps(&a.m[4]);  // a10 a11 a12 a13
  __m128 row2 = _mm_loadu_ps(&a.m[8]);  // a20 a21 a22 a23
  __m128 row3 = _mm_loadu_ps(&a.m[12]); // a30 a31 a32 a33

  // Transpose using _MM_TRANSPOSE4_PS macro equivalent
  __m128 tmp0 = _mm_unpacklo_ps(row0, row1); // a00 a10 a01 a11
  __m128 tmp1 = _mm_unpacklo_ps(row2, row3); // a20 a30 a21 a31
  __m128 tmp2 = _mm_unpackhi_ps(row0, row1); // a02 a12 a03 a13
  __m128 tmp3 = _mm_unpackhi_ps(row2, row3); // a22 a32 a23 a33

  __m128 col0 = _mm_movelh_ps(tmp0, tmp1); // a00 a10 a20 a30
  __m128 col1 = _mm_movehl_ps(tmp1, tmp0); // a01 a11 a21 a31
  __m128 col2 = _mm_movelh_ps(tmp2, tmp3); // a02 a12 a22 a32
  __m128 col3 = _mm_movehl_ps(tmp3, tmp2); // a03 a13 a23 a33

  _mm_storeu_ps(&result.m[0], col0);
  _mm_storeu_ps(&result.m[4], col1);
  _mm_storeu_ps(&result.m[8], col2);
  _mm_storeu_ps(&result.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation using transpose operations
  float32x4_t row0 = vld1q_f32(&a.m[0]);
  float32x4_t row1 = vld1q_f32(&a.m[4]);
  float32x4_t row2 = vld1q_f32(&a.m[8]);
  float32x4_t row3 = vld1q_f32(&a.m[12]);

  // Transpose using vtrn (vector transpose) and vuzp/vzip operations
  float32x4x2_t t01 = vtrnq_f32(row0, row1); // Interleave rows 0 and 1
  float32x4x2_t t23 = vtrnq_f32(row2, row3); // Interleave rows 2 and 3

  // Final transpose step
  float32x4_t col0 =
      vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
  float32x4_t col1 =
      vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
  float32x4_t col2 =
      vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
  float32x4_t col3 =
      vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

  vst1q_f32(&result.m[0], col0);
  vst1q_f32(&result.m[4], col1);
  vst1q_f32(&result.m[8], col2);
  vst1q_f32(&result.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation using shuffle operations for transpose
  v128_t row0 = wasm_v128_load(&a.m[0]);  // a00 a01 a02 a03
  v128_t row1 = wasm_v128_load(&a.m[4]);  // a10 a11 a12 a13
  v128_t row2 = wasm_v128_load(&a.m[8]);  // a20 a21 a22 a23
  v128_t row3 = wasm_v128_load(&a.m[12]); // a30 a31 a32 a33

  // Transpose using shuffle operations
  // First shuffle to get diagonal elements
  v128_t tmp0 = wasm_i32x4_shuffle(row0, row1, 0, 1, 0, 1); // a00 a01 a10 a11
  v128_t tmp1 = wasm_i32x4_shuffle(row2, row3, 0, 1, 0, 1); // a20 a21 a30 a31
  v128_t tmp2 = wasm_i32x4_shuffle(row0, row1, 2, 3, 2, 3); // a02 a03 a12 a13
  v128_t tmp3 = wasm_i32x4_shuffle(row2, row3, 2, 3, 2, 3); // a22 a23 a32 a33

  // Final shuffles to get columns
  v128_t col0 = wasm_i32x4_shuffle(tmp0, tmp1, 0, 2, 0, 2); // a00 a10 a20 a30
  v128_t col1 = wasm_i32x4_shuffle(tmp0, tmp1, 1, 3, 1, 3); // a01 a11 a21 a31
  v128_t col2 = wasm_i32x4_shuffle(tmp2, tmp3, 0, 2, 0, 2); // a02 a12 a22 a32
  v128_t col3 = wasm_i32x4_shuffle(tmp2, tmp3, 1, 3, 1, 3); // a03 a13 a23 a33

  wasm_v128_store(&result.m[0], col0);
  wasm_v128_store(&result.m[4], col1);
  wasm_v128_store(&result.m[8], col2);
  wasm_v128_store(&result.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation for transpose
  size_t vl = __riscv_vsetvli(4, RV32, e32, m1);
  vfloat32m1_t row0 = __riscv_vle32_v_f32m1(&a.m[0], vl);
  vfloat32m1_t row1 = __riscv_vle32_v_f32m1(&a.m[4], vl);
  vfloat32m1_t row2 = __riscv_vle32_v_f32m1(&a.m[8], vl);
  vfloat32m1_t row3 = __riscv_vle32_v_f32m1(&a.m[12], vl);

  // Transpose by extracting elements from each row to form columns
  vfloat32m1_t col0 = __riscv_vcreate_v_f32m1(__riscv_vfmv_f_s_f32m1_f32(row0),
                                               __riscv_vfmv_f_s_f32m1_f32(row1),
                                               __riscv_vfmv_f_s_f32m1_f32(row2),
                                               __riscv_vfmv_f_s_f32m1_f32(row3));

  // Extract second elements
  vfloat32m1_t col1 = __riscv_vcreate_v_f32m1(__riscv_vextractf32x4_f32(row0, 1),
                                               __riscv_vextractf32x4_f32(row1, 1),
                                               __riscv_vextractf32x4_f32(row2, 1),
                                               __riscv_vextractf32x4_f32(row3, 1));

  // Extract third elements
  vfloat32m1_t col2 = __riscv_vcreate_v_f32m1(__riscv_vextractf32x4_f32(row0, 2),
                                               __riscv_vextractf32x4_f32(row1, 2),
                                               __riscv_vextractf32x4_f32(row2, 2),
                                               __riscv_vextractf32x4_f32(row3, 2));

  // Extract fourth elements
  vfloat32m1_t col3 = __riscv_vcreate_v_f32m1(__riscv_vextractf32x4_f32(row0, 3),
                                               __riscv_vextractf32x4_f32(row1, 3),
                                               __riscv_vextractf32x4_f32(row2, 3),
                                               __riscv_vextractf32x4_f32(row3, 3));

  // Store the transposed matrix
  __riscv_vse32_v_f32m1(&result.m[0], col0, vl);
  __riscv_vse32_v_f32m1(&result.m[4], col1, vl);
  __riscv_vse32_v_f32m1(&result.m[8], col2, vl);
  __riscv_vse32_v_f32m1(&result.m[12], col3, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation for transpose
  __m128 row0 = (__m128)__lsx_vld(&a.m[0], 0);  // a00 a01 a02 a03
  __m128 row1 = (__m128)__lsx_vld(&a.m[4], 0);  // a10 a11 a12 a13
  __m128 row2 = (__m128)__lsx_vld(&a.m[8], 0);  // a20 a21 a22 a23
  __m128 row3 = (__m128)__lsx_vld(&a.m[12], 0); // a30 a31 a32 a33

  // Transpose using shuffle operations
  __m128 tmp0 = __lsx_vilvl_w((__m128i)row1, (__m128i)row0); // a00 a10 a01 a11
  __m128 tmp1 = __lsx_vilvl_w((__m128i)row3, (__m128i)row2); // a20 a30 a21 a31
  __m128 tmp2 = __lsx_vilvh_w((__m128i)row1, (__m128i)row0); // a02 a12 a03 a13
  __m128 tmp3 = __lsx_vilvh_w((__m128i)row3, (__m128i)row2); // a22 a32 a23 a33

  __m128 col0 = __lsx_vilvl_d((__m128i)tmp1, (__m128i)tmp0); // a00 a10 a20 a30
  __m128 col1 = __lsx_vilvh_d((__m128i)tmp1, (__m128i)tmp0); // a01 a11 a21 a31
  __m128 col2 = __lsx_vilvl_d((__m128i)tmp3, (__m128i)tmp2); // a02 a12 a22 a32
  __m128 col3 = __lsx_vilvh_d((__m128i)tmp3, (__m128i)tmp2); // a03 a13 a23 a33

  __lsx_vst((__m128i)col0, &result.m[0], 0);
  __lsx_vst((__m128i)col1, &result.m[4], 0);
  __lsx_vst((__m128i)col2, &result.m[8], 0);
  __lsx_vst((__m128i)col3, &result.m[12], 0);

#else
  // Scalar fallback
  result = WMATH_SET(Mat4)(WMATH_COPY(Mat4)(a),              // Self
                         a.m[0], a.m[4], a.m[8], a.m[12],  // 0
                         a.m[1], a.m[5], a.m[9], a.m[13],  // 1
                         a.m[2], a.m[6], a.m[10], a.m[14], // 2
                         a.m[3], a.m[7], a.m[11], a.m[15]  // 3
  );
#endif

  return result;
}

float WMATH_DETERMINANT(Mat4)(WMATH_TYPE(Mat4) m) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized version using SIMD for parallel computation
  _mm_loadu_ps(&m.m[0]);
  _mm_loadu_ps(&m.m[4]);
  _mm_loadu_ps(&m.m[8]);
  _mm_loadu_ps(&m.m[12]);

  // Extract elements for computation
  float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
  float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
  float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
  float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

  // Calculate temporary values using SIMD where possible
  __m128 vec_m22 = _mm_set1_ps(m22);
  __m128 vec_m33 = _mm_set1_ps(m33);
  __m128 vec_m32 = _mm_set1_ps(m32);
  __m128 vec_m23 = _mm_set1_ps(m23);
  __m128 vec_m12 = _mm_set1_ps(m12);
  __m128 vec_m13 = _mm_set1_ps(m13);
  __m128 vec_m02 = _mm_set1_ps(m02);
  __m128 vec_m03 = _mm_set1_ps(m03);

  // Calculate tmp values using SIMD
  __m128 tmp0 = _mm_mul_ps(vec_m22, vec_m33);
  __m128 tmp1 = _mm_mul_ps(vec_m32, vec_m23);
  __m128 tmp2 = _mm_mul_ps(vec_m12, vec_m33);
  __m128 tmp3 = _mm_mul_ps(vec_m32, vec_m13);
  __m128 tmp4 = _mm_mul_ps(vec_m12, vec_m23);
  __m128 tmp5 = _mm_mul_ps(vec_m22, vec_m13);
  __m128 tmp6 = _mm_mul_ps(vec_m02, vec_m33);
  __m128 tmp7 = _mm_mul_ps(vec_m32, vec_m03);
  __m128 tmp8 = _mm_mul_ps(vec_m02, vec_m23);
  __m128 tmp9 = _mm_mul_ps(vec_m22, vec_m03);
  __m128 tmp10 = _mm_mul_ps(vec_m02, vec_m13);
  __m128 tmp11 = _mm_mul_ps(vec_m12, vec_m03);

  // Extract scalar values
  float tmp0_val = _mm_cvtss_f32(tmp0);
  float tmp1_val = _mm_cvtss_f32(tmp1);
  float tmp2_val = _mm_cvtss_f32(tmp2);
  float tmp3_val = _mm_cvtss_f32(tmp3);
  float tmp4_val = _mm_cvtss_f32(tmp4);
  float tmp5_val = _mm_cvtss_f32(tmp5);
  float tmp6_val = _mm_cvtss_f32(tmp6);
  float tmp7_val = _mm_cvtss_f32(tmp7);
  float tmp8_val = _mm_cvtss_f32(tmp8);
  float tmp9_val = _mm_cvtss_f32(tmp9);
  float tmp10_val = _mm_cvtss_f32(tmp10);
  float tmp11_val = _mm_cvtss_f32(tmp11);

  // Calculate t values (these are complex cross terms, done scalar for
  // precision)
  float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
             (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
  float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
             (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
  float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
             (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
  float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
             (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

  // Calculate determinant using SIMD
  __m128 vec_m00 = _mm_set1_ps(m00);
  __m128 vec_m10 = _mm_set1_ps(m10);
  __m128 vec_m20 = _mm_set1_ps(m20);
  __m128 vec_m30 = _mm_set1_ps(m30);
  __m128 vec_t0 = _mm_set1_ps(t0);
  __m128 vec_t1 = _mm_set1_ps(t1);
  __m128 vec_t2 = _mm_set1_ps(t2);
  __m128 vec_t3 = _mm_set1_ps(t3);

  __m128 det_part0 = _mm_mul_ps(vec_m00, vec_t0);
  __m128 det_part1 = _mm_mul_ps(vec_m10, vec_t1);
  __m128 det_part2 = _mm_mul_ps(vec_m20, vec_t2);
  __m128 det_part3 = _mm_mul_ps(vec_m30, vec_t3);

  __m128 det_sum01 = _mm_add_ps(det_part0, det_part1);
  __m128 det_sum23 = _mm_add_ps(det_part2, det_part3);
  __m128 det_total = _mm_add_ps(det_sum01, det_sum23);

  return _mm_cvtss_f32(det_total);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized version using SIMD for parallel computation
  float32x4_t row0 = vld1q_f32(&m.m[0]);
  float32x4_t row1 = vld1q_f32(&m.m[4]);
  float32x4_t row2 = vld1q_f32(&m.m[8]);
  float32x4_t row3 = vld1q_f32(&m.m[12]);

  // Extract elements for computation
  float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
  float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
  float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
  float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

  // Calculate temporary values using SIMD where possible
  float32x4_t vec_m22 = vdupq_n_f32(m22);
  float32x4_t vec_m33 = vdupq_n_f32(m33);
  float32x4_t vec_m32 = vdupq_n_f32(m32);
  float32x4_t vec_m23 = vdupq_n_f32(m23);
  float32x4_t vec_m12 = vdupq_n_f32(m12);
  float32x4_t vec_m13 = vdupq_n_f32(m13);
  float32x4_t vec_m02 = vdupq_n_f32(m02);
  float32x4_t vec_m03 = vdupq_n_f32(m03);

  // Calculate tmp values using SIMD
  float32x4_t tmp0 = vmulq_f32(vec_m22, vec_m33);
  float32x4_t tmp1 = vmulq_f32(vec_m32, vec_m23);
  float32x4_t tmp2 = vmulq_f32(vec_m12, vec_m33);
  float32x4_t tmp3 = vmulq_f32(vec_m32, vec_m13);
  float32x4_t tmp4 = vmulq_f32(vec_m12, vec_m23);
  float32x4_t tmp5 = vmulq_f32(vec_m22, vec_m13);
  float32x4_t tmp6 = vmulq_f32(vec_m02, vec_m33);
  float32x4_t tmp7 = vmulq_f32(vec_m32, vec_m03);
  float32x4_t tmp8 = vmulq_f32(vec_m02, vec_m23);
  float32x4_t tmp9 = vmulq_f32(vec_m22, vec_m03);
  float32x4_t tmp10 = vmulq_f32(vec_m02, vec_m13);
  float32x4_t tmp11 = vmulq_f32(vec_m12, vec_m03);

  // Extract scalar values
  float tmp0_val = vgetq_lane_f32(tmp0, 0);
  float tmp1_val = vgetq_lane_f32(tmp1, 0);
  float tmp2_val = vgetq_lane_f32(tmp2, 0);
  float tmp3_val = vgetq_lane_f32(tmp3, 0);
  float tmp4_val = vgetq_lane_f32(tmp4, 0);
  float tmp5_val = vgetq_lane_f32(tmp5, 0);
  float tmp6_val = vgetq_lane_f32(tmp6, 0);
  float tmp7_val = vgetq_lane_f32(tmp7, 0);
  float tmp8_val = vgetq_lane_f32(tmp8, 0);
  float tmp9_val = vgetq_lane_f32(tmp9, 0);
  float tmp10_val = vgetq_lane_f32(tmp10, 0);
  float tmp11_val = vgetq_lane_f32(tmp11, 0);

  // Calculate t values (these are complex cross terms, done scalar for
  // precision)
  float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
             (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
  float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
             (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
  float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
             (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
  float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
             (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

  // Calculate determinant using SIMD
  float32x4_t vec_m00 = vdupq_n_f32(m00);
  float32x4_t vec_m10 = vdupq_n_f32(m10);
  float32x4_t vec_m20 = vdupq_n_f32(m20);
  float32x4_t vec_m30 = vdupq_n_f32(m30);
  float32x4_t vec_t0 = vdupq_n_f32(t0);
  float32x4_t vec_t1 = vdupq_n_f32(t1);
  float32x4_t vec_t2 = vdupq_n_f32(t2);
  float32x4_t vec_t3 = vdupq_n_f32(t3);

  float32x4_t det_part0 = vmulq_f32(vec_m00, vec_t0);
  float32x4_t det_part1 = vmulq_f32(vec_m10, vec_t1);
  float32x4_t det_part2 = vmulq_f32(vec_m20, vec_t2);
  float32x4_t det_part3 = vmulq_f32(vec_m30, vec_t3);

  float32x4_t det_sum01 = vaddq_f32(det_part0, det_part1);
  float32x4_t det_sum23 = vaddq_f32(det_part2, det_part3);
  float32x4_t det_total = vaddq_f32(det_sum01, det_sum23);

  return vgetq_lane_f32(det_total, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - optimized version using SIMD for parallel computation
  v128_t row0 = wasm_v128_load(&m.m[0]);
  v128_t row1 = wasm_v128_load(&m.m[4]);
  v128_t row2 = wasm_v128_load(&m.m[8]);
  v128_t row3 = wasm_v128_load(&m.m[12]);

  // Extract elements for computation
  float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
  float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
  float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
  float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

  // Calculate temporary values using WASM SIMD
  v128_t vec_m22 = wasm_f32x4_splat(m22);
  v128_t vec_m33 = wasm_f32x4_splat(m33);
  v128_t vec_m32 = wasm_f32x4_splat(m32);
  v128_t vec_m23 = wasm_f32x4_splat(m23);
  v128_t vec_m12 = wasm_f32x4_splat(m12);
  v128_t vec_m13 = wasm_f32x4_splat(m13);
  v128_t vec_m02 = wasm_f32x4_splat(m02);
  v128_t vec_m03 = wasm_f32x4_splat(m03);

  // Calculate tmp values using SIMD
  v128_t tmp0 = wasm_f32x4_mul(vec_m22, vec_m33);
  v128_t tmp1 = wasm_f32x4_mul(vec_m32, vec_m23);
  v128_t tmp2 = wasm_f32x4_mul(vec_m12, vec_m33);
  v128_t tmp3 = wasm_f32x4_mul(vec_m32, vec_m13);
  v128_t tmp4 = wasm_f32x4_mul(vec_m12, vec_m23);
  v128_t tmp5 = wasm_f32x4_mul(vec_m22, vec_m13);
  v128_t tmp6 = wasm_f32x4_mul(vec_m02, vec_m33);
  v128_t tmp7 = wasm_f32x4_mul(vec_m32, vec_m03);
  v128_t tmp8 = wasm_f32x4_mul(vec_m02, vec_m23);
  v128_t tmp9 = wasm_f32x4_mul(vec_m22, vec_m03);
  v128_t tmp10 = wasm_f32x4_mul(vec_m02, vec_m13);
  v128_t tmp11 = wasm_f32x4_mul(vec_m12, vec_m03);

  // Extract scalar values
  float tmp0_val = wasm_f32x4_extract_lane(tmp0, 0);
  float tmp1_val = wasm_f32x4_extract_lane(tmp1, 0);
  float tmp2_val = wasm_f32x4_extract_lane(tmp2, 0);
  float tmp3_val = wasm_f32x4_extract_lane(tmp3, 0);
  float tmp4_val = wasm_f32x4_extract_lane(tmp4, 0);
  float tmp5_val = wasm_f32x4_extract_lane(tmp5, 0);
  float tmp6_val = wasm_f32x4_extract_lane(tmp6, 0);
  float tmp7_val = wasm_f32x4_extract_lane(tmp7, 0);
  float tmp8_val = wasm_f32x4_extract_lane(tmp8, 0);
  float tmp9_val = wasm_f32x4_extract_lane(tmp9, 0);
  float tmp10_val = wasm_f32x4_extract_lane(tmp10, 0);
  float tmp11_val = wasm_f32x4_extract_lane(tmp11, 0);

  // Calculate t values (these are complex cross terms, done scalar for
  // precision)
  float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
             (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
  float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
             (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
  float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
             (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
  float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
             (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

  // Calculate determinant using SIMD
  v128_t vec_m00 = wasm_f32x4_splat(m00);
  v128_t vec_m10 = wasm_f32x4_splat(m10);
  v128_t vec_m20 = wasm_f32x4_splat(m20);
  v128_t vec_m30 = wasm_f32x4_splat(m30);
  v128_t vec_t0 = wasm_f32x4_splat(t0);
  v128_t vec_t1 = wasm_f32x4_splat(t1);
  v128_t vec_t2 = wasm_f32x4_splat(t2);
  v128_t vec_t3 = wasm_f32x4_splat(t3);

  v128_t det_part0 = wasm_f32x4_mul(vec_m00, vec_t0);
  v128_t det_part1 = wasm_f32x4_mul(vec_m10, vec_t1);
  v128_t det_part2 = wasm_f32x4_mul(vec_m20, vec_t2);
  v128_t det_part3 = wasm_f32x4_mul(vec_m30, vec_t3);

  v128_t det_sum01 = wasm_f32x4_add(det_part0, det_part1);
  v128_t det_sum23 = wasm_f32x4_add(det_part2, det_part3);
  v128_t det_total = wasm_f32x4_add(det_sum01, det_sum23);

  return wasm_f32x4_extract_lane(det_total, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - optimized version using SIMD for parallel computation
  size_t vl = __riscv_vsetvli(4, RV32, e32, m1);
  vfloat32m1_t row0 = __riscv_vle32_v_f32m1(&m.m[0], vl);
  vfloat32m1_t row1 = __riscv_vle32_v_f32m1(&m.m[4], vl);
  vfloat32m1_t row2 = __riscv_vle32_v_f32m1(&m.m[8], vl);
  vfloat32m1_t row3 = __riscv_vle32_v_f32m1(&m.m[12], vl);

  // Extract elements for computation
  float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
  float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
  float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
  float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

  // Calculate temporary values using RISC-V SIMD
  vfloat32m1_t vec_m22 = __riscv_vfmv_v_f_f32m1(m22, vl);
  vfloat32m1_t vec_m33 = __riscv_vfmv_v_f_f32m1(m33, vl);
  vfloat32m1_t vec_m32 = __riscv_vfmv_v_f_f32m1(m32, vl);
  vfloat32m1_t vec_m23 = __riscv_vfmv_v_f_f32m1(m23, vl);
  vfloat32m1_t vec_m12 = __riscv_vfmv_v_f_f32m1(m12, vl);
  vfloat32m1_t vec_m13 = __riscv_vfmv_v_f_f32m1(m13, vl);
  vfloat32m1_t vec_m02 = __riscv_vfmv_v_f_f32m1(m02, vl);
  vfloat32m1_t vec_m03 = __riscv_vfmv_v_f_f32m1(m03, vl);

  // Calculate tmp values using SIMD
  vfloat32m1_t tmp0 = __riscv_vfmul_vv_f32m1(vec_m22, vec_m33, vl);
  vfloat32m1_t tmp1 = __riscv_vfmul_vv_f32m1(vec_m32, vec_m23, vl);
  vfloat32m1_t tmp2 = __riscv_vfmul_vv_f32m1(vec_m12, vec_m33, vl);
  vfloat32m1_t tmp3 = __riscv_vfmul_vv_f32m1(vec_m32, vec_m13, vl);
  vfloat32m1_t tmp4 = __riscv_vfmul_vv_f32m1(vec_m12, vec_m23, vl);
  vfloat32m1_t tmp5 = __riscv_vfmul_vv_f32m1(vec_m22, vec_m13, vl);
  vfloat32m1_t tmp6 = __riscv_vfmul_vv_f32m1(vec_m02, vec_m33, vl);
  vfloat32m1_t tmp7 = __riscv_vfmul_vv_f32m1(vec_m32, vec_m03, vl);
  vfloat32m1_t tmp8 = __riscv_vfmul_vv_f32m1(vec_m02, vec_m23, vl);
  vfloat32m1_t tmp9 = __riscv_vfmul_vv_f32m1(vec_m22, vec_m03, vl);
  vfloat32m1_t tmp10 = __riscv_vfmul_vv_f32m1(vec_m02, vec_m13, vl);
  vfloat32m1_t tmp11 = __riscv_vfmul_vv_f32m1(vec_m12, vec_m03, vl);

  // Extract scalar values
  float tmp0_val = __riscv_vfmv_f_s_f32m1_f32(tmp0);
  float tmp1_val = __riscv_vfmv_f_s_f32m1_f32(tmp1);
  float tmp2_val = __riscv_vfmv_f_s_f32m1_f32(tmp2);
  float tmp3_val = __riscv_vfmv_f_s_f32m1_f32(tmp3);
  float tmp4_val = __riscv_vfmv_f_s_f32m1_f32(tmp4);
  float tmp5_val = __riscv_vfmv_f_s_f32m1_f32(tmp5);
  float tmp6_val = __riscv_vfmv_f_s_f32m1_f32(tmp6);
  float tmp7_val = __riscv_vfmv_f_s_f32m1_f32(tmp7);
  float tmp8_val = __riscv_vfmv_f_s_f32m1_f32(tmp8);
  float tmp9_val = __riscv_vfmv_f_s_f32m1_f32(tmp9);
  float tmp10_val = __riscv_vfmv_f_s_f32m1_f32(tmp10);
  float tmp11_val = __riscv_vfmv_f_s_f32m1_f32(tmp11);

  // Calculate t values (these are complex cross terms, done scalar for
  // precision)
  float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
             (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
  float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
             (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
  float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
             (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
  float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
             (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

  // Calculate determinant using SIMD
  vfloat32m1_t vec_m00 = __riscv_vfmv_v_f_f32m1(m00, vl);
  vfloat32m1_t vec_m10 = __riscv_vfmv_v_f_f32m1(m10, vl);
  vfloat32m1_t vec_m20 = __riscv_vfmv_v_f_f32m1(m20, vl);
  vfloat32m1_t vec_m30 = __riscv_vfmv_v_f_f32m1(m30, vl);
  vfloat32m1_t vec_t0 = __riscv_vfmv_v_f_f32m1(t0, vl);
  vfloat32m1_t vec_t1 = __riscv_vfmv_v_f_f32m1(t1, vl);
  vfloat32m1_t vec_t2 = __riscv_vfmv_v_f_f32m1(t2, vl);
  vfloat32m1_t vec_t3 = __riscv_vfmv_v_f_f32m1(t3, vl);

  vfloat32m1_t det_part0 = __riscv_vfmul_vv_f32m1(vec_m00, vec_t0, vl);
  vfloat32m1_t det_part1 = __riscv_vfmul_vv_f32m1(vec_m10, vec_t1, vl);
  vfloat32m1_t det_part2 = __riscv_vfmul_vv_f32m1(vec_m20, vec_t2, vl);
  vfloat32m1_t det_part3 = __riscv_vfmul_vv_f32m1(vec_m30, vec_t3, vl);

  vfloat32m1_t det_sum01 = __riscv_vfadd_vv_f32m1(det_part0, det_part1, vl);
  vfloat32m1_t det_sum23 = __riscv_vfadd_vv_f32m1(det_part2, det_part3, vl);
  vfloat32m1_t det_total = __riscv_vfadd_vv_f32m1(det_sum01, det_sum23, vl);

  return __riscv_vfmv_f_s_f32m1_f32(det_total);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized version using SIMD for parallel computation
  __m128 row0 = (__m128)__lsx_vld(&m.m[0], 0);
  __m128 row1 = (__m128)__lsx_vld(&m.m[4], 0);
  __m128 row2 = (__m128)__lsx_vld(&m.m[8], 0);
  __m128 row3 = (__m128)__lsx_vld(&m.m[12], 0);

  // Extract elements for computation
  float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
  float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
  float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
  float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

  // Calculate temporary values using LoongArch SIMD
  __m128 vec_m22 = __lsx_vldrepl_w(&m22, 0);
  __m128 vec_m33 = __lsx_vldrepl_w(&m33, 0);
  __m128 vec_m32 = __lsx_vldrepl_w(&m32, 0);
  __m128 vec_m23 = __lsx_vldrepl_w(&m23, 0);
  __m128 vec_m12 = __lsx_vldrepl_w(&m12, 0);
  __m128 vec_m13 = __lsx_vldrepl_w(&m13, 0);
  __m128 vec_m02 = __lsx_vldrepl_w(&m02, 0);
  __m128 vec_m03 = __lsx_vldrepl_w(&m03, 0);

  // Calculate tmp values using SIMD
  __m128 tmp0 = (__m128)__lsx_vfmul_s((__m128i)vec_m22, (__m128i)vec_m33);
  __m128 tmp1 = (__m128)__lsx_vfmul_s((__m128i)vec_m32, (__m128i)vec_m23);
  __m128 tmp2 = (__m128)__lsx_vfmul_s((__m128i)vec_m12, (__m128i)vec_m33);
  __m128 tmp3 = (__m128)__lsx_vfmul_s((__m128i)vec_m32, (__m128i)vec_m13);
  __m128 tmp4 = (__m128)__lsx_vfmul_s((__m128i)vec_m12, (__m128i)vec_m23);
  __m128 tmp5 = (__m128)__lsx_vfmul_s((__m128i)vec_m22, (__m128i)vec_m13);
  __m128 tmp6 = (__m128)__lsx_vfmul_s((__m128i)vec_m02, (__m128i)vec_m33);
  __m128 tmp7 = (__m128)__lsx_vfmul_s((__m128i)vec_m32, (__m128i)vec_m03);
  __m128 tmp8 = (__m128)__lsx_vfmul_s((__m128i)vec_m02, (__m128i)vec_m23);
  __m128 tmp9 = (__m128)__lsx_vfmul_s((__m128i)vec_m22, (__m128i)vec_m03);
  __m128 tmp10 = (__m128)__lsx_vfmul_s((__m128i)vec_m02, (__m128i)vec_m13);
  __m128 tmp11 = (__m128)__lsx_vfmul_s((__m128i)vec_m12, (__m128i)vec_m03);

  // Extract scalar values
  float tmp0_val = __lsx_vfmov_s_f(tmp0);
  float tmp1_val = __lsx_vfmov_s_f(tmp1);
  float tmp2_val = __lsx_vfmov_s_f(tmp2);
  float tmp3_val = __lsx_vfmov_s_f(tmp3);
  float tmp4_val = __lsx_vfmov_s_f(tmp4);
  float tmp5_val = __lsx_vfmov_s_f(tmp5);
  float tmp6_val = __lsx_vfmov_s_f(tmp6);
  float tmp7_val = __lsx_vfmov_s_f(tmp7);
  float tmp8_val = __lsx_vfmov_s_f(tmp8);
  float tmp9_val = __lsx_vfmov_s_f(tmp9);
  float tmp10_val = __lsx_vfmov_s_f(tmp10);
  float tmp11_val = __lsx_vfmov_s_f(tmp11);

  // Calculate t values (these are complex cross terms, done scalar for
  // precision)
  float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
             (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
  float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
             (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
  float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
             (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
  float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
             (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

  // Calculate determinant using SIMD
  __m128 vec_m00 = __lsx_vldrepl_w(&m00, 0);
  __m128 vec_m10 = __lsx_vldrepl_w(&m10, 0);
  __m128 vec_m20 = __lsx_vldrepl_w(&m20, 0);
  __m128 vec_m30 = __lsx_vldrepl_w(&m30, 0);
  __m128 vec_t0 = __lsx_vldrepl_w(&t0, 0);
  __m128 vec_t1 = __lsx_vldrepl_w(&t1, 0);
  __m128 vec_t2 = __lsx_vldrepl_w(&t2, 0);
  __m128 vec_t3 = __lsx_vldrepl_w(&t3, 0);

  __m128 det_part0 = (__m128)__lsx_vfmul_s((__m128i)vec_m00, (__m128i)vec_t0);
  __m128 det_part1 = (__m128)__lsx_vfmul_s((__m128i)vec_m10, (__m128i)vec_t1);
  __m128 det_part2 = (__m128)__lsx_vfmul_s((__m128i)vec_m20, (__m128i)vec_t2);
  __m128 det_part3 = (__m128)__lsx_vfmul_s((__m128i)vec_m30, (__m128i)vec_t3);

  __m128 det_sum01 = (__m128)__lsx_vfadd_s((__m128i)det_part0, (__m128i)det_part1);
  __m128 det_sum23 = (__m128)__lsx_vfadd_s((__m128i)det_part2, (__m128i)det_part3);
  __m128 det_total = (__m128)__lsx_vfadd_s((__m128i)det_sum01, (__m128i)det_sum23);

  return __lsx_vfmov_s_f(det_total);

#else
  // Scalar fallback implementation
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m03 = m.m[0 * 4 + 3];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];
  float m13 = m.m[1 * 4 + 3];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];
  float m23 = m.m[2 * 4 + 3];
  float m30 = m.m[3 * 4 + 0];
  float m31 = m.m[3 * 4 + 1];
  float m32 = m.m[3 * 4 + 2];
  float m33 = m.m[3 * 4 + 3];

  float tmp0 = m22 * m33;
  float tmp1 = m32 * m23;
  float tmp2 = m12 * m33;
  float tmp3 = m32 * m13;
  float tmp4 = m12 * m23;
  float tmp5 = m22 * m13;
  float tmp6 = m02 * m33;
  float tmp7 = m32 * m03;
  float tmp8 = m02 * m23;
  float tmp9 = m22 * m03;
  float tmp10 = m02 * m13;
  float tmp11 = m12 * m03;

  float t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) -
             (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
  float t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) -
             (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
  float t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) -
             (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
  float t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) -
             (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);

  return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
#endif
}

// aim
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, aim)
(WMATH_TYPE(Vec3) position, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - optimized camera aim matrix creation
  __m128 vec_position = wcn_load_vec3_partial(position.v);
  __m128 vec_target = wcn_load_vec3_partial(target.v);
  __m128 vec_up = wcn_load_vec3_partial(up.v);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Calculate z_axis = normalize(target - position)
  __m128 z_axis_unnorm = _mm_sub_ps(vec_target, vec_position);
  float z_axis_len_sq =
      _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(z_axis_unnorm, z_axis_unnorm)));

  __m128 z_axis;
  if (z_axis_len_sq > 0.00001f) {
    __m128 z_axis_inv_len = wcn_fast_inv_sqrt_ps(_mm_set1_ps(z_axis_len_sq));
    z_axis = _mm_mul_ps(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f); // Default forward vector
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  __m128 x_axis_unnorm = wcn_cross_platform(vec_up, z_axis);
  float x_axis_len_sq =
      _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(x_axis_unnorm, x_axis_unnorm)));

  __m128 x_axis;
  if (x_axis_len_sq > 0.00001f) {
    __m128 x_axis_inv_len = wcn_fast_inv_sqrt_ps(_mm_set1_ps(x_axis_len_sq));
    x_axis = _mm_mul_ps(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f); // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  __m128 y_axis_unnorm = wcn_cross_platform(z_axis, x_axis);
  float y_axis_len_sq =
      _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(y_axis_unnorm, y_axis_unnorm)));

  __m128 y_axis;
  if (y_axis_len_sq > 0.00001f) {
    __m128 y_axis_inv_len = wcn_fast_inv_sqrt_ps(_mm_set1_ps(y_axis_len_sq));
    y_axis = _mm_mul_ps(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // Default up vector
  }

  // Create camera aim matrix rows
  // Row0: [x_axis.x, x_axis.y, x_axis.z, 0]
  __m128 row0 = _mm_move_ss(x_axis, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row1: [y_axis.x, y_axis.y, y_axis.z, 0]
  __m128 row1 = _mm_move_ss(y_axis, vec_zero);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row2: [z_axis.x, z_axis.y, z_axis.z, 0]
  __m128 row2 = _mm_move_ss(z_axis, vec_zero);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row3: [position.x, position.y, position.z, 1]
  __m128 row3 = _mm_move_ss(vec_position, vec_one);
  row3 = _mm_shuffle_ps(row3, vec_zero, _MM_SHUFFLE(3, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&result.m[0], row0);
  _mm_storeu_ps(&result.m[4], row1);
  _mm_storeu_ps(&result.m[8], row2);
  _mm_storeu_ps(&result.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - optimized camera aim matrix creation
  float32x4_t vec_position = wcn_load_vec3_partial(position.v);
  float32x4_t vec_target = wcn_load_vec3_partial(target.v);
  float32x4_t vec_up = wcn_load_vec3_partial(up.v);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Calculate z_axis = normalize(target - position)
  float32x4_t z_axis_unnorm = vsubq_f32(vec_target, vec_position);
  float z_axis_len_sq = wcn_hadd_f32(vmulq_f32(z_axis_unnorm, z_axis_unnorm));

  float32x4_t z_axis;
  if (z_axis_len_sq > 0.00001f) {
    float z_axis_inv_len = wcn_fast_inv_sqrt(z_axis_len_sq);
    z_axis = vmulq_n_f32(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = (float32x4_t){0.0f, 0.0f, 1.0f, 0.0f}; // Default forward vector
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  float32x4_t x_axis_unnorm = wcn_cross_platform(vec_up, z_axis);
  float x_axis_len_sq = wcn_hadd_f32(vmulq_f32(x_axis_unnorm, x_axis_unnorm));

  float32x4_t x_axis;
  if (x_axis_len_sq > 0.00001f) {
    float x_axis_inv_len = wcn_fast_inv_sqrt(x_axis_len_sq);
    x_axis = vmulq_n_f32(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = (float32x4_t){0.0f, 1.0f, 0.0f, 0.0f}; // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  float32x4_t y_axis_unnorm = wcn_cross_platform(z_axis, x_axis);
  float y_axis_len_sq = wcn_hadd_f32(vmulq_f32(y_axis_unnorm, y_axis_unnorm));

  float32x4_t y_axis;
  if (y_axis_len_sq > 0.00001f) {
    float y_axis_inv_len = wcn_fast_inv_sqrt(y_axis_len_sq);
    y_axis = vmulq_n_f32(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = (float32x4_t){1.0f, 0.0f, 0.0f, 0.0f}; // Default up vector
  }

  // Create camera aim matrix rows
  float32x4_t row0 = x_axis;
  row0 = vsetq_lane_f32(0.0f, row0, 3);

  float32x4_t row1 = y_axis;
  row1 = vsetq_lane_f32(0.0f, row1, 3);

  float32x4_t row2 = z_axis;
  row2 = vsetq_lane_f32(0.0f, row2, 3);

  float32x4_t row3 = vec_position;
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  // Store results
  vst1q_f32(&result.m[0], row0);
  vst1q_f32(&result.m[4], row1);
  vst1q_f32(&result.m[8], row2);
  vst1q_f32(&result.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation - optimized camera aim matrix creation
  v128_t vec_position = wcn_load_vec3_partial(position.v);
  v128_t vec_target = wcn_load_vec3_partial(target.v);
  v128_t vec_up = wcn_load_vec3_partial(up.v);
  v128_t vec_zero = wasm_f32x4_splat(0.0f);
  v128_t vec_one = wasm_f32x4_splat(1.0f);

  // Calculate z_axis = normalize(target - position)
  v128_t z_axis_unnorm = wasm_f32x4_sub(vec_target, vec_position);
  v128_t z_axis_sq = wasm_f32x4_mul(z_axis_unnorm, z_axis_unnorm);

  // Horizontal add to get sum of squares
  v128_t shuf = wasm_i32x4_shuffle(z_axis_sq, z_axis_sq, 2, 3, 0, 1);
  v128_t sums = wasm_f32x4_add(z_axis_sq, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  float z_axis_len_sq = wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);

  v128_t z_axis;
  if (z_axis_len_sq > 0.00001f) {
    v128_t z_axis_inv_len = wcn_fast_inv_sqrt_platform(wasm_f32x4_splat(z_axis_len_sq));
    z_axis = wasm_f32x4_mul(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = wasm_f32x4_make(0.0f, 0.0f, 1.0f, 0.0f); // Default forward vector
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  v128_t x_axis_unnorm = wcn_cross_platform(vec_up, z_axis);
  v128_t x_axis_sq = wasm_f32x4_mul(x_axis_unnorm, x_axis_unnorm);

  // Horizontal add to get sum of squares
  shuf = wasm_i32x4_shuffle(x_axis_sq, x_axis_sq, 2, 3, 0, 1);
  sums = wasm_f32x4_add(x_axis_sq, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  float x_axis_len_sq = wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);

  v128_t x_axis;
  if (x_axis_len_sq > 0.00001f) {
    v128_t x_axis_inv_len = wcn_fast_inv_sqrt_platform(wasm_f32x4_splat(x_axis_len_sq));
    x_axis = wasm_f32x4_mul(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = wasm_f32x4_make(0.0f, 1.0f, 0.0f, 0.0f); // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  v128_t y_axis_unnorm = wcn_cross_platform(z_axis, x_axis);
  v128_t y_axis_sq = wasm_f32x4_mul(y_axis_unnorm, y_axis_unnorm);

  // Horizontal add to get sum of squares
  shuf = wasm_i32x4_shuffle(y_axis_sq, y_axis_sq, 2, 3, 0, 1);
  sums = wasm_f32x4_add(y_axis_sq, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  float y_axis_len_sq = wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);

  v128_t y_axis;
  if (y_axis_len_sq > 0.00001f) {
    v128_t y_axis_inv_len = wcn_fast_inv_sqrt_platform(wasm_f32x4_splat(y_axis_len_sq));
    y_axis = wasm_f32x4_mul(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = wasm_f32x4_make(1.0f, 0.0f, 0.0f, 0.0f); // Default up vector
  }

  // Create camera aim matrix rows
  v128_t row0 = x_axis;
  row0 = wasm_f32x4_replace_lane(row0, 3, 0.0f);

  v128_t row1 = y_axis;
  row1 = wasm_f32x4_replace_lane(row1, 3, 0.0f);

  v128_t row2 = z_axis;
  row2 = wasm_f32x4_replace_lane(row2, 3, 0.0f);

  v128_t row3 = vec_position;
  row3 = wasm_f32x4_replace_lane(row3, 3, 1.0f);

  // Store results
  wasm_v128_store(&result.m[0], row0);
  wasm_v128_store(&result.m[4], row1);
  wasm_v128_store(&result.m[8], row2);
  wasm_v128_store(&result.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - optimized camera aim matrix creation
  vfloat32m1_t vec_position = wcn_load_vec3_partial(position.v);
  vfloat32m1_t vec_target = wcn_load_vec3_partial(target.v);
  vfloat32m1_t vec_up = wcn_load_vec3_partial(up.v);
  vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0.0f, 4);
  vfloat32m1_t vec_one = __riscv_vfmv_v_f_f32m1(1.0f, 4);

  // Calculate z_axis = normalize(target - position)
  vfloat32m1_t z_axis_unnorm = __riscv_vfsub_vv_f32m1(vec_target, vec_position, 4);
  vfloat32m1_t z_axis_sq = __riscv_vfmul_vv_f32m1(z_axis_unnorm, z_axis_unnorm, 4);
  float z_axis_len_sq = __riscv_vfredusum_vs_f32m1_f32(z_axis_sq, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

  vfloat32m1_t z_axis;
  if (z_axis_len_sq > 0.00001f) {
    vfloat32m1_t z_axis_inv_len = wcn_fast_inv_sqrt_platform(__riscv_vfmv_v_f_f32m1(z_axis_len_sq, 4));
    z_axis = __riscv_vfmul_vv_f32m1(z_axis_unnorm, z_axis_inv_len, 4);
  } else {
    float temp[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    z_axis = __riscv_vle32_v_f32m1(temp, 4); // Default forward vector
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  vfloat32m1_t x_axis_unnorm = wcn_cross_platform(vec_up, z_axis);
  vfloat32m1_t x_axis_sq = __riscv_vfmul_vv_f32m1(x_axis_unnorm, x_axis_unnorm, 4);
  float x_axis_len_sq = __riscv_vfredusum_vs_f32m1_f32(x_axis_sq, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

  vfloat32m1_t x_axis;
  if (x_axis_len_sq > 0.00001f) {
    vfloat32m1_t x_axis_inv_len = wcn_fast_inv_sqrt_platform(__riscv_vfmv_v_f_f32m1(x_axis_len_sq, 4));
    x_axis = __riscv_vfmul_vv_f32m1(x_axis_unnorm, x_axis_inv_len, 4);
  } else {
    float temp[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    x_axis = __riscv_vle32_v_f32m1(temp, 4); // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  vfloat32m1_t y_axis_unnorm = wcn_cross_platform(z_axis, x_axis);
  vfloat32m1_t y_axis_sq = __riscv_vfmul_vv_f32m1(y_axis_unnorm, y_axis_unnorm, 4);
  float y_axis_len_sq = __riscv_vfredusum_vs_f32m1_f32(y_axis_sq, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);

  vfloat32m1_t y_axis;
  if (y_axis_len_sq > 0.00001f) {
    vfloat32m1_t y_axis_inv_len = wcn_fast_inv_sqrt_platform(__riscv_vfmv_v_f_f32m1(y_axis_len_sq, 4));
    y_axis = __riscv_vfmul_vv_f32m1(y_axis_unnorm, y_axis_inv_len, 4);
  } else {
    float temp[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    y_axis = __riscv_vle32_v_f32m1(temp, 4); // Default up vector
  }

  // Create camera aim matrix rows
  vfloat32m1_t row0 = __riscv_vfmerge_vfm_f32m1(x_axis, 0.0f, 0x8, 4); // Set lane 3 to 0
  vfloat32m1_t row1 = __riscv_vfmerge_vfm_f32m1(y_axis, 0.0f, 0x8, 4); // Set lane 3 to 0
  vfloat32m1_t row2 = __riscv_vfmerge_vfm_f32m1(z_axis, 0.0f, 0x8, 4); // Set lane 3 to 0
  vfloat32m1_t row3 = __riscv_vfmerge_vfm_f32m1(vec_position, 1.0f, 0x8, 4); // Set lane 3 to 1

  // Store results
  __riscv_vse32_v_f32m1(&result.m[0], row0, 4);
  __riscv_vse32_v_f32m1(&result.m[4], row1, 4);
  __riscv_vse32_v_f32m1(&result.m[8], row2, 4);
  __riscv_vse32_v_f32m1(&result.m[12], row3, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - optimized camera aim matrix creation
  __m128 vec_position = wcn_load_vec3_partial(position.v);
  __m128 vec_target = wcn_load_vec3_partial(target.v);
  __m128 vec_up = wcn_load_vec3_partial(up.v);
  __m128 vec_zero = __lsx_vldi(0x0); // Load zero
  __m128 vec_one = __lsx_vldrepl_w(&(const float){1.0f}, 0);

  // Calculate z_axis = normalize(target - position)
  __m128 z_axis_unnorm = __lsx_vfsub_s(vec_target, vec_position);
  __m128 z_axis_sq = __lsx_vfmul_s(z_axis_unnorm, z_axis_unnorm);

  // Horizontal add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(z_axis_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(z_axis_sq, high);
  float z_axis_len_sq = __lsx_vfrep2vr_w(sum)[0];

  __m128 z_axis;
  if (z_axis_len_sq > 0.00001f) {
    __m128 z_axis_inv_len = wcn_fast_inv_sqrt_platform(__lsx_vldrepl_w(&z_axis_len_sq, 0));
    z_axis = __lsx_vfmul_s(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = __lsx_vldrepl_w(&(const float){1.0f}, 2); // Default forward vector (z=1)
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  __m128 x_axis_unnorm = wcn_cross_platform(vec_up, z_axis);
  __m128 x_axis_sq = __lsx_vfmul_s(x_axis_unnorm, x_axis_unnorm);

  // Horizontal add to get sum of squares
  high = __lsx_vshuf4i_w(x_axis_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(x_axis_sq, high);
  float x_axis_len_sq = __lsx_vfrep2vr_w(sum)[0];

  __m128 x_axis;
  if (x_axis_len_sq > 0.00001f) {
    __m128 x_axis_inv_len = wcn_fast_inv_sqrt_platform(__lsx_vldrepl_w(&x_axis_len_sq, 0));
    x_axis = __lsx_vfmul_s(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = __lsx_vldrepl_w(&(const float){1.0f}, 1); // Default right vector (y=1)
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  __m128 y_axis_unnorm = wcn_cross_platform(z_axis, x_axis);
  __m128 y_axis_sq = __lsx_vfmul_s(y_axis_unnorm, y_axis_unnorm);

  // Horizontal add to get sum of squares
  high = __lsx_vshuf4i_w(y_axis_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(y_axis_sq, high);
  float y_axis_len_sq = __lsx_vfrep2vr_w(sum)[0];

  __m128 y_axis;
  if (y_axis_len_sq > 0.00001f) {
    __m128 y_axis_inv_len = wcn_fast_inv_sqrt_platform(__lsx_vldrepl_w(&y_axis_len_sq, 0));
    y_axis = __lsx_vfmul_s(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = __lsx_vldrepl_w(&(const float){1.0f}, 0); // Default up vector (x=1)
  }

  // Create camera aim matrix rows
  __m128 row0 = __lsx_vinsgr2vr_w(x_axis, 0, 3); // Set lane 3 to 0
  __m128 row1 = __lsx_vinsgr2vr_w(y_axis, 0, 3); // Set lane 3 to 0
  __m128 row2 = __lsx_vinsgr2vr_w(z_axis, 0, 3); // Set lane 3 to 0
  __m128 row3 = __lsx_vinsgr2vr_w(vec_position, 1, 3); // Set lane 3 to 1

  // Store results
  __lsx_vst(row0, &result.m[0], 0);
  __lsx_vst(row1, &result.m[4], 0);
  __lsx_vst(row2, &result.m[8], 0);
  __lsx_vst(row3, &result.m[12], 0);

#else
  // Scalar fallback - optimized with fast inverse square root
  WMATH_TYPE(Vec3) z_axis = WMATH_SUB(Vec3)(target, position);
  wcn_fast_normalize_vec3(z_axis.v);

  WMATH_TYPE(Vec3) x_axis = WMATH_CROSS(Vec3)(up, z_axis);
  wcn_fast_normalize_vec3(x_axis.v);

  WMATH_TYPE(Vec3) y_axis = WMATH_CROSS(Vec3)(z_axis, x_axis);
  wcn_fast_normalize_vec3(y_axis.v);

  // Direct assignment is more efficient than memset
  result.m[0] = x_axis.v[0];
  result.m[1] = x_axis.v[1];
  result.m[2] = x_axis.v[2];
  result.m[3] = 0.0f;
  result.m[4] = y_axis.v[0];
  result.m[5] = y_axis.v[1];
  result.m[6] = y_axis.v[2];
  result.m[7] = 0.0f;
  result.m[8] = z_axis.v[0];
  result.m[9] = z_axis.v[1];
  result.m[10] = z_axis.v[2];
  result.m[11] = 0.0f;
  result.m[12] = position.v[0];
  result.m[13] = position.v[1];
  result.m[14] = position.v[2];
  result.m[15] = 1.0f;
#endif

  return result;
}

// lookAt
// lookAt
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, look_at)
(WMATH_TYPE(Vec3) eye, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE Implementation
  __m128 vec_eye = wcn_load_vec3_partial(eye.v);
  __m128 vec_target = wcn_load_vec3_partial(target.v);
  __m128 vec_up = wcn_load_vec3_partial(up.v);
  __m128 vec_zero = _mm_setzero_ps();

  // 1. Z Axis = normalize(eye - target)
  __m128 z_axis = _mm_sub_ps(vec_eye, vec_target);
  __m128 z_len_sq = _mm_dp_ps(z_axis, z_axis, 0x7F);

  // FIX: Use sqrt + div for high precision instead of rsqrt
  float z_len = _mm_cvtss_f32(_mm_sqrt_ss(z_len_sq));
  if (z_len > 1e-6f) {
    z_axis = _mm_div_ps(z_axis, _mm_set1_ps(z_len));
  } else {
    z_axis = _mm_setr_ps(0.0f, 0.0f, -1.0f, 0.0f);
  }

  // 2. X Axis = normalize(cross(up, z_axis))
  __m128 x_axis = wcn_cross_platform(vec_up, z_axis);
  __m128 x_len_sq = _mm_dp_ps(x_axis, x_axis, 0x7F);

  // FIX: Use sqrt + div
  float x_len = _mm_cvtss_f32(_mm_sqrt_ss(x_len_sq));
  if (x_len > 1e-6f) {
    x_axis = _mm_div_ps(x_axis, _mm_set1_ps(x_len));
  } else {
    x_axis = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
  }

  // 3. Y Axis = cross(z_axis, x_axis)
  __m128 y_axis = wcn_cross_platform(z_axis, x_axis);
  // Y is already normalized if Z and X are orthonormal

  // 4. Translation
  float tx = -_mm_cvtss_f32(_mm_dp_ps(x_axis, vec_eye, 0x71));
  float ty = -_mm_cvtss_f32(_mm_dp_ps(y_axis, vec_eye, 0x71));
  float tz = -_mm_cvtss_f32(_mm_dp_ps(z_axis, vec_eye, 0x71));

  // 5. Construct Matrix columns
  __m128 t0 = _mm_unpacklo_ps(x_axis, y_axis); // [xx, yx, xy, yy]
  __m128 t1 = _mm_unpackhi_ps(x_axis, y_axis); // [xz, yz, 0,  0 ]
  __m128 t2 = _mm_unpacklo_ps(z_axis, vec_zero); // [zx, 0, zy, 0]
  // __m128 t3 = _mm_unpackhi_ps(z_axis, vec_zero); // [zz, 0, 0, 0] - Not needed for col2 logic

  __m128 col0 = _mm_movelh_ps(t0, t2); // [xx, yx, zx, 0]
  __m128 col1 = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)); // [xy, yy, zy, 0]

  // Need to reconstruct col2 carefully to pick up zz
  // Low of t1 is [xz, yz]. We need [zz, 0] for high part.
  // z_axis is [zx, zy, zz, 0].
  // shuffle(z_axis, z_axis, 2, 2, 2, 2) -> [zz, zz, zz, zz]
  // This is slightly annoying with unpacks. Let's use direct shuffle.
  // We want [xz, yz, zz, 0]
  // t1 has [xz, yz, ...]
  // z_axis has [..., ..., zz, ...]
  __m128 z_shuf = _mm_shuffle_ps(z_axis, vec_zero, _MM_SHUFFLE(0, 0, 2, 2)); // [zz, zz, 0, 0]
  __m128 col2 = _mm_shuffle_ps(t1, z_shuf, _MM_SHUFFLE(2, 0, 1, 0)); // t1[0], t1[1], z_shuf[0], z_shuf[2]? No
  // _mm_movelh_ps(t1, z_shuf) -> [xz, yz, zz, zz] -> Close, but w needs to be 0.
  // Let's rely on t3 logic from previous code if it was cleaner, or just fix it:
  __m128 t3 = _mm_unpackhi_ps(z_axis, vec_zero); // [zz, 0, 0, 0]
  col2 = _mm_movelh_ps(t1, t3); // [xz, yz, zz, 0] - This is correct.

  __m128 col3 = _mm_setr_ps(tx, ty, tz, 1.0f);

  _mm_storeu_ps(&result.m[0], col0);
  _mm_storeu_ps(&result.m[4], col1);
  _mm_storeu_ps(&result.m[8], col2);
  _mm_storeu_ps(&result.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON Implementation
  float32x4_t vec_eye = wcn_load_vec3_partial(eye.v);
  float32x4_t vec_target = wcn_load_vec3_partial(target.v);
  float32x4_t vec_up = wcn_load_vec3_partial(up.v);

  float32x4_t z_axis = vsubq_f32(vec_eye, vec_target);
  float z_len = sqrtf(vaddvq_f32(vmulq_f32(z_axis, z_axis))); // High precision sqrt

  if (z_len > 1e-6f) {
      z_axis = vdivq_f32(z_axis, vdupq_n_f32(z_len));
  } else {
      float vals[] = {0,0,-1,0};
      z_axis = vld1q_f32(vals);
  }

  float32x4_t x_axis = wcn_cross_platform(vec_up, z_axis);
  float x_len = sqrtf(vaddvq_f32(vmulq_f32(x_axis, x_axis)));

  if (x_len > 1e-6f) {
      x_axis = vdivq_f32(x_axis, vdupq_n_f32(x_len));
  } else {
      float vals[] = {0,1,0,0};
      x_axis = vld1q_f32(vals);
  }

  float32x4_t y_axis = wcn_cross_platform(z_axis, x_axis);

  float tx = -vaddvq_f32(vmulq_f32(x_axis, vec_eye));
  float ty = -vaddvq_f32(vmulq_f32(y_axis, vec_eye));
  float tz = -vaddvq_f32(vmulq_f32(z_axis, vec_eye));

  // Transpose logic matches previous correct implementation
  float32x4_t t0 = vzip1q_f32(x_axis, y_axis);
  float32x4_t t1 = vzip2q_f32(x_axis, y_axis);
  float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t t2 = vzip1q_f32(z_axis, zero);
  float32x4_t t3 = vzip2q_f32(z_axis, zero);

  float32x4_t col0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
  float32x4_t col1 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
  float32x4_t col2 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));

  float vals_t[] = {tx, ty, tz, 1.0f};
  float32x4_t col3 = vld1q_f32(vals_t);

  vst1q_f32(&result.m[0], col0);
  vst1q_f32(&result.m[4], col1);
  vst1q_f32(&result.m[8], col2);
  vst1q_f32(&result.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD
  v128_t vec_eye = wcn_load_vec3_partial(eye.v);
  v128_t vec_target = wcn_load_vec3_partial(target.v);
  v128_t vec_up = wcn_load_vec3_partial(up.v);

  v128_t z_axis = wasm_f32x4_sub(vec_eye, vec_target);
  float z_len_sq = wcn_hadd_f32(wasm_f32x4_mul(z_axis, z_axis));
  if (z_len_sq > 1e-6f) {
      float inv = 1.0f / sqrtf(z_len_sq); // High precision
      z_axis = wasm_f32x4_mul(z_axis, wasm_f32x4_splat(inv));
  } else {
      z_axis = wasm_f32x4_make(0,0,-1,0);
  }

  v128_t x_axis = wcn_cross_platform(vec_up, z_axis);
  float x_len_sq = wcn_hadd_f32(wasm_f32x4_mul(x_axis, x_axis));
  if (x_len_sq > 1e-6f) {
      float inv = 1.0f / sqrtf(x_len_sq);
      x_axis = wasm_f32x4_mul(x_axis, wasm_f32x4_splat(inv));
  } else {
      x_axis = wasm_f32x4_make(0,1,0,0);
  }

  v128_t y_axis = wcn_cross_platform(z_axis, x_axis);

  float tx = -wcn_hadd_f32(wasm_f32x4_mul(x_axis, vec_eye));
  float ty = -wcn_hadd_f32(wasm_f32x4_mul(y_axis, vec_eye));
  float tz = -wcn_hadd_f32(wasm_f32x4_mul(z_axis, vec_eye));

  v128_t col0 = wasm_f32x4_make(wasm_f32x4_extract_lane(x_axis, 0), wasm_f32x4_extract_lane(y_axis, 0), wasm_f32x4_extract_lane(z_axis, 0), 0.0f);
  v128_t col1 = wasm_f32x4_make(wasm_f32x4_extract_lane(x_axis, 1), wasm_f32x4_extract_lane(y_axis, 1), wasm_f32x4_extract_lane(z_axis, 1), 0.0f);
  v128_t col2 = wasm_f32x4_make(wasm_f32x4_extract_lane(x_axis, 2), wasm_f32x4_extract_lane(y_axis, 2), wasm_f32x4_extract_lane(z_axis, 2), 0.0f);
  v128_t col3 = wasm_f32x4_make(tx, ty, tz, 1.0f);

  wasm_v128_store(&result.m[0], col0);
  wasm_v128_store(&result.m[4], col1);
  wasm_v128_store(&result.m[8], col2);
  wasm_v128_store(&result.m[12], col3);

#else
  // Scalar fallback
  // Use sqrtf instead of fast approximation if it was being used before
  float z0 = eye.v[0] - target.v[0];
  float z1 = eye.v[1] - target.v[1];
  float z2 = eye.v[2] - target.v[2];
  float len = sqrtf(z0*z0 + z1*z1 + z2*z2);
  if (len > 1e-6f) {
      len = 1.0f / len;
      z0 *= len; z1 *= len; z2 *= len;
  } else {
      z0 = 0.0f; z1 = 0.0f; z2 = -1.0f;
  }

  // X = Up x Z
  float x0 = up.v[1]*z2 - up.v[2]*z1;
  float x1 = up.v[2]*z0 - up.v[0]*z2;
  float x2 = up.v[0]*z1 - up.v[1]*z0;
  len = sqrtf(x0*x0 + x1*x1 + x2*x2);
  if (len > 1e-6f) {
      len = 1.0f / len;
      x0 *= len; x1 *= len; x2 *= len;
  } else {
      x0 = 1.0f; x1 = 0.0f; x2 = 0.0f; // Note: Default right is typically X
  }

  // Y = Z x X
  float y0 = z1*x2 - z2*x1;
  float y1 = z2*x0 - z0*x2;
  float y2 = z0*x1 - z1*x0;

  float tx = -(x0*eye.v[0] + x1*eye.v[1] + x2*eye.v[2]);
  float ty = -(y0*eye.v[0] + y1*eye.v[1] + y2*eye.v[2]);
  float tz = -(z0*eye.v[0] + z1*eye.v[1] + z2*eye.v[2]);

  result.m[0] = x0;  result.m[4] = y0;  result.m[8]  = z0;  result.m[12] = tx;
  result.m[1] = x1;  result.m[5] = y1;  result.m[9]  = z1;  result.m[13] = ty;
  result.m[2] = x2;  result.m[6] = y2;  result.m[10] = z2;  result.m[14] = tz;
  result.m[3] = 0.0f;result.m[7] = 0.0f;result.m[11] = 0.0f;result.m[15] = 1.0f;
#endif

  return result;
}

// END Mat4
