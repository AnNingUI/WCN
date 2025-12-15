#include "wcn_math_internal.h"
#include <string.h>

// BEGIN Mat3

WMATH_TYPE(Mat3) WMATH_IDENTITY(Mat3)() {
  return (WMATH_TYPE(Mat3)){
      1.0f, 0.0f, 0.0f, 0.0f, //
      0.0f, 1.0f, 0.0f, 0.0f, //
      0.0f, 0.0f, 1.0f, 0.0f  //
  };
};

WMATH_TYPE(Mat3) WMATH_ZERO(Mat3)() {
  return (WMATH_TYPE(Mat3)){
      0.0f, 0.0f, 0.0f, 0.0f, //
      0.0f, 0.0f, 0.0f, 0.0f, //
      0.0f, 0.0f, 0.0f, 0.0f  //
  };
}

WMATH_TYPE(Mat3)
WMATH_CREATE(Mat3)(const WMATH_CREATE_TYPE(Mat3) mat3_c) {
  WMATH_TYPE(Mat3) mat = {0};

  mat.m[0] = WMATH_OR_ELSE_ZERO(mat3_c.m_00);
  mat.m[1] = WMATH_OR_ELSE_ZERO(mat3_c.m_01);
  mat.m[2] = WMATH_OR_ELSE_ZERO(mat3_c.m_02);
  mat.m[4] = WMATH_OR_ELSE_ZERO(mat3_c.m_10);
  mat.m[5] = WMATH_OR_ELSE_ZERO(mat3_c.m_11);
  mat.m[6] = WMATH_OR_ELSE_ZERO(mat3_c.m_12);
  mat.m[8] = WMATH_OR_ELSE_ZERO(mat3_c.m_20);
  mat.m[9] = WMATH_OR_ELSE_ZERO(mat3_c.m_21);
  mat.m[10] = WMATH_OR_ELSE_ZERO(mat3_c.m_22);

  return mat;
}

WMATH_TYPE(Mat3)
WMATH_COPY(Mat3)(WMATH_TYPE(Mat3) mat) {
  WMATH_TYPE(Mat3) mat_copy;
  memcpy(&mat_copy, &mat, sizeof(WMATH_TYPE(Mat3)));
  return mat_copy;
}

bool WMATH_EQUALS(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] &&
          a.m[4] == b.m[4] && a.m[5] == b.m[5] && a.m[6] == b.m[6] &&
          a.m[8] == b.m[8] && a.m[9] == b.m[9] && a.m[10] == b.m[10]);
}

bool WMATH_EQUALS_APPROXIMATELY(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  return (fabsf(a.m[0] - b.m[0]) < WCN_GET_EPSILON() &&
          fabsf(a.m[1] - b.m[1]) < WCN_GET_EPSILON() &&
          fabsf(a.m[2] - b.m[2]) < WCN_GET_EPSILON() &&
          fabsf(a.m[4] - b.m[4]) < WCN_GET_EPSILON() &&
          fabsf(a.m[5] - b.m[5]) < WCN_GET_EPSILON() &&
          fabsf(a.m[6] - b.m[6]) < WCN_GET_EPSILON() &&
          fabsf(a.m[8] - b.m[8]) < WCN_GET_EPSILON() &&
          fabsf(a.m[9] - b.m[9]) < WCN_GET_EPSILON() &&
          fabsf(a.m[10] - b.m[10]) < WCN_GET_EPSILON());
}

WMATH_TYPE(Mat3)
WMATH_SET(Mat3)(WMATH_TYPE(Mat3) mat, float m00, float m01, float m02,
                float m10, float m11, float m12, float m20, float m21,
                float m22) {
  // Using SIMD-friendly layout
  mat.m[0] = m00;
  mat.m[1] = m01;
  mat.m[2] = m02;
  mat.m[4] = m10;
  mat.m[5] = m11;
  mat.m[6] = m12;
  mat.m[8] = m20;
  mat.m[9] = m21;
  mat.m[10] = m22;

  // Ensure unused elements are 0 for consistency
  mat.m[3] = mat.m[7] = mat.m[11] = 0.0f;

  return mat;
}

WMATH_TYPE(Mat3)
WMATH_NEGATE(Mat3)(WMATH_TYPE(Mat3) mat) {
  WMATH_TYPE(Mat3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - negate using XOR with sign bit mask
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements

  // Process first 4 elements (indices 0-3)
  __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = _mm_loadu_ps(&mat.m[4]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = _mm_loadu_ps(&mat.m[8]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a, vec_res;

  // Process first 4 elements (indices 0-3)
  vec_a = vld1q_f32(&mat.m[0]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = vld1q_f32(&mat.m[4]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = vld1q_f32(&mat.m[8]);
  vec_res = vnegq_f32(vec_a);
  vst1q_f32(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation - negate using multiply with -1.0f
  v128_t neg_one = wasm_f32x4_splat(-1.0f);

  // Process first 4 elements (indices 0-3)
  v128_t vec_a = wasm_v128_load(&mat.m[0]);
  v128_t vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = wasm_v128_load(&mat.m[4]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = wasm_v128_load(&mat.m[8]);
  vec_res = wasm_f32x4_mul(vec_a, neg_one);
  wasm_v128_store(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvli(4, __riscv_e32, __riscv_m1);

  // Process first 4 elements (indices 0-3)
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&mat.m[0], vl);
  vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, vl);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
  __riscv_vse32_v_f32m1(&result.m[0], vec_res, vl);

  // Process next 4 elements (indices 4-7)
  vec_a = __riscv_vle32_v_f32m1(&mat.m[4], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
  __riscv_vse32_v_f32m1(&result.m[4], vec_res, vl);

  // Process last 4 elements (indices 8-11)
  vec_a = __riscv_vle32_v_f32m1(&mat.m[8], vl);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
  __riscv_vse32_v_f32m1(&result.m[8], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation - negate using XOR with sign bit mask
  __m128 sign_mask = __lsx_vldrepl_w(&(-0.0f), 0); // Load -0.0f into all lanes

  // Process first 4 elements (indices 0-3)
  __m128 vec_a = __lsx_vld(&mat.m[0], 0);
  __m128 vec_res = __lsx_vxor_v(vec_a, sign_mask);
  __lsx_vst(vec_res, &result.m[0], 0);

  // Process next 4 elements (indices 4-7)
  vec_a = __lsx_vld(&mat.m[4], 0);
  vec_res = __lsx_vxor_v(vec_a, sign_mask);
  __lsx_vst(vec_res, &result.m[4], 0);

  // Process last 4 elements (indices 8-11)
  vec_a = __lsx_vld(&mat.m[8], 0);
  vec_res = __lsx_vxor_v(vec_a, sign_mask);
  __lsx_vst(vec_res, &result.m[8], 0);

#else
  // Scalar fallback
  return WMATH_SET(Mat3)(WMATH_COPY(Mat3)(mat),           // Self(Mat3)
                         -mat.m[0], -mat.m[1], -mat.m[2], // 00 ~ 02
                         -mat.m[4], -mat.m[5], -mat.m[6], // 10 ~ 12
                         -mat.m[8], -mat.m[9], -mat.m[10] // 20 ~ 22
  );
#endif

  return result;
}

WMATH_TYPE(Mat3)
WMATH_TRANSPOSE(Mat3)(WMATH_TYPE(Mat3) mat) {
  WMATH_TYPE(Mat3) result;

  // Direct assignment is optimal for 3x3 matrices regardless of SIMD support
  result.m[0] = mat.m[0]; // [0,0] -> [0,0]
  result.m[1] = mat.m[4]; // [1,0] -> [0,1]
  result.m[2] = mat.m[8]; // [2,0] -> [0,2]
  result.m[3] = 0.0f;

  result.m[4] = mat.m[1]; // [0,1] -> [1,0]
  result.m[5] = mat.m[5]; // [1,1] -> [1,1]
  result.m[6] = mat.m[9]; // [2,1] -> [1,2]
  result.m[7] = 0.0f;

  result.m[8] = mat.m[2];   // [0,2] -> [2,0]
  result.m[9] = mat.m[6];   // [1,2] -> [2,1]
  result.m[10] = mat.m[10]; // [2,2] -> [2,2]
  result.m[11] = 0.0f;

  return result;
}

// SIMD optimized matrix addition
WMATH_TYPE(Mat3)
WMATH_ADD(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  WMATH_TYPE(Mat3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - add all 3 rows at once using helper functions

  __m128 row_a = wcn_mat3_get_row(&a, 0);
  __m128 row_b = wcn_mat3_get_row(&b, 0);
  __m128 row_res = _mm_add_ps(row_a, row_b);
  wcn_mat3_set_row(&result, 0, row_res);

  row_a = wcn_mat3_get_row(&a, 1);
  row_b = wcn_mat3_get_row(&b, 1);
  row_res = _mm_add_ps(row_a, row_b);
  wcn_mat3_set_row(&result, 1, row_res);

  row_a = wcn_mat3_get_row(&a, 2);
  row_b = wcn_mat3_get_row(&b, 2);
  row_res = _mm_add_ps(row_a, row_b);
  wcn_mat3_set_row(&result, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - add all 3 rows at once using helper functions
  float32x4_t row_a, row_b, row_res;

  row_a = wcn_mat3_get_row(&a, 0);
  row_b = wcn_mat3_get_row(&b, 0);
  row_res = vaddq_f32(row_a, row_b);
  wcn_mat3_set_row(&result, 0, row_res);

  row_a = wcn_mat3_get_row(&a, 1);
  row_b = wcn_mat3_get_row(&b, 1);
  row_res = vaddq_f32(row_a, row_b);
  wcn_mat3_set_row(&result, 1, row_res);

  row_a = wcn_mat3_get_row(&a, 2);
  row_b = wcn_mat3_get_row(&b, 2);
  row_res = vaddq_f32(row_a, row_b);
  wcn_mat3_set_row(&result, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t row_a, row_b, row_res;

  row_a = wcn_mat3_get_row(&a, 0);
  row_b = wcn_mat3_get_row(&b, 0);
  row_res = wasm_f32x4_add(row_a, row_b);
  wcn_mat3_set_row(&result, 0, row_res);

  row_a = wcn_mat3_get_row(&a, 1);
  row_b = wcn_mat3_get_row(&b, 1);
  row_res = wasm_f32x4_add(row_a, row_b);
  wcn_mat3_set_row(&result, 1, row_res);

  row_a = wcn_mat3_get_row(&a, 2);
  row_b = wcn_mat3_get_row(&b, 2);
  row_res = wasm_f32x4_add(row_a, row_b);
  wcn_mat3_set_row(&result, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t row_a, row_b, row_res;

  row_a = wcn_mat3_get_row(&a, 0);
  row_b = wcn_mat3_get_row(&b, 0);
  row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
  wcn_mat3_set_row(&result, 0, row_res);

  row_a = wcn_mat3_get_row(&a, 1);
  row_b = wcn_mat3_get_row(&b, 1);
  row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
  wcn_mat3_set_row(&result, 1, row_res);

  row_a = wcn_mat3_get_row(&a, 2);
  row_b = wcn_mat3_get_row(&b, 2);
  row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
  wcn_mat3_set_row(&result, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 row_a, row_b, row_res;

  row_a = wcn_mat3_get_row(&a, 0);
  row_b = wcn_mat3_get_row(&b, 0);
  row_res = __lsx_vfadd_s(row_a, row_b);
  wcn_mat3_set_row(&result, 0, row_res);

  row_a = wcn_mat3_get_row(&a, 1);
  row_b = wcn_mat3_get_row(&b, 1);
  row_res = __lsx_vfadd_s(row_a, row_b);
  wcn_mat3_set_row(&result, 1, row_res);

  row_a = wcn_mat3_get_row(&a, 2);
  row_b = wcn_mat3_get_row(&b, 2);
  row_res = __lsx_vfadd_s(row_a, row_b);
  wcn_mat3_set_row(&result, 2, row_res);

#else
  // Scalar fallback implementation
  result.m[0] = a.m[0] + b.m[0];
  result.m[1] = a.m[1] + b.m[1];
  result.m[2] = a.m[2] + b.m[2];
  result.m[3] = 0.0f;
  result.m[4] = a.m[4] + b.m[4];
  result.m[5] = a.m[5] + b.m[5];
  result.m[6] = a.m[6] + b.m[6];
  result.m[7] = 0.0f;
  result.m[8] = a.m[8] + b.m[8];
  result.m[9] = a.m[9] + b.m[9];
  result.m[10] = a.m[10] + b.m[10];
  result.m[11] = 0.0f;
#endif

  return result;
}

// SIMD optimized matrix subtraction
WMATH_TYPE(Mat3)
WMATH_SUB(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  WMATH_TYPE(Mat3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation

  // Process first 4 elements
  __m128 vec_a = _mm_loadu_ps(&a.m[0]);
  __m128 vec_b = _mm_loadu_ps(&b.m[0]);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_b = _mm_loadu_ps(&b.m[4]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_b = _mm_loadu_ps(&b.m[8]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a, vec_b, vec_res;

  // Process first 4 elements
  vec_a = vld1q_f32(&a.m[0]);
  vec_b = vld1q_f32(&b.m[0]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = vld1q_f32(&a.m[4]);
  vec_b = vld1q_f32(&b.m[4]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = vld1q_f32(&a.m[8]);
  vec_b = vld1q_f32(&b.m[8]);
  vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a, vec_b, vec_res;

  // Process first 4 elements
  vec_a = wasm_v128_load(&a.m[0]);
  vec_b = wasm_v128_load(&b.m[0]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = wasm_v128_load(&a.m[4]);
  vec_b = wasm_v128_load(&b.m[4]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = wasm_v128_load(&a.m[8]);
  vec_b = wasm_v128_load(&b.m[8]);
  vec_res = wasm_f32x4_sub(vec_a, vec_b);
  wasm_v128_store(&result.m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_a, vec_b, vec_res;

  // Process first 4 elements
  vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
  vec_b = __riscv_vle32_v_f32m1(&b.m[0], 4);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[0], vec_res, 4);

  // Process next 4 elements
  vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
  vec_b = __riscv_vle32_v_f32m1(&b.m[4], 4);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[4], vec_res, 4);

  // Process last 4 elements
  vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
  vec_b = __riscv_vle32_v_f32m1(&b.m[8], 4);
  vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[8], vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a, vec_b, vec_res;

  // Process first 4 elements
  vec_a = __lsx_vld(&a.m[0], 0);
  vec_b = __lsx_vld(&b.m[0], 0);
  vec_res = __lsx_vfsub_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[0], 0);

  // Process next 4 elements
  vec_a = __lsx_vld(&a.m[4], 0);
  vec_b = __lsx_vld(&b.m[4], 0);
  vec_res = __lsx_vfsub_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[4], 0);

  // Process last 4 elements
  vec_a = __lsx_vld(&a.m[8], 0);
  vec_b = __lsx_vld(&b.m[8], 0);
  vec_res = __lsx_vfsub_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[8], 0);

#else
  // Scalar fallback
  result.m[0] = a.m[0] - b.m[0];
  result.m[1] = a.m[1] - b.m[1];
  result.m[2] = a.m[2] - b.m[2];
  result.m[3] = 0.0f;
  result.m[4] = a.m[4] - b.m[4];
  result.m[5] = a.m[5] - b.m[5];
  result.m[6] = a.m[6] - b.m[6];
  result.m[7] = 0.0f;
  result.m[8] = a.m[8] - b.m[8];
  result.m[9] = a.m[9] - b.m[9];
  result.m[10] = a.m[10] - b.m[10];
  result.m[11] = 0.0f;
#endif

  return result;
}

// SIMD optimized scalar multiplication
WMATH_TYPE(Mat3)
WMATH_MULTIPLY_SCALAR(Mat3)(WMATH_TYPE(Mat3) a, float b) {
  WMATH_TYPE(Mat3) result;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - broadcast scalar to all vector elements and multiply
  __m128 vec_b = _mm_set1_ps(b); // Broadcast scalar to 4 elements

  // Process first 4 elements
  __m128 vec_a = _mm_loadu_ps(&a.m[0]);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[8], vec_res);

  // Ensure unused elements remain 0
  result.m[3] = result.m[7] = result.m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float32x4_t vec_a, vec_b, vec_res;
  vec_b = vdupq_n_f32(b); // Broadcast scalar to 4 elements

  // Process first 4 elements
  vec_a = vld1q_f32(&a.m[0]);
  vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = vld1q_f32(&a.m[4]);
  vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = vld1q_f32(&a.m[8]);
  vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[8], vec_res);

  // Ensure unused elements remain 0
  result.m[3] = result.m[7] = result.m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_b = wasm_f32x4_splat(b); // Broadcast scalar to 4 elements

  // Process first 4 elements
  v128_t vec_a = wasm_v128_load(&a.m[0]);
  v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wasm_v128_store(&result.m[0], vec_res);

  // Process next 4 elements
  vec_a = wasm_v128_load(&a.m[4]);
  vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wasm_v128_store(&result.m[4], vec_res);

  // Process last 4 elements
  vec_a = wasm_v128_load(&a.m[8]);
  vec_res = wasm_f32x4_mul(vec_a, vec_b);
  wasm_v128_store(&result.m[8], vec_res);

  // Ensure unused elements remain 0
  result.m[3] = result.m[7] = result.m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_b = __riscv_vfmv_v_f_f32m1(b, 4); // Broadcast scalar to 4 elements

  // Process first 4 elements
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
  vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[0], vec_res, 4);

  // Process next 4 elements
  vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[4], vec_res, 4);

  // Process last 4 elements
  vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
  vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b, 4);
  __riscv_vse32_v_f32m1(&result.m[8], vec_res, 4);

  // Ensure unused elements remain 0
  result.m[3] = result.m[7] = result.m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_b = __lsx_vldrepl_w(&b, 0); // Broadcast scalar to 4 elements

  // Process first 4 elements
  __m128 vec_a = __lsx_vld(&a.m[0], 0);
  __m128 vec_res = __lsx_vfmul_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[0], 0);

  // Process next 4 elements
  vec_a = __lsx_vld(&a.m[4], 0);
  vec_res = __lsx_vfmul_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[4], 0);

  // Process last 4 elements
  vec_a = __lsx_vld(&a.m[8], 0);
  vec_res = __lsx_vfmul_s(vec_a, vec_b);
  __lsx_vst(vec_res, &result.m[8], 0);

  // Ensure unused elements remain 0
  result.m[3] = result.m[7] = result.m[11] = 0.0f;

#else
  // Scalar fallback
  result.m[0] = a.m[0] * b;
  result.m[1] = a.m[1] * b;
  result.m[2] = a.m[2] * b;
  result.m[3] = 0.0f;
  result.m[4] = a.m[4] * b;
  result.m[5] = a.m[5] * b;
  result.m[6] = a.m[6] * b;
  result.m[7] = 0.0f;
  result.m[8] = a.m[8] * b;
  result.m[9] = a.m[9] * b;
  result.m[10] = a.m[10] * b;
  result.m[11] = 0.0f;
#endif

  return result;
}

WMATH_TYPE(Mat3)
WMATH_INVERSE(Mat3)(const WMATH_TYPE(Mat3) a) {
    WMATH_TYPE(Mat3) out;

    // 标量取值方便算 det
    float m00 = a.m[0], m01 = a.m[1], m02 = a.m[2];
    float m10 = a.m[4], m11 = a.m[5], m12 = a.m[6];
    float m20 = a.m[8], m21 = a.m[9], m22 = a.m[10];

    const float det = m00 * (m11 * m22 - m12 * m21)
              - m01 * (m10 * m22 - m12 * m20)
              + m02 * (m10 * m21 - m11 * m20);

    if (fabsf(det) < 1e-12f) {
        // fallback identity
        out.m[0] = 1; out.m[1] = 0; out.m[2] = 0; out.m[3] = 0;
        out.m[4] = 0; out.m[5] = 1; out.m[6] = 0; out.m[7] = 0;
        out.m[8] = 0; out.m[9] = 0; out.m[10]= 1; out.m[11]= 0;
        return out;
    }

    float inv_det = 1.0f / det;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // ========================= SSE2 实现 =========================

    __m128 inv_r0 = _mm_set_ps(0.0f,
        (m01*m12 - m02*m11) * inv_det,
        (m02*m21 - m01*m22) * inv_det,
        (m11*m22 - m12*m21) * inv_det
    );
    __m128 inv_r1 = _mm_set_ps(0.0f,
        (m02*m10 - m00*m12) * inv_det,
        (m00*m22 - m02*m20) * inv_det,
        (m12*m20 - m10*m22) * inv_det
    );
    __m128 inv_r2 = _mm_set_ps(0.0f,
        (m00*m11 - m01*m10) * inv_det,
        (m01*m20 - m00*m21) * inv_det,
        (m10*m21 - m11*m20) * inv_det
    );

    _mm_storeu_ps(&out.m[0],  inv_r0);
    _mm_storeu_ps(&out.m[4],  inv_r1);
    _mm_storeu_ps(&out.m[8],  inv_r2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ========================= ARM NEON 实现 =========================

    float32x4_t inv_r0 = {
        (m11*m22 - m12*m21) * inv_det,
        (m02*m21 - m01*m22) * inv_det,
        (m01*m12 - m02*m11) * inv_det,
        0.0f
    };
    float32x4_t inv_r1 = {
        (m12*m20 - m10*m22) * inv_det,
        (m00*m22 - m02*m20) * inv_det,
        (m02*m10 - m00*m12) * inv_det,
        0.0f
    };
    float32x4_t inv_r2 = {
        (m10*m21 - m11*m20) * inv_det,
        (m01*m20 - m00*m21) * inv_det,
        (m00*m11 - m01*m10) * inv_det,
        0.0f
    };

    vst1q_f32(&out.m[0],  inv_r0);
    vst1q_f32(&out.m[4],  inv_r1);
    vst1q_f32(&out.m[8],  inv_r2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // ========================= WebAssembly SIMD 实现 =========================

    v128_t inv_r0 = wasm_f32x4_make(
        0.0f,
        (m01*m12 - m02*m11) * inv_det,
        (m02*m21 - m01*m22) * inv_det,
        (m11*m22 - m12*m21) * inv_det
    );

    v128_t inv_r1 = wasm_f32x4_make(
        0.0f,
        (m02*m10 - m00*m12) * inv_det,
        (m00*m22 - m02*m20) * inv_det,
        (m12*m20 - m10*m22) * inv_det
    );

    v128_t inv_r2 = wasm_f32x4_make(
        0.0f,
        (m00*m11 - m01*m10) * inv_det,
        (m01*m20 - m00*m21) * inv_det,
        (m10*m21 - m11*m20) * inv_det
    );

    wasm_v128_store(&out.m[0],  inv_r0);
    wasm_v128_store(&out.m[4],  inv_r1);
    wasm_v128_store(&out.m[8],  inv_r2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // ========================= RISC-V Vector 实现 =========================

    float temp_r0[4] = {
        (m11*m22 - m12*m21) * inv_det,
        (m02*m21 - m01*m22) * inv_det,
        (m01*m12 - m02*m11) * inv_det,
        0.0f
    };
    float temp_r1[4] = {
        (m12*m20 - m10*m22) * inv_det,
        (m00*m22 - m02*m20) * inv_det,
        (m02*m10 - m00*m12) * inv_det,
        0.0f
    };
    float temp_r2[4] = {
        (m10*m21 - m11*m20) * inv_det,
        (m01*m20 - m00*m21) * inv_det,
        (m00*m11 - m01*m10) * inv_det,
        0.0f
    };

    vfloat32m1_t inv_r0 = __riscv_vle32_v_f32m1(temp_r0, 4);
    vfloat32m1_t inv_r1 = __riscv_vle32_v_f32m1(temp_r1, 4);
    vfloat32m1_t inv_r2 = __riscv_vle32_v_f32m1(temp_r2, 4);

    __riscv_vse32_v_f32m1(&out.m[0],  inv_r0, 4);
    __riscv_vse32_v_f32m1(&out.m[4],  inv_r1, 4);
    __riscv_vse32_v_f32m1(&out.m[8],  inv_r2, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // ========================= LoongArch LSX 实现 =========================

    __m128 inv_r0 = (__m128){
        (m11*m22 - m12*m21) * inv_det,
        (m02*m21 - m01*m22) * inv_det,
        (m01*m12 - m02*m11) * inv_det,
        0.0f
    };

    __m128 inv_r1 = (__m128){
        (m12*m20 - m10*m22) * inv_det,
        (m00*m22 - m02*m20) * inv_det,
        (m02*m10 - m00*m12) * inv_det,
        0.0f
    };

    __m128 inv_r2 = (__m128){
        (m10*m21 - m11*m20) * inv_det,
        (m01*m20 - m00*m21) * inv_det,
        (m00*m11 - m01*m10) * inv_det,
        0.0f
    };

    __lsx_vst(inv_r0, &out.m[0], 0);
    __lsx_vst(inv_r1, &out.m[4], 0);
    __lsx_vst(inv_r2, &out.m[8], 0);

#else
    // ========================= 标量 fallback =========================
    out.m[0]  = (m11*m22 - m12*m21) * inv_det;
    out.m[1]  = (m02*m21 - m01*m22) * inv_det;
    out.m[2]  = (m01*m12 - m02*m11) * inv_det;
    out.m[3]  = 0.0f;

    out.m[4]  = (m12*m20 - m10*m22) * inv_det;
    out.m[5]  = (m00*m22 - m02*m20) * inv_det;
    out.m[6]  = (m02*m10 - m00*m12) * inv_det;
    out.m[7]  = 0.0f;

    out.m[8]  = (m10*m21 - m11*m20) * inv_det;
    out.m[9]  = (m01*m20 - m00*m21) * inv_det;
    out.m[10] = (m00*m11 - m01*m10) * inv_det;
    out.m[11] = 0.0f;
#endif

    return out;
}



// Optimized matrix multiplication
WMATH_TYPE(Mat3)
WMATH_MULTIPLY(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  WMATH_TYPE(Mat3) result = {0};

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE optimized matrix multiplication

  // Calculate the first row of a result
  // [0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  __m128 row = _mm_set_ps(0.0f, a.m[2], a.m[1], a.m[0]);
  __m128 col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
  __m128 prod = _mm_mul_ps(row, col);
  __m128 sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[0] = _mm_cvtss_f32(sum);

  // result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
  col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[1] = _mm_cvtss_f32(sum);

  // result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
  col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[2] = _mm_cvtss_f32(sum);

  // Calculate second row of result
  // result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
  row = _mm_set_ps(0.0f, a.m[6], a.m[5], a.m[4]);
  col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[4] = _mm_cvtss_f32(sum);

  // result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
  col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[5] = _mm_cvtss_f32(sum);

  // result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
  col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[6] = _mm_cvtss_f32(sum);

  // Calculate the third row of a result
  // [8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
  row = _mm_set_ps(0.0f, a.m[10], a.m[9], a.m[8]);
  col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[8] = _mm_cvtss_f32(sum);

  // result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
  col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[9] = _mm_cvtss_f32(sum);

  // result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
  col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[10] = _mm_cvtss_f32(sum);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON optimized matrix multiplication
  float32x4_t row, col, prod, sum;

  // Calculate the first row of a result
  row = vld1q_f32(&a.m[0]); // a[0], a[1], a[2], a[3] (a[3] is 0)
  // result[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  col = vsetq_lane_f32(
      b.m[0],
      vsetq_lane_f32(b.m[4], vsetq_lane_f32(b.m[8], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[0] = vgetq_lane_f32(sum, 0);

  // result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
  col = vsetq_lane_f32(
      b.m[1],
      vsetq_lane_f32(b.m[5], vsetq_lane_f32(b.m[9], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[1] = vgetq_lane_f32(sum, 0);

  // result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
  col = vsetq_lane_f32(
      b.m[2],
      vsetq_lane_f32(b.m[6], vsetq_lane_f32(b.m[10], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[2] = vgetq_lane_f32(sum, 0);

  // Calculate the second row of a result
  row = vld1q_f32(&a.m[4]); // a[4], a[5], a[6], a[7] (a[7] is 0)

  // result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
  col = vsetq_lane_f32(
      b.m[0],
      vsetq_lane_f32(b.m[4], vsetq_lane_f32(b.m[8], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[4] = vgetq_lane_f32(sum, 0);

  // result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
  col = vsetq_lane_f32(
      b.m[1],
      vsetq_lane_f32(b.m[5], vsetq_lane_f32(b.m[9], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[5] = vgetq_lane_f32(sum, 0);

  // result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
  col = vsetq_lane_f32(
      b.m[2],
      vsetq_lane_f32(b.m[6], vsetq_lane_f32(b.m[10], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[6] = vgetq_lane_f32(sum, 0);

  // Calculate the third row of a result
  row = vld1q_f32(&a.m[8]); // a[8], a[9], a[10], a[11] (a[11] is 0)

  // result[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
  col = vsetq_lane_f32(
      b.m[0],
      vsetq_lane_f32(b.m[4], vsetq_lane_f32(b.m[8], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[8] = vgetq_lane_f32(sum, 0);

  // result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
  col = vsetq_lane_f32(
      b.m[1],
      vsetq_lane_f32(b.m[5], vsetq_lane_f32(b.m[9], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[9] = vgetq_lane_f32(sum, 0);

  // result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
  col = vsetq_lane_f32(
      b.m[2],
      vsetq_lane_f32(b.m[6], vsetq_lane_f32(b.m[10], vdupq_n_f32(0.0f), 2), 1),
      0);
  prod = vmulq_f32(row, col);
  sum = vpaddq_f32(prod, prod);
  sum = vpaddq_f32(sum, sum);
  result.m[10] = vgetq_lane_f32(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD optimized matrix multiplication

  // Calculate the first row of a result
  // [0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  v128_t row = wasm_f32x4_make(0.0f, a.m[2], a.m[1], a.m[0]);
  v128_t col = wasm_f32x4_make(0.0f, b.m[8], b.m[4], b.m[0]);
  v128_t prod = wasm_f32x4_mul(row, col);
  v128_t sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[0] = wasm_f32x4_extract_lane(sum, 0);

  // result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
  col = wasm_f32x4_make(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[1] = wasm_f32x4_extract_lane(sum, 0);

  // result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
  col = wasm_f32x4_make(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[2] = wasm_f32x4_extract_lane(sum, 0);

  // Calculate second row of result
  // result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
  row = wasm_f32x4_make(0.0f, a.m[6], a.m[5], a.m[4]);
  col = wasm_f32x4_make(0.0f, b.m[8], b.m[4], b.m[0]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[4] = wasm_f32x4_extract_lane(sum, 0);

  // result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
  col = wasm_f32x4_make(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[5] = wasm_f32x4_extract_lane(sum, 0);

  // result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
  col = wasm_f32x4_make(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[6] = wasm_f32x4_extract_lane(sum, 0);

  // Calculate the third row of a result
  // [8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
  row = wasm_f32x4_make(0.0f, a.m[10], a.m[9], a.m[8]);
  col = wasm_f32x4_make(0.0f, b.m[8], b.m[4], b.m[0]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[8] = wasm_f32x4_extract_lane(sum, 0);

  // result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
  col = wasm_f32x4_make(0.0f, b.m[9], b.m[5], b.m[1]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[9] = wasm_f32x4_extract_lane(sum, 0);

  // result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
  col = wasm_f32x4_make(0.0f, b.m[10], b.m[6], b.m[2]);
  prod = wasm_f32x4_mul(row, col);
  sum = wasm_f32x4_add(prod, wasm_i32x4_shuffle(prod, prod, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 0, 0));
  result.m[10] = wasm_f32x4_extract_lane(sum, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension optimized matrix multiplication

  // Calculate the first row of a result
  // [0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  float temp_row[4] = {a.m[0], a.m[1], a.m[2], 0.0f};
  float temp_col[4] = {b.m[0], b.m[4], b.m[8], 0.0f};
  vfloat32m1_t row = __riscv_vle32_v_f32m1(temp_row, 4);
  vfloat32m1_t col = __riscv_vle32_v_f32m1(temp_col, 4);
  vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  float sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[0] = sum;

  // result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
  temp_col[0] = b.m[1]; temp_col[1] = b.m[5]; temp_col[2] = b.m[9];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[1] = sum;

  // result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
  temp_col[0] = b.m[2]; temp_col[1] = b.m[6]; temp_col[2] = b.m[10];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[2] = sum;

  // Calculate second row of result
  // result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
  temp_row[0] = a.m[4]; temp_row[1] = a.m[5]; temp_row[2] = a.m[6];
  row = __riscv_vle32_v_f32m1(temp_row, 4);
  temp_col[0] = b.m[0]; temp_col[1] = b.m[4]; temp_col[2] = b.m[8];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[4] = sum;

  // result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
  temp_col[0] = b.m[1]; temp_col[1] = b.m[5]; temp_col[2] = b.m[9];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[5] = sum;

  // result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
  temp_col[0] = b.m[2]; temp_col[1] = b.m[6]; temp_col[2] = b.m[10];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[6] = sum;

  // Calculate the third row of a result
  // [8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
  temp_row[0] = a.m[8]; temp_row[1] = a.m[9]; temp_row[2] = a.m[10];
  row = __riscv_vle32_v_f32m1(temp_row, 4);
  temp_col[0] = b.m[0]; temp_col[1] = b.m[4]; temp_col[2] = b.m[8];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[8] = sum;

  // result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
  temp_col[0] = b.m[1]; temp_col[1] = b.m[5]; temp_col[2] = b.m[9];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[9] = sum;

  // result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
  temp_col[0] = b.m[2]; temp_col[1] = b.m[6]; temp_col[2] = b.m[10];
  col = __riscv_vle32_v_f32m1(temp_col, 4);
  prod = __riscv_vfmul_vv_f32m1(row, col, 4);
  sum = __riscv_vfredusum_vs_f32m1_f32(prod, __riscv_vfmv_v_f_f32m1(0.0f, 4), 4);
  result.m[10] = sum;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX optimized matrix multiplication

  // Calculate the first row of a result
  // [0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  __m128 row = __lsx_vldrepl_w(&a.m[0], 0);
  row = __lsx_vinsgr2vr_w(row, a.m[1], 1);
  row = __lsx_vinsgr2vr_w(row, a.m[2], 2);
  __m128 col = __lsx_vldrepl_w(&b.m[0], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[4], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[8], 2);
  __m128 prod = __lsx_vfmul_s(row, col);
  __m128 shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(prod, shuf);
  result.m[0] = __lsx_vfrep2vr_w(sum)[0];

  // result[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
  col = __lsx_vldrepl_w(&b.m[1], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[5], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[9], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[1] = __lsx_vfrep2vr_w(sum)[0];

  // result[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
  col = __lsx_vldrepl_w(&b.m[2], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[6], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[10], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[2] = __lsx_vfrep2vr_w(sum)[0];

  // Calculate second row of result
  // result[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
  row = __lsx_vldrepl_w(&a.m[4], 0);
  row = __lsx_vinsgr2vr_w(row, a.m[5], 1);
  row = __lsx_vinsgr2vr_w(row, a.m[6], 2);
  col = __lsx_vldrepl_w(&b.m[0], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[4], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[8], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[4] = __lsx_vfrep2vr_w(sum)[0];

  // result[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
  col = __lsx_vldrepl_w(&b.m[1], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[5], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[9], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[5] = __lsx_vfrep2vr_w(sum)[0];

  // result[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
  col = __lsx_vldrepl_w(&b.m[2], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[6], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[10], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[6] = __lsx_vfrep2vr_w(sum)[0];

  // Calculate the third row of a result
  // [8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
  row = __lsx_vldrepl_w(&a.m[8], 0);
  row = __lsx_vinsgr2vr_w(row, a.m[9], 1);
  row = __lsx_vinsgr2vr_w(row, a.m[10], 2);
  col = __lsx_vldrepl_w(&b.m[0], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[4], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[8], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[8] = __lsx_vfrep2vr_w(sum)[0];

  // result[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
  col = __lsx_vldrepl_w(&b.m[1], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[5], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[9], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[9] = __lsx_vfrep2vr_w(sum)[0];

  // result[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
  col = __lsx_vldrepl_w(&b.m[2], 0);
  col = __lsx_vinsgr2vr_w(col, b.m[6], 1);
  col = __lsx_vinsgr2vr_w(col, b.m[10], 2);
  prod = __lsx_vfmul_s(row, col);
  shuf = __lsx_vshuf4i_w(prod, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(prod, shuf);
  result.m[10] = __lsx_vfrep2vr_w(sum)[0];

#else
  // Original scalar implementation
  result.m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[4] + a.m[2] * b.m[8];
  result.m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[5] + a.m[2] * b.m[9];
  result.m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[6] + a.m[2] * b.m[10];
  result.m[4] = a.m[4] * b.m[0] + a.m[5] * b.m[4] + a.m[6] * b.m[8];
  result.m[5] = a.m[4] * b.m[1] + a.m[5] * b.m[5] + a.m[6] * b.m[9];
  result.m[6] = a.m[4] * b.m[2] + a.m[5] * b.m[6] + a.m[6] * b.m[10];
  result.m[8] = a.m[8] * b.m[0] + a.m[9] * b.m[4] + a.m[10] * b.m[8];
  result.m[9] = a.m[8] * b.m[1] + a.m[9] * b.m[5] + a.m[10] * b.m[9];
  result.m[10] = a.m[8] * b.m[2] + a.m[9] * b.m[6] + a.m[10] * b.m[10];
#endif

  return result;
}

float WMATH_DETERMINANT(Mat3)(const WMATH_TYPE(Mat3) m) {
  const float m00 = m.m[0 * 4 + 0];
  const float m01 = m.m[0 * 4 + 1];
  const float m02 = m.m[0 * 4 + 2];
  const float m10 = m.m[1 * 4 + 0];
  const float m11 = m.m[1 * 4 + 1];
  const float m12 = m.m[1 * 4 + 2];
  const float m20 = m.m[2 * 4 + 0];
  const float m21 = m.m[2 * 4 + 1];
  const float m22 = m.m[2 * 4 + 2];

  return m00 * (m11 * m22 - m21 * m12) - m10 * (m01 * m22 - m21 * m02) +
         m20 * (m01 * m12 - m11 * m02);
}

// END Mat3
