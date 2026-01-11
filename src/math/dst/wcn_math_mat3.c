#include "WCN/WCN_MATH_DST.h"
#include "math/common/wcn_math_internal.h"
#include <string.h>

// BEGIN Mat3

void WMATH_IDENTITY(Mat3)(DST_MAT3) {
    dst->m[0] = 1.0f;
    dst->m[1] = 0.0f;
    dst->m[2] = 0.0f;
    dst->m[3] = 0.0f;
    dst->m[4] = 0.0f;
    dst->m[5] = 1.0f;
    dst->m[6] = 0.0f;
    dst->m[7] = 0.0f;
    dst->m[8] = 0.0f;
    dst->m[9] = 0.0f;
    dst->m[10] = 1.0f;
    dst->m[11] = 0.0f;
}

void WMATH_ZERO(Mat3)(DST_MAT3) {
    for (int i = 0; i < 12; i++) {
        dst->m[i] = 0.0f;
    }
}

void WMATH_CREATE(Mat3)(DST_MAT3, const WMATH_CREATE_TYPE(Mat3) mat3_c) {
    dst->m[0] = WMATH_OR_ELSE_ZERO(mat3_c.m_00);
    dst->m[1] = WMATH_OR_ELSE_ZERO(mat3_c.m_01);
    dst->m[2] = WMATH_OR_ELSE_ZERO(mat3_c.m_02);
    dst->m[4] = WMATH_OR_ELSE_ZERO(mat3_c.m_10);
    dst->m[5] = WMATH_OR_ELSE_ZERO(mat3_c.m_11);
    dst->m[6] = WMATH_OR_ELSE_ZERO(mat3_c.m_12);
    dst->m[8] = WMATH_OR_ELSE_ZERO(mat3_c.m_20);
    dst->m[9] = WMATH_OR_ELSE_ZERO(mat3_c.m_21);
    dst->m[10] = WMATH_OR_ELSE_ZERO(mat3_c.m_22);
    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;
}

void WMATH_COPY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {
    memcpy(dst, &mat, sizeof(WMATH_TYPE(Mat3)));
}

bool WMATH_EQUALS(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {
    return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] && a.m[4] == b.m[4] &&
            a.m[5] == b.m[5] && a.m[6] == b.m[6] && a.m[8] == b.m[8] && a.m[9] == b.m[9] &&
            a.m[10] == b.m[10]);
}

bool WMATH_EQUALS_APPROXIMATELY(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {
    const float ep = WCN_GET_EPSILON();
    return (fabsf(a.m[0] - b.m[0]) < ep && fabsf(a.m[1] - b.m[1]) < ep &&
            fabsf(a.m[2] - b.m[2]) < ep && fabsf(a.m[4] - b.m[4]) < ep &&
            fabsf(a.m[5] - b.m[5]) < ep && fabsf(a.m[6] - b.m[6]) < ep &&
            fabsf(a.m[8] - b.m[8]) < ep && fabsf(a.m[9] - b.m[9]) < ep &&
            fabsf(a.m[10] - b.m[10]) < ep);
}

void WMATH_SET(Mat3)(DST_MAT3, const float m00, const float m01, const float m02, const float m10,
                     const float m11, const float m12, const float m20, const float m21,
                     const float m22) {
    dst->m[0] = m00;
    dst->m[1] = m01;
    dst->m[2] = m02;
    dst->m[4] = m10;
    dst->m[5] = m11;
    dst->m[6] = m12;
    dst->m[8] = m20;
    dst->m[9] = m21;
    dst->m[10] = m22;
    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;
}

void WMATH_NEGATE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 sign_mask = _mm_set1_ps(-0.0f);

    __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
    __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[4]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[8]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_res;

    vec_a = vld1q_f32(&mat.m[0]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&mat.m[4]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&mat.m[8]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t neg_one = wasm_f32x4_splat(-1.0f);

    v128_t vec_a = wasm_v128_load(&mat.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&mat.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&mat.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl = __riscv_vsetvli(4, __riscv_e32, __riscv_m1);

    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&mat.m[0], vl);
    vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, vl);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&mat.m[4], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&mat.m[8], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 sign_mask = __lsx_vldrepl_w(&(-0.0f), 0);

    __m128 vec_a = __lsx_vld(&mat.m[0], 0);
    __m128 vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&mat.m[4], 0);
    vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&mat.m[8], 0);
    vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[8], 0);

#else
    WMATH_SET(Mat3)(dst, -mat.m[0], -mat.m[1], -mat.m[2], -mat.m[4], -mat.m[5], -mat.m[6],
                    -mat.m[8], -mat.m[9], -mat.m[10]);
#endif
}

void WMATH_TRANSPOSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {
    dst->m[0] = mat.m[0];
    dst->m[1] = mat.m[4];
    dst->m[2] = mat.m[8];
    dst->m[3] = 0.0f;

    dst->m[4] = mat.m[1];
    dst->m[5] = mat.m[5];
    dst->m[6] = mat.m[9];
    dst->m[7] = 0.0f;

    dst->m[8] = mat.m[2];
    dst->m[9] = mat.m[6];
    dst->m[10] = mat.m[10];
    dst->m[11] = 0.0f;
}

void WMATH_ADD(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 row_a = wcn_mat3_get_row(&a, 0);
    __m128 row_b = wcn_mat3_get_row(&b, 0);
    __m128 row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#else
    dst->m[0] = a.m[0] + b.m[0];
    dst->m[1] = a.m[1] + b.m[1];
    dst->m[2] = a.m[2] + b.m[2];
    dst->m[3] = 0.0f;
    dst->m[4] = a.m[4] + b.m[4];
    dst->m[5] = a.m[5] + b.m[5];
    dst->m[6] = a.m[6] + b.m[6];
    dst->m[7] = 0.0f;
    dst->m[8] = a.m[8] + b.m[8];
    dst->m[9] = a.m[9] + b.m[9];
    dst->m[10] = a.m[10] + b.m[10];
    dst->m[11] = 0.0f;
#endif
}

void WMATH_SUB(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_b = _mm_loadu_ps(&b.m[0]);
    __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_b = _mm_loadu_ps(&b.m[4]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_b = _mm_loadu_ps(&b.m[8]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b, vec_res;

    vec_a = vld1q_f32(&a.m[0]);
    vec_b = vld1q_f32(&b.m[0]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&a.m[4]);
    vec_b = vld1q_f32(&b.m[4]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&a.m[8]);
    vec_b = vld1q_f32(&b.m[8]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_b, vec_res;

    vec_a = wasm_v128_load(&a.m[0]);
    vec_b = wasm_v128_load(&b.m[0]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&a.m[4]);
    vec_b = wasm_v128_load(&b.m[4]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&a.m[8]);
    vec_b = wasm_v128_load(&b.m[8]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t vec_a, vec_b, vec_res;

    vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[0], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[4], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[8], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;

    vec_a = __lsx_vld(&a.m[0], 0);
    vec_b = __lsx_vld(&b.m[0], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&a.m[4], 0);
    vec_b = __lsx_vld(&b.m[4], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&a.m[8], 0);
    vec_b = __lsx_vld(&b.m[8], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[8], 0);

#else
    dst->m[0] = a.m[0] - b.m[0];
    dst->m[1] = a.m[1] - b.m[1];
    dst->m[2] = a.m[2] - b.m[2];
    dst->m[3] = 0.0f;
    dst->m[4] = a.m[4] - b.m[4];
    dst->m[5] = a.m[5] - b.m[5];
    dst->m[6] = a.m[6] - b.m[6];
    dst->m[7] = 0.0f;
    dst->m[8] = a.m[8] - b.m[8];
    dst->m[9] = a.m[9] - b.m[9];
    dst->m[10] = a.m[10] - b.m[10];
    dst->m[11] = 0.0f;
#endif
}

void WMATH_MULTIPLY_SCALAR(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const float b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_scalar = _mm_set1_ps(b);

    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_scalar, vec_res;
    vec_scalar = vdupq_n_f32(b);

    vec_a = vld1q_f32(&a.m[0]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&a.m[4]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&a.m[8]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_scalar = wasm_f32x4_splat(b);

    v128_t vec_a = wasm_v128_load(&a.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&a.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&a.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t vec_b_scalar = __riscv_vfmv_v_f_f32m1(b, 4);

    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, 4);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_scalar = __lsx_vldrepl_w(&b, 0);

    __m128 vec_a = __lsx_vld(&a.m[0], 0);
    __m128 vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&a.m[4], 0);
    vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&a.m[8], 0);
    vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[8], 0);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#else
    for (int i = 0; i < 12; i++) {
        dst->m[i] = a.m[i] * b;
    }
#endif
}

void WMATH_INVERSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a) {
    const float m00 = a.m[0], m01 = a.m[1], m02 = a.m[2];
    const float m10 = a.m[4], m11 = a.m[5], m12 = a.m[6];
    const float m20 = a.m[8], m21 = a.m[9], m22 = a.m[10];

    const float det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
                      m02 * (m10 * m21 - m11 * m20);

    if (fabsf(det) < 1e-12f) {
        WMATH_IDENTITY(Mat3)(dst);
        return;
    }

    const float inv_det = 1.0f / det;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 inv_r0 =
        _mm_set_ps(0.0f, (m01 * m12 - m02 * m11) * inv_det, (m02 * m21 - m01 * m22) * inv_det,
                   (m11 * m22 - m12 * m21) * inv_det);
    __m128 inv_r1 =
        _mm_set_ps(0.0f, (m02 * m10 - m00 * m12) * inv_det, (m00 * m22 - m02 * m20) * inv_det,
                   (m12 * m20 - m10 * m22) * inv_det);
    __m128 inv_r2 =
        _mm_set_ps(0.0f, (m00 * m11 - m01 * m10) * inv_det, (m01 * m20 - m00 * m21) * inv_det,
                   (m10 * m21 - m11 * m20) * inv_det);

    _mm_storeu_ps(&dst->m[0], inv_r0);
    _mm_storeu_ps(&dst->m[4], inv_r1);
    _mm_storeu_ps(&dst->m[8], inv_r2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t inv_r0 = {(m11 * m22 - m12 * m21) * inv_det, (m02 * m21 - m01 * m22) * inv_det,
                          (m01 * m12 - m02 * m11) * inv_det, 0.0f};
    float32x4_t inv_r1 = {(m12 * m20 - m10 * m22) * inv_det, (m00 * m22 - m02 * m20) * inv_det,
                          (m02 * m10 - m00 * m12) * inv_det, 0.0f};
    float32x4_t inv_r2 = {(m10 * m21 - m11 * m20) * inv_det, (m01 * m20 - m00 * m21) * inv_det,
                          (m00 * m11 - m01 * m10) * inv_det, 0.0f};

    vst1q_f32(&dst->m[0], inv_r0);
    vst1q_f32(&dst->m[4], inv_r1);
    vst1q_f32(&dst->m[8], inv_r2);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t inv_r0 =
        wasm_f32x4_make(0.0f, (m01 * m12 - m02 * m11) * inv_det, (m02 * m21 - m01 * m22) * inv_det,
                        (m11 * m22 - m12 * m21) * inv_det);
    v128_t inv_r1 =
        wasm_f32x4_make(0.0f, (m02 * m10 - m00 * m12) * inv_det, (m00 * m22 - m02 * m20) * inv_det,
                        (m12 * m20 - m10 * m22) * inv_det);
    v128_t inv_r2 =
        wasm_f32x4_make(0.0f, (m00 * m11 - m01 * m10) * inv_det, (m01 * m20 - m00 * m21) * inv_det,
                        (m10 * m21 - m11 * m20) * inv_det);

    wasm_v128_store(&dst->m[0], inv_r0);
    wasm_v128_store(&dst->m[4], inv_r1);
    wasm_v128_store(&dst->m[8], inv_r2);

#else
    dst->m[0] = (m11 * m22 - m12 * m21) * inv_det;
    dst->m[1] = (m02 * m21 - m01 * m22) * inv_det;
    dst->m[2] = (m01 * m12 - m02 * m11) * inv_det;
    dst->m[3] = 0.0f;

    dst->m[4] = (m12 * m20 - m10 * m22) * inv_det;
    dst->m[5] = (m00 * m22 - m02 * m20) * inv_det;
    dst->m[6] = (m02 * m10 - m00 * m12) * inv_det;
    dst->m[7] = 0.0f;

    dst->m[8] = (m10 * m21 - m11 * m20) * inv_det;
    dst->m[9] = (m01 * m20 - m00 * m21) * inv_det;
    dst->m[10] = (m00 * m11 - m01 * m10) * inv_det;
    dst->m[11] = 0.0f;
#endif
}

void WMATH_MULTIPLY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {
    WMATH_TYPE(Mat3) result = {0};

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 row = _mm_set_ps(0.0f, a.m[2], a.m[1], a.m[0]);
    __m128 col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
    __m128 prod = _mm_mul_ps(row, col);
    __m128 sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[0] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[1] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[2] = _mm_cvtss_f32(sum);

    row = _mm_set_ps(0.0f, a.m[6], a.m[5], a.m[4]);
    col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[4] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[5] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[6] = _mm_cvtss_f32(sum);

    row = _mm_set_ps(0.0f, a.m[10], a.m[9], a.m[8]);
    col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[8] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[9], b.m[5], b.m[1]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[9] = _mm_cvtss_f32(sum);

    col = _mm_set_ps(0.0f, b.m[10], b.m[6], b.m[2]);
    prod = _mm_mul_ps(row, col);
    sum = _mm_hadd_ps(prod, prod);
    sum = _mm_hadd_ps(sum, sum);
    result.m[10] = _mm_cvtss_f32(sum);

#else
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

    result.m[3] = result.m[7] = result.m[11] = 0.0f;
    WMATH_COPY(Mat3)(dst, result);
}

float WMATH_DETERMINANT(Mat3)(const WMATH_TYPE(Mat3) m) {
    return m.m[0] * (m.m[5] * m.m[10] - m.m[9] * m.m[6]) -
           m.m[4] * (m.m[1] * m.m[10] - m.m[9] * m.m[2]) +
           m.m[8] * (m.m[1] * m.m[6] - m.m[5] * m.m[2]);
}

// END Mat3
