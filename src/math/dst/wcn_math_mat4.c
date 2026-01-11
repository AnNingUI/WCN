#include "WCN/WCN_MATH_DST.h"
#include "math/common/wcn_math_internal.h"
#include <string.h>
static inline void __prev_clangd_check() {
wcn_math_get_epsilon();
}
// BEGIN Mat4

void WMATH_IDENTITY(Mat4)(DST_MAT4) {
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
    dst->m[12] = 0.0f;
    dst->m[13] = 0.0f;
    dst->m[14] = 0.0f;
    dst->m[15] = 1.0f;
}

void WMATH_ZERO(Mat4)(DST_MAT4) {
    for (int i = 0; i < 16; i++) {
        dst->m[i] = 0.0f;
    }
}

void WMATH_CREATE(Mat4)(DST_MAT4, const WMATH_CREATE_TYPE(Mat4) mat4_c) {
    dst->m[0] = WMATH_OR_ELSE_ZERO(mat4_c.m_00);
    dst->m[1] = WMATH_OR_ELSE_ZERO(mat4_c.m_01);
    dst->m[2] = WMATH_OR_ELSE_ZERO(mat4_c.m_02);
    dst->m[3] = WMATH_OR_ELSE_ZERO(mat4_c.m_03);
    dst->m[4] = WMATH_OR_ELSE_ZERO(mat4_c.m_10);
    dst->m[5] = WMATH_OR_ELSE_ZERO(mat4_c.m_11);
    dst->m[6] = WMATH_OR_ELSE_ZERO(mat4_c.m_12);
    dst->m[7] = WMATH_OR_ELSE_ZERO(mat4_c.m_13);
    dst->m[8] = WMATH_OR_ELSE_ZERO(mat4_c.m_20);
    dst->m[9] = WMATH_OR_ELSE_ZERO(mat4_c.m_21);
    dst->m[10] = WMATH_OR_ELSE_ZERO(mat4_c.m_22);
    dst->m[11] = WMATH_OR_ELSE_ZERO(mat4_c.m_23);
    dst->m[12] = WMATH_OR_ELSE_ZERO(mat4_c.m_30);
    dst->m[13] = WMATH_OR_ELSE_ZERO(mat4_c.m_31);
    dst->m[14] = WMATH_OR_ELSE_ZERO(mat4_c.m_32);
    dst->m[15] = WMATH_OR_ELSE_ZERO(mat4_c.m_33);
}

void WMATH_COPY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat) {
    memcpy(dst, &mat, sizeof(WMATH_TYPE(Mat4)));
}

void WMATH_SET(Mat4)(DST_MAT4, const float m00, const float m01, const float m02, const float m03,
                     const float m10, const float m11, const float m12, const float m13,
                     const float m20, const float m21, const float m22, const float m23,
                     const float m30, const float m31, const float m32, const float m33) {
    dst->m[0] = m00;
    dst->m[1] = m01;
    dst->m[2] = m02;
    dst->m[3] = m03;
    dst->m[4] = m10;
    dst->m[5] = m11;
    dst->m[6] = m12;
    dst->m[7] = m13;
    dst->m[8] = m20;
    dst->m[9] = m21;
    dst->m[10] = m22;
    dst->m[11] = m23;
    dst->m[12] = m30;
    dst->m[13] = m31;
    dst->m[14] = m32;
    dst->m[15] = m33;
}

void WMATH_NEGATE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
    __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[4]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[8]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[8], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[12]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[12], vec_res);

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

    vec_a = vld1q_f32(&mat.m[12]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_res;
    const v128_t neg_one = wasm_f32x4_splat(-1.0f);

    vec_a = wasm_v128_load(&mat.m[0]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&mat.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&mat.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[8], vec_res);

    vec_a = wasm_v128_load(&mat.m[12]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_res;
    float* a_ptr = (float*)&mat;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvli(4, RV32, e32, m1);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_res;
    const __m128 sign_mask = (__m128)__lsx_vldrepl_w(&wcn_minus_zero_bits, 0);

    vec_a = (__m128)__lsx_vld(&mat.m[0], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[4], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[8], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[12], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);

#else
    for (int i = 0; i < 16; i++) {
        dst->m[i] = -mat.m[i];
    }
#endif
}

bool WMATH_EQUALS(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
    return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] && a.m[3] == b.m[3] &&
            a.m[4] == b.m[4] && a.m[5] == b.m[5] && a.m[6] == b.m[6] && a.m[7] == b.m[7] &&
            a.m[8] == b.m[8] && a.m[9] == b.m[9] && a.m[10] == b.m[10] && a.m[11] == b.m[11] &&
            a.m[12] == b.m[12] && a.m[13] == b.m[13] && a.m[14] == b.m[14] && a.m[15] == b.m[15]);
}

bool WMATH_EQUALS_APPROXIMATELY(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
    return (
        fabsf(a.m[0] - b.m[0]) < WCN_GET_EPSILON() && fabsf(a.m[1] - b.m[1]) < WCN_GET_EPSILON() &&
        fabsf(a.m[2] - b.m[2]) < WCN_GET_EPSILON() && fabsf(a.m[3] - b.m[3]) < WCN_GET_EPSILON() &&
        fabsf(a.m[4] - b.m[4]) < WCN_GET_EPSILON() && fabsf(a.m[5] - b.m[5]) < WCN_GET_EPSILON() &&
        fabsf(a.m[6] - b.m[6]) < WCN_GET_EPSILON() && fabsf(a.m[7] - b.m[7]) < WCN_GET_EPSILON() &&
        fabsf(a.m[8] - b.m[8]) < WCN_GET_EPSILON() && fabsf(a.m[9] - b.m[9]) < WCN_GET_EPSILON() &&
        fabsf(a.m[10] - b.m[10]) < WCN_GET_EPSILON() &&
        fabsf(a.m[11] - b.m[11]) < WCN_GET_EPSILON() &&
        fabsf(a.m[12] - b.m[12]) < WCN_GET_EPSILON() &&
        fabsf(a.m[13] - b.m[13]) < WCN_GET_EPSILON() &&
        fabsf(a.m[14] - b.m[14]) < WCN_GET_EPSILON() &&
        fabsf(a.m[15] - b.m[15]) < WCN_GET_EPSILON());
}

void WMATH_ADD(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_b = _mm_loadu_ps(&b.m[0]);
    __m128 vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[0], vec_res);
    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_b = _mm_loadu_ps(&b.m[4]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[4], vec_res);
    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_b = _mm_loadu_ps(&b.m[8]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[8], vec_res);
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_b = _mm_loadu_ps(&b.m[12]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b, vec_res;
    vec_a = vld1q_f32(&a.m[0]);
    vec_b = vld1q_f32(&b.m[0]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[0], vec_res);
    vec_a = vld1q_f32(&a.m[4]);
    vec_b = vld1q_f32(&b.m[4]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[4], vec_res);
    vec_a = vld1q_f32(&a.m[8]);
    vec_b = vld1q_f32(&b.m[8]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[8], vec_res);
    vec_a = vld1q_f32(&a.m[12]);
    vec_b = vld1q_f32(&b.m[12]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_b, vec_res;
    vec_a = wasm_v128_load(&a.m[0]);
    vec_b = wasm_v128_load(&b.m[0]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[0], vec_res);
    vec_a = wasm_v128_load(&a.m[4]);
    vec_b = wasm_v128_load(&b.m[4]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[4], vec_res);
    vec_a = wasm_v128_load(&a.m[8]);
    vec_b = wasm_v128_load(&b.m[8]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[8], vec_res);
    vec_a = wasm_v128_load(&a.m[12]);
    vec_b = wasm_v128_load(&b.m[12]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b, vec_res;
    float* a_ptr = (float*)&a;
    float* b_ptr = (float*)&b;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvli(4, RV32, e32, m1);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;
    vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    vec_b = (__m128)__lsx_vld(&b.m[0], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_b = (__m128)__lsx_vld(&b.m[4], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_b = (__m128)__lsx_vld(&b.m[8], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_b = (__m128)__lsx_vld(&b.m[12], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] + b.m[i];
    }
#endif
}

void WMATH_SUB(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
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
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_b = _mm_loadu_ps(&b.m[12]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[12], vec_res);
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
    vec_a = vld1q_f32(&a.m[12]);
    vec_b = vld1q_f32(&b.m[12]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[12], vec_res);
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
    vec_a = wasm_v128_load(&a.m[12]);
    vec_b = wasm_v128_load(&b.m[12]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b, vec_res;
    float* a_ptr = (float*)&a;
    float* b_ptr = (float*)&b;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvli(4, RV32, e32, m1);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;
    vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    vec_b = (__m128)__lsx_vld(&b.m[0], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_b = (__m128)__lsx_vld(&b.m[4], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_b = (__m128)__lsx_vld(&b.m[8], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_b = (__m128)__lsx_vld(&b.m[12], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] - b.m[i];
    }
#endif
}

void WMATH_MULTIPLY_SCALAR(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const float b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_b_scalar = _mm_set1_ps(b);
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[0], vec_res);
    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[4], vec_res);
    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[8], vec_res);
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b_scalar, vec_res;
    vec_b_scalar = vdupq_n_f32(b);
    vec_a = vld1q_f32(&a.m[0]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[0], vec_res);
    vec_a = vld1q_f32(&a.m[4]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[4], vec_res);
    vec_a = vld1q_f32(&a.m[8]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[8], vec_res);
    vec_a = vld1q_f32(&a.m[12]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_b_scalar = wasm_f32x4_splat(b);
    v128_t vec_a = wasm_v128_load(&a.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[0], vec_res);
    vec_a = wasm_v128_load(&a.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[4], vec_res);
    vec_a = wasm_v128_load(&a.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[8], vec_res);
    vec_a = wasm_v128_load(&a.m[12]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b_scalar, vec_res;
    float* a_ptr = (float*)&a;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvli(4, RV32, e32, m1);
    vec_b_scalar = __riscv_vfmv_v_f_f32m1(b, vl);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_b_scalar = __lsx_vldrepl_w(&b, 0);
    __m128 vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    __m128 vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] * b;
    }
#endif
}

void WMATH_MULTIPLY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 b_col0 = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
    __m128 b_col1 = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
    __m128 b_col2 = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
    __m128 b_col3 = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);

    for (int i = 0; i < 4; i++) {
        __m128 a_row = _mm_loadu_ps(&a.m[i * 4]);
        __m128 temp = _mm_mul_ps(a_row, b_col0);
        temp = _mm_hadd_ps(temp, temp);
        temp = _mm_hadd_ps(temp, temp);
        dst->m[i * 4 + 0] = _mm_cvtss_f32(temp);

        temp = _mm_mul_ps(a_row, b_col1);
        temp = _mm_hadd_ps(temp, temp);
        temp = _mm_hadd_ps(temp, temp);
        dst->m[i * 4 + 1] = _mm_cvtss_f32(temp);

        temp = _mm_mul_ps(a_row, b_col2);
        temp = _mm_hadd_ps(temp, temp);
        temp = _mm_hadd_ps(temp, temp);
        dst->m[i * 4 + 2] = _mm_cvtss_f32(temp);

        temp = _mm_mul_ps(a_row, b_col3);
        temp = _mm_hadd_ps(temp, temp);
        temp = _mm_hadd_ps(temp, temp);
        dst->m[i * 4 + 3] = _mm_cvtss_f32(temp);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t row, col, prod, sum;
    float32x4_t col0 = {b.m[0], b.m[4], b.m[8], b.m[12]};
    float32x4_t col1 = {b.m[1], b.m[5], b.m[9], b.m[13]};
    float32x4_t col2 = {b.m[2], b.m[6], b.m[10], b.m[14]};
    float32x4_t col3 = {b.m[3], b.m[7], b.m[11], b.m[15]};

    for (int i = 0; i < 4; i++) {
        row = vld1q_f32(&a.m[i * 4]);
        prod = vmulq_f32(row, col0);
        sum = vpaddq_f32(prod, prod);
        sum = vpaddq_f32(sum, sum);
        dst->m[i * 4 + 0] = vgetq_lane_f32(sum, 0);

        prod = vmulq_f32(row, col1);
        sum = vpaddq_f32(prod, prod);
        sum = vpaddq_f32(sum, sum);
        dst->m[i * 4 + 1] = vgetq_lane_f32(sum, 0);

        prod = vmulq_f32(row, col2);
        sum = vpaddq_f32(prod, prod);
        sum = vpaddq_f32(sum, sum);
        dst->m[i * 4 + 2] = vgetq_lane_f32(sum, 0);

        prod = vmulq_f32(row, col3);
        sum = vpaddq_f32(prod, prod);
        sum = vpaddq_f32(sum, sum);
        dst->m[i * 4 + 3] = vgetq_lane_f32(sum, 0);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t b_col0 = wasm_f32x4_make(b.m[0], b.m[4], b.m[8], b.m[12]);
    v128_t b_col1 = wasm_f32x4_make(b.m[1], b.m[5], b.m[9], b.m[13]);
    v128_t b_col2 = wasm_f32x4_make(b.m[2], b.m[6], b.m[10], b.m[14]);
    v128_t b_col3 = wasm_f32x4_make(b.m[3], b.m[7], b.m[11], b.m[15]);

    for (int i = 0; i < 4; i++) {
        v128_t a_row = wasm_v128_load(&a.m[i * 4]);
        v128_t temp = wasm_f32x4_mul(a_row, b_col0);
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
        dst->m[i * 4 + 0] = wasm_f32x4_extract_lane(temp, 0);

        temp = wasm_f32x4_mul(a_row, b_col1);
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
        dst->m[i * 4 + 1] = wasm_f32x4_extract_lane(temp, 0);

        temp = wasm_f32x4_mul(a_row, b_col2);
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
        dst->m[i * 4 + 2] = wasm_f32x4_extract_lane(temp, 0);

        temp = wasm_f32x4_mul(a_row, b_col3);
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 1, 1, 1, 1));
        temp = wasm_f32x4_add(temp, wasm_i32x4_shuffle(temp, temp, 2, 2, 2, 2));
        dst->m[i * 4 + 3] = wasm_f32x4_extract_lane(temp, 0);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl = __riscv_vsetvli(4, RV32, e32, m1);
    vfloat32m1_t a_row, b_col, temp_vec, sum_vec;

    for (int i = 0; i < 4; i++) {
        a_row = __riscv_vle32_v_f32m1(&a.m[i * 4], vl);

        b_col =
            __riscv_vcreate_v_f32m1(b.m[0 + 4 * 0], b.m[1 + 4 * 0], b.m[2 + 4 * 0], b.m[3 + 4 * 0]);
        temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
        sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        dst->m[i * 4 + 0] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

        b_col =
            __riscv_vcreate_v_f32m1(b.m[0 + 4 * 1], b.m[1 + 4 * 1], b.m[2 + 4 * 1], b.m[3 + 4 * 1]);
        temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
        sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        dst->m[i * 4 + 1] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

        b_col =
            __riscv_vcreate_v_f32m1(b.m[0 + 4 * 2], b.m[1 + 4 * 2], b.m[2 + 4 * 2], b.m[3 + 4 * 2]);
        temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
        sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        dst->m[i * 4 + 2] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

        b_col =
            __riscv_vcreate_v_f32m1(b.m[0 + 4 * 3], b.m[1 + 4 * 3], b.m[2 + 4 * 3], b.m[3 + 4 * 3]);
        temp_vec = __riscv_vfmul_vv_f32m1(a_row, b_col, vl);
        sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(temp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        dst->m[i * 4 + 3] = __riscv_vfmv_f_s_f32m1_f32(sum_vec);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 a_row, b_col, temp_vec;
    float dot_result;

    for (int i = 0; i < 4; i++) {
        a_row = (__m128)__lsx_vld(&a.m[i * 4], 0);

        b_col = __lsx_vldrepl_w(&b.m[0 + 4 * 0], 0);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4 * 0], 1);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4 * 0], 2);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4 * 0], 3);
        temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
        dot_result = __lsx_vfmov_s_f(temp_vec);
        dot_result += __lsx_vextract_float(temp_vec, 1);
        dot_result += __lsx_vextract_float(temp_vec, 2);
        dot_result += __lsx_vextract_float(temp_vec, 3);
        dst->m[i * 4 + 0] = dot_result;

        b_col = __lsx_vldrepl_w(&b.m[0 + 4 * 1], 0);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4 * 1], 1);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4 * 1], 2);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4 * 1], 3);
        temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
        dot_result = __lsx_vfmov_s_f(temp_vec);
        dot_result += __lsx_vextract_float(temp_vec, 1);
        dot_result += __lsx_vextract_float(temp_vec, 2);
        dot_result += __lsx_vextract_float(temp_vec, 3);
        dst->m[i * 4 + 1] = dot_result;

        b_col = __lsx_vldrepl_w(&b.m[0 + 4 * 2], 0);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4 * 2], 1);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4 * 2], 2);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4 * 2], 3);
        temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
        dot_result = __lsx_vfmov_s_f(temp_vec);
        dot_result += __lsx_vextract_float(temp_vec, 1);
        dot_result += __lsx_vextract_float(temp_vec, 2);
        dot_result += __lsx_vextract_float(temp_vec, 3);
        dst->m[i * 4 + 2] = dot_result;

        b_col = __lsx_vldrepl_w(&b.m[0 + 4 * 3], 0);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[1 + 4 * 3], 1);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[2 + 4 * 3], 2);
        b_col = __lsx_vinsgr2vr_w(b_col, b.m[3 + 4 * 3], 3);
        temp_vec = (__m128)__lsx_vfmul_s((__m128i)a_row, (__m128i)b_col);
        dot_result = __lsx_vfmov_s_f(temp_vec);
        dot_result += __lsx_vextract_float(temp_vec, 1);
        dot_result += __lsx_vextract_float(temp_vec, 2);
        dot_result += __lsx_vextract_float(temp_vec, 3);
        dst->m[i * 4 + 3] = dot_result;
    }

#else
    dst->m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[4] + a.m[2] * b.m[8] + a.m[3] * b.m[12];
    dst->m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[5] + a.m[2] * b.m[9] + a.m[3] * b.m[13];
    dst->m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[6] + a.m[2] * b.m[10] + a.m[3] * b.m[14];
    dst->m[3] = a.m[0] * b.m[3] + a.m[1] * b.m[7] + a.m[2] * b.m[11] + a.m[3] * b.m[15];

    dst->m[4] = a.m[4] * b.m[0] + a.m[5] * b.m[4] + a.m[6] * b.m[8] + a.m[7] * b.m[12];
    dst->m[5] = a.m[4] * b.m[1] + a.m[5] * b.m[5] + a.m[6] * b.m[9] + a.m[7] * b.m[13];
    dst->m[6] = a.m[4] * b.m[2] + a.m[5] * b.m[6] + a.m[6] * b.m[10] + a.m[7] * b.m[14];
    dst->m[7] = a.m[4] * b.m[3] + a.m[5] * b.m[7] + a.m[6] * b.m[11] + a.m[7] * b.m[15];

    dst->m[8] = a.m[8] * b.m[0] + a.m[9] * b.m[4] + a.m[10] * b.m[8] + a.m[11] * b.m[12];
    dst->m[9] = a.m[8] * b.m[1] + a.m[9] * b.m[5] + a.m[10] * b.m[9] + a.m[11] * b.m[13];
    dst->m[10] = a.m[8] * b.m[2] + a.m[9] * b.m[6] + a.m[10] * b.m[10] + a.m[11] * b.m[14];
    dst->m[11] = a.m[8] * b.m[3] + a.m[9] * b.m[7] + a.m[10] * b.m[11] + a.m[11] * b.m[15];

    dst->m[12] = a.m[12] * b.m[0] + a.m[13] * b.m[4] + a.m[14] * b.m[8] + a.m[15] * b.m[12];
    dst->m[13] = a.m[12] * b.m[1] + a.m[13] * b.m[5] + a.m[14] * b.m[9] + a.m[15] * b.m[13];
    dst->m[14] = a.m[12] * b.m[2] + a.m[13] * b.m[6] + a.m[14] * b.m[10] + a.m[15] * b.m[14];
    dst->m[15] = a.m[12] * b.m[3] + a.m[13] * b.m[7] + a.m[14] * b.m[11] + a.m[15] * b.m[15];
#endif
}

void WMATH_INVERSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    _mm_loadu_ps(&a.m[0]);
    _mm_loadu_ps(&a.m[4]);
    _mm_loadu_ps(&a.m[8]);
    _mm_loadu_ps(&a.m[12]);

    float m_00 = a.m[0], m_01 = a.m[1], m_02 = a.m[2], m_03 = a.m[3];
    float m_10 = a.m[4], m_11 = a.m[5], m_12 = a.m[6], m_13 = a.m[7];
    float m_20 = a.m[8], m_21 = a.m[9], m_22 = a.m[10], m_23 = a.m[11];
    float m_30 = a.m[12], m_31 = a.m[13], m_32 = a.m[14], m_33 = a.m[15];

    __m128 vec_m22 = _mm_set1_ps(m_22);
    __m128 vec_m33 = _mm_set1_ps(m_33);
    __m128 vec_m32 = _mm_set1_ps(m_32);
    __m128 vec_m23 = _mm_set1_ps(m_23);
    __m128 vec_m12 = _mm_set1_ps(m_12);
    __m128 vec_m13 = _mm_set1_ps(m_13);

    __m128 tmp_0 = _mm_mul_ps(vec_m22, vec_m33);
    __m128 tmp_1 = _mm_mul_ps(vec_m32, vec_m23);
    __m128 tmp_2 = _mm_mul_ps(vec_m12, vec_m33);
    __m128 tmp_3 = _mm_mul_ps(vec_m32, vec_m13);

    float tmp_0_val = _mm_cvtss_f32(tmp_0);
    float tmp_1_val = _mm_cvtss_f32(tmp_1);
    float tmp_2_val = _mm_cvtss_f32(tmp_2);
    float tmp_3_val = _mm_cvtss_f32(tmp_3);

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

    float t_0 = (tmp_0_val * m_11 + tmp_3_val * m_21 + tmp_4 * m_31) -
                (tmp_1_val * m_11 + tmp_2_val * m_21 + tmp_5 * m_31);
    float t_1 = (tmp_1_val * m_01 + tmp_6 * m_21 + tmp_9 * m_31) -
                (tmp_0_val * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
    float t_2 = (tmp_2_val * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
                (tmp_3_val * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
    float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
                (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

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

    __m128 vec_d = _mm_set1_ps(d);

    __m128 m_00_inv = _mm_mul_ps(vec_d, vec_t0);
    __m128 m_01_inv = _mm_mul_ps(vec_d, vec_t1);
    __m128 m_02_inv = _mm_mul_ps(vec_d, vec_t2);
    __m128 m_03_inv = _mm_mul_ps(vec_d, vec_t3);

    float m_00_val = _mm_cvtss_f32(m_00_inv);
    float m_01_val = _mm_cvtss_f32(m_01_inv);
    float m_02_val = _mm_cvtss_f32(m_02_inv);
    float m_03_val = _mm_cvtss_f32(m_03_inv);

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

    WMATH_CREATE(Mat4)(dst, (WMATH_CREATE_TYPE(Mat4)){.m_00 = m_00_val,
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

#else
    float m_00 = a.m[0 * 4 + 0], m_01 = a.m[0 * 4 + 1], m_02 = a.m[0 * 4 + 2],
          m_03 = a.m[0 * 4 + 3];
    float m_10 = a.m[1 * 4 + 0], m_11 = a.m[1 * 4 + 1], m_12 = a.m[1 * 4 + 2],
          m_13 = a.m[1 * 4 + 3];
    float m_20 = a.m[2 * 4 + 0], m_21 = a.m[2 * 4 + 1], m_22 = a.m[2 * 4 + 2],
          m_23 = a.m[2 * 4 + 3];
    float m_30 = a.m[3 * 4 + 0], m_31 = a.m[3 * 4 + 1], m_32 = a.m[3 * 4 + 2],
          m_33 = a.m[3 * 4 + 3];

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

    float t_0 =
        (tmp_0 * m_11 + tmp_3 * m_21 + tmp_4 * m_31) - (tmp_1 * m_11 + tmp_2 * m_21 + tmp_5 * m_31);
    float t_1 =
        (tmp_1 * m_01 + tmp_6 * m_21 + tmp_9 * m_31) - (tmp_0 * m_01 + tmp_7 * m_21 + tmp_8 * m_31);
    float t_2 = (tmp_2 * m_01 + tmp_7 * m_11 + tmp_10 * m_31) -
                (tmp_3 * m_01 + tmp_6 * m_11 + tmp_11 * m_31);
    float t_3 = (tmp_5 * m_01 + tmp_8 * m_11 + tmp_11 * m_21) -
                (tmp_4 * m_01 + tmp_9 * m_11 + tmp_10 * m_21);

    float d = 1.0f / (m_00 * t_0 + m_10 * t_1 + m_20 * t_2 + m_30 * t_3);

    WMATH_CREATE(Mat4)(dst, (WMATH_CREATE_TYPE(Mat4)){
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
}

void WMATH_TRANSPOSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 row0 = _mm_loadu_ps(&a.m[0]);
    __m128 row1 = _mm_loadu_ps(&a.m[4]);
    __m128 row2 = _mm_loadu_ps(&a.m[8]);
    __m128 row3 = _mm_loadu_ps(&a.m[12]);

    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp1 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp2 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);

    __m128 col0 = _mm_movelh_ps(tmp0, tmp1);
    __m128 col1 = _mm_movehl_ps(tmp1, tmp0);
    __m128 col2 = _mm_movelh_ps(tmp2, tmp3);
    __m128 col3 = _mm_movehl_ps(tmp3, tmp2);

    _mm_storeu_ps(&dst->m[0], col0);
    _mm_storeu_ps(&dst->m[4], col1);
    _mm_storeu_ps(&dst->m[8], col2);
    _mm_storeu_ps(&dst->m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t row0 = vld1q_f32(&a.m[0]);
    float32x4_t row1 = vld1q_f32(&a.m[4]);
    float32x4_t row2 = vld1q_f32(&a.m[8]);
    float32x4_t row3 = vld1q_f32(&a.m[12]);

    float32x4x2_t t01 = vtrnq_f32(row0, row1);
    float32x4x2_t t23 = vtrnq_f32(row2, row3);

    float32x4_t col0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    float32x4_t col1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    float32x4_t col2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    float32x4_t col3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

    vst1q_f32(&dst->m[0], col0);
    vst1q_f32(&dst->m[4], col1);
    vst1q_f32(&dst->m[8], col2);
    vst1q_f32(&dst->m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t row0 = wasm_v128_load(&a.m[0]);
    v128_t row1 = wasm_v128_load(&a.m[4]);
    v128_t row2 = wasm_v128_load(&a.m[8]);
    v128_t row3 = wasm_v128_load(&a.m[12]);

    v128_t tmp0 = wasm_i32x4_shuffle(row0, row1, 0, 1, 0, 1);
    v128_t tmp1 = wasm_i32x4_shuffle(row2, row3, 0, 1, 0, 1);
    v128_t tmp2 = wasm_i32x4_shuffle(row0, row1, 2, 3, 2, 3);
    v128_t tmp3 = wasm_i32x4_shuffle(row2, row3, 2, 3, 2, 3);

    v128_t col0 = wasm_i32x4_shuffle(tmp0, tmp1, 0, 2, 0, 2);
    v128_t col1 = wasm_i32x4_shuffle(tmp0, tmp1, 1, 3, 1, 3);
    v128_t col2 = wasm_i32x4_shuffle(tmp2, tmp3, 0, 2, 0, 2);
    v128_t col3 = wasm_i32x4_shuffle(tmp2, tmp3, 1, 3, 1, 3);

    wasm_v128_store(&dst->m[0], col0);
    wasm_v128_store(&dst->m[4], col1);
    wasm_v128_store(&dst->m[8], col2);
    wasm_v128_store(&dst->m[12], col3);

#else
    dst->m[0] = a.m[0];
    dst->m[1] = a.m[4];
    dst->m[2] = a.m[8];
    dst->m[3] = a.m[12];
    dst->m[4] = a.m[1];
    dst->m[5] = a.m[5];
    dst->m[6] = a.m[9];
    dst->m[7] = a.m[13];
    dst->m[8] = a.m[2];
    dst->m[9] = a.m[6];
    dst->m[10] = a.m[10];
    dst->m[11] = a.m[14];
    dst->m[12] = a.m[3];
    dst->m[13] = a.m[7];
    dst->m[14] = a.m[11];
    dst->m[15] = a.m[15];
#endif
}

float WMATH_DETERMINANT(Mat4)(const WMATH_TYPE(Mat4) m) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    _mm_loadu_ps(&m.m[0]);
    _mm_loadu_ps(&m.m[4]);
    _mm_loadu_ps(&m.m[8]);
    _mm_loadu_ps(&m.m[12]);

    float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
    float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
    float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
    float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

    __m128 vec_m22 = _mm_set1_ps(m22);
    __m128 vec_m33 = _mm_set1_ps(m33);
    __m128 vec_m32 = _mm_set1_ps(m32);
    __m128 vec_m23 = _mm_set1_ps(m23);
    __m128 vec_m12 = _mm_set1_ps(m12);
    __m128 vec_m13 = _mm_set1_ps(m13);
    __m128 vec_m02 = _mm_set1_ps(m02);
    __m128 vec_m03 = _mm_set1_ps(m03);

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

    float t0 = (tmp0_val * m11 + tmp3_val * m21 + tmp4_val * m31) -
               (tmp1_val * m11 + tmp2_val * m21 + tmp5_val * m31);
    float t1 = (tmp1_val * m01 + tmp6_val * m21 + tmp9_val * m31) -
               (tmp0_val * m01 + tmp7_val * m21 + tmp8_val * m31);
    float t2 = (tmp2_val * m01 + tmp7_val * m11 + tmp10_val * m31) -
               (tmp3_val * m01 + tmp6_val * m11 + tmp11_val * m31);
    float t3 = (tmp5_val * m01 + tmp8_val * m11 + tmp11_val * m21) -
               (tmp4_val * m01 + tmp9_val * m11 + tmp10_val * m21);

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

#else
    float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
    float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
    float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
    float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

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

    float t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) - (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
    float t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) - (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
    float t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) - (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
    float t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) - (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);

    return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
#endif
}

void WMATH_CALL(Mat4, aim)(DST_MAT4, const WMATH_TYPE(Vec3) position, const WMATH_TYPE(Vec3) target,
                           const WMATH_TYPE(Vec3) up) {
    WMATH_TYPE(Vec3) z_axis;
    WMATH_TYPE(Vec3) x_axis;
    WMATH_TYPE(Vec3) y_axis;

    WMATH_SUB(Vec3)(&z_axis, target, position);
    WMATH_NORMALIZE(Vec3)(&z_axis, z_axis);

    WMATH_CALL(Vec3, cross)(&x_axis, up, z_axis);
    WMATH_NORMALIZE(Vec3)(&x_axis, x_axis);

    WMATH_CALL(Vec3, cross)(&y_axis, z_axis, x_axis);

    WMATH_SET(Mat4)(dst, x_axis.v[0], x_axis.v[1], x_axis.v[2], 0.0f, y_axis.v[0], y_axis.v[1],
                    y_axis.v[2], 0.0f, z_axis.v[0], z_axis.v[1], z_axis.v[2], 0.0f, position.v[0],
                    position.v[1], position.v[2], 1.0f);
}

void WMATH_CALL(Mat4, look_at)(DST_MAT4, const WMATH_TYPE(Vec3) eye, const WMATH_TYPE(Vec3) target,
                               const WMATH_TYPE(Vec3) up) {
    WMATH_TYPE(Vec3) z_axis;
    WMATH_TYPE(Vec3) x_axis;
    WMATH_TYPE(Vec3) y_axis;

    WMATH_SUB(Vec3)(&z_axis, eye, target);
    WMATH_NORMALIZE(Vec3)(&z_axis, z_axis);

    WMATH_CALL(Vec3, cross)(&x_axis, up, z_axis);
    WMATH_NORMALIZE(Vec3)(&x_axis, x_axis);

    WMATH_CALL(Vec3, cross)(&y_axis, z_axis, x_axis);

    WMATH_SET(Mat4)(dst, x_axis.v[0], x_axis.v[1], x_axis.v[2], -WMATH_DOT(Vec3)(x_axis, eye),
                    y_axis.v[0], y_axis.v[1], y_axis.v[2], -WMATH_DOT(Vec3)(y_axis, eye),
                    z_axis.v[0], z_axis.v[1], z_axis.v[2], -WMATH_DOT(Vec3)(z_axis, eye), 0.0f,
                    0.0f, 0.0f, 1.0f);
}

// END Mat4
