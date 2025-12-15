#ifndef WCN_MATH_INTERNAL_H
#define WCN_MATH_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif
#include "WCN/WCN_PLATFORM_MACROS.h"

// ========================================================================

#include "WCN/WCN_Math.h"
#include <stdbool.h>

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
#include <emmintrin.h>
#include <immintrin.h> // For additional SSE/AVX intrinsics
#include <smmintrin.h> // For SSE 4.1, which has better float operations

// Check for AVX/AVX2 support at compile time
#if defined(__AVX2__)
#define WCN_HAS_AVX2 1
#elif defined(__AVX__)
#define WCN_HAS_AVX 1
#endif

// SIMD helper functions for x86
static inline __m128 wcn_load_vec2_partial(const float *v) {
  return _mm_set_ps(0.0f, 0.0f, v[1], v[0]);
}

// Cross product helper function for x86 SSE
static inline __m128 wcn_cross_ps(__m128 a, __m128 b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y -
  // a.y*b.x)
  const __m128 a_yzx =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 2, 1)); // a.y, a.z, a.x, a.y
  const __m128 b_zxy =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 0, 2)); // b.z, b.x, b.y, b.z
  const __m128 a_zxy =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 2)); // a.z, a.x, a.y, a.z
  const __m128 b_yzx =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 2, 1)); // b.y, b.z, b.x, b.y

  const __m128 mul1 = _mm_mul_ps(a_yzx, b_zxy);
  const __m128 mul2 = _mm_mul_ps(a_zxy, b_yzx);
  return _mm_sub_ps(mul1, mul2);
}

static inline void wcn_store_vec2_partial(float *v, const __m128 vec) {
  float temp[4];
  _mm_storeu_ps(temp, vec);
  v[0] = temp[0];
  v[1] = temp[1];
}

static inline __m128 wcn_load_vec3_partial(const float *v) {
  return _mm_set_ps(0.0f, v[2], v[1], v[0]);
}

static inline void wcn_store_vec3_partial(float *v, const __m128 vec) {
  float temp[4];
  _mm_storeu_ps(temp, vec);
  v[0] = temp[0];
  v[1] = temp[1];
  v[2] = temp[2];
}

static inline __m128 wcn_hadd_ps(const __m128 vec) {
  const __m128 temp = _mm_hadd_ps(vec, vec);
  return _mm_hadd_ps(temp, temp);
}

// Matrix helper functions
static inline __m128 wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, const int row) {
  return _mm_loadu_ps(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, const int row,
                                    const __m128 vec) {
  _mm_storeu_ps(&mat->m[row * 4], vec);
}

static inline __m128 wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, const int row) {
  return _mm_loadu_ps(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, const int row,
                                    const __m128 vec) {
  _mm_storeu_ps(&mat->m[row * 4], vec);
}

static inline __m128 wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, const int col) {
  return _mm_set_ps(mat->m[col + 12], mat->m[col + 8], mat->m[col + 4],
                    mat->m[col]);
}

// AVX/AVX2 helper functions
#if defined(WCN_HAS_AVX) || defined(WCN_HAS_AVX2)

#endif

// FMA helper functions
#if defined(WCN_HAS_FMA)

// FMA-optimized vector multiply-add: a * b + c
static inline __m128 wcn_fma_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fmadd_ps(a, b, c);
}

// FMA-optimized vector multiply-sub: a * b - c
static inline __m128 wcn_fma_mul_sub_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fmsub_ps(a, b, c);
}

// FMA-optimized vector negate-multiply-add: -(a * b) + c
static inline __m128 wcn_fma_neg_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fnmadd_ps(a, b, c);
}

// AVX FMA versions
#if defined(WCN_HAS_AVX) || defined(WCN_HAS_AVX2)

// FMA-optimized AVX vector multiply-add: a * b + c
static inline __m256 wcn_avx_fma_mul_add_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}

// FMA-optimized AVX vector multiply-sub: a * b - c
static inline __m256 wcn_avx_fma_mul_sub_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fmsub_ps(a, b, c);
}

// FMA-optimized AVX vector negate-multiply-add: -(a * b) + c
static inline __m256 wcn_avx_fma_neg_mul_add_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fnmadd_ps(a, b, c);
}

#endif

#endif

// AVX2 specific helper functions
#if defined(WCN_HAS_AVX2)

#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
#include <arm_neon.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// Helper: store a float32x4_t into a 4-element array and copy first 3 elements
static inline void wcn_neon_store_vec3_to_array(float32x4_t v, float out[3]) {
  float tmp[4];
  vst1q_f32(tmp, v);
  out[0] = tmp[0];
  out[1] = tmp[1];
  out[2] = tmp[2];
}
#endif

// SIMD helper functions for ARM NEON
static inline float32x4_t wcn_load_vec2_partial(const float *v) {
  return (float32x4_t){v[0], v[1], 0.0f, 0.0f};
}

// NEON cross product helper for 3D vectors
static inline float32x4_t wcn_cross_neon(float32x4_t a, float32x4_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y -
  // a.y*b.x)
  float32x4_t a_yzx = (float32x4_t){vgetq_lane_f32(a, 1), vgetq_lane_f32(a, 2),
                                    vgetq_lane_f32(a, 0), 0.0f};
  float32x4_t b_zxy = (float32x4_t){vgetq_lane_f32(b, 2), vgetq_lane_f32(b, 0),
                                    vgetq_lane_f32(b, 1), 0.0f};
  float32x4_t a_zxy = (float32x4_t){vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 0),
                                    vgetq_lane_f32(a, 1), 0.0f};
  float32x4_t b_yzx = (float32x4_t){vgetq_lane_f32(b, 1), vgetq_lane_f32(b, 2),
                                    vgetq_lane_f32(b, 0), 0.0f};

  float32x4_t mul1 = vmulq_f32(a_yzx, b_zxy);
  float32x4_t mul2 = vmulq_f32(a_zxy, b_yzx);
  return vsubq_f32(mul1, mul2);
}

static inline void wcn_store_vec2_partial(float *v, float32x4_t vec) {
  v[0] = vgetq_lane_f32(vec, 0);
  v[1] = vgetq_lane_f32(vec, 1);
}

static inline float32x4_t wcn_load_vec3_partial(const float *v) {
  return (float32x4_t){v[0], v[1], v[2], 0.0f};
}

static inline void wcn_store_vec3_partial(float *v, float32x4_t vec) {
  v[0] = vgetq_lane_f32(vec, 0);
  v[1] = vgetq_lane_f32(vec, 1);
  v[2] = vgetq_lane_f32(vec, 2);
}

static inline float wcn_hadd_f32(float32x4_t vec) {
  float32x2_t low = vget_low_f32(vec);
  float32x2_t high = vget_high_f32(vec);
  float32x2_t sum = vadd_f32(low, high);
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}

// Matrix helper functions
static inline float32x4_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat,
                                           int row) {
  return vld1q_f32(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row,
                                    float32x4_t vec) {
  vst1q_f32(&mat->m[row * 4], vec);
}

static inline float32x4_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat,
                                           int row) {
  return vld1q_f32(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row,
                                    float32x4_t vec) {
  vst1q_f32(&mat->m[row * 4], vec);
}

static inline float32x4_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat,
                                           int col) {
  return (float32x4_t){mat->m[col], mat->m[col + 4], mat->m[col + 8],
                       mat->m[col + 12]};
}

#endif

// Platform-specific SIMD implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
// RISC-V Vector Extension
#include <riscv_vector.h>
// Helper functions for RISC-V
static inline vfloat32m1_t wcn_load_vec2_partial(const float *v) {
  float temp[4] = {v[0], v[1], 0.0f, 0.0f};
  return __riscv_vle32_v_f32m1(temp, 4);
}

static inline void wcn_store_vec2_partial(float *v, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(v, vec, 2); // Only store first 2 elements
}

static inline vfloat32m1_t wcn_load_vec3_partial(const float *v) {
  float temp[4] = {v[0], v[1], v[2], 0.0f};
  return __riscv_vle32_v_f32m1(temp, 4);
}

static inline void wcn_store_vec3_partial(float *v, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(v, vec, 3); // Only store first 3 elements
}

static inline float wcn_hadd_f32(vfloat32m1_t vec) {
  // Perform horizontal sum of vector elements
  vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(0.0f, 4); // broadcast 0.0f to all elements
  vfloat32m1_t sum = __riscv_vredsum_vs_f32m1_f32m1(vec, initial, 4);
  return __riscv_vfmv_f_s_f32m1(sum, 0);
}

// RISC-V cross product helper for 3D vectors
static inline vfloat32m1_t wcn_cross_rvv(vfloat32m1_t a, vfloat32m1_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  float a_vals[4] = {__riscv_vfmv_f_s_f32m1(a, 1), __riscv_vfmv_f_s_f32m1(a, 2),
                     __riscv_vfmv_f_s_f32m1(a, 0), 0.0f};
  vfloat32m1_t a_yzx = __riscv_vle32_v_f32m1(a_vals, 4);

  float b_vals[4] = {__riscv_vfmv_f_s_f32m1(b, 2), __riscv_vfmv_f_s_f32m1(b, 0),
                     __riscv_vfmv_f_s_f32m1(b, 1), 0.0f};
  vfloat32m1_t b_zxy = __riscv_vle32_v_f32m1(b_vals, 4);

  float a_vals2[4] = {__riscv_vfmv_f_s_f32m1(a, 2), __riscv_vfmv_f_s_f32m1(a, 0),
                      __riscv_vfmv_f_s_f32m1(a, 1), 0.0f};
  vfloat32m1_t a_zxy = __riscv_vle32_v_f32m1(a_vals2, 4);

  float b_vals2[4] = {__riscv_vfmv_f_s_f32m1(b, 1), __riscv_vfmv_f_s_f32m1(b, 2),
                      __riscv_vfmv_f_s_f32m1(b, 0), 0.0f};
  vfloat32m1_t b_yzx = __riscv_vle32_v_f32m1(b_vals2, 4);

  vfloat32m1_t mul1 = __riscv_vfmul_vv_f32m1(a_yzx, b_zxy, 4);
  vfloat32m1_t mul2 = __riscv_vfmul_vv_f32m1(a_zxy, b_yzx, 4);
  return __riscv_vfsub_vv_f32m1(mul1, mul2, 4);
}

// Matrix helper functions for RISC-V
static inline vfloat32m1_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return __riscv_vle32_v_f32m1(&mat->m[row * 4], 4);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(&mat->m[row * 4], vec, 4);
}

static inline vfloat32m1_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return __riscv_vle32_v_f32m1(&mat->m[row * 4], 4);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(&mat->m[row * 4], vec, 4);
}

static inline vfloat32m1_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  float temp[4] = {mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]};
  return __riscv_vle32_v_f32m1(temp, 4);
}

#endif

// WebAssembly SIMD
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
#include <wasm_simd128.h>

// Helper functions for WebAssembly SIMD
static inline v128_t wcn_load_vec2_partial(const float *v) {
  // Vec2 只有 8 字节 (2 floats)，直接读取避免越界
  return wasm_f32x4_make(v[0], v[1], 0.0f, 0.0f);
}

static inline void wcn_store_vec2_partial(float *v, v128_t vec) {
  v[0] = wasm_f32x4_extract_lane(vec, 0);
  v[1] = wasm_f32x4_extract_lane(vec, 1);
}

static inline v128_t wcn_load_vec3_partial(const float *v) {
  return wasm_f32x4_make(v[0], v[1], v[2], 0.0f);
}

static inline void wcn_store_vec3_partial(float *v, v128_t vec) {
  v[0] = wasm_f32x4_extract_lane(vec, 0);
  v[1] = wasm_f32x4_extract_lane(vec, 1);
  v[2] = wasm_f32x4_extract_lane(vec, 2);
}

static inline float wcn_hadd_f32(v128_t vec) {
  // Perform horizontal sum using WASM intrinsics
  v128_t shuf = wasm_i32x4_shuffle(vec, vec, 2, 3, 0, 1);
  v128_t sums = wasm_f32x4_add(vec, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  return wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);
}

// WASM cross product helper for 3D vectors
static inline v128_t wcn_cross_wasm(v128_t a, v128_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  v128_t a_yzx = wasm_i32x4_shuffle(a, a, 1, 2, 0, 3);
  v128_t b_zxy = wasm_i32x4_shuffle(b, b, 2, 0, 1, 3);
  v128_t a_zxy = wasm_i32x4_shuffle(a, a, 2, 0, 1, 3);
  v128_t b_yzx = wasm_i32x4_shuffle(b, b, 1, 2, 0, 3);

  v128_t mul1 = wasm_f32x4_mul(a_yzx, b_zxy);
  v128_t mul2 = wasm_f32x4_mul(a_zxy, b_yzx);
  return wasm_f32x4_sub(mul1, mul2);
}

// Matrix helper functions for WASM
static inline v128_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return wasm_v128_load(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, v128_t vec) {
  wasm_v128_store(&mat->m[row * 4], vec);
}

static inline v128_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return wasm_v128_load(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, v128_t vec) {
  wasm_v128_store(&mat->m[row * 4], vec);
}

static inline v128_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  return wasm_f32x4_make(mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]);
}

#endif

// LoongArch SIMD
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
#include <lsxintrin.h>

// Helper functions for LoongArch LSX
static inline __m128 wcn_load_vec2_partial(const float *v) {
  __m128 temp = __lsx_vldrepl_w(v, 0);
  temp = __lsx_vinsgr2vr_w(temp, v[1], 1);
  return temp;
}

static inline void wcn_store_vec2_partial(float *v, __m128 vec) {
  v[0] = __lsx_vextractw(vec, 0);
  v[1] = __lsx_vextractw(vec, 1);
}

static inline __m128 wcn_load_vec3_partial(const float *v) {
  __m128 temp = __lsx_vldrepl_w(v, 0);
  temp = __lsx_vinsgr2vr_w(temp, v[1], 1);
  temp = __lsx_vinsgr2vr_w(temp, v[2], 2);
  return temp;
}

static inline void wcn_store_vec3_partial(float *v, __m128 vec) {
  v[0] = __lsx_vextractw(vec, 0);
  v[1] = __lsx_vextractw(vec, 1);
  v[2] = __lsx_vextractw(vec, 2);
}

static inline float wcn_hadd_f32(__m128 vec) {
  __m128 shuf = __lsx_vshuf4i_w(vec, 0x1B);  // Shuffle: [2, 3, 0, 1]
  __m128 sums = __lsx_vfadd_s(vec, shuf);
  shuf = __lsx_vshuf4i_w(sums, 0x01);  // Shuffle: [1, 0, x, x]
  sums = __lsx_vfadd_s(sums, shuf);
  return __lsx_vextractw(sums, 0);
}

// LoongArch cross product helper for 3D vectors
static inline __m128 wcn_cross_lsx(__m128 a, __m128 b) {
  __m128 a_yzx = __lsx_vshuf4i_w(a, 0x69);  // Shuffle: [1, 2, 0, 3] -> 01101001 binary = 0x69
  __m128 b_zxy = __lsx_vshuf4i_w(b, 0x96);  // Shuffle: [2, 0, 1, 3] -> 10010110 binary = 0x96
  __m128 a_zxy = __lsx_vshuf4i_w(a, 0x96);  // Shuffle: [2, 0, 1, 3] -> 10010110 binary = 0x96
  __m128 b_yzx = __lsx_vshuf4i_w(b, 0x69);  // Shuffle: [1, 2, 0, 3] -> 01101001 binary = 0x69

  __m128 mul1 = __lsx_vfmul_s(a_yzx, b_zxy);
  __m128 mul2 = __lsx_vfmul_s(a_zxy, b_yzx);
  return __lsx_vfsub_s(mul1, mul2);
}

// Matrix helper functions for LoongArch
static inline __m128 wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return __lsx_vld(&mat->m[row * 4], 0);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, __m128 vec) {
  __lsx_vst(vec, &mat->m[row * 4], 0);
}

static inline __m128 wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return __lsx_vld(&mat->m[row * 4], 0);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, __m128 vec) {
  __lsx_vst(vec, &mat->m[row * 4], 0);
}

static inline __m128 wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  return (const __m128){mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]};
}

#endif

// Missing FMA functions for x86_64 SSE (when FMA not available)
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64 && !defined(WCN_HAS_FMA)
static inline __m128 wcn_fma_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_add_ps(_mm_mul_ps(a, b), c);
}

static inline __m128 wcn_fma_mul_sub_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_sub_ps(_mm_mul_ps(a, b), c);
}

static inline __m128 wcn_fma_neg_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_sub_ps(c, _mm_mul_ps(a, b));
}
#endif

// Missing cross product functions for RISC-V
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_cross_rvv(vfloat32m1_t a, vfloat32m1_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  float a_vals[4] = {__riscv_vfmv_f_s_f32m1(a, 1), __riscv_vfmv_f_s_f32m1(a, 2),
                     __riscv_vfmv_f_s_f32m1(a, 0), 0.0f};
  vfloat32m1_t a_yzx = __riscv_vle32_v_f32m1(a_vals, 4);

  float b_vals[4] = {__riscv_vfmv_f_s_f32m1(b, 2), __riscv_vfmv_f_s_f32m1(b, 0),
                     __riscv_vfmv_f_s_f32m1(b, 1), 0.0f};
  vfloat32m1_t b_zxy = __riscv_vle32_v_f32m1(b_vals, 4);

  float a_vals2[4] = {__riscv_vfmv_f_s_f32m1(a, 2), __riscv_vfmv_f_s_f32m1(a, 0),
                      __riscv_vfmv_f_s_f32m1(a, 1), 0.0f};
  vfloat32m1_t a_zxy = __riscv_vle32_v_f32m1(a_vals2, 4);

  float b_vals2[4] = {__riscv_vfmv_f_s_f32m1(b, 1), __riscv_vfmv_f_s_f32m1(b, 2),
                      __riscv_vfmv_f_s_f32m1(b, 0), 0.0f};
  vfloat32m1_t b_yzx = __riscv_vle32_v_f32m1(b_vals2, 4);

  vfloat32m1_t mul1 = __riscv_vfmul_vv_f32m1(a_yzx, b_zxy, 4);
  vfloat32m1_t mul2 = __riscv_vfmul_vv_f32m1(a_zxy, b_yzx, 4);
  return __riscv_vfsub_vv_f32m1(mul1, mul2, 4);
}
#endif

// SIMD branchless selection for SSE
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_select_ps(const __m128 condition_mask, const __m128 a, const __m128 b) {
  // Use bitwise operations to select between a and b without branching
  return _mm_or_ps(_mm_and_ps(condition_mask, a),
                   _mm_andnot_ps(condition_mask, b));
}

#endif

// SIMD-optimized fast inverse square root for SSE
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_fast_inv_sqrt_ps(const __m128 x) {
  // Use sqrt instruction if available (SSE and later)
  const __m128 approx = _mm_rsqrt_ps(x);

  // One Newton-Phonographs iteration to improve precision
  // y = approx * (1.5-0.5 * x * approx * approx)
  __m128 x2 = _mm_mul_ps(x, approx);
  x2 = _mm_mul_ps(x2, approx);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 one_point_five = _mm_set1_ps(1.5f);
  __m128 correction = _mm_sub_ps(one_point_five, _mm_mul_ps(half, x2));
  return _mm_mul_ps(approx, correction);
}

#endif

// Platform-specific inverse square root implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_fast_inv_sqrt_wasm(const v128_t x) {
  // WASM does not have a direct reciprocal square root approximation
  // Use standard division and square root for precision
  v128_t one = wasm_f32x4_splat(1.0f);
  return wasm_f32x4_div(one, wasm_f32x4_sqrt(x));
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_fast_inv_sqrt_lsx(const __m128 x) {
  // Use standard division and sqrt for precision with LoongArch
  __m128 one = __lsx_vldrepl_w((float[]){1.0f}, 0);
  __m128 sqrt_x = __lsx_vfsqrt_s(x);
  return __lsx_vfdiv_s(one, sqrt_x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_fast_inv_sqrt_rvv(const vfloat32m1_t x) {
  // Use standard division and sqrt for precision with RISC-V vector extension
  vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, 4);
  vfloat32m1_t sqrt_x = __riscv_vfsqrt_v_f32m1(x, 4); // assuming SEW=32, LMUL=1
  return __riscv_vfdiv_vv_f32m1(one, sqrt_x, 4);
}
#endif

// Unified cross product function that uses platform-specific implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_cross_platform(__m128 a, __m128 b) {
  return wcn_cross_ps(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
static inline float32x4_t wcn_cross_platform(float32x4_t a, float32x4_t b) {
  return wcn_cross_neon(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_cross_platform(vfloat32m1_t a, vfloat32m1_t b) {
  return wcn_cross_rvv(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_cross_platform(v128_t a, v128_t b) {
  return wcn_cross_wasm(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_cross_platform(__m128 a, __m128 b) {
  return wcn_cross_lsx(a, b);
}
#endif

// Unified fast inverse square root function that uses platform-specific implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_fast_inv_sqrt_platform(__m128 x) {
  return wcn_fast_inv_sqrt_ps(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
static inline float32x4_t wcn_fast_inv_sqrt_platform(float32x4_t x) {
  // Use NEON reciprocal square root estimate with Newton-Raphson refinement
  float32x4_t est = vrsqrteq_f32(x);
  // One Newton-Raphson iteration for better accuracy
  return vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(x, est), est));
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_fast_inv_sqrt_platform(vfloat32m1_t x) {
  return wcn_fast_inv_sqrt_rvv(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_fast_inv_sqrt_platform(v128_t x) {
  return wcn_fast_inv_sqrt_wasm(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_fast_inv_sqrt_platform(__m128 x) {
  return wcn_fast_inv_sqrt_lsx(x);
}
#endif

// Scalar fast inverse square root function
static inline float wcn_fast_inv_sqrt(const float x) {
  // Fast inverse square root using Quake's famous algorithm
  float x_half = 0.5f * x;
  int i = *(int*)&x;          // Evil bit-level hacking
  i = 0x5f3759df - (i >> 1);  // Magic number and bit shift
  float y = *(float*)&i;
  // One Newton-Raphson iteration to improve precision
  y = y * (1.5f - x_half * y * y);
  return y;
}

extern const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4];

extern float EPSILON;

extern bool EPSILON_IS_SET;

float wcn_math_set_epsilon(const float epsilon);

float wcn_math_get_epsilon();

// ========================================================================
#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_INTERNAL_H
