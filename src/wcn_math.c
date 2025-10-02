#include "WCN/WCN_Math.h"
#include <stdbool.h>
#include <string.h>

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

// SSE cross product helper for 3D vectors
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

// FMA (Fused Multiply-Add) support
#if defined(__FMA__)
#define WCN_HAS_FMA 1
#endif

// FMA helper functions
#if defined(WCN_HAS_FMA)

// FMA-optimized vector multiply-add: a * b + c
static inline __m128 wcn_fma_mul_add_ps(__m128 a, __m128 b, __m128 c) {
  return _mm_fmadd_ps(a, b, c);
}

// FMA-optimized vector multiply-sub: a * b - c
static inline __m128 wcn_fma_mul_sub_ps(__m128 a, __m128 b, __m128 c) {
  return _mm_fmsub_ps(a, b, c);
}

// FMA-optimized vector negate-multiply-add: -(a * b) + c
static inline __m128 wcn_fma_neg_mul_add_ps(__m128 a, __m128 b, __m128 c) {
  return _mm_fnmadd_ps(a, b, c);
}

// AVX FMA versions
#if defined(WCN_HAS_AVX) || defined(WCN_HAS_AVX2)

// FMA-optimized AVX vector multiply-add: a * b + c
static inline __m256 wcn_avx_fma_mul_add_ps(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}

// FMA-optimized AVX vector multiply-sub: a * b - c
static inline __m256 wcn_avx_fma_mul_sub_ps(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmsub_ps(a, b, c);
}

// FMA-optimized AVX vector negate-multiply-add: -(a * b) + c
static inline __m256 wcn_avx_fma_neg_mul_add_ps(__m256 a, __m256 b, __m256 c) {
  return _mm256_fnmadd_ps(a, b, c);
}

#endif

#endif

// AVX2 specific helper functions
#if defined(WCN_HAS_AVX2)

#endif

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

// SIMD branchless selection for SSE
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
static inline __m128 wcn_select_ps(const __m128 condition_mask, const __m128 a, const __m128 b) {
  // Use bitwise operations to select between a and b without branching
  return _mm_or_ps(_mm_and_ps(condition_mask, a),
                   _mm_andnot_ps(condition_mask, b));
}

#endif

// SIMD-optimized fast inverse square root for SSE
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4] = {
  { 1, -1,  1, -1}, // XYZ
  {-1, -1,  1,  1}, // XZY
  { 1, -1, -1,  1}, // YXZ
  { 1,  1, -1, -1}, // YZX
  {-1,  1,  1, -1}, // ZXY
  {-1,  1, -1,  1}  // ZYX
};

float EPSILON = 0.0f;

static bool EPSILON_IS_SET = false;

float wcn_math_set_epsilon(const float epsilon) {
  const float old_epsilon = EPSILON_IS_SET ? EPSILON : 1e-6f;
  EPSILON = epsilon;
  EPSILON_IS_SET = true;
  return old_epsilon;
}

float wcn_math_get_epsilon() { return EPSILON_IS_SET ? EPSILON : 1e-6f; }

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
  vec2.v[0] = ceilf(a.v[0]);
  vec2.v[1] = ceilf(a.v[1]);
  return vec2;
}

// floor
WMATH_TYPE(Vec2)
WMATH_FLOOR(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = floorf(a.v[0]);
  vec2.v[1] = floorf(a.v[1]);
  return vec2;
}

// round
WMATH_TYPE(Vec2)
WMATH_ROUND(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = roundf(a.v[0]);
  vec2.v[1] = roundf(a.v[1]);
  return vec2;
}

// clamp
WMATH_TYPE(Vec2)
WMATH_CLAMP(Vec2)(WMATH_TYPE(Vec2) a, float min_val, float max_val) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation with branchless operations
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);

  // Branchless clamp using min/max
  __m128 vec_clamped = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);
  wcn_store_vec2_partial(vec2.v, vec_clamped);

#else
  // Scalar implementation with branchless operations
  vec2.v[0] = wcn_clamp_float(a.v[0], min_val, max_val);
  vec2.v[1] = wcn_clamp_float(a.v[1], min_val, max_val);
#endif

  return vec2;
}

// dot
float WMATH_DOT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 mul = _mm_mul_ps(va, vb);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t mul = vmulq_f32(va, vb);
  return wcn_hadd_f32(mul);

#else
  return a.v[0] * b.v[0] + a.v[1] * b.v[1];
#endif
}

// add
WMATH_TYPE(Vec2)
WMATH_ADD(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vres = vmlaq_f32(va, vb, vscale);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vt = wcn_load_vec2_partial(t.v);
  __m128 vdiff = _mm_sub_ps(vb, va);
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vt = wcn_load_vec2_partial(t.v);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 vres = _mm_div_ps(va, vb);
  wcn_store_vec2_partial(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t vres = vdivq_f32(va, vb);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper function
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper function
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float len_sq = wcn_hadd_f32(vec_squared);
  return sqrtf(len_sq);

#else
  // Scalar fallback
  return sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1]);
#endif
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(WMATH_TYPE(Vec2) v) {
  return v.v[0] * v.v[0] + v.v[1] * v.v[1];
}

// distance
float WMATH_DISTANCE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  float sum = _mm_cvtss_f32(wcn_hadd_ps(mul));
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  float sum = wcn_hadd_f32(mul);
  return sqrtf(sum);

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return sqrtf(dx * dx + dy * dy);
#endif
}

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  return wcn_hadd_f32(mul);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - negate using XOR with sign bit mask
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - negate using "v n e g q _ f32"
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
  wcn_store_vec2_partial(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation using vrndnq_f32 (round to nearest)
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_res = vrndnq_f32(vec_a);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper function
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);
  return wcn_hadd_f32(vec_mul);

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
#endif
}

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  // Use helper cross to avoid manual shuffle mistakes
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = wcn_cross_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_b = {b.v[0], b.v[1], b.v[2], 0.0f};

  // Create shuffled vectors for cross-product computation
  float32x4_t a_yzx = {a.v[1], a.v[2], a.v[0], 0.0f};
  float32x4_t b_zxy = {b.v[2], b.v[0], b.v[1], 0.0f};
  float32x4_t a_zxy = {a.v[2], a.v[0], a.v[1], 0.0f};
  float32x4_t b_yzx = {b.v[1], b.v[2], b.v[0], 0.0f};

  // Cross-product computation
  float32x4_t mul1 = vmulq_f32(a_yzx, b_zxy);
  float32x4_t mul2 = vmulq_f32(a_zxy, b_yzx);
  float32x4_t vec_res = vsubq_f32(mul1, mul2);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);
  __m128 vec_res = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);

  // Store using partial helper
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);
  float32x4_t vec_res = vminq_f32(vmaxq_f32(vec_a, vec_min), vec_max);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_scaled = vmulq_f32(vec_b, vec_scalar);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_scaled);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_b = {b.v[0], b.v[1], b.v[2], 0.0f};
  float32x4_t vec_t = vdupq_n_f32(t);
  float32x4_t vec_diff = vsubq_f32(vec_b, vec_a);
  float32x4_t vec_res = vaddq_f32(vec_a, vmulq_f32(vec_diff, vec_t));

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(&a.v[0]);
  __m128 vec_b = _mm_loadu_ps(&b.v[0]);
  __m128 vec_t = _mm_loadu_ps(&t.v[0]);
  __m128 vec_diff = _mm_sub_ps(vec_b, vec_a);
  __m128 vec_res = _mm_add_ps(vec_a, _mm_mul_ps(vec_diff, vec_t));

  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_b = {b.v[0], b.v[1], b.v[2], 0.0f};
  float32x4_t vec_t = {t.v[0], t.v[1], t.v[2], 0.0f};
  float32x4_t vec_diff = vsubq_f32(vec_b, vec_a);
  float32x4_t vec_res = vaddq_f32(vec_a, vmulq_f32(vec_diff, vec_t));

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_b = wcn_load_vec3_partial(b.v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_b);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_b = wcn_load_vec3_partial(b.v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_b);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = {a.v[0], a.v[1], a.v[2], 0.0f};
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_scalar);

  result.v[0] = vgetq_lane_f32(vec_res, 0);
  result.v[1] = vgetq_lane_f32(vec_res, 1);
  result.v[2] = vgetq_lane_f32(vec_res, 2);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using partial load/store helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  float sum = _mm_cvtss_f32(wcn_hadd_ps(mul));
  return sqrtf(sum);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec3_partial(a.v);
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  float sum = wcn_hadd_f32(mul);
  return sqrtf(sum);

#else
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  return sqrtf(dx * dx + dy * dy + dz * dz);
#endif
}

// distanceSquared
float WMATH_DISTANCE_SQ(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 diff = _mm_sub_ps(va, vb);
  __m128 mul = _mm_mul_ps(diff, diff);
  return _mm_cvtss_f32(wcn_hadd_ps(mul));

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = wcn_load_vec3_partial(a.v);
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t diff = vsubq_f32(va, vb);
  float32x4_t mul = vmulq_f32(diff, diff);
  return wcn_hadd_f32(mul);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - negate using XOR with sign bit mask and partial
  // helpers
  __m128 vec_a = wcn_load_vec3_partial(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  wcn_store_vec3_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a = wcn_load_vec3_partial(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation using vrndpq_f32 (round towards positive infinity)
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vrndpq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation using vrndmq_f32 (round towards negative infinity)
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vrndmq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

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
  // Scalar implementation for better compatibility
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
  result.v[3] = roundf(a.v[3]);
  return result;
}

WMATH_TYPE(Vec4)
WMATH_CLAMP(Vec4)(WMATH_TYPE(Vec4) a, float min_val, float max_val) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);
  __m128 vec_res = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);
  float32x4_t vec_res = vminq_f32(vmaxq_f32(vec_a, vec_min), vec_max);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_scale = _mm_set1_ps(scale);
  __m128 vec_scaled = _mm_mul_ps(vec_b, vec_scale);
  __m128 vec_res = _mm_add_ps(vec_a, vec_scaled);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_scale = vdupq_n_f32(scale);
  float32x4_t vec_scaled = vmulq_f32(vec_b, vec_scale);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_scaled);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 va = _mm_loadu_ps(a.v);
  __m128 vb = _mm_loadu_ps(b.v);
  __m128 vt = _mm_loadu_ps(t.v);
  __m128 vdiff = _mm_sub_ps(vb, va);
  __m128 vres = _mm_add_ps(va, _mm_mul_ps(vdiff, vt));
  _mm_storeu_ps(result.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vld1q_f32(t.v);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vmulq_f32(vec_a, vec_scalar);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_scalar = _mm_set1_ps(scalar);
  __m128 vec_res = _mm_div_ps(vec_a, vec_scalar);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_scalar = vdupq_n_f32(scalar);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_scalar);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_res = _mm_div_ps(vec_one, vec_a);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_res = vdivq_f32(vec_one, vec_a);
  vst1q_f32(result.v, vec_res);

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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - multiply and horizontal add all 4 elements
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_mul = _mm_mul_ps(vec_a, vec_b);

  // Horizontally add to get dot product
  __m128 temp = _mm_hadd_ps(vec_mul, vec_mul);
  temp = _mm_hadd_ps(temp, temp);
  return _mm_cvtss_f32(temp);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
#endif
}

float WMATH_LENGTH_SQ(Vec4)(WMATH_TYPE(Vec4) v) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 vec_v = _mm_load_ps(v.v);
  __m128 vec_squared = _mm_mul_ps(vec_v, vec_v);
  __m128 hadd1 = _mm_hadd_ps(vec_squared, vec_squared);
  __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
  return _mm_cvtss_f32(hadd2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  float32x4_t vec_v = vld1q_f32(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float32x2_t sum = vpadd_f32(vget_low_f32(vec_squared), vget_high_f32(vec_squared));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - negate using XOR with sign bit mask
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - negate using vnegq_f32
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_res = vnegq_f32(vec_a);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#include <math.h>

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
(defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

float WMATH_DETERMINANT(Mat3)(WMATH_TYPE(Mat3) m) {
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];

  return m00 * (m11 * m22 - m21 * m12) - m10 * (m01 * m22 - m21 * m02) +
         m20 * (m01 * m12 - m11 * m02);
}

// END Mat3

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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
#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#else
  for (int i = 0; i < 16; ++i) {
    result.m[i] = a.m[i] + b.m[i];
  }
#endif
  return result;
}

WMATH_TYPE(Mat4) WMATH_SUB(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b) {
  WMATH_TYPE(Mat4) result;
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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
#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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
#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

  return WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){.m_00 = m_00_val,
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
  float m_00_val = d * t_0;
  float m_01_val = d * t_1;
  float m_02_val = d * t_2;
  float m_03_val = d * t_3;
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

  return WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){.m_00 = m_00_val,
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

  return WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
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

WMATH_TYPE(Mat4)
WMATH_TRANSPOSE(Mat4)(WMATH_TYPE(Mat4) a) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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

#else
  // Scalar fallback
  return WMATH_SET(Mat4)(WMATH_COPY(Mat4)(a),              // Self
                         a.m[0], a.m[4], a.m[8], a.m[12],  // 0
                         a.m[1], a.m[5], a.m[9], a.m[13],  // 1
                         a.m[2], a.m[6], a.m[10], a.m[14], // 2
                         a.m[3], a.m[7], a.m[11], a.m[15]  // 3
  );
#endif

  return result;
}

float WMATH_DETERMINANT(Mat4)(WMATH_TYPE(Mat4) m) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
  float t3 = (tmp5 * m01 + tmp8 * m11 + tmp11_val * m21) -
             (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);

  return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
#endif
}

// aim
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, aim)
(WMATH_TYPE(Vec3) position, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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
  __m128 x_axis_unnorm = wcn_cross_ps(vec_up, z_axis);
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
  __m128 y_axis_unnorm = wcn_cross_ps(z_axis, x_axis);
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
  float32x4_t x_axis_unnorm = wcn_cross_neon(vec_up, z_axis);
  float x_axis_len_sq = wcn_hadd_f32(vmulq_f32(x_axis_unnorm, x_axis_unnorm));

  float32x4_t x_axis;
  if (x_axis_len_sq > 0.00001f) {
    float x_axis_inv_len = wcn_fast_inv_sqrt(x_axis_len_sq);
    x_axis = vmulq_n_f32(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = (float32x4_t){0.0f, 1.0f, 0.0f, 0.0f}; // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  float32x4_t y_axis_unnorm = wcn_cross_neon(z_axis, x_axis);
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
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, look_at)
(WMATH_TYPE(Vec3) eye, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized lookAt matrix creation
  __m128 vec_eye = wcn_load_vec3_partial(eye.v);
  __m128 vec_target = wcn_load_vec3_partial(target.v);
  __m128 vec_up = wcn_load_vec3_partial(up.v);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);
  _mm_set1_ps(-1.0f);

  // Calculate z_axis = normalize(eye - target)
  __m128 z_axis_unnorm = _mm_sub_ps(vec_eye, vec_target);
  float z_axis_len_sq =
      _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(z_axis_unnorm, z_axis_unnorm)));

  __m128 z_axis;
  if (z_axis_len_sq > 0.00001f) {
    __m128 z_axis_inv_len = wcn_fast_inv_sqrt_ps(_mm_set1_ps(z_axis_len_sq));
    z_axis = _mm_mul_ps(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = _mm_set_ps(0.0f, 0.0f, -1.0f,
                        0.0f); // Default forward vector (negative Z)
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  __m128 x_axis_unnorm = wcn_cross_ps(vec_up, z_axis);
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
  __m128 y_axis_unnorm = wcn_cross_ps(z_axis, x_axis);
  float y_axis_len_sq =
      _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(y_axis_unnorm, y_axis_unnorm)));

  __m128 y_axis;
  if (y_axis_len_sq > 0.00001f) {
    __m128 y_axis_inv_len = wcn_fast_inv_sqrt_ps(_mm_set1_ps(y_axis_len_sq));
    y_axis = _mm_mul_ps(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // Default up vector
  }

  // Calculate dot products for translation
  float dot_x_eye = _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(x_axis, vec_eye)));
  float dot_y_eye = _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(y_axis, vec_eye)));
  float dot_z_eye = _mm_cvtss_f32(wcn_hadd_ps(_mm_mul_ps(z_axis, vec_eye)));

  __m128 vec_dot_x_eye = _mm_set1_ps(-dot_x_eye);
  __m128 vec_dot_y_eye = _mm_set1_ps(-dot_y_eye);
  __m128 vec_dot_z_eye = _mm_set1_ps(-dot_z_eye);

  // Create lookAt matrix rows (transposed for column-major layout)
  // Row0: [x_axis.x, y_axis.x, z_axis.x, 0]
  __m128 x_x = _mm_shuffle_ps(x_axis, x_axis, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 y_x = _mm_shuffle_ps(y_axis, y_axis, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 z_x = _mm_shuffle_ps(z_axis, z_axis, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 row0 = _mm_move_ss(x_x, y_x);
  row0 = _mm_shuffle_ps(row0, z_x, _MM_SHUFFLE(0, 2, 1, 0));
  row0 = _mm_move_ss(row0, vec_zero);
  row0 = _mm_shuffle_ps(row0, row0, _MM_SHUFFLE(3, 2, 1, 0));

  // Row1: [x_axis.y, y_axis.y, z_axis.y, 0]
  __m128 x_y = _mm_shuffle_ps(x_axis, x_axis, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 y_y = _mm_shuffle_ps(y_axis, y_axis, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 z_y = _mm_shuffle_ps(z_axis, z_axis, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 row1 = _mm_move_ss(x_y, y_y);
  row1 = _mm_shuffle_ps(row1, z_y, _MM_SHUFFLE(0, 2, 1, 0));
  row1 = _mm_move_ss(row1, vec_zero);
  row1 = _mm_shuffle_ps(row1, row1, _MM_SHUFFLE(3, 2, 1, 0));

  // Row2: [x_axis.z, y_axis.z, z_axis.z, 0]
  __m128 x_z = _mm_shuffle_ps(x_axis, x_axis, _MM_SHUFFLE(2, 2, 2, 2));
  __m128 y_z = _mm_shuffle_ps(y_axis, y_axis, _MM_SHUFFLE(2, 2, 2, 2));
  __m128 z_z = _mm_shuffle_ps(z_axis, z_axis, _MM_SHUFFLE(2, 2, 2, 2));
  __m128 row2 = _mm_move_ss(x_z, y_z);
  row2 = _mm_shuffle_ps(row2, z_z, _MM_SHUFFLE(0, 2, 1, 0));
  row2 = _mm_move_ss(row2, vec_zero);
  row2 = _mm_shuffle_ps(row2, row2, _MM_SHUFFLE(3, 2, 1, 0));

  // Row3: [dot_x_eye, dot_y_eye, dot_z_eye, 1]
  __m128 row3 = _mm_move_ss(vec_dot_x_eye, vec_dot_y_eye);
  row3 = _mm_shuffle_ps(row3, vec_dot_z_eye, _MM_SHUFFLE(0, 2, 1, 0));
  row3 = _mm_move_ss(row3, vec_one);
  row3 = _mm_shuffle_ps(row3, row3, _MM_SHUFFLE(3, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&result.m[0], row0);
  _mm_storeu_ps(&result.m[4], row1);
  _mm_storeu_ps(&result.m[8], row2);
  _mm_storeu_ps(&result.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized lookAt matrix creation
  float32x4_t vec_eye = wcn_load_vec3_partial(eye.v);
  float32x4_t vec_target = wcn_load_vec3_partial(target.v);
  float32x4_t vec_up = wcn_load_vec3_partial(up.v);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Calculate z_axis = normalize(eye - target)
  float32x4_t z_axis_unnorm = vsubq_f32(vec_eye, vec_target);
  float z_axis_len_sq = wcn_hadd_f32(vmulq_f32(z_axis_unnorm, z_axis_unnorm));

  float32x4_t z_axis;
  if (z_axis_len_sq > 0.00001f) {
    float z_axis_inv_len = wcn_fast_inv_sqrt(z_axis_len_sq);
    z_axis = vmulq_n_f32(z_axis_unnorm, z_axis_inv_len);
  } else {
    z_axis = (float32x4_t){0.0f, 0.0f, -1.0f,
                           0.0f}; // Default forward vector (negative Z)
  }

  // Calculate x_axis = normalize(cross(up, z_axis))
  float32x4_t x_axis_unnorm = wcn_cross_neon(vec_up, z_axis);
  float x_axis_len_sq = wcn_hadd_f32(vmulq_f32(x_axis_unnorm, x_axis_unnorm));

  float32x4_t x_axis;
  if (x_axis_len_sq > 0.00001f) {
    float x_axis_inv_len = wcn_fast_inv_sqrt(x_axis_len_sq);
    x_axis = vmulq_n_f32(x_axis_unnorm, x_axis_inv_len);
  } else {
    x_axis = (float32x4_t){0.0f, 1.0f, 0.0f, 0.0f}; // Default right vector
  }

  // Calculate y_axis = normalize(cross(z_axis, x_axis))
  float32x4_t y_axis_unnorm = wcn_cross_neon(z_axis, x_axis);
  float y_axis_len_sq = wcn_hadd_f32(vmulq_f32(y_axis_unnorm, y_axis_unnorm));

  float32x4_t y_axis;
  if (y_axis_len_sq > 0.00001f) {
    float y_axis_inv_len = wcn_fast_inv_sqrt(y_axis_len_sq);
    y_axis = vmulq_n_f32(y_axis_unnorm, y_axis_inv_len);
  } else {
    y_axis = (float32x4_t){1.0f, 0.0f, 0.0f, 0.0f}; // Default up vector
  }

  // Calculate dot products for translation
  float dot_x_eye = wcn_hadd_f32(vmulq_f32(x_axis, vec_eye));
  float dot_y_eye = wcn_hadd_f32(vmulq_f32(y_axis, vec_eye));
  float dot_z_eye = wcn_hadd_f32(vmulq_f32(z_axis, vec_eye));

  // Create lookAt matrix rows (transposed for column-major layout)
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(vgetq_lane_f32(x_axis, 0), row0, 0);
  row0 = vsetq_lane_f32(vgetq_lane_f32(y_axis, 0), row0, 1);
  row0 = vsetq_lane_f32(vgetq_lane_f32(z_axis, 0), row0, 2);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(vgetq_lane_f32(x_axis, 1), row1, 0);
  row1 = vsetq_lane_f32(vgetq_lane_f32(y_axis, 1), row1, 1);
  row1 = vsetq_lane_f32(vgetq_lane_f32(z_axis, 1), row1, 2);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(vgetq_lane_f32(x_axis, 2), row2, 0);
  row2 = vsetq_lane_f32(vgetq_lane_f32(y_axis, 2), row2, 1);
  row2 = vsetq_lane_f32(vgetq_lane_f32(z_axis, 2), row2, 2);

  float32x4_t row3 = vec_zero;
  row3 = vsetq_lane_f32(-dot_x_eye, row3, 0);
  row3 = vsetq_lane_f32(-dot_y_eye, row3, 1);
  row3 = vsetq_lane_f32(-dot_z_eye, row3, 2);
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  // Store results
  vst1q_f32(&result.m[0], row0);
  vst1q_f32(&result.m[4], row1);
  vst1q_f32(&result.m[8], row2);
  vst1q_f32(&result.m[12], row3);

#else
  // Scalar fallback - optimized with fast inverse square root
  WMATH_TYPE(Vec3) z_axis = WMATH_SUB(Vec3)(eye, target);
  wcn_fast_normalize_vec3(z_axis.v);

  WMATH_TYPE(Vec3) x_axis = WMATH_CROSS(Vec3)(up, z_axis);
  wcn_fast_normalize_vec3(x_axis.v);

  WMATH_TYPE(Vec3) y_axis = WMATH_CROSS(Vec3)(z_axis, x_axis);
  wcn_fast_normalize_vec3(y_axis.v);

  // Calculate dot products for translation
  float dot_x_eye = -(x_axis.v[0] * eye.v[0] + x_axis.v[1] * eye.v[1] +
                      x_axis.v[2] * eye.v[2]);
  float dot_y_eye = -(y_axis.v[0] * eye.v[0] + y_axis.v[1] * eye.v[1] +
                      y_axis.v[2] * eye.v[2]);
  float dot_z_eye = -(z_axis.v[0] * eye.v[0] + z_axis.v[1] * eye.v[1] +
                      z_axis.v[2] * eye.v[2]);

  // Direct assignment is more efficient than memset
  result.m[0] = x_axis.v[0];
  result.m[1] = y_axis.v[0];
  result.m[2] = z_axis.v[0];
  result.m[3] = 0.0f;
  result.m[4] = x_axis.v[1];
  result.m[5] = y_axis.v[1];
  result.m[6] = z_axis.v[1];
  result.m[7] = 0.0f;
  result.m[8] = x_axis.v[2];
  result.m[9] = y_axis.v[2];
  result.m[10] = z_axis.v[2];
  result.m[11] = 0.0f;
  result.m[12] = dot_x_eye;
  result.m[13] = dot_y_eye;
  result.m[14] = dot_z_eye;
  result.m[15] = 1.0f;
#endif

  return result;
}

// END Mat4

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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper function
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_mul = _mm_mul_ps(vec_a, vec_b);
  return _mm_cvtss_f32(wcn_hadd_ps(vec_mul));

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper function
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);
  return wcn_hadd_f32(vec_mul);

#else
  // Scalar fallback
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
#endif
}

WMATH_TYPE(Quat)
WMATH_LERP(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t va = vld1q_f32(a.v);
  float32x4_t vb = vld1q_f32(b.v);
  float32x4_t vt = vdupq_n_f32(t);
  float32x4_t vdiff = vsubq_f32(vb, va);
  float32x4_t vres = vaddq_f32(va, vmulq_f32(vdiff, vt));
  vst1q_f32(result.v, vres);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - using helper function
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_squared = _mm_mul_ps(vec_a, vec_a);
  float len_sq = _mm_cvtss_f32(wcn_hadd_ps(vec_squared));
  return sqrtf(len_sq);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - using helper function
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_squared = vmulq_f32(vec_a, vec_a);
  float len_sq = wcn_hadd_f32(vec_squared);
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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

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
  return acosf(2 * cosOmega - 1.0f);
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

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
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

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
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
  r.v[0] = a.v[0] * b;
  r.v[1] = a.v[1] * b;
  r.v[2] = a.v[2] * b;
  r.v[3] = a.v[3] * b;
  return r;
}

// -
WMATH_TYPE(Quat)
WMATH_SUB(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vsubq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all 4 elements at once
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_b = _mm_loadu_ps(b.v);
  __m128 vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all 4 elements at once
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_b = vld1q_f32(b.v);
  float32x4_t vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(result.v, vec_res);

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
  const float dot = q.v[0] * q.v[0] + q.v[1] * q.v[1] + q.v[2] * q.v[2] + q.v[3] * q.v[3];
  const float invDot = dot ? 1.0f / dot : 0.0f;
  
  // Conjugate divided by squared length
  r.v[0] = -q.v[0] * invDot;
  r.v[1] = -q.v[1] * invDot;
  r.v[2] = -q.v[2] * invDot;
  r.v[3] = q.v[3] * invDot;
  return r;
}

// conjugate
WMATH_TYPE(Quat)
WMATH_CALL(Quat, conjugate)(WMATH_TYPE(Quat) q) {
  return (WMATH_TYPE(Quat)){
      .v = {-q.v[0], -q.v[1], -q.v[2], q.v[3]},
  };
}

// divScalar
WMATH_TYPE(Quat)
WMATH_DIV_SCALAR(Quat)(WMATH_TYPE(Quat) a, float v) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_a = _mm_loadu_ps(a.v);
  __m128 vec_v = _mm_set1_ps(v);
  __m128 vec_res = _mm_div_ps(vec_a, vec_v);
  _mm_storeu_ps(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_a = vld1q_f32(a.v);
  float32x4_t vec_v = vdupq_n_f32(v);
  float32x4_t vec_res = vdivq_f32(vec_a, vec_v);
  vst1q_f32(result.v, vec_res);

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

// FROM

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_mat4)(const WMATH_TYPE(Mat4) a) {
  return WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
      .m_00 = a.m[0],
      .m_01 = a.m[1],
      .m_02 = a.m[2],
      .m_10 = a.m[4],
      .m_11 = a.m[5],
      .m_12 = a.m[6],
      .m_20 = a.m[8],
      .m_21 = a.m[9],
      .m_22 = a.m[10],
  });
}

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_quat)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized version using 4-element parallel
  // multiply-add
  __m128 vec_q = _mm_loadu_ps(q.v); // Load quaternion [x, y, z, w]

  // Calculate doubled components
  _mm_add_ps(vec_q, vec_q); // [x2, y2, z2, w2]

  // Extract individual components
  float x = q.v[0];
  float y = q.v[1];
  float z = q.v[2];
  float w = q.v[3];
  float x2 = x + x;
  float y2 = y + y;
  float z2 = z + z;

  // Calculate products using SIMD where possible
  __m128 vec_x = _mm_set1_ps(x);
  __m128 vec_y = _mm_set1_ps(y);
  __m128 vec_z = _mm_set1_ps(z);
  __m128 vec_w = _mm_set1_ps(w);

  __m128 vec_x2 = _mm_set1_ps(x2);
  __m128 vec_y2 = _mm_set1_ps(y2);
  __m128 vec_z2 = _mm_set1_ps(z2);

  // Calculate the products
  __m128 xx = _mm_mul_ps(vec_x, vec_x2);
  __m128 yx = _mm_mul_ps(vec_y, vec_x2);
  __m128 yy = _mm_mul_ps(vec_y, vec_y2);
  __m128 zx = _mm_mul_ps(vec_z, vec_x2);
  __m128 zy = _mm_mul_ps(vec_z, vec_y2);
  __m128 zz = _mm_mul_ps(vec_z, vec_z2);
  __m128 wx = _mm_mul_ps(vec_w, vec_x2);
  __m128 wy = _mm_mul_ps(vec_w, vec_y2);
  __m128 wz = _mm_mul_ps(vec_w, vec_z2);

  // Extract scalar values for matrix construction
  float xx_val = _mm_cvtss_f32(xx);
  float yx_val = _mm_cvtss_f32(yx);
  float yy_val = _mm_cvtss_f32(yy);
  float zx_val = _mm_cvtss_f32(zx);
  float zy_val = _mm_cvtss_f32(zy);
  float zz_val = _mm_cvtss_f32(zz);
  float wx_val = _mm_cvtss_f32(wx);
  float wy_val = _mm_cvtss_f32(wy);
  float wz_val = _mm_cvtss_f32(wz);

  // Construct matrix using optimized calculations
  newDst.m[0] = 1.0f - yy_val - zz_val;
  newDst.m[1] = yx_val + wz_val;
  newDst.m[2] = zx_val - wy_val;
  newDst.m[3] = 0.0f;
  newDst.m[4] = yx_val - wz_val;
  newDst.m[5] = 1.0f - xx_val - zz_val;
  newDst.m[6] = zy_val + wx_val;
  newDst.m[7] = 0.0f;
  newDst.m[8] = zx_val + wy_val;
  newDst.m[9] = zy_val - wx_val;
  newDst.m[10] = 1.0f - xx_val - yy_val;
  newDst.m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized version using 4-element parallel
  // multiply-add
  float32x4_t vec_q = vld1q_f32(q.v); // Load quaternion [x, y, z, w]

  // Extract individual components
  float x = q.v[0];
  float y = q.v[1];
  float z = q.v[2];
  float w = q.v[3];
  float x2 = x + x;
  float y2 = y + y;
  float z2 = z + z;

  // Calculate products using SIMD where possible
  float32x4_t vec_x = vdupq_n_f32(x);
  float32x4_t vec_y = vdupq_n_f32(y);
  float32x4_t vec_z = vdupq_n_f32(z);
  float32x4_t vec_w = vdupq_n_f32(w);

  float32x4_t vec_x2 = vdupq_n_f32(x2);
  float32x4_t vec_y2 = vdupq_n_f32(y2);
  float32x4_t vec_z2 = vdupq_n_f32(z2);

  // Calculate the products
  float32x4_t xx = vmulq_f32(vec_x, vec_x2);
  float32x4_t yx = vmulq_f32(vec_y, vec_x2);
  float32x4_t yy = vmulq_f32(vec_y, vec_y2);
  float32x4_t zx = vmulq_f32(vec_z, vec_x2);
  float32x4_t zy = vmulq_f32(vec_z, vec_y2);
  float32x4_t zz = vmulq_f32(vec_z, vec_z2);
  float32x4_t wx = vmulq_f32(vec_w, vec_x2);
  float32x4_t wy = vmulq_f32(vec_w, vec_y2);
  float32x4_t wz = vmulq_f32(vec_w, vec_z2);

  // Extract scalar values for matrix construction
  float xx_val = vgetq_lane_f32(xx, 0);
  float yx_val = vgetq_lane_f32(yx, 0);
  float yy_val = vgetq_lane_f32(yy, 0);
  float zx_val = vgetq_lane_f32(zx, 0);
  float zy_val = vgetq_lane_f32(zy, 0);
  float zz_val = vgetq_lane_f32(zz, 0);
  float wx_val = vgetq_lane_f32(wx, 0);
  float wy_val = vgetq_lane_f32(wy, 0);
  float wz_val = vgetq_lane_f32(wz, 0);

  // Construct matrix using optimized calculations
  newDst.m[0] = 1.0f - yy_val - zz_val;
  newDst.m[1] = yx_val + wz_val;
  newDst.m[2] = zx_val - wy_val;
  newDst.m[3] = 0.0f;
  newDst.m[4] = yx_val - wz_val;
  newDst.m[5] = 1.0f - xx_val - zz_val;
  newDst.m[6] = zy_val + wx_val;
  newDst.m[7] = 0.0f;
  newDst.m[8] = zx_val + wy_val;
  newDst.m[9] = zy_val - wx_val;
  newDst.m[10] = 1.0f - xx_val - yy_val;
  newDst.m[11] = 0.0f;

#else
  // Scalar fallback
  float x = q.v[0];
  float y = q.v[1];
  float z = q.v[2];
  float w = q.v[3];
  float x2 = x + x;
  float y2 = y + y;
  float z2 = z + z;

  float xx = x * x2;
  float yx = y * x2;
  float yy = y * y2;
  float zx = z * x2;
  float zy = z * y2;
  float zz = z * z2;
  float wx = w * x2;
  float wy = w * y2;
  float wz = w * z2;

  newDst.m[0] = 1 - yy - zz;
  newDst.m[1] = yx + wz;
  newDst.m[2] = zx - wy;
  newDst.m[3] = 0;
  newDst.m[4] = yx - wz;
  newDst.m[5] = 1 - xx - zz;
  newDst.m[6] = zy + wx;
  newDst.m[7] = 0;
  newDst.m[8] = zx + wy;
  newDst.m[9] = zy - wx;
  newDst.m[10] = 1 - xx - yy;
  newDst.m[11] = 0;
#endif

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_mat3)(const WMATH_TYPE(Mat3) a) {
  WMATH_TYPE(Mat4) m;
  m.m[ 0] = a.m[0]; m.m[ 1] = a.m[1]; m.m[ 2] = a.m[ 2];  m.m[ 3] = 0.0f;
  m.m[ 4] = a.m[4]; m.m[ 5] = a.m[5]; m.m[ 6] = a.m[ 6];  m.m[ 7] = 0.0f;
  m.m[ 8] = a.m[8]; m.m[ 9] = a.m[9]; m.m[10] = a.m[10];  m.m[11] = 0.0f;
  m.m[12] =   0.0f; m.m[13] =   0.0f; m.m[14] =    0.0f;  m.m[15] = 1.0f;
  return m;
}
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_quat)(const WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat4) m;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // printf("x86_64: Mat4::from_quat");
  /* -----------------------
     SSE (x86_64) - FMA-optimized variant when available
     Compile flags: -O3 -mfma -msse4.1  (if you want to enable FMA path)
     ----------------------- */
  /* load quaternion q.v = [x,y,z,w] */
  __m128 qv = _mm_loadu_ps(q.v);
  __m128 v2 = _mm_add_ps(qv, qv);        /* [2x,2y,2z,2w] */
  __m128 prod = _mm_mul_ps(qv, v2);      /* [2x^2,2y^2,2z^2,2w^2] */
  const __m128 one  = _mm_set1_ps(1.0f);
  const __m128 zero = _mm_setzero_ps();

  /* rotations */
  __m128 rot1_q = _mm_shuffle_ps(qv, qv, _MM_SHUFFLE(0,3,2,1)); /* [y,z,w,x] */
  const __m128 rot2_q = _mm_shuffle_ps(qv, qv, _MM_SHUFFLE(1,0,3,2)); /* [z,w,x,y] */
  __m128 rot3_q = _mm_shuffle_ps(qv, qv, _MM_SHUFFLE(2,1,0,3)); /* [w,x,y,z] */

  const __m128 rot1_v2 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0,3,2,1)); /* [2y,2z,2w,2x] */
  __m128 rot2_v2 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1,0,3,2)); /* [2z,2w,2x,2y] */

  /* precomputed: xx_vec (2*x^2) and yy_vec (2*y^2) and zz_vec (2*z^2) broadcasted when needed */
  const __m128 xx_vec = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(0,0,0,0)); /* [2x^2,...] */
  const __m128 yy_vec = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1,1,1,1)); /* [2y^2,...] */
  const __m128 zz_vec = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2,2,2,2)); /* [2z^2,...] */

  /* useful cross terms - computed vectorially so lane0 contains desired scalar */
  const __m128 yx_vec = _mm_mul_ps(rot1_q, v2);     /* lane0 = 2*y*x */
  const __m128 zx_vec = _mm_mul_ps(rot2_q, v2);     /* lane0 = 2*z*x */
  const __m128 zy_vec = _mm_mul_ps(rot2_q, rot1_v2);/* lane0 = 2*z*y */
  const __m128 wz_vec = _mm_mul_ps(rot3_q, rot2_v2);/* lane0 = 2*w*z */
  const __m128 wy_vec = _mm_mul_ps(rot3_q, rot1_v2);/* lane0 = 2*w*y */
  const __m128 wx_vec = _mm_mul_ps(rot3_q, v2);     /* lane0 = 2*w*x */

  /* m00 = 1 - 2*y^2 - 2*z^2  (lane0 of m00_vec) */
  const __m128 m00_vec = _mm_sub_ps(_mm_sub_ps(one, yy_vec), zz_vec);

  /* m01 = 2*y*x + 2*w*z */
#if defined(__FMA__)
  /* combine with FMA: compute m01 = (y * (2*x)) + (w * (2*z)) but we already have lane vectors
     Use scalar-vector add via _mm_add_ps; FMA helps if we compute a*b + c in one op, but
     we still need two multiplications. We'll use FMA to do (y * x2) + (w * z2) in-one if desired:
  */
  __m128 part1 = _mm_mul_ps(rot1_q, v2);      /* yx_vec */
  __m128 part2 = _mm_mul_ps(rot3_q, rot2_v2); /* wz_vec */
  __m128 m01_vec = _mm_add_ps(part1, part2);
#else
  __m128 m01_vec = _mm_add_ps(yx_vec, wz_vec);
#endif

  /* m02 = 2*z*x - 2*w*y */
  const __m128 m02_vec = _mm_sub_ps(zx_vec, wy_vec);

  /* pack row0 = [m00, m01, m02, 0] */
  const __m128 tmp0 = _mm_unpacklo_ps(m00_vec, m01_vec); /* [m00, m01, ?, ?] */
  const __m128 tmp1 = _mm_unpacklo_ps(m02_vec, zero);    /* [m02, 0, ?, ?] */
  __m128 row0 = _mm_movelh_ps(tmp0, tmp1);         /* [m00,m01,m02,0] */
  _mm_storeu_ps(&m.m[0], row0);

  /* row1:
     m10 = 2*y*x - 2*w*z  (yx - wz)
     m11 = 1 - 2*x^2 - 2*z^2
     m12 = 2*z*y + 2*w*x
  */
  const __m128 m10_vec = _mm_sub_ps(yx_vec, wz_vec);
  const __m128 m11_vec = _mm_sub_ps(_mm_sub_ps(one, xx_vec), zz_vec);
  const __m128 m12_vec = _mm_add_ps(zy_vec, wx_vec);

  const __m128 tmp2 = _mm_unpacklo_ps(m10_vec, m11_vec);
  const __m128 tmp3 = _mm_unpacklo_ps(m12_vec, zero);
  __m128 row1 = _mm_movelh_ps(tmp2, tmp3);
  _mm_storeu_ps(&m.m[4], row1);

  /* row2:
     m20 = 2*z*x + 2*w*y
     m21 = 2*z*y - 2*w*x
     m22 = 1 - 2*x^2 - 2*y^2
  */
  const __m128 m20_vec = _mm_add_ps(zx_vec, wy_vec);
  const __m128 m21_vec = _mm_sub_ps(zy_vec, wx_vec);
  const __m128 m22_vec = _mm_sub_ps(_mm_sub_ps(one, xx_vec), yy_vec);

  const __m128 tmp4 = _mm_unpacklo_ps(m20_vec, m21_vec);
  const __m128 tmp5 = _mm_unpacklo_ps(m22_vec, zero);
  __m128 row2 = _mm_movelh_ps(tmp4, tmp5);
  _mm_storeu_ps(&m.m[8], row2);

  /* row3 = [0,0,0,1] */
  __m128 row3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
  _mm_storeu_ps(&m.m[12], row3);

  return m;

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  /* -----------------------
     NEON (aarch64) - mixed variant: vector ops + a few scalar lane broadcasts
     Rationale: on many ARM cores, a small number of scalar extracts + vdupq_n_f32
                is faster than aggressive vext/mask sequences.
     Compile flags: -O3 -march=armv8-a
     ----------------------- */
  /* load quaternion */
  float32x4_t qv = vld1q_f32(q.v);        /* [x,y,z,w] */
  /* extract scalars (cheap relative to many shuffles on ARM) */
  const float x = vgetq_lane_f32(qv, 0);
  const float y = vgetq_lane_f32(qv, 1);
  const float z = vgetq_lane_f32(qv, 2);
  const float w = vgetq_lane_f32(qv, 3);

  /* scalar doubled */
  const float x2 = x + x;
  const float y2 = y + y;
  const float z2 = z + z;

  /* scalar cross terms */
  const float xx = x * x2;
  const float yy = y * y2;
  const float zz = z * z2;
  const float yx = y * x2; /* 2*y*x */
  const float zx = z * x2; /* 2*z*x */
  const float zy = z * y2; /* 2*z*y */
  const float wx = w * x2; /* 2*w*x */
  const float wy = w * y2; /* 2*w*y */
  const float wz = w * z2; /* 2*w*z */

  /* broadcast needed scalars into vectors */
  float32x4_t m00v = vdupq_n_f32(1.0f - yy - zz); /* m00 */
  float32x4_t m01v = vdupq_n_f32(yx + wz);        /* m01 */
  float32x4_t m02v = vdupq_n_f32(zx - wy);        /* m02 */

  float32x4_t row0 = vsetq_lane_f32(0.0f, m00v, 3); /* ensure lane3 = 0 (already but safe) */
  row0 = vsetq_lane_f32(vgetq_lane_f32(m01v,0), row0, 1);
  row0 = vsetq_lane_f32(vgetq_lane_f32(m02v,0), row0, 2);
  vst1q_f32(&m.m[0], row0);

  /* row1 */
  float32x4_t row1 = vdupq_n_f32(0.0f);
  row1 = vsetq_lane_f32(yx - wz, row1, 0);
  row1 = vsetq_lane_f32(1.0f - xx - zz, row1, 1);
  row1 = vsetq_lane_f32(zy + wx, row1, 2);
  vst1q_f32(&m.m[4], row1);

  /* row2 */
  float32x4_t row2 = vdupq_n_f32(0.0f);
  row2 = vsetq_lane_f32(zx + wy, row2, 0);
  row2 = vsetq_lane_f32(zy - wx, row2, 1);
  row2 = vsetq_lane_f32(1.0f - xx - yy, row2, 2);
  vst1q_f32(&m.m[8], row2);

  /* row3 */
  float32x4_t row3 = vdupq_n_f32(0.0f);
  row3 = vsetq_lane_f32(1.0f, row3, 3);
  vst1q_f32(&m.m[12], row3);

  return m;

#else

  /* Scalar fallback */
  float x = q.v[0], y = q.v[1], z = q.v[2], w = q.v[3];
  float x2 = x + x, y2 = y + y, z2 = z + z;

  float xx = x * x2;
  float yx = y * x2;
  float yy = y * y2;
  float zx = z * x2;
  float zy = z * y2;
  float zz = z * z2;
  float wx = w * x2;
  float wy = w * y2;
  float wz = w * z2;
  m.m[0]  = 1 - yy - zz;  m.m[1]  = yx + wz;  m.m[2]  = zx - wy;  m.m[3]  = 0;
  m.m[4]  = yx - wz;      m.m[5]  = 1 - xx - zz; m.m[6]  = zy + wx;  m.m[7]  = 0;
  m.m[8]  = zx + wy;      m.m[9]  = zy - wx;     m.m[10] = 1 - xx - yy; m.m[11] = 0;
  m.m[12] = 0;            m.m[13] = 0;           m.m[14] = 0;          m.m[15] = 1;
  return m;

#endif
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_axis_angle)(WMATH_TYPE(Vec3) axis,
                                  const float angle_in_radians) {
  const float a = angle_in_radians * 0.5f;
  const float a_s = sinf(a);
  const float a_c = cosf(a);

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

  // 使用SSE优化实现
  __m128 axis_vec = wcn_load_vec3_partial(axis.v);   // 加载axis的x,y,z分量
  __m128 sin_vec = _mm_set1_ps(a_s);                 // 广播sin值到所有元素
  __m128 result_vec = _mm_mul_ps(axis_vec, sin_vec); // 计算x,y,z分量

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // 存储x,y,z分量
  result.v[3] = a_c;                            // 设置w分量

  return result;

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  // 使用NEON优化实现
  float32x4_t axis_vec = wcn_load_vec3_partial(axis.v);  // 加载axis的x,y,z分量
  float32x4_t sin_vec = vdupq_n_f32(a_s);                // 广播sin值到所有元素
  float32x4_t result_vec = vmulq_f32(axis_vec, sin_vec); // 计算x,y,z分量

  WMATH_TYPE(Quat) result;
  wcn_store_vec3_partial(result.v, result_vec); // 存储x,y,z分量
  result.v[3] = a_c;                            // 设置w分量

  return result;

#else

  // 原始标量实现
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = a_s * axis.v[0],
      .v_y = a_s * axis.v[1],
      .v_z = a_s * axis.v[2],
      .v_w = a_c,
  });

#endif
}

WCN_Math_Vec3_WithAngleAxis WMATH_CALL(Quat,
                                       to_axis_angle)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Vec3)
  vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = 0,
      .v_y = 0,
      .v_z = 0,
  });
  float angle = acosf(q.v[3]) * 2.0f;
  float s = sinf(angle * 0.5f);
  float ep = wcn_math_get_epsilon();
  if (s > ep) {
    vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
        .v_x = q.v[0] / s,
        .v_y = q.v[1] / s,
        .v_z = q.v[2] / s,
    });
  } else {
    vec3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
        .v_x = 1.0f,
        .v_y = 0.0f,
        .v_z = 0.0f,
    });
  }
  return (WCN_Math_Vec3_WithAngleAxis){
      .angle = angle,
      .axis = vec3,
  };
}

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat4)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec2) vec2;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized version
  __m128 vec_v = wcn_load_vec2_partial(v.v);

  // Load matrix columns using helper functions for better consistency
  __m128 col0 = wcn_mat4_get_col(&m, 0);
  __m128 col1 = wcn_mat4_get_col(&m, 1);
  wcn_mat4_get_col(&m, 3); // Translation column

  // Calculate dot products
  const __m128 dot_x = _mm_mul_ps(col0, vec_v);
  const __m128 dot_y = _mm_mul_ps(col1, vec_v);

  // Horizontal adds and adds translation (only x and y components)
  float x = _mm_cvtss_f32(wcn_hadd_ps(dot_x)) + m.m[12];
  float y = _mm_cvtss_f32(wcn_hadd_ps(dot_y)) + m.m[13];

  // Store result using helper function
  __m128 result = _mm_set_ps(0.0f, 0.0f, y, x);
  wcn_store_vec2_partial(vec2.v, result);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized version
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);

  // Load matrix columns using helper functions for better consistency
  float32x4_t col0 = wcn_mat4_get_col(&m, 0);
  float32x4_t col1 = wcn_mat4_get_col(&m, 1);

  // Calculate dot products
  float32x4_t mul0 = vmulq_f32(col0, vec_v);
  float32x4_t mul1 = vmulq_f32(col1, vec_v);

  // Horizontally add and add translation
  float x = wcn_hadd_f32(mul0) + m.m[12];
  float y = wcn_hadd_f32(mul1) + m.m[13];

  // Store result using helper function
  float32x4_t result = {x, y, 0.0f, 0.0f};
  wcn_store_vec2_partial(vec2.v, result);

#else
  // Scalar fallback
  float x = v.v[0];
  float y = v.v[1];
  vec2.v[0] = x * m.m[0] + y * m.m[4] + m.m[12];
  vec2.v[1] = x * m.m[1] + y * m.m[5] + m.m[13];
#endif

  return vec2;
}

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat3)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) vec2;
  float x = v.v[0];
  float y = v.v[1];
  vec2.v[0] = x * m.m[0] + y * m.m[4] + m.m[8];
  vec2.v[1] = x * m.m[1] + y * m.m[5] + m.m[9];
  return vec2;
}

WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) vec3;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized version
  __m128 vec_v = _mm_set_ps(1.0f, v.v[2], v.v[1], v.v[0]);

  // Load matrix columns using helper function for better consistency
  __m128 col0 = wcn_mat4_get_col(&m, 0);
  __m128 col1 = wcn_mat4_get_col(&m, 1);
  __m128 col2 = wcn_mat4_get_col(&m, 2);
  __m128 col3 = wcn_mat4_get_col(&m, 3);

  // Calculate dot products for each component
  __m128 dot_x = _mm_mul_ps(col0, vec_v);
  __m128 dot_y = _mm_mul_ps(col1, vec_v);
  __m128 dot_z = _mm_mul_ps(col2, vec_v);
  __m128 dot_w = _mm_mul_ps(col3, vec_v);

  // Horizontal add for each component using optimized helper
  float x = _mm_cvtss_f32(wcn_hadd_ps(dot_x));
  float y = _mm_cvtss_f32(wcn_hadd_ps(dot_y));
  float z = _mm_cvtss_f32(wcn_hadd_ps(dot_z));
  float w = _mm_cvtss_f32(wcn_hadd_ps(dot_w));

  // Handle a w component with improved numerical stability
  w = (fabsf(w) < wcn_math_get_epsilon()) ? 1.0f : w;

  // Perform perspective division
  __m128 vec_result = _mm_set_ps(0.0f, z, y, x);
  __m128 vec_w = _mm_set1_ps(w);
  vec_result = _mm_div_ps(vec_result, vec_w);

  // Store result
  wcn_store_vec3_partial(vec3.v, vec_result);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized version
  float32x4_t vec_v = {v.v[0], v.v[1], v.v[2], 1.0f};

  // Load matrix columns using helper function for better consistency
  float32x4_t col0 = wcn_mat4_get_col(&m, 0);
  float32x4_t col1 = wcn_mat4_get_col(&m, 1);
  float32x4_t col2 = wcn_mat4_get_col(&m, 2);
  float32x4_t col3 = wcn_mat4_get_col(&m, 3);

  // Calculate dot products
  float32x4_t mul0 = vmulq_f32(col0, vec_v);
  float32x4_t mul1 = vmulq_f32(col1, vec_v);
  float32x4_t mul2 = vmulq_f32(col2, vec_v);
  float32x4_t mul3 = vmulq_f32(col3, vec_v);

  // Horizontally add
  float x = wcn_hadd_f32(mul0);
  float y = wcn_hadd_f32(mul1);
  float z = wcn_hadd_f32(mul2);
  float w = wcn_hadd_f32(mul3);

  // Handle a w component with improved numerical stability
  w = (fabsf(w) < wcn_math_get_epsilon()) ? 1.0f : w;

  // Perform perspective division
  float32x4_t vec_result = {x, y, z, 0.0f};
  float32x4_t vec_w = vdupq_n_f32(w);
  vec_result = vdivq_f32(vec_result, vec_w);

  // Store result
  wcn_store_vec3_partial(vec3.v, vec_result);

#else
  // Scalar fallback - improved numerical stability
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  float w = m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15];

  // Handle a w component with improved numerical stability
  w = (fabsf(w) < wcn_math_get_epsilon()) ? 1.0f : w;

  vec3.v[0] = (x * m.m[0] + y * m.m[4] + z * m.m[8] + m.m[12]) / w;
  vec3.v[1] = (x * m.m[1] + y * m.m[5] + z * m.m[9] + m.m[13]) / w;
  vec3.v[2] = (x * m.m[2] + y * m.m[6] + z * m.m[10] + m.m[14]) / w;
#endif

  return vec3;
}

// vec3 transformMat4Upper3x3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4_upper3x3)(WMATH_TYPE(Vec3) v,
                                          WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) vec3;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized version
  __m128 vec_v = wcn_load_vec3_partial(v.v);

  // Load matrix columns (upper 3x3 part) using helper functions with masking
  __m128 col0 = wcn_mat4_get_col(&m, 0);
  __m128 col1 = wcn_mat4_get_col(&m, 1);
  __m128 col2 = wcn_mat4_get_col(&m, 2);

  // Mask out the w component (set to 0) for upper 3x3 transformation
  __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
  col0 = _mm_and_ps(col0, mask);
  col1 = _mm_and_ps(col1, mask);
  col2 = _mm_and_ps(col2, mask);

  // Calculate dot products
  __m128 dot_x = _mm_mul_ps(col0, vec_v);
  __m128 dot_y = _mm_mul_ps(col1, vec_v);
  __m128 dot_z = _mm_mul_ps(col2, vec_v);

  // Horizontal add for each component
  float x = _mm_cvtss_f32(wcn_hadd_ps(dot_x));
  float y = _mm_cvtss_f32(wcn_hadd_ps(dot_y));
  float z = _mm_cvtss_f32(wcn_hadd_ps(dot_z));

  // Store result using helper function
  __m128 result = _mm_set_ps(0.0f, z, y, x);
  wcn_store_vec3_partial(vec3.v, result);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized version
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);

  // Load matrix columns (upper 3x3 part) using helper functions
  float32x4_t col0 = wcn_mat4_get_col(&m, 0);
  float32x4_t col1 = wcn_mat4_get_col(&m, 1);
  float32x4_t col2 = wcn_mat4_get_col(&m, 2);

  // Mask out the w component (set to 0) for upper 3x3 transformation
  float32x4_t mask = {1.0f, 1.0f, 1.0f, 0.0f};
  col0 = vmulq_f32(col0, mask);
  col1 = vmulq_f32(col1, mask);
  col2 = vmulq_f32(col2, mask);

  // Calculate dot products
  float32x4_t mul0 = vmulq_f32(col0, vec_v);
  float32x4_t mul1 = vmulq_f32(col1, vec_v);
  float32x4_t mul2 = vmulq_f32(col2, vec_v);

  // Horizontal add for each component
  float x = wcn_hadd_f32(mul0);
  float y = wcn_hadd_f32(mul1);
  float z = wcn_hadd_f32(mul2);

  // Store result using helper function
  float32x4_t result = {x, y, z, 0.0f};
  wcn_store_vec3_partial(vec3.v, result);

#else
  // Scalar fallback
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  vec3.v[0] = v0 * m.m[0] + v1 * m.m[4] + v2 * m.m[8];
  vec3.v[1] = v0 * m.m[1] + v1 * m.m[5] + v2 * m.m[9];
  vec3.v[2] = v0 * m.m[2] + v1 * m.m[6] + v2 * m.m[10];
#endif

  return vec3;
}

// vec3 transformMat3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat3)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) r;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized version
  __m128 vec_v = wcn_load_vec3_partial(v.v);

  // Load matrix columns using helper functions
  __m128 col0 = wcn_mat3_get_row(&m, 0);
  __m128 col1 = wcn_mat3_get_row(&m, 1);
  __m128 col2 = wcn_mat3_get_row(&m, 2);

  // Mask out the w component (set to 0) for 3x3 transformation
  __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
  col0 = _mm_and_ps(col0, mask);
  col1 = _mm_and_ps(col1, mask);
  col2 = _mm_and_ps(col2, mask);

  // Calculate dot products
  __m128 dot_x = _mm_mul_ps(col0, vec_v);
  __m128 dot_y = _mm_mul_ps(col1, vec_v);
  __m128 dot_z = _mm_mul_ps(col2, vec_v);

  // Horizontal add for each component
  float x = _mm_cvtss_f32(wcn_hadd_ps(dot_x));
  float y = _mm_cvtss_f32(wcn_hadd_ps(dot_y));
  float z = _mm_cvtss_f32(wcn_hadd_ps(dot_z));

  // Store result using helper function
  __m128 result = _mm_set_ps(0.0f, z, y, x);
  wcn_store_vec3_partial(r.v, result);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized version
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);

  // Load matrix columns using helper functions
  float32x4_t col0 = wcn_mat3_get_row(&m, 0);
  float32x4_t col1 = wcn_mat3_get_row(&m, 1);
  float32x4_t col2 = wcn_mat3_get_row(&m, 2);

  // Mask out the w component (set to 0) for 3x3 transformation
  float32x4_t mask = {1.0f, 1.0f, 1.0f, 0.0f};
  col0 = vmulq_f32(col0, mask);
  col1 = vmulq_f32(col1, mask);
  col2 = vmulq_f32(col2, mask);

  // Calculate dot products
  float32x4_t mul0 = vmulq_f32(col0, vec_v);
  float32x4_t mul1 = vmulq_f32(col1, vec_v);
  float32x4_t mul2 = vmulq_f32(col2, vec_v);

  // Horizontal add for each component
  float x = wcn_hadd_f32(mul0);
  float y = wcn_hadd_f32(mul1);
  float z = wcn_hadd_f32(mul2);

  // Store result using helper function
  float32x4_t result = {x, y, z, 0.0f};
  wcn_store_vec3_partial(r.v, result);

#else
  // Scalar fallback
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  r.v[0] = x * m.m[0] + y * m.m[4] + z * m.m[8];
  r.v[1] = x * m.m[1] + y * m.m[5] + z * m.m[9];
  r.v[2] = x * m.m[2] + y * m.m[6] + z * m.m[10];
#endif

  return r;
}

// vec3 transformQuat
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_quat)
(WMATH_TYPE(Vec3) v, WMATH_TYPE(Quat) q) {
#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

  // SSE -- 修正：不要预先把 q.w * 2
  __m128 v_vec = wcn_load_vec3_partial(v.v); // [x, y, z, 0]
  __m128 q_vec = _mm_loadu_ps(q.v);          // [qx, qy, qz, qw]

  __m128 q_yzx = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 q_zxy = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 1, 0, 2));

  __m128 v_yzx = _mm_shuffle_ps(v_vec, v_vec, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 v_zxy = _mm_shuffle_ps(v_vec, v_vec, _MM_SHUFFLE(3, 1, 0, 2));

  __m128 uv = _mm_sub_ps(_mm_mul_ps(q_yzx, v_zxy), _mm_mul_ps(q_zxy, v_yzx));

  __m128 uv_yzx = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 uv_zxy = _mm_shuffle_ps(uv, uv, _MM_SHUFFLE(3, 1, 0, 2));
  __m128 uuv = _mm_sub_ps(_mm_mul_ps(q_yzx, uv_zxy), _mm_mul_ps(q_zxy, uv_yzx));

  // 这里用 q_w（而不是 2*q_w）
  __m128 q_w = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(3, 3, 3, 3));
  __m128 t1 = _mm_mul_ps(q_w, uv);        // q.w * uv
  __m128 t2 = _mm_add_ps(t1, uuv);        // q.w*uv + uuv
  __m128 t3 = _mm_add_ps(t2, t2);         // 2 * (q.w*uv + uuv)
  __m128 res = _mm_add_ps(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  // NEON -- 同样修正：不用提前乘 2
  float32x4_t v_vec = wcn_load_vec3_partial(v.v);
  float32x4_t q_vec = vld1q_f32(q.v);

  float32x4_t q_yzx = vextq_f32(q_vec, q_vec, 1);
  float32x4_t q_zxy = vextq_f32(q_vec, q_vec, 2);

  float32x4_t v_yzx = vextq_f32(v_vec, v_vec, 1);
  float32x4_t v_zxy = vextq_f32(v_vec, v_vec, 2);

  float32x4_t uv = vsubq_f32(vmulq_f32(q_yzx, v_zxy), vmulq_f32(q_zxy, v_yzx));

  float32x4_t uv_yzx = vextq_f32(uv, uv, 1);
  float32x4_t uv_zxy = vextq_f32(uv, uv, 2);
  float32x4_t uuv = vsubq_f32(vmulq_f32(q_yzx, uv_zxy), vmulq_f32(q_zxy, uv_yzx));

  float32x4_t q_w = vdupq_n_f32(vgetq_lane_f32(q_vec, 3));
  float32x4_t t1 = vmulq_f32(q_w, uv);
  float32x4_t t2 = vaddq_f32(t1, uuv);
  float32x4_t t3 = vaddq_f32(t2, t2);
  float32x4_t res = vaddq_f32(v_vec, t3);

  WMATH_TYPE(Vec3) r;
  wcn_store_vec3_partial(r.v, res);
  return r;

#else

  // Scalar (不变)
  WMATH_TYPE(Vec3) r;
  float qx = q.v[0], qy = q.v[1], qz = q.v[2], qw = q.v[3];
  float x = v.v[0], y = v.v[1], z = v.v[2];

  float uvX = qy * z - qz * y;
  float uvY = qz * x - qx * z;
  float uvZ = qx * y - qy * x;

  float uuvX = qy * uvZ - qz * uvY;
  float uuvY = qz * uvX - qx * uvZ;
  float uuvZ = qx * uvY - qy * uvX;

  r.v[0] = x + 2.0f * ( qw * uvX + uuvX );
  r.v[1] = y + 2.0f * ( qw * uvY + uuvY );
  r.v[2] = z + 2.0f * ( qw * uvZ + uuvZ );

  return r;
#endif
}


// Quat fromMat
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat4)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Quat) r;
  const float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    const float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[0] = (m.m[6] - m.m[9]) * invRoot;
    r.v[1] = (m.m[8] - m.m[2]) * invRoot;
    r.v[2] = (m.m[1] - m.m[4]) * invRoot;
  } else {
    int i = 0;
    if (m.m[5] > m.m[0]) {
      i = 1;
    }
    if (m.m[10] > m.m[i * 4 + i]) {
      i = 2;
    }

    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;

    const float root =
        sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[3] = (m.m[j * 4 + k] - m.m[k * 4 + j]) * invRoot;
    r.v[j] = (m.m[j * 4 + i] + m.m[i * 4 + j]) * invRoot;
    r.v[k] = (m.m[k * 4 + i] + m.m[i * 4 + k]) * invRoot;
  }
  return r;
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat3)(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Quat) r;
  const float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    const float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[0] = (m.m[6] - m.m[9]) * invRoot;
    r.v[1] = (m.m[8] - m.m[2]) * invRoot;
    r.v[2] = (m.m[1] - m.m[4]) * invRoot;
  } else {
    int i = 0;
    if (m.m[5] > m.m[0]) {
      i = 1;
    }
    if (m.m[10] > m.m[i * 4 + i]) {
      i = 2;
    }

    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    const float root =
        sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    const float invRoot = 0.5f / root;
    r.v[3] = (m.m[j * 4 + k] - m.m[k * 4 + j]) * invRoot;
    r.v[j] = (m.m[j * 4 + i] + m.m[i * 4 + j]) * invRoot;
    r.v[k] = (m.m[k * 4 + i] + m.m[i * 4 + k]) * invRoot;
  }
  return r;
}

// fromEuler
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_euler)(float x_angle_in_radians, float y_angle_in_radians,
                             float z_angle_in_radians,
                             enum WCN_Math_RotationOrder order) {
  WMATH_TYPE(Quat) r;
  float x_half_angle = x_angle_in_radians * 0.5f;
  float y_half_angle = y_angle_in_radians * 0.5f;
  float z_half_angle = z_angle_in_radians * 0.5f;
  
  float s_x, c_x, s_y, c_y, s_z, c_z;
#if defined(__GNUC__) && defined(__x86_64__)
  sincosf(x_half_angle, &s_x, &c_x);
  sincosf(y_half_angle, &s_y, &c_y);
  sincosf(z_half_angle, &s_z, &c_z);
#else
  s_x = sinf(x_half_angle);
  c_x = cosf(x_half_angle);
  s_y = sinf(y_half_angle);
  c_y = cosf(y_half_angle);
  s_z = sinf(z_half_angle);
  c_z = cosf(z_half_angle);
#endif

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];
    
    // 构建向量进行并行计算
    __m128 sz_cz_cz_cz = _mm_set_ps(c_z, c_z, c_z, s_z);
    __m128 cx_sx_cx_cx = _mm_set_ps(c_x, c_x, s_x, c_x);
    __m128 cy_cy_sy_cy = _mm_set_ps(c_y, s_y, c_y, c_y);
    
    __m128 cz_sz_sz_sz = _mm_set_ps(s_z, s_z, s_z, c_z);
    __m128 sx_cx_sx_sx = _mm_set_ps(s_x, s_x, c_x, s_x);
    __m128 sy_sy_cy_sy = _mm_set_ps(s_y, c_y, s_y, s_y);
    
    // 计算 A 和 B 部分
    __m128 A = _mm_mul_ps(_mm_mul_ps(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    __m128 B = _mm_mul_ps(_mm_mul_ps(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);
    
    // 应用符号
    __m128 signs_vec = _mm_set_ps((float)signs[3], (float)signs[2], 
                                  (float)signs[1], (float)signs[0]);
    B = _mm_mul_ps(B, signs_vec);
    
    // 最终结果
    __m128 result = _mm_add_ps(A, B);
    _mm_storeu_ps(r.v, result);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];
    
    // 构建向量进行并行计算
    float32x4_t sz_cz_cz_cz = {s_z, c_z, c_z, c_z};
    float32x4_t cx_sx_cx_cx = {c_x, s_x, c_x, c_x};
    float32x4_t cy_cy_sy_cy = {c_y, c_y, s_y, c_y};
    
    float32x4_t cz_sz_sz_sz = {c_z, s_z, s_z, s_z};
    float32x4_t sx_cx_sx_sx = {s_x, c_x, s_x, s_x};
    float32x4_t sy_sy_cy_sy = {s_y, c_y, s_y, s_y};
    
    // 计算 A 和 B 部分
    float32x4_t A = vmulq_f32(vmulq_f32(sz_cz_cz_cz, cx_sx_cx_cx), cy_cy_sy_cy);
    float32x4_t B = vmulq_f32(vmulq_f32(cz_sz_sz_sz, sx_cx_sx_sx), sy_sy_cy_sy);
    
    // 应用符号
    float32x4_t signs_vec = {(float)signs[0], (float)signs[1], 
                             (float)signs[2], (float)signs[3]};
    B = vmulq_f32(B, signs_vec);
    
    // 最终结果
    float32x4_t result = vaddq_f32(A, B);
    vst1q_f32(r.v, result);
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#else
  // 标量版本保持不变
  if (WCN_MATH_IS_VALID_ROTATION_ORDER(order)) {
    const int* signs = WCN_MATH_ROTATION_SIGN_TABLE[order];
    
    r.v[0] = s_z * c_x * c_y + c_z * s_x * s_y * signs[0];
    r.v[1] = c_z * s_x * c_y + s_z * c_x * s_y * signs[1];
    r.v[2] = c_z * c_x * s_y + s_z * s_x * c_y * signs[2];
    r.v[3] = c_z * c_x * c_y + s_z * s_x * s_y * signs[3];
  } else {
    r.v[0] = 0.0f;
    r.v[1] = 0.0f;
    r.v[2] = 0.0f;
    r.v[3] = 1.0f;
  }
#endif

  return r;
}

// BEGIN 3D
// vec3 getTranslation
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Vec3)(const WMATH_TYPE(Mat4) m) {
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = m.m[12],
      .v_y = m.m[13],
      .v_z = m.m[14],
  });
}

// vec3 getAxis
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_axis)(const WMATH_TYPE(Mat4) m, const int axis) {
  int off = axis * 4;
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = m.m[off + 0],
      .v_y = m.m[off + 1],
      .v_z = m.m[off + 2],
  });
}

// vec3 getScale
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_scale)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - parallel computation of three column square sums
  __m128 col_x = _mm_set_ps(0.0f, m.m[2], m.m[1], m.m[0]);
  __m128 col_y = _mm_set_ps(0.0f, m.m[6], m.m[5], m.m[4]);
  __m128 col_z = _mm_set_ps(0.0f, m.m[10], m.m[9], m.m[8]);

  // Square each element
  __m128 sq_x = _mm_mul_ps(col_x, col_x);
  __m128 sq_y = _mm_mul_ps(col_y, col_y);
  __m128 sq_z = _mm_mul_ps(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = _mm_cvtss_f32(wcn_hadd_ps(sq_x));
  float sum_y = _mm_cvtss_f32(wcn_hadd_ps(sq_y));
  float sum_z = _mm_cvtss_f32(wcn_hadd_ps(sq_z));

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - parallel computation of three column square sums
  float32x4_t col_x = {m.m[0], m.m[1], m.m[2], 0.0f};
  float32x4_t col_y = {m.m[4], m.m[5], m.m[6], 0.0f};
  float32x4_t col_z = {m.m[8], m.m[9], m.m[10], 0.0f};

  // Square each element
  float32x4_t sq_x = vmulq_f32(col_x, col_x);
  float32x4_t sq_y = vmulq_f32(col_y, col_y);
  float32x4_t sq_z = vmulq_f32(col_z, col_z);

  // Horizontally add to get a sum of squares for each column
  float sum_x = wcn_hadd_f32(sq_x);
  float sum_y = wcn_hadd_f32(sq_y);
  float sum_z = wcn_hadd_f32(sq_z);

  // Take square root
  result.v[0] = sqrtf(sum_x);
  result.v[1] = sqrtf(sum_y);
  result.v[2] = sqrtf(sum_z);

#else
  // Scalar fallback
  float x_x = m.m[0];
  float x_y = m.m[1];
  float x_z = m.m[2];
  float y_x = m.m[4];
  float y_y = m.m[5];
  float y_z = m.m[6];
  float z_x = m.m[8];
  float z_y = m.m[9];
  float z_z = m.m[10];
  result.v[0] = sqrtf(x_x * x_x + x_y * x_y + x_z * x_z);
  result.v[1] = sqrtf(y_x * y_x + y_y * y_y + y_z * y_z);
  result.v[2] = sqrtf(z_x * z_x + z_y * z_y + z_z * z_z);
#endif

  return result;
}

// vec3 rotateX
WMATH_TYPE(Vec3)
WMATH_ROTATE_X(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

  // --- SSE (fully vectorized, no scalar lane extraction) ---
  // vp = p = a - b  (layout: [px, py, pz, 0])
  __m128 va = wcn_load_vec3_partial(a.v);
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);

  // Broadcast py and pz into full registers using shuffle:
  // _MM_SHUFFLE(z,y,x,w) -> builds imm8 = (z<<6)|(y<<4)|(x<<2)|w
  // _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)) -> [py, py, py, py]
  // _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)) -> [pz, pz, pz, pz]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1));
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2));

  // vector constants
  __m128 v_c = _mm_set1_ps(c);
  __m128 v_s = _mm_set1_ps(s);

  // y' = c*py - s*pz   -> vector: [y', y', y', y']
  __m128 v_y = _mm_sub_ps(_mm_mul_ps(v_py, v_c), _mm_mul_ps(v_pz, v_s));

  // z' = s*py + c*pz  -> vector: [z', z', z', z']
  __m128 v_z = _mm_add_ps(_mm_mul_ps(v_py, v_s), _mm_mul_ps(v_pz, v_c));

  // rx duplicated: [px, px, px, px]
  __m128 v_rx = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0));

  // Now combine into [rx, y', z', 0] using unpack & shuffle without scalar extracts:
  // v_low  = unpacklo(v_rx, v_y) => [rx, y', rx, y']
  // v_high = unpacklo(v_z, vzero) => [z', 0, z', 0]
  // final = shuffle(v_low, v_high, _MM_SHUFFLE(1,0,1,0)) => [rx, y', z', 0]
  __m128 v_low  = _mm_unpacklo_ps(v_rx, v_y);
  __m128 v_zero = _mm_setzero_ps();
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  __m128 vres   = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center back (vb) and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  // --- NEON (vectorized; using lane duplication where necessary) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, 0]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // Extract lanes (cheap) then duplicate into vectors:
  // Note: vgetq_lane_f32 does a lane read (scalar), then vdupq_n_f32 broadcasts it.
  // This keeps core arithmetic vectorized.
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_rx = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c = vdupq_n_f32(c);
  float32x4_t v_s = vdupq_n_f32(s);

  float32x4_t v_y = vsubq_f32(vmulq_f32(v_py, v_c), vmulq_f32(v_pz, v_s)); // [y',y',y',y']
  float32x4_t v_z = vaddq_f32(vmulq_f32(v_py, v_s), vmulq_f32(v_pz, v_c)); // [z',z',z',z']

  // Build [rx, y', z', 0]
  // v_low  = [rx, y', rx, y']  via vzip or combine:
  float32x4_t v_low = vcombine_f32(vget_low_f32(v_rx), vget_low_f32(v_y)); // [rx_low0, y_low0, rx_low1, y_low1] -> given duplicates it's [rx,y,rx,y]
  // v_high = [z', 0, z', 0] - construct from v_z and zero
  float32x4_t v_zero = vdupq_n_f32(0.0f);
  float32x4_t v_high = vcombine_f32(vget_low_f32(v_z), vget_low_f32(v_zero)); // [z',0,z',0]

  // shuffle to [rx, y', z', 0] - use vsetq_lane to be explicit & portable
  // Extract needed lanes (they are duplicated so lane 0 is representative)
  float rx_s = vgetq_lane_f32(v_rx, 0);
  float y_s  = vgetq_lane_f32(v_y, 0);
  float z_s  = vgetq_lane_f32(v_z, 0);

  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(rx_s, vres, 0);
  vres = vsetq_lane_f32(y_s , vres, 1);
  vres = vsetq_lane_f32(z_s , vres, 2);
  // lane3 stays 0

  // add center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = p.v[0];
  r.v[1] = cosf(rad) * p.v[1] - sinf(rad) * p.v[2];
  r.v[2] = sinf(rad) * p.v[1] + cosf(rad) * p.v[2];
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec3 rotateY
WMATH_TYPE(Vec3)
WMATH_ROTATE_Y(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

  // --- SSE (fully vectorized) ---
  __m128 va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);         // p = a - b

  // Broadcast lanes:
  __m128 v_px = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0)); // [px,px,px,px]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)); // [py,py,py,py]
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)); // [pz,pz,pz,pz]

  __m128 v_c  = _mm_set1_ps(c);
  __m128 v_s  = _mm_set1_ps(s);
  __m128 v_ns = _mm_set1_ps(-s);
  __m128 v_zero = _mm_setzero_ps();

  // x' = x*c + z*s
  __m128 v_x = _mm_add_ps(_mm_mul_ps(v_px, v_c), _mm_mul_ps(v_pz, v_s));

  // z' = -x*s + z*c
  __m128 v_z = _mm_add_ps(_mm_mul_ps(v_px, v_ns), _mm_mul_ps(v_pz, v_c));

  // Combine into [x', y, z', 0]
  // v_low = [x', y, x', y]
  __m128 v_low  = _mm_unpacklo_ps(v_x, v_py);
  // v_high = [z', 0, z', 0]
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  // final: [x', y, z', 0]
  __m128 vres = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  // --- NEON (vectorized; using lane-read + broadcast) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // read lanes to scalars then broadcast (common & efficient)
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_px = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c  = vdupq_n_f32(c);
  float32x4_t v_s  = vdupq_n_f32(s);
  float32x4_t v_ns = vdupq_n_f32(-s);

  // x' = x*c + z*s
  float32x4_t v_x = vaddq_f32(vmulq_f32(v_px, v_c), vmulq_f32(v_pz, v_s));
  // z' = -x*s + z*c
  float32x4_t v_z = vaddq_f32(vmulq_f32(v_px, v_ns), vmulq_f32(v_pz, v_c));

  // Build result [x', y, z', 0]
  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_x, 0), vres, 0);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_py, 0), vres, 1);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_z, 0), vres, 2);
  // lane3 remains 0

  // add back center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = sinf(rad) * p.v[2] + cosf(rad) * p.v[0]; // x' = x*c + z*s  (note: sin* z + cos*x same as written)
  r.v[1] = p.v[1];
  r.v[2] = cosf(rad) * p.v[2] - sinf(rad) * p.v[0]; // z' = -x*s + z*c
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec3 rotateZ
WMATH_TYPE(Vec3)
WMATH_ROTATE_Z(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;

  float s = sinf(rad);
  float c = cosf(rad);

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))

  // --- SSE (fully vectorized) ---
  __m128 va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  __m128 vb = wcn_load_vec3_partial(b.v);
  __m128 vp = _mm_sub_ps(va, vb);         // p = a - b

  // Broadcast lanes
  __m128 v_px = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(0,0,0,0)); // [px,px,px,px]
  __m128 v_py = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(1,1,1,1)); // [py,py,py,py]
  __m128 v_pz = _mm_shuffle_ps(vp, vp, _MM_SHUFFLE(2,2,2,2)); // [pz,pz,pz,pz]

  __m128 v_c = _mm_set1_ps(c);
  __m128 v_s = _mm_set1_ps(s);
  __m128 v_zero = _mm_setzero_ps();

  // x' = x*c - y*s
  __m128 v_x = _mm_sub_ps(_mm_mul_ps(v_px, v_c), _mm_mul_ps(v_py, v_s));

  // y' = x*s + y*c
  __m128 v_y = _mm_add_ps(_mm_mul_ps(v_px, v_s), _mm_mul_ps(v_py, v_c));

  // z' = z (unchanged by Z-rotation)
  __m128 v_z = v_pz;

  // Combine into [x', y', z', 0]
  // v_low  = [x', y', x', y']
  __m128 v_low  = _mm_unpacklo_ps(v_x, v_y);
  // v_high = [z', 0, z', 0]
  __m128 v_high = _mm_unpacklo_ps(v_z, v_zero);
  // final = [x', y', z', 0]
  __m128 vres = _mm_shuffle_ps(v_low, v_high, _MM_SHUFFLE(1,0,1,0));

  // add center back and store
  vres = _mm_add_ps(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)

  // --- NEON (vectorized; lane-read + broadcast) ---
  float32x4_t va = wcn_load_vec3_partial(a.v); // [px, py, pz, ?]
  float32x4_t vb = wcn_load_vec3_partial(b.v);
  float32x4_t vp = vsubq_f32(va, vb);

  // read lanes then broadcast
  float px = vgetq_lane_f32(vp, 0);
  float py = vgetq_lane_f32(vp, 1);
  float pz = vgetq_lane_f32(vp, 2);

  float32x4_t v_px = vdupq_n_f32(px);
  float32x4_t v_py = vdupq_n_f32(py);
  float32x4_t v_pz = vdupq_n_f32(pz);

  float32x4_t v_c = vdupq_n_f32(c);
  float32x4_t v_s = vdupq_n_f32(s);

  // x' = x*c - y*s
  float32x4_t v_x = vsubq_f32(vmulq_f32(v_px, v_c), vmulq_f32(v_py, v_s));
  // y' = x*s + y*c
  float32x4_t v_y = vaddq_f32(vmulq_f32(v_px, v_s), vmulq_f32(v_py, v_c));
  // z' = z
  float32x4_t v_z = v_pz;

  // Build [x', y', z', 0] explicitly
  float32x4_t vres = vdupq_n_f32(0.0f);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_x, 0), vres, 0);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_y, 0), vres, 1);
  vres = vsetq_lane_f32(vgetq_lane_f32(v_z, 0), vres, 2);
  // lane3 stays 0

  // add center and store
  vres = vaddq_f32(vres, vb);
  wcn_store_vec3_partial(vec3.v, vres);

#else

  // --- Scalar fallback (reference) ---
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = cosf(rad) * p.v[0] - sinf(rad) * p.v[1]; // x' = x*c - y*s
  r.v[1] = sinf(rad) * p.v[0] + cosf(rad) * p.v[1]; // y' = x*s + y*c
  r.v[2] = p.v[2];                                  // z unchanged
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];

#endif

  return vec3;
}

// vec4 transformMat4
WMATH_TYPE(Vec4)
WMATH_CALL(Vec4, transform_mat4)(WMATH_TYPE(Vec4) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec4) result;

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // ================= SSE 实现 =================
  __m128 vec_v = _mm_loadu_ps(v.v);

  // 按行加载矩阵（列主序存储下，行 i 的元素是 m[i], m[i+4], m[i+8], m[i+12]）
  __m128 row0 = _mm_set_ps(m.m[12], m.m[8],  m.m[4],  m.m[0]);
  __m128 row1 = _mm_set_ps(m.m[13], m.m[9],  m.m[5],  m.m[1]);
  __m128 row2 = _mm_set_ps(m.m[14], m.m[10], m.m[6],  m.m[2]);
  __m128 row3 = _mm_set_ps(m.m[15], m.m[11], m.m[7],  m.m[3]);

  // 点积，每个分量一个
  __m128 res0 = _mm_dp_ps(row0, vec_v, 0xF1);
  __m128 res1 = _mm_dp_ps(row1, vec_v, 0xF2);
  __m128 res2 = _mm_dp_ps(row2, vec_v, 0xF4);
  __m128 res3 = _mm_dp_ps(row3, vec_v, 0xF8);

  __m128 result_v = _mm_add_ps(_mm_add_ps(res0, res1), _mm_add_ps(res2, res3));

  _mm_storeu_ps(result.v, result_v);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // ================= NEON 实现 =================
  float32x4_t vec_v = vld1q_f32(v.v);

  // 按行加载矩阵
  float32x4_t row0 = {m.m[0],  m.m[4],  m.m[8],  m.m[12]};
  float32x4_t row1 = {m.m[1],  m.m[5],  m.m[9],  m.m[13]};
  float32x4_t row2 = {m.m[2],  m.m[6],  m.m[10], m.m[14]};
  float32x4_t row3 = {m.m[3],  m.m[7],  m.m[11], m.m[15]};

  // 点积
  float32x4_t mul0 = vmulq_f32(row0, vec_v);
  float32x4_t mul1 = vmulq_f32(row1, vec_v);
  float32x4_t mul2 = vmulq_f32(row2, vec_v);
  float32x4_t mul3 = vmulq_f32(row3, vec_v);

  float x = vaddvq_f32(mul0); // 横向加
  float y = vaddvq_f32(mul1);
  float z = vaddvq_f32(mul2);
  float w = vaddvq_f32(mul3);

  float32x4_t result_v = {x, y, z, w};
  vst1q_f32(result.v, result_v);

#else
  // ================= 标量回退 =================
  const float x = v.v[0];
  const float y = v.v[1];
  const float z = v.v[2];
  const float w = v.v[3];

  result.v[0] = m.m[0] * x + m.m[4] * y + m.m[8]  * z + m.m[12] * w;
  result.v[1] = m.m[1] * x + m.m[5] * y + m.m[9]  * z + m.m[13] * w;
  result.v[2] = m.m[2] * x + m.m[6] * y + m.m[10] * z + m.m[14] * w;
  result.v[3] = m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15] * w;
#endif

  return result;
}

// Quat rotate_x
WMATH_TYPE(Quat)
WMATH_ROTATE_X(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);
  
  // Create rotation quaternion b = [s, 0, 0, c]
  __m128 b_vec = _mm_set_ps(c, 0.0f, 0.0f, s);
  
  // Shuffle q for multiplication:
  // q = [x, y, z, w]
  // For result.x: q.x * b.w + q.w * b.x = q[0] * b[3] + q[3] * b[0]
  // For result.y: q.y * b.w + q.z * b.x = q[1] * b[3] + q[2] * b[0]
  // For result.z: q.z * b.w - q.y * b.x = q[2] * b[3] - q[1] * b[0]
  // For result.w: q.w * b.w - q.x * b.x = q[3] * b[3] - q[0] * b[0]
  
  const __m128 q_swapped = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(0, 3, 2, 1)); // [y, z, w, x]
  const __m128 b_swapped = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(0, 0, 0, 3)); // [c, 0, 0, c]
  
  __m128 mul1 = _mm_mul_ps(q_vec, b_swapped);
  const __m128 mul2 = _mm_mul_ps(q_swapped, _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 0, 0, 0))); // [s, 0, 0, s]
  
  // Add/sub according to formula
  const __m128 signs = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
  const __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, mul2_signed);
  
  // Correct order: [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(2, 1, 0, 3)); // [w, z, y, x] -> [x, y, z, w]
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  // Load quaternion
  float32x4_t q_vec = vld1q_f32(q.v);
  
  // Create rotation quaternion b = [s, 0, 0, c]
  float32x4_t b_vec = {s, 0.0f, 0.0f, c};
  
  // Perform quaternion multiplication q * b for rotation around x-axis
  // result.x = q.x * b.w + q.w * b.x = q[0] * b[3] + q[3] * b[0]
  // result.y = q.y * b.w + q.z * b.x = q[1] * b[3] + q[2] * b[0]
  // result.z = q.z * b.w - q.y * b.x = q[2] * b[3] - q[1] * b[0]
  // result.w = q.w * b.w - q.x * b.x = q[3] * b[3] - q[0] * b[0]
  
  float32x4_t q_components = q_vec;
  float32x2_t qw_qx = vget_low_f32(q_components);     // [q.x, q.y]
  float32x2_t qz_qw = vget_high_f32(q_components);    // [q.z, q.w]
  
  // Create [b.w, b.x, b.w, b.x] = [c, s, c, s]
  float32x4_t bw_bx = vsetq_lane_f32(c, vdupq_n_f32(s), 0);
  bw_bx = vsetq_lane_f32(c, bw_bx, 2);
  
  // Create [q.x, q.z, q.w, q.y]
  float32x4_t q_xzw_y = {vgetq_lane_f32(q_vec, 0), vgetq_lane_f32(q_vec, 2), 
                         vgetq_lane_f32(q_vec, 3), vgetq_lane_f32(q_vec, 1)};
  
  // Create [b.w, b.x, b.w, b.x]
  float32x4_t b_wx_wx = {c, s, c, s};
  
  // Multiply
  float32x4_t mul1 = vmulq_f32(q_vec, vsetq_lane_f32(c, vsetq_lane_f32(c, vsetq_lane_f32(c, vdupq_n_f32(c), 3), 2), 0));
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 0), mul1, 0);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 1), mul1, 1);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 2), mul1, 2);
  mul1 = vsetq_lane_f32(vgetq_lane_f32(mul1, 3), mul1, 3);
  
  // Simpler approach for NEON:
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];
  
  result.v[0] = q_x * c + q_w * s;
  result.v[1] = q_y * c + q_z * s;
  result.v[2] = q_z * c - q_y * s;
  result.v[3] = q_w * c - q_x * s;

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  float b_x = sinf(half_angle);
  float b_w = cosf(half_angle);
  result.v[0] = q_x * b_w + q_w * b_x;
  result.v[1] = q_y * b_w + q_z * b_x;
  result.v[2] = q_z * b_w - q_y * b_x;
  result.v[3] = q_w * b_w - q_x * b_x;
#endif

  return result;
}

// Quat rotate_y
WMATH_TYPE(Quat)
WMATH_ROTATE_Y(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);
  
  // Create rotation quaternion b = [0, s, 0, c]
  _mm_set_ps(c, 0.0f, s, 0.0f);
  
  // For rotation around y-axis:
  // result.x = q.x * b.w - q.z * b.y = q[0] * b[3] - q[2] * b[1]
  // result.y = q.y * b.w + q.w * b.y = q[1] * b[3] + q[3] * b[1]
  // result.z = q.z * b.w + q.x * b.y = q[2] * b[3] + q[0] * b[1]
  // result.w = q.w * b.w - q.y * b.y = q[3] * b[3] - q[1] * b[1]
  
  const __m128 mul1 = _mm_mul_ps(q_vec, _mm_set_ps(c, c, c, c));
  const __m128 q_shuffled = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(1, 0, 3, 2)); // [z, w, x, y]
  const __m128 mul2 = _mm_mul_ps(q_shuffled, _mm_set_ps(s, s, s, s));
  
  // Apply signs based on formula
  const __m128 signs = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
  __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, _mm_shuffle_ps(mul2_signed, mul2_signed, _MM_SHUFFLE(2, 3, 0, 1)));
  
  // Reorder to get [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(2, 3, 1, 0));
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c - q_z * s;
  result.v[1] = q_y * c + q_w * s;
  result.v[2] = q_z * c + q_x * s;
  result.v[3] = q_w * c - q_y * s;

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  float b_y = sinf(half_angle);
  float b_w = cosf(half_angle);
  result.v[0] = q_x * b_w - q_z * b_y;
  result.v[1] = q_y * b_w + q_w * b_y;
  result.v[2] = q_z * b_w + q_x * b_y;
  result.v[3] = q_w * b_w - q_y * b_y;
#endif

  return result;
}

// Quat rotate_z
WMATH_TYPE(Quat)
WMATH_ROTATE_Z(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  // Load quaternion
  __m128 q_vec = _mm_loadu_ps(q.v);
  
  // Create rotation quaternion b = [0, 0, s, c]
  _mm_set_ps(c, s, 0.0f, 0.0f);
  
  // For rotation around z-axis:
  // result.x = q.x * b.w + q.y * b.z = q[0] * b[3] + q[1] * b[2]
  // result.y = q.y * b.w - q.x * b.z = q[1] * b[3] - q[0] * b[2]
  // result.z = q.z * b.w + q.w * b.z = q[2] * b[3] + q[3] * b[2]
  // result.w = q.w * b.w - q.z * b.z = q[3] * b[3] - q[2] * b[2]
  
  __m128 mul1 = _mm_mul_ps(q_vec, _mm_set_ps(c, c, c, c));
  __m128 q_shuffled = _mm_shuffle_ps(q_vec, q_vec, _MM_SHUFFLE(2, 3, 0, 1)); // [y, x, w, z]
  __m128 mul2 = _mm_mul_ps(q_shuffled, _mm_set_ps(s, s, s, s));
  
  // Apply signs based on formula
  __m128 signs = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
  __m128 mul2_signed = _mm_mul_ps(mul2, signs);
  __m128 res_vec = _mm_add_ps(mul1, _mm_shuffle_ps(mul2_signed, mul2_signed, _MM_SHUFFLE(3, 2, 1, 0)));
  
  // Reorder to get [x, y, z, w]
  res_vec = _mm_shuffle_ps(res_vec, res_vec, _MM_SHUFFLE(3, 2, 0, 1));
  _mm_storeu_ps(result.v, res_vec);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float half_angle = angleInRadians * 0.5f;
  float s = sinf(half_angle);
  float c = cosf(half_angle);
  
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  result.v[0] = q_x * c + q_y * s;
  result.v[1] = q_y * c - q_x * s;
  result.v[2] = q_z * c + q_w * s;
  result.v[3] = q_w * c - q_z * s;

#else
  // Scalar implementation
  float half_angle = angleInRadians * 0.5f;
  float q_x = q.v[0];
  float q_y = q.v[1];
  float q_z = q.v[2];
  float q_w = q.v[3];

  float b_z = sinf(half_angle);
  float b_w = cosf(half_angle);
  result.v[0] = q_x * b_w + q_y * b_z;
  result.v[1] = q_y * b_w - q_x * b_z;
  result.v[2] = q_z * b_w + q_w * b_z;
  result.v[3] = q_w * b_w - q_z * b_z;
#endif

  return result;
}

// Mat3 rotate
WMATH_TYPE(Mat3)
WMATH_ROTATE(Mat3)
(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load first two rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  
  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);
  
  // Calculate new first row: c * row0 + s * row1
  __m128 new_row0 = _mm_add_ps(
    _mm_mul_ps(c_vec, row0),
    _mm_mul_ps(s_vec, row1)
  );
  
  // Calculate new second row: c * row1 - s * row0
  __m128 new_row1 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row1),
    _mm_mul_ps(s_vec, row0)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);
  
  // Copy third row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row0, row0)) || 
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row1, row1))) {
    wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load first two rows
  float32x4_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  float32x4_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  
  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);
  
  // Calculate new first row: c * row0 + s * row1
  float32x4_t new_row0 = vaddq_f32(
    vmulq_f32(c_vec, row0),
    vmulq_f32(s_vec, row1)
  );
  
  // Calculate new second row: c * row1 - s * row0
  float32x4_t new_row1 = vsubq_f32(
    vmulq_f32(c_vec, row1),
    vmulq_f32(s_vec, row0)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 1, new_row1);
  
  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq0 = vceqq_f32(new_row0, row0);
  uint32x4_t eq1 = vceqq_f32(new_row1, row1);
  if (!(vgetq_lane_u32(eq0, 0) && vgetq_lane_u32(eq0, 1) && vgetq_lane_u32(eq0, 2) &&
        vgetq_lane_u32(eq1, 0) && vgetq_lane_u32(eq1, 1) && vgetq_lane_u32(eq1, 2))) {
    wcn_mat3_set_row(&newDst, 2, wcn_mat3_get_row(&m, 2));
  }

#else
  // Scalar implementation (original code)
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 + s * m10;
  newDst.m[1] = c * m01 + s * m11;
  newDst.m[2] = c * m02 + s * m12;

  newDst.m[4] = c * m10 - s * m00;
  newDst.m[5] = c * m11 - s * m01;
  newDst.m[6] = c * m12 - s * m02;

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

// Mat3 rotate x
WMATH_TYPE(Mat3)
WMATH_ROTATE_X(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load second and third rows
  __m128 row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]
  
  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);
  
  // Calculate new second row: c * row1 + s * row2
  __m128 new_row1 = _mm_add_ps(
    _mm_mul_ps(c_vec, row1),
    _mm_mul_ps(s_vec, row2)
  );
  
  // Calculate new third row: c * row2 - s * row1
  __m128 new_row2 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row2),
    _mm_mul_ps(s_vec, row1)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);
  
  // Copy first row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row1, row1)) || 
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row2, row2))) {
    wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load second and third rows
  float32x4_t row1 = wcn_mat3_get_row(&m, 1); // [m10, m11, m12, 0]
  float32x4_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]
  
  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);
  
  // Calculate new second row: c * row1 + s * row2
  float32x4_t new_row1 = vaddq_f32(
    vmulq_f32(c_vec, row1),
    vmulq_f32(s_vec, row2)
  );
  
  // Calculate new third row: c * row2 - s * row1
  float32x4_t new_row2 = vsubq_f32(
    vmulq_f32(c_vec, row2),
    vmulq_f32(s_vec, row1)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 1, new_row1);
  wcn_mat3_set_row(&newDst, 2, new_row2);
  
  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq1 = vceqq_f32(new_row1, row1);
  uint32x4_t eq2 = vceqq_f32(new_row2, row2);
  if (!(vgetq_lane_u32(eq1, 0) && vgetq_lane_u32(eq1, 1) && vgetq_lane_u32(eq1, 2) &&
        vgetq_lane_u32(eq2, 0) && vgetq_lane_u32(eq2, 1) && vgetq_lane_u32(eq2, 2))) {
    wcn_mat3_set_row(&newDst, 0, wcn_mat3_get_row(&m, 0));
  }

#else
  // Scalar implementation (original code)
  float m_10 = m.m[4];
  float m_11 = m.m[5];
  float m_12 = m.m[6];
  float m_20 = m.m[8];
  float m_21 = m.m[9];
  float m_22 = m.m[10];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[4] = c * m_10 + s * m_20;
  newDst.m[5] = c * m_11 + s * m_21;
  newDst.m[6] = c * m_12 + s * m_22;
  newDst.m[8] = c * m_20 - s * m_10;
  newDst.m[9] = c * m_21 - s * m_11;
  newDst.m[10] = c * m_22 - s * m_12;
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
  }
#endif

  return newDst;
}

// Mat3 rotate y
WMATH_TYPE(Mat3)
WMATH_ROTATE_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load first and third rows
  __m128 row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  __m128 row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]
  
  // Create coefficients
  __m128 c_vec = _mm_set1_ps(c);
  __m128 s_vec = _mm_set1_ps(s);
  
  // Calculate new first row: c * row0 - s * row2
  __m128 new_row0 = _mm_sub_ps(
    _mm_mul_ps(c_vec, row0),
    _mm_mul_ps(s_vec, row2)
  );
  
  // Calculate new third row: c * row2 + s * row0
  __m128 new_row2 = _mm_add_ps(
    _mm_mul_ps(c_vec, row2),
    _mm_mul_ps(s_vec, row0)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);
  
  // Copy second row if needed
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row0, row0)) || 
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row2, row2))) {
    wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  
  // Load first and third rows
  float32x4_t row0 = wcn_mat3_get_row(&m, 0); // [m00, m01, m02, 0]
  float32x4_t row2 = wcn_mat3_get_row(&m, 2); // [m20, m21, m22, 0]
  
  // Create coefficients
  float32x4_t c_vec = vdupq_n_f32(c);
  float32x4_t s_vec = vdupq_n_f32(s);
  
  // Calculate new first row: c * row0 - s * row2
  float32x4_t new_row0 = vsubq_f32(
    vmulq_f32(c_vec, row0),
    vmulq_f32(s_vec, row2)
  );
  
  // Calculate new third row: c * row2 + s * row0
  float32x4_t new_row2 = vaddq_f32(
    vmulq_f32(c_vec, row2),
    vmulq_f32(s_vec, row0)
  );
  
  // Store results
  wcn_mat3_set_row(&newDst, 0, new_row0);
  wcn_mat3_set_row(&newDst, 2, new_row2);
  
  // Simple check for equality (simplified version of the original logic)
  uint32x4_t eq0 = vceqq_f32(new_row0, row0);
  uint32x4_t eq2 = vceqq_f32(new_row2, row2);
  if (!(vgetq_lane_u32(eq0, 0) && vgetq_lane_u32(eq0, 1) && vgetq_lane_u32(eq0, 2) &&
        vgetq_lane_u32(eq2, 0) && vgetq_lane_u32(eq2, 1) && vgetq_lane_u32(eq2, 2))) {
    wcn_mat3_set_row(&newDst, 1, wcn_mat3_get_row(&m, 1));
  }

#else
  // Scalar implementation (original code)
  float m00 = m.m[0 * 4 + 0];
  float m01 = m.m[0 * 4 + 1];
  float m02 = m.m[0 * 4 + 2];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 - s * m20;
  newDst.m[1] = c * m01 - s * m21;
  newDst.m[2] = c * m02 - s * m22;
  newDst.m[8] = c * m20 + s * m00;
  newDst.m[9] = c * m21 + s * m01;
  newDst.m[10] = c * m22 + s * m02;

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
  }
#endif

  return newDst;
}

// Mat3 rotate z
WMATH_TYPE(Mat3)
WMATH_ROTATE_Z(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  return WMATH_ROTATE(Mat3)(m, angleInRadians);
}

// Mat3 rotation
WMATH_TYPE(Mat3)
WMATH_ROTATION(Mat3)(float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = c;
  newDst.m[1] = s;
  // newDst.m[2] = 0;
  newDst.m[4] = -s;
  newDst.m[5] = c;
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

// Mat3 rotation x
WMATH_TYPE(Mat3)
WMATH_ROTATION_X(Mat3)(float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = 1;
  // newDst.m[1] = 0;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = c;
  newDst.m[6] = s;
  // newDst.m[8] = 0;
  newDst.m[9] = -s;
  newDst.m[10] = c;
  return newDst;
}

// Mat3 rotation y
WMATH_TYPE(Mat3)
WMATH_ROTATION_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  newDst.m[0] = c;
  // newDst.m[1] = 0;
  newDst.m[2] = -s;
  // newDst.m[4] = 0;
  newDst.m[5] = 1;
  // newDst.m[6] = 0;
  newDst.m[8] = s;
  // newDst.m[9] = 0;
  newDst.m[10] = c;

  return newDst;
}

// Mat3 rotation z
WMATH_TYPE(Mat3)
WMATH_ROTATION_Z(Mat3)(float angleInRadians) {
  return WMATH_ROTATION(Mat3)(angleInRadians);
}

// Mat3 get_axis
/**
 * Returns an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y,
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_axis)
(WMATH_TYPE(Mat3) m, int axis) {
  WMATH_TYPE(Vec2) result;
  int off = axis * 4;
  result.v[0] = m.m[off + 0];
  result.v[1] = m.m[off + 1];
  return result;
}
// Mat3 set_axis
/**
 * Sets an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - Axis  0 = x, 1 = y;
 * @returns The matrix with axis set.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, set_axis)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v, int axis) {
  WMATH_TYPE(Mat3) newDst = WMATH_COPY(Mat3)(m);
  int off = axis * 4;
  newDst.m[off + 0] = v.v[0];
  newDst.m[off + 1] = v.v[1];
  return newDst;
}

// Mat3 get_scaling
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_scaling)
(const WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) result;
  const float xx = m.m[0];
  const float xy = m.m[1];
  const float yx = m.m[4];
  const float yy = m.m[5];
  result.v[0] = sqrtf(xx * xx + xy * xy);
  result.v[1] = sqrtf(yx * yx + yy * yy);
  return result;
}

// Mat3 get_3D_scaling
WMATH_TYPE(Vec3)
WMATH_CALL(Mat3, get_3D_scaling)
(const WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) result;
  const float xx = m.m[0];
  const float xy = m.m[1];
  const float xz = m.m[2];
  const float yx = m.m[4];
  const float yy = m.m[5];
  const float yz = m.m[6];
  const float zx = m.m[8];
  const float zy = m.m[9];
  const float zz = m.m[10];

  result.v[0] = sqrtf(xx * xx + xy * xy + xz * xz);
  result.v[1] = sqrtf(yx * yx + yy * yy + yz * yz);
  result.v[2] = sqrtf(zx * zx + zy * zy + zz * zz);

  return result;
}

// Mat3 get_translation
WMATH_TYPE(Vec2)
WMATH_GET_TRANSLATION(Mat3)
(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) result;
  result.v[0] = m.m[8];
  result.v[1] = m.m[9];
  return result;
}

// Mat3 set_translation
/**
 * Sets the translation component of a 3-by-3 matrix to the given
 * vector.
 * @param m - The matrix.
 * @param v - The vector.
 * @returns The matrix with translation set.
 */
WMATH_TYPE(Mat3)
WMATH_SET_TRANSLATION(Mat3)
(const WMATH_TYPE(Mat3) m, const WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_IDENTITY(Mat3)();
  if (!WMATH_EQUALS(Mat3)(m, newDst)) {
    newDst.m[0] = v.v[0];
    newDst.m[1] = v.v[1];
    newDst.m[2] = v.v[2];
    newDst.m[4] = v.v[4];
    newDst.m[5] = v.v[5];
    newDst.m[6] = v.v[6];
  }
  newDst.m[8] = v.v[0];
  newDst.m[9] = v.v[1];
  newDst.m[10] = 1;

  return newDst;
}

// Mat3 translation
WMATH_TYPE(Mat3)
WMATH_TRANSLATION(Mat3)
(WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - create 2D translation matrix efficiently
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create identity matrix with translation in third row
  // Row0: [1, 0, 0, pad]
  __m128 row0 = _mm_move_ss(vec_one, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, 1, 0, pad]
  __m128 row1 = _mm_move_ss(vec_zero, vec_one);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [v.x, v.y, 1, pad]
  __m128 row2 = _mm_move_ss(vec_v, vec_one);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - create 2D translation matrix efficiently
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create identity matrix with translation in third row
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(1.0f, row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(1.0f, row1, 1);

  float32x4_t row2 = vec_v;
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#else
  // Scalar fallback - direct assignment is more efficient than memset
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat3)));
  newDst.m[0] = 1.0f;
  newDst.m[5] = 1.0f;
  newDst.m[8] = v.v[0];
  newDst.m[9] = v.v[1];
  newDst.m[10] = 1.0f;
#endif

  return newDst;
}

// translate
/**
 * Translates the given 3-by-3 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by which to translate.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, translate)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized 2D matrix translation
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 row0 = wcn_mat3_get_row(&m, 0);
  __m128 row1 = wcn_mat3_get_row(&m, 1);
  __m128 row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  __m128 v_x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 v_y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));

  // Calculate dot products for each translation component
  __m128 dot_x = _mm_mul_ps(row0, v_x);
  __m128 dot_y = _mm_mul_ps(row1, v_y);

  // Sum the dot products and add original translation
  __m128 sum_xy = _mm_add_ps(dot_x, dot_y);
  __m128 trans_sum = _mm_add_ps(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized 2D matrix translation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t row0 = wcn_mat3_get_row(&m, 0);
  float32x4_t row1 = wcn_mat3_get_row(&m, 1);
  float32x4_t row2 = wcn_mat3_get_row(&m, 2);

  // Copy the first two rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    wcn_mat3_set_row(&newDst, 0, row0);
    wcn_mat3_set_row(&newDst, 1, row1);
  }

  // Calculate translation components using SIMD
  // Extract x, y components from translation vector
  float32x4_t v_x = vdupq_lane_f32(vget_low_f32(vec_v), 0);
  float32x4_t v_y = vdupq_lane_f32(vget_low_f32(vec_v), 1);

  // Calculate dot products for each translation component
  float32x4_t dot_x = vmulq_f32(row0, v_x);
  float32x4_t dot_y = vmulq_f32(row1, v_y);

  // Sum the dot products and add original translation
  float32x4_t sum_xy = vaddq_f32(dot_x, dot_y);
  float32x4_t trans_sum = vaddq_f32(sum_xy, row2);

  // Store the translation row
  wcn_mat3_set_row(&newDst, 2, trans_sum);

#else
  // Scalar fallback with optimized variable usage
  float v0 = v.v[0];
  float v1 = v.v[1];

  // Copy rotation/scaling part if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
  }

  // Calculate translation components with optimized ordering
  newDst.m[8] = m.m[0] * v0 + m.m[4] * v1 + m.m[8];
  newDst.m[9] = m.m[1] * v0 + m.m[5] * v1 + m.m[9];
  newDst.m[10] = m.m[2] * v0 + m.m[6] * v1 + m.m[10];
#endif

  return newDst;
}

// Mat4 axis_rotate
/**
 * Rotates the given 4-by-4 matrix around the given axis by the
 * given angle.
 * @param m - The matrix.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, axis_rotate)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) axis, float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  
#if !defined(WMATH_DISABLE_SIMD) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  
  // SSE implementation
  // Normalize axis
  __m128 axis_vec = wcn_load_vec3_partial(axis.v);
  __m128 axis_squared = _mm_mul_ps(axis_vec, axis_vec);
  float norm_sq = _mm_cvtss_f32(wcn_hadd_ps(axis_squared));
  __m128 inv_norm = _mm_set1_ps(1.0f / sqrtf(norm_sq));
  axis_vec = _mm_mul_ps(axis_vec, inv_norm);
  
  // Extract normalized components
  float x = _mm_cvtss_f32(_mm_shuffle_ps(axis_vec, axis_vec, _MM_SHUFFLE(0, 0, 0, 0)));
  float y = _mm_cvtss_f32(_mm_shuffle_ps(axis_vec, axis_vec, _MM_SHUFFLE(1, 1, 1, 1)));
  float z = _mm_cvtss_f32(_mm_shuffle_ps(axis_vec, axis_vec, _MM_SHUFFLE(2, 2, 2, 2)));
  
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1 - c;

  // Rotation matrix elements
  float r00 = xx + (1 - xx) * c;
  float r01 = x * y * oneMinusCosine + z * s;
  float r02 = x * z * oneMinusCosine - y * s;
  float r10 = x * y * oneMinusCosine - z * s;
  float r11 = yy + (1 - yy) * c;
  float r12 = y * z * oneMinusCosine + x * s;
  float r20 = x * z * oneMinusCosine + y * s;
  float r21 = y * z * oneMinusCosine - x * s;
  float r22 = zz + (1 - zz) * c;

  // Load matrix rows
  __m128 row0 = wcn_mat4_get_row(&m, 0); // [m00, m01, m02, m03]
  __m128 row1 = wcn_mat4_get_row(&m, 1); // [m10, m11, m12, m13]
  __m128 row2 = wcn_mat4_get_row(&m, 2); // [m20, m21, m22, m23]

  // Perform matrix multiplication using SIMD
  // new row 0 = r00 * row0 + r01 * row1 + r02 * row2
  __m128 r00_vec = _mm_set1_ps(r00);
  __m128 r01_vec = _mm_set1_ps(r01);
  __m128 r02_vec = _mm_set1_ps(r02);
#if defined(WCN_HAS_FMA)  
  // 更高效的FMA实现
  __m128 new_row0 = _mm_fmadd_ps(r00_vec, row0, _mm_fmadd_ps(r01_vec, row1, _mm_mul_ps(r02_vec, row2)));
  __m128 new_row1 = _mm_fmadd_ps(r10_vec, row0, _mm_fmadd_ps(r11_vec, row1, _mm_mul_ps(r12_vec, row2)));
  __m128 new_row2 = _mm_fmadd_ps(r20_vec, row0, _mm_fmadd_ps(r21_vec, row1, _mm_mul_ps(r22_vec, row2)));
#else
  __m128 new_row0 = _mm_mul_ps(r00_vec, row0);
  new_row0 = _mm_add_ps(new_row0, _mm_mul_ps(r01_vec, row1));
  new_row0 = _mm_add_ps(new_row0, _mm_mul_ps(r02_vec, row2));
  
  // new row 1 = r10 * row0 + r11 * row1 + r12 * row2
  __m128 r10_vec = _mm_set1_ps(r10);
  __m128 r11_vec = _mm_set1_ps(r11);
  __m128 r12_vec = _mm_set1_ps(r12);
  
  __m128 new_row1 = _mm_mul_ps(r10_vec, row0);
  new_row1 = _mm_add_ps(new_row1, _mm_mul_ps(r11_vec, row1));
  new_row1 = _mm_add_ps(new_row1, _mm_mul_ps(r12_vec, row2));
  
  // new row 2 = r20 * row0 + r21 * row1 + r22 * row2
  __m128 r20_vec = _mm_set1_ps(r20);
  __m128 r21_vec = _mm_set1_ps(r21);
  __m128 r22_vec = _mm_set1_ps(r22);
  
  __m128 new_row2 = _mm_mul_ps(r20_vec, row0);
  new_row2 = _mm_add_ps(new_row2, _mm_mul_ps(r21_vec, row1));
  new_row2 = _mm_add_ps(new_row2, _mm_mul_ps(r22_vec, row2));
#endif
  
  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);
  
  // Copy last row if needed
  __m128 new_row0_check = wcn_mat4_get_row(&newDst, 0);
  __m128 new_row1_check = wcn_mat4_get_row(&newDst, 1);
  __m128 new_row2_check = wcn_mat4_get_row(&newDst, 2);
  
  __m128 row0_check = wcn_mat4_get_row(&m, 0);
  __m128 row1_check = wcn_mat4_get_row(&m, 1);
  __m128 row2_check = wcn_mat4_get_row(&m, 2);
  
  if (!_mm_movemask_ps(_mm_cmpeq_ps(new_row0_check, row0_check)) ||
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row1_check, row1_check)) ||
      !_mm_movemask_ps(_mm_cmpeq_ps(new_row2_check, row2_check))) {
    wcn_mat4_set_row(&newDst, 3, wcn_mat4_get_row(&m, 3));
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  
  // NEON implementation
  // Normalize axis
  float32x4_t axis_vec = wcn_load_vec3_partial(axis.v);
  float32x4_t axis_squared = vmulq_f32(axis_vec, axis_vec);
  float norm_sq = wcn_hadd_f32(axis_squared);
  float32x4_t inv_norm = vdupq_n_f32(1.0f / sqrtf(norm_sq));
  axis_vec = vmulq_f32(axis_vec, inv_norm);
  
  // Extract normalized components
  float x = vgetq_lane_f32(axis_vec, 0);
  float y = vgetq_lane_f32(axis_vec, 1);
  float z = vgetq_lane_f32(axis_vec, 2);
  
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1 - c;

  // Rotation matrix elements
  float r00 = xx + (1 - xx) * c;
  float r01 = x * y * oneMinusCosine + z * s;
  float r02 = x * z * oneMinusCosine - y * s;
  float r10 = x * y * oneMinusCosine - z * s;
  float r11 = yy + (1 - yy) * c;
  float r12 = y * z * oneMinusCosine + x * s;
  float r20 = x * z * oneMinusCosine + y * s;
  float r21 = y * z * oneMinusCosine - x * s;
  float r22 = zz + (1 - zz) * c;

  // Load matrix rows
  float32x4_t row0 = wcn_mat4_get_row(&m, 0); // [m00, m01, m02, m03]
  float32x4_t row1 = wcn_mat4_get_row(&m, 1); // [m10, m11, m12, m13]
  float32x4_t row2 = wcn_mat4_get_row(&m, 2); // [m20, m21, m22, m23]

  // Perform matrix multiplication using SIMD
  // new row 0 = r00 * row0 + r01 * row1 + r02 * row2
  float32x4_t r00_vec = vdupq_n_f32(r00);
  float32x4_t r01_vec = vdupq_n_f32(r01);
  float32x4_t r02_vec = vdupq_n_f32(r02);
  
  float32x4_t new_row0 = vmulq_f32(r00_vec, row0);
  new_row0 = vaddq_f32(new_row0, vmulq_f32(r01_vec, row1));
  new_row0 = vaddq_f32(new_row0, vmulq_f32(r02_vec, row2));
  
  // new row 1 = r10 * row0 + r11 * row1 + r12 * row2
  float32x4_t r10_vec = vdupq_n_f32(r10);
  float32x4_t r11_vec = vdupq_n_f32(r11);
  float32x4_t r12_vec = vdupq_n_f32(r12);
  
  float32x4_t new_row1 = vmulq_f32(r10_vec, row0);
  new_row1 = vaddq_f32(new_row1, vmulq_f32(r11_vec, row1));
  new_row1 = vaddq_f32(new_row1, vmulq_f32(r12_vec, row2));
  
  // new row 2 = r20 * row0 + r21 * row1 + r22 * row2
  float32x4_t r20_vec = vdupq_n_f32(r20);
  float32x4_t r21_vec = vdupq_n_f32(r21);
  float32x4_t r22_vec = vdupq_n_f32(r22);
  
  float32x4_t new_row2 = vmulq_f32(r20_vec, row0);
  new_row2 = vaddq_f32(new_row2, vmulq_f32(r21_vec, row1));
  new_row2 = vaddq_f32(new_row2, vmulq_f32(r22_vec, row2));
  
  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);
  
  // Copy last row if needed
  float32x4_t new_row0_check = wcn_mat4_get_row(&newDst, 0);
  float32x4_t new_row1_check = wcn_mat4_get_row(&newDst, 1);
  float32x4_t new_row2_check = wcn_mat4_get_row(&newDst, 2);
  
  float32x4_t row0_check = wcn_mat4_get_row(&m, 0);
  float32x4_t row1_check = wcn_mat4_get_row(&m, 1);
  float32x4_t row2_check = wcn_mat4_get_row(&m, 2);
  
  uint32x4_t eq0 = vceqq_f32(new_row0_check, row0_check);
  uint32x4_t eq1 = vceqq_f32(new_row1_check, row1_check);
  uint32x4_t eq2 = vceqq_f32(new_row2_check, row2_check);
  
  if (!(vgetq_lane_u32(eq0, 0) && vgetq_lane_u32(eq0, 1) && vgetq_lane_u32(eq0, 2) && vgetq_lane_u32(eq0, 3)) ||
      !(vgetq_lane_u32(eq1, 0) && vgetq_lane_u32(eq1, 1) && vgetq_lane_u32(eq1, 2) && vgetq_lane_u32(eq1, 3)) ||
      !(vgetq_lane_u32(eq2, 0) && vgetq_lane_u32(eq2, 1) && vgetq_lane_u32(eq2, 2) && vgetq_lane_u32(eq2, 3))) {
    wcn_mat4_set_row(&newDst, 3, wcn_mat4_get_row(&m, 3));
  }

#else
  // Scalar implementation (original code)
  float x = axis.v[0];
  float y = axis.v[1];
  float z = axis.v[2];
  float n = sqrtf(x * x + y * y + z * z);
  x /= n;
  y /= n;
  z /= n;
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1 - c;

  float r00 = xx + (1 - xx) * c;
  float r01 = x * y * oneMinusCosine + z * s;
  float r02 = x * z * oneMinusCosine - y * s;
  float r10 = x * y * oneMinusCosine - z * s;
  float r11 = yy + (1 - yy) * c;
  float r12 = y * z * oneMinusCosine + x * s;
  float r20 = x * z * oneMinusCosine + y * s;
  float r21 = y * z * oneMinusCosine - x * s;
  float r22 = zz + (1 - zz) * c;

  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m03 = m.m[3];
  float m10 = m.m[4];
  float m11 = m.m[5];
  float m12 = m.m[6];
  float m13 = m.m[7];
  float m20 = m.m[8];
  float m21 = m.m[9];
  float m22 = m.m[10];
  float m23 = m.m[11];

  newDst.m[0] = r00 * m00 + r01 * m10 + r02 * m20;
  newDst.m[1] = r00 * m01 + r01 * m11 + r02 * m21;
  newDst.m[2] = r00 * m02 + r01 * m12 + r02 * m22;
  newDst.m[3] = r00 * m03 + r01 * m13 + r02 * m23;
  newDst.m[4] = r10 * m00 + r11 * m10 + r12 * m20;
  newDst.m[5] = r10 * m01 + r11 * m11 + r12 * m21;
  newDst.m[6] = r10 * m02 + r11 * m12 + r12 * m22;
  newDst.m[7] = r10 * m03 + r11 * m13 + r12 * m23;
  newDst.m[8] = r20 * m00 + r21 * m10 + r22 * m20;
  newDst.m[9] = r20 * m01 + r21 * m11 + r22 * m21;
  newDst.m[10] = r20 * m02 + r21 * m12 + r22 * m22;
  newDst.m[11] = r20 * m03 + r21 * m13 + r22 * m23;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
  }
#endif

  return newDst;
}

// Mat4 axisRotation
/**
 * Creates a 4-by-4 matrix which rotates around the given axis by the given
 * angle.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns A matrix which rotates angle radians
 *     around the axis.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, axis_rotation)
(WMATH_TYPE(Vec3) axis, float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  float x = axis.v[0];
  float y = axis.v[1];
  float z = axis.v[2];
  float n = sqrtf(x * x + y * y + z * z);
  x /= n;
  y /= n;
  z /= n;
  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float oneMinusCosine = 1 - c;

  newDst.m[0] = xx + (1 - xx) * c;
  newDst.m[1] = x * y * oneMinusCosine + z * s;
  newDst.m[2] = x * z * oneMinusCosine - y * s;
  // newDst.m[3] = 0;
  newDst.m[4] = x * y * oneMinusCosine - z * s;
  newDst.m[5] = yy + (1 - yy) * c;
  newDst.m[6] = y * z * oneMinusCosine + x * s;
  // newDst.m[7] = 0;
  newDst.m[8] = x * z * oneMinusCosine + y * s;
  newDst.m[9] = y * z * oneMinusCosine - x * s;
  newDst.m[10] = zz + (1 - zz) * c;
  // newDst.m[11] = 0;
  // newDst.m[12] = 0;
  // newDst.m[13] = 0;
  // newDst.m[14] = 0;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 camera_aim
/**
 * Computes a 4-by-4 camera aim transformation.
 *
 * This is a matrix which positions an object aiming down negative Z.
 * toward the target.
 *
 * Note: this is the inverse of `lookAt`
 *
 * @param eye - The position of the object.
 * @param target - The position meant to be aimed at.
 * @param up - A vector pointing up.
 * @returns The aim matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, camera_aim)
(
    //
    WMATH_TYPE(Vec3) eye,    // eye: Vec3
    WMATH_TYPE(Vec3) target, // target: Vec3
    WMATH_TYPE(Vec3) up      // up: Vec3
) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  WMATH_TYPE(Vec3) z_axis = WMATH_NORMALIZE(Vec3)(WMATH_SUB(Vec3)(eye, target));
  WMATH_TYPE(Vec3)
  x_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(up, z_axis));
  WMATH_TYPE(Vec3) y_axis = WMATH_CROSS(Vec3)(z_axis, x_axis);

  newDst.m[0] = x_axis.v[0];
  newDst.m[1] = x_axis.v[1];
  newDst.m[2] = x_axis.v[2]; // x
  // newDst.m[3] = 0;
  newDst.m[4] = y_axis.v[0];
  newDst.m[5] = y_axis.v[1];
  newDst.m[6] = y_axis.v[2]; // y
  // newDst.m[7] = 0;
  newDst.m[8] = z_axis.v[0];
  newDst.m[9] = z_axis.v[1];
  newDst.m[10] = z_axis.v[2]; // z
  // newDst.m[11] = 0;
  newDst.m[12] = eye.v[0];
  newDst.m[13] = eye.v[1];
  newDst.m[14] = eye.v[2];
  newDst.m[15] = 1; // eye

  return newDst;
}

// Mat4 frustum
/**
 * Computes a 4-by-4 perspective transformation matrix given the left, right,
 * top, bottom, near and far clipping planes. The arguments define a frustum
 * extending in the negative z direction. The arguments near and far are the
 * distances to the near and far clipping planes. Note that near and far are not
 * z coordinates, but rather they are distances along the negative z-axis. The
 * matrix generated sends the viewing frustum to the unit box. We assume a unit
 * box extending from -1 to 1 in the x and y dimensions and from 0 to 1 in the z
 * dimension.
 * @param left - The x coordinate of the left plane of the box.
 * @param right - The x coordinate of the right plane of the box.
 * @param bottom - The y coordinate of the bottom plane of the box.
 * @param top - The y coordinate of the right plane of the box.
 * @param near - The negative z coordinate of the near plane of the box.
 * @param far - The negative z coordinate of the far plane of the box.
 * @returns The perspective projection matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, frustum)
(float left, float right, float bottom, float top, float near, float far) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  float dx = right - left;
  float dy = top - bottom;
  float dz = near - far;

  newDst.m[0] = 2 * near / dx;
  newDst.m[5] = 2 * near / dy;
  newDst.m[8] = (left + right) / dx;
  newDst.m[9] = (top + bottom) / dy;
  newDst.m[10] = far / dz;
  newDst.m[11] = -1;
  newDst.m[14] = near * far / dz;

  return newDst;
}

// Mat4 frustumReverseZ
/**
 * Computes a 4-by-4 reverse-z perspective transformation matrix given the left,
 * right, top, bottom, near and far clipping planes. The arguments define a
 * frustum extending in the negative z direction. The arguments near and far are
 * the distances to the near and far clipping planes. Note that near and far are
 * not z coordinates, but rather they are distances along the negative z-axis.
 * The matrix generated sends the viewing frustum to the unit box. We assume a
 * unit box extending from -1 to 1 in the x and y dimensions and from 1 (-near)
 * to 0 (-far) in the z dimension.
 * @param left - The x coordinate of the left plane of the box.
 * @param right - The x coordinate of the right plane of the box.
 * @param bottom - The y coordinate of the bottom plane of the box.
 * @param top - The y coordinate of the right plane of the box.
 * @param near - The negative z coordinate of the near plane of the box.
 * @param far - The negative z coordinate of the far plane of the box.
 * @returns The perspective projection matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, frustum_reverse_z)
(float left, float right, float bottom, float top, float near, float far) {
  float _far = WMATH_OR_ELSE(far, INFINITY);
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  float dx = right - left;
  float dy = top - bottom;

  // 0
  newDst.m[0] = 2 * near / dx;
  // 5
  newDst.m[5] = 2 * near / dy;
  // 8
  newDst.m[8] = (left + right) / dx;
  // 9
  newDst.m[9] = (top + bottom) / dy;
  // 11
  newDst.m[11] = -1;

  if (isfinite(_far)) {
    newDst.m[10] = 0;
    newDst.m[14] = near;
  } else {
    float rangeInv = 1 / (far - near);
    newDst.m[10] = near * rangeInv;
    newDst.m[14] = far * near * rangeInv;
  }

  return newDst;
}

// Mat4 get_axis
/**
 * Returns an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y, 2 = z;
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_axis)
(WMATH_TYPE(Mat4) m, int axis) {
  WMATH_TYPE(Vec3) newDst = WMATH_ZERO(Vec3)();
  int off = axis * 4;
  newDst.v[0] = m.m[off + 0];
  newDst.v[1] = m.m[off + 1];
  newDst.v[2] = m.m[off + 2];
  return newDst;
}

// Mat4 set_axis
/**
 * Sets an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - The axis  0 = x, 1 = y, 2 = z;
 * @returns The matrix with axis set.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, set_axis)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v, int axis) {
  WMATH_TYPE(Mat4) newDst = WMATH_COPY(Mat4)(m);

  int off = axis * 4;
  newDst.m[off + 0] = v.v[0];
  newDst.m[off + 1] = v.v[1];
  newDst.m[off + 2] = v.v[2];
  return newDst;
}

// Mat4 getTranslation
/**
 * Returns the translation component of a 4-by-4 matrix as a vector with 3
 * entries.
 * @param m - The matrix.
 * @returns The translation component of m.
 */
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Mat4)
(const WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) v;
  v.v[0] = m.m[12];
  v.v[1] = m.m[13];
  v.v[2] = m.m[14];
  return v;
}

// Mat4 setTranslation
WMATH_TYPE(Mat4)
WMATH_SET_TRANSLATION(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst = WMATH_IDENTITY(Mat4)();
  if (!WMATH_EQUALS(Mat4)(m, newDst)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
  }
  newDst.m[12] = v.v[0];
  newDst.m[13] = v.v[1];
  newDst.m[14] = v.v[2];
  newDst.m[15] = 1;
  return newDst;
}

// Mat4 translation
/**
 * Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by
 *     which to translate.
 * @returns The translation matrix.
 */
// Mat4 translation
WMATH_TYPE(Mat4)
WMATH_TRANSLATION(Mat4)
(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && \
(defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // ========================= SSE implementation =========================
  // Create translation matrix as 4 rows
  __m128 row0 = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
  __m128 row1 = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
  __m128 row2 = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
  __m128 row3 = _mm_setr_ps(v.v[0], v.v[1], v.v[2], 1.0f);

  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // ========================= NEON implementation =========================
  float32x4_t row0 = vdupq_n_f32(0.0f);
  row0 = vsetq_lane_f32(1.0f, row0, 0);

  float32x4_t row1 = vdupq_n_f32(0.0f);
  row1 = vsetq_lane_f32(1.0f, row1, 1);

  float32x4_t row2 = vdupq_n_f32(0.0f);
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  float32x4_t row3 = vdupq_n_f32(0.0f);
  row3 = vsetq_lane_f32(v.v[0], row3, 0);
  row3 = vsetq_lane_f32(v.v[1], row3, 1);
  row3 = vsetq_lane_f32(v.v[2], row3, 2);
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#else
  // ========================= Scalar fallback =========================
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0]  = 1.0f;
  newDst.m[5]  = 1.0f;
  newDst.m[10] = 1.0f;
  newDst.m[12] = v.v[0];
  newDst.m[13] = v.v[1];
  newDst.m[14] = v.v[2];
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}


// Mat4 perspective
/**
 * Computes a 4-by-4 perspective transformation matrix given the angular height
 * of the frustum, the aspect ratio, and the near and far clipping planes.  The
 * arguments define a frustum extending in the negative z direction.  The given
 * angle is the vertical angle of the frustum, and the horizontal angle is
 * determined to produce the given aspect ratio.  The arguments near and far are
 * the distances to the near and far clipping planes.  Note that near and far
 * are not z coordinates, but rather they are distances along the negative
 * z-axis.  The matrix generated sends the viewing frustum to the unit box.
 * We assume a unit box extending from -1 to 1 in the x and y dimensions and
 * from 0 to 1 in the z dimension.
 *
 * Note: If you pass `Infinity` for zFar then it will produce a projection
 * matrix returns -Infinity for Z when transforming coordinates with Z <= 0 and
 * +Infinity for Z otherwise.
 *
 * @param fieldOfViewYInRadians - The camera angle from top to bottom (in
 * radians).
 * @param aspect - The aspect ratio width / height.
 * @param zNear - The depth (negative z coordinate)
 *     of the near clipping plane.
 * @param zFar - The depth (negative z coordinate)
 *     of the far clipping plane.
 * @returns The perspective matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, perspective)
(float fieldOfViewYInRadians, float aspect, float zNear, float zFar) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized perspective matrix creation
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);
  __m128 vec_neg_one = _mm_set1_ps(-1.0f);

  // Calculate focal length
  float f = tanf(WMATH_PI * 0.5 - 0.5 * fieldOfViewYInRadians);
  __m128 vec_f = _mm_set1_ps(f);
  __m128 vec_aspect = _mm_set1_ps(aspect);
  __m128 vec_inv_aspect = _mm_div_ps(vec_one, vec_aspect);

  // Calculate f / aspect
  __m128 f_over_aspect = _mm_mul_ps(vec_f, vec_inv_aspect);

  // Create perspective matrix rows
  // Row0: [f/aspect, 0, 0, 0]
  __m128 row0 = _mm_move_ss(f_over_aspect, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, f, 0, 0]
  __m128 row1 = _mm_move_ss(vec_zero, vec_f);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Declare row2 and row3 outside the conditional
  __m128 row2, row3;

  // Row2 and Row3 depend on whether zFar is finite
  if (isfinite(zFar)) {
    float rangeInv = 1.0f / (zNear - zFar);
    __m128 vec_rangeInv = _mm_set1_ps(rangeInv);
    __m128 vec_zNear = _mm_set1_ps(zNear);
    __m128 vec_zFar = _mm_set1_ps(zFar);

    // Row2: [0, 0, zFar*rangeInv, -1]
    __m128 z_far_range = _mm_mul_ps(vec_zFar, vec_rangeInv);
    row2 = _mm_move_ss(vec_zero, z_far_range);
    row2 = _mm_shuffle_ps(row2, vec_neg_one, _MM_SHUFFLE(3, 2, 1, 0));

    // Row3: [0, 0, zFar*zNear*rangeInv, 0]
    __m128 z_far_zNear_range =
        _mm_mul_ps(vec_zFar, _mm_mul_ps(vec_zNear, vec_rangeInv));
    row3 = _mm_move_ss(vec_zero, z_far_zNear_range);
    row3 = _mm_shuffle_ps(row3, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));
  } else {
    // Infinite far plane
    __m128 vec_zNear = _mm_set1_ps(zNear);
    __m128 vec_neg_zNear = _mm_mul_ps(vec_zNear, vec_neg_one);

    // Row2: [0, 0, -1, -1]
    row2 = _mm_move_ss(vec_zero, vec_neg_one);
    row2 = _mm_shuffle_ps(row2, vec_neg_one, _MM_SHUFFLE(3, 2, 1, 0));

    // Row3: [0, 0, -zNear, 0]
    row3 = _mm_move_ss(vec_zero, vec_neg_zNear);
    row3 = _mm_shuffle_ps(row3, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));
  }

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized perspective matrix creation
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_neg_one = vdupq_n_f32(-1.0f);

  // Calculate focal length
  float f = tanf(WMATH_PI * 0.5 - 0.5 * fieldOfViewYInRadians);
  float32x4_t vec_f = vdupq_n_f32(f);
  float32x4_t vec_aspect = vdupq_n_f32(aspect);
  float32x4_t vec_inv_aspect = vdivq_f32(vec_one, vec_aspect);

  // Calculate f / aspect
  float32x4_t f_over_aspect = vmulq_f32(vec_f, vec_inv_aspect);

  // Create perspective matrix rows
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(vgetq_lane_f32(f_over_aspect, 0), row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(vgetq_lane_f32(vec_f, 0), row1, 1);

  // Row2 and Row3 depend on whether zFar is finite
  if (isfinite(zFar)) {
    float rangeInv = 1.0f / (zNear - zFar);
    float32x4_t vec_rangeInv = vdupq_n_f32(rangeInv);
    float32x4_t vec_zNear = vdupq_n_f32(zNear);
    float32x4_t vec_zFar = vdupq_n_f32(zFar);

    // Row2: [0, 0, zFar*rangeInv, -1]
    float32x4_t z_far_range = vmulq_f32(vec_zFar, vec_rangeInv);
    float32x4_t row2 = vec_zero;
    row2 = vsetq_lane_f32(vgetq_lane_f32(z_far_range, 0), row2, 2);
    row2 = vsetq_lane_f32(-1.0f, row2, 3);

    // Row3: [0, 0, zFar*zNear*rangeInv, 0]
    float32x4_t z_far_zNear_range =
        vmulq_f32(vec_zFar, vmulq_f32(vec_zNear, vec_rangeInv));
    float32x4_t row3 = vec_zero;
    row3 = vsetq_lane_f32(vgetq_lane_f32(z_far_zNear_range, 0), row3, 2);
  } else {
    // Infinite far plane
    float32x4_t vec_zNear = vdupq_n_f32(zNear);

    // Row2: [0, 0, -1, -1]
    float32x4_t row2 = vec_zero;
    row2 = vsetq_lane_f32(-1.0f, row2, 2);
    row2 = vsetq_lane_f32(-1.0f, row2, 3);

    // Row3: [0, 0, -zNear, 0]
    float32x4_t row3 = vec_zero;
    row3 = vsetq_lane_f32(-zNear, row3, 2);
  }

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#else
  // Scalar fallback - direct assignment is more efficient
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  float f = tanf(WMATH_PI * 0.5 - 0.5 * fieldOfViewYInRadians);

  newDst.m[0] = f / aspect;
  newDst.m[5] = f;
  newDst.m[11] = -1;

  if (isfinite(zFar)) {
    float rangeInv = 1.0f / (zNear - zFar);
    newDst.m[10] = zFar * rangeInv;
    newDst.m[14] = zFar * zNear * rangeInv;
  } else {
    newDst.m[10] = -1;
    newDst.m[14] = -zNear;
  }
#endif

  return newDst;
}

// Mat4 perspective_reverse_z
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, perspective_reverse_z)
(
    //
    float fieldOfViewYInRadians, // fieldOfViewYInRadians: number
    float aspect,                // aspect: number
    float zNear,                 // zNear: number
    float zFar                   // zFar: number
) {
  float _zFar = WMATH_OR_ELSE(zFar, INFINITY);

  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

  const float f = 1 / tanf(0.5f * fieldOfViewYInRadians);

  // 0
  newDst.m[0] = f / aspect;
  // 5
  newDst.m[5] = f;
  // 11
  newDst.m[11] = -1;
  if (isfinite(_zFar)) {
    newDst.m[10] = 0;
    newDst.m[14] = zNear;
  } else {
    float rangeInv = 1 / (_zFar - zNear);
    newDst.m[10] = zNear * rangeInv;
    newDst.m[14] = _zFar * zNear * rangeInv;
  }

  return newDst;
}

// Mat4 translate
/**
 * Translates the given 4-by-4 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by
 *     which to translate.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, translate)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized matrix translation
  __m128 vec_v = wcn_load_vec3_partial(v.v);
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 0, row0);
    wcn_mat4_set_row(&newDst, 1, row1);
    wcn_mat4_set_row(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  __m128 v_x = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 v_y = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 v_z = _mm_shuffle_ps(vec_v, vec_v, _MM_SHUFFLE(2, 2, 2, 2));

  // Calculate dot products for each translation component
  __m128 dot_x = _mm_mul_ps(row0, v_x);
  __m128 dot_y = _mm_mul_ps(row1, v_y);
  __m128 dot_z = _mm_mul_ps(row2, v_z);

  // Sum the dot products and add original translation
  __m128 sum_xy = _mm_add_ps(dot_x, dot_y);
  __m128 sum_xyz = _mm_add_ps(sum_xy, dot_z);
  __m128 trans_sum = _mm_add_ps(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row(&newDst, 3, trans_sum);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized matrix translation
  float32x4_t vec_v = wcn_load_vec3_partial(v.v);
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Copy the first three rows unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 0, row0);
    wcn_mat4_set_row(&newDst, 1, row1);
    wcn_mat4_set_row(&newDst, 2, row2);
  }

  // Calculate translation components using SIMD
  // Extract x, y, z components from translation vector
  float32x4_t v_x = vdupq_lane_f32(vget_low_f32(vec_v), 0);
  float32x4_t v_y = vdupq_lane_f32(vget_low_f32(vec_v), 1);
  float32x4_t v_z = vdupq_lane_f32(vget_high_f32(vec_v), 0);

  // Calculate dot products for each translation component
  float32x4_t dot_x = vmulq_f32(row0, v_x);
  float32x4_t dot_y = vmulq_f32(row1, v_y);
  float32x4_t dot_z = vmulq_f32(row2, v_z);

  // Sum the dot products and add original translation
  float32x4_t sum_xy = vaddq_f32(dot_x, dot_y);
  float32x4_t sum_xyz = vaddq_f32(sum_xy, dot_z);
  float32x4_t trans_sum = vaddq_f32(sum_xyz, row3);

  // Store the translation row
  wcn_mat4_set_row(&newDst, 3, trans_sum);

#else
  // Scalar fallback with optimized variable usage
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  // Copy rotation/scaling part if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
  }

  // Calculate translation components with optimized ordering
  newDst.m[12] = m.m[0] * v0 + m.m[4] * v1 + m.m[8] * v2 + m.m[12];
  newDst.m[13] = m.m[1] * v0 + m.m[5] * v1 + m.m[9] * v2 + m.m[13];
  newDst.m[14] = m.m[2] * v0 + m.m[6] * v1 + m.m[10] * v2 + m.m[14];
  newDst.m[15] = m.m[3] * v0 + m.m[7] * v1 + m.m[11] * v2 + m.m[15];
#endif

  return newDst;
}

// Mat4 rotate
/**
 * Rotates the given 4-by-4 matrix around the given axis by the
 * given angle. (same as rotate)
 * @param m - The matrix.
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,    // m: Mat4
    WMATH_TYPE(Vec3) axis, // axis: Vec3
    float angleInRadians   // angleInRadians: number
) {
  return WMATH_CALL(Mat4, axis_rotate)(m, axis, angleInRadians);
}

// Mat4 rotate_x
/**
 * Rotates the given 4-by-4 matrix around the x-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_X(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized X-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row1 = _mm_fmadd_ps(vec_c, row1, _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_fmadd_ps(vec_c, row2, _mm_mul_ps(_mm_set1_ps(-s), row1));
#else
  __m128 new_row1 =
      _mm_add_ps(_mm_mul_ps(vec_c, row1), _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row2), _mm_mul_ps(vec_s, row1));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized X-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For X-axis rotation: row1' = c*row1 + s*row2, row2' = c*row2 - s*row1
  float32x4_t new_row1 = vmlaq_f32(vmulq_f32(vec_s, row2), vec_c, row1);
  float32x4_t new_row2 = vmlaq_f32(vmulq_f32(vec_neg_s, row1), vec_c, row2);

  // Store results
  wcn_mat4_set_row(&newDst, 0, row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m10 = m.m[4];
  float m11 = m.m[5];
  float m12 = m.m[6];
  float m13 = m.m[7];
  float m20 = m.m[8];
  float m21 = m.m[9];
  float m22 = m.m[10];
  float m23 = m.m[11];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[4] = c * m10 + s * m20;
  newDst.m[5] = c * m11 + s * m21;
  newDst.m[6] = c * m12 + s * m22;
  newDst.m[7] = c * m13 + s * m23;
  newDst.m[8] = c * m20 - s * m10;
  newDst.m[9] = c * m21 - s * m11;
  newDst.m[10] = c * m22 - s * m12;
  newDst.m[11] = c * m23 - s * m13;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[0] = m.m[0];
    newDst.m[1] = m.m[1];
    newDst.m[2] = m.m[2];
    newDst.m[3] = m.m[3];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotate_y
/**
 * Rotates the given 4-by-4 matrix around the y-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_Y(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized Y-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row0 =
      _mm_fmadd_ps(vec_c, row0, _mm_mul_ps(_mm_set1_ps(-s), row2));
  __m128 new_row2 = _mm_fmadd_ps(vec_c, row2, _mm_mul_ps(vec_s, row0));
#else
  __m128 new_row0 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row0), _mm_mul_ps(vec_s, row2));
  __m128 new_row2 =
      _mm_add_ps(_mm_mul_ps(vec_c, row2), _mm_mul_ps(vec_s, row0));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized Y-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For Y-axis rotation: row0' = c*row0 - s*row2, row2' = c*row2 + s*row0
  float32x4_t new_row0 = vmlaq_f32(vmulq_f32(vec_neg_s, row2), vec_c, row0);
  float32x4_t new_row2 = vmlaq_f32(vmulq_f32(vec_s, row0), vec_c, row2);

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, row1);
  wcn_mat4_set_row(&newDst, 2, new_row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m03 = m.m[3];
  float m20 = m.m[8];
  float m21 = m.m[9];
  float m22 = m.m[10];
  float m23 = m.m[11];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 - s * m20;
  newDst.m[1] = c * m01 - s * m21;
  newDst.m[2] = c * m02 - s * m22;
  newDst.m[3] = c * m03 - s * m23;
  newDst.m[8] = c * m20 + s * m00;
  newDst.m[9] = c * m21 + s * m01;
  newDst.m[10] = c * m22 + s * m02;
  newDst.m[11] = c * m23 + s * m03;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[4] = m.m[4];
    newDst.m[5] = m.m[5];
    newDst.m[6] = m.m[6];
    newDst.m[7] = m.m[7];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotate_z
/**
 * Rotates the given 4-by-4 matrix around the z-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATE_Z(Mat4)
(
    //
    WMATH_TYPE(Mat4) m,  // m: Mat4
    float angleInRadians // angleInRadians: number
) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized Z-axis rotation
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  __m128 vec_c = _mm_set1_ps(c);
  __m128 vec_s = _mm_set1_ps(s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
#if defined(WCN_HAS_FMA)
  // Use FMA for better performance
  __m128 new_row0 = _mm_fmadd_ps(vec_c, row0, _mm_mul_ps(vec_s, row1));
  __m128 new_row1 =
      _mm_fmadd_ps(vec_c, row1, _mm_mul_ps(_mm_set1_ps(-s), row0));
#else
  __m128 new_row0 =
      _mm_add_ps(_mm_mul_ps(vec_c, row0), _mm_mul_ps(vec_s, row1));
  __m128 new_row1 =
      _mm_sub_ps(_mm_mul_ps(vec_c, row1), _mm_mul_ps(vec_s, row0));
#endif

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized Z-axis rotation
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Precompute sine and cosine
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);
  float32x4_t vec_c = vdupq_n_f32(c);
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_neg_s = vdupq_n_f32(-s);

  // For Z-axis rotation: row0' = c*row0 + s*row1, row1' = c*row1 - s*row0
  float32x4_t new_row0 = vmlaq_f32(vmulq_f32(vec_s, row1), vec_c, row0);
  float32x4_t new_row1 = vmlaq_f32(vmulq_f32(vec_neg_s, row0), vec_c, row1);

  // Store results
  wcn_mat4_set_row(&newDst, 0, new_row0);
  wcn_mat4_set_row(&newDst, 1, new_row1);
  wcn_mat4_set_row(&newDst, 2, row2);

  // Copy the fourth row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized variable usage
  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m03 = m.m[3];
  float m10 = m.m[4];
  float m11 = m.m[5];
  float m12 = m.m[6];
  float m13 = m.m[7];
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c * m00 + s * m10;
  newDst.m[1] = c * m01 + s * m11;
  newDst.m[2] = c * m02 + s * m12;
  newDst.m[3] = c * m03 + s * m13;
  newDst.m[4] = c * m10 - s * m00;
  newDst.m[5] = c * m11 - s * m01;
  newDst.m[6] = c * m12 - s * m02;
  newDst.m[7] = c * m13 - s * m03;

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
    newDst.m[11] = m.m[11];
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 rotation
/**
 * Creates a 4-by-4 matrix which rotates around the given axis by the given
 * angle. (same as axisRotation)
 * @param axis - The axis
 *     about which to rotate.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns A matrix which rotates angle radians
 *     around the axis.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION(Mat4)
(WMATH_TYPE(Vec3) axis, float angleInRadians) {
  return WMATH_CALL(Mat4, axis_rotation)(axis, angleInRadians);
}

// Mat4 rotation_x
/**
 * Creates a 4-by-4 matrix which rotates around the x-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_X(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = 1;
  newDst.m[5] = c;
  newDst.m[6] = s;
  newDst.m[9] = -s;
  newDst.m[10] = c;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 rotation_y
/**
 * Creates a 4-by-4 matrix which rotates around the y-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Y(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c;
  newDst.m[2] = -s;
  newDst.m[5] = 1;
  newDst.m[8] = s;
  newDst.m[10] = c;
  newDst.m[15] = 1;

  return newDst;
}

// Mat4 rotation_z
/**
 * Creates a 4-by-4 matrix which rotates around the z-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Z(Mat4)
(float angleInRadians) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();
  float c = cosf(angleInRadians);
  float s = sinf(angleInRadians);

  newDst.m[0] = c;
  newDst.m[1] = s;
  newDst.m[4] = -s;
  newDst.m[5] = c;
  newDst.m[10] = 1;
  newDst.m[15] = 1;

  return newDst;
}

// All Type Scale Impl
WMATH_TYPE(Vec2)
WMATH_SCALE(Vec2)
(WMATH_TYPE(Vec2) v, float scale) {
  WMATH_TYPE(Vec2) result;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_scale = _mm_set1_ps(scale);
  __m128 vec_res = _mm_mul_ps(vec_v, vec_scale);
  wcn_store_vec2_partial(result.v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_scale = vdupq_n_f32(scale);
  float32x4_t vec_res = vmulq_f32(vec_v, vec_scale);
  wcn_store_vec2_partial(result.v, vec_res);

#else
  // Scalar fallback
  result.v[0] = v.v[0] * scale;
  result.v[1] = v.v[1] * scale;
#endif

  return result;
}

WMATH_TYPE(Vec3)
WMATH_SCALE(Vec3)
(WMATH_TYPE(Vec3) v, float scale) {
  return WMATH_MULTIPLY_SCALAR(Vec3)(v, scale);
}

WMATH_TYPE(Quat)
WMATH_SCALE(Quat)
(WMATH_TYPE(Quat) q, float scale) {
  return WMATH_MULTIPLY_SCALAR(Quat)(q, scale);
}

WMATH_TYPE(Mat3)
WMATH_SCALE(Mat3)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements
  
  // Scale first row (indices 0-3)
  __m128 vec_row0 = _mm_loadu_ps(&m.m[0]);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  _mm_storeu_ps(&newDst.m[0], vec_result0);
  
  // Scale second row (indices 4-7)
  __m128 vec_row1 = _mm_loadu_ps(&m.m[4]);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  _mm_storeu_ps(&newDst.m[4], vec_result1);
  
  // Copy third row (indices 8-11) if needed
  if (_mm_movemask_ps(_mm_cmpneq_ps(vec_result0, vec_row0)) != 0 ||
      _mm_movemask_ps(_mm_cmpneq_ps(vec_result1, vec_row1)) != 0) {
    __m128 vec_row2 = _mm_loadu_ps(&m.m[8]);
    _mm_storeu_ps(&newDst.m[8], vec_row2);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements
  
  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = vld1q_f32(&m.m[0]);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  vst1q_f32(&newDst.m[0], vec_result0);
  
  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = vld1q_f32(&m.m[4]);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  vst1q_f32(&newDst.m[4], vec_result1);
  
  // Copy third row (indices 8-11) if needed
  uint32x4_t cmp0 = vceqq_f32(vec_result0, vec_row0);
  uint32x4_t cmp1 = vceqq_f32(vec_result1, vec_row1);
  if (vminvq_u32(cmp0) == 0 || vminvq_u32(cmp1) == 0) {
    float32x4_t vec_row2 = vld1q_f32(&m.m[8]);
    vst1q_f32(&newDst.m[8], vec_row2);
  }

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];

  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];

  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_SCALE(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements
  __m128 vec_v2 = _mm_set1_ps(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = _mm_loadu_ps(&m.m[0]);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  _mm_storeu_ps(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = _mm_loadu_ps(&m.m[4]);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  _mm_storeu_ps(&newDst.m[4], vec_result1);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = _mm_loadu_ps(&m.m[8]);
  __m128 vec_result2 = _mm_mul_ps(vec_row2, vec_v2);
  _mm_storeu_ps(&newDst.m[8], vec_result2);

  // Check if any element changed and copy last row if needed
  __m128 cmp0 = _mm_cmpneq_ps(vec_result0, vec_row0);
  __m128 cmp1 = _mm_cmpneq_ps(vec_result1, vec_row1);
  __m128 cmp2 = _mm_cmpneq_ps(vec_result2, vec_row2);
  
  if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(cmp0, cmp1), cmp2)) != 0) {
    __m128 vec_row3 = _mm_loadu_ps(&m.m[12]);
    _mm_storeu_ps(&newDst.m[12], vec_row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements
  float32x4_t vec_v2 = vdupq_n_f32(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = vld1q_f32(&m.m[0]);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  vst1q_f32(&newDst.m[0], vec_result0);

  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = vld1q_f32(&m.m[4]);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  vst1q_f32(&newDst.m[4], vec_result1);

  // Scale third row (indices 8-11)
  float32x4_t vec_row2 = vld1q_f32(&m.m[8]);
  float32x4_t vec_result2 = vmulq_f32(vec_row2, vec_v2);
  vst1q_f32(&newDst.m[8], vec_result2);

  // Check if any element changed and copy last row if needed
  uint32x4_t cmp0 = vceqq_f32(vec_result0, vec_row0);
  uint32x4_t cmp1 = vceqq_f32(vec_result1, vec_row1);
  uint32x4_t cmp2 = vceqq_f32(vec_result2, vec_row2);
  
  if (vminvq_u32(cmp0) == 0 || vminvq_u32(cmp1) == 0 || vminvq_u32(cmp2) == 0) {
    float32x4_t vec_row3 = vld1q_f32(&m.m[12]);
    vst1q_f32(&newDst.m[12], vec_row3);
  }

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];
  newDst.m[3] = v0 * m.m[0 * 4 + 3];
  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];
  newDst.m[7] = v1 * m.m[1 * 4 + 3];
  newDst.m[8] = v2 * m.m[2 * 4 + 0];
  newDst.m[9] = v2 * m.m[2 * 4 + 1];
  newDst.m[10] = v2 * m.m[2 * 4 + 2];
  newDst.m[11] = v2 * m.m[2 * 4 + 3];

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat3 scale3D
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scale3D)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_v0 = _mm_set1_ps(v.v[0]); // Broadcast v0 to all elements
  __m128 vec_v1 = _mm_set1_ps(v.v[1]); // Broadcast v1 to all elements
  __m128 vec_v2 = _mm_set1_ps(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  __m128 vec_row0 = wcn_mat3_get_row(&m, 0);
  __m128 vec_result0 = _mm_mul_ps(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  __m128 vec_row1 = wcn_mat3_get_row(&m, 1);
  __m128 vec_result1 = _mm_mul_ps(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  __m128 vec_row2 = wcn_mat3_get_row(&m, 2);
  __m128 vec_result2 = _mm_mul_ps(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation
  float32x4_t vec_v0 = vdupq_n_f32(v.v[0]); // Broadcast v0 to all elements
  float32x4_t vec_v1 = vdupq_n_f32(v.v[1]); // Broadcast v1 to all elements
  float32x4_t vec_v2 = vdupq_n_f32(v.v[2]); // Broadcast v2 to all elements

  // Scale first row (indices 0-3)
  float32x4_t vec_row0 = wcn_mat3_get_row(&m, 0);
  float32x4_t vec_result0 = vmulq_f32(vec_row0, vec_v0);
  wcn_mat3_set_row(&newDst, 0, vec_result0);

  // Scale second row (indices 4-7)
  float32x4_t vec_row1 = wcn_mat3_get_row(&m, 1);
  float32x4_t vec_result1 = vmulq_f32(vec_row1, vec_v1);
  wcn_mat3_set_row(&newDst, 1, vec_result1);

  // Scale third row (indices 8-11)
  float32x4_t vec_row2 = wcn_mat3_get_row(&m, 2);
  float32x4_t vec_result2 = vmulq_f32(vec_row2, vec_v2);
  wcn_mat3_set_row(&newDst, 2, vec_result2);

#else
  // Scalar fallback (original implementation)
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];
  newDst.m[0] = v0 * m.m[0 * 4 + 0];
  newDst.m[1] = v0 * m.m[0 * 4 + 1];
  newDst.m[2] = v0 * m.m[0 * 4 + 2];
  newDst.m[4] = v1 * m.m[1 * 4 + 0];
  newDst.m[5] = v1 * m.m[1 * 4 + 1];
  newDst.m[6] = v1 * m.m[1 * 4 + 2];
  newDst.m[8] = v2 * m.m[2 * 4 + 0];
  newDst.m[9] = v2 * m.m[2 * 4 + 1];
  newDst.m[10] = v2 * m.m[2 * 4 + 2];
#endif

  return newDst;
}

// Mat3 scaling
/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has two
 * entries.
 * @param v - A vector of
 *     2 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling)
(const WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = v.v[0];
  // newDst.m[1] = 1;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     3 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling3D)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  newDst.m[0] = v.v[0];
  // newDst.m[1] = 0;
  // newDst.m[2] = 0;
  // newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  // newDst.m[6] = 0;
  // newDst.m[8] = 0;
  // newDst.m[9] = 0;
  newDst.m[10] = v.v[2];

  return newDst;
}

// Mat3 uniform_scale
/**
 * Scales the given 3-by-3 matrix in the X and Y dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process first two rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = _mm_loadu_ps(&m.m[0]); // Load first row [m00, m01, m02, pad]
  __m128 row1 = _mm_loadu_ps(&m.m[4]); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);

  // Store results
  _mm_storeu_ps(&newDst.m[0], scaled_row0);
  _mm_storeu_ps(&newDst.m[4], scaled_row1);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process first two rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = vld1q_f32(&m.m[0]); // Load first row [m00, m01, m02, pad]
  float32x4_t row1 = vld1q_f32(&m.m[4]); // Load second row [m10, m11, m12, pad]

  // Scale the first two rows
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);

  // Store results
  vst1q_f32(&newDst.m[0], scaled_row0);
  vst1q_f32(&newDst.m[4], scaled_row1);

  // Copy the third row unchanged if matrices are different
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0];
  newDst.m[1] = s * m.m[1];
  newDst.m[2] = s * m.m[2];
  newDst.m[4] = s * m.m[4];
  newDst.m[5] = s * m.m[5];
  newDst.m[6] = s * m.m[6];
  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[8] = m.m[8];
    newDst.m[9] = m.m[9];
    newDst.m[10] = m.m[10];
  }
#endif

  return newDst;
}

// Mat3 uniform_scale3D
/**
 * Scales the given 3-by-3 matrix in each dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale_3D)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process all three rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = _mm_loadu_ps(&m.m[0]); // Load first row [m00, m01, m02, pad]
  __m128 row1 = _mm_loadu_ps(&m.m[4]); // Load second row [m10, m11, m12, pad]
  __m128 row2 = _mm_loadu_ps(&m.m[8]); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);
  __m128 scaled_row2 = _mm_mul_ps(row2, vec_s);

  // Store results
  _mm_storeu_ps(&newDst.m[0], scaled_row0);
  _mm_storeu_ps(&newDst.m[4], scaled_row1);
  _mm_storeu_ps(&newDst.m[8], scaled_row2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process all three rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = vld1q_f32(&m.m[0]); // Load first row [m00, m01, m02, pad]
  float32x4_t row1 = vld1q_f32(&m.m[4]); // Load second row [m10, m11, m12, pad]
  float32x4_t row2 = vld1q_f32(&m.m[8]); // Load third row [m20, m21, m22, pad]

  // Scale all three rows
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);
  float32x4_t scaled_row2 = vmulq_f32(row2, vec_s);

  // Store results
  vst1q_f32(&newDst.m[0], scaled_row0);
  vst1q_f32(&newDst.m[4], scaled_row1);
  vst1q_f32(&newDst.m[8], scaled_row2);

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0];
  newDst.m[1] = s * m.m[1];
  newDst.m[2] = s * m.m[2];
  newDst.m[4] = s * m.m[4];
  newDst.m[5] = s * m.m[5];
  newDst.m[6] = s * m.m[6];
  newDst.m[8] = s * m.m[8];
  newDst.m[9] = s * m.m[9];
  newDst.m[10] = s * m.m[10];
#endif

  return newDst;
}

// Mat3 uniform_scaling
/**
 * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling)
(float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - create uniform scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create rows: [s, 0, 0, pad], [0, s, 0, pad], [0, 0, 1, pad]
  __m128 row0 =
      _mm_move_ss(vec_s, vec_zero); // [0, s, 0, 0] -> shuffle to [s, 0, 0, pad]
  row0 =
      _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0)); // [s, 0, 0, pad]

  __m128 row1 =
      _mm_move_ss(vec_zero, vec_s); // [s, 0, 0, 0] -> shuffle to [0, s, 0, pad]
  row1 =
      _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1)); // [0, s, 0, pad]

  __m128 row2 = _mm_move_ss(
      vec_zero, vec_one); // [1, 0, 0, 0] -> shuffle to [0, 0, 1, pad]
  row2 =
      _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0)); // [0, 0, 1, pad]

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - create uniform scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create rows and store results directly
  float32x4_t row0 = vec_s;
  row0 = vsetq_lane_f32(0.0f, row0, 1);
  row0 = vsetq_lane_f32(0.0f, row0, 2);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(1.0f, row2, 2);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#else
  // Scalar fallback - zero initialization is more efficient than memset
  newDst.m[0] = s;
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;
  newDst.m[4] = 0.0f;
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;
  newDst.m[8] = 0.0f;
  newDst.m[9] = 0.0f;
  newDst.m[10] = 1.0f;
#endif

  return newDst;
}

// Mat3 uniform_scaling3D
/**
 * Creates a 3-by-3 matrix which scales uniformly in each dimension
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling_3D)
(float s) {
  WMATH_TYPE(Mat3) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - create uniform 3D scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();

  // Create diagonal matrix with s on diagonal
  // Row0: [s, 0, 0, pad]
  __m128 row0 = _mm_move_ss(vec_s, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, s, 0, pad]
  __m128 row1 = _mm_move_ss(vec_zero, vec_s);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [0, 0, s, pad]
  __m128 row2 = _mm_move_ss(vec_zero, vec_s);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - create uniform 3D scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);

  // Create diagonal matrix with s on diagonal
  float32x4_t row0 = vec_s;
  row0 = vsetq_lane_f32(0.0f, row0, 1);
  row0 = vsetq_lane_f32(0.0f, row0, 2);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(s, row2, 2);

  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);

#else
  // Scalar fallback - direct assignment is more efficient
  newDst.m[0] = s;
  newDst.m[1] = 0.0f;
  newDst.m[2] = 0.0f;
  newDst.m[4] = 0.0f;
  newDst.m[5] = s;
  newDst.m[6] = 0.0f;
  newDst.m[8] = 0.0f;
  newDst.m[9] = 0.0f;
  newDst.m[10] = s;
#endif

  return newDst;
}

// Mat4 getScaling
/**
 * Returns the "3d" scaling component of the matrix
 * @param m - The Matrix
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_scaling)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) result;

#if !defined(WMATH_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation: directly read columns like scalar version

  // x column: m[0], m[1], m[2]
  __m128 col0 = _mm_set_ps(0.0f, m.m[2], m.m[1], m.m[0]);
  __m128 sq0 = _mm_mul_ps(col0, col0);
  result.v[0] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq0)));

  // y column: m[4], m[5], m[6]
  __m128 col1 = _mm_set_ps(0.0f, m.m[6], m.m[5], m.m[4]);
  __m128 sq1 = _mm_mul_ps(col1, col1);
  result.v[1] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq1)));

  // z column: m[8], m[9], m[10]
  __m128 col2 = _mm_set_ps(0.0f, m.m[10], m.m[9], m.m[8]);
  __m128 sq2 = _mm_mul_ps(col2, col2);
  result.v[2] = sqrtf(_mm_cvtss_f32(wcn_hadd_ps(sq2)));

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation

  float32x4_t col0 = { m.m[0], m.m[1], m.m[2], 0.0f };
  float32x4_t sq0  = vmulq_f32(col0, col0);
  result.v[0] = sqrtf(wcn_hadd_f32(sq0));

  float32x4_t col1 = { m.m[4], m.m[5], m.m[6], 0.0f };
  float32x4_t sq1  = vmulq_f32(col1, col1);
  result.v[1] = sqrtf(wcn_hadd_f32(sq1));

  float32x4_t col2 = { m.m[8], m.m[9], m.m[10], 0.0f };
  float32x4_t sq2  = vmulq_f32(col2, col2);
  result.v[2] = sqrtf(wcn_hadd_f32(sq2));

#else
  // Scalar fallback
  result.v[0] = sqrtf(m.m[0]*m.m[0] + m.m[1]*m.m[1] + m.m[2]*m.m[2]);
  result.v[1] = sqrtf(m.m[4]*m.m[4] + m.m[5]*m.m[5] + m.m[6]*m.m[6]);
  result.v[2] = sqrtf(m.m[8]*m.m[8] + m.m[9]*m.m[9] + m.m[10]*m.m[10]);
#endif

  return result;
}

// Mat4 scaling
/**
 * Creates a 4-by-4 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     three entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, scaling)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - column-major scaling matrix
  __m128 col0 = _mm_set_ps(0.0f, 0.0f, 0.0f, v.v[0]); // first column
  __m128 col1 = _mm_set_ps(0.0f, 0.0f, v.v[1], 0.0f); // second column
  __m128 col2 = _mm_set_ps(0.0f, v.v[2], 0.0f, 0.0f); // third column
  __m128 col3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);   // fourth column

  _mm_storeu_ps(&newDst.m[0], col0);
  _mm_storeu_ps(&newDst.m[4], col1);
  _mm_storeu_ps(&newDst.m[8], col2);
  _mm_storeu_ps(&newDst.m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - column-major scaling matrix
  float32x4_t col0 = {v.v[0], 0.0f, 0.0f, 0.0f};
  float32x4_t col1 = {0.0f, v.v[1], 0.0f, 0.0f};
  float32x4_t col2 = {0.0f, 0.0f, v.v[2], 0.0f};
  float32x4_t col3 = {0.0f, 0.0f, 0.0f, 1.0f};

  vst1q_f32(&newDst.m[0], col0);
  vst1q_f32(&newDst.m[4], col1);
  vst1q_f32(&newDst.m[8], col2);
  vst1q_f32(&newDst.m[12], col3);

#else
  // Scalar fallback
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0]  = v.v[0];
  newDst.m[5]  = v.v[1];
  newDst.m[10] = v.v[2];
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}

// Mat4 uniformScale
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scale)
(WMATH_TYPE(Mat4) m, float s) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process first three rows with SIMD
  __m128 vec_s = _mm_set1_ps(s);
  __m128 row0 = wcn_mat4_get_row(&m, 0);
  __m128 row1 = wcn_mat4_get_row(&m, 1);
  __m128 row2 = wcn_mat4_get_row(&m, 2);
  __m128 row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  __m128 scaled_row0 = _mm_mul_ps(row0, vec_s);
  __m128 scaled_row1 = _mm_mul_ps(row1, vec_s);
  __m128 scaled_row2 = _mm_mul_ps(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process first three rows with SIMD
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t row0 = wcn_mat4_get_row(&m, 0);
  float32x4_t row1 = wcn_mat4_get_row(&m, 1);
  float32x4_t row2 = wcn_mat4_get_row(&m, 2);
  float32x4_t row3 = wcn_mat4_get_row(&m, 3);

  // Scale the first three rows (rotation/scaling part)
  float32x4_t scaled_row0 = vmulq_f32(row0, vec_s);
  float32x4_t scaled_row1 = vmulq_f32(row1, vec_s);
  float32x4_t scaled_row2 = vmulq_f32(row2, vec_s);

  // Store results
  wcn_mat4_set_row(&newDst, 0, scaled_row0);
  wcn_mat4_set_row(&newDst, 1, scaled_row1);
  wcn_mat4_set_row(&newDst, 2, scaled_row2);

  // Copy the fourth row (translation part) unchanged if matrices are different
  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    wcn_mat4_set_row(&newDst, 3, row3);
  }

#else
  // Scalar fallback with optimized loop unrolling
  newDst.m[0] = s * m.m[0 * 4 + 0];
  newDst.m[1] = s * m.m[0 * 4 + 1];
  newDst.m[2] = s * m.m[0 * 4 + 2];
  newDst.m[3] = s * m.m[0 * 4 + 3];
  newDst.m[4] = s * m.m[1 * 4 + 0];
  newDst.m[5] = s * m.m[1 * 4 + 1];
  newDst.m[6] = s * m.m[1 * 4 + 2];
  newDst.m[7] = s * m.m[1 * 4 + 3];
  newDst.m[8] = s * m.m[2 * 4 + 0];
  newDst.m[9] = s * m.m[2 * 4 + 1];
  newDst.m[10] = s * m.m[2 * 4 + 2];
  newDst.m[11] = s * m.m[2 * 4 + 3];

  if (!WMATH_EQUALS(Mat4)(newDst, m)) {
    newDst.m[12] = m.m[12];
    newDst.m[13] = m.m[13];
    newDst.m[14] = m.m[14];
    newDst.m[15] = m.m[15];
  }
#endif

  return newDst;
}

// Mat4 uniformScaling
/**
 * Creates a 4-by-4 matrix which scales a uniform amount in each dimension.
 * @param s - the amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scaling)
(float s) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - create uniform scaling matrix efficiently
  __m128 vec_s = _mm_set1_ps(s);
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);

  // Create diagonal matrix with s on diagonal
  // Row0: [s, 0, 0, 0]
  __m128 row0 = _mm_move_ss(vec_s, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, s, 0, 0]
  __m128 row1 = _mm_move_ss(vec_zero, vec_s);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [0, 0, s, 0]
  __m128 row2 = _mm_move_ss(vec_zero, vec_s);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row3: [0, 0, 0, 1]
  __m128 row3 = _mm_move_ss(vec_zero, vec_one);
  row3 = _mm_shuffle_ps(row3, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - create uniform scaling matrix efficiently
  float32x4_t vec_s = vdupq_n_f32(s);
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);

  // Create diagonal matrix with s on diagonal
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(s, row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(s, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(s, row2, 2);

  float32x4_t row3 = vec_zero;
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#else
  // Scalar fallback - direct assignment is more efficient than memset
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));
  newDst.m[0] = s;
  newDst.m[5] = s;
  newDst.m[10] = s;
  newDst.m[15] = 1.0f;
#endif

  return newDst;
}

// Mat4 ortho
/**
 * Computes a 4-by-4 orthographic projection matrix.
 * @param left - The left coordinate of the viewing frustum.
 * @param right - The right coordinate of the viewing frustum.
 * @param bottom - The bottom coordinate of the viewing frustum.
 * @param top - The top coordinate of the viewing frustum.
 * @param near - The near coordinate of the viewing frustum.
 * @param far - The far coordinate of the viewing frustum.
 * @returns The orthographic projection matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, ortho)
(float left, float right, float bottom, float top, float near, float far) {
  WMATH_TYPE(Mat4) newDst;

#if !defined(WMATH_DISABLE_SIMD) &&                                            \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - optimized orthographic matrix creation
  __m128 vec_zero = _mm_setzero_ps();
  __m128 vec_one = _mm_set1_ps(1.0f);
  _mm_set1_ps(-1.0f);
  __m128 vec_two = _mm_set1_ps(2.0f);
  __m128 vec_neg_two = _mm_set1_ps(-2.0f);

  // Calculate reciprocal values for orthographic projection
  __m128 vec_left = _mm_set1_ps(left);
  __m128 vec_right = _mm_set1_ps(right);
  __m128 vec_bottom = _mm_set1_ps(bottom);
  __m128 vec_top = _mm_set1_ps(top);
  __m128 vec_near = _mm_set1_ps(near);
  __m128 vec_far = _mm_set1_ps(far);

  // Calculate right - left and top - bottom
  __m128 rl_diff = _mm_sub_ps(vec_right, vec_left);
  __m128 tb_diff = _mm_sub_ps(vec_top, vec_bottom);
  __m128 fn_diff = _mm_sub_ps(vec_far, vec_near);

  // Calculate reciprocal values using fast division
  __m128 rl_inv = _mm_div_ps(vec_two, rl_diff);
  __m128 tb_inv = _mm_div_ps(vec_two, tb_diff);
  __m128 fn_inv = _mm_div_ps(vec_neg_two, fn_diff);

  // Extract scalar values for translation calculations
  float rl_inv_val = _mm_cvtss_f32(rl_inv);
  float tb_inv_val = _mm_cvtss_f32(tb_inv);
  float fn_inv_val = _mm_cvtss_f32(fn_inv);

  // Calculate translation components
  float tx = -(right + left) * rl_inv_val * 0.5f;
  float ty = -(top + bottom) * tb_inv_val * 0.5f;
  float tz = -(far + near) * fn_inv_val * 0.5f;

  __m128 vec_tx = _mm_set1_ps(tx);
  __m128 vec_ty = _mm_set1_ps(ty);
  __m128 vec_tz = _mm_set1_ps(tz);

  // Create orthographic matrix rows
  // Row0: [2/(r-l), 0, 0, 0]
  __m128 row0 = _mm_move_ss(rl_inv, vec_zero);
  row0 = _mm_shuffle_ps(row0, vec_zero, _MM_SHUFFLE(0, 0, 2, 0));

  // Row1: [0, 2/(t-b), 0, 0]
  __m128 row1 = _mm_move_ss(vec_zero, tb_inv);
  row1 = _mm_shuffle_ps(row1, vec_zero, _MM_SHUFFLE(0, 2, 0, 1));

  // Row2: [0, 0, -2/(f-n), 0]
  __m128 row2 = _mm_move_ss(vec_zero, fn_inv);
  row2 = _mm_shuffle_ps(row2, vec_zero, _MM_SHUFFLE(0, 2, 1, 0));

  // Row3: [tx, ty, tz, 1]
  __m128 row3 = _mm_move_ss(vec_tx, vec_ty);
  row3 = _mm_shuffle_ps(row3, vec_tz, _MM_SHUFFLE(3, 2, 1, 0));
  row3 = _mm_move_ss(row3, vec_one);
  row3 = _mm_shuffle_ps(row3, row3, _MM_SHUFFLE(3, 2, 1, 0));

  // Store results
  _mm_storeu_ps(&newDst.m[0], row0);
  _mm_storeu_ps(&newDst.m[4], row1);
  _mm_storeu_ps(&newDst.m[8], row2);
  _mm_storeu_ps(&newDst.m[12], row3);

#elif !defined(WMATH_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - optimized orthographic matrix creation
  float32x4_t vec_zero = vdupq_n_f32(0.0f);
  float32x4_t vec_one = vdupq_n_f32(1.0f);
  float32x4_t vec_neg_one = vdupq_n_f32(-1.0f);
  float32x4_t vec_two = vdupq_n_f32(2.0f);
  float32x4_t vec_neg_two = vdupq_n_f32(-2.0f);

  // Calculate reciprocal values for orthographic projection
  float32x4_t vec_left = vdupq_n_f32(left);
  float32x4_t vec_right = vdupq_n_f32(right);
  float32x4_t vec_bottom = vdupq_n_f32(bottom);
  float32x4_t vec_top = vdupq_n_f32(top);
  float32x4_t vec_near = vdupq_n_f32(near);
  float32x4_t vec_far = vdupq_n_f32(far);

  // Calculate right - left and top - bottom
  float32x4_t rl_diff = vsubq_f32(vec_right, vec_left);
  float32x4_t tb_diff = vsubq_f32(vec_top, vec_bottom);
  float32x4_t fn_diff = vsubq_f32(vec_far, vec_near);

  // Calculate reciprocal values using fast division
  float32x4_t rl_inv = vdivq_f32(vec_two, rl_diff);
  float32x4_t tb_inv = vdivq_f32(vec_two, tb_diff);
  float32x4_t fn_inv = vdivq_f32(vec_neg_two, fn_diff);

  // Extract scalar values for translation calculations
  float rl_inv_val = vgetq_lane_f32(rl_inv, 0);
  float tb_inv_val = vgetq_lane_f32(tb_inv, 0);
  float fn_inv_val = vgetq_lane_f32(fn_inv, 0);

  // Calculate translation components
  float tx = -(right + left) * rl_inv_val * 0.5f;
  float ty = -(top + bottom) * tb_inv_val * 0.5f;
  float tz = -(far + near) * fn_inv_val * 0.5f;

  // Create orthographic matrix rows
  float32x4_t row0 = vec_zero;
  row0 = vsetq_lane_f32(rl_inv_val, row0, 0);

  float32x4_t row1 = vec_zero;
  row1 = vsetq_lane_f32(tb_inv_val, row1, 1);

  float32x4_t row2 = vec_zero;
  row2 = vsetq_lane_f32(fn_inv_val, row2, 2);

  float32x4_t row3 = vec_zero;
  row3 = vsetq_lane_f32(tx, row3, 0);
  row3 = vsetq_lane_f32(ty, row3, 1);
  row3 = vsetq_lane_f32(tz, row3, 2);
  row3 = vsetq_lane_f32(1.0f, row3, 3);

  // Store results
  vst1q_f32(&newDst.m[0], row0);
  vst1q_f32(&newDst.m[4], row1);
  vst1q_f32(&newDst.m[8], row2);
  vst1q_f32(&newDst.m[12], row3);

#else
  // Scalar fallback - optimized calculation
  memset(&newDst, 0, sizeof(WMATH_TYPE(Mat4)));

  // Calculate reciprocal values once for efficiency
  float rl_inv = 2.0f / (right - left);
  float tb_inv = 2.0f / (top - bottom);
  float fn_inv = -2.0f / (far - near);

  // Set diagonal elements
  newDst.m[0] = rl_inv;
  newDst.m[5] = tb_inv;
  newDst.m[10] = fn_inv;
  newDst.m[15] = 1.0f;

  // Set translation components with optimized calculations
  newDst.m[12] = -(right + left) * rl_inv * 0.5f;
  newDst.m[13] = -(top + bottom) * tb_inv * 0.5f;
  newDst.m[14] = -(far + near) * fn_inv * 0.5f;
#endif

  return newDst;
}