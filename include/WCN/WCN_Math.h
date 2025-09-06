#ifndef WCN_MATH_H
#define WCN_MATH_H

// SIMD includes on supported platforms
#include <math.h>
#include <stdbool.h>
#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
#include <emmintrin.h>
#include <smmintrin.h> // For SSE4.1 which has better float operations
#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <string.h> // For memcpy

#ifdef __cplusplus
extern "C" {
#endif

// CONST

static float EPSILON = 0.000001f;
float wcn_math_set_epsilon(float epsilon) {
  float old_epsilon = EPSILON;
  EPSILON = epsilon;
  return old_epsilon;
}
float wcn_math_get_epsilon() { return EPSILON; }
#define WCN_GET_EPSILON() wcn_math_get_epsilon()

// MACRO

// ()
#define WMATH_CALL(TYPE, FUNC) wcn_math_##TYPE##_##FUNC

// ?
#define WMATH_OR_ELSE(value, other) (value ? value : other)

// ?0
#define WMATH_OR_ELSE_ZERO(value) (value ? value : 0.0f)

// 1
#define WMATH_IDENTITY(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_identity

// 0
#define WMATH_ZERO(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_zero

// TYPE
#define WMATH_TYPE(WCN_Math_TYPE) WCN_Math_##WCN_Math_TYPE

#define WMATH_CREATE_TYPE(WCN_Math_TYPE) WCN_Math_##WCN_Math_TYPE##_Create

// set
#define WMATH_SET(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_set

// create
#define WMATH_CREATE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_create

// copy
#define WMATH_COPY(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_copy

// equals
#define WMATH_EQUALS(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_equals

// equalsApproximately
#define WMATH_EQUALS_APPROXIMATELY(WCN_Math_TYPE)                              \
  wcn_math_##WCN_Math_TYPE##_equalsApproximately

// negate
#define WMATH_NEGATE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_negate

// transpose
#define WMATH_TRANSPOSE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_transpose

// add
#define WMATH_ADD(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_add

// sub
#define WMATH_SUB(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_sub

// multiplyScalar
#define WMATH_MULTIPLY_SCALAR(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##_multiplyScalar

// multiply
#define WMATH_MULTIPLY(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_multiply

// inverse
#define WMATH_INVERSE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_inverse

// invert
#define WMATH_INVERT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_invert

// vec dot
#define WMATH_DOT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_dot

// <T> interface lerp
#define WMATH_LERP(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lerp

// vec length
#define WMATH_LENGTH(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_length

// vec length squared
#define WMATH_LENGTH_SQ(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lengthSquared

// vec normalize
#define WMATH_NORMALIZE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_normalize

// BEGIN Utils

#define WMATH_DEG2RED(degrees) (degrees * 0.017453292519943295f)

#define WMATH_RED2DEG(radians) (radians * 57.29577951308232f)

// #define WMATH_NUM_LERP(a, b, t) ((a) + ((b) - (a)) * (t))
// Impl of lerp for float, double, int, and float_t
// ==================================================================
int WMATH_LERP(int)(int a, int b, float t) {
  return (int)(a + ((b) - (a)) * t);
}
float WMATH_LERP(float)(float a, float b, float t) {
  return (a + ((b) - (a)) * t);
}
double WMATH_LERP(double)(double a, double b, double t) {
  return (a + ((b) - (a)) * t);
}

float_t WMATH_LERP(float_t)(float_t a, float_t b, float_t t) {
  return (a + ((b) - (a)) * t);
}

double_t WMATH_LERP(double_t)(double_t a, double_t b, double_t t) {
  return (a + ((b) - (a)) * t);
}
// ==================================================================

#define WMATH_INVERSE_LERP(a, b, t)                                            \
  (fabsf(b - a) < wcn_math_get_epsilon() ? a : (((b - a) - a) / d))

#define WMATH_EUCLIDEAN_MODULO(n, m) ((n) - floorf((n) / (m)) * (m))
// END Utils

// TYPE

// Mat3 Type

typedef struct {
  float m[12]; // Using 12 elements for better SIMD alignment
} WMATH_TYPE(Mat3);

typedef struct {
  float m_00;
  float m_01;
  float m_02;
  // next row
  float m_10;
  float m_11;
  float m_12;
  // next row
  float m_20;
  float m_21;
  float m_22;
} WMATH_CREATE_TYPE(Mat3);

// Mat4 Type

typedef struct {
  float m[16]; // Using 9 elements for better SIMD alignment
} WMATH_TYPE(Mat4);

typedef struct {
  float m_00;
  float m_01;
  float m_02;
  float m_03;
  // next row
  float m_10;
  float m_11;
  float m_12;
  float m_13;
  // next row
  float m_20;
  float m_21;
  float m_22;
  float m_23;
  // next row
  float m_30;
  float m_31;
  float m_32;
  float m_33;
} WMATH_CREATE_TYPE(Mat4);

// Quat Type

typedef struct {
  float v[4];
} WMATH_TYPE(Quat);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
  float v_w;
} WMATH_CREATE_TYPE(Quat);

enum WCN_Math_RotationOrder {
  WCN_Math_RotationOrder_XYZ,
  WCN_Math_RotationOrder_XZY,
  WCN_Math_RotationOrder_YXZ,
  WCN_Math_RotationOrder_YZX,
  WCN_Math_RotationOrder_ZXY,
  WCN_Math_RotationOrder_ZYX,
};

// Vec2 Type

typedef struct {
  float v[2];
} WMATH_TYPE(Vec2);

typedef struct {
  float v_x;
  float v_y;
} WMATH_CREATE_TYPE(Vec2);

// Vec3 Type

typedef struct {
  float v[3];
} WMATH_TYPE(Vec3);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
} WMATH_CREATE_TYPE(Vec3);

typedef struct {
  float angle;
  WMATH_TYPE(Vec3) axis;
} WCN_Math_Vec3_WithAngleAxis;

// Vec4 Type

typedef struct {
  float v[4];
} WMATH_TYPE(Vec4);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
  float v_w;
} WMATH_CREATE_TYPE(Vec4);

// BEGIN Vec2
// END Vec2

// BEGIN Vec3

WMATH_TYPE(Vec3) WMATH_CREATE(Vec3)(WMATH_CREATE_TYPE(Vec3) vec3_c) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = WMATH_OR_ELSE_ZERO(vec3_c.v_x);
  vec3.v[1] = WMATH_OR_ELSE_ZERO(vec3_c.v_y);
  vec3.v[2] = WMATH_OR_ELSE_ZERO(vec3_c.v_z);
  return vec3;
}

// END Vec3

// BEGIN Vec4
// END Vec4

// BEGIN Mat3

WMATH_TYPE(Mat3) WMATH_IDENTITY(Mat3)() {
  return (WMATH_TYPE(Mat3)){1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
};

WMATH_TYPE(Mat3) WMATH_ZERO(Mat3)() {
  return (WMATH_TYPE(Mat3)){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

WMATH_TYPE(Mat3)
WMATH_CREATE(Mat3)(WMATH_CREATE_TYPE(Mat3) mat3_c) {
  WMATH_TYPE(Mat3) mat;
  // Initialize all elements to 0 first for consistent results
  memset(&mat, 0, sizeof(WMATH_TYPE(Mat3)));

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

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - negate using XOR with sign bit mask
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_a, vec_res;

  // Process first 4 elements (indices 0-3)
  vec_a = _mm_loadu_ps(&mat.m[0]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = _mm_loadu_ps(&mat.m[4]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = _mm_loadu_ps(&mat.m[8]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
  _mm_storeu_ps(&result.m[8], vec_res);

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - process 4 floats at a time
  __m128 vec_a, vec_b, vec_res;

  // Process first 4 elements (indices 0-3)
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_b = _mm_loadu_ps(&b.m[0]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = _mm_loadu_ps(&a.m[4]);
  vec_b = _mm_loadu_ps(&b.m[4]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = _mm_loadu_ps(&a.m[8]);
  vec_b = _mm_loadu_ps(&b.m[8]);
  vec_res = _mm_add_ps(vec_a, vec_b);
  _mm_storeu_ps(&result.m[8], vec_res);

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - process 4 floats at a time
  float32x4_t vec_a, vec_b, vec_res;

  // Process first 4 elements (indices 0-3)
  vec_a = vld1q_f32(&a.m[0]);
  vec_b = vld1q_f32(&b.m[0]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[0], vec_res);

  // Process next 4 elements (indices 4-7)
  vec_a = vld1q_f32(&a.m[4]);
  vec_b = vld1q_f32(&b.m[4]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[4], vec_res);

  // Process last 4 elements (indices 8-11)
  vec_a = vld1q_f32(&a.m[8]);
  vec_b = vld1q_f32(&b.m[8]);
  vec_res = vaddq_f32(vec_a, vec_b);
  vst1q_f32(&result.m[8], vec_res);

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

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation
  __m128 vec_a, vec_b, vec_res;

  // Process first 4 elements
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_b = _mm_loadu_ps(&b.m[0]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
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

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - broadcast scalar to all vector elements and multiply
  __m128 vec_a, vec_b, vec_res;
  vec_b = _mm_set1_ps(b); // Broadcast scalar to 4 elements

  // Process first 4 elements
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_res = _mm_mul_ps(vec_a, vec_b);
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

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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

WMATH_TYPE(Mat3)
WMATH_INVERSE(Mat3)(WMATH_TYPE(Mat3) a) {
  float m_00 = a.m[0 * 4 + 0];
  float m_01 = a.m[0 * 4 + 1];
  float m_02 = a.m[0 * 4 + 2];
  float m_10 = a.m[1 * 4 + 0];
  float m_11 = a.m[1 * 4 + 1];
  float m_12 = a.m[1 * 4 + 2];
  float m_20 = a.m[2 * 4 + 0];
  float m_21 = a.m[2 * 4 + 1];
  float m_22 = a.m[2 * 4 + 2];

  float b_01 = m_22 * m_11 - m_21 * m_12;
  float b_11 = -m_22 * m_01 + m_21 * m_02;
  float b_21 = m_12 * m_01 - m_11 * m_02;

  float inv_det = 1.0f / (m_00 * b_01 - m_10 * b_11 + m_20 * b_21);

  return WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
      .m_00 = b_01 * inv_det,
      .m_01 = (-m_22 * m_10 + m_20 * m_12) * inv_det,
      .m_02 = (m_12 * m_00 - m_10 * m_02) * inv_det,
      .m_10 = b_11 * inv_det,
      .m_11 = (m_22 * m_00 - m_20 * m_02) * inv_det,
      .m_12 = (-m_12 * m_00 + m_10 * m_02) * inv_det,
      .m_20 = b_21 * inv_det,
      .m_21 = (-m_21 * m_00 + m_20 * m_01) * inv_det,
      .m_22 = (m_11 * m_00 - m_10 * m_01) * inv_det,
  });
}

WMATH_TYPE(Mat3)
WMATH_INVERT(Mat3)(WMATH_TYPE(Mat3) a) { return WMATH_INVERSE(Mat3)(a); }

// Optimized matrix multiplication
WMATH_TYPE(Mat3)
WMATH_MULTIPLY(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b) {
  WMATH_TYPE(Mat3) result;
  memset(&result, 0, sizeof(WMATH_TYPE(Mat3))); // Initialize all to 0

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE optimized matrix multiplication
  __m128 row, col, prod, sum;

  // Calculate first row of result
  // result[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
  row = _mm_set_ps(0.0f, a.m[2], a.m[1], a.m[0]);
  col = _mm_set_ps(0.0f, b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
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

  // Calculate third row of result
  // result[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
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

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
  // NEON optimized matrix multiplication
  float32x4_t row, col, prod, sum;

  // Calculate first row of result
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

  // Calculate second row of result
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

  // Calculate third row of result
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
// 创建绕Z轴旋转矩阵
WMATH_TYPE(Mat3) wcn_math_Mat3_rotationZ(float angle) {
  WMATH_TYPE(Mat3) result;
  float cos_a = cosf(angle);
  float sin_a = sinf(angle);

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - faster initialization
  __m128 zero = _mm_setzero_ps();
  _mm_storeu_ps(&result.m[0], zero);
  _mm_storeu_ps(&result.m[4], zero);
  _mm_storeu_ps(&result.m[8], zero);

  // Set rotation values
  result.m[0] = cos_a;  // [0,0]
  result.m[1] = -sin_a; // [0,1]
  result.m[4] = sin_a;  // [1,0]
  result.m[5] = cos_a;  // [1,1]
  result.m[10] = 1.0f;  // [2,2]

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
  // NEON implementation - faster initialization
  float32x4_t zero = vdupq_n_f32(0.0f);
  vst1q_f32(&result.m[0], zero);
  vst1q_f32(&result.m[4], zero);
  vst1q_f32(&result.m[8], zero);

  // Set rotation values
  result.m[0] = cos_a;  // [0,0]
  result.m[1] = -sin_a; // [0,1]
  result.m[4] = sin_a;  // [1,0]
  result.m[5] = cos_a;  // [1,1]
  result.m[10] = 1.0f;  // [2,2]

#else
  // Scalar fallback
  memset(&result, 0, sizeof(WMATH_TYPE(Mat3)));
  result.m[0] = cos_a;
  result.m[1] = -sin_a;
  result.m[4] = sin_a;
  result.m[5] = cos_a;
  result.m[10] = 1.0f;
#endif

  return result;
}

// END Mat3

// BEGIN Mat4

// 0 add 1 Mat4

WMATH_TYPE(Mat4) WMATH_IDENTITY(Mat4)() {
  WMATH_TYPE(Mat4) result;
  memset(&result, 0, sizeof(WMATH_TYPE(Mat4)));
  result.m[0] = 1.0f;
  result.m[5] = 1.0f;
  result.m[10] = 1.0f;
  result.m[15] = 1.0f;
  return result;
}

WMATH_TYPE(Mat4) WMATH_ZERO(Mat4)() {
  WMATH_TYPE(Mat4) result;
  memset(&result, 0, sizeof(WMATH_TYPE(Mat4)));
  return result;
}

// Init Mat4

WMATH_TYPE(Mat4) WMATH_CREATE(Mat4)(WMATH_CREATE_TYPE(Mat4) mat4_c) {
  WMATH_TYPE(Mat4) mat;
  memset(&mat, 0, sizeof(WMATH_TYPE(Mat4)));
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

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE implementation - negate using XOR with sign bit mask
  __m128 sign_mask = _mm_set1_ps(-0.0f); // 0x80000000 for all elements
  __m128 vec_a, vec_res;

  // Process all 16 elements in groups of 4
  vec_a = _mm_loadu_ps(&mat.m[0]);
  vec_res = _mm_xor_ps(vec_a, sign_mask);
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

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 vec_a, vec_b, vec_res;
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_b = _mm_loadu_ps(&b.m[0]);
  vec_res = _mm_add_ps(vec_a, vec_b);
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
#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 vec_a, vec_b, vec_res;
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_b = _mm_loadu_ps(&b.m[0]);
  vec_res = _mm_sub_ps(vec_a, vec_b);
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
#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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
#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  __m128 vec_a, vec_b_scalar, vec_res;
  vec_b_scalar = _mm_set1_ps(b);
  vec_a = _mm_loadu_ps(&a.m[0]);
  vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
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
#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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
  memset(&result, 0, sizeof(WMATH_TYPE(Mat4))); // Initialize all to 0

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
  // SSE optimized matrix multiplication
  __m128 row, col, prod, sum;

  // Calculate first row of result
  row = _mm_loadu_ps(&a.m[0]); // Load first row of matrix a

  // result.m[0] = a.m[0]*b.m[0] + a.m[1]*b.m[4] + a.m[2]*b.m[8] +
  // a.m[3]*b.m[12]
  col = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[0] = _mm_cvtss_f32(sum);

  // result.m[1] = a.m[0]*b.m[1] + a.m[1]*b.m[5] + a.m[2]*b.m[9] +
  // a.m[3]*b.m[13]
  col = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[1] = _mm_cvtss_f32(sum);

  // result.m[2] = a.m[0]*b.m[2] + a.m[1]*b.m[6] + a.m[2]*b.m[10] +
  // a.m[3]*b.m[14]
  col = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[2] = _mm_cvtss_f32(sum);

  // result.m[3] = a.m[0]*b.m[3] + a.m[1]*b.m[7] + a.m[2]*b.m[11] +
  // a.m[3]*b.m[15]
  col = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[3] = _mm_cvtss_f32(sum);

  // Calculate second row of result
  row = _mm_loadu_ps(&a.m[4]); // Load second row of matrix a

  // result.m[4] = a.m[4]*b.m[0] + a.m[5]*b.m[4] + a.m[6]*b.m[8] +
  // a.m[7]*b.m[12]
  col = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[4] = _mm_cvtss_f32(sum);

  // result.m[5] = a.m[4]*b.m[1] + a.m[5]*b.m[5] + a.m[6]*b.m[9] +
  // a.m[7]*b.m[13]
  col = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[5] = _mm_cvtss_f32(sum);

  // result.m[6] = a.m[4]*b.m[2] + a.m[5]*b.m[6] + a.m[6]*b.m[10] +
  // a.m[7]*b.m[14]
  col = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[6] = _mm_cvtss_f32(sum);

  // result.m[7] = a.m[4]*b.m[3] + a.m[5]*b.m[7] + a.m[6]*b.m[11] +
  // a.m[7]*b.m[15]
  col = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[7] = _mm_cvtss_f32(sum);

  // Calculate third row of result
  row = _mm_loadu_ps(&a.m[8]); // Load third row of matrix a

  // result.m[8] = a.m[8]*b.m[0] + a.m[9]*b.m[4] + a.m[10]*b.m[8] +
  // a.m[11]*b.m[12]
  col = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[8] = _mm_cvtss_f32(sum);

  // result.m[9] = a.m[8]*b.m[1] + a.m[9]*b.m[5] + a.m[10]*b.m[9] +
  // a.m[11]*b.m[13]
  col = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[9] = _mm_cvtss_f32(sum);

  // result.m[10] = a.m[8]*b.m[2] + a.m[9]*b.m[6] + a.m[10]*b.m[10] +
  // a.m[11]*b.m[14]
  col = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[10] = _mm_cvtss_f32(sum);

  // result.m[11] = a.m[8]*b.m[3] + a.m[9]*b.m[7] + a.m[10]*b.m[11] +
  // a.m[11]*b.m[15]
  col = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[11] = _mm_cvtss_f32(sum);

  // Calculate fourth row of result
  row = _mm_loadu_ps(&a.m[12]); // Load fourth row of matrix a

  // result.m[12] = a.m[12]*b.m[0] + a.m[13]*b.m[4] + a.m[14]*b.m[8] +
  // a.m[15]*b.m[12]
  col = _mm_set_ps(b.m[12], b.m[8], b.m[4], b.m[0]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[12] = _mm_cvtss_f32(sum);

  // result.m[13] = a.m[12]*b.m[1] + a.m[13]*b.m[5] + a.m[14]*b.m[9] +
  // a.m[15]*b.m[13]
  col = _mm_set_ps(b.m[13], b.m[9], b.m[5], b.m[1]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[13] = _mm_cvtss_f32(sum);

  // result.m[14] = a.m[12]*b.m[2] + a.m[13]*b.m[6] + a.m[14]*b.m[10] +
  // a.m[15]*b.m[14]
  col = _mm_set_ps(b.m[14], b.m[10], b.m[6], b.m[2]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[14] = _mm_cvtss_f32(sum);

  // result.m[15] = a.m[12]*b.m[3] + a.m[13]*b.m[7] + a.m[14]*b.m[11] +
  // a.m[15]*b.m[15]
  col = _mm_set_ps(b.m[15], b.m[11], b.m[7], b.m[3]);
  prod = _mm_mul_ps(row, col);
  sum = _mm_hadd_ps(prod, prod);
  sum = _mm_hadd_ps(sum, sum);
  result.m[15] = _mm_cvtss_f32(sum);

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
  // NEON optimized matrix multiplication
  float32x4_t row, col, prod, sum;

  // Calculate first row of result
  row = vld1q_f32(&a.m[0]); // Load first row of matrix a

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

  // Calculate second row of result
  row = vld1q_f32(&a.m[4]); // Load second row of matrix a

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

  // Calculate third row of result
  row = vld1q_f32(&a.m[8]); // Load third row of matrix a

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

  // Calculate fourth row of result
  row = vld1q_f32(&a.m[12]); // Load fourth row of matrix a

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
}

WMATH_TYPE(Mat4)
WMATH_INVERT(Mat4)(WMATH_TYPE(Mat4) a) { return WMATH_INVERSE(Mat4)(a); }

WMATH_TYPE(Mat4)
WMATH_TRANSPOSE(Mat4)(WMATH_TYPE(Mat4) a) {
  WMATH_TYPE(Mat4) result;

#if !defined(CLAY_DISABLE_SIMD) &&                                             \
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

#elif !defined(CLAY_DISABLE_SIMD) && defined(__aarch64__)
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

// END Mat4

// BEGIN Quat

WMATH_TYPE(Quat)
WMATH_ZERO(Quat)(void) {
  return (WMATH_TYPE(Quat)){
      .v = {0.0f, 0.0f, 0.0f, 0.0f},
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
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
}

WMATH_TYPE(Quat)
WMATH_LERP(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t) {
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = a.v[0] + (b.v[0] - a.v[0]) * t,
      .v_y = a.v[1] + (b.v[1] - a.v[1]) * t,
      .v_z = a.v[2] + (b.v[2] - a.v[2]) * t,
      .v_w = a.v[3] + (b.v[3] - a.v[3]) * t,
  });
}

const WMATH_TYPE(Quat) TEMP_QUAT_1 = (WMATH_TYPE(Quat)){
    .v = {0.0f, 0.0f, 0.0f, 0.0f},
};

const WMATH_TYPE(Quat) TEMP_QUAT_2 = (WMATH_TYPE(Quat)){
    .v = {0.0f, 0.0f, 0.0f, 0.0f},
};

// sqlerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, sqlerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b,
                         WMATH_TYPE(Quat) c, WMATH_TYPE(Quat) d, float t) {
  WMATH_TYPE(Quat) result;
  float cosOmega = WMATH_DOT(Quat)(a, b);
  float ep = wcn_math_get_epsilon();
  if (cosOmega < 0.0f) {
    cosOmega = -cosOmega;
    b.v[0] = -b.v[0];
    b.v[1] = -b.v[1];
    b.v[2] = -b.v[2];
    b.v[3] = -b.v[3];
  }
  float scale0;
  float scale1;
  if (1.0f - cosOmega > ep) {
    float omega = acosf(cosOmega);
    float sinOmega = sinf(omega);
    scale0 = sinf((1.0f - t) * omega) / sinOmega;
    scale1 = sinf(t * omega) / sinOmega;
  } else {
    scale0 = 1.0f - t;
    scale1 = t;
  }
  result.v[0] = scale0 * a.v[0] + scale1 * b.v[0];
  result.v[1] = scale0 * a.v[1] + scale1 * b.v[1];
  result.v[2] = scale0 * a.v[2] + scale1 * b.v[2];
  result.v[3] = scale0 * a.v[3] + scale1 * b.v[3];
  return result; 
}

float WMATH_LENGTH(Quat)(WMATH_TYPE(Quat) a) {
  return sqrtf(WMATH_DOT(Quat)(a, a));
}

float WMATH_LENGTH_SQ(Quat)(WMATH_TYPE(Quat) a) {
  return WMATH_DOT(Quat)(a, a);
}

WMATH_TYPE(Quat)
WMATH_NORMALIZE(Quat)(WMATH_TYPE(Quat) a) {
  float len = sqrtf(a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2] +
                    a.v[3] * a.v[3]);
  if (len > 0.00001f) {
    return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
        .v_x = a.v[0] / len,
        .v_y = a.v[1] / len,
        .v_z = a.v[2] / len,
        .v_w = a.v[3] / len,
    });
  } else {
    return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
        .v_x = 0.0f,
        .v_y = 0.0f,
        .v_z = 0.0f,
        .v_w = 1.0f,
    });
  }
}

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  float ep = wcn_math_get_epsilon();
  return fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep &&
         fabsf(a.v[2] - b.v[2]) < ep && fabsf(a.v[3] - b.v[3]) < ep;
}

// ==
bool WMATH_EQUALS(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] &&
         a.v[3] == b.v[3];
}

// identity
WMATH_TYPE(Quat)
WMATH_IDENTITY(Quat)(void) {
  return (WMATH_TYPE(Quat)){
    .v = {0.0f, 0.0f, 0.0f, 1.0f},
  };
}

// END Quat

// FROM

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_mat4)(WMATH_TYPE(Mat4) a) {
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

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_mat3)(WMATH_TYPE(Mat3) a) {
  return WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
      .m_00 = a.m[0],
      .m_01 = a.m[1],
      .m_02 = a.m[2],
      .m_03 = 0.0f,
      // next row
      .m_10 = a.m[3],
      .m_11 = a.m[4],
      .m_12 = a.m[5],
      .m_13 = 0.0f,
      // next row
      .m_20 = a.m[6],
      .m_21 = a.m[7],
      .m_22 = a.m[8],
      .m_23 = 0.0f,
      // next row
      .m_30 = 0.0f,
      .m_31 = 0.0f,
      .m_32 = 0.0f,
      .m_33 = 1.0f,
  });
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_axis_angle)(WMATH_TYPE(Vec3) axis,
                                  float angle_in_radians) {
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = sinf(angle_in_radians * 0.5f) * axis.v[0],
      .v_y = sinf(angle_in_radians * 0.5f) * axis.v[1],
      .v_z = sinf(angle_in_radians * 0.5f) * axis.v[2],
      .v_w = cosf(angle_in_radians * 0.5f),
  });
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

#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_H
