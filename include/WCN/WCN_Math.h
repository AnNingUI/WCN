#ifndef WCN_MATH_H
#define WCN_MATH_H

// SIMD includes on supported platforms
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
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

extern float EPSILON;
float wcn_math_set_epsilon(float epsilon) {
  float old_epsilon = EPSILON;
  EPSILON = epsilon;
  return old_epsilon ? old_epsilon : epsilon;
}
float wcn_math_get_epsilon() { return EPSILON ? EPSILON : 1e-6f; }
#define WCN_GET_EPSILON() wcn_math_get_epsilon()

#define WMATH_PI 3.14159265358979323846f
#define WMATH_2PI 6.28318530717958647693f
#define WMATH_PI_2 1.57079632679489661923f

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

// scale
#define WMATH_SCALE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_scale

// multiply
#define WMATH_MULTIPLY(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_multiply

// inverse
#define WMATH_INVERSE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_inverse

// invert
#define WMATH_INVERT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_inverse

// vec dot
#define WMATH_DOT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_dot

// vec cross
#define WMATH_CROSS(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_cross

// <T> interface lerp
#define WMATH_LERP(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lerp

// vec lerpV
#define WMATH_LERP_V(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lerpV

// vec length
#define WMATH_LENGTH(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_length

// vec length squared
#define WMATH_LENGTH_SQ(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lengthSquared

// vec set_length
#define WMATH_SET_LENGTH(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_setLength

// vec normalize
#define WMATH_NORMALIZE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_normalize

// vec ceil
#define WMATH_CEIL(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_ceil

// vec floor
#define WMATH_FLOOR(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_floor

// vec round
#define WMATH_ROUND(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_round

// vec clamp
#define WMATH_CLAMP(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_clamp

// vec add_scaled
#define WMATH_ADD_SCALED(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_addScaled

// vec angle
#define WMATH_ANGLE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_angle

// vec fmax
#define WMATH_FMAX(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_fmax

// vec fmin
#define WMATH_FMIN(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_fmin

// vec div
#define WMATH_DIV(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_div

// vec div_scalar
#define WMATH_DIV_SCALAR(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_divScalar

// distance
#define WMATH_DISTANCE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distance

// dist
#define WMATH_DIST(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distance

// distSquared
#define WMATH_DISTANCE_SQ(WCN_Math_TYPE)                                       \
  wcn_math_##WCN_Math_TYPE##_distanceSquared

// dist_sq
#define WMATH_DIST_SQ(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distanceSquared

// vec2/3 random
#define WMATH_RANDOM(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_random

// vec truncate
#define WMATH_TRUNCATE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_truncate

// vec midpoint
#define WMATH_MIDPOINT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_midpoint

// mat determinant
#define WMATH_DETERMINANT(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_determinant

// rotate
#define WMATH_ROTATE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotate

// rotate_x
#define WMATH_ROTATE_X(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotateX

// rotate_y
#define WMATH_ROTATE_Y(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotateY

// rotate_z
#define WMATH_ROTATE_Z(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotateZ

// mat rotation
#define WMATH_ROTATION(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotation

// mat rotation_x
#define WMATH_ROTATION_X(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotationX

// mat rotation_y
#define WMATH_ROTATION_Y(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotationY

// mat rotation_z
#define WMATH_ROTATION_Z(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotationZ

// getTranslation
#define WMATH_GET_TRANSLATION(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##getTranslation

#define T$(WCN_Math_TYPE) WMATH_TYPE(WCN_Math_TYPE)

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

// Impl of random for float, double, int, and float_t
// ==================================================================
int WMATH_RANDOM(int)() { return rand(); }

float WMATH_RANDOM(float)() { return ((float)rand()) / RAND_MAX; }

double WMATH_RANDOM(double)() { return ((double)rand()) / RAND_MAX; }

float_t WMATH_RANDOM(float_t)() { return ((float_t)rand()) / RAND_MAX; }

double_t WMATH_RANDOM(double_t)() { return ((double_t)rand()) / RAND_MAX; }
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
  memcpy(&result, &vec2, sizeof(WMATH_TYPE(Vec2)));
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
  vec2.v[0] = fminf(fmaxf(a.v[0], min_val), max_val);
  vec2.v[1] = fminf(fmaxf(a.v[1], min_val), max_val);
  return vec2;
}

// dot
float WMATH_DOT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  return a.v[0] * b.v[0] + a.v[1] * b.v[1];
}

// add
WMATH_TYPE(Vec2)
WMATH_ADD(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] + b.v[0];
  vec2.v[1] = a.v[1] + b.v[1];
  return vec2;
}

// addScaled
WMATH_TYPE(Vec2)
WMATH_ADD_SCALED(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float scale) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] + b.v[0] * scale;
  vec2.v[1] = a.v[1] + b.v[1] * scale;
  return vec2;
}

// sub
WMATH_TYPE(Vec2)
WMATH_SUB(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] - b.v[0];
  vec2.v[1] = a.v[1] - b.v[1];
  return vec2;
}

// angle
float WMATH_ANGLE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  float mag_1 = sqrtf(a.v[0] * a.v[0] + a.v[1] * a.v[1]);
  float mag_2 = sqrtf(b.v[0] * b.v[0] + b.v[1] * b.v[1]);
  float mag = mag_1 * mag_2;
  float cosine = mag && WMATH_DOT(Vec2)(a, b) / mag;
  return acosf(cosine);
}

// equalsApproximately
bool WMATH_EQUALS_APPROXIMATELY(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  float ep = wcn_math_get_epsilon();
  return fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep;
}

// equals
bool WMATH_EQUALS(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1]);
}

// lerp
WMATH_TYPE(Vec2)
WMATH_LERP(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float t) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t);
  vec2.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t);
  return vec2;
}

// lerpV
WMATH_TYPE(Vec2)
WMATH_LERP_V(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, WMATH_TYPE(Vec2) t) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t.v[0]);
  vec2.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t.v[1]);
  return vec2;
}

// fmax
WMATH_TYPE(Vec2)
WMATH_FMAX(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = fmaxf(a.v[0], b.v[0]);
  vec2.v[1] = fmaxf(a.v[1], b.v[1]);
  return vec2;
}

// fmin
WMATH_TYPE(Vec2)
WMATH_FMIN(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = fminf(a.v[0], b.v[0]);
  vec2.v[1] = fminf(a.v[1], b.v[1]);
  return vec2;
}

// multiplyScalar
WMATH_TYPE(Vec2)
WMATH_MULTIPLY_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] * scalar;
  vec2.v[1] = a.v[1] * scalar;
  return vec2;
}

// multiply
WMATH_TYPE(Vec2)
WMATH_MULTIPLY(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] * b.v[0];
  vec2.v[1] = a.v[1] * b.v[1];
  return vec2;
}

// divScalar
/**
 * (divScalar) if scalar is 0, returns a zero vector
 */
WMATH_TYPE(Vec2)
WMATH_DIV_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar) {
  WMATH_TYPE(Vec2) vec2;
  if (scalar == 0) {
    memset(&vec2, 0, sizeof(WMATH_TYPE(Vec2)));
    return vec2;
  }
  vec2.v[0] = a.v[0] / scalar;
  vec2.v[1] = a.v[1] / scalar;
  return vec2;
}

// div
WMATH_TYPE(Vec2)
WMATH_DIV(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = a.v[0] / b.v[0];
  vec2.v[1] = a.v[1] / b.v[1];
  return vec2;
}

// inverse
WMATH_TYPE(Vec2)
WMATH_INVERSE(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = 1.0f / a.v[0];
  vec2.v[1] = 1.0f / a.v[1];
  return vec2;
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
  return sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1]);
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(WMATH_TYPE(Vec2) v) {
  return v.v[0] * v.v[0] + v.v[1] * v.v[1];
}

// distance
float WMATH_DISTANCE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return sqrtf(dx * dx + dy * dy);
}

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return dx * dx + dy * dy;
}

// negate
WMATH_TYPE(Vec2)
WMATH_NEGATE(Vec2)(WMATH_TYPE(Vec2) a) {
  WMATH_TYPE(Vec2) vec2;
  vec2.v[0] = -a.v[0];
  vec2.v[1] = -a.v[1];
  return vec2;
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
  float len = sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1]);
  if (len > 0.00001f) {
    vec2.v[0] = v.v[0] / len;
    vec2.v[1] = v.v[1] / len;
  } else {
    vec2.v[0] = 0;
    vec2.v[1] = 0;
  }
  return vec2;
}

// rotate
WMATH_TYPE(Vec2)
WMATH_ROTATE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float rad) {
  WMATH_TYPE(Vec2) vec2;
  float p0 = a.v[0] - b.v[0];
  float p1 = a.v[1] - b.v[1];
  float s = sinf(rad);
  float c = cosf(rad);
  vec2.v[0] = p0 * c - p1 * s + b.v[0];
  vec2.v[1] = p0 * s + p1 * c + b.v[1];
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
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = ceilf(a.v[0]);
  vec3.v[1] = ceilf(a.v[1]);
  vec3.v[2] = ceilf(a.v[2]);
  return vec3;
}

// floor
WMATH_TYPE(Vec3)
WMATH_FLOOR(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = floorf(a.v[0]);
  vec3.v[1] = floorf(a.v[1]);
  vec3.v[2] = floorf(a.v[2]);
  return vec3;
}

// round
WMATH_TYPE(Vec3)
WMATH_ROUND(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = roundf(a.v[0]);
  vec3.v[1] = roundf(a.v[1]);
  vec3.v[2] = roundf(a.v[2]);
  return vec3;
}

// dot
float WMATH_DOT(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
}

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) result;
  result.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
  result.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
  result.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
  return result;
}

// length
float WMATH_LENGTH(Vec3)(WMATH_TYPE(Vec3) v) {
  return sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2]);
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec3)(WMATH_TYPE(Vec3) v) {
  return v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2];
}

// normalize
WMATH_TYPE(Vec3)
WMATH_NORMALIZE(Vec3)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Vec3) result;
  float len = sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2]);
  if (len > 0.00001f) {
    result.v[0] = v.v[0] / len;
    result.v[1] = v.v[1] / len;
    result.v[2] = v.v[2] / len;
  } else {
    result.v[0] = 0.0f;
    result.v[1] = 0.0f;
    result.v[2] = 0.0f;
  }
  return result;
}

// clamp
WMATH_TYPE(Vec3)
WMATH_CLAMP(Vec3)(WMATH_TYPE(Vec3) a, float min_val, float max_val) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = fminf(fmaxf(a.v[0], min_val), max_val);
  vec3.v[1] = fminf(fmaxf(a.v[1], min_val), max_val);
  vec3.v[2] = fminf(fmaxf(a.v[2], min_val), max_val);
  return vec3;
}

// +
WMATH_TYPE(Vec3) WMATH_ADD(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] + b.v[0];
  vec3.v[1] = a.v[1] + b.v[1];
  vec3.v[2] = a.v[2] + b.v[2];
  return vec3;
}

WMATH_TYPE(Vec3)
WMATH_ADD_SCALED(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float scalar) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] + b.v[0] * scalar;
  vec3.v[1] = a.v[1] + b.v[1] * scalar;
  vec3.v[2] = a.v[2] + b.v[2] * scalar;
  return vec3;
}

// -
WMATH_TYPE(Vec3) WMATH_SUB(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] - b.v[0];
  vec3.v[1] = a.v[1] - b.v[1];
  vec3.v[2] = a.v[2] - b.v[2];
  return vec3;
}

// angle
float WMATH_ANGLE(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  float mag_1 = WMATH_LENGTH(Vec3)(a);
  float mag_2 = WMATH_LENGTH(Vec3)(b);
  float mag = mag_1 * mag_2;
  float cosine = mag && WMATH_DOT(Vec3)(a, b) / mag;
  return acosf(cosine);
}

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  float ep = WCN_GET_EPSILON();
  return (fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep &&
          fabsf(a.v[2] - b.v[2]));
}

// =
bool WMATH_EQUALS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2]);
}

// lerp
WMATH_TYPE(Vec3)
WMATH_LERP(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float t) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t;
  vec3.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t;
  vec3.v[2] = a.v[2] + (b.v[2] - a.v[2]) * t;
  return vec3;
}

// lerpV
WMATH_TYPE(Vec3)
WMATH_LERP_V(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, WMATH_TYPE(Vec3) t) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] + (b.v[0] - a.v[0]) * t.v[0];
  vec3.v[1] = a.v[1] + (b.v[1] - a.v[1]) * t.v[1];
  vec3.v[2] = a.v[2] + (b.v[2] - a.v[2]) * t.v[2];
  return vec3;
}

// fmax
WMATH_TYPE(Vec3)
WMATH_FMAX(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = fmaxf(a.v[0], b.v[0]);
  vec3.v[1] = fmaxf(a.v[1], b.v[1]);
  vec3.v[2] = fmaxf(a.v[2], b.v[2]);
  return vec3;
}

// fmin
WMATH_TYPE(Vec3)
WMATH_FMIN(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = fminf(a.v[0], b.v[0]);
  vec3.v[1] = fminf(a.v[1], b.v[1]);
  vec3.v[2] = fminf(a.v[2], b.v[2]);
  return vec3;
}

// *
WMATH_TYPE(Vec3) WMATH_MULTIPLY(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] * b.v[0];
  vec3.v[1] = a.v[1] * b.v[1];
  vec3.v[2] = a.v[2] * b.v[2];
  return vec3;
}

// .*
WMATH_TYPE(Vec3) WMATH_MULTIPLY_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] * scalar;
  vec3.v[1] = a.v[1] * scalar;
  vec3.v[2] = a.v[2] * scalar;
}

// div
WMATH_TYPE(Vec3)
WMATH_DIV(Vec3)
(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] / b.v[0];
  vec3.v[1] = a.v[1] / b.v[1];
  vec3.v[2] = a.v[2] / b.v[2];
  return vec3;
}

// .div
WMATH_TYPE(Vec3)
WMATH_DIV_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = a.v[0] / scalar;
  vec3.v[1] = a.v[1] / scalar;
  vec3.v[2] = a.v[2] / scalar;
  return vec3;
}

// inverse
WMATH_TYPE(Vec3)
WMATH_INVERSE(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = 1.0f / a.v[0];
  vec3.v[1] = 1.0f / a.v[1];
  vec3.v[2] = 1.0f / a.v[2];
  return vec3;
}

// distance
float WMATH_DISTANCE(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  return sqrtf(dx * dx + dy * dy + dz * dz);
}

// distanceSquared
float WMATH_DISTANCE_SQ(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  float dz = a.v[2] - b.v[2];
  return dx * dx + dy * dy + dz * dz;
}

// negate
WMATH_TYPE(Vec3)
WMATH_NEGATE(Vec3)(WMATH_TYPE(Vec3) a) {
  WMATH_TYPE(Vec3) vec3;
  vec3.v[0] = -a.v[0];
  vec3.v[1] = -a.v[1];
  vec3.v[2] = -a.v[2];
  return vec3;
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
  result.v[0] = ceilf(a.v[0]);
  result.v[1] = ceilf(a.v[1]);
  result.v[2] = ceilf(a.v[2]);
  result.v[3] = ceilf(a.v[3]);
  return result;
}

WMATH_TYPE(Vec4) WMATH_FLOOR(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = floorf(a.v[0]);
  result.v[1] = floorf(a.v[1]);
  result.v[2] = floorf(a.v[2]);
  result.v[3] = floorf(a.v[3]);
  return result;
}

WMATH_TYPE(Vec4) WMATH_ROUND(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = roundf(a.v[0]);
  result.v[1] = roundf(a.v[1]);
  result.v[2] = roundf(a.v[2]);
  result.v[3] = roundf(a.v[3]);
  return result;
}

WMATH_TYPE(Vec4)
WMATH_CLAMP(Vec4)(WMATH_TYPE(Vec4) a, float min_val, float max_val) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = fminf(fmaxf(a.v[0], min_val), max_val);
  result.v[1] = fminf(fmaxf(a.v[1], min_val), max_val);
  result.v[2] = fminf(fmaxf(a.v[2], min_val), max_val);
  result.v[3] = fminf(fmaxf(a.v[3], min_val), max_val);
  return result;
}

WMATH_TYPE(Vec4) WMATH_ADD(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] + b.v[0];
  result.v[1] = a.v[1] + b.v[1];
  result.v[2] = a.v[2] + b.v[2];
  result.v[3] = a.v[3] + b.v[3];
  return result;
}

WMATH_TYPE(Vec4)
WMATH_ADD_SCALED(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, float scale) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] + b.v[0] * scale;
  result.v[1] = a.v[1] + b.v[1] * scale;
  result.v[2] = a.v[2] + b.v[2] * scale;
  result.v[3] = a.v[3] + b.v[3] * scale;
  return result;
}

WMATH_TYPE(Vec4) WMATH_SUB(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] - b.v[0];
  result.v[1] = a.v[1] - b.v[1];
  result.v[2] = a.v[2] - b.v[2];
  result.v[3] = a.v[3] - b.v[3];
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
  result.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t);
  result.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t);
  result.v[2] = WMATH_LERP(float)(a.v[2], b.v[2], t);
  result.v[3] = WMATH_LERP(float)(a.v[3], b.v[3], t);
  return result;
}

WMATH_TYPE(Vec4)
WMATH_LERP_V(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, WMATH_TYPE(Vec4) t) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = WMATH_LERP(float)(a.v[0], b.v[0], t.v[0]);
  result.v[1] = WMATH_LERP(float)(a.v[1], b.v[1], t.v[1]);
  result.v[2] = WMATH_LERP(float)(a.v[2], b.v[2], t.v[2]);
  result.v[3] = WMATH_LERP(float)(a.v[3], b.v[3], t.v[3]);
  return result;
}

WMATH_TYPE(Vec4) WMATH_FMAX(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = fmaxf(a.v[0], b.v[0]);
  result.v[1] = fmaxf(a.v[1], b.v[1]);
  result.v[2] = fmaxf(a.v[2], b.v[2]);
  result.v[3] = fmaxf(a.v[3], b.v[3]);
  return result;
}

WMATH_TYPE(Vec4) WMATH_FMIN(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = fminf(a.v[0], b.v[0]);
  result.v[1] = fminf(a.v[1], b.v[1]);
  result.v[2] = fminf(a.v[2], b.v[2]);
  result.v[3] = fminf(a.v[3], b.v[3]);
  return result;
}

WMATH_TYPE(Vec4) WMATH_MULTIPLY(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] * b.v[0];
  result.v[1] = a.v[1] * b.v[1];
  result.v[2] = a.v[2] * b.v[2];
  result.v[3] = a.v[3] * b.v[3];
  return result;
}

WMATH_TYPE(Vec4) WMATH_MULTIPLY_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] * scalar;
  result.v[1] = a.v[1] * scalar;
  result.v[2] = a.v[2] * scalar;
  result.v[3] = a.v[3] * scalar;
  return result;
}

WMATH_TYPE(Vec4) WMATH_DIV(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] / b.v[0];
  result.v[1] = a.v[1] / b.v[1];
  result.v[2] = a.v[2] / b.v[2];
  result.v[3] = a.v[3] / b.v[3];
  return result;
}

WMATH_TYPE(Vec4) WMATH_DIV_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar) {
  if (scalar == 0) {
    return WMATH_ZERO(Vec4)();
  }
  WMATH_TYPE(Vec4) result;
  result.v[0] = a.v[0] / scalar;
  result.v[1] = a.v[1] / scalar;
  result.v[2] = a.v[2] / scalar;
  result.v[3] = a.v[3] / scalar;
  return result;
}

WMATH_TYPE(Vec4) WMATH_INVERSE(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = 1.0f / a.v[0];
  result.v[1] = 1.0f / a.v[1];
  result.v[2] = 1.0f / a.v[2];
  result.v[3] = 1.0f / a.v[3];
  return result;
}

float WMATH_DOT(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b) {
  return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
}

float WMATH_LENGTH_SQ(Vec4)(WMATH_TYPE(Vec4) v) {
  return v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2] + v.v[3] * v.v[3];
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
  float len = WMATH_LENGTH(Vec4)(v);

  if (len > 0.00001f) {
    result.v[0] = v.v[0] / len;
    result.v[1] = v.v[1] / len;
    result.v[2] = v.v[2] / len;
    result.v[3] = v.v[3] / len;
  } else {
    result = WMATH_ZERO(Vec4)();
  }

  return result;
}

WMATH_TYPE(Vec4) WMATH_NEGATE(Vec4)(WMATH_TYPE(Vec4) a) {
  WMATH_TYPE(Vec4) result;
  result.v[0] = -a.v[0];
  result.v[1] = -a.v[1];
  result.v[2] = -a.v[2];
  result.v[3] = -a.v[3];
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

float WMATH_DETERMINANT(Mat4)(WMATH_TYPE(Mat4) m) {
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
}

// aim
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, aim)
(WMATH_TYPE(Vec3) position, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result = WMATH_ZERO(Mat4)();
  WMATH_TYPE(Vec3)
  z_axis = WMATH_NORMALIZE(Vec3)(WMATH_SUB(Vec3)(target, position));

  WMATH_TYPE(Vec3)
  x_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(up, z_axis));

  WMATH_TYPE(Vec3)
  y_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(z_axis, x_axis));

  result.m[0] = x_axis.v[0];
  result.m[4] = y_axis.v[0];
  result.m[8] = z_axis.v[0];
  result.m[12] = position.v[0];
}

// lookAt
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, look_at)
(WMATH_TYPE(Vec3) eye, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up) {
  WMATH_TYPE(Mat4) result = WMATH_ZERO(Mat4)();
  WMATH_TYPE(Vec3)
  z_axis = WMATH_NORMALIZE(Vec3)(WMATH_SUB(Vec3)(eye, target));
  WMATH_TYPE(Vec3)
  x_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(up, z_axis));
  WMATH_TYPE(Vec3)
  y_axis = WMATH_NORMALIZE(Vec3)(WMATH_CROSS(Vec3)(z_axis, x_axis));

  result.m[0] = x_axis.v[0];
  result.m[1] = y_axis.v[0];
  result.m[2] = z_axis.v[0];
  result.m[3] = 0;
  result.m[4] = x_axis.v[1];
  result.m[5] = y_axis.v[1];
  result.m[6] = z_axis.v[1];
  result.m[7] = 0;
  result.m[8] = x_axis.v[2];
  result.m[9] = y_axis.v[2];
  result.m[10] = z_axis.v[2];
  result.m[11] = 0;
  result.m[12] = -WMATH_DOT(Vec3)(x_axis, eye);
  result.m[13] = -WMATH_DOT(Vec3)(y_axis, eye);
  result.m[14] = -WMATH_DOT(Vec3)(z_axis, eye);
  result.m[15] = 1;

  return result;
}

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, ortho)
(float left, float right, float bottom, float top, float near, float far) {
  WMATH_TYPE(Mat4) newDst;

  newDst.m[0] = 2 / (right - left);
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[3] = 0;

  newDst.m[4] = 0;
  newDst.m[5] = 2 / (top - bottom);
  newDst.m[6] = 0;
  newDst.m[7] = 0;

  newDst.m[8] = 0;
  newDst.m[9] = 0;
  newDst.m[10] = 1 / (near - far);
  newDst.m[11] = 0;

  newDst.m[12] = (right + left) / (left - right);
  newDst.m[13] = (top + bottom) / (bottom - top);
  newDst.m[14] = near / (near - far);
  newDst.m[15] = 1;

  return newDst;
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
// slerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, slerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t) {
  WMATH_TYPE(Quat) result;
  float a_x = a.v[0];
  float a_y = a.v[1];
  float a_z = a.v[2];
  float a_w = a.v[3];
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

float WMATH_ANGLE(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  float cosOmega = WMATH_DOT(Quat)(a, b);
  return acosf(2 * cosOmega - 1.0f);
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, rotation_to)(WMATH_TYPE(Vec3) a_unit,
                              WMATH_TYPE(Vec3) b_unit) {
  float dot = WMATH_DOT(Vec3)(a_unit, b_unit);
  if (dot < -0.999999f) {
  }
}

// *
WMATH_TYPE(Quat)
WMATH_MULTIPLY(Quat)
(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) r;
  float a_x = a.v[0];
  float a_y = a.v[1];
  float a_z = a.v[2];
  float a_w = a.v[3];
  float b_x = b.v[0];
  float b_y = b.v[1];
  float b_z = b.v[2];
  float b_w = b.v[3];
  r.v[0] = a_x * b_w + a_w * b_x + a_y * b_z - a_z * b_y;
  r.v[1] = a_y * b_w + a_w * b_y + a_z * b_x - a_x * b_z;
  r.v[2] = a_z * b_w + a_w * b_z + a_x * b_y - a_y * b_x;
  r.v[3] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z;
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
  WMATH_TYPE(Quat) r;
  r.v[0] = a.v[0] - b.v[0];
  r.v[1] = a.v[1] - b.v[1];
  r.v[2] = a.v[2] - b.v[2];
  r.v[3] = a.v[3] - b.v[3];
  return r;
}

// +
WMATH_TYPE(Quat)
WMATH_ADD(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b) {
  WMATH_TYPE(Quat) r;
  r.v[0] = a.v[0] + b.v[0];
  r.v[1] = a.v[1] + b.v[1];
  r.v[2] = a.v[2] + b.v[2];
  r.v[3] = a.v[3] + b.v[3];
  return r;
}

// inverse
WMATH_TYPE(Quat)
WMATH_INVERSE(Quat)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Quat) r;
  float a_0 = q.v[0];
  float a_1 = q.v[1];
  float a_2 = q.v[2];
  float a_3 = q.v[3];
  float dot = a_0 * a_0 + a_1 * a_1 + a_2 * a_2 + a_3 * a_3;
  float invDot = dot ? 1 / dot : 0;
  r.v[0] = -a_0 * invDot;
  r.v[1] = -a_1 * invDot;
  r.v[2] = -a_2 * invDot;
  r.v[3] = a_3 * invDot;
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
  return WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){
      .v_x = a.v[0] / v,
      .v_y = a.v[1] / v,
      .v_z = a.v[2] / v,
      .v_w = a.v[3] / v,
  });
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

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_quat)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat3) newDst;

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

  return newDst;
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
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_quat)(WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Mat4) newDst;

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
  newDst.m[3] = 0; // 0
  newDst.m[4] = yx - wz;
  newDst.m[5] = 1 - xx - zz;
  newDst.m[6] = zy + wx;
  newDst.m[7] = 0; // 1
  newDst.m[8] = zx + wy;
  newDst.m[9] = zy - wx;
  newDst.m[10] = 1 - xx - yy;
  newDst.m[11] = 0; // 2
  newDst.m[12] = 0;
  newDst.m[13] = 0;
  newDst.m[14] = 0;
  newDst.m[15] = 1; // 3

  return newDst;
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

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat4)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec2) vec2;
  float x = v.v[0];
  float y = v.v[1];
  vec2.v[0] = x * m.m[0] + y * m.m[4] + m.m[12];
  vec2.v[1] = x * m.m[1] + y * m.m[5] + m.m[13];
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
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  float w = (m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15]) || 1;
  vec3.v[0] = (x * m.m[0] + y * m.m[4] + z * m.m[8] + m.m[12]) / w;
  vec3.v[1] = (x * m.m[1] + y * m.m[5] + z * m.m[9] + m.m[13]) / w;
  vec3.v[2] = (x * m.m[2] + y * m.m[6] + z * m.m[10] + m.m[14]) / w;
  return vec3;
}

// vec3 transformMat4Upper3x3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4_upper3x3)(WMATH_TYPE(Vec3) v,
                                          WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec3) vec3;
  // v0 v1 v2
  float v0 = v.v[0];
  float v1 = v.v[1];
  float v2 = v.v[2];

  vec3.v[0] = v0 * m.m[0] + v1 * m.m[4] + v2 * m.m[8];
  vec3.v[1] = v0 * m.m[1] + v1 * m.m[5] + v2 * m.m[9];
  vec3.v[2] = v0 * m.m[2] + v1 * m.m[6] + v2 * m.m[10];
  return vec3;
}

// vec3 transformMat3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat3)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) r;

  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  r.v[0] = x * m.m[0] + y * m.m[4] + z * m.m[8];
  r.v[1] = x * m.m[1] + y * m.m[5] + z * m.m[9];
  r.v[2] = x * m.m[2] + y * m.m[6] + z * m.m[10];

  return r;
}

// vec3 transformQuat
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_quat)
(WMATH_TYPE(Vec3) v, WMATH_TYPE(Quat) q) {
  WMATH_TYPE(Vec3) r;

  float qx = q.v[0];
  float qy = q.v[1];
  float qz = q.v[2];
  float w2 = q.v[3] * 2;

  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];

  float uvX = qy * z - qz * y;
  float uvY = qz * x - qx * z;
  float uvZ = qx * y - qy * x;

  r.v[0] = x + uvX * w2 + (qy * uvZ - qz * uvY) * 2;
  r.v[1] = y + uvY * w2 + (qz * uvX - qx * uvZ) * 2;
  r.v[2] = z + uvZ * w2 + (qx * uvY - qy * uvX) * 2;

  return r;
}

// Quat fromMat
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat4)(WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Quat) r;
  float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    float invRoot = 0.5f / root;
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

    int root = sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    float invRoot = 0.5f / root;
    r.v[3] = (m.m[j * 4 + k] - m.m[k * 4 + j]) * invRoot;
    r.v[j] = (m.m[j * 4 + i] + m.m[i * 4 + j]) * invRoot;
    r.v[k] = (m.m[k * 4 + i] + m.m[i * 4 + k]) * invRoot;
  }
  return r;
}

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat3)(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Quat) r;
  float trace = m.m[0] + m.m[5] + m.m[10];
  if (trace > 0.0) {
    float root = sqrtf(trace + 1.0f);
    r.v[3] = 0.5f * root;
    float invRoot = 0.5f / root;
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

    int root = sqrtf(m.m[i * 4 + i] - m.m[j * 4 + j] - m.m[k * 4 + k] + 1.0f);
    r.v[i] = 0.5f * root;
    float invRoot = 0.5f / root;
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
  float s_x = sinf(x_half_angle);
  float c_x = cosf(x_half_angle);
  float s_y = sinf(y_half_angle);
  float c_y = cosf(y_half_angle);
  float s_z = sinf(z_half_angle);
  float c_z = cosf(z_half_angle);
  switch (order) {
  case WCN_Math_RotationOrder_XYZ:
    r.v[0] = s_x * c_y * c_z + c_x * s_y * s_z;
    r.v[1] = c_x * s_y * c_z - s_x * c_y * s_z;
    r.v[2] = c_x * c_y * s_z + s_x * s_y * c_z;
    r.v[3] = c_x * c_y * c_z - s_x * s_y * s_z;
    break;
  case WCN_Math_RotationOrder_XZY:
    r.v[0] = s_x * c_y * c_z - c_x * s_y * s_z;
    r.v[1] = c_x * s_y * c_z - s_x * c_y * s_z;
    r.v[2] = c_x * c_y * s_z + s_x * s_y * c_z;
    r.v[3] = c_x * c_y * c_z + s_x * s_y * s_z;
    break;
  case WCN_Math_RotationOrder_YXZ:
    r.v[0] = s_y * c_x * c_z + c_y * s_x * s_z;
    r.v[1] = c_y * s_x * c_z - s_y * c_x * s_z;
    r.v[2] = c_y * c_x * s_z - s_y * s_x * c_z;
    r.v[3] = c_y * c_x * c_z + s_y * s_x * s_z;
    break;
  case WCN_Math_RotationOrder_YZX:
    r.v[0] = s_y * c_x * c_z + c_y * s_x * s_z;
    r.v[1] = c_y * s_x * c_z + s_y * c_x * s_z;
    r.v[2] = c_y * c_x * s_z - s_y * s_x * c_z;
    r.v[3] = c_y * c_x * c_z - s_y * s_x * s_z;
    break;
  case WCN_Math_RotationOrder_ZXY:
    r.v[0] = s_z * c_x * c_y - c_z * s_x * s_y;
    r.v[1] = c_z * s_x * c_y + s_z * c_x * s_y;
    r.v[2] = c_z * c_x * s_y + s_z * s_x * c_y;
    r.v[3] = c_z * c_x * c_y - s_z * s_x * s_y;
    break;
  case WCN_Math_RotationOrder_ZYX:
    r.v[0] = s_z * c_x * c_y - c_z * s_x * s_y;
    r.v[1] = c_z * s_x * c_y + s_z * c_x * s_y;
    r.v[2] = c_z * c_x * s_y - s_z * s_x * c_y;
    r.v[3] = c_z * c_x * c_y + s_z * s_x * s_y;
    break;
  default:
    break;
    return r;
  }
}

// 3D
// vec3 getTranslation
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_translation)(WMATH_TYPE(Mat4) m) {
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = m.m[12],
      .v_y = m.m[13],
      .v_z = m.m[14],
  });
}

// vec3 getAxis
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_axis)(WMATH_TYPE(Mat4) m, int axis) {
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
  float x_x = m.m[0];
  float x_y = m.m[1];
  float x_z = m.m[2];
  float y_x = m.m[4];
  float y_y = m.m[5];
  float y_z = m.m[6];
  float z_x = m.m[8];
  float z_y = m.m[9];
  float z_z = m.m[10];
  return WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){
      .v_x = sqrtf(x_x * x_x + x_y * x_y + x_z * x_z),
      .v_y = sqrtf(y_x * y_x + y_y * y_y + y_z * y_z),
      .v_z = sqrtf(z_x * z_x + z_y * z_y + z_z * z_z),
  });
}

// vec3 rotateX
WMATH_TYPE(Vec3)
WMATH_ROTATE_X(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;
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
  return vec3;
}

// vec3 rotateY
WMATH_TYPE(Vec3)
WMATH_ROTATE_Y(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = sinf(rad) * p.v[2] + cosf(rad) * p.v[0];
  r.v[1] = p.v[1];
  r.v[2] = cosf(rad) * p.v[2] - sinf(rad) * p.v[0];
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];
  return vec3;
}

// vec3 rotateZ
WMATH_TYPE(Vec3)
WMATH_ROTATE_Z(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad) {
  WMATH_TYPE(Vec3) vec3;
  WMATH_TYPE(Vec3) p;
  WMATH_TYPE(Vec3) r;
  p.v[0] = a.v[0] - b.v[0];
  p.v[1] = a.v[1] - b.v[1];
  p.v[2] = a.v[2] - b.v[2];
  r.v[0] = cosf(rad) * p.v[0] - sinf(rad) * p.v[1];
  r.v[1] = sinf(rad) * p.v[0] + cosf(rad) * p.v[1];
  r.v[2] = p.v[2];
  vec3.v[0] = r.v[0] + b.v[0];
  vec3.v[1] = r.v[1] + b.v[1];
  vec3.v[2] = r.v[2] + b.v[2];
  return vec3;
}

// vec4 transformMat4
WMATH_TYPE(Vec4)
WMATH_CALL(Vec4, transform_mat4)(WMATH_TYPE(Vec4) v, WMATH_TYPE(Mat4) m) {
  WMATH_TYPE(Vec4) result;
  float x = v.v[0];
  float y = v.v[1];
  float z = v.v[2];
  float w = v.v[3];

  result.v[0] = m.m[0] * x + m.m[4] * y + m.m[8] * z + m.m[12] * w;
  result.v[1] = m.m[1] * x + m.m[5] * y + m.m[9] * z + m.m[13] * w;
  result.v[2] = m.m[2] * x + m.m[6] * y + m.m[10] * z + m.m[14] * w;
  result.v[3] = m.m[3] * x + m.m[7] * y + m.m[11] * z + m.m[15] * w;

  return result;
}

// Quat rotate_x
WMATH_TYPE(Quat)
WMATH_ROTATE_X(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;
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
  return result;
}

// Quat rotate_y
WMATH_TYPE(Quat)
WMATH_ROTATE_Y(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;
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
  return result;
}

// Quat rotate_z
WMATH_TYPE(Quat)
WMATH_ROTATE_Z(Quat)(WMATH_TYPE(Quat) q, float angleInRadians) {
  WMATH_TYPE(Quat) result;
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
  return result;
}

// Mat3 rotate
WMATH_TYPE(Mat3)
WMATH_ROTATE(Mat3)
(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

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

  return newDst;
}

// Mat3 rotate x
WMATH_TYPE(Mat3)
WMATH_ROTATE_X(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
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
  return newDst;
}

// Mat3 rotate y
WMATH_TYPE(Mat3)
WMATH_ROTATE_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

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
  newDst.m[2] = 0;
  newDst.m[4] = -s;
  newDst.m[5] = c;
  newDst.m[6] = 0;
  newDst.m[8] = 0;
  newDst.m[9] = 0;
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
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = c;
  newDst.m[6] = s;
  newDst.m[8] = 0;
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
  newDst.m[1] = 0;
  newDst.m[2] = -s;
  newDst.m[4] = 0;
  newDst.m[5] = 1;
  newDst.m[6] = 0;
  newDst.m[8] = s;
  newDst.m[9] = 0;
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
 * @param axis - The axis  0 = x, 1 = y;
 * @param dst - The matrix to set. If not passed a new one is created.
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
(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec2) result;
  float xx = m.m[0];
  float xy = m.m[1];
  float yx = m.m[4];
  float yy = m.m[5];
  result.v[0] = sqrtf(xx * xx + xy * xy);
  result.v[1] = sqrtf(yx * yx + yy * yy);
  return result;
}

// Mat3 get_3D_scaling
WMATH_TYPE(Vec3)
WMATH_CALL(Mat3, get_3D_scaling)
(WMATH_TYPE(Mat3) m) {
  WMATH_TYPE(Vec3) result;
  float xx = m.m[0];
  float xy = m.m[1];
  float xz = m.m[2];
  float yx = m.m[4];
  float yy = m.m[5];
  float yz = m.m[6];
  float zx = m.m[8];
  float zy = m.m[9];
  float zz = m.m[10];

  result.v[0] = sqrtf(xx * xx + xy * xy + xz * xz);
  result.v[1] = sqrtf(yx * yx + yy * yy + yz * yz);
  result.v[2] = sqrtf(zx * zx + zy * zy + zz * zz);

  return result;
}

// Mat3 get_translation
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_translation)
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
 * @param a - The matrix.
 * @param v - The vector.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The matrix with translation set.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, set_translation)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
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
}

// Mat3 translation
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, translation)
(WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  newDst.m[0] = 1;
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = 1;
  newDst.m[6] = 0;
  newDst.m[8] = v.v[0];
  newDst.m[9] = v.v[1];
  newDst.m[10] = 1;

  return newDst;
}

// translate
/**
 * Translates the given 3-by-3 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by which to translate.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, translate)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_COPY(Mat3)(m);
  float v0 = v.v[0];
  float v1 = v.v[1];

  float m00 = m.m[0];
  float m01 = m.m[1];
  float m02 = m.m[2];
  float m10 = m.m[1 * 4 + 0];
  float m11 = m.m[1 * 4 + 1];
  float m12 = m.m[1 * 4 + 2];
  float m20 = m.m[2 * 4 + 0];
  float m21 = m.m[2 * 4 + 1];
  float m22 = m.m[2 * 4 + 2];

  if (!WMATH_EQUALS(Mat3)(newDst, m)) {
    newDst.m[0] = m00;
    newDst.m[1] = m01;
    newDst.m[2] = m02;
    newDst.m[4] = m10;
    newDst.m[5] = m11;
    newDst.m[6] = m12;
  }

  newDst.m[8] = m00 * v0 + m10 * v1 + m20;
  newDst.m[9] = m01 * v0 + m11 * v1 + m21;
  newDst.m[10] = m02 * v0 + m12 * v1 + m22;

  return newDst;
}

// All Type Scale Impl
WMATH_TYPE(Vec2)
WMATH_SCALE(Vec2)
(WMATH_TYPE(Vec2) v, float scale) {
  WMATH_TYPE(Vec2) result;
  result.v[0] = v.v[0] * scale;
  result.v[1] = v.v[1] * scale;
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

  return newDst;
}

WMATH_TYPE(Mat4)
WMATH_SCALE(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat4) newDst = WMATH_ZERO(Mat4)();

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

  return newDst;
}

// Mat3 scale3D
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scale3D)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
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
  return newDst;
}

// Mat3 scaling
/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has two
 * entries.
 * @param v - A vector of
 *     2 entries specifying the factor by which to scale in each dimension.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling)
(WMATH_TYPE(Vec2) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = v.v[0];
  newDst.m[1] = 1;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  newDst.m[6] = 0;
  newDst.m[8] = 0;
  newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     3 entries specifying the factor by which to scale in each dimension.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling3D)(WMATH_TYPE(Vec3) v) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();

  newDst.m[0] = v.v[0];
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = v.v[1];
  newDst.m[6] = 0;
  newDst.m[8] = 0;
  newDst.m[9] = 0;
  newDst.m[10] = v.v[2];

  return newDst;
}

// Mat3 uniform_scale
/**
 * Scales the given 3-by-3 matrix in the X and Y dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
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
  return newDst;
}

// Mat3 uniform_scale_3D
/**
 * Scales the given 3-by-3 matrix in each dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale_3D)
(WMATH_TYPE(Mat3) m, float s) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = s * m.m[0];
  newDst.m[1] = s * m.m[1];
  newDst.m[2] = s * m.m[2];
  newDst.m[4] = s * m.m[4];
  newDst.m[5] = s * m.m[5];
  newDst.m[6] = s * m.m[6];
  newDst.m[8] = s * m.m[8];
  newDst.m[9] = s * m.m[9];
  newDst.m[10] = s * m.m[10];
  return newDst;
}

// Mat3 uniform_scaling
/**
 * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
 * @param s - Amount to scale
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling)
(float s) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = s;
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = s;
  newDst.m[6] = 0;
  newDst.m[8] = 0;
  newDst.m[9] = 0;
  newDst.m[10] = 1;
  return newDst;
}

// Mat3 uniform_scaling_3D
/**
 * Creates a 3-by-3 matrix which scales uniformly in each dimension
 * @param s - Amount to scale
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling_3D)
(float s) {
  WMATH_TYPE(Mat3) newDst = WMATH_ZERO(Mat3)();
  newDst.m[0] = s;
  newDst.m[1] = 0;
  newDst.m[2] = 0;
  newDst.m[4] = 0;
  newDst.m[5] = s;
  newDst.m[6] = 0;
  newDst.m[8] = 0;
  newDst.m[9] = 0;
  newDst.m[10] = s;
  return newDst;
}

#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_H
