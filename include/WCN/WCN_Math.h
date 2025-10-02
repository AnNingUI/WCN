#ifndef WCN_MATH_H
#define WCN_MATH_H

// SIMD includes on supported platforms
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Config
extern float EPSILON;
float wcn_math_set_epsilon(float epsilon);
float wcn_math_get_epsilon();
#define WCN_GET_EPSILON() wcn_math_get_epsilon()

// CONST

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
  wcn_math_##WCN_Math_TYPE##_equals_approximately

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
  wcn_math_##WCN_Math_TYPE##_multiply_scalar

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
#define WMATH_LERP_V(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_lerp_v

// vec length
#define WMATH_LENGTH(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_length

// vec length squared
#define WMATH_LENGTH_SQ(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_length_squared

// vec set_length
#define WMATH_SET_LENGTH(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_set_length

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
#define WMATH_ADD_SCALED(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_add_scaled

// vec angle
#define WMATH_ANGLE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_angle

// vec fmax
#define WMATH_FMAX(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_fmax

// vec fmin
#define WMATH_FMIN(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_fmin

// vec div
#define WMATH_DIV(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_div

// vec div_scalar
#define WMATH_DIV_SCALAR(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_div_scalar

// distance
#define WMATH_DISTANCE(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distance

// dist
#define WMATH_DIST(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distance

// distSquared
#define WMATH_DISTANCE_SQ(WCN_Math_TYPE)                                       \
  wcn_math_##WCN_Math_TYPE##_distanceSquared

// dist_sq
#define WMATH_DIST_SQ(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_distance_squared

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
#define WMATH_ROTATE_X(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotate_x

// rotate_y
#define WMATH_ROTATE_Y(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotate_y

// rotate_z
#define WMATH_ROTATE_Z(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotate_z

// mat rotation
#define WMATH_ROTATION(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotation

// mat rotation_x
#define WMATH_ROTATION_X(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotation_x

// mat rotation_y
#define WMATH_ROTATION_Y(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotation_y

// mat rotation_z
#define WMATH_ROTATION_Z(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_rotation_z

// getTranslation
#define WMATH_GET_TRANSLATION(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##_get_translation

// setTranslation
#define WMATH_SET_TRANSLATION(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##_set_translation

// translation
#define WMATH_TRANSLATION(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_translation

#define T$(WCN_Math_TYPE) WMATH_TYPE(WCN_Math_TYPE)

#define INIT$(WCN_Math_TYPE, ...)               \
    WMATH_CREATE(WCN_Math_TYPE)(                \
       (WMATH_CREATE_TYPE(WCN_Math_TYPE)){      \
          __VA_ARGS__                           \
        }                                       \
    )

// BEGIN Utils

#define WMATH_DEG2RED(degrees) (degrees * 0.017453292519943295f)

#define WMATH_RED2DEG(radians) (radians * 57.29577951308232f)

// #define WMATH_NUM_LERP(a, b, t) ((a) + ((b) - (a)) * (t))
// Impl of lerp for float, double, int, and float_t
// ==================================================================
static  int WMATH_LERP(int)(const int a, const int b, const float t) {
  return (int)(a + ((b) - (a)) * t);
}
static  float WMATH_LERP(float)(const float a, const float b, const float t) {
  return (a + ((b) - (a)) * t);
}
static  double WMATH_LERP(double)(const double a, const double b, const double t) {
  return (a + ((b) - (a)) * t);
}

static  float_t WMATH_LERP(float_t)(const float_t a, const float_t b, const float_t t) {
  return (a + ((b) - (a)) * t);
}

static  double_t WMATH_LERP(double_t)(const double_t a, const double_t b, const double_t t) {
  return (a + ((b) - (a)) * t);
}
// ==================================================================

// Impl of random for float, double, int, and float_t
// ==================================================================
static inline int WMATH_RANDOM(int)() { return rand(); }

static inline float WMATH_RANDOM(float)() { return ((float)rand()) / RAND_MAX; }

static inline double WMATH_RANDOM(double)() { return ((double)rand()) / RAND_MAX; }

static inline float_t WMATH_RANDOM(float_t)() { return ((float_t)rand()) / RAND_MAX; }

static inline double_t WMATH_RANDOM(double_t)() { return ((double_t)rand()) / RAND_MAX; }
// ==================================================================

#define WMATH_INVERSE_LERP(a, b, t)                                            \
  (fabsf((b) - (a)) < wcn_math_get_epsilon() ? 0.0f                            \
                                             : (((t) - (a)) / ((b) - (a))))
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
  float v[4] __attribute__((aligned(16)));
} WMATH_TYPE(Quat);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
  float v_w;
} WMATH_CREATE_TYPE(Quat);

enum WCN_Math_RotationOrder {
  WCN_Math_RotationOrder_XYZ = 0,
  WCN_Math_RotationOrder_XZY = 1,
  WCN_Math_RotationOrder_YXZ = 2,
  WCN_Math_RotationOrder_YZX = 3,
  WCN_Math_RotationOrder_ZXY = 4,
  WCN_Math_RotationOrder_ZYX = 5,
};
#define WCN_MATH_IS_VALID_ROTATION_ORDER(order)                                \
  ((order) >= WCN_Math_RotationOrder_XYZ &&                                    \
   (order) <= WCN_Math_RotationOrder_ZYX)

// 或者使用数组大小来确保一致性
#define WCN_MATH_ROTATION_ORDER_COUNT 6
extern const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4];

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
  float v[4] __attribute__((aligned(16)));
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
WMATH_CREATE(Vec2)(WMATH_CREATE_TYPE(Vec2) vec2_c);

// set
WMATH_TYPE(Vec2)
WMATH_SET(Vec2)(WMATH_TYPE(Vec2) vec2, float x, float y);

// copy
WMATH_TYPE(Vec2)
WMATH_COPY(Vec2)(WMATH_TYPE(Vec2) vec2);

// 0
WMATH_TYPE(Vec2)
WMATH_ZERO(Vec2)();

// 1
WMATH_TYPE(Vec2)
WMATH_IDENTITY(Vec2)();

// ceil
WMATH_TYPE(Vec2)
WMATH_CEIL(Vec2)(WMATH_TYPE(Vec2) a);

// floor
WMATH_TYPE(Vec2)
WMATH_FLOOR(Vec2)(WMATH_TYPE(Vec2) a);

// round
WMATH_TYPE(Vec2)
WMATH_ROUND(Vec2)(WMATH_TYPE(Vec2) a);

// clamp
WMATH_TYPE(Vec2)
WMATH_CLAMP(Vec2)(WMATH_TYPE(Vec2) a, float min_val, float max_val);

// dot
float WMATH_DOT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// add
WMATH_TYPE(Vec2)
WMATH_ADD(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// addScaled
WMATH_TYPE(Vec2)
WMATH_ADD_SCALED(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float scale);

// sub
WMATH_TYPE(Vec2)
WMATH_SUB(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// angle
float WMATH_ANGLE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// equalsApproximately
bool WMATH_EQUALS_APPROXIMATELY(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// equals
bool WMATH_EQUALS(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// lerp
WMATH_TYPE(Vec2)
WMATH_LERP(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float t);

// lerpV
WMATH_TYPE(Vec2)
WMATH_LERP_V(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, WMATH_TYPE(Vec2) t);

// fmax
WMATH_TYPE(Vec2)
WMATH_FMAX(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// fmin
WMATH_TYPE(Vec2)
WMATH_FMIN(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// multiplyScalar
WMATH_TYPE(Vec2)
WMATH_MULTIPLY_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar);

// multiply
WMATH_TYPE(Vec2)
WMATH_MULTIPLY(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// divScalar
/**
 * (divScalar) if scalar is 0, returns a zero vector
 */
WMATH_TYPE(Vec2)
WMATH_DIV_SCALAR(Vec2)(WMATH_TYPE(Vec2) a, float scalar);

// div
WMATH_TYPE(Vec2)
WMATH_DIV(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// inverse
WMATH_TYPE(Vec2)
WMATH_INVERSE(Vec2)(WMATH_TYPE(Vec2) a);

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// length
float WMATH_LENGTH(Vec2)(WMATH_TYPE(Vec2) v);

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(WMATH_TYPE(Vec2) v);

// distance
float WMATH_DISTANCE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// negate
WMATH_TYPE(Vec2)
WMATH_NEGATE(Vec2)(WMATH_TYPE(Vec2) a);

// random
WMATH_TYPE(Vec2)
WMATH_RANDOM(Vec2)(float scale);

// normalize
WMATH_TYPE(Vec2)
WMATH_NORMALIZE(Vec2)(WMATH_TYPE(Vec2) v);

// rotate
WMATH_TYPE(Vec2)
WMATH_ROTATE(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b, float rad);

// set length
WMATH_TYPE(Vec2)
WMATH_SET_LENGTH(Vec2)(WMATH_TYPE(Vec2) a, float length);

// truncate
WMATH_TYPE(Vec2)
WMATH_TRUNCATE(Vec2)(WMATH_TYPE(Vec2) a, float length);

// midpoint
WMATH_TYPE(Vec2)
WMATH_MIDPOINT(Vec2)(WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b);

// END Vec2

// BEGIN Vec3

WMATH_TYPE(Vec3) WMATH_CREATE(Vec3)(WMATH_CREATE_TYPE(Vec3) vec3_c);

// copy
WMATH_TYPE(Vec3)
WMATH_COPY(Vec3)(WMATH_TYPE(Vec3) a);

// set
WMATH_TYPE(Vec3)
WMATH_SET(Vec3)(WMATH_TYPE(Vec3) a, float x, float y, float z);

// 0
WMATH_TYPE(Vec3)
WMATH_ZERO(Vec3)();

// ceil
WMATH_TYPE(Vec3)
WMATH_CEIL(Vec3)(WMATH_TYPE(Vec3) a);

// floor
WMATH_TYPE(Vec3)
WMATH_FLOOR(Vec3)(WMATH_TYPE(Vec3) a);

// round
WMATH_TYPE(Vec3)
WMATH_ROUND(Vec3)(WMATH_TYPE(Vec3) a);

// dot
float WMATH_DOT(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// cross
WMATH_TYPE(Vec3)
WMATH_CROSS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// length
float WMATH_LENGTH(Vec3)(WMATH_TYPE(Vec3) v);

// lengthSquared
float WMATH_LENGTH_SQ(Vec3)(WMATH_TYPE(Vec3) v);

// normalize
WMATH_TYPE(Vec3)
WMATH_NORMALIZE(Vec3)(WMATH_TYPE(Vec3) v);

// clamp
WMATH_TYPE(Vec3)
WMATH_CLAMP(Vec3)(WMATH_TYPE(Vec3) a, float min_val, float max_val);

// +
WMATH_TYPE(Vec3) WMATH_ADD(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

WMATH_TYPE(Vec3)
WMATH_ADD_SCALED(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float scalar);

// -
WMATH_TYPE(Vec3) WMATH_SUB(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// angle
float WMATH_ANGLE(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// =
bool WMATH_EQUALS(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// lerp
WMATH_TYPE(Vec3)
WMATH_LERP(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float t);

// lerpV
WMATH_TYPE(Vec3)
WMATH_LERP_V(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, WMATH_TYPE(Vec3) t);

// fmax
WMATH_TYPE(Vec3)
WMATH_FMAX(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// fmin
WMATH_TYPE(Vec3)
WMATH_FMIN(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// *
WMATH_TYPE(Vec3) WMATH_MULTIPLY(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// .*
WMATH_TYPE(Vec3) WMATH_MULTIPLY_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar);

// div
WMATH_TYPE(Vec3)
WMATH_DIV(Vec3)
(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// .div
WMATH_TYPE(Vec3)
WMATH_DIV_SCALAR(Vec3)(WMATH_TYPE(Vec3) a, float scalar);

// inverse
WMATH_TYPE(Vec3)
WMATH_INVERSE(Vec3)(WMATH_TYPE(Vec3) a);

// distance
float WMATH_DISTANCE(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// distanceSquared
float WMATH_DISTANCE_SQ(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// negate
WMATH_TYPE(Vec3)
WMATH_NEGATE(Vec3)(WMATH_TYPE(Vec3) a);

// random
WMATH_TYPE(Vec3)
WMATH_RANDOM(Vec3)(float scale);

// setLength
WMATH_TYPE(Vec3)
WMATH_SET_LENGTH(Vec3)(WMATH_TYPE(Vec3) v, float length);

// truncate
WMATH_TYPE(Vec3)
WMATH_TRUNCATE(Vec3)(WMATH_TYPE(Vec3) v, float max_length);

// midpoint
WMATH_TYPE(Vec3)
WMATH_MIDPOINT(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b);

// END Vec3

// BEGIN Vec4

WMATH_TYPE(Vec4) WMATH_CREATE(Vec4)(WMATH_CREATE_TYPE(Vec4) vec4_c);

WMATH_TYPE(Vec4)
WMATH_SET(Vec4)(WMATH_TYPE(Vec4) vec4, float x, float y, float z, float w);

WMATH_TYPE(Vec4) WMATH_COPY(Vec4)(WMATH_TYPE(Vec4) vec4);

// 0
WMATH_TYPE(Vec4) WMATH_ZERO(Vec4)();

// 1
WMATH_TYPE(Vec4) WMATH_IDENTITY(Vec4)();

WMATH_TYPE(Vec4) WMATH_CEIL(Vec4)(WMATH_TYPE(Vec4) a);

WMATH_TYPE(Vec4) WMATH_FLOOR(Vec4)(WMATH_TYPE(Vec4) a);

WMATH_TYPE(Vec4) WMATH_ROUND(Vec4)(WMATH_TYPE(Vec4) a);

WMATH_TYPE(Vec4)
WMATH_CLAMP(Vec4)(WMATH_TYPE(Vec4) a, float min_val, float max_val);

WMATH_TYPE(Vec4) WMATH_ADD(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4)
WMATH_ADD_SCALED(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, float scale);

WMATH_TYPE(Vec4) WMATH_SUB(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

bool WMATH_EQUALS_APPROXIMATELY(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

bool WMATH_EQUALS(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4)
WMATH_LERP(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, float t);

WMATH_TYPE(Vec4)
WMATH_LERP_V(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b, WMATH_TYPE(Vec4) t);

WMATH_TYPE(Vec4) WMATH_FMAX(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4) WMATH_FMIN(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4) WMATH_MULTIPLY(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4) WMATH_MULTIPLY_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar);

WMATH_TYPE(Vec4) WMATH_DIV(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4) WMATH_DIV_SCALAR(Vec4)(WMATH_TYPE(Vec4) a, float scalar);

WMATH_TYPE(Vec4) WMATH_INVERSE(Vec4)(WMATH_TYPE(Vec4) a);

float WMATH_DOT(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

float WMATH_LENGTH_SQ(Vec4)(WMATH_TYPE(Vec4) v);

float WMATH_LENGTH(Vec4)(WMATH_TYPE(Vec4) v);

float WMATH_DISTANCE_SQ(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

float WMATH_DISTANCE(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

WMATH_TYPE(Vec4) WMATH_NORMALIZE(Vec4)(WMATH_TYPE(Vec4) v);

WMATH_TYPE(Vec4) WMATH_NEGATE(Vec4)(WMATH_TYPE(Vec4) a);

WMATH_TYPE(Vec4) WMATH_SET_LENGTH(Vec4)(WMATH_TYPE(Vec4) v, float length);

WMATH_TYPE(Vec4) WMATH_TRUNCATE(Vec4)(WMATH_TYPE(Vec4) v, float max_length);

WMATH_TYPE(Vec4) WMATH_MIDPOINT(Vec4)(WMATH_TYPE(Vec4) a, WMATH_TYPE(Vec4) b);

// END Vec4

// BEGIN Mat3

WMATH_TYPE(Mat3) WMATH_IDENTITY(Mat3)();

WMATH_TYPE(Mat3) WMATH_ZERO(Mat3)();

WMATH_TYPE(Mat3)
WMATH_CREATE(Mat3)(WMATH_CREATE_TYPE(Mat3) mat3_c);

WMATH_TYPE(Mat3)
WMATH_COPY(Mat3)(WMATH_TYPE(Mat3) mat);

WMATH_TYPE(Mat3)
WMATH_SET(Mat3)(WMATH_TYPE(Mat3) mat, float m00, float m01, float m02,
                float m10, float m11, float m12, float m20, float m21,
                float m22);

WMATH_TYPE(Mat3)
WMATH_NEGATE(Mat3)(WMATH_TYPE(Mat3) mat);

WMATH_TYPE(Mat3)
WMATH_TRANSPOSE(Mat3)(WMATH_TYPE(Mat3) mat);

// SIMD optimized matrix addition
WMATH_TYPE(Mat3)
WMATH_ADD(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b);

// SIMD optimized matrix subtraction
WMATH_TYPE(Mat3)
WMATH_SUB(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b);

// SIMD optimized scalar multiplication
WMATH_TYPE(Mat3)
WMATH_MULTIPLY_SCALAR(Mat3)(WMATH_TYPE(Mat3) a, float b);

WMATH_TYPE(Mat3)
WMATH_INVERSE(Mat3)(WMATH_TYPE(Mat3) a);

// Optimized matrix multiplication
WMATH_TYPE(Mat3)
WMATH_MULTIPLY(Mat3)(WMATH_TYPE(Mat3) a, WMATH_TYPE(Mat3) b);

float WMATH_DETERMINANT(Mat3)(WMATH_TYPE(Mat3) m);

// END Mat3

// BEGIN Mat4

// 0 add 1 Mat4

WMATH_TYPE(Mat4) WMATH_IDENTITY(Mat4)();

WMATH_TYPE(Mat4) WMATH_ZERO(Mat4)();

// Init Mat4

WMATH_TYPE(Mat4) WMATH_CREATE(Mat4)(WMATH_CREATE_TYPE(Mat4) mat4_c);

WMATH_TYPE(Mat4) WMATH_COPY(Mat4)(WMATH_TYPE(Mat4) mat);

WMATH_TYPE(Mat4)
WMATH_SET(Mat4)(WMATH_TYPE(Mat4) mat, float m00, float m01, float m02,
                float m03, float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23, float m30,
                float m31, float m32, float m33);

WMATH_TYPE(Mat4)
WMATH_NEGATE(Mat4)(WMATH_TYPE(Mat4) mat);

bool WMATH_EQUALS(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b);

bool WMATH_EQUALS_APPROXIMATELY(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b);

// + add - Mat4
WMATH_TYPE(Mat4) WMATH_ADD(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b);

WMATH_TYPE(Mat4) WMATH_SUB(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b);

// .* Mat4

WMATH_TYPE(Mat4) WMATH_MULTIPLY_SCALAR(Mat4)(WMATH_TYPE(Mat4) a, float b);

// * Mat4

WMATH_TYPE(Mat4)
WMATH_MULTIPLY(Mat4)(WMATH_TYPE(Mat4) a, WMATH_TYPE(Mat4) b);

WMATH_TYPE(Mat4)
WMATH_INVERSE(Mat4)(WMATH_TYPE(Mat4) a);

WMATH_TYPE(Mat4)
WMATH_TRANSPOSE(Mat4)(WMATH_TYPE(Mat4) a);

float WMATH_DETERMINANT(Mat4)(WMATH_TYPE(Mat4) m);

// aim
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, aim)
(WMATH_TYPE(Vec3) position, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up);

// lookAt
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, look_at)
(WMATH_TYPE(Vec3) eye, WMATH_TYPE(Vec3) target, WMATH_TYPE(Vec3) up);

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, ortho)
(float left, float right, float bottom, float top, float near, float far);

// END Mat4

// BEGIN Quat

// 0
WMATH_TYPE(Quat)
WMATH_ZERO(Quat)(void);

// 1
WMATH_TYPE(Quat)
WMATH_IDENTITY(Quat)(void);

WMATH_TYPE(Quat)
WMATH_CREATE(Quat)(WMATH_CREATE_TYPE(Quat) c);

WMATH_TYPE(Quat)
WMATH_SET(Quat)(WMATH_TYPE(Quat) a, float x, float y, float z, float w);

WMATH_TYPE(Quat)
WMATH_COPY(Quat)(WMATH_TYPE(Quat) a);

float WMATH_DOT(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

WMATH_TYPE(Quat)
WMATH_LERP(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t);

// slerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, slerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b, float t);

// sqlerp
WMATH_TYPE(Quat)
WMATH_CALL(Quat, sqlerp)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b,
                         WMATH_TYPE(Quat) c, WMATH_TYPE(Quat) d, float t);

float WMATH_LENGTH(Quat)(WMATH_TYPE(Quat) a);

float WMATH_LENGTH_SQ(Quat)(WMATH_TYPE(Quat) a);

WMATH_TYPE(Quat)
WMATH_NORMALIZE(Quat)(WMATH_TYPE(Quat) a);

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);
// ==
bool WMATH_EQUALS(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

float WMATH_ANGLE(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

WMATH_TYPE(Quat)
WMATH_CALL(Quat, rotation_to)(WMATH_TYPE(Vec3) a_unit, WMATH_TYPE(Vec3) b_unit);

// *
WMATH_TYPE(Quat)
WMATH_MULTIPLY(Quat)
(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

// .*
WMATH_TYPE(Quat)
WMATH_MULTIPLY_SCALAR(Quat)(WMATH_TYPE(Quat) a, float b);

// -
WMATH_TYPE(Quat)
WMATH_SUB(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

// +
WMATH_TYPE(Quat)
WMATH_ADD(Quat)(WMATH_TYPE(Quat) a, WMATH_TYPE(Quat) b);

// inverse
WMATH_TYPE(Quat)
WMATH_INVERSE(Quat)(WMATH_TYPE(Quat) q);

// conjugate
WMATH_TYPE(Quat)
WMATH_CALL(Quat, conjugate)(WMATH_TYPE(Quat) q);

// divScalar
WMATH_TYPE(Quat)
WMATH_DIV_SCALAR(Quat)(WMATH_TYPE(Quat) a, float v);

// END Quat

// FROM

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_mat4)(WMATH_TYPE(Mat4) a);

WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, from_quat)(WMATH_TYPE(Quat) q);

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_mat3)(WMATH_TYPE(Mat3) a);

WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, from_quat)(WMATH_TYPE(Quat) q);

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_axis_angle)(WMATH_TYPE(Vec3) axis,
                                  float angle_in_radians);

WCN_Math_Vec3_WithAngleAxis WMATH_CALL(Quat, to_axis_angle)(WMATH_TYPE(Quat) q);

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat4)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat4) m);

WMATH_TYPE(Vec2)
WMATH_CALL(Vec2, transform_mat3)(WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat3) m);

WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m);

// vec3 transformMat4Upper3x3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat4_upper3x3)(WMATH_TYPE(Vec3) v,
                                          WMATH_TYPE(Mat4) m);

// vec3 transformMat3
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_mat3)(WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat3) m);

// vec3 transformQuat
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, transform_quat)
(WMATH_TYPE(Vec3) v, WMATH_TYPE(Quat) q);

// Quat fromMat
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat4)(WMATH_TYPE(Mat4) m);

WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_mat3)(WMATH_TYPE(Mat3) m);

// fromEuler
WMATH_TYPE(Quat)
WMATH_CALL(Quat, from_euler)(float x_angle_in_radians, float y_angle_in_radians,
                             float z_angle_in_radians,
                             enum WCN_Math_RotationOrder order);

// BEGIN 3D
// vec3 getTranslation
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Vec3)(WMATH_TYPE(Mat4) m);

// vec3 getAxis
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_axis)(WMATH_TYPE(Mat4) m, int axis);

// vec3 getScale
WMATH_TYPE(Vec3)
WMATH_CALL(Vec3, get_scale)(WMATH_TYPE(Mat4) m);

// vec3 rotateX
WMATH_TYPE(Vec3)
WMATH_ROTATE_X(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec3 rotateY
WMATH_TYPE(Vec3)
WMATH_ROTATE_Y(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec3 rotateZ
WMATH_TYPE(Vec3)
WMATH_ROTATE_Z(Vec3)(WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec4 transformMat4
WMATH_TYPE(Vec4)
WMATH_CALL(Vec4, transform_mat4)(WMATH_TYPE(Vec4) v, WMATH_TYPE(Mat4) m);

// Quat rotate_x
WMATH_TYPE(Quat)
WMATH_ROTATE_X(Quat)(WMATH_TYPE(Quat) q, float angleInRadians);

// Quat rotate_y
WMATH_TYPE(Quat)
WMATH_ROTATE_Y(Quat)(WMATH_TYPE(Quat) q, float angleInRadians);

// Quat rotate_z
WMATH_TYPE(Quat)
WMATH_ROTATE_Z(Quat)(WMATH_TYPE(Quat) q, float angleInRadians);

// Mat3 rotate
WMATH_TYPE(Mat3)
WMATH_ROTATE(Mat3)
(WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate x
WMATH_TYPE(Mat3)
WMATH_ROTATE_X(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate y
WMATH_TYPE(Mat3)
WMATH_ROTATE_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate z
WMATH_TYPE(Mat3)
WMATH_ROTATE_Z(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotation
WMATH_TYPE(Mat3)
WMATH_ROTATION(Mat3)(float angleInRadians);

// Mat3 rotation x
WMATH_TYPE(Mat3)
WMATH_ROTATION_X(Mat3)(float angleInRadians);

// Mat3 rotation y
WMATH_TYPE(Mat3)
WMATH_ROTATION_Y(Mat3)(WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotation z
WMATH_TYPE(Mat3)
WMATH_ROTATION_Z(Mat3)(float angleInRadians);

// Mat3 get_axis
/**
 * Returns an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y,
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_axis)
(WMATH_TYPE(Mat3) m, int axis);

// Mat3 set_axis
/**
 * Sets an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - The axis  0 = x, 1 = y;
 * @returns The matrix with axis set.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, set_axis)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v, int axis);

// Mat3 get_scaling
WMATH_TYPE(Vec2)
WMATH_CALL(Mat3, get_scaling)
(WMATH_TYPE(Mat3) m);

// Mat3 get_3D_scaling
WMATH_TYPE(Vec3)
WMATH_CALL(Mat3, get_3D_scaling)
(WMATH_TYPE(Mat3) m);

// Mat3 get_translation
WMATH_TYPE(Vec2)
WMATH_GET_TRANSLATION(Mat3)
(WMATH_TYPE(Mat3) m);

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
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

// Mat3 translation
WMATH_TYPE(Mat3)
WMATH_TRANSLATION(Mat3)
(WMATH_TYPE(Vec2) v);

// translate
/**
 * Translates the given 3-by-3 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by which to translate.
 * @returns The translated matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, translate)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

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
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) axis, float angleInRadians);

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
(WMATH_TYPE(Vec3) axis, float angleInRadians);
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
);

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
(float left, float right, float bottom, float top, float near, float far);

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
(float left, float right, float bottom, float top, float near, float far);

// Mat4 get_axis
/**
 * Returns an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y, 2 = z;
 * @returns The axis component of m.
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_axis)
(WMATH_TYPE(Mat4) m, int axis);

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
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v, int axis);

// Mat4 getTranslation
///**
// * Returns the translation component of a 4-by-4 matrix as a vector with 3
// * entries.
// * @param m - The matrix.
// * @returns The translation component of m.
// */
WMATH_TYPE(Vec3)
WMATH_GET_TRANSLATION(Mat4)
(WMATH_TYPE(Mat4) m);

// Mat4 setTranslation
WMATH_TYPE(Mat4)
WMATH_SET_TRANSLATION(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

// Mat4 translation
/**
 * Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by
 *     which to translate.
 * @returns The translation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_TRANSLATION(Mat4)
(WMATH_TYPE(Vec3) v);

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
(float fieldOfViewYInRadians, float aspect, float zNear, float zFar);

// Mat4 perspective_reverse_z
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, perspective_reverse_z)
(
    //
    float fieldOfViewYInRadians, // fieldOfViewYInRadians: number
    float aspect,                // aspect: number
    float zNear,                 // zNear: number
    float zFar                   // zFar: number
);

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
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

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
);

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
);

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
);

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
);

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
(WMATH_TYPE(Vec3) axis, float angleInRadians);

// Mat4 rotation_x
/**
 * Creates a 4-by-4 matrix which rotates around the x-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_X(Mat4)
(float angleInRadians);

// Mat4 rotation_y
/**
 * Creates a 4-by-4 matrix which rotates around the y-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Y(Mat4)
(float angleInRadians);

// Mat4 rotation_z
/**
 * Creates a 4-by-4 matrix which rotates around the z-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
WMATH_TYPE(Mat4)
WMATH_ROTATION_Z(Mat4)
(float angleInRadians);

// All Type Scale Impl
WMATH_TYPE(Vec2)
WMATH_SCALE(Vec2)
(WMATH_TYPE(Vec2) v, float scale);

WMATH_TYPE(Vec3)
WMATH_SCALE(Vec3)
(WMATH_TYPE(Vec3) v, float scale);

WMATH_TYPE(Quat)
WMATH_SCALE(Quat)
(WMATH_TYPE(Quat) q, float scale);

WMATH_TYPE(Mat3)
WMATH_SCALE(Mat3)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

WMATH_TYPE(Mat4)
WMATH_SCALE(Mat4)
(WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

// Mat3 scale3D
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scale3D)
(WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec3) v);

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
(WMATH_TYPE(Vec2) v);

/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     3 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, scaling3D)(WMATH_TYPE(Vec3) v);

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
(WMATH_TYPE(Mat3) m, float s);

// Mat3 uniform_scale_3D
/**
 * Scales the given 3-by-3 matrix in each dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scale_3D)
(WMATH_TYPE(Mat3) m, float s);

// Mat3 uniform_scaling
/**
 * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling)
(float s);

/**
 * Creates a 3-by-3 matrix which scales uniformly in each dimension
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat3)
WMATH_CALL(Mat3, uniform_scaling_3D)
(float s);

// Mat4 getScaling
/**
 * Returns the "3d" scaling component of the matrix
 * @param m - The Matrix
 */
WMATH_TYPE(Vec3)
WMATH_CALL(Mat4, get_scaling)
(WMATH_TYPE(Mat4) m);

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
WMATH_CALL(Mat4, scaling)
(WMATH_TYPE(Vec3) v);

// Mat4 uniformScale
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scale)
(WMATH_TYPE(Mat4) m, float s);
// Mat4 uniformScaling
/**
 * Creates a 4-by-4 matrix which scales a uniform amount in each dimension.
 * @param s - the amount to scale
 * @returns The scaling matrix.
 */
WMATH_TYPE(Mat4)
WMATH_CALL(Mat4, uniform_scaling)
(float s);

// END 3D

#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_H
