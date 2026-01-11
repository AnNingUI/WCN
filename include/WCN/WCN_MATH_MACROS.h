#ifndef WCN_MATH_MACROS_H
#define WCN_MATH_MACROS_H


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

// dist_squared
#define WMATH_DISTANCE_SQ(WCN_Math_TYPE)                                       \
  wcn_math_##WCN_Math_TYPE##_distance_squared

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

// get_translation
#define WMATH_GET_TRANSLATION(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##_get_translation

// set_translation
#define WMATH_SET_TRANSLATION(WCN_Math_TYPE)                                   \
  wcn_math_##WCN_Math_TYPE##_set_translation

// translation
#define WMATH_TRANSLATION(WCN_Math_TYPE) wcn_math_##WCN_Math_TYPE##_translation

#define T$(WCN_Math_TYPE) WMATH_TYPE(WCN_Math_TYPE)

// BEGIN Utils

#define WMATH_DEG2RED(degrees) (degrees * 0.017453292519943295f)

#define WMATH_RED2DEG(radians) (radians * 57.29577951308232f)

// #define WMATH_NUM_LERP(a, b, t) ((a) + ((b) - (a)) * (t))
// Impl of lerp for float, double, int, and float_t
// ==================================================================
int WMATH_LERP(int)(const int a, const int b, const float t);
float WMATH_LERP(float)(const float a, const float b, const float t);
double WMATH_LERP(double)(const double a, const double b, const double t);
// ==================================================================

// Impl of random for float, double, int
// ==================================================================
int WMATH_RANDOM(int)();
float WMATH_RANDOM(float)();
double WMATH_RANDOM(double)();
// ==================================================================

// Impl of clamp for float, double, int
int WMATH_CLAMP(int)(int v, int min, int max);
float WMATH_CLAMP(float)(float v, float min, float max);
double WMATH_CLAMP(double)(double v, double min, double max);

#define WMATH_INVERSE_LERP(a, b, t)                                            \
  (fabsf((b) - (a)) < wcn_math_get_epsilon() ? 0.0f                            \
                                             : (((t) - (a)) / ((b) - (a))))
#define WMATH_EUCLIDEAN_MODULO(n, m) ((n) - floorf((n) / (m)) * (m))

#endif // WCN_MATH_MACROS_H
