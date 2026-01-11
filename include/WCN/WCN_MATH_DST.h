#ifndef WCN_MATH_DST_H
#define WCN_MATH_DST_H

// SIMD includes on supported platforms
#include "WCN/WCN_MATH_MACROS.h"
#include "WCN/WCN_MATH_TYPES.h"
#include "WCN/WCN_PLATFORM_MACROS.h"
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DST_VEC2 WMATH_TYPE(Vec2) * dst
#define DST_VEC3 WMATH_TYPE(Vec3) * dst
#define DST_VEC4 WMATH_TYPE(Vec4) * dst
#define DST_QUAT WMATH_TYPE(Quat) * dst
#define DST_MAT3 WMATH_TYPE(Mat3) * dst
#define DST_MAT4 WMATH_TYPE(Mat4) * dst

// Config
extern float EPSILON;
WCN_WASM_EXPORT float wcn_math_set_epsilon(float epsilon);
WCN_WASM_EXPORT float wcn_math_get_epsilon();
#define WCN_GET_EPSILON() wcn_math_get_epsilon()

// BEGIN Vec2

// create
void WMATH_CREATE(Vec2)(DST_VEC2, const WMATH_CREATE_TYPE(Vec2) vec2_c);

// set
void WMATH_SET(Vec2)(DST_VEC2, const float x, const float y);

// copy
void WMATH_COPY(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) vec2);

// 0
void WMATH_ZERO(Vec2)(DST_VEC2);

// 1
void WMATH_IDENTITY(Vec2)(DST_VEC2);

// ceil
void WMATH_CEIL(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a);

// floor
void WMATH_FLOOR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a);

// round
void WMATH_ROUND(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a);

// clamp
void WMATH_CLAMP(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, float min_val, float max_val);

// dot
float WMATH_DOT(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// add
void WMATH_ADD(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// addScaled
void WMATH_ADD_SCALED(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                            float scale);

// sub
void WMATH_SUB(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// angle
float WMATH_ANGLE(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// equalsApproximately
bool WMATH_EQUALS_APPROXIMATELY(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// equals
bool WMATH_EQUALS(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// lerp
void WMATH_LERP(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b, float t);

// lerpV
void WMATH_LERP_V(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                        const WMATH_TYPE(Vec2) t);

// fmax
void WMATH_FMAX(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// fmin
void WMATH_FMIN(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// multiplyScalar
void WMATH_MULTIPLY_SCALAR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float scalar);

// multiply
void WMATH_MULTIPLY(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// divScalar
/**
 * (divScalar) if scalar is 0, returns a zero vector
 */
void WMATH_DIV_SCALAR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float scalar);

// div
void WMATH_DIV(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// inverse
void WMATH_INVERSE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a);

// cross
void WMATH_CROSS(Vec2)(DST_VEC3, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// length
float WMATH_LENGTH(Vec2)(const WMATH_TYPE(Vec2) v);

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(const WMATH_TYPE(Vec2) v);

// distance
float WMATH_DISTANCE(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// negate
void WMATH_NEGATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a);

// random
void WMATH_RANDOM(Vec2)(DST_VEC2, const float scale);

// normalize
void WMATH_NORMALIZE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) v);

// rotate
void WMATH_ROTATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                        const float rad);

// set length
void WMATH_SET_LENGTH(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float length);

// truncate
void WMATH_TRUNCATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float length);

// midpoint
void WMATH_MIDPOINT(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b);

// END Vec2

// BEGIN Vec3

void WMATH_CREATE(Vec3)(DST_VEC3, const WMATH_CREATE_TYPE(Vec3) vec3_c);

// copy
void WMATH_COPY(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// set
void WMATH_SET(Vec3)(DST_VEC3, const float x, const float y, const float z);

// 0
void WMATH_ZERO(Vec3)(DST_VEC3);

// ceil
void WMATH_CEIL(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// floor
void WMATH_FLOOR(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// round
void WMATH_ROUND(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// dot
float WMATH_DOT(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// cross
void WMATH_CROSS(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// length
float WMATH_LENGTH(Vec3)(const WMATH_TYPE(Vec3) v);

// lengthSquared
float WMATH_LENGTH_SQ(Vec3)(const WMATH_TYPE(Vec3) v);

// normalize
void WMATH_NORMALIZE(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) v);

// clamp
void WMATH_CLAMP(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const float min_val,
                       const float max_val);

// +
void WMATH_ADD(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

void WMATH_ADD_SCALED(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b,
                            const float scalar);

// -
void WMATH_SUB(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// angle
float WMATH_ANGLE(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// ~=
bool WMATH_EQUALS_APPROXIMATELY(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// =
bool WMATH_EQUALS(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// lerp
void WMATH_LERP(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b, const float t);

// lerpV
void WMATH_LERP_V(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b,
                        const WMATH_TYPE(Vec3) t);

// fmax
void WMATH_FMAX(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// fmin
void WMATH_FMIN(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// *
void WMATH_MULTIPLY(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// .*
void WMATH_MULTIPLY_SCALAR(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const float scalar);

// div
void WMATH_DIV(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// .div
void WMATH_DIV_SCALAR(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const float scalar);

// inverse
void WMATH_INVERSE(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// distance
float WMATH_DISTANCE(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// distanceSquared
float WMATH_DISTANCE_SQ(Vec3)(const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// negate
void WMATH_NEGATE(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a);

// random
void WMATH_RANDOM(Vec3)(DST_VEC3, const float scale);

// setLength
void WMATH_SET_LENGTH(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) v, const float length);

// truncate
void WMATH_TRUNCATE(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) v, const float max_length);

// midpoint
void WMATH_MIDPOINT(Vec3)(DST_VEC3, const WMATH_TYPE(Vec3) a, const WMATH_TYPE(Vec3) b);

// END Vec3

// BEGIN Vec4

void WMATH_CREATE(Vec4)(DST_VEC4, const WMATH_CREATE_TYPE(Vec4) vec4_c);

void WMATH_SET(Vec4)(DST_VEC4, const float x, const float y, const float z, const float w);

void WMATH_COPY(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) vec4);

void WMATH_ZERO(Vec4)(DST_VEC4);

void WMATH_IDENTITY(Vec4)(DST_VEC4);

void WMATH_CEIL(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a);

void WMATH_FLOOR(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a);

void WMATH_ROUND(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a);

void WMATH_CLAMP(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const float min_val,
                       const float max_val);

void WMATH_ADD(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_ADD_SCALED(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b,
                            const float scale);

void WMATH_SUB(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

bool WMATH_EQUALS_APPROXIMATELY(Vec4)(const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

bool WMATH_EQUALS(Vec4)(const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_LERP(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b, const float t);

void WMATH_LERP_V(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b,
                        const WMATH_TYPE(Vec4) t);

void WMATH_FMAX(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_FMIN(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_MULTIPLY(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_MULTIPLY_SCALAR(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const float scalar);

void WMATH_DIV(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_DIV_SCALAR(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const float scalar);

void WMATH_INVERSE(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a);

float WMATH_DOT(Vec4)(const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

float WMATH_LENGTH_SQ(Vec4)(const WMATH_TYPE(Vec4) v);

float WMATH_LENGTH(Vec4)(const WMATH_TYPE(Vec4) v);

float WMATH_DISTANCE_SQ(Vec4)(const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

float WMATH_DISTANCE(Vec4)(const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

void WMATH_NORMALIZE(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) v);

void WMATH_NEGATE(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a);

void WMATH_SET_LENGTH(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) v, const float length);

void WMATH_TRUNCATE(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) v, const float max_length);

void WMATH_MIDPOINT(Vec4)(DST_VEC4, const WMATH_TYPE(Vec4) a, const WMATH_TYPE(Vec4) b);

// END Vec4

// BEGIN Mat3

void WMATH_IDENTITY(Mat3)(DST_MAT3);

void WMATH_ZERO(Mat3)(DST_MAT3);

void WMATH_CREATE(Mat3)(DST_MAT3, const WMATH_CREATE_TYPE(Mat3) mat3_c);

void WMATH_COPY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat);

bool WMATH_EQUALS(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b);
bool WMATH_EQUALS_APPROXIMATELY(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b);

void WMATH_SET(Mat3)(DST_MAT3, const float m00, const float m01, const float m02, const float m10,
                     const float m11, const float m12, const float m20, const float m21,
                     const float m22);

void WMATH_NEGATE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat);

void WMATH_TRANSPOSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat);

// SIMD optimized matrix addition
void WMATH_ADD(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b);

// SIMD optimized matrix subtraction
void WMATH_SUB(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b);

// SIMD optimized scalar multiplication
void WMATH_MULTIPLY_SCALAR(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const float b);

void WMATH_INVERSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a);

// Optimized matrix multiplication
void WMATH_MULTIPLY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b);

float WMATH_DETERMINANT(Mat3)(const WMATH_TYPE(Mat3) m);

// END Mat3

// BEGIN Mat4

// 0 add 1 Mat4

void WMATH_IDENTITY(Mat4)(DST_MAT4);

void WMATH_ZERO(Mat4)(DST_MAT4);

// Init Mat4

void WMATH_CREATE(Mat4)(DST_MAT4, const WMATH_CREATE_TYPE(Mat4) mat4_c);

void WMATH_COPY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat);

void WMATH_SET(Mat4)(DST_MAT4, const float m00, const float m01, const float m02, const float m03,
                     const float m10, const float m11, const float m12, const float m13,
                     const float m20, const float m21, const float m22, const float m23,
                     const float m30, const float m31, const float m32, const float m33);

void WMATH_NEGATE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat);

bool WMATH_EQUALS(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b);

bool WMATH_EQUALS_APPROXIMATELY(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b);

// + add - Mat4
void WMATH_ADD(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b);

void WMATH_SUB(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b);

// .* Mat4

void WMATH_MULTIPLY_SCALAR(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const float b);

// * Mat4

void WMATH_MULTIPLY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b);

void WMATH_INVERSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a);

void WMATH_TRANSPOSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a);

float WMATH_DETERMINANT(Mat4)(const WMATH_TYPE(Mat4) m);

// aim
void WMATH_CALL(Mat4, aim)(DST_MAT4, const WMATH_TYPE(Vec3) position, const WMATH_TYPE(Vec3) target,
                           const WMATH_TYPE(Vec3) up);

// lookAt
void WMATH_CALL(Mat4, look_at)(DST_MAT4, const WMATH_TYPE(Vec3) eye, const WMATH_TYPE(Vec3) target,
                               const WMATH_TYPE(Vec3) up);

void WMATH_CALL(Mat4, ortho)(DST_MAT4, const float left, const float right, const float bottom,
                             const float top, const float near, const float far);

// END Mat4

// BEGIN Quat

void WMATH_ZERO(Quat)(DST_QUAT);

void WMATH_IDENTITY(Quat)(DST_QUAT);

void WMATH_CREATE(Quat)(DST_QUAT, const WMATH_CREATE_TYPE(Quat) c);

void WMATH_SET(Quat)(DST_QUAT, const float x, const float y, const float z, const float w);

void WMATH_COPY(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a);

float WMATH_DOT(Quat)(const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

void WMATH_LERP(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b, const float t);

void WMATH_CALL(Quat, slerp)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b,
                             const float t);

void WMATH_CALL(Quat, sqlerp)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b,
                              const WMATH_TYPE(Quat) c, const WMATH_TYPE(Quat) d, const float t);

float WMATH_LENGTH(Quat)(const WMATH_TYPE(Quat) a);

float WMATH_LENGTH_SQ(Quat)(const WMATH_TYPE(Quat) a);

void WMATH_NORMALIZE(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a);

bool WMATH_EQUALS_APPROXIMATELY(Quat)(const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

bool WMATH_EQUALS(Quat)(const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

float WMATH_ANGLE(Quat)(const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

void WMATH_CALL(Quat, rotation_to)(DST_QUAT, const WMATH_TYPE(Vec3) a_unit,
                                   const WMATH_TYPE(Vec3) b_unit);

void WMATH_MULTIPLY(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

void WMATH_MULTIPLY_SCALAR(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const float b);

void WMATH_SUB(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

void WMATH_ADD(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const WMATH_TYPE(Quat) b);

void WMATH_INVERSE(Quat)(DST_QUAT, const WMATH_TYPE(Quat) q);

void WMATH_CALL(Quat, conjugate)(DST_QUAT, const WMATH_TYPE(Quat) q);

void WMATH_DIV_SCALAR(Quat)(DST_QUAT, const WMATH_TYPE(Quat) a, const float v);

// END Quat

// FROM

// FROM

void WMATH_CALL(Mat3, from_mat4)(DST_MAT3, WMATH_TYPE(Mat4) a);

void WMATH_CALL(Mat3, from_quat)(DST_MAT3, WMATH_TYPE(Quat) q);

void WMATH_CALL(Mat4, from_mat3)(DST_MAT4, WMATH_TYPE(Mat3) a);

void WMATH_CALL(Mat4, from_quat)(DST_MAT4, WMATH_TYPE(Quat) q);

void WMATH_CALL(Quat, from_axis_angle)(DST_QUAT, WMATH_TYPE(Vec3) axis, float angle_in_radians);

void WMATH_CALL(Quat, to_axis_angle)(WCN_Math_Vec3_WithAngleAxis* dst, WMATH_TYPE(Quat) q);

void WMATH_CALL(Vec2, transform_mat4)(DST_VEC2, WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat4) m);

void WMATH_CALL(Vec2, transform_mat3)(DST_VEC2, WMATH_TYPE(Vec2) v, WMATH_TYPE(Mat3) m);

void WMATH_CALL(Vec3, transform_mat4)(DST_VEC3, WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m);

// vec3 transformMat4Upper3x3
void WMATH_CALL(Vec3, transform_mat4_upper3x3)(DST_VEC3, WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat4) m);

// vec3 transformMat3
void WMATH_CALL(Vec3, transform_mat3)(DST_VEC3, WMATH_TYPE(Vec3) v, WMATH_TYPE(Mat3) m);

// vec3 transformQuat
void WMATH_CALL(Vec3, transform_quat)(DST_VEC3, WMATH_TYPE(Vec3) v, WMATH_TYPE(Quat) q);

// Quat fromMat
void WMATH_CALL(Quat, from_mat4)(DST_QUAT, WMATH_TYPE(Mat4) m);

void WMATH_CALL(Quat, from_mat3)(DST_QUAT, WMATH_TYPE(Mat3) m);

// fromEuler
void WMATH_CALL(Quat, from_euler)(DST_QUAT, float x_angle_in_radians, float y_angle_in_radians,
                                  float z_angle_in_radians, enum WCN_Math_RotationOrder order);

// BEGIN 3D
// vec3 getTranslation
void WMATH_GET_TRANSLATION(Vec3)(DST_VEC3, WMATH_TYPE(Mat4) m);

// vec3 getAxis
void WMATH_CALL(Vec3, get_axis)(DST_VEC3, WMATH_TYPE(Mat4) m, int axis);

// vec3 getScale
void WMATH_CALL(Vec3, get_scale)(DST_VEC3, WMATH_TYPE(Mat4) m);

// vec3 rotateX
void WMATH_ROTATE_X(Vec3)(DST_VEC3, WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec3 rotateY
void WMATH_ROTATE_Y(Vec3)(DST_VEC3, WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec3 rotateZ
void WMATH_ROTATE_Z(Vec3)(DST_VEC3, WMATH_TYPE(Vec3) a, WMATH_TYPE(Vec3) b, float rad);

// vec4 transformMat4
void WMATH_CALL(Vec4, transform_mat4)(DST_VEC4, WMATH_TYPE(Vec4) v, WMATH_TYPE(Mat4) m);

// Quat rotate_x
void WMATH_ROTATE_X(Quat)(DST_QUAT, WMATH_TYPE(Quat) q, float angleInRadians);

// Quat rotate_y
void WMATH_ROTATE_Y(Quat)(DST_QUAT, WMATH_TYPE(Quat) q, float angleInRadians);

// Quat rotate_z
void WMATH_ROTATE_Z(Quat)(DST_QUAT, WMATH_TYPE(Quat) q, float angleInRadians);

// Mat3 rotate
void WMATH_ROTATE(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate x
void WMATH_ROTATE_X(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate y
void WMATH_ROTATE_Y(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotate z
void WMATH_ROTATE_Z(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, float angleInRadians);

// Mat3 rotation
void WMATH_ROTATION(Mat3)(DST_MAT3, float angleInRadians);

// Mat3 rotation x
void WMATH_ROTATION_X(Mat3)(DST_MAT3, float angleInRadians);

// Mat3 rotation y
void WMATH_ROTATION_Y(Mat3)(DST_MAT3, float angleInRadians);

// Mat3 rotation z
void WMATH_ROTATION_Z(Mat3)(DST_MAT3, float angleInRadians);

// Mat3 get_axis
/**
 * Returns an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y,
 * @returns The axis component of m.
 */
void WMATH_CALL(Mat3, get_axis)(DST_VEC2, WMATH_TYPE(Mat3) m, int axis);

// Mat3 set_axis
/**
 * Sets an axis of a 3x3 matrix as a vector with 2 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - The axis  0 = x, 1 = y;
 * @returns The matrix with axis set.
 */
void WMATH_CALL(Mat3, set_axis)(DST_MAT3, WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v, int axis);

// Mat3 get_scaling
void WMATH_CALL(Mat3, get_scaling)(DST_VEC2, WMATH_TYPE(Mat3) m);

// Mat3 get_3D_scaling
void WMATH_CALL(Mat3, get_3D_scaling)(DST_VEC3, WMATH_TYPE(Mat3) m);

// Mat3 get_translation
void WMATH_GET_TRANSLATION(Mat3)(DST_VEC2, WMATH_TYPE(Mat3) m);

// Mat3 set_translation
/**
 * Sets the translation component of a 3-by-3 matrix to the given
 * vector.
 * @param m - The matrix.
 * @param v - The vector.
 * @returns The matrix with translation set.
 */
void WMATH_SET_TRANSLATION(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

// Mat3 translation
void WMATH_TRANSLATION(Mat3)(DST_MAT3, WMATH_TYPE(Vec2) v);

// translate
/**
 * Translates the given 3-by-3 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by which to translate.
 * @returns The translated matrix.
 */
void WMATH_CALL(Mat3, translate)(DST_MAT3, WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

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
void WMATH_CALL(Mat4, axis_rotate)(DST_MAT4, WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) axis,
                                   float angleInRadians);

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
void WMATH_CALL(Mat4, axis_rotation)(DST_MAT4, WMATH_TYPE(Vec3) axis, float angleInRadians);

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
void WMATH_CALL(Mat4, camera_aim)(DST_MAT4,
                                  WMATH_TYPE(Vec3) eye,     // eye: Vec3
                                  WMATH_TYPE(Vec3) target,  // target: Vec3
                                  WMATH_TYPE(Vec3) up       // up: Vec3
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
void WMATH_CALL(Mat4, frustum)(DST_MAT4, float left, float right, float bottom, float top,
                               float near, float far);

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
void WMATH_CALL(Mat4, frustum_reverse_z)(DST_MAT4, float left, float right, float bottom, float top,
                                         float near, float far);

// Mat4 get_axis
/**
 * Returns an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param axis - The axis 0 = x, 1 = y, 2 = z;
 * @returns The axis component of m.
 */
void WMATH_CALL(Mat4, get_axis)(DST_VEC3, WMATH_TYPE(Mat4) m, int axis);

// Mat4 set_axis
/**
 * Sets an axis of a 4x4 matrix as a vector with 3 entries
 * @param m - The matrix.
 * @param v - the axis vector
 * @param axis - The axis  0 = x, 1 = y, 2 = z;
 * @returns The matrix with axis set.
 */
void WMATH_CALL(Mat4, set_axis)(DST_MAT4, WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v, int axis);

// Mat4 getTranslation
void WMATH_GET_TRANSLATION(Mat4)(DST_VEC3, WMATH_TYPE(Mat4) m);

// Mat4 setTranslation
void WMATH_SET_TRANSLATION(Mat4)(DST_MAT4, WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

// Mat4 translation
/**
 * Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by
 *     which to translate.
 * @returns The translation matrix.
 */
void WMATH_TRANSLATION(Mat4)(DST_MAT4, WMATH_TYPE(Vec3) v);

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
void WMATH_CALL(Mat4, perspective)(DST_MAT4, float fieldOfViewYInRadians, float aspect, float zNear,
                                   float zFar);

// Mat4 perspective_reverse_z
void WMATH_CALL(Mat4, perspective_reverse_z)(
    DST_MAT4,
    float fieldOfViewYInRadians,  // fieldOfViewYInRadians: number
    float aspect,                 // aspect: number
    float zNear,                  // zNear: number
    float zFar                    // zFar: number
);

// Mat4 translate
/**
 * Translates the given 4-by-4 matrix by the given vector v.
 * @param m - The matrix.
 * @param v - The vector by
 *     which to translate.
 * @returns The translated matrix.
 */
void WMATH_CALL(Mat4, translate)(DST_MAT4, WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

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
void WMATH_ROTATE(Mat4)(DST_MAT4,
                        WMATH_TYPE(Mat4) m,     // m: Mat4
                        WMATH_TYPE(Vec3) axis,  // axis: Vec3
                        float angleInRadians    // angleInRadians: number
);

// Mat4 rotate_x
/**
 * Rotates the given 4-by-4 matrix around the x-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
void WMATH_ROTATE_X(Mat4)(DST_MAT4,
                          WMATH_TYPE(Mat4) m,   // m: Mat4
                          float angleInRadians  // angleInRadians: number
);

// Mat4 rotate_y
/**
 * Rotates the given 4-by-4 matrix around the y-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
void WMATH_ROTATE_Y(Mat4)(DST_MAT4,
                          WMATH_TYPE(Mat4) m,   // m: Mat4
                          float angleInRadians  // angleInRadians: number
);

// Mat4 rotate_z
/**
 * Rotates the given 4-by-4 matrix around the z-axis by the given
 * angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix.
 */
void WMATH_ROTATE_Z(Mat4)(DST_MAT4,
                          WMATH_TYPE(Mat4) m,   // m: Mat4
                          float angleInRadians  // angleInRadians: number
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
void WMATH_ROTATION(Mat4)(DST_MAT4, WMATH_TYPE(Vec3) axis, float angleInRadians);

// Mat4 rotation_x
/**
 * Creates a 4-by-4 matrix which rotates around the x-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
void WMATH_ROTATION_X(Mat4)(DST_MAT4, float angleInRadians);

// Mat4 rotation_y
/**
 * Creates a 4-by-4 matrix which rotates around the y-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
void WMATH_ROTATION_Y(Mat4)(DST_MAT4, float angleInRadians);

// Mat4 rotation_z
/**
 * Creates a 4-by-4 matrix which rotates around the z-axis by the given angle.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotation matrix.
 */
void WMATH_ROTATION_Z(Mat4)(DST_MAT4, float angleInRadians);

// All Type Scale Impl
void WMATH_SCALE(Vec2)(DST_VEC2, WMATH_TYPE(Vec2) v, float scale);

void WMATH_SCALE(Vec3)(DST_VEC3, WMATH_TYPE(Vec3) v, float scale);

void WMATH_SCALE(Quat)(DST_QUAT, WMATH_TYPE(Quat) q, float scale);

void WMATH_SCALE(Mat3)(DST_MAT3, WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec2) v);

void WMATH_SCALE(Mat4)(DST_MAT4, WMATH_TYPE(Mat4) m, WMATH_TYPE(Vec3) v);

// Mat3 scale3D
void WMATH_CALL(Mat3, scale3D)(DST_MAT3, WMATH_TYPE(Mat3) m, WMATH_TYPE(Vec3) v);

// Mat3 scaling
/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has two
 * entries.
 * @param v - A vector of
 *     2 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat3, scaling)(DST_MAT3, WMATH_TYPE(Vec2) v);

/**
 * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     3 entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat3, scaling3D)(DST_MAT3, WMATH_TYPE(Vec3) v);

// Mat3 uniform_scale
/**
 * Scales the given 3-by-3 matrix in the X and Y dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
void WMATH_CALL(Mat3, uniform_scale)(DST_MAT3, WMATH_TYPE(Mat3) m, float s);

// Mat3 uniform_scale_3D
/**
 * Scales the given 3-by-3 matrix in each dimension by an amount
 * given.
 * @param m - The matrix to be modified.
 * @param s - Amount to scale.
 * @returns The scaled matrix.
 */
void WMATH_CALL(Mat3, uniform_scale_3D)(DST_MAT3, WMATH_TYPE(Mat3) m, float s);

// Mat3 uniform_scaling
/**
 * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat3, uniform_scaling)(DST_MAT3, float s);

/**
 * Creates a 3-by-3 matrix which scales uniformly in each dimension
 * @param s - Amount to scale
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat3, uniform_scaling_3D)(DST_MAT3, float s);

// Mat4 getScaling
/**
 * Returns the "3d" scaling component of the matrix
 * @param m - The Matrix
 */
void WMATH_CALL(Mat4, get_scaling)(DST_VEC3, WMATH_TYPE(Mat4) m);

// Mat4 scaling
/**
 * Creates a 4-by-4 matrix which scales in each dimension by an amount given by
 * the corresponding entry in the given vector; assumes the vector has three
 * entries.
 * @param v - A vector of
 *     three entries specifying the factor by which to scale in each dimension.
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat4, scaling)(DST_MAT4, WMATH_TYPE(Vec3) v);

// Mat4 uniformScale
void WMATH_CALL(Mat4, uniform_scale)(DST_MAT4, WMATH_TYPE(Mat4) m, float s);

// Mat4 uniformScaling
/**
 * Creates a 4-by-4 matrix which scales a uniform amount in each dimension.
 * @param s - the amount to scale
 * @returns The scaling matrix.
 */
void WMATH_CALL(Mat4, uniform_scaling)(DST_MAT4, float s);

// END 3D

#ifdef __cplusplus
}
#endif

#endif  // WCN_MATH_DST_H
