#ifdef __EMSCRIPTEN__

#include <WCN/WCN_Math.h>

#define vec2_t  WMATH_TYPE(Vec2)
#define vec3_t  WMATH_TYPE(Vec3)
#define vec4_t  WMATH_TYPE(Vec4)
#define quat_t  WMATH_TYPE(Quat)
#define mat3_t  WMATH_TYPE(Mat3)
#define mat4_t  WMATH_TYPE(Mat4)
#define vec3_aa WCN_Math_Vec3_WithAngleAxis

// 由于 js 侧无法生成结构体，则取消用create宏的导出，采用具体值参数导出
// BEGIN easy init functions
WCN_WASM_EXPORT void wcn_math_Vec2_create_wasm(vec2_t *out, const float x, const float y)
{ 
  *out = INIT$(Vec2, .v_x = x, .v_y = y);
}
WCN_WASM_EXPORT void wcn_math_Vec3_create_wasm(vec3_t *out, const float x, const float y, const float z) {
  *out = INIT$(Vec3, .v_x = x, .v_y = y, .v_z = z);  
}
WCN_WASM_EXPORT void wcn_math_Vec4_create_wasm(vec4_t *out, const float x, const float y, const float z, const float w) {
  *out = INIT$(Vec4, .v_x = x, .v_y = y, .v_z = z, .v_w = w); 
};
WCN_WASM_EXPORT void wcn_math_Quat_create_wasm(quat_t *out, const float x, const float y, const float z, const float w) {
  *out = INIT$(Quat, .v_x = x, .v_y = y, .v_z = z, .v_w = w);
};
WCN_WASM_EXPORT void wcn_math_Mat3_create_wasm(
  mat3_t *out,
  const float m00,
  const float m01,
  const float m02,
  // next row
  const float m10,
  const float m11,
  const float m12,
  // next row
  const float m20,
  const float m21,
  const float m22
)
{
  *out = INIT$(Mat3,
    .m_00 = m00, .m_01 = m01, .m_02 = m02,
    .m_10 = m10, .m_11 = m11, .m_12 = m12,
    .m_20 = m20, .m_21 = m21, .m_22 = m22
  );
}
WCN_WASM_EXPORT void wcn_math_Mat4_create_wasm(
  mat4_t *out,
  const float m00,
  const float m01,
  const float m02,
  const float m03,
  // next row
  const float m10,
  const float m11,
  const float m12,
  const float m13,
  // next row
  const float m20,
  const float m21,
  const float m22,
  const float m23,
  // next row
  const float m30,
  const float m31,
  const float m32,
  const float m33
)
{
  *out = INIT$(Mat4,
    .m_00 = m00, .m_01 = m01, .m_02 = m02, .m_03 = m03,
    .m_10 = m10, .m_11 = m11, .m_12 = m12, .m_13 = m13,
    .m_20 = m20, .m_21 = m21, .m_22 = m22, .m_23 = m23,
    .m_30 = m30, .m_31 = m31, .m_32 = m32, .m_33 = m33
  );
}

// END easy init functions

// BEGIN WCN_Math_Vec3_WithAngleAxis
WCN_WASM_EXPORT vec3_t*  wcn_math_Vec3_WithAngleAxis_get_vec3_wasm(vec3_aa *out) {
  return &out->axis;
}
WCN_WASM_EXPORT float    wcn_math_Vec3_WithAngleAxis_get_angle(const vec3_aa *out) {
  return out->angle;
}
// END WCN_Math_Vec3_WithAngleAxis

// BEGIN Vec2
WCN_WASM_EXPORT void  wcn_math_Vec2_set_wasm(vec2_t *out, const float x, const float y) {
  out->v[0] = x;
  out->v[1] = y;
}
WCN_WASM_EXPORT void  wcn_math_Vec2_set_x_wasm(vec2_t *out, const float x) {
  out->v[0] = x;
}
WCN_WASM_EXPORT void  wcn_math_Vec2_set_y_wasm(vec2_t *out, const float y) {
  out->v[1] = y;
}
WCN_WASM_EXPORT void  wcn_math_Vec2_copy_wasm(vec2_t *out, const vec2_t * in) {
  out->v[0] = in->v[0];
  out->v[1] = in->v[1];
}
WCN_WASM_EXPORT void  wcn_math_Vec2_zero_wasm(vec2_t *out) {
  *out = WMATH_ZERO(Vec2)();
}
WCN_WASM_EXPORT void  wcn_math_Vec2_identity_wasm(vec2_t *out) {
  *out = WMATH_ZERO(Vec2)();
}
WCN_WASM_EXPORT void  wcn_math_Vec2_ceil_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_CEIL(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_floor_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_FLOOR(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_round_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_ROUND(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_clamp_wasm(vec2_t *out, const vec2_t *a, const float min_val, const float max_val) {
  *out = WMATH_CLAMP(Vec2)(*a, min_val, max_val);
}
WCN_WASM_EXPORT float wcn_math_Vec2_dot_wasm(const vec2_t *a, const vec2_t *b) {
  return WMATH_DOT(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_add_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_ADD(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_add_scaled_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b, const float scale) {
  *out = WMATH_ADD_SCALED(Vec2)(*a, *b, scale);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_sub_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_SUB(Vec2)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec2_angle_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  return WMATH_ANGLE(Vec2)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec2_equals_approximately_wasm(const vec2_t *a, const vec2_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Vec2)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec2_equals_wasm(const vec2_t *a, const vec2_t *b) {
  return WMATH_EQUALS(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_lerp_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b, const float t) {
  *out = WMATH_LERP(Vec2)(*a, *b, t);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_lerp_v_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b, const vec2_t *t) {
  *out = WMATH_LERP_V(Vec2)(*a, *b, *t);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_fmax_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_FMAX(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_fmin_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_FMIN(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_multiply_scalar_wasm(vec2_t *out, const vec2_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Vec2)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_multiply_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_MULTIPLY(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_div_scalar_wasm(vec2_t *out, const vec2_t *a, const float scalar) {
  *out = WMATH_DIV_SCALAR(Vec2)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_div_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_DIV(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_inverse_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_INVERSE(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_cross_wasm(vec3_t *out,const vec2_t *a, const vec2_t *b) {
  *out = WMATH_CROSS(Vec2)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec2_length_wasm(const vec2_t *v) {
  return WMATH_LENGTH(Vec2)(*v);
}
WCN_WASM_EXPORT float wcn_math_Vec2_length_squared_wasm(const vec2_t *v) {
  return WMATH_LENGTH_SQ(Vec2)(*v);
}
WCN_WASM_EXPORT float wcn_math_Vec2_distance_wasm(const vec2_t *a, const vec2_t *b) {
  return WMATH_DISTANCE(Vec2)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec2_distance_squared_wasm(const vec2_t *a, const vec2_t *b) {
  return WMATH_DISTANCE_SQ(Vec2)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_negate_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_NEGATE(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_random_wasm(vec2_t *out, const float scale) {
  *out = WMATH_RANDOM(Vec2)(scale);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_normalize_wasm(vec2_t *out, const vec2_t *in) {
  *out = WMATH_NORMALIZE(Vec2)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_rotate_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b, const float rad) {
  *out = WMATH_ROTATE(Vec2)(*a, *b, rad);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_set_length_wasm(vec2_t *out, const vec2_t *in, const float length) {
  *out = WMATH_SET_LENGTH(Vec2)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_truncate_wasm(vec2_t *out, const vec2_t *in, const float length) {
  *out = WMATH_TRUNCATE(Vec2)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec2_midpoint_wasm(vec2_t *out, const vec2_t *a, const vec2_t *b) {
  *out = WMATH_MIDPOINT(Vec2)(*a, *b);
}
// END Vec2

// BEGIN Vec3
WCN_WASM_EXPORT void  wcn_math_Vec3_set_wasm(vec3_t *out, const float x, const float y, const float z) {
  out->v[0] = x;
  out->v[1] = y;
  out->v[2] = z;
}
WCN_WASM_EXPORT void  wcn_math_Vec3_set_x_wasm(vec3_t *out, const float x) {
  out->v[0] = x;
}
WCN_WASM_EXPORT void  wcn_math_Vec3_set_y_wasm(vec3_t *out, const float y) {
  out->v[1] = y;
}
WCN_WASM_EXPORT void  wcn_math_Vec3_set_z_wasm(vec3_t *out, const float z) {
  out->v[2] = z;
}
WCN_WASM_EXPORT void  wcn_math_Vec3_copy_wasm(vec3_t *out, const vec3_t * in) {
  out->v[0] = in->v[0];
  out->v[1] = in->v[1];
  out->v[2] = in->v[2];
}
WCN_WASM_EXPORT void  wcn_math_Vec3_zero_wasm(vec3_t *out) {
  *out = WMATH_ZERO(Vec3)();
}
WCN_WASM_EXPORT void  wcn_math_Vec3_identity_wasm(vec3_t *out) {
  *out = WMATH_ZERO(Vec3)();
}
WCN_WASM_EXPORT void  wcn_math_Vec3_ceil_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_CEIL(Vec3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_floor_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_FLOOR(Vec3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_round_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_ROUND(Vec3)(*in);
}
WCN_WASM_EXPORT float wcn_math_Vec3_dot_wasm(const vec3_t *a, const vec3_t *b) {
  return WMATH_DOT(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_cross_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_CROSS(Vec3)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec3_length_wasm(const vec3_t *v) {
  return WMATH_LENGTH(Vec3)(*v);
}
WCN_WASM_EXPORT float wcn_math_Vec3_length_squared_wasm(const vec3_t *v) {
  return WMATH_LENGTH_SQ(Vec3)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_normalize_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_NORMALIZE(Vec3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_clamp_wasm(vec3_t *out, const vec3_t *a, const float min_val, const float max_val) {
  *out = WMATH_CLAMP(Vec3)(*a, min_val, max_val);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_add_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_ADD(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_add_scaled_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const float scale) {
  *out = WMATH_ADD_SCALED(Vec3)(*a, *b, scale);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_sub_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_SUB(Vec3)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec3_angle_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  return WMATH_ANGLE(Vec3)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec3_equals_approximately_wasm(const vec3_t *a, const vec3_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Vec3)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec3_equals_wasm(const vec3_t *a, const vec3_t *b) {
  return WMATH_EQUALS(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_lerp_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const float t) {
  *out = WMATH_LERP(Vec3)(*a, *b, t);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_lerp_v_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const vec3_t *t) {
  *out = WMATH_LERP_V(Vec3)(*a, *b, *t);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_fmax_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_FMAX(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_fmin_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_FMIN(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_multiply_scalar_wasm(vec3_t *out, const vec3_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Vec3)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_multiply_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_MULTIPLY(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_div_scalar_wasm(vec3_t *out, const vec3_t *a, const float scalar) {
  *out = WMATH_DIV_SCALAR(Vec3)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_div_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_DIV(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_inverse_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_INVERSE(Vec3)(*in);
}
WCN_WASM_EXPORT float wcn_math_Vec3_distance_wasm(const vec3_t *a, const vec3_t *b) {
  return WMATH_DISTANCE(Vec3)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec3_distance_squared_wasm(const vec3_t *a, const vec3_t *b) {
  return WMATH_DISTANCE_SQ(Vec3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_negate_wasm(vec3_t *out, const vec3_t *in) {
  *out = WMATH_NEGATE(Vec3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_random_wasm(vec3_t *out, const float scale) {
  *out = WMATH_RANDOM(Vec3)(scale);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_set_length_wasm(vec3_t *out, const vec3_t *in, const float length) {
  *out = WMATH_SET_LENGTH(Vec3)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_truncate_wasm(vec3_t *out, const vec3_t *in, const float length) {
  *out = WMATH_TRUNCATE(Vec3)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_midpoint_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b) {
  *out = WMATH_MIDPOINT(Vec3)(*a, *b);
}
// EMD Vec3

// BEGIN Vec4
WCN_WASM_EXPORT void  wcn_math_Vec4_set_wasm(vec4_t *out, const float x, const float y, const float z, const float w) {
  out->v[0] = x;
  out->v[1] = y;
  out->v[2] = z;
  out->v[3] = w;
}
WCN_WASM_EXPORT void  wcn_math_Vec4_set_x_wasm(vec4_t *out, const float x) {
  out->v[0] = x;
}
WCN_WASM_EXPORT void  wcn_math_Vec4_set_y_wasm(vec4_t *out, const float y) {
  out->v[1] = y;
}
WCN_WASM_EXPORT void  wcn_math_Vec4_set_z_wasm(vec4_t *out, const float z) {
  out->v[2] = z;
}
WCN_WASM_EXPORT void  wcn_math_Vec4_set_w_wasm(vec4_t *out, const float w) {
  out->v[3] = w;
}
WCN_WASM_EXPORT void  wcn_math_Vec4_copy_wasm(vec4_t *out, const vec4_t * in) {
  out->v[0] = in->v[0];
  out->v[1] = in->v[1];
  out->v[2] = in->v[2];
  out->v[3] = in->v[3];
}
WCN_WASM_EXPORT void  wcn_math_Vec4_zero_wasm(vec4_t *out) {
  *out = WMATH_ZERO(Vec4)();
}
WCN_WASM_EXPORT void  wcn_math_Vec4_identity_wasm(vec4_t *out) {
  *out = WMATH_ZERO(Vec4)();
}
WCN_WASM_EXPORT void  wcn_math_Vec4_ceil_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_CEIL(Vec4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_floor_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_FLOOR(Vec4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_round_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_ROUND(Vec4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_clamp_wasm(vec4_t *out, const vec4_t *a, const float min_val, const float max_val) {
  *out = WMATH_CLAMP(Vec4)(*a, min_val, max_val);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_add_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_ADD(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_add_scaled_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b, const float scale) {
  *out = WMATH_ADD_SCALED(Vec4)(*a, *b, scale);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_sub_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_SUB(Vec4)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec4_equals_approximately_wasm(const vec4_t *a, const vec4_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Vec4)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Vec4_equals_wasm(const vec4_t *a, const vec4_t *b) {
  return WMATH_EQUALS(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_lerp_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b, const float t) {
  *out = WMATH_LERP(Vec4)(*a, *b, t);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_lerp_v_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b, const vec4_t *t) {
  *out = WMATH_LERP_V(Vec4)(*a, *b, *t);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_fmax_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_FMAX(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_fmin_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_FMIN(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_multiply_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_MULTIPLY(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_multiply_scalar_wasm(vec4_t *out, const vec4_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Vec4)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_div_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_DIV(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_div_scalar_wasm(vec4_t *out, const vec4_t *a, const float scalar) {
  *out = WMATH_DIV_SCALAR(Vec4)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_inverse_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_INVERSE(Vec4)(*in);
}
WCN_WASM_EXPORT float wcn_math_Vec4_dot_wasm(const vec4_t *a, const vec4_t *b) {
  return WMATH_DOT(Vec4)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec4_length_squared_wasm(const vec4_t *in) {
  return WMATH_LENGTH_SQ(Vec4)(*in);
}
WCN_WASM_EXPORT float wcn_math_Vec4_length_wasm(const vec4_t *in) {
  return WMATH_LENGTH_SQ(Vec4)(*in);
}
WCN_WASM_EXPORT float wcn_math_Vec4_distance_wasm(const vec4_t *a, const vec4_t *b) {
  return WMATH_DISTANCE(Vec4)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Vec4_distance_squared_wasm(const vec4_t *a, const vec4_t *b) {
  return WMATH_DISTANCE_SQ(Vec4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_normalize_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_NORMALIZE(Vec4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_negate_wasm(vec4_t *out, const vec4_t *in) {
  *out = WMATH_NEGATE(Vec4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_set_length_wasm(vec4_t *out, const vec4_t *in, const float length) {
  *out = WMATH_SET_LENGTH(Vec4)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_truncate_wasm(vec4_t *out, const vec4_t *in, const float length) {
  *out = WMATH_TRUNCATE(Vec4)(*in, length);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_midpoint_wasm(vec4_t *out, const vec4_t *a, const vec4_t *b) {
  *out = WMATH_MIDPOINT(Vec4)(*a, *b);
}
// END Vec4

// BEGIN Mat3
WCN_WASM_EXPORT void  wcn_math_Mat3_set_wasm(
  mat3_t *out,
  const float m00,
  const float m01,
  const float m02,
  // next row
  const float m10,
  const float m11,
  const float m12,
  // next row
  const float m20,
  const float m21,
  const float m22
)
{
  // Using SIMD-friendly layout
  out->m[0] = m00;
  out->m[1] = m01;
  out->m[2] = m02;
  out->m[4] = m10;
  out->m[5] = m11;
  out->m[6] = m12;
  out->m[8] = m20;
  out->m[9] = m21;
  out->m[10] = m22;

  // Ensure unused elements are 0 for consistency
  out->m[3] = out->m[7] = out->m[11] = 0.0f;
}
/**
 * 设置矩阵指定位置的值
 * @param out 矩阵指针
 * @param index 坐标编码: 0x00 | 0x01 | 0x02 | 0x10 | 0x11 | 0x12 | 0x20 | 0x21 | 0x22 (High nibble = row, Low nibble = col)
 * @param value 要设置的浮点数值 (假设你需要传值)
 */
WCN_WASM_EXPORT void  wcn_math_Mat3_set_with_index_wasm(mat3_t *out, const int index, const float value) {
  // 0x12 >> 4 = 1 (Row), 0x12 & 0x0F = 2 (Col)
  const int row = (index >> 4) & 0x0F;
  const int col = index & 0x0F;
  // Row 0: 0, 1, 2, (3是padding)
  // Row 1: 4, 5, 6, (7是padding)
  // Row 2: 8, 9, 10, (11是padding)
  // Stride is 4
  const int linear_i = row * 4 + col;

  // 4. 赋值
  out->m[linear_i] = value;
}
WCN_WASM_EXPORT void  wcn_math_Mat3_copy_wasm(mat3_t *out, const mat3_t *in) {
  out->m[0] = in->m[0];
  out->m[1] = in->m[1];
  out->m[2] = in->m[2];
  out->m[4] = in->m[4];
  out->m[5] = in->m[5];
  out->m[6] = in->m[6];
  out->m[8] = in->m[8];
  out->m[9] = in->m[9];
  out->m[10] = in->m[10];
}
WCN_WASM_EXPORT void  wcn_math_Mat3_zero_wasm(mat3_t *out) {
  *out = WMATH_ZERO(Mat3)();
}
WCN_WASM_EXPORT void  wcn_math_Mat3_identity_wasm(mat3_t *out) {
  *out = WMATH_ZERO(Mat3)();
}
WCN_WASM_EXPORT bool  wcn_math_Mat3_equals_wasm(const mat3_t *a, const mat3_t *b) {
  return WMATH_EQUALS(Mat3)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Mat3_equals_approximately_wasm(const mat3_t *a, const mat3_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Mat3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_negate_wasm(mat3_t *out, const mat3_t *in) {
  *out = WMATH_NEGATE(Mat3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_transpose_wasm(mat3_t *out, const mat3_t *in) {
  *out = WMATH_TRANSPOSE(Mat3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_add_wasm(mat3_t *out, const mat3_t *a, const mat3_t *b) {
  *out = WMATH_ADD(Mat3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_sub_wasm(mat3_t *out, const mat3_t *a, const mat3_t *b) {
  *out = WMATH_SUB(Mat3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_multiply_wasm(mat3_t *out, const mat3_t *a, const mat3_t *b) {
  *out = WMATH_MULTIPLY(Mat3)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_multiply_scalar_wasm(mat3_t *out, const mat3_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Mat3)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_inverse_wasm(mat3_t *out, const mat3_t *in) {
  *out = WMATH_INVERSE(Mat3)(*in);
}
WCN_WASM_EXPORT float wcn_math_Mat3_determinant_wasm(const mat3_t *in) {
  return WMATH_DETERMINANT(Mat3)(*in);
}
// END Mat3

// BEGIN Mat4
WCN_WASM_EXPORT void  wcn_math_Mat4_set_wasm(
  mat4_t *out,
  const float m00, const float m01, const float m02, const float m03,
  const float m10, const float m11, const float m12, const float m13,
  const float m20, const float m21, const float m22, const float m23,
  const float m30, const float m31, const float m32, const float m33
) {
  out->m[0] = m00;
  out->m[1] = m01;
  out->m[2] = m02;
  out->m[3] = m03;
  out->m[4] = m10;
  out->m[5] = m11;
  out->m[6] = m12;
  out->m[7] = m13;
  out->m[8] = m20;
  out->m[9] = m21;
  out->m[10] = m22;
  out->m[11] = m23;
  out->m[12] = m30;
  out->m[13] = m31;
  out->m[14] = m32;
  out->m[15] = m33;
}
WCN_WASM_EXPORT void  wcn_math_Mat4_set_with_index_wasm(mat4_t *out, const int index, const float value) {
  const int row = (index >> 4) & 0x0F;
  const int col = index & 0x0F;
  const int linear_i = row * 4 + col;

  out->m[linear_i] = value;
}
WCN_WASM_EXPORT void  wcn_math_Mat4_copy_wasm(mat4_t *out, const mat4_t *in) {
  out->m[0] = in->m[0];
  out->m[1] = in->m[1];
  out->m[2] = in->m[2];
  out->m[3] = in->m[3];
  out->m[4] = in->m[4];
  out->m[5] = in->m[5];
  out->m[6] = in->m[6];
  out->m[7] = in->m[7];
  out->m[8] = in->m[8];
  out->m[9] = in->m[9];
  out->m[10] = in->m[10];
  out->m[11] = in->m[11];
  out->m[12] = in->m[12];
  out->m[13] = in->m[13];
  out->m[14] = in->m[14];
  out->m[15] = in->m[15];
}
WCN_WASM_EXPORT void  wcn_math_Mat4_zero_wasm(mat4_t *out) {
  *out = WMATH_ZERO(Mat4)();
}
WCN_WASM_EXPORT void  wcn_math_Mat4_identity_wasm(mat4_t *out) {
  *out = WMATH_IDENTITY(Mat4)();
}
// WMATH_NEGATE
WCN_WASM_EXPORT void  wcn_math_Mat4_negate_wasm(mat4_t *out, const mat4_t *in) {
  *out = WMATH_NEGATE(Mat4)(*in);
}
WCN_WASM_EXPORT bool  wcn_math_Mat4_equals_wasm(const mat4_t *a, const mat4_t *b) {
  return WMATH_EQUALS(Mat4)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Mat4_equals_approximately_wasm(const mat4_t *a, const mat4_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Mat4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_add_wasm(mat4_t *out, const mat4_t *a, const mat4_t *b) {
  *out = WMATH_ADD(Mat4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_sub_wasm(mat4_t *out, const mat4_t *a, const mat4_t *b) {
  *out = WMATH_SUB(Mat4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_multiply_wasm(mat4_t *out, const mat4_t *a, const mat4_t *b) {
  *out = WMATH_MULTIPLY(Mat4)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_multiply_scalar_wasm(mat4_t *out, const mat4_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Mat4)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_inverse_wasm(mat4_t *out, const mat4_t *in) {
  *out = WMATH_INVERSE(Mat4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_transpose_wasm(mat4_t *out, const mat4_t *in) {
  *out = WMATH_TRANSPOSE(Mat4)(*in);
}
WCN_WASM_EXPORT float wcn_math_Mat4_determinant_wasm(const mat4_t *in) {
  return WMATH_DETERMINANT(Mat4)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_aim_wasm(mat4_t *out, const vec3_t *pos, const vec3_t *target, const vec3_t *up) {
  *out = WMATH_CALL(Mat4, aim)(*pos, *target, *up);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_look_at_wasm(mat4_t *out, const vec3_t *eye, const vec3_t *target, const vec3_t *up) {
  *out = WMATH_CALL(Mat4, look_at)(*eye, *target, *up);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_ortho_wasm(mat4_t *out, const float left, const float right, const float bottom, const float top, const float near, const float far) {
  *out = WMATH_CALL(Mat4, ortho)(left, right, bottom, top, near, far);
}
// END Mat4

// BEGIN Quat
WCN_WASM_EXPORT void  wcn_math_Quat_set_wasm(quat_t *out, const float x, const float y, const float z, const float w) {
  out->v[0] = x;
  out->v[1] = y;
  out->v[2] = z;
  out->v[3] = w;
}
WCN_WASM_EXPORT void  wcn_math_Quat_set_x_wasm(quat_t *out, const float x) {
  out->v[0] = x;
}
WCN_WASM_EXPORT void  wcn_math_Quat_set_y_wasm(quat_t *out, const float y) {
  out->v[1] = y;
}
WCN_WASM_EXPORT void  wcn_math_Quat_set_z_wasm(quat_t *out, const float z) {
  out->v[2] = z;
}
WCN_WASM_EXPORT void  wcn_math_Quat_set_w_wasm(quat_t *out, const float w) {
  out->v[3] = w;
}
WCN_WASM_EXPORT void  wcn_math_Quat_zero_wasm(quat_t *out) {
  *out = WMATH_ZERO(Quat)();
}
WCN_WASM_EXPORT void  wcn_math_Quat_identity_wasm(quat_t *out) {
  *out = WMATH_IDENTITY(Quat)();
}
WCN_WASM_EXPORT void  wcn_math_Quat_copy_wasm(quat_t *out, const quat_t *in) {
  out->v[0] = in->v[0];
  out->v[1] = in->v[1];
  out->v[2] = in->v[2];
  out->v[3] = in->v[3];
}
WCN_WASM_EXPORT float wcn_math_Quat_dot_wasm(const quat_t *a, const quat_t *b) {
  return WMATH_DOT(Quat)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Quat_lerp_wasm(quat_t *out, const quat_t *a, const quat_t *b, const float t) {
  *out = WMATH_CALL(Quat, lerp)(*a, *b, t);
}
WCN_WASM_EXPORT void  wcn_math_Quat_slerp_wasm(quat_t *out, const quat_t *a, const quat_t *b, const float t) {
  *out = WMATH_CALL(Quat, slerp)(*a, *b, t);
}
WCN_WASM_EXPORT void  wcn_math_Quat_sqlerp_wasm(quat_t *out, const quat_t *a, const quat_t *b, const quat_t *c, const quat_t *d, const float t) {
  *out = WMATH_CALL(Quat, sqlerp)(*a, *b, *c, *d, t);
}
WCN_WASM_EXPORT float wcn_math_Quat_length_wasm(const quat_t *in) {
  return WMATH_LENGTH(Quat)(*in);
}
WCN_WASM_EXPORT float wcn_math_Quat_length_squared_wasm(const quat_t *in) {
  return WMATH_LENGTH_SQ(Quat)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Quat_normalize_wasm(quat_t *out, const quat_t *in) {
  *out = WMATH_NORMALIZE(Quat)(*in);
}
WCN_WASM_EXPORT bool  wcn_math_Quat_equals_approximately_wasm(const quat_t *a, const quat_t *b) {
  return WMATH_EQUALS_APPROXIMATELY(Quat)(*a, *b);
}
WCN_WASM_EXPORT bool  wcn_math_Quat_equals_wasm(const quat_t *a, const quat_t *b) {
  return WMATH_EQUALS(Quat)(*a, *b);
}
WCN_WASM_EXPORT float wcn_math_Quat_angle_wasm(const quat_t *a, const quat_t *b) {
  return WMATH_CALL(Quat, angle)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Quat_rotation_to_wasm(quat_t *out, const vec3_t *a_unit, const vec3_t *b_unit) {
  *out = WMATH_CALL(Quat, rotation_to)(*a_unit, *b_unit);
}
WCN_WASM_EXPORT void  wcn_math_Quat_multiply_wasm(quat_t *out, const quat_t *a, const quat_t *b) {
  *out = WMATH_MULTIPLY(Quat)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Quat_multiply_scalar_wasm(quat_t *out, const quat_t *a, const float scalar) {
  *out = WMATH_MULTIPLY_SCALAR(Quat)(*a, scalar);
}
WCN_WASM_EXPORT void  wcn_math_Quat_sub_wasm(quat_t *out, const quat_t *a, const quat_t *b) {
  *out = WMATH_SUB(Quat)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Quat_add_wasm(quat_t *out, const quat_t *a, const quat_t *b) {
  *out = WMATH_ADD(Quat)(*a, *b);
}
WCN_WASM_EXPORT void  wcn_math_Quat_inverse_wasm(quat_t *out, const quat_t *in) {
  *out = WMATH_INVERSE(Quat)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Quat_conjugate_wasm(quat_t *out, const quat_t *in) {
  *out = WMATH_CALL(Quat, conjugate)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Quat_div_scalar_wasm(quat_t *out, const quat_t *a, const float scalar) {
  *out = WMATH_DIV_SCALAR(Quat)(*a, scalar);
}
// END Quat

// FROM
// WMATH_CALL(Mat3, from_mat4)
WCN_WASM_EXPORT void  wcn_math_Mat3_from_mat4_wasm(mat3_t *out, const mat4_t *in) {
  *out = WMATH_CALL(Mat3, from_mat4)(*in);
}
// WMATH_CALL(Mat3, from_quat)
WCN_WASM_EXPORT void  wcn_math_Mat3_from_quat_wasm(mat3_t *out, const quat_t *in) {
  *out = WMATH_CALL(Mat3, from_quat)(*in);
}
// WMATH_CALL(Mat4, from_mat3)
WCN_WASM_EXPORT void  wcn_math_Mat4_from_mat3_wasm(mat4_t *out, const mat3_t *in) {
  *out = WMATH_CALL(Mat4, from_mat3)(*in);
}
// WMATH_CALL(Mat4, from_quat)
WCN_WASM_EXPORT void  wcn_math_Mat4_from_quat_wasm(mat4_t *out, const quat_t *in) {
  *out = WMATH_CALL(Mat4, from_quat)(*in);
}
// WMATH_CALL(Quat, from_axis_angle)
WCN_WASM_EXPORT void  wcn_math_Quat_from_axis_angle_wasm(quat_t *out, const vec3_t *axis, const float angle) {
  *out = WMATH_CALL(Quat, from_axis_angle)(*axis, angle);
}
// WMATH_CALL(Quat, to_axis_angle)
WCN_WASM_EXPORT void  wcn_math_Quat_to_axis_angle_wasm(vec3_aa *out, const quat_t *in) {
  *out = WMATH_CALL(Quat, to_axis_angle)(*in);
}
// WMATH_CALL(Vec2, transform_mat4)
WCN_WASM_EXPORT void  wcn_math_Vec2_transform_mat4_wasm(vec2_t *out, const vec2_t *in, const mat4_t *m) {
  *out = WMATH_CALL(Vec2, transform_mat4)(*in, *m);
}
// WMATH_CALL(Vec2, transform_mat3)
WCN_WASM_EXPORT void  wcn_math_Vec2_transform_mat3_wasm(vec2_t *out, const vec2_t *in, const mat3_t *m) {
  *out = WMATH_CALL(Vec2, transform_mat3)(*in, *m);
}
// WMATH_CALL(Vec3, transform_mat4)
WCN_WASM_EXPORT void  wcn_math_Vec3_transform_mat4_wasm(vec3_t *out, const vec3_t *in, const mat4_t *m) {
  *out = WMATH_CALL(Vec3, transform_mat4)(*in, *m);
}
// WMATH_CALL(Vec3, transform_mat4_upper3x3)
WCN_WASM_EXPORT void  wcn_math_Vec3_transform_mat4_upper3x3_wasm(vec3_t *out, const vec3_t *in, const mat4_t *m) {
  *out = WMATH_CALL(Vec3, transform_mat4_upper3x3(*in, *m));
}
// WMATH_CALL(Vec3, transform_mat3)
WCN_WASM_EXPORT void  wcn_math_Vec3_transform_mat3_wasm(vec3_t *out, const vec3_t *in, const mat3_t *m) {
  *out = WMATH_CALL(Vec3, transform_mat3(*in, *m));
}
// WMATH_CALL(Vec3, transform_quat)
WCN_WASM_EXPORT void  wcn_math_Vec3_transform_quat_wasm(vec3_t *out, const vec3_t *in, const quat_t *q) {
  *out = WMATH_CALL(Vec3, transform_quat)(*in, *q);
}
// WMATH_CALL(Quat, from_mat4)
WCN_WASM_EXPORT void  wcn_math_Quat_from_mat4_wasm(quat_t *out, const mat4_t *in) {
  *out = WMATH_CALL(Quat, from_mat4)(*in);
}
// WMATH_CALL(Quat, from_mat3)
WCN_WASM_EXPORT void  wcn_math_Quat_from_mat3_wasm(quat_t *out, const mat3_t *in) {
  *out = WMATH_CALL(Quat, from_mat3(*in));
}
// WMATH_CALL(Quat, from_euler)
WCN_WASM_EXPORT void  wcn_math_Quat_from_euler_wasm(quat_t *out, const float x_angle_in_radians, const float y_angle_in_radians,
                                    const float z_angle_in_radians,
                                    const enum WCN_Math_RotationOrder order) {
  *out = WMATH_CALL(Quat, from_euler)(x_angle_in_radians, y_angle_in_radians, z_angle_in_radians, order);
}
// END FROM

// BEGIN 3D
WCN_WASM_EXPORT void  wcn_math_Vec3_get_translation_wasm(vec3_t* out, const mat4_t* in) {
  *out = WMATH_GET_TRANSLATION(Vec3)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_get_axis_wasm(vec3_t *out, const mat4_t *in, const int axis) {
  *out = WMATH_CALL(Vec3, get_axis)(*in, axis);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_get_scale_wasm(vec3_t *out, const mat4_t *in) {
  *out = WMATH_CALL(Vec3, get_scale)(*in);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_rotate_x_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const float angle) {
  *out = WMATH_ROTATE_X(Vec3)(*a, *b, angle);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_rotate_y_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const float angle) {
  *out = WMATH_ROTATE_Y(Vec3)(*a, *b, angle);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_rotate_z_wasm(vec3_t *out, const vec3_t *a, const vec3_t *b, const float angle) {
  *out = WMATH_ROTATE_Z(Vec3)(*a, *b, angle);
}
WCN_WASM_EXPORT void  wcn_math_Vec4_transform_mat4_wasm(vec4_t *out, const vec4_t *v, const mat4_t *m) {
  *out = WMATH_CALL(Vec4, transform_mat4)(*v, *m);
}
WCN_WASM_EXPORT void  wcn_math_Quat_rotate_x_wasm(quat_t *out, const quat_t *q, const float angle) {
  *out = WMATH_ROTATE_X(Quat)(*q, angle);
}
WCN_WASM_EXPORT void  wcn_math_Quat_rotate_y_wasm(quat_t *out, const quat_t *q, const float angle) {
  *out = WMATH_ROTATE_Y(Quat)(*q, angle);
}
WCN_WASM_EXPORT void  wcn_math_Quat_rotate_z_wasm(quat_t *out, const quat_t *q, const float angle) {
  *out = WMATH_ROTATE_Z(Quat)(*q, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotate_wasm(mat3_t *out, const mat3_t *m, const float angle) {
  *out = WMATH_ROTATE(Mat3)(*m, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotate_x_wasm(mat3_t *out, const mat3_t *m, const float angle) {
  *out = WMATH_ROTATE(Mat3)(*m, -angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotate_y_wasm(mat3_t *out, const mat3_t *m, const float angle) {
  *out = WMATH_ROTATE(Mat3)(*m, -angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotate_z_wasm(mat3_t *out, const mat3_t *m, const float angle) {
  *out = WMATH_ROTATE(Mat3)(*m, -angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotation_wasm(mat3_t *out, const float angle) {
  *out = WMATH_ROTATION(Mat3)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotation_x_wasm(mat3_t *out, const float angle) {
  *out = WMATH_ROTATION_X(Mat3)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotation_y_wasm(mat3_t *out, const float angle) {
  *out = WMATH_ROTATION_Y(Mat3)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_rotation_z_wasm(mat3_t *out, const float angle) {
  *out = WMATH_ROTATION_Z(Mat3)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_get_axis_wasm(vec2_t *out, const mat3_t *m, const int axis) {
  *out = WMATH_CALL(Mat3, get_axis)(*m, axis);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_set_axis_wasm(mat3_t *out, const mat3_t *m, const vec2_t *v, const int axis) {
  *out = WMATH_CALL(Mat3, set_axis)(*m, *v, axis);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_get_scaling_wasm(vec2_t *out, const mat3_t *m) {
  *out = WMATH_CALL(Mat3, get_scaling)(*m);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_get_3D_scaling_wasm(vec3_t *out, const mat3_t *m) {
  *out = WMATH_CALL(Mat3, get_3D_scaling)(*m);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_get_translation_wasm(vec2_t *out, const mat3_t *m) {
  *out = WMATH_GET_TRANSLATION(Mat3)(*m);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_set_translation_wasm(mat3_t *out, const mat3_t *m, const vec2_t *v) {
  *out = WMATH_SET_TRANSLATION(Mat3)(*m, *v);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_translation_wasm(mat3_t *out, const vec2_t *v) {
  *out = WMATH_TRANSLATION(Mat3)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_translate_wasm(mat3_t *out, const mat3_t *m, const vec2_t *v) {
  *out = WMATH_CALL(Mat3, translate)(*m, *v);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_axis_rotate_wasm(mat4_t *out, const mat4_t *m, const vec3_t *axis, const float angle) {
  *out = WMATH_CALL(Mat4, axis_rotate)(*m, *axis, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_axis_rotation_wasm(mat4_t *out, const vec3_t *axis, const float angle) {
  *out = WMATH_CALL(Mat4, axis_rotation)(*axis, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_camera_aim_wasm(mat4_t *out, const vec3_t *eye, const vec3_t *target, const vec3_t *up) {
  *out = WMATH_CALL(Mat4, camera_aim)(*eye, *target, *up);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_frustum_wasm(mat4_t *out, const float left, const float right, const float bottom, const float top, const float near, const float far) {
  *out = WMATH_CALL(Mat4, frustum)(left, right, bottom, top, near, far);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_frustum_reverse_z_wasm(mat4_t *out, const float left, const float right, const float bottom, const float top, const float near, const float far) {
  *out = WMATH_CALL(Mat4, frustum_reverse_z)(left, right, bottom, top, near, far);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_get_axis_wasm(vec3_t *out, const mat4_t *m, const int axis) {
  *out = WMATH_CALL(Mat4, get_axis)(*m, axis);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_set_axis_wasm(mat4_t *out, const mat4_t *m, const vec3_t *v, const int axis) {
  *out = WMATH_CALL(Mat4, set_axis)(*m, *v, axis);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_get_translation_wasm(vec3_t* out, const mat4_t *m) {
  *out = WMATH_GET_TRANSLATION(Mat4)(*m);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_set_translation_wasm(mat4_t *out, const mat4_t *m, const vec3_t *v) {
  *out = WMATH_SET_TRANSLATION(Mat4)(*m, *v);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_translation_wasm(mat4_t *out, const vec3_t *v) {
  *out = WMATH_TRANSLATION(Mat4)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_perspective_wasm(mat4_t *out, const float fov, const float aspect, const float near, const float far) {
  *out = WMATH_CALL(Mat4, perspective)(fov, aspect, near, far);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_perspective_reverse_z_wasm(mat4_t *out, const float fov, const float aspect, const float near, const float far) {
  *out = WMATH_CALL(Mat4, perspective_reverse_z)(fov, aspect, near, far);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_translate_wasm(mat4_t *out, const mat4_t *m, const vec3_t *v) {
  *out = WMATH_TRANSLATION(Mat4)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotate_wasm(mat4_t *out, const mat4_t *m, const vec3_t *axis, const float angle) {
  *out = WMATH_ROTATION(Mat4)(*axis, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotate_x_wasm(mat4_t *out, const mat4_t *m, const float angle) {
  *out = WMATH_ROTATION_X(Mat4)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotate_y_wasm(mat4_t *out, const mat4_t *m, const float angle) {
  *out = WMATH_ROTATION_Y(Mat4)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotate_z_wasm(mat4_t *out, const mat4_t *m, const float angle) {
  *out = WMATH_ROTATION_Z(Mat4)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotation_wasm(mat4_t *out, const vec3_t *axis, const float angle) {
  *out = WMATH_ROTATION(Mat4)(*axis, angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotation_x_wasm(mat4_t *out, const float angle) {
  *out = WMATH_ROTATION_X(Mat4)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotation_y_wasm(mat4_t *out, const float angle) {
  *out = WMATH_ROTATION_Y(Mat4)(angle);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_rotation_z_wasm(mat4_t *out, const float angle) {
  *out = WMATH_ROTATION_Z(Mat4)(angle);
}
// END 3D
// All Type Scale Impl
WCN_WASM_EXPORT void  wcn_math_Vec2_scale_wasm(vec2_t *out, const vec2_t *v, const float s) {
  *out = WMATH_SCALE(Vec2)(*v, s);
}
WCN_WASM_EXPORT void  wcn_math_Vec3_scale_wasm(vec3_t *out, const vec3_t *v, const float s) {
  *out = WMATH_SCALE(Vec3)(*v, s);
}
WCN_WASM_EXPORT void  wcn_math_Quat_scale_wasm(quat_t *out, const quat_t *v, const float s) {
  *out = WMATH_SCALE(Quat)(*v, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_scale_wasm(mat3_t *out, const mat3_t *m, const vec2_t s) {
  *out = WMATH_SCALE(Mat3)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_scale_wasm(mat4_t *out, const mat4_t *m, const vec3_t s) {
  *out = WMATH_SCALE(Mat4)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_scale3D_wasm(mat3_t *out, const mat3_t *m, const vec3_t s) {
  *out = WMATH_CALL(Mat3, scale3D)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_scaling_wasm(mat3_t *out, const vec2_t *v) {
  *out = WMATH_CALL(Mat3, scaling)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_scaling3D_wasm(mat3_t *out, const vec3_t *v) {
  *out = WMATH_CALL(Mat3, scaling3D)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_uniform_scale_wasm(mat3_t *out, const mat3_t *m, const float s) {
  *out = WMATH_CALL(Mat3, uniform_scale)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_uniform_scale_3D_wasm(mat3_t *out, const mat3_t *m, const float s) {
  *out = WMATH_CALL(Mat3, uniform_scale_3D)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_uniform_scaling_wasm(mat3_t *out, const float s) {
  *out = WMATH_CALL(Mat3, uniform_scaling)(s);
}
WCN_WASM_EXPORT void  wcn_math_Mat3_uniform_scaling_3D_wasm(mat3_t *out, const float s) {
  *out = WMATH_CALL(Mat3, uniform_scaling_3D)(s);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_get_scaling_wasm(vec3_t *out, const mat4_t *m) {
  *out = WMATH_CALL(Mat4, get_scaling)(*m);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_scaling_wasm(mat4_t *out, const vec3_t *v) {
  *out = WMATH_CALL(Mat4, scaling)(*v);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_uniform_scale_wasm(mat4_t *out, const mat4_t *m, const float s) {
  *out = WMATH_CALL(Mat4, uniform_scale)(*m, s);
}
WCN_WASM_EXPORT void  wcn_math_Mat4_uniform_scaling_wasm(mat4_t *out, const float s) {
  *out = WMATH_CALL(Mat4, uniform_scaling)(s);
}
#endif