#pragma once

#include "WCN/WCN_Math.h"


namespace WCN
{
    inline namespace Math {

    // =========================================================================
    // 1. 类型定义与枚举 (Types & Enums)
    // =========================================================================
    
    using                              Vec2                       =                                                                                                            WMATH_TYPE(Vec2);
    using                              Vec3                       =                                                                                                            WMATH_TYPE(Vec3);
    using                              Vec4                       =                                                                                                            WMATH_TYPE(Vec4);
    using                              Quat                       =                                                                                                            WMATH_TYPE(Quat);
    using                              Mat3                       =                                                                                                            WMATH_TYPE(Mat3);
    using                              Mat4                       =                                                                                                            WMATH_TYPE(Mat4);

    using                              RotationOrder              =                                                                                                            WCN_Math_RotationOrder;
    // 暴露旋转顺序枚举值
    constexpr                          RotationOrder RotOrder_XYZ =                                                                                                            WCN_Math_RotationOrder_XYZ;
    constexpr                          RotationOrder RotOrder_XZY =                                                                                                            WCN_Math_RotationOrder_XZY;
    constexpr                          RotationOrder RotOrder_YXZ =                                                                                                            WCN_Math_RotationOrder_YXZ;
    constexpr                          RotationOrder RotOrder_YZX =                                                                                                            WCN_Math_RotationOrder_YZX;
    constexpr                          RotationOrder RotOrder_ZXY =                                                                                                            WCN_Math_RotationOrder_ZXY;
    constexpr                          RotationOrder RotOrder_ZYX =                                                                                                            WCN_Math_RotationOrder_ZYX;

    // =========================================================================
    // 2. 全局配置与基础数学 (Config & Scalar Utils)
    // =========================================================================

    static constexpr                   float PI                   =                                                                                                            WMATH_PI;
    static constexpr                   float TWO_PI               =                                                                                                            WMATH_2PI;
    static constexpr                   float PI_OVER_2            =                                                                                                            WMATH_PI_2;
    /**
    inline WCN_Math_Vec3_WithAngleAxis transform_vector_upper_3x3 (const float left, const float right, const float bottom, const float top, const float near, const float far) {}
     */
    inline float                       get_epsilon                ()                                                                                                           { return wcn_math_get_epsilon(); }
    inline float                       set_epsilon                (const float eps)                                                                                            { return wcn_math_set_epsilon(eps); }
                      
    inline float                       to_radians                 (const float degrees)                                                                                        { return WMATH_DEG2RED(degrees); }
    inline float                       to_degrees                 (const float radians)                                                                                        { return WMATH_RED2DEG(radians); }
                      
    // 标量运算                      
    inline float                       lerp                       (const float  a, const float  b, const float  t)                                                             { return WMATH_LERP(float)(a, b, t); }
    inline double                      lerp                       (const double a, const double b, const double t)                                                             { return WMATH_LERP(double)(a, b, t); }
    inline int                         lerp                       (const int    a, const int    b, const float  t)                                                             { return WMATH_LERP(int)(a, b, t); }
                      
    inline float                       clamp                      (const float v, const float min, const float max)                                                            { return WMATH_CLAMP(float)(v, min, max); }
    inline int                         clamp                      (const int   v, const int   min, const int   max)                                                            { return WMATH_CLAMP(int)(v, min, max); }
    inline int                         clamp                      (const double v, const double min, const double max)                                                         { return WMATH_CLAMP(double)(v, min, max); }
                      
    inline float                       inverse_lerp               (const float a, const float b, const float t)                                                                { return WMATH_INVERSE_LERP(a, b, t); }
    inline float                       euclidean_modulo           (const float n, const float m)                                                                               { return WMATH_EUCLIDEAN_MODULO(n, m); }
                      
    // 随机数                      
    inline float                       random_float               ()                                                                                                           { return WMATH_RANDOM(float)(); }
    inline float                       random_range               (const float min, const float max)                                                                           { return min + random_float() * (max - min); }

    // =========================================================================
    // 3. Vec2 完整封装
    // =========================================================================

    // 构造与常量
    inline Vec2                        make_vec2                  (const float x, const float y)                                                                               { return WMATH_CREATE(Vec2)({x, y}); }
    inline Vec2                        vec2_zero                  ()                                                                                                           { return WMATH_ZERO(Vec2)(); }
    inline Vec2                        vec2_identity              ()                                                                                                           { return WMATH_IDENTITY(Vec2)(); }
    inline Vec2                        random_vec2                (const float scale = 1.0f)                                                                                   { return WMATH_RANDOM(Vec2)(scale); }

    // 运算符重载
    inline Vec2                        operator+                  (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_ADD(Vec2)(a, b); }
    inline Vec2                        operator-                  (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_SUB(Vec2)(a, b); }
    inline Vec2                        operator*                  (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_MULTIPLY(Vec2)(a, b); } // Component-wise
    inline Vec2                        operator/                  (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_DIV(Vec2)(a, b); }
    inline Vec2                        operator*                  (const Vec2 a, const float s)                                                                                { return WMATH_MULTIPLY_SCALAR(Vec2)(a, s); }
    inline Vec2                        operator*                  (const float s, const Vec2 a)                                                                                { return WMATH_MULTIPLY_SCALAR(Vec2)(a, s); }
    inline Vec2                        operator/                  (const Vec2 a, const float s)                                                                                { return WMATH_DIV_SCALAR(Vec2)(a, s); }
    inline Vec2                        operator-                  (const Vec2 a)                                                                                               { return WMATH_NEGATE(Vec2)(a); }
    inline bool                        operator==                 (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_EQUALS(Vec2)(a, b); }
    inline bool                        operator!=                 (const Vec2 a, const Vec2 b)                                                                                 { return !WMATH_EQUALS(Vec2)(a, b); }

    // 功能函数
    inline bool                        equals_approx              (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_EQUALS_APPROXIMATELY(Vec2)(a, b); }
    inline float                       dot                        (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_DOT(Vec2)(a, b); }
    inline Vec3                        cross                      (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_CROSS(Vec2)(a, b); } // 2D Cross returns Vec3 (Z component)
    inline float                       length                     (const Vec2 v)                                                                                               { return WMATH_LENGTH(Vec2)(v); }
    inline float                       length_sq                  (const Vec2 v)                                                                                               { return WMATH_LENGTH_SQ(Vec2)(v); }
    inline float                       distance                   (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_DISTANCE(Vec2)(a, b); }
    inline float                       distance_sq                (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_DISTANCE_SQ(Vec2)(a, b); }
    inline float                       angle                      (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_ANGLE(Vec2)(a, b); }
                     
    inline Vec2                        normalize                  (const Vec2 v)                                                                                               { return WMATH_NORMALIZE(Vec2)(v); }
    inline Vec2                        set_length                 (const Vec2 v, const float len)                                                                              { return WMATH_SET_LENGTH(Vec2)(v, len); }
    inline Vec2                        truncate                   (const Vec2 v, const float max_len)                                                                          { return WMATH_TRUNCATE(Vec2)(v, max_len); }
    inline Vec2                        midpoint                   (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_MIDPOINT(Vec2)(a, b); }
    inline Vec2                        rotate                     (const Vec2 v, const Vec2 center, const float rad)                                                           { return WMATH_ROTATE(Vec2)(v, center, rad); }
    inline Vec2                        inverse_component          (const Vec2 v)                                                                                               { return WMATH_INVERSE(Vec2)(v); }
                     
    // 数学工具                      
    inline Vec2                        lerp                       (const Vec2 a, const Vec2 b, const float t)                                                                  { return WMATH_LERP(Vec2)(a, b, t); }
    inline Vec2                        lerp                       (const Vec2 a, const Vec2 b, const Vec2 t)                                                                   { return WMATH_LERP_V(Vec2)(a, b, t); }
    inline Vec2                        ceil                       (const Vec2 v)                                                                                               { return WMATH_CEIL(Vec2)(v); }
    inline Vec2                        floor                      (const Vec2 v)                                                                                               { return WMATH_FLOOR(Vec2)(v); }
    inline Vec2                        round                      (const Vec2 v)                                                                                               { return WMATH_ROUND(Vec2)(v); }
    inline Vec2                        clamp                      (const Vec2 v, const float min, const float max)                                                             { return WMATH_CLAMP(Vec2)(v, min, max); }
    inline Vec2                        max                        (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_FMAX(Vec2)(a, b); }
    inline Vec2                        min                        (const Vec2 a, const Vec2 b)                                                                                 { return WMATH_FMIN(Vec2)(a, b); }
    inline Vec2                        add_scaled                 (const Vec2 a, const Vec2 b, const float s)                                                                  { return WMATH_ADD_SCALED(Vec2)(a, b, s); } // a + b * s

    // =========================================================================
    // 4. Vec3 完整封装
    // =========================================================================

    inline Vec3                        make_vec3                   (const float x, const float y, const float z)                                                               { return WMATH_CREATE(Vec3)({x, y, z}); }
    inline Vec3                        vec3_zero                   ()                                                                                                          { return WMATH_ZERO(Vec3)(); }
    inline Vec3                        random_vec3                 (const float scale = 1.0f)                                                                                  { return WMATH_RANDOM(Vec3)(scale); }
                
    inline Vec3                        operator+                   (const Vec3 a, const Vec3 b)                                                                                { return WMATH_ADD(Vec3)(a, b); }
    inline Vec3                        operator-                   (const Vec3 a, const Vec3 b)                                                                                { return WMATH_SUB(Vec3)(a, b); }
    inline Vec3                        operator*                   (const Vec3 a, const Vec3 b)                                                                                { return WMATH_MULTIPLY(Vec3)(a, b); }
    inline Vec3                        operator/                   (const Vec3 a, const Vec3 b)                                                                                { return WMATH_DIV(Vec3)(a, b); }
    inline Vec3                        operator*                   (const Vec3 a, const float s)                                                                               { return WMATH_MULTIPLY_SCALAR(Vec3)(a, s); }
    inline Vec3                        operator*                   (const float s, const Vec3 a)                                                                               { return WMATH_MULTIPLY_SCALAR(Vec3)(a, s); }
    inline Vec3                        operator/                   (const Vec3 a, const float s)                                                                               { return WMATH_DIV_SCALAR(Vec3)(a, s); }
    inline Vec3                        operator-                   (const Vec3 a)                                                                                              { return WMATH_NEGATE(Vec3)(a); }
    inline bool                        operator==                  (const Vec3 a, const Vec3 b)                                                                                { return WMATH_EQUALS(Vec3)(a, b); }
    inline bool                        operator!=                  (const Vec3 a, const Vec3 b)                                                                                { return !WMATH_EQUALS(Vec3)(a, b); }
                                                         
    inline bool                        equals_approx               (const Vec3 a, const Vec3 b)                                                                                { return WMATH_EQUALS_APPROXIMATELY(Vec3)(a, b); }
    inline float                       dot                         (const Vec3 a, const Vec3 b)                                                                                { return WMATH_DOT(Vec3)(a, b); }
    inline Vec3                        cross                       (const Vec3 a, const Vec3 b)                                                                                { return WMATH_CROSS(Vec3)(a, b); }
    inline float                       length                      (const Vec3 v)                                                                                              { return WMATH_LENGTH(Vec3)(v); }
    inline float                       length_sq                   (const Vec3 v)                                                                                              { return WMATH_LENGTH_SQ(Vec3)(v); }
    inline float                       distance                    (const Vec3 a, const Vec3 b)                                                                                { return WMATH_DISTANCE(Vec3)(a, b); }
    inline float                       distance_sq                 (const Vec3 a, const Vec3 b)                                                                                { return WMATH_DISTANCE_SQ(Vec3)(a, b); }
    inline float                       angle                       (const Vec3 a, const Vec3 b)                                                                                { return WMATH_ANGLE(Vec3)(a, b); }
                      
    inline Vec3                        normalize                   (const Vec3 v)                                                                                              { return WMATH_NORMALIZE(Vec3)(v); }
    inline Vec3                        set_length                  (const Vec3 v, const float len)                                                                             { return WMATH_SET_LENGTH(Vec3)(v, len); }
    inline Vec3                        truncate                    (const Vec3 v, const float max_len)                                                                         { return WMATH_TRUNCATE(Vec3)(v, max_len); }
    inline Vec3                        midpoint                    (const Vec3 a, const Vec3 b)                                                                                { return WMATH_MIDPOINT(Vec3)(a, b); }
    inline Vec3                        inverse_component           (const Vec3 v)                                                                                              { return WMATH_INVERSE(Vec3)(v); }
                     
    // 3D 旋转操作                     (Vector rotation) 
    inline Vec3                        rotate_x                    (const Vec3 v, const Vec3 center, const float rad)                                                          { return WMATH_ROTATE_X(Vec3)(v, center, rad); }
    inline Vec3                        rotate_y                    (const Vec3 v, const Vec3 center, const float rad)                                                          { return WMATH_ROTATE_Y(Vec3)(v, center, rad); }
    inline Vec3                        rotate_z                    (const Vec3 v, const Vec3 center, const float rad)                                                          { return WMATH_ROTATE_Z(Vec3)(v, center, rad); }
                      
    inline Vec3                        lerp                        (const Vec3 a, const Vec3 b, const float t)                                                                 { return WMATH_LERP(Vec3)(a, b, t); }
    inline Vec3                        lerp                        (const Vec3 a, const Vec3 b, const Vec3 t)                                                                  { return WMATH_LERP_V(Vec3)(a, b, t); }
    inline Vec3                        ceil                        (const Vec3 v)                                                                                              { return WMATH_CEIL(Vec3)(v); }
    inline Vec3                        floor                       (const Vec3 v)                                                                                              { return WMATH_FLOOR(Vec3)(v); }
    inline Vec3                        round                       (const Vec3 v)                                                                                              { return WMATH_ROUND(Vec3)(v); }
    inline Vec3                        clamp                       (const Vec3 v, const float min, const float max)                                                            { return WMATH_CLAMP(Vec3)(v, min, max); }
    inline Vec3                        max                         (const Vec3 a, const Vec3 b)                                                                                { return WMATH_FMAX(Vec3)(a, b); }
    inline Vec3                        min                         (const Vec3 a, const Vec3 b)                                                                                { return WMATH_FMIN(Vec3)(a, b); }
    inline Vec3                        add_scaled                  (const Vec3 a, const Vec3 b, const float s)                                                                 { return WMATH_ADD_SCALED(Vec3)(a, b, s); }

    // =========================================================================
    // 5. Vec4 完整封装
    // =========================================================================
 
    inline Vec4                        make_vec4                   (const float x, const float y, const float z, const float w)                                                { return WMATH_CREATE(Vec4)({x, y, z, w}); }
    inline Vec4                        vec4_zero                   ()                                                                                                          { return WMATH_ZERO(Vec4)(); }
    inline Vec4                        vec4_identity               ()                                                                                                          { return WMATH_IDENTITY(Vec4)(); }
                    
    inline Vec4                        operator+                   (const Vec4 a, const Vec4 b)                                                                                { return WMATH_ADD(Vec4)(a, b); }
    inline Vec4                        operator-                   (const Vec4 a, const Vec4 b)                                                                                { return WMATH_SUB(Vec4)(a, b); }
    inline Vec4                        operator*                   (const Vec4 a, const Vec4 b)                                                                                { return WMATH_MULTIPLY(Vec4)(a, b); }
    inline Vec4                        operator/                   (const Vec4 a, const Vec4 b)                                                                                { return WMATH_DIV(Vec4)(a, b); }
    inline Vec4                        operator*                   (const Vec4 a, const float s)                                                                               { return WMATH_MULTIPLY_SCALAR(Vec4)(a, s); }
    inline Vec4                        operator*                   (const float s, const Vec4 a)                                                                               { return WMATH_MULTIPLY_SCALAR(Vec4)(a, s); }
    inline Vec4                        operator/                   (const Vec4 a, const float s)                                                                               { return WMATH_DIV_SCALAR(Vec4)(a, s); }
    inline Vec4                        operator-                   (const Vec4 a)                                                                                              { return WMATH_NEGATE(Vec4)(a); }
    inline bool                        operator==                  (const Vec4 a, const Vec4 b)                                                                                { return WMATH_EQUALS(Vec4)(a, b); }
    inline bool                        operator!=                  (const Vec4 a, const Vec4 b)                                                                                { return !WMATH_EQUALS(Vec4)(a, b); }

    inline bool                        equals_approx               (const Vec4 a, const Vec4 b)                                                                                { return WMATH_EQUALS_APPROXIMATELY(Vec4)(a, b); }
    inline float                       dot                         (const Vec4 a, const Vec4 b)                                                                                { return WMATH_DOT(Vec4)(a, b); }
    inline float                       length                      (const Vec4 v)                                                                                              { return WMATH_LENGTH(Vec4)(v); }
    inline float                       length_sq                   (const Vec4 v)                                                                                              { return WMATH_LENGTH_SQ(Vec4)(v); }
    inline float                       distance                    (const Vec4 a, const Vec4 b)                                                                                { return WMATH_DISTANCE(Vec4)(a, b); }
    inline float                       distance_sq                 (const Vec4 a, const Vec4 b)                                                                                { return WMATH_DISTANCE_SQ(Vec4)(a, b); }
    
    inline Vec4                        normalize                   (const Vec4 v)                                                                                              { return WMATH_NORMALIZE(Vec4)(v); }
    inline Vec4                        set_length                  (const Vec4 v, const float len)                                                                             { return WMATH_SET_LENGTH(Vec4)(v, len); }
    inline Vec4                        truncate                    (const Vec4 v, const float max_len)                                                                         { return WMATH_TRUNCATE(Vec4)(v, max_len); }
    inline Vec4                        midpoint                    (const Vec4 a, const Vec4 b)                                                                                { return WMATH_MIDPOINT(Vec4)(a, b); }
    inline Vec4                        inverse_component           (const Vec4 v)                                                                                              { return WMATH_INVERSE(Vec4)(v); }

    inline Vec4                        lerp                        (const Vec4 a, const Vec4 b, const float t)                                                                 { return WMATH_LERP(Vec4)(a, b, t); }
    inline Vec4                        lerp                        (const Vec4 a, const Vec4 b, const Vec4 t)                                                                  { return WMATH_LERP_V(Vec4)(a, b, t); }
    inline Vec4                        ceil                        (const Vec4 v)                                                                                              { return WMATH_CEIL(Vec4)(v); }
    inline Vec4                        floor                       (const Vec4 v)                                                                                              { return WMATH_FLOOR(Vec4)(v); }
    inline Vec4                        round                       (const Vec4 v)                                                                                              { return WMATH_ROUND(Vec4)(v); }
    inline Vec4                        clamp                       (const Vec4 v, const float min, const float max)                                                            { return WMATH_CLAMP(Vec4)(v, min, max); }
    inline Vec4                        max                         (const Vec4 a, const Vec4 b)                                                                                { return WMATH_FMAX(Vec4)(a, b); }
    inline Vec4                        min                         (const Vec4 a, const Vec4 b)                                                                                { return WMATH_FMIN(Vec4)(a, b); }
    inline Vec4                        add_scaled                  (const Vec4 a, const Vec4 b, const float s)                                                                 { return WMATH_ADD_SCALED(Vec4)(a, b, s); }

    // =========================================================================
    // 6. Quat 完整封装
    // =========================================================================

    inline Quat                        make_quat                   (const float x, const float y, const float z, const float w)                                                { return WMATH_CREATE(Quat)({x, y, z, w}); }
    inline Quat                        quat_identity               ()                                                                                                          { return WMATH_IDENTITY(Quat)(); }
    inline Quat                        quat_zero                   ()                                                                                                          { return WMATH_ZERO(Quat)(); }
                      
    inline Quat                        operator+                   (const Quat a, const Quat b)                                                                                { return WMATH_ADD(Quat)(a, b); }
    inline Quat                        operator-                   (const Quat a, const Quat b)                                                                                { return WMATH_SUB(Quat)(a, b); }
    inline Quat                        operator*                   (const Quat a, const Quat b)                                                                                { return WMATH_MULTIPLY(Quat)(a, b); }
    inline Quat                        operator*                   (const Quat a, const float s)                                                                               { return WMATH_MULTIPLY_SCALAR(Quat)(a, s); }
    inline Quat                        operator*                   (const float s, const Quat a)                                                                               { return WMATH_MULTIPLY_SCALAR(Quat)(a, s); }
    inline Quat                        operator/                   (const Quat a, const float s)                                                                               { return WMATH_DIV_SCALAR(Quat)(a, s); }
    inline bool                        operator==                  (const Quat a, const Quat b)                                                                                { return WMATH_EQUALS(Quat)(a, b); }
    inline bool                        operator!=                  (const Quat a, const Quat b)                                                                                { return !WMATH_EQUALS(Quat)(a, b); }
                      
    inline bool                        equals_approx               (const Quat a, const Quat b)                                                                                { return WMATH_EQUALS_APPROXIMATELY(Quat)(a, b); }
    inline float                       dot                         (const Quat a, const Quat b)                                                                                { return WMATH_DOT(Quat)(a, b); }
    inline float                       length                      (const Quat q)                                                                                              { return WMATH_LENGTH(Quat)(q); }
    inline float                       length_sq                   (const Quat q)                                                                                              { return WMATH_LENGTH_SQ(Quat)(q); }
    inline float                       angle                       (const Quat a, const Quat b)                                                                                { return WMATH_ANGLE(Quat)(a, b); }
                      
    inline Quat                        normalize                   (const Quat q)                                                                                              { return WMATH_NORMALIZE(Quat)(q); }
    inline Quat                        invert                      (const Quat q)                                                                                              { return WMATH_INVERSE(Quat)(q); }
    inline Quat                        conjugate                   (const Quat q)                                                                                              { return WMATH_CALL(Quat, conjugate)(q); }
                          
    // 旋转操作 (Quaternion rotation)
    inline Quat                        rotate_x                    (const Quat q, const float rad)                                                                             { return WMATH_ROTATE_X(Quat)(q, rad); }
    inline Quat                        rotate_y                    (const Quat q, const float rad)                                                                             { return WMATH_ROTATE_Y(Quat)(q, rad); }
    inline Quat                        rotate_z                    (const Quat q, const float rad)                                                                             { return WMATH_ROTATE_Z(Quat)(q, rad); }
                      
    // 插值与构造                      
    inline Quat                        lerp                        (const Quat a, const Quat b, const float t)                                                                 { return WMATH_LERP(Quat)(a, b, t); }
    inline Quat                        slerp                       (const Quat a, const Quat b, const float t)                                                                 { return WMATH_CALL(Quat, slerp)(a, b, t); }
    inline Quat                        sqlerp                      (const Quat a, const Quat b, const Quat c, const Quat d, const float t)                                     { return WMATH_CALL(Quat, sqlerp)(a, b, c, d, t); }
                         
    inline Quat                        from_axis_angle             (const Vec3 axis, const float rad)                                                                          { return WMATH_CALL(Quat, from_axis_angle)(axis, rad); }
    inline Quat                        from_euler                  (const float x, const float y, const float z, const RotationOrder order = RotOrder_XYZ)                     { return WMATH_CALL(Quat, from_euler)(x, y, z, order); }
    inline Quat                        rotation_to                 (const Vec3 from, const Vec3 to)                                                                            { return WMATH_CALL(Quat, rotation_to)(from, to); }
    inline WCN_Math_Vec3_WithAngleAxis to_axis_angle               (const Quat q)                                                                                              { return WMATH_CALL(Quat, to_axis_angle)(q); }

    // =========================================================================
    // 7. Mat3 完整封装
    // =========================================================================

    inline Mat3                        mat3_identity               ()                                                                                                          { return WMATH_IDENTITY(Mat3)(); }
    inline Mat3                        mat3_zero                   ()                                                                                                          { return WMATH_ZERO(Mat3)(); }
    
    // 创建
    inline Mat3                        make_mat3_translation       (const Vec2 v)                                                                                              { return WMATH_TRANSLATION(Mat3)(v); }
    inline Mat3                        make_mat3_scaling           (const Vec2 v)                                                                                              { return WMATH_CALL(Mat3, scaling)(v); }
    inline Mat3                        make_mat3_scaling           (const Vec3 v)                                                                                              { return WMATH_CALL(Mat3, scaling3D)(v); } // 3D scaling in 3x3
    inline Mat3                        make_mat3_rotation          (const float rad)                                                                                           { return WMATH_ROTATION(Mat3)(rad); }
    inline Mat3                        make_mat3_rotation_x        (const float rad)                                                                                           { return WMATH_ROTATION_X(Mat3)(rad); }
    inline Mat3                        make_mat3_rotation_y        (const float rad)                                                                                           { return WMATH_ROTATION_Y(Mat3)(rad); }
    inline Mat3                        make_mat3_rotation_z        (const float rad)                                                                                           { return WMATH_ROTATION_Z(Mat3)(rad); }
                  
    // 运算符                  
    inline Mat3                        operator+                   (const Mat3 &a, const Mat3 &b)                                                                              { return WMATH_ADD(Mat3)(a, b); }
    inline Mat3                        operator-                   (const Mat3 &a, const Mat3 &b)                                                                              { return WMATH_SUB(Mat3)(a, b); }
    inline Mat3                        operator*                   (const Mat3 &a, const Mat3 &b)                                                                              { return WMATH_MULTIPLY(Mat3)(a, b); }
    inline Mat3                        operator*                   (const Mat3 &a, const float s)                                                                              { return WMATH_MULTIPLY_SCALAR(Mat3)(a, s); }
    inline Mat3                        operator-                   (const Mat3 &a)                                                                                             { return WMATH_NEGATE(Mat3)(a); }
    inline bool                        operator==                  (const Mat3 &a, const Mat3 &b)                                                                              { return WMATH_EQUALS(Mat3)(a, b); }
    inline bool                        operator!=                  (const Mat3 &a, const Mat3 &b)                                                                              { return !WMATH_EQUALS(Mat3)(a, b); }

    // 变换应用 (Apply Transform)
    inline Vec2                        operator*                   (const Mat3 &m, const Vec2 v)                                                                               { return WMATH_CALL(Vec2, transform_mat3)(v, m); }
    inline Vec3                        operator*                   (const Mat3 &m, const Vec3 v)                                                                               { return WMATH_CALL(Vec3, transform_mat3)(v, m); }

    // 操作
    inline bool                        equals_approx               (const Mat3 &a, const Mat3 &b)                                                                              { return WMATH_EQUALS_APPROXIMATELY(Mat3)(a, b); }
    inline Mat3                        transpose                   (const Mat3 &m)                                                                                             { return WMATH_TRANSPOSE(Mat3)(m); }
    inline Mat3                        inverse                     (const Mat3 &m)                                                                                             { return WMATH_INVERSE(Mat3)(m); }
    inline float                       determinant                 (const Mat3 &m)                                                                                             { return WMATH_DETERMINANT(Mat3)(m); }

    // 变换修改 (Modifiers)
    inline Mat3                        translate                   (const Mat3 &m, const Vec2 v)                                                                               { return WMATH_CALL(Mat3, translate)(m, v); }
    inline Mat3                        rotate                      (const Mat3 &m, const float rad)                                                                            { return WMATH_ROTATE(Mat3)(m, rad); }
    inline Mat3                        rotate_x                    (const Mat3 &m, const float rad)                                                                            { return WMATH_ROTATE_X(Mat3)(m, rad); }
    inline Mat3                        rotate_y                    (const Mat3 &m, const float rad)                                                                            { return WMATH_ROTATE_Y(Mat3)(m, rad); }
    inline Mat3                        rotate_z                    (const Mat3 &m, const float rad)                                                                            { return WMATH_ROTATE_Z(Mat3)(m, rad); }
    inline Mat3                        scale                       (const Mat3 &m, const Vec2 v)                                                                               { return WMATH_SCALE(Mat3)(m, v); }
    inline Mat3                        scale                       (const Mat3 &m, const Vec3 v)                                                                               { return WMATH_CALL(Mat3, scale3D)(m, v); }
    inline Mat3                        uniform_scale               (const Mat3 &m, const float s)                                                                              { return WMATH_CALL(Mat3, uniform_scale)(m, s); }
    inline Mat3                        uniform_scale_3d            (const Mat3 &m, const float s)                                                                              { return WMATH_CALL(Mat3, uniform_scale_3D)(m, s); }

    // 访问器
    inline Vec2                        get_translation             (const Mat3 &m)                                                                                             { return WMATH_GET_TRANSLATION(Mat3)(m); }
    inline Mat3                        set_translation             (const Mat3 &m, const Vec2 v)                                                                               { return WMATH_SET_TRANSLATION(Mat3)(m, v); }
    inline Vec2                        get_axis                    (const Mat3 &m, const int axis)                                                                             { return WMATH_CALL(Mat3, get_axis)(m, axis); }
    inline Mat3                        set_axis                    (const Mat3 &m, const Vec2 v, const int axis)                                                               { return WMATH_CALL(Mat3, set_axis)(m, v, axis); }
    inline Vec2                        get_scaling                 (const Mat3 &m)                                                                                             { return WMATH_CALL(Mat3, get_scaling)(m); }
    inline Vec3                        get_scaling_3d              (const Mat3 &m)                                                                                             { return WMATH_CALL(Mat3, get_3D_scaling)(m); }

    // =========================================================================
    // 8. Mat4 完整封装
    // =========================================================================

    inline Mat4                        mat4_identity               ()                                                                                                          { return WMATH_IDENTITY(Mat4)(); }
    inline Mat4                        mat4_zero                   ()                                                                                                          { return WMATH_ZERO(Mat4)(); }

    // 创建 (Factories)
    inline Mat4                        make_mat4_translation       (const Vec3 v)                                                                                              { return WMATH_TRANSLATION(Mat4)(v); }
    inline Mat4                        make_mat4_scaling           (const Vec3 v)                                                                                              { return WMATH_CALL(Mat4, scaling)(v); }
    inline Mat4                        make_mat4_uniform_scaling   (const float s)                                                                                             { return WMATH_CALL(Mat4, uniform_scaling)(s); }
    inline Mat4                        make_mat4_rotation          (const Vec3 axis, const float rad)                                                                          { return WMATH_ROTATION(Mat4)(axis, rad); }
    inline Mat4                        make_mat4_rotation_x        (const float rad)                                                                                           { return WMATH_ROTATION_X(Mat4)(rad); }
    inline Mat4                        make_mat4_rotation_y        (const float rad)                                                                                           { return WMATH_ROTATION_Y(Mat4)(rad); }
    inline Mat4                        make_mat4_rotation_z        (const float rad)                                                                                           { return WMATH_ROTATION_Z(Mat4)(rad); }
    inline Mat4                        make_mat4_axis_rotation     (const Vec3 axis, const float rad)                                                                          { return WMATH_CALL(Mat4, axis_rotation)(axis, rad); }

    // 摄像机与投影
    inline Mat4                        look_at                     (const Vec3 eye, const Vec3 target, const Vec3 up)                                                          { return WMATH_CALL(Mat4, look_at)(eye, target, up); }
    inline Mat4                        aim                         (const Vec3 pos, const Vec3 target, const Vec3 up)                                                          { return WMATH_CALL(Mat4, aim)(pos, target, up); }
    inline Mat4                        camera_aim                  (const Vec3 eye, const Vec3 target, const Vec3 up)                                                          { return WMATH_CALL(Mat4, camera_aim)(eye, target, up); } // Inverse of lookAt
    
    inline Mat4                        ortho                       (const float left, const float right, const float bottom, const float top, const float near, const float far)
    {
           return                      WMATH_CALL(Mat4, ortho)                                                                             (left, right, bottom, top, near, far);
    }
    inline Mat4                        frustum                     (const float left, const float right, const float bottom, const float top, const float near, const float far)
    {
           return                      WMATH_CALL(Mat4, frustum)                                                                           (left, right, bottom, top, near, far);
    }
    inline Mat4                        frustum_reverse_z           (const float left, const float right, const float bottom, const float top, const float near, const float far) 
    {
           return                      WMATH_CALL(Mat4, frustum_reverse_z)                                                                 (left, right, bottom, top, near, far);
    }
    inline Mat4                        perspective                 (const float fovY, const float aspect, const float zNear, const float zFar)
    {
           return                      WMATH_CALL(Mat4, perspective)                        (fovY, aspect, zNear, zFar);
    }
    inline Mat4                        perspective_reverse_z       (const float fovY, const float aspect, const float zNear, const float zFar)
    {
           return                      WMATH_CALL(Mat4, perspective_reverse_z)              (fovY, aspect, zNear, zFar);
    }

    // 运算符
    inline Mat4                        operator+                    (const Mat4 &a, const Mat4 &b)                                                                             { return WMATH_ADD(Mat4)(a, b); }
    inline Mat4                        operator-                    (const Mat4 &a, const Mat4 &b)                                                                             { return WMATH_SUB(Mat4)(a, b); }
    inline Mat4                        operator*                    (const Mat4 &a, const Mat4 &b)                                                                             { return WMATH_MULTIPLY(Mat4)(a, b); }
    inline Mat4                        operator*                    (const Mat4 &a, const float s)                                                                             { return WMATH_MULTIPLY_SCALAR(Mat4)(a, s); }
    inline Mat4                        operator-                    (const Mat4 &a)                                                                                            { return WMATH_NEGATE(Mat4)(a); }
    inline bool                        operator==                   (const Mat4 &a, const Mat4 &b)                                                                             { return WMATH_EQUALS(Mat4)(a, b); }
    inline bool                        operator!=                   (const Mat4 &a, const Mat4 &b)                                                                             { return !WMATH_EQUALS(Mat4)(a, b); }

    // 变换应用
    inline Vec4                        operator*                    (const Mat4 &m, const Vec4 v)                                                                              { return WMATH_CALL(Vec4, transform_mat4)(v, m); }
    inline Vec3                        operator*                    (const Mat4 &m, const Vec3 v)                                                                              { return WMATH_CALL(Vec3, transform_mat4)(v, m); }
    inline Vec2                        operator*                    (const Mat4 &m, const Vec2 v)                                                                              { return WMATH_CALL(Vec2, transform_mat4)(v, m); }
    // 特殊：仅变换 Vec3 上半部分 3x3 (用于法线等)
    inline Vec3                        transform_vector_upper_3x3   (const Mat4 &m, const Vec3 v)                                                                              { return WMATH_CALL(Vec3, transform_mat4_upper3x3)(v, m); }

    // 操作
    inline bool                        equals_approx                (const Mat4 &a, const Mat4 &b)                                                                             { return WMATH_EQUALS_APPROXIMATELY(Mat4)(a, b); }
    inline Mat4                        transpose                    (const Mat4 &m)                                                                                            { return WMATH_TRANSPOSE(Mat4)(m); }
    inline Mat4                        inverse                      (const Mat4 &m)                                                                                            { return WMATH_INVERSE(Mat4)(m); }
    inline float                       determinant                  (const Mat4 &m)                                                                                            { return WMATH_DETERMINANT(Mat4)(m); }

    // 变换修改 (Modifiers)
    inline Mat4                        translate                    (const Mat4 &m, const Vec3 v)                                                                              { return WMATH_CALL(Mat4, translate)(m, v); }
    inline Mat4                        rotate                       (const Mat4 &m, const Vec3 axis, const float rad)                                                          { return WMATH_ROTATE(Mat4)(m, axis, rad); }
    inline Mat4                        rotate_x                     (const Mat4 &m, const float rad)                                                                           { return WMATH_ROTATE_X(Mat4)(m, rad); }
    inline Mat4                        rotate_y                     (const Mat4 &m, const float rad)                                                                           { return WMATH_ROTATE_Y(Mat4)(m, rad); }
    inline Mat4                        rotate_z                     (const Mat4 &m, const float rad)                                                                           { return WMATH_ROTATE_Z(Mat4)(m, rad); }
    inline Mat4                        axis_rotate                  (const Mat4 &m, const Vec3 axis, const float rad)                                                          { return WMATH_CALL(Mat4, axis_rotate)(m, axis, rad); }
    inline Mat4                        scale                        (const Mat4 &m, const Vec3 v)                                                                              { return WMATH_SCALE(Mat4)(m, v); }
    inline Mat4                        uniform_scale                (const Mat4 &m, const float s)                                                                             { return WMATH_CALL(Mat4, uniform_scale)(m, s); }

    // 访问器
    inline Vec3                        get_translation              (const Mat4 &m)                                                                                            { return WMATH_GET_TRANSLATION(Mat4)(m); }
    inline Mat4                        set_translation              (const Mat4 &m, const Vec3 v)                                                                              { return WMATH_SET_TRANSLATION(Mat4)(m, v); }
    inline Vec3                        get_axis                     (const Mat4 &m, const int axis)                                                                            { return WMATH_CALL(Mat4, get_axis)(m, axis); }
    inline Mat4                        set_axis                     (const Mat4 &m, const Vec3 v, const int axis)                                                              { return WMATH_CALL(Mat4, set_axis)(m, v, axis); }
    inline Vec3                        get_scaling                  (const Mat4 &m)                                                                                            { return WMATH_CALL(Mat4, get_scaling)(m); }

    // =========================================================================
    // 9. 类型转换 (Conversions)
    // =========================================================================

    inline Mat3                        to_mat3                      (const Mat4 &m)                                                                                            { return WMATH_CALL(Mat3, from_mat4)(m); }
    inline Mat3                        to_mat3                      (const Quat q)                                                                                             { return WMATH_CALL(Mat3, from_quat)(q); }
                                                                                                                                            
    inline Mat4                        to_mat4                      (const Mat3 &m)                                                                                            { return WMATH_CALL(Mat4, from_mat3)(m); }
    inline Mat4                        to_mat4                      (const Quat q)                                                                                             { return WMATH_CALL(Mat4, from_quat)(q); }
                                                                                                                                        
    inline Quat                        to_quat                      (const Mat4 &m)                                                                                            { return WMATH_CALL(Quat, from_mat4)(m); }
    inline Quat                        to_quat                      (const Mat3 &m)                                                                                            { return WMATH_CALL(Quat, from_mat3)(m); }
                                                                                                     
    inline Vec3                        transform_quat               (const Quat q, const Vec3 v)                                                                               { return WMATH_CALL(Vec3, transform_quat)(v, q); }
    } 
} // namespace WCN::Math
