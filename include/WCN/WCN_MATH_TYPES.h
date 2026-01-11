#ifndef WCN_MATH_TYPES_H
#define WCN_MATH_TYPES_H

#include "WCN/WCN_MATH_MACROS.h"

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
#endif // WCN_MATH_TYPES_H