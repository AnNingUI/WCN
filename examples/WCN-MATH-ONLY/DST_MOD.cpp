#include "WCN/WCN_MATH_MACROS.h"
#define WCN_USE_DST_MODE
#include "WCN/WCN_Math.h"
#include <iostream>
using vec2f = WMATH_TYPE(Vec2);
using vec3f = WMATH_TYPE(Vec3);
using vec4f = WMATH_TYPE(Vec4);
using quatf = WMATH_TYPE(Quat);
using mat3f = WMATH_TYPE(Mat3);
using mat4f = WMATH_TYPE(Mat4);
int main() {
    // in c
    vec2f v1;
    vec2f v2 = {1, 2};
    vec2f v3 = {2, 3};
    #ifdef WCN_USE_DST_MODE
    WMATH_ADD(Vec2)(&v1, v2, v3);
    #else 
    v1 = WMATH_ADD(Vec2)(v2, v3);
    #endif
    std::cout << "v1.x = " << v1.v[0] << "v1.y = " << v1.v[1] << "\n";
}