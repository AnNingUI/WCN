#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "WCN/WCN_Math.h"

/**
 *
 * 准确性测试
 */
bool TestVec2() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-6f);
    
    // Test create
    WMATH_TYPE(Vec2) v1 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    if (v1.v[0] != 1.0f || v1.v[1] != 2.0f) {
        printf("Vec2 create test failed\n");
        return false;
    }
    
    // Test set
    WMATH_TYPE(Vec2) v2 = WMATH_SET(Vec2)(v1, 3.0f, 4.0f);
    if (v2.v[0] != 3.0f || v2.v[1] != 4.0f) {
        printf("Vec2 set test failed\n");
        return false;
    }
    
    // Test copy
    WMATH_TYPE(Vec2) v3 = WMATH_COPY(Vec2)(v2);
    if (v3.v[0] != 3.0f || v3.v[1] != 4.0f) {
        printf("Vec2 copy test failed\n");
        return false;
    }
    
    // Test zero
    WMATH_TYPE(Vec2) v4 = WMATH_ZERO(Vec2)();
    if (v4.v[0] != 0.0f || v4.v[1] != 0.0f) {
        printf("Vec2 zero test failed\n");
        return false;
    }
    
    // Test identity
    WMATH_TYPE(Vec2) v5 = WMATH_IDENTITY(Vec2)();
    if (v5.v[0] != 1.0f || v5.v[1] != 1.0f) {
        printf("Vec2 identity test failed\n");
        return false;
    }
    
    // Test ceil
    WMATH_TYPE(Vec2) v6 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.3f, .v_y = 2.7f});
    WMATH_TYPE(Vec2) v7 = WMATH_CEIL(Vec2)(v6);
    if (v7.v[0] != ceilf(1.3f) || v7.v[1] != ceilf(2.7f)) {
        printf("Vec2 ceil test failed\n");
        return false;
    }
    
    // Test floor
    WMATH_TYPE(Vec2) v8 = WMATH_FLOOR(Vec2)(v6);
    if (v8.v[0] != floorf(1.3f) || v8.v[1] != floorf(2.7f)) {
        printf("Vec2 floor test failed\n");
        return false;
    }
    
    // Test round
    WMATH_TYPE(Vec2) v9 = WMATH_ROUND(Vec2)(v6);
    if (v9.v[0] != roundf(1.3f) || v9.v[1] != roundf(2.7f)) {
        printf("Vec2 round test failed\n");
        return false;
    }
    
    // Test clamp
    WMATH_TYPE(Vec2) v10 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = -2.0f, .v_y = 5.0f});
    WMATH_TYPE(Vec2) v11 = WMATH_CLAMP(Vec2)(v10, 0.0f, 3.0f);
    if (v11.v[0] != 0.0f || v11.v[1] != 3.0f) {
        printf("Vec2 clamp test failed\n");
        return false;
    }
    
    // Test add
    WMATH_TYPE(Vec2) v12 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Vec2) v13 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 3.0f, .v_y = 4.0f});
    WMATH_TYPE(Vec2) v14 = WMATH_ADD(Vec2)(v12, v13);
    if (v14.v[0] != 4.0f || v14.v[1] != 6.0f) {
        printf("Vec2 add test failed\n");
        return false;
    }
    
    // Test addScaled
    WMATH_TYPE(Vec2) v15 = WMATH_ADD_SCALED(Vec2)(v12, v13, 2.0f);
    if (v15.v[0] != 7.0f || v15.v[1] != 10.0f) {
        printf("Vec2 addScaled test failed\n");
        return false;
    }
    
    // Test angle
    WMATH_TYPE(Vec2) v16 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 0.0f});
    WMATH_TYPE(Vec2) v17 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 0.0f, .v_y = 1.0f});
    float angle = WMATH_ANGLE(Vec2)(v16, v17);
    if (fabsf(angle - WMATH_PI_2) > wcn_math_get_epsilon()) {
        printf("Vec2 angle test failed\n");
        return false;
    }
    
    // Test subtract
    WMATH_TYPE(Vec2) v18 = WMATH_SUB(Vec2)(v13, v12);
    if (v18.v[0] != 2.0f || v18.v[1] != 2.0f) {
        printf("Vec2 subtract test failed\n");
        return false;
    }
    
    // Test equalsApproximately
    WMATH_TYPE(Vec2) v19 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Vec2) v20 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f + wcn_math_get_epsilon()/2, .v_y = 2.0f});
    if (!WMATH_EQUALS_APPROXIMATELY(Vec2)(v19, v20)) {
        printf("Vec2 equalsApproximately test failed\n");
        return false;
    }
    
    // Test equals
    WMATH_TYPE(Vec2) v21 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Vec2) v22 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    if (!WMATH_EQUALS(Vec2)(v21, v22)) {
        printf("Vec2 equals test failed\n");
        return false;
    }
    
    // Test lerp
    WMATH_TYPE(Vec2) v23 = WMATH_LERP(Vec2)(v12, v13, 0.5f);
    if (v23.v[0] != 2.0f || v23.v[1] != 3.0f) {
        printf("Vec2 lerp test failed\n");
        return false;
    }
    
    // Test lerpV
    WMATH_TYPE(Vec2) v24 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 0.5f, .v_y = 0.5f});
    WMATH_TYPE(Vec2) v25 = WMATH_LERP_V(Vec2)(v12, v13, v24);
    if (v25.v[0] != 2.0f || v25.v[1] != 3.0f) {
        printf("Vec2 lerpV test failed\n");
        return false;
    }
    
    // Test max
    WMATH_TYPE(Vec2) v26 = WMATH_FMAX(Vec2)(v12, v13);
    if (v26.v[0] != 3.0f || v26.v[1] != 4.0f) {
        printf("Vec2 max test failed\n");
        return false;
    }
    
    // Test min
    WMATH_TYPE(Vec2) v27 = WMATH_FMIN(Vec2)(v12, v13);
    if (v27.v[0] != 1.0f || v27.v[1] != 2.0f) {
        printf("Vec2 min test failed\n");
        return false;
    }
    
    // Test mulScalar
    WMATH_TYPE(Vec2) v28 = WMATH_MULTIPLY_SCALAR(Vec2)(v12, 2.0f);
    if (v28.v[0] != 2.0f || v28.v[1] != 4.0f) {
        printf("Vec2 mulScalar test failed\n");
        return false;
    }
    
    // Test multiply
    WMATH_TYPE(Vec2) v29 = WMATH_MULTIPLY(Vec2)(v12, v13);
    if (v29.v[0] != 3.0f || v29.v[1] != 8.0f) {
        printf("Vec2 multiply test failed\n");
        return false;
    }
    
    // Test divScalar
    WMATH_TYPE(Vec2) v30 = WMATH_DIV_SCALAR(Vec2)(v13, 2.0f);
    if (v30.v[0] != 1.5f || v30.v[1] != 2.0f) {
        printf("Vec2 divScalar test failed\n");
        return false;
    }
    
    // Test inverse
    WMATH_TYPE(Vec2) v31 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 2.0f, .v_y = 4.0f});
    WMATH_TYPE(Vec2) v32 = WMATH_INVERSE(Vec2)(v31);
    if (fabsf(v32.v[0] - 0.5f) > wcn_math_get_epsilon() || fabsf(v32.v[1] - 0.25f) > wcn_math_get_epsilon()) {
        printf("Vec2 inverse test failed\n");
        return false;
    }
    
    // Test cross
    WMATH_TYPE(Vec2) v33 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 0.0f});
    WMATH_TYPE(Vec2) v34 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 0.0f, .v_y = 1.0f});
    WMATH_TYPE(Vec3) v35 = WMATH_CROSS(Vec2)(v33, v34);
    if (v35.v[0] != 0.0f || v35.v[1] != 0.0f || v35.v[2] != 1.0f) {
        printf("Vec2 cross test failed\n");
        return false;
    }
    
    // Test dot
    float dot = WMATH_DOT(Vec2)(v12, v13);
    if (dot != 11.0f) {
        printf("Vec2 dot test failed\n");
        return false;
    }
    
    // Test length
    float length = WMATH_LENGTH(Vec2)(v12);
    if (fabsf(length - sqrtf(5.0f)) > wcn_math_get_epsilon()) {
        printf("Vec2 length test failed\n");
        return false;
    }
    
    // Test lengthSq
    float lengthSq = WMATH_LENGTH_SQ(Vec2)(v12);
    if (lengthSq != 5.0f) {
        printf("Vec2 lengthSq test failed\n");
        return false;
    }
    
    // Test distance
    float distance = WMATH_DISTANCE(Vec2)(v12, v13);
    if (fabsf(distance - sqrtf(8.0f)) > wcn_math_get_epsilon()) {
        printf("Vec2 distance test failed\n");
        return false;
    }
    
    // Test distanceSq
    float distanceSq = WMATH_DISTANCE_SQ(Vec2)(v12, v13);
    if (distanceSq != 8.0f) {
        printf("Vec2 distanceSq test failed\n");
        return false;
    }
    
    // Test normalize
    WMATH_TYPE(Vec2) v36 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 3.0f, .v_y = 4.0f});
    WMATH_TYPE(Vec2) v37 = WMATH_NORMALIZE(Vec2)(v36);
    float len = sqrtf(3.0f*3.0f + 4.0f*4.0f);
    if (fabsf(v37.v[0] - 3.0f/len) > wcn_math_get_epsilon() || 
        fabsf(v37.v[1] - 4.0f/len) > wcn_math_get_epsilon()) {
        printf("Vec2 normalize test failed\n");
        return false;
    }
    
    // Test negate
    WMATH_TYPE(Vec2) v38 = WMATH_NEGATE(Vec2)(v12);
    if (v38.v[0] != -1.0f || v38.v[1] != -2.0f) {
        printf("Vec2 negate test failed\n");
        return false;
    }
    
    // Test random
    WMATH_TYPE(Vec2) v39 = WMATH_RANDOM(Vec2)(1.0f);
    float randomLength = WMATH_LENGTH(Vec2)(v39);
    if (fabsf(randomLength - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 random test failed\n");
        return false;
    }
    
    // Test rotate
    WMATH_TYPE(Vec2) v40 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 0.0f});
    WMATH_TYPE(Vec2) v41 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 0.0f, .v_y = 0.0f});
    WMATH_TYPE(Vec2) v42 = WMATH_ROTATE(Vec2)(v40, v41, WMATH_PI_2);
    if (fabsf(v42.v[0] - 0.0f) > wcn_math_get_epsilon() || 
        fabsf(v42.v[1] - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 rotate test failed\n");
        return false;
    }
    
    // Test setLength
    WMATH_TYPE(Vec2) v43 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 3.0f, .v_y = 4.0f});
    WMATH_TYPE(Vec2) v44 = WMATH_SET_LENGTH(Vec2)(v43, 10.0f);
    float newLength = WMATH_LENGTH(Vec2)(v44);
    if (fabsf(newLength - 10.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 setLength test failed\n");
        return false;
    }
    
    // Test midpoint
    WMATH_TYPE(Vec2) v45 = WMATH_MIDPOINT(Vec2)(v12, v13);
    if (v45.v[0] != 2.0f || v45.v[1] != 3.0f) {
        printf("Vec2 midpoint test failed\n");
        return false;
    }

    // Test truncate
    WMATH_TYPE(Vec2) v46 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 6.0f, .v_y = 8.0f});
    WMATH_TYPE(Vec2) v47 = WMATH_TRUNCATE(Vec2)(v46, 5.0f);
    float truncLength = WMATH_LENGTH(Vec2)(v47);
    if (fabsf(truncLength - 5.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 truncate test failed\n");
        return false;
    }

    printf("All Vec2 tests passed!\n");
    return true;
}

bool TestVec3() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-6f);

    // Test create
    WMATH_TYPE(Vec3) v1 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    if (v1.v[0] != 1.0f || v1.v[1] != 2.0f || v1.v[2] != 3.0f) {
        printf("Vec3 create test failed\n");
        return false;
    }

    // Test copy
    WMATH_TYPE(Vec3) v2 = WMATH_COPY(Vec3)(v1);
    if (v2.v[0] != 1.0f || v2.v[1] != 2.0f || v2.v[2] != 3.0f) {
        printf("Vec3 copy test failed\n");
        return false;
    }

    // Test set
    WMATH_TYPE(Vec3) v3 = WMATH_SET(Vec3)(v1, 4.0f, 5.0f, 6.0f);
    if (v3.v[0] != 4.0f || v3.v[1] != 5.0f || v3.v[2] != 6.0f) {
        printf("Vec3 set test failed\n");
        return false;
    }

    // Test zero
    WMATH_TYPE(Vec3) v4 = WMATH_ZERO(Vec3)();
    if (v4.v[0] != 0.0f || v4.v[1] != 0.0f || v4.v[2] != 0.0f) {
        printf("Vec3 zero test failed\n");
        return false;
    }



    // Test ceil
    WMATH_TYPE(Vec3) v5 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.3f, .v_y = 2.7f, .v_z = 3.2f});
    WMATH_TYPE(Vec3) v6 = WMATH_CEIL(Vec3)(v5);
    if (v6.v[0] != ceilf(1.3f) || v6.v[1] != ceilf(2.7f) || v6.v[2] != ceilf(3.2f)) {
        printf("Vec3 ceil test failed\n");
        return false;
    }

    // Test floor
    WMATH_TYPE(Vec3) v7 = WMATH_FLOOR(Vec3)(v5);
    if (v7.v[0] != floorf(1.3f) || v7.v[1] != floorf(2.7f) || v7.v[2] != floorf(3.2f)) {
        printf("Vec3 floor test failed\n");
        return false;
    }

    // Test round
    WMATH_TYPE(Vec3) v8 = WMATH_ROUND(Vec3)(v5);
    if (v8.v[0] != roundf(1.3f) || v8.v[1] != roundf(2.7f) || v8.v[2] != roundf(3.2f)) {
        printf("Vec3 round test failed\n");
        return false;
    }

    // Test clamp
    WMATH_TYPE(Vec3) v9 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = -2.0f, .v_y = 5.0f, .v_z = 8.0f});
    WMATH_TYPE(Vec3) v10 = WMATH_CLAMP(Vec3)(v9, 0.0f, 6.0f);
    if (v10.v[0] != 0.0f || v10.v[1] != 5.0f || v10.v[2] != 6.0f) {
        printf("Vec3 clamp test failed\n");
        return false;
    }

    // Test add
    WMATH_TYPE(Vec3) v11 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Vec3) v12 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 4.0f, .v_y = 5.0f, .v_z = 6.0f});
    WMATH_TYPE(Vec3) v13 = WMATH_ADD(Vec3)(v11, v12);
    if (v13.v[0] != 5.0f || v13.v[1] != 7.0f || v13.v[2] != 9.0f) {
        printf("Vec3 add test failed\n");
        return false;
    }

    // Test addScaled
    WMATH_TYPE(Vec3) v14 = WMATH_ADD_SCALED(Vec3)(v11, v12, 2.0f);
    if (v14.v[0] != 9.0f || v14.v[1] != 12.0f || v14.v[2] != 15.0f) {
        printf("Vec3 addScaled test failed\n");
        return false;
    }

    // Test subtract
    WMATH_TYPE(Vec3) v15 = WMATH_SUB(Vec3)(v12, v11);
    if (v15.v[0] != 3.0f || v15.v[1] != 3.0f || v15.v[2] != 3.0f) {
        printf("Vec3 subtract test failed\n");
        return false;
    }

    // Test equalsApproximately
    WMATH_TYPE(Vec3) v16 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Vec3) v17 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f + wcn_math_get_epsilon()/2, .v_y = 2.0f, .v_z = 3.0f});
    if (!WMATH_EQUALS_APPROXIMATELY(Vec3)(v16, v17)) {
        printf("Vec3 equalsApproximately test failed\n");
        return false;
    }

    // Test equals
    WMATH_TYPE(Vec3) v18 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Vec3) v19 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    if (!WMATH_EQUALS(Vec3)(v18, v19)) {
        printf("Vec3 equals test failed\n");
        return false;
    }

    // Test lerp
    WMATH_TYPE(Vec3) v20 = WMATH_LERP(Vec3)(v11, v12, 0.5f);
    if (v20.v[0] != 2.5f || v20.v[1] != 3.5f || v20.v[2] != 4.5f) {
        printf("Vec3 lerp test failed\n");
        return false;
    }

    // Test lerpV
    WMATH_TYPE(Vec3) v21 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.5f, .v_y = 0.5f, .v_z = 0.5f});
    WMATH_TYPE(Vec3) v22 = WMATH_LERP_V(Vec3)(v11, v12, v21);
    if (v22.v[0] != 2.5f || v22.v[1] != 3.5f || v22.v[2] != 4.5f) {
        printf("Vec3 lerpV test failed\n");
        return false;
    }

    // Test fmax
    WMATH_TYPE(Vec3) v23 = WMATH_FMAX(Vec3)(v11, v12);
    if (v23.v[0] != 4.0f || v23.v[1] != 5.0f || v23.v[2] != 6.0f) {
        printf("Vec3 fmax test failed\n");
        return false;
    }

    // Test fmin
    WMATH_TYPE(Vec3) v24 = WMATH_FMIN(Vec3)(v11, v12);
    if (v24.v[0] != 1.0f || v24.v[1] != 2.0f || v24.v[2] != 3.0f) {
        printf("Vec3 fmin test failed\n");
        return false;
    }

    // Test multiply
    WMATH_TYPE(Vec3) v25 = WMATH_MULTIPLY(Vec3)(v11, v12);
    if (v25.v[0] != 4.0f || v25.v[1] != 10.0f || v25.v[2] != 18.0f) {
        printf("Vec3 multiply test failed\n");
        return false;
    }

    // Test multiplyScalar
    WMATH_TYPE(Vec3) v26 = WMATH_MULTIPLY_SCALAR(Vec3)(v11, 2.0f);
    if (v26.v[0] != 2.0f || v26.v[1] != 4.0f || v26.v[2] != 6.0f) {
        printf("Vec3 multiplyScalar test failed\n");
        return false;
    }

    // Test div
    WMATH_TYPE(Vec3) v27 = WMATH_DIV(Vec3)(v12, v11);
    if (fabsf(v27.v[0] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(v27.v[1] - 2.5f) > wcn_math_get_epsilon() ||
        fabsf(v27.v[2] - 2.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 div test failed\n");
        return false;
    }

    // Test divScalar
    WMATH_TYPE(Vec3) v28 = WMATH_DIV_SCALAR(Vec3)(v12, 2.0f);
    if (fabsf(v28.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(v28.v[1] - 2.5f) > wcn_math_get_epsilon() ||
        fabsf(v28.v[2] - 3.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 divScalar test failed\n");
        return false;
    }

    // Test inverse
    WMATH_TYPE(Vec3) v29 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 4.0f, .v_z = 8.0f});
    WMATH_TYPE(Vec3) v30 = WMATH_INVERSE(Vec3)(v29);
    if (fabsf(v30.v[0] - 0.5f) > wcn_math_get_epsilon() ||
        fabsf(v30.v[1] - 0.25f) > wcn_math_get_epsilon() ||
        fabsf(v30.v[2] - 0.125f) > wcn_math_get_epsilon()) {
        printf("Vec3 inverse test failed\n");
        return false;
    }

    // Test dot
    float dot = WMATH_DOT(Vec3)(v11, v12);
    if (fabsf(dot - 32.0f) > wcn_math_get_epsilon()) { // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        printf("Vec3 dot test failed: %f\n", dot);
        return false;
    }

    // Test cross
    WMATH_TYPE(Vec3) v31 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v32 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v33 = WMATH_CROSS(Vec3)(v31, v32);
    if (fabsf(v33.v[0] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v33.v[1] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v33.v[2] - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 cross test failed\n");
        return false;
    }

    // Test length
    WMATH_TYPE(Vec3) v34 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    float length = WMATH_LENGTH(Vec3)(v34);
    // 使用相对误差检查，因为快速平方根可能有轻微的精度损失
    float relError1 = fabsf(length - sqrtf(14.0f)) / sqrtf(14.0f);
    if (relError1 > 1e-3f) {
        printf("Vec3 length test failed: %f, expected: %f, relative error: %f\n", length, sqrtf(14.0f), relError1);
        return false;
    }

    // Test lengthSq
    float lengthSq = WMATH_LENGTH_SQ(Vec3)(v34);
    if (fabsf(lengthSq - 14.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 lengthSq test failed: %f\n", lengthSq);
        return false;
    }

    // Test distance
    float distance = WMATH_DISTANCE(Vec3)(v11, v12);
    if (fabsf(distance - sqrtf(27.0f)) > wcn_math_get_epsilon()) {
        printf("Vec3 distance test failed: %f\n", distance);
        return false;
    }

    // Test distanceSq
    float distanceSq = WMATH_DISTANCE_SQ(Vec3)(v11, v12);
    if (fabsf(distanceSq - 27.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 distanceSq test failed: %f\n", distanceSq);
        return false;
    }

    // Test normalize
    WMATH_TYPE(Vec3) v35 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f});
    WMATH_TYPE(Vec3) v36 = WMATH_NORMALIZE(Vec3)(v35);
    float len = sqrtf(3.0f*3.0f + 4.0f*4.0f + 5.0f*5.0f);
    if (fabsf(v36.v[0] - 3.0f/len) > wcn_math_get_epsilon() ||
        fabsf(v36.v[1] - 4.0f/len) > wcn_math_get_epsilon() ||
        fabsf(v36.v[2] - 5.0f/len) > wcn_math_get_epsilon()) {
        printf("Vec3 normalize test failed\n");
        return false;
    }

    // Test negate
    WMATH_TYPE(Vec3) v37 = WMATH_NEGATE(Vec3)(v11);
    if (v37.v[0] != -1.0f || v37.v[1] != -2.0f || v37.v[2] != -3.0f) {
        printf("Vec3 negate test failed\n");
        return false;
    }

    // Test random
    WMATH_TYPE(Vec3) v38 = WMATH_RANDOM(Vec3)(1.0f);
    float randomLength = WMATH_LENGTH(Vec3)(v38);
    if (fabsf(randomLength - 1.0f) > 0.01f) { // Use larger epsilon for random test
        printf("Vec3 random test failed, length: %f\n", randomLength);
        return false;
    }

    // Test setLength
    WMATH_TYPE(Vec3) v39 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f});
    WMATH_TYPE(Vec3) v40 = WMATH_SET_LENGTH(Vec3)(v39, 10.0f);
    float newLength = WMATH_LENGTH(Vec3)(v40);
    // 使用相对误差检查，因为快速平方根可能有轻微的精度损失
    float relError2 = fabsf(newLength - 10.0f) / 10.0f;
    if (relError2 > 1e-2f) {  // 放宽到1%的误差容忍度
        printf("Vec3 setLength test failed: length = %f, relative error: %f\n", newLength, relError2);
        return false;
    }

    // Test truncate
    WMATH_TYPE(Vec3) v41 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f});
    WMATH_TYPE(Vec3) v42 = WMATH_TRUNCATE(Vec3)(v41, 5.0f);
    float truncLength = WMATH_LENGTH(Vec3)(v42);
    if (fabsf(truncLength - 5.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 truncate test failed\n");
        return false;
    }

    // Test midpoint
    WMATH_TYPE(Vec3) v43 = WMATH_MIDPOINT(Vec3)(v11, v12);
    if (fabsf(v43.v[0] - 2.5f) > wcn_math_get_epsilon() ||
        fabsf(v43.v[1] - 3.5f) > wcn_math_get_epsilon() ||
        fabsf(v43.v[2] - 4.5f) > wcn_math_get_epsilon()) {
        printf("Vec3 midpoint test failed\n");
        return false;
    }

    // Test angle
    WMATH_TYPE(Vec3) v44 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v45 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    float angle = WMATH_ANGLE(Vec3)(v44, v45);
    if (fabsf(angle - WMATH_PI_2) > wcn_math_get_epsilon()) {
        printf("Vec3 angle test failed: %f\n", angle);
        return false;
    }

    // Test rotate_x
    WMATH_TYPE(Vec3) v46 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v47 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v48 = WMATH_ROTATE_X(Vec3)(v46, v47, WMATH_PI_2);
    if (fabsf(v48.v[0] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v48.v[1] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v48.v[2] - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 rotate_x test failed\n");
        return false;
    }

    // Test rotate_y
    WMATH_TYPE(Vec3) v49 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v50 = WMATH_ROTATE_Y(Vec3)(v49, v47, WMATH_PI_2);
    if (fabsf(v50.v[0] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v50.v[1] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v50.v[2] + 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 rotate_y test failed\n");
        return false;
    }

    // Test rotate_z
    WMATH_TYPE(Vec3) v51 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) v52 = WMATH_ROTATE_Z(Vec3)(v51, v47, WMATH_PI_2);
    if (fabsf(v52.v[0] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(v52.v[1] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(v52.v[2] - 0.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 rotate_z test failed\n");
        return false;
    }

    printf("All Vec3 tests passed!\n");
    return true;
}

bool TestVec4() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-6f);

    // Test create
    WMATH_TYPE(Vec4) v1 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (v1.v[0] != 1.0f || v1.v[1] != 2.0f || v1.v[2] != 3.0f || v1.v[3] != 4.0f) {
        printf("Vec4 create test failed\n");
        return false;
    }

    // Test set
    WMATH_TYPE(Vec4) v2 = WMATH_SET(Vec4)(v1, 5.0f, 6.0f, 7.0f, 8.0f);
    if (v2.v[0] != 5.0f || v2.v[1] != 6.0f || v2.v[2] != 7.0f || v2.v[3] != 8.0f) {
        printf("Vec4 set test failed\n");
        return false;
    }

    // Test copy
    WMATH_TYPE(Vec4) v3 = WMATH_COPY(Vec4)(v1);
    if (v3.v[0] != 1.0f || v3.v[1] != 2.0f || v3.v[2] != 3.0f || v3.v[3] != 4.0f) {
        printf("Vec4 copy test failed\n");
        return false;
    }

    // Test zero
    WMATH_TYPE(Vec4) v4 = WMATH_ZERO(Vec4)();
    if (v4.v[0] != 0.0f || v4.v[1] != 0.0f || v4.v[2] != 0.0f || v4.v[3] != 0.0f) {
        printf("Vec4 zero test failed\n");
        return false;
    }

    // Test identity
    WMATH_TYPE(Vec4) v5 = WMATH_IDENTITY(Vec4)();
    if (v5.v[0] != 0.0f || v5.v[1] != 0.0f || v5.v[2] != 0.0f || v5.v[3] != 1.0f) {
        printf("Vec4 identity test failed\n");
        return false;
    }

    // Test ceil
    const WMATH_TYPE(Vec4) v6 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.3f, .v_y = 2.7f, .v_z = 3.2f, .v_w = 4.8f});
    const WMATH_TYPE(Vec4) v7 = WMATH_CEIL(Vec4)(v6);
    if (v7.v[0] != ceilf(1.3f) || v7.v[1] != ceilf(2.7f) || v7.v[2] != ceilf(3.2f) || v7.v[3] != ceilf(4.8f)) {
        printf("Vec4 ceil test failed\n");
        return false;
    }

    // Test floor
    WMATH_TYPE(Vec4) v8 = WMATH_FLOOR(Vec4)(v6);
    if (v8.v[0] != floorf(1.3f) || v8.v[1] != floorf(2.7f) || v8.v[2] != floorf(3.2f) || v8.v[3] != floorf(4.8f)) {
        printf("Vec4 floor test failed\n");
        return false;
    }

    // Test round
    WMATH_TYPE(Vec4) v9 = WMATH_ROUND(Vec4)(v6);
    if (v9.v[0] != roundf(1.3f) || v9.v[1] != roundf(2.7f) || v9.v[2] != roundf(3.2f) || v9.v[3] != roundf(4.8f)) {
        printf("Vec4 round test failed\n");
        return false;
    }

    // Test clamp
    WMATH_TYPE(Vec4) v10 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = -2.0f, .v_y = 5.0f, .v_z = 8.0f, .v_w = 12.0f});
    WMATH_TYPE(Vec4) v11 = WMATH_CLAMP(Vec4)(v10, 0.0f, 10.0f);
    if (v11.v[0] != 0.0f || v11.v[1] != 5.0f || v11.v[2] != 8.0f || v11.v[3] != 10.0f) {
        printf("Vec4 clamp test failed\n");
        return false;
    }

    // Test add
    WMATH_TYPE(Vec4) v12 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Vec4) v13 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 5.0f, .v_y = 6.0f, .v_z = 7.0f, .v_w = 8.0f});
    WMATH_TYPE(Vec4) v14 = WMATH_ADD(Vec4)(v12, v13);
    if (v14.v[0] != 6.0f || v14.v[1] != 8.0f || v14.v[2] != 10.0f || v14.v[3] != 12.0f) {
        printf("Vec4 add test failed\n");
        return false;
    }

    // Test addScaled
    WMATH_TYPE(Vec4) v15 = WMATH_ADD_SCALED(Vec4)(v12, v13, 2.0f);
    if (v15.v[0] != 11.0f || v15.v[1] != 14.0f || v15.v[2] != 17.0f || v15.v[3] != 20.0f) {
        printf("Vec4 addScaled test failed\n");
        return false;
    }

    // Test subtract
    WMATH_TYPE(Vec4) v16 = WMATH_SUB(Vec4)(v13, v12);
    if (v16.v[0] != 4.0f || v16.v[1] != 4.0f || v16.v[2] != 4.0f || v16.v[3] != 4.0f) {
        printf("Vec4 subtract test failed\n");
        return false;
    }

    // Test equalsApproximately
    WMATH_TYPE(Vec4) v17 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Vec4) v18 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f + wcn_math_get_epsilon()/2, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (!WMATH_EQUALS_APPROXIMATELY(Vec4)(v17, v18)) {
        printf("Vec4 equalsApproximately test failed\n");
        return false;
    }

    // Test equals
    WMATH_TYPE(Vec4) v19 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Vec4) v20 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (!WMATH_EQUALS(Vec4)(v19, v20)) {
        printf("Vec4 equals test failed\n");
        return false;
    }

    // Test lerp
    WMATH_TYPE(Vec4) v21 = WMATH_LERP(Vec4)(v12, v13, 0.5f);
    if (fabsf(v21.v[0] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v21.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(v21.v[2] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(v21.v[3] - 6.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 lerp test failed\n");
        return false;
    }

    // Test lerpV
    WMATH_TYPE(Vec4) v22 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 0.5f, .v_y = 0.5f, .v_z = 0.5f, .v_w = 0.5f});
    WMATH_TYPE(Vec4) v23 = WMATH_LERP_V(Vec4)(v12, v13, v22);
    if (fabsf(v23.v[0] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v23.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(v23.v[2] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(v23.v[3] - 6.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 lerpV test failed\n");
        return false;
    }

    // Test fmax
    WMATH_TYPE(Vec4) v24 = WMATH_FMAX(Vec4)(v12, v13);
    if (v24.v[0] != 5.0f || v24.v[1] != 6.0f || v24.v[2] != 7.0f || v24.v[3] != 8.0f) {
        printf("Vec4 fmax test failed\n");
        return false;
    }

    // Test fmin
    WMATH_TYPE(Vec4) v25 = WMATH_FMIN(Vec4)(v12, v13);
    if (v25.v[0] != 1.0f || v25.v[1] != 2.0f || v25.v[2] != 3.0f || v25.v[3] != 4.0f) {
        printf("Vec4 fmin test failed\n");
        return false;
    }

    // Test multiply
    WMATH_TYPE(Vec4) v26 = WMATH_MULTIPLY(Vec4)(v12, v13);
    if (v26.v[0] != 5.0f || v26.v[1] != 12.0f || v26.v[2] != 21.0f || v26.v[3] != 32.0f) {
        printf("Vec4 multiply test failed\n");
        return false;
    }

    // Test multiplyScalar
    WMATH_TYPE(Vec4) v27 = WMATH_MULTIPLY_SCALAR(Vec4)(v12, 2.0f);
    if (v27.v[0] != 2.0f || v27.v[1] != 4.0f || v27.v[2] != 6.0f || v27.v[3] != 8.0f) {
        printf("Vec4 multiplyScalar test failed\n");
        return false;
    }

    // Test div
    WMATH_TYPE(Vec4) v28 = WMATH_DIV(Vec4)(v13, v12);
    if (fabsf(v28.v[0] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(v28.v[1] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v28.v[2] - 7.0f/3.0f) > wcn_math_get_epsilon() ||
        fabsf(v28.v[3] - 2.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 div test failed\n");
        return false;
    }

    // Test divScalar
    WMATH_TYPE(Vec4) v29 = WMATH_DIV_SCALAR(Vec4)(v13, 2.0f);
    if (fabsf(v29.v[0] - 2.5f) > wcn_math_get_epsilon() ||
        fabsf(v29.v[1] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v29.v[2] - 3.5f) > wcn_math_get_epsilon() ||
        fabsf(v29.v[3] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 divScalar test failed\n");
        return false;
    }

    // Test inverse
    WMATH_TYPE(Vec4) v30 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 2.0f, .v_y = 4.0f, .v_z = 8.0f, .v_w = 16.0f});
    WMATH_TYPE(Vec4) v31 = WMATH_INVERSE(Vec4)(v30);
    if (fabsf(v31.v[0] - 0.5f) > wcn_math_get_epsilon() ||
        fabsf(v31.v[1] - 0.25f) > wcn_math_get_epsilon() ||
        fabsf(v31.v[2] - 0.125f) > wcn_math_get_epsilon() ||
        fabsf(v31.v[3] - 0.0625f) > wcn_math_get_epsilon()) {
        printf("Vec4 inverse test failed\n");
        return false;
    }

    // Test dot
    float dot = WMATH_DOT(Vec4)(v12, v13);
    if (fabsf(dot - 70.0f) > wcn_math_get_epsilon()) { // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        printf("Vec4 dot test failed: %f\n", dot);
        return false;
    }

    // Test lengthSq
    float lengthSq = WMATH_LENGTH_SQ(Vec4)(v12);
    if (fabsf(lengthSq - 30.0f) > wcn_math_get_epsilon()) { // 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        printf("Vec4 lengthSq test failed: %f\n", lengthSq);
        return false;
    }

    // Test length
    float length = WMATH_LENGTH(Vec4)(v12);
    if (fabsf(length - sqrtf(30.0f)) > wcn_math_get_epsilon()) {
        printf("Vec4 length test failed: %f\n", length);
        return false;
    }

    // Test distanceSq
    float distanceSq = WMATH_DISTANCE_SQ(Vec4)(v12, v13);
    if (fabsf(distanceSq - 64.0f) > wcn_math_get_epsilon()) { // (5-1)^2 + (6-2)^2 + (7-3)^2 + (8-4)^2 = 16 + 16 + 16 + 16 = 64
        printf("Vec4 distanceSq test failed: %f\n", distanceSq);
        return false;
    }

    // Test distance
    float distance = WMATH_DISTANCE(Vec4)(v12, v13);
    if (fabsf(distance - 8.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 distance test failed: %f\n", distance);
        return false;
    }

    // Test normalize
    WMATH_TYPE(Vec4) v32 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f, .v_w = 6.0f});
    WMATH_TYPE(Vec4) v33 = WMATH_NORMALIZE(Vec4)(v32);
    float len = sqrtf(3.0f*3.0f + 4.0f*4.0f + 5.0f*5.0f + 6.0f*6.0f);
    if (fabsf(v33.v[0] - 3.0f/len) > wcn_math_get_epsilon() ||
        fabsf(v33.v[1] - 4.0f/len) > wcn_math_get_epsilon() ||
        fabsf(v33.v[2] - 5.0f/len) > wcn_math_get_epsilon() ||
        fabsf(v33.v[3] - 6.0f/len) > wcn_math_get_epsilon()) {
        printf("Vec4 normalize test failed\n");
        return false;
    }

    // Test negate
    WMATH_TYPE(Vec4) v34 = WMATH_NEGATE(Vec4)(v12);
    if (v34.v[0] != -1.0f || v34.v[1] != -2.0f || v34.v[2] != -3.0f || v34.v[3] != -4.0f) {
        printf("Vec4 negate test failed\n");
        return false;
    }

    // Test setLength
    WMATH_TYPE(Vec4) v35 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f, .v_w = 6.0f});
    WMATH_TYPE(Vec4) v36 = WMATH_SET_LENGTH(Vec4)(v35, 10.0f);
    float newLength = WMATH_LENGTH(Vec4)(v36);
    if (fabsf(newLength - 10.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 setLength test failed\n");
        return false;
    }

    // Test truncate
    WMATH_TYPE(Vec4) v37 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 5.0f, .v_w = 6.0f});
    WMATH_TYPE(Vec4) v38 = WMATH_TRUNCATE(Vec4)(v37, 5.0f);
    float truncLength = WMATH_LENGTH(Vec4)(v38);
    if (fabsf(truncLength - 5.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 truncate test failed\n");
        return false;
    }

    // Test midpoint
    WMATH_TYPE(Vec4) v39 = WMATH_MIDPOINT(Vec4)(v12, v13);
    if (fabsf(v39.v[0] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v39.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(v39.v[2] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(v39.v[3] - 6.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 midpoint test failed\n");
        return false;
    }



    printf("All Vec4 tests passed!\n");
    return true;
}

bool TestQuat() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-6f);

    // Test zero
    WMATH_TYPE(Quat) q0 = WMATH_ZERO(Quat)();
    if (q0.v[0] != 0.0f || q0.v[1] != 0.0f || q0.v[2] != 0.0f || q0.v[3] != 0.0f) {
        printf("Quat zero test failed\n");
        return false;
    }

    // Test identity
    WMATH_TYPE(Quat) q1 = WMATH_IDENTITY(Quat)();
    if (q1.v[0] != 0.0f || q1.v[1] != 0.0f || q1.v[2] != 0.0f || q1.v[3] != 1.0f) {
        printf("Quat identity test failed\n");
        return false;
    }

    // Test create
    WMATH_TYPE(Quat) q2 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (q2.v[0] != 1.0f || q2.v[1] != 2.0f || q2.v[2] != 3.0f || q2.v[3] != 4.0f) {
        printf("Quat create test failed\n");
        return false;
    }

    // Test set
    WMATH_TYPE(Quat) q3 = WMATH_SET(Quat)(q2, 5.0f, 6.0f, 7.0f, 8.0f);
    if (q3.v[0] != 5.0f || q3.v[1] != 6.0f || q3.v[2] != 7.0f || q3.v[3] != 8.0f) {
        printf("Quat set test failed\n");
        return false;
    }

    // Test copy
    WMATH_TYPE(Quat) q4 = WMATH_COPY(Quat)(q2);
    if (q4.v[0] != 1.0f || q4.v[1] != 2.0f || q4.v[2] != 3.0f || q4.v[3] != 4.0f) {
        printf("Quat copy test failed\n");
        return false;
    }

    // Test dot
    float dot = WMATH_DOT(Quat)(q2, q3);
    float expectedDot = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f + 4.0f*8.0f; // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    if (fabsf(dot - expectedDot) > wcn_math_get_epsilon()) {
        printf("Quat dot test failed: %f, expected: %f\n", dot, expectedDot);
        return false;
    }

    // Test lengthSq
    float lengthSq = WMATH_LENGTH_SQ(Quat)(q2);
    float expectedLengthSq = 1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f; // 1 + 4 + 9 + 16 = 30
    if (fabsf(lengthSq - expectedLengthSq) > wcn_math_get_epsilon()) {
        printf("Quat lengthSq test failed: %f, expected: %f\n", lengthSq, expectedLengthSq);
        return false;
    }

    // Test length
    float length = WMATH_LENGTH(Quat)(q2);
    float expectedLength = sqrtf(expectedLengthSq);
    if (fabsf(length - expectedLength) > wcn_math_get_epsilon()) {
        printf("Quat length test failed: %f, expected: %f\n", length, expectedLength);
        return false;
    }

    // Test normalize
    WMATH_TYPE(Quat) q5 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 3.0f, .v_y = 4.0f, .v_z = 0.0f, .v_w = 0.0f});
    WMATH_TYPE(Quat) q6 = WMATH_NORMALIZE(Quat)(q5);
    float len5 = WMATH_LENGTH(Quat)(q6);
    if (fabsf(len5 - 1.0f) > wcn_math_get_epsilon()) {
        printf("Quat normalize test failed: length = %f\n", len5);
        return false;
    }

    // Test equals
    WMATH_TYPE(Quat) q7 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Quat) q8 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (!WMATH_EQUALS(Quat)(q7, q8)) {
        printf("Quat equals test failed\n");
        return false;
    }

    // Test equalsApproximately
    WMATH_TYPE(Quat) q9 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Quat) q10 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f + wcn_math_get_epsilon()/2, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    if (!WMATH_EQUALS_APPROXIMATELY(Quat)(q9, q10)) {
        printf("Quat equalsApproximately test failed\n");
        return false;
    }

    // Test add
    WMATH_TYPE(Quat) q11 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Quat) q12 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 5.0f, .v_y = 6.0f, .v_z = 7.0f, .v_w = 8.0f});
    WMATH_TYPE(Quat) q13 = WMATH_ADD(Quat)(q11, q12);
    if (fabsf(q13.v[0] - 6.0f) > wcn_math_get_epsilon() ||
        fabsf(q13.v[1] - 8.0f) > wcn_math_get_epsilon() ||
        fabsf(q13.v[2] - 10.0f) > wcn_math_get_epsilon() ||
        fabsf(q13.v[3] - 12.0f) > wcn_math_get_epsilon()) {
        printf("Quat add test failed\n");
        return false;
    }

    // Test sub
    WMATH_TYPE(Quat) q14 = WMATH_SUB(Quat)(q12, q11);
    if (fabsf(q14.v[0] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q14.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q14.v[2] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q14.v[3] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Quat sub test failed\n");
        return false;
    }

    // Test multiplyScalar
    WMATH_TYPE(Quat) q15 = WMATH_MULTIPLY_SCALAR(Quat)(q11, 2.0f);
    if (fabsf(q15.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(q15.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q15.v[2] - 6.0f) > wcn_math_get_epsilon() ||
        fabsf(q15.v[3] - 8.0f) > wcn_math_get_epsilon()) {
        printf("Quat multiplyScalar test failed\n");
        return false;
    }

    // Test divScalar
    WMATH_TYPE(Quat) q16 = WMATH_DIV_SCALAR(Quat)(q12, 2.0f);
    if (fabsf(q16.v[0] - 2.5f) > wcn_math_get_epsilon() ||
        fabsf(q16.v[1] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(q16.v[2] - 3.5f) > wcn_math_get_epsilon() ||
        fabsf(q16.v[3] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Quat divScalar test failed\n");
        return false;
    }

    // Test conjugate
    WMATH_TYPE(Quat) q17 = WMATH_CALL(Quat, conjugate)(q11);
    if (fabsf(q17.v[0] + 1.0f) > wcn_math_get_epsilon() ||
        fabsf(q17.v[1] + 2.0f) > wcn_math_get_epsilon() ||
        fabsf(q17.v[2] + 3.0f) > wcn_math_get_epsilon() ||
        fabsf(q17.v[3] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Quat conjugate test failed\n");
        return false;
    }

    // Test inverse
    WMATH_TYPE(Quat) q18 = WMATH_INVERSE(Quat)(q11);
    WMATH_TYPE(Quat) q19 = WMATH_MULTIPLY(Quat)(q11, q18);
    // The result should be close to identity quaternion [0, 0, 0, 1]
    if (fabsf(q19.v[0]) > 1e-4f ||
        fabsf(q19.v[1]) > 1e-4f ||
        fabsf(q19.v[2]) > 1e-4f ||
        fabsf(q19.v[3] - 1.0f) > 1e-4f) {
        printf("Quat inverse test failed: [%f, %f, %f, %f]\n", q19.v[0], q19.v[1], q19.v[2], q19.v[3]);
        printf("q11: [%f, %f, %f, %f]\n", q11.v[0], q11.v[1], q11.v[2], q11.v[3]);
        printf("q18: [%f, %f, %f, %f]\n", q18.v[0], q18.v[1], q18.v[2], q18.v[3]);
        // Manual calculation:
        // q11 = [1, 2, 3, 4]
        // length squared = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        // inverse = [-1/30, -2/30, -3/30, 4/30] = [-0.0333, -0.0667, -0.1, 0.1333]
        // multiply result should be [0, 0, 0, 1]
        return false;
    }

    // Test multiply
    const WMATH_TYPE(Quat) q20 = WMATH_MULTIPLY(Quat)(q11, q12);
    // Manual calculation:
    // q11 = [1, 2, 3, 4]
    // q12 = [5, 6, 7, 8]
    // Result should be:
    // x = 1*8 + 4*5 + 2*7 - 3*6 = 8 + 20 + 14 - 18 = 24
    // y = 2*8 + 4*6 + 3*5 - 1*7 = 16 + 24 + 15 - 7 = 48
    // z = 3*8 + 4*7 + 1*6 - 2*5 = 24 + 28 + 6 - 10 = 48
    // w = 4*8 - 1*5 - 2*6 - 3*7 = 32 - 5 - 12 - 21 = -6
    if (fabsf(q20.v[0] - 24.0f) > wcn_math_get_epsilon() ||
        fabsf(q20.v[1] - 48.0f) > wcn_math_get_epsilon() ||
        fabsf(q20.v[2] - 48.0f) > wcn_math_get_epsilon() ||
        fabsf(q20.v[3] + 6.0f) > wcn_math_get_epsilon()) {
        printf("Quat multiply test failed\n");
        return false;
    }

    // Test lerp
    WMATH_TYPE(Quat) q21 = WMATH_LERP(Quat)(q11, q12, 0.5f);
    if (fabsf(q21.v[0] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(q21.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q21.v[2] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(q21.v[3] - 6.0f) > wcn_math_get_epsilon()) {
        printf("Quat lerp test failed\n");
        return false;
    }

    // Test slerp
    WMATH_TYPE(Quat) q22 = WMATH_IDENTITY(Quat)();
    WMATH_TYPE(Quat) q23 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 1.0f, .v_w = 0.0f});
    WMATH_TYPE(Quat) q24 = WMATH_CALL(Quat, slerp)(q22, q23, 0.5f);
    // Should be close to 45 degree rotation around Z axis
    if (fabsf(WMATH_LENGTH(Quat)(q24) - 1.0f) > wcn_math_get_epsilon()) {
        printf("Quat slerp test failed - not normalized\n");
        return false;
    }

    // Test from_axis_angle and to_axis_angle
    WMATH_TYPE(Vec3) axis = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 1.0f});
    float angle = WMATH_PI_2;
    WMATH_TYPE(Quat) q25 = WMATH_CALL(Quat, from_axis_angle)(axis, angle);
    WCN_Math_Vec3_WithAngleAxis result = WMATH_CALL(Quat, to_axis_angle)(q25);
    if (fabsf(result.angle - angle) > wcn_math_get_epsilon() ||
        fabsf(WMATH_DOT(Vec3)(result.axis, axis) - 1.0f) > wcn_math_get_epsilon()) {
        printf("Quat from/to axis_angle test failed\n");
        return false;
    }

    // Test rotation_to
    const WMATH_TYPE(Vec3) from = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    const WMATH_TYPE(Vec3) to = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    const WMATH_TYPE(Quat) q26 = WMATH_CALL(Quat, rotation_to)(from, to);
    // printf("q26: [%f, %f, %f, %f]\n", q26.v[0], q26.v[1], q26.v[2], q26.v[3]);
    const WMATH_TYPE(Vec3) rotated = WMATH_CALL(Vec3, transform_quat)(from, q26);
    // printf("rotated: [%f, %f, %f]\n", rotated.v[0], rotated.v[1], rotated.v[2]);
    const float ep = fabsf(WMATH_DOT(Vec3)(rotated, to) - 1.0f);
    // printf("Q~t rotation_to test ep %f \n", ep);
    if (ep > 1e-4f) {
        printf("Quat rotation_to test failed\n");
        return false;
    }

    // Test angle
    float quatAngle = WMATH_ANGLE(Quat)(q11, q12);
    if (quatAngle < 0.0f) {
        printf("Quat angle test failed - negative angle\n");
        return false;
    }

    // Test rotate_x, rotate_y, rotate_z
    WMATH_TYPE(Quat) q27 = WMATH_IDENTITY(Quat)();
    WMATH_TYPE(Quat) q28 = WMATH_ROTATE_X(Quat)(q27, WMATH_PI_2);
    WMATH_TYPE(Quat) q29 = WMATH_ROTATE_Y(Quat)(q27, WMATH_PI_2);
    WMATH_TYPE(Quat) q30 = WMATH_ROTATE_Z(Quat)(q27, WMATH_PI_2);

    if (fabsf(WMATH_LENGTH(Quat)(q28) - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(WMATH_LENGTH(Quat)(q29) - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(WMATH_LENGTH(Quat)(q30) - 1.0f) > wcn_math_get_epsilon()) {
        printf("Quat rotate x/y/z test failed\n");
        return false;
    }

    // Test from_euler
    WMATH_TYPE(Quat) q31 = WMATH_CALL(Quat, from_euler)(WMATH_PI_2, 0.0f, 0.0f, WCN_Math_RotationOrder_XYZ);
    if (fabsf(WMATH_LENGTH(Quat)(q31) - 1.0f) > wcn_math_get_epsilon()) {
        printf("Quat from_euler test failed\n");
        return false;
    }

    printf("All Quat tests passed!\n");
    return true;
}

bool TestMat3() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-5f);

    // Test zero
    WMATH_TYPE(Mat3) m0 = WMATH_ZERO(Mat3)();
    for (int i = 0; i < 12; i++) {
        if (i == 3 || i == 7 || i == 11) continue; // Skip padding elements
        if (fabsf(m0.m[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 zero test failed at index %d\n", i);
            return false;
        }
    }

    // Test identity
    WMATH_TYPE(Mat3) m1 = WMATH_IDENTITY(Mat3)();
    // 检查所有元素，包括padding元素应该为0
    for (int i = 0; i < 12; i++) {const float identity_expected[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // 第一行: 1 0 0 0 (最后一个0是padding)
        0.0f, 1.0f, 0.0f, 0.0f,  // 第二行: 0 1 0 0 (最后一个0是padding)
        0.0f, 0.0f, 1.0f, 0.0f   // 第三行: 0 0 1 0 (最后一个0是padding)
    };
        if (fabsf(m1.m[i] - identity_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 identity test failed at index %d, expected %f, got %f\n",
                   i, identity_expected[i], m1.m[i]);
            return false;
        }
    }

    // Test create
    WMATH_TYPE(Mat3) m2 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 1.0f, .m_01 = 2.0f, .m_02 = 3.0f,
        .m_10 = 4.0f, .m_11 = 5.0f, .m_12 = 6.0f,
        .m_20 = 7.0f, .m_21 = 8.0f, .m_22 = 9.0f
    });
    float create_expected[12] = {
        1.0f, 2.0f, 3.0f, 0.0f,  // 第一行: 1 2 3 0 (最后一个0是padding)
        4.0f, 5.0f, 6.0f, 0.0f,  // 第二行: 4 5 6 0 (最后一个0是padding)
        7.0f, 8.0f, 9.0f, 0.0f   // 第三行: 7 8 9 0 (最后一个0是padding)
    };
    for (int i = 0; i < 12; i++) {
        if (fabsf(m2.m[i] - create_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 create test failed at index %d, expected %f, got %f\n",
                   i, create_expected[i], m2.m[i]);
            return false;
        }
    }

    // Test copy
    WMATH_TYPE(Mat3) m3 = WMATH_COPY(Mat3)(m2);
    for (int i = 0; i < 12; i++) {
        if (fabsf(m3.m[i] - create_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 copy test failed at index %d, expected %f, got %f\n",
                   i, create_expected[i], m3.m[i]);
            return false;
        }
    }

    // Test add
    const WMATH_TYPE(Mat3) m4 = WMATH_ADD(Mat3)(m2, m1);
    for (int i = 0; i < 12; i++) {const float add_expected[12] = {
        2.0f, 2.0f, 3.0f, 0.0f,  // 第一行: 2 2 3 0
        4.0f, 6.0f, 6.0f, 0.0f,  // 第二行: 4 6 6 0
        7.0f, 8.0f, 10.0f, 0.0f  // 第三行: 7 8 10 0
    };
        if (fabsf(m4.m[i] - add_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 add test failed at index %d, expected %f, got %f\n",
                   i, add_expected[i], m4.m[i]);
            printf("Matrix values: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
                   m4.m[0], m4.m[1], m4.m[2], m4.m[3],
                   m4.m[4], m4.m[5], m4.m[6], m4.m[7],
                   m4.m[8], m4.m[9], m4.m[10], m4.m[11]);
            return false;
        }
    }

    // Test multiply
    WMATH_TYPE(Mat3) m5 = WMATH_MULTIPLY(Mat3)(m2, m1);
    for (int i = 0; i < 12; i++) {
        if (fabsf(m5.m[i] - create_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 multiply test failed at index %d, expected %f, got %f\n",
                   i, create_expected[i], m5.m[i]);
            return false;
        }
    }

    // Test multiply with actual multiplication
    WMATH_TYPE(Mat3) m6 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f
    });
    const WMATH_TYPE(Mat3) m7 = WMATH_MULTIPLY(Mat3)(m2, m6);
    for (int i = 0; i < 12; i++) {const float multiply_expected[12] = {
        2.0f, 4.0f, 6.0f, 0.0f,   // 第一行: 2 4 6 0
        8.0f, 10.0f, 12.0f, 0.0f, // 第二行: 8 10 12 0
        14.0f, 16.0f, 18.0f, 0.0f // 第三行: 14 16 18 0
    };
        if (fabsf(m7.m[i] - multiply_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 multiply test 2 failed at index %d, expected %f, got %f\n",
                   i, multiply_expected[i], m7.m[i]);
            return false;
        }
    }

    // Test transpose
    WMATH_TYPE(Mat3) m8 = WMATH_CALL(Mat3, transpose)(m2);
    for (int i = 0; i < 12; i++) {const float transpose_expected[12] = {
        1.0f, 4.0f, 7.0f, 0.0f,  // 第一行: 1 4 7 0
        2.0f, 5.0f, 8.0f, 0.0f,  // 第二行: 2 5 8 0
        3.0f, 6.0f, 9.0f, 0.0f   // 第三行: 3 6 9 0
    };
        if (fabsf(m8.m[i] - transpose_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 transpose test failed at index %d, expected %f, got %f\n",
                   i, transpose_expected[i], m8.m[i]);
            return false;
        }
    }

    // Test determinant
    float det = WMATH_DETERMINANT(Mat3)(m2);
    // For matrix:
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    // Determinant = 1(5*9-6*8) - 2(4*9-6*7) + 3(4*8-5*7)
    //             = 1(45-48) - 2(36-42) + 3(32-35)
    //             = 1(-3) - 2(-6) + 3(-3)
    //             = -3 + 12 - 9 = 0
    if (fabsf(det - 0.0f) > wcn_math_get_epsilon()) {
        printf("Mat3 determinant test failed: %f\n", det);
        return false;
    }

    // Test determinant with non-singular matrix
    WMATH_TYPE(Mat3) m9 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 2.0f, .m_01 = 1.0f, .m_02 = 1.0f,
        .m_10 = 1.0f, .m_11 = 3.0f, .m_12 = 2.0f,
        .m_20 = 3.0f, .m_21 = 2.0f, .m_22 = 4.0f
    });
    float det2 = WMATH_DETERMINANT(Mat3)(m9);
    // Determinant = 2(3*4-2*2) - 1(1*4-2*3) + 1(1*2-3*3)
    //             = 2(12-4) - 1(4-6) + 1(2-9)
    //             = 2(8) - 1(-2) + 1(-7)
    //             = 16 + 2 - 7 = 11
    if (fabsf(det2 - 11.0f) > wcn_math_get_epsilon()) {
        printf("Mat3 determinant test 2 failed: %f\n", det2);
        return false;
    }

    // Test set
    WMATH_TYPE(Mat3) m10 = WMATH_SET(Mat3)(m1,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f);
    for (int i = 0; i < 12; i++) {
        const float set_expected[12] = {
        10.0f, 11.0f, 12.0f, 0.0f,
        13.0f, 14.0f, 15.0f, 0.0f,
        16.0f, 17.0f, 18.0f, 0.0f
        };
        if (fabsf(m10.m[i] - set_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat3 set test failed at index %d\n", i);
            return false;
        }
    }

    // Test negate
    WMATH_TYPE(Mat3) m11 = WMATH_NEGATE(Mat3)(m2);
    for (int i = 0; i < 12; i++) {
        const int expected = ((i + 1) % 4 == 0) ? 0 : -(i - i/4 + 1);
        if (fabsf(m11.m[i] - (float)expected) > wcn_math_get_epsilon()) {
            printf("Mat3 negate test failed at index %d\n", i);
            return false;
        }
    }

    // Test sub
    WMATH_TYPE(Mat3) m12 = WMATH_SUB(Mat3)(m2, m1);
    for (int i = 0; i < 12; i++) {
        int expected = (i % 4 == 3) ? 0 : (i - i/4 + 1);
        if (i % 4 != 3) {
            if (i == 0 || i == 5 || i == 10) expected -= 1; // Diagonal elements
        }
        if (fabsf(m12.m[i] - (float)expected) > wcn_math_get_epsilon()) {
            printf("Mat3 sub test failed at index %d\n", i);
            return false;
        }
    }

    // Test multiply_scalar
    WMATH_TYPE(Mat3) m13 = WMATH_MULTIPLY_SCALAR(Mat3)(m2, 2.0f);
    for (int i = 0; i < 12; i++) {
        const int expected = (i % 4 == 3) ? 0 : 2 * (i - i/4 + 1);
        if (fabsf(m13.m[i] - (float)expected) > wcn_math_get_epsilon()) {
            printf("Mat3 multiply_scalar test failed at index %d\n", i);
            return false;
        }
    }

    // Test inverse
    /**
    WMATH_TYPE(Mat3) m9 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 2.0f, .m_01 = 1.0f, .m_02 = 1.0f,
        .m_10 = 1.0f, .m_11 = 3.0f, .m_12 = 2.0f,
        .m_20 = 3.0f, .m_21 = 2.0f, .m_22 = 4.0f
    });
     */
    WMATH_TYPE(Mat3) m14 = WMATH_INVERSE(Mat3)(m9);
    WMATH_TYPE(Mat3) m15 = WMATH_MULTIPLY(Mat3)(m9, m14);
    // The Result should be close to identity
    for (int i = 0; i < 12; i++) {
        const float expected = (i % 4 == 3) ? 0.0f : ((i == 0 || i == 5 || i == 10) ? 1.0f : 0.0f);
        if (fabsf(m15.m[i] - expected) > 1e-4f) {
            printf("Mat3 inverse test failed at index %d\n", i);
            return false;
        }
    }

    // Test rotation functions
    WMATH_TYPE(Mat3) m16 = WMATH_ROTATION(Mat3)(WMATH_PI_2);
    WMATH_TYPE(Mat3) m17 = WMATH_ROTATION_X(Mat3)(WMATH_PI_2);
    WMATH_TYPE(Mat3) m18 = WMATH_ROTATION_Y(Mat3)(WMATH_PI_2);
    WMATH_TYPE(Mat3) m19 = WMATH_ROTATION_Z(Mat3)(WMATH_PI_2);

    // Test rotate functions
    WMATH_TYPE(Mat3) m20 = WMATH_ROTATE(Mat3)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat3) m21 = WMATH_ROTATE_X(Mat3)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat3) m22 = WMATH_ROTATE_Y(Mat3)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat3) m23 = WMATH_ROTATE_Z(Mat3)(m1, WMATH_PI_2);

    // Test scaling functions
    WMATH_TYPE(Vec2) scale2d = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 2.0f, .v_y = 3.0f});
    WMATH_TYPE(Vec3) scale3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});

    WMATH_TYPE(Mat3) m24 = WMATH_CALL(Mat3, scaling)(scale2d);
    WMATH_TYPE(Mat3) m25 = WMATH_CALL(Mat3, scaling3D)(scale3d);
    WMATH_TYPE(Mat3) m26 = WMATH_CALL(Mat3, uniform_scaling)(2.0f);
    WMATH_TYPE(Mat3) m27 = WMATH_CALL(Mat3, uniform_scaling_3D)(3.0f);

    WMATH_TYPE(Mat3) m28 = WMATH_SCALE(Mat3)(m1, scale2d);
    WMATH_TYPE(Mat3) m29 = WMATH_CALL(Mat3, scale3D)(m1, scale3d);
    WMATH_TYPE(Mat3) m30 = WMATH_CALL(Mat3, uniform_scale)(m1, 2.0f);
    WMATH_TYPE(Mat3) m31 = WMATH_CALL(Mat3, uniform_scale_3D)(m1, 3.0f);

    // Test translation functions
    WMATH_TYPE(Vec2) translation2d = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 5.0f, .v_y = 10.0f});
    WMATH_TYPE(Mat3) m32 = WMATH_TRANSLATION(Mat3)(translation2d);
    WMATH_TYPE(Mat3) m33 = WMATH_SET_TRANSLATION(Mat3)(m1, translation2d);
    WMATH_TYPE(Vec2) getTranslation = WMATH_GET_TRANSLATION(Mat3)(m33);

    if (fabsf(getTranslation.v[0] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(getTranslation.v[1] - 10.0f) > wcn_math_get_epsilon()) {
        printf("Mat3 get_translation test failed\n");
        return false;
    }

    WMATH_TYPE(Mat3) m34 = WMATH_CALL(Mat3, translate)(m1, translation2d);

    // Test axis functions
    WMATH_TYPE(Vec2) testAxis = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Mat3) m35 = WMATH_CALL(Mat3, set_axis)(m1, testAxis, 0);
    WMATH_TYPE(Vec2) gotAxis = WMATH_CALL(Mat3, get_axis)(m35, 0);

    if (fabsf(gotAxis.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(gotAxis.v[1] - 2.0f) > wcn_math_get_epsilon()) {
        printf("Mat3 get/set_axis test failed\n");
        return false;
    }

    // Test get_scaling functions
    /**
    WMATH_TYPE(Vec2) scale2d = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 2.0f, .v_y = 3.0f});
    WMATH_TYPE(Mat3) m24 = WMATH_CALL(Mat3, scaling)(scale2d);
     */
    WMATH_TYPE(Vec2) scaling2d = WMATH_CALL(Mat3, get_scaling)(m24);

    if (fabsf(scaling2d.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(scaling2d.v[1] - 3.0f) > wcn_math_get_epsilon()) {
        printf("Mat3 get_scaling test failed\n");
        return false;
    }

    // Test conversion functions
    WMATH_TYPE(Quat) testQuat = WMATH_IDENTITY(Quat)();
    WMATH_TYPE(Mat4) testMat4 = WMATH_IDENTITY(Mat4)();

    WMATH_TYPE(Mat3) m36 = WMATH_CALL(Mat3, from_quat)(testQuat);
    WMATH_TYPE(Mat3) m37 = WMATH_CALL(Mat3, from_mat4)(testMat4);

    // Should be close to identity
    for (int i = 0; i < 12; i++) {
        const float expected = (i % 4 == 3) ? 0.0f : ((i == 0 || i == 5 || i == 10) ? 1.0f : 0.0f);
        if (fabsf(m36.m[i] - expected) > wcn_math_get_epsilon() ||
            fabsf(m37.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat3 from_quat/from_mat4 test failed at index %d\n", i);
            return false;
        }
    }

    printf("All Mat3 tests passed!\n");
    return true;
}

bool TestMat4() {
    // Initialize epsilon for approximate comparisons
    wcn_math_set_epsilon(1e-5f);

    // Test zero
    WMATH_TYPE(Mat4) m0 = WMATH_ZERO(Mat4)();
    for (int i = 0; i < 16; i++) {
        if (fabsf(m0.m[i]) > wcn_math_get_epsilon()) {
            printf("Mat4 zero test failed at index %d\n", i);
            return false;
        }
    }

    // Test identity
    WMATH_TYPE(Mat4) m1 = WMATH_IDENTITY(Mat4)();
    for (int i = 0; i < 16; i++) {
        float expected = (i % 5 == 0) ? 1.0f : 0.0f; // Diagonal elements should be 1
        if (fabsf(m1.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 identity test failed at index %d\n", i);
            return false;
        }
    }

    // Test create
    WMATH_TYPE(Mat4) m2 = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 1.0f, .m_01 = 2.0f, .m_02 = 3.0f, .m_03 = 4.0f,
        .m_10 = 5.0f, .m_11 = 6.0f, .m_12 = 7.0f, .m_13 = 8.0f,
        .m_20 = 9.0f, .m_21 = 10.0f, .m_22 = 11.0f, .m_23 = 12.0f,
        .m_30 = 13.0f, .m_31 = 14.0f, .m_32 = 15.0f, .m_33 = 16.0f
    });
    for (int i = 0; i < 16; i++) {
        float expected = (float)(i + 1);
        if (fabsf(m2.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 create test failed at index %d\n", i);
            return false;
        }
    }

    // Test copy
    WMATH_TYPE(Mat4) m3 = WMATH_COPY(Mat4)(m2);
    for (int i = 0; i < 16; i++) {
        float expected = (float)(i + 1);
        if (fabsf(m3.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 copy test failed at index %d\n", i);
            return false;
        }
    }

    // Test add
    const WMATH_TYPE(Mat4) m4 = WMATH_ADD(Mat4)(m2, m1);
    for (int i = 0; i < 16; i++) {const float add_expected[16] = {
        2.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 7.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 12.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 17.0f
    };
        if (fabsf(m4.m[i] - add_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat4 add test failed at index %d\n", i);
            return false;
        }
    }

    // Test multiply
    WMATH_TYPE(Mat4) m5 = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f, .m_03 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f, .m_13 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f, .m_23 = 0.0f,
        .m_30 = 0.0f, .m_31 = 0.0f, .m_32 = 0.0f, .m_33 = 2.0f
    });
    const WMATH_TYPE(Mat4) m6 = WMATH_MULTIPLY(Mat4)(m2, m5);
    for (int i = 0; i < 16; i++) {const float multiply_expected[16] = {
        2.0f, 4.0f, 6.0f, 8.0f,
        10.0f, 12.0f, 14.0f, 16.0f,
        18.0f, 20.0f, 22.0f, 24.0f,
        26.0f, 28.0f, 30.0f, 32.0f
    };
        if (fabsf(m6.m[i] - multiply_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat4 multiply test failed at index %d\n", i);
            return false;
        }
    }

    // Test transpose
    const WMATH_TYPE(Mat4) m7 = WMATH_CALL(Mat4, transpose)(m2);
    for (int i = 0; i < 16; i++) {const float transpose_expected[16] = {
        1.0f, 5.0f, 9.0f, 13.0f,
        2.0f, 6.0f, 10.0f, 14.0f,
        3.0f, 7.0f, 11.0f, 15.0f,
        4.0f, 8.0f, 12.0f, 16.0f
    };
        if (fabsf(m7.m[i] - transpose_expected[i]) > wcn_math_get_epsilon()) {
            printf("Mat4 transpose test failed at index %d\n", i);
            return false;
        }
    }

    // Test determinant
    float det = WMATH_DETERMINANT(Mat4)(m2);
    // For this specific matrix, the determinant should be 0
    if (fabsf(det - 0.0f) > wcn_math_get_epsilon()) {
        printf("Mat4 determinant test failed: %f\n", det);
        return false;
    }

    // Test set
    WMATH_TYPE(Mat4) m8 = WMATH_SET(Mat4)(m1,
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
        28.0f, 29.0f, 30.0f, 31.0f,
        32.0f, 33.0f, 34.0f, 35.0f);
    for (int i = 0; i < 16; i++) {
        float expected = 20.0f + (float)i;
        if (fabsf(m8.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 set test failed at index %d\n", i);
            return false;
        }
    }

    // Test negate
    WMATH_TYPE(Mat4) m9 = WMATH_NEGATE(Mat4)(m2);
    for (int i = 0; i < 16; i++) {
        float expected = -(float)(i + 1);
        if (fabsf(m9.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 negate test failed at index %d\n", i);
            return false;
        }
    }

    // Test sub
    WMATH_TYPE(Mat4) m10 = WMATH_SUB(Mat4)(m2, m1);
    for (int i = 0; i < 16; i++) {
        float expected = (float)(i + 1);
        if (i % 5 == 0) expected -= 1.0f; // Diagonal elements
        if (fabsf(m10.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 sub test failed at index %d\n", i);
            return false;
        }
    }

    // Test multiply_scalar
    WMATH_TYPE(Mat4) m11 = WMATH_MULTIPLY_SCALAR(Mat4)(m2, 3.0f);
    for (int i = 0; i < 16; i++) {
        float expected = 3.0f * (float)(i + 1);
        if (fabsf(m11.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 multiply_scalar test failed at index %d\n", i);
            return false;
        }
    }

    // Test get_translation
    WMATH_TYPE(Mat4) m12 = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 1.0f, .m_01 = 0.0f, .m_02 = 0.0f, .m_03 = 0.0f,
        .m_10 = 0.0f, .m_11 = 1.0f, .m_12 = 0.0f, .m_13 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 1.0f, .m_23 = 0.0f,
        .m_30 = 10.0f, .m_31 = 20.0f, .m_32 = 30.0f, .m_33 = 1.0f
    });
    WMATH_TYPE(Vec3) translation = WMATH_GET_TRANSLATION(Mat4)(m12);
    if (fabsf(translation.v[0] - 10.0f) > wcn_math_get_epsilon() ||
        fabsf(translation.v[1] - 20.0f) > wcn_math_get_epsilon() ||
        fabsf(translation.v[2] - 30.0f) > wcn_math_get_epsilon()) {
        printf("Mat4 get_translation test failed\n");
        return false;
    }

    // Test 3D transformation functions
    WMATH_TYPE(Vec3) scale3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});
    WMATH_TYPE(Vec3) translation3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 5.0f, .v_y = 10.0f, .v_z = 15.0f});
    WMATH_TYPE(Vec3) axis = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 1.0f});

    // Test rotation functions
    WMATH_TYPE(Mat4) m13 = WMATH_ROTATION_X(Mat4)(WMATH_PI_2);
    WMATH_TYPE(Mat4) m14 = WMATH_ROTATION_Y(Mat4)(WMATH_PI_2);
    WMATH_TYPE(Mat4) m15 = WMATH_ROTATION_Z(Mat4)(WMATH_PI_2);
    WMATH_TYPE(Mat4) m16 = WMATH_ROTATION(Mat4)(axis, WMATH_PI_2);
    WMATH_TYPE(Mat4) m17 = WMATH_CALL(Mat4, axis_rotation)(axis, WMATH_PI_2);

    // Test rotate functions
    WMATH_TYPE(Mat4) m18 = WMATH_ROTATE_X(Mat4)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat4) m19 = WMATH_ROTATE_Y(Mat4)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat4) m20 = WMATH_ROTATE_Z(Mat4)(m1, WMATH_PI_2);
    WMATH_TYPE(Mat4) m21 = WMATH_ROTATE(Mat4)(m1, axis, WMATH_PI_2);
    WMATH_TYPE(Mat4) m22 = WMATH_CALL(Mat4, axis_rotate)(m1, axis, WMATH_PI_2);

    // Test scaling functions
    /**
    WMATH_TYPE(Vec3) scale3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});
     */
    WMATH_TYPE(Mat4) m23 = WMATH_CALL(Mat4, scaling)(scale3d);

    WMATH_TYPE(Vec3) gotScaling = WMATH_CALL(Mat4, get_scaling)(m23);
    if (fabsf(gotScaling.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(gotScaling.v[1] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(gotScaling.v[2] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Mat4 get_scaling test failed\n");
        return false;
    }

    // Test translation functions
    /**
    WMATH_TYPE(Vec3) translation3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 5.0f, .v_y = 10.0f, .v_z = 15.0f});
     */
    WMATH_TYPE(Mat4) m27 = WMATH_TRANSLATION(Mat4)(translation3d);

    WMATH_TYPE(Vec3) gotTranslation = WMATH_GET_TRANSLATION(Mat4)(m27);
    if (fabsf(gotTranslation.v[0] - 5.0f) > wcn_math_get_epsilon() ||
        fabsf(gotTranslation.v[1] - 10.0f) > wcn_math_get_epsilon() ||
        fabsf(gotTranslation.v[2] - 15.0f) > wcn_math_get_epsilon()) {
        printf("Mat4 translation test failed\n");
        return false;
    }

    // Test axis functions
    WMATH_TYPE(Vec3) testAxis3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Mat4) m30 = WMATH_CALL(Mat4, set_axis)(m1, testAxis3d, 0);
    WMATH_TYPE(Vec3) gotAxis3d = WMATH_CALL(Mat4, get_axis)(m30, 0);

    if (fabsf(gotAxis3d.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(gotAxis3d.v[1] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(gotAxis3d.v[2] - 3.0f) > wcn_math_get_epsilon()) {
        printf("Mat4 get/set_axis test failed\n");
        return false;
    }

    // Test projection functions
    WMATH_TYPE(Mat4) m31 = WMATH_CALL(Mat4, perspective)(WMATH_PI / 4.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    WMATH_TYPE(Mat4) m32 = WMATH_CALL(Mat4, perspective_reverse_z)(WMATH_PI / 4.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    WMATH_TYPE(Mat4) m33 = WMATH_CALL(Mat4, ortho)(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    WMATH_TYPE(Mat4) m34 = WMATH_CALL(Mat4, frustum)(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    WMATH_TYPE(Mat4) m35 = WMATH_CALL(Mat4, frustum_reverse_z)(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

    // Validate projection matrices - check basic properties
    // Perspective matrix should have specific structure
    // Let's check some basic properties instead of specific values
    float perspective_det = WMATH_DETERMINANT(Mat4)(m31);
    if (fabsf(perspective_det) < wcn_math_get_epsilon()) {
        printf("Mat4 perspective matrix determinant test failed: %f\n", perspective_det);
        return false;
    }

    // Perspective reverse Z should also have non-zero determinant
    float perspective_revz_det = WMATH_DETERMINANT(Mat4)(m32);
    if (fabsf(perspective_revz_det) < wcn_math_get_epsilon()) {
        printf("Mat4 perspective_reverse_z matrix determinant test failed: %f\n", perspective_revz_det);
        return false;
    }

    // Ortho matrix should have non-zero determinant
    float ortho_det = WMATH_DETERMINANT(Mat4)(m33);
    if (fabsf(ortho_det) < wcn_math_get_epsilon()) {
        printf("Mat4 ortho matrix determinant test failed: %f\n", ortho_det);
        return false;
    }

    // 4. Frustum matrices (INCORRECT in original code)
    // FIX: They are structurally identical to perspective matrices and MUST be invertible.
    // Determinant should be NON-ZERO.
    float frustum_det = WMATH_DETERMINANT(Mat4)(m34);
    if (fabsf(frustum_det) < wcn_math_get_epsilon()) { // Changed form > to <
        printf("Mat4 frustum matrix determinant test failed: got %f (Expected non-zero)\n", frustum_det);
        return false;
    }

    float frustum_revz_det = WMATH_DETERMINANT(Mat4)(m35);
    if (fabsf(frustum_revz_det) < wcn_math_get_epsilon()) { // Changed form > to <
        printf("Mat4 frustum_reverse_z matrix determinant test failed: got %f (Expected non-zero)\n", frustum_revz_det);
        return false;
    }

    // Test view functions
    WMATH_TYPE(Vec3) eye = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 5.0f});
    WMATH_TYPE(Vec3) target = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) up = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});

    WMATH_TYPE(Mat4) m36 = WMATH_CALL(Mat4, look_at)(eye, target, up);
    WMATH_TYPE(Mat4) m37 = WMATH_CALL(Mat4, aim)(eye, target, up);
    WMATH_TYPE(Mat4) m38 = WMATH_CALL(Mat4, camera_aim)(eye, target, up);

    // Validate view matrices - check that they transform points correctly
    // look_at should create a view matrix that moves the eye to origin
    WMATH_TYPE(Vec4) testPoint = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 0.0f, .v_w = 1.0f});
    WMATH_TYPE(Vec4) transformedPoint = WMATH_CALL(Vec4, transform_mat4)(testPoint, m36);
    
    // The target point (0,0,0) should be transformed to (0,0,-5) in view space
    if (fabsf(transformedPoint.v[0] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(transformedPoint.v[1] - 0.0f) > wcn_math_get_epsilon() ||
        fabsf(transformedPoint.v[2] - (-5.0f)) > wcn_math_get_epsilon()) {
        printf("Mat4 look_at transformation test failed\n");
        return false;
    }

    // Test that aim and camera_aim produce reasonable results (at least create valid matrices)
    // Check that they don't produce NaN or infinity values
    for (int i = 0; i < 16; i++) {
        if (isnan(m37.m[i]) || isinf(m37.m[i]) || isnan(m38.m[i]) || isinf(m38.m[i])) {
            printf("Mat4 aim/camera_aim produced invalid values test failed\n");
            return false;
        }
    }

    // Test conversion functions
    WMATH_TYPE(Quat) testQuat = WMATH_IDENTITY(Quat)();
    WMATH_TYPE(Mat3) testMat3 = WMATH_IDENTITY(Mat3)();
    WMATH_TYPE(Mat4) m39 = WMATH_CALL(Mat4, from_quat)(testQuat);
    WMATH_TYPE(Mat4) m40 = WMATH_CALL(Mat4, from_mat3)(testMat3);

    // Should be close to identity
    for (int i = 0; i < 16; i++) {
        const float expected = (i % 5 == 0) ? 1.0f : 0.0f;
        if (fabsf(m39.m[i] - expected) > wcn_math_get_epsilon() ||
            fabsf(m40.m[i] - expected) > wcn_math_get_epsilon()) {
            printf("Mat4 from_quat/from_mat3 test failed at index %d\n", i);
            return false;
        }
    }

    // Test inverse for a simple matrix
    WMATH_TYPE(Mat4) simpleMatrix = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f, .m_03 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f, .m_13 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f, .m_23 = 0.0f,
        .m_30 = 0.0f, .m_31 = 0.0f, .m_32 = 0.0f, .m_33 = 1.0f
    });
    WMATH_TYPE(Mat4) invMatrix = WMATH_INVERSE(Mat4)(simpleMatrix);
    WMATH_TYPE(Mat4) product = WMATH_MULTIPLY(Mat4)(simpleMatrix, invMatrix);

    // Should be close to identity
    for (int i = 0; i < 16; i++) {
        const float expected = (i % 5 == 0) ? 1.0f : 0.0f;
        if (fabsf(product.m[i] - expected) > wcn_math_get_epsilon() * 100.0f) {
            printf("Mat4 inverse test failed at index %d\n", i);
            return false;
        }
    }

    printf("All Mat4 tests passed!\n");
    return true;
}

bool TestTransformations() {
    wcn_math_set_epsilon(1e-5f);

    // Test Vec2 transformations
    WMATH_TYPE(Vec2) v2 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Mat3) m3 = WMATH_IDENTITY(Mat3)();
    WMATH_TYPE(Mat4) m4 = WMATH_IDENTITY(Mat4)();

    WMATH_TYPE(Vec2) v2_transformed_m3 = WMATH_CALL(Vec2, transform_mat3)(v2, m3);
    WMATH_TYPE(Vec2) v2_transformed_m4 = WMATH_CALL(Vec2, transform_mat4)(v2, m4);

    if (fabsf(v2_transformed_m3.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(v2_transformed_m3.v[1] - 2.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 transform_mat3 test failed\n");
        return false;
    }

    // Test Vec3 transformations
    WMATH_TYPE(Vec3) v3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Quat) q = WMATH_IDENTITY(Quat)();

    WMATH_TYPE(Vec3) v3_transformed_m3 = WMATH_CALL(Vec3, transform_mat3)(v3, m3);
    WMATH_TYPE(Vec3) v3_transformed_m4 = WMATH_CALL(Vec3, transform_mat4)(v3, m4);
    WMATH_TYPE(Vec3) v3_transformed_m4_upper = WMATH_CALL(Vec3, transform_mat4_upper3x3)(v3, m4);
    WMATH_TYPE(Vec3) v3_transformed_quat = WMATH_CALL(Vec3, transform_quat)(v3, q);

    if (fabsf(v3_transformed_m3.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_transformed_m3.v[1] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_transformed_m3.v[2] - 3.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 transform_mat3 test failed\n");
        return false;
    }

    if (fabsf(v3_transformed_quat.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_transformed_quat.v[1] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_transformed_quat.v[2] - 3.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 transform_quat test failed\n");
        return false;
    }

    // Test Vec4 transformations
    /// m4
    /// 1 0 0 0
    /// 0 1 0 0
    /// 0 0 1 0
    /// 0 0 0 1
    const WMATH_TYPE(Vec4) v4 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 1.0f});
    const WMATH_TYPE(Vec4) v4_transformed = WMATH_CALL(Vec4, transform_mat4)(v4, m4);

    if (fabsf(v4_transformed.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(v4_transformed.v[1] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(v4_transformed.v[2] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(v4_transformed.v[3] - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec4 transform_mat4 test failed\n");
        return false;
    }

    // Test get/set scale for Vec3
    WMATH_TYPE(Vec3) scaleVec = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});
    WMATH_TYPE(Mat4) scaleMat = WMATH_CALL(Mat4, scaling)(scaleVec);
    WMATH_TYPE(Vec3) gotScale = WMATH_CALL(Vec3, get_scale)(scaleMat);

    if (fabsf(gotScale.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(gotScale.v[1] - 3.0f) > wcn_math_get_epsilon() ||
        fabsf(gotScale.v[2] - 4.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 get_scale test failed\n");
        return false;
    }

    // Test get axis
    WMATH_TYPE(Vec3) xAxis = WMATH_CALL(Vec3, get_axis)(m4, 0);
    WMATH_TYPE(Vec3) yAxis = WMATH_CALL(Vec3, get_axis)(m4, 1);
    WMATH_TYPE(Vec3) zAxis = WMATH_CALL(Vec3, get_axis)(m4, 2);

    if (fabsf(xAxis.v[0] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(yAxis.v[1] - 1.0f) > wcn_math_get_epsilon() ||
        fabsf(zAxis.v[2] - 1.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 get_axis test failed\n");
        return false;
    }

    printf("All Transformation tests passed!\n");
    return true;
}

bool TestScaleOperations() {
    wcn_math_set_epsilon(1e-5f);

    // Test Vec2 scale
    WMATH_TYPE(Vec2) v2 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 2.0f, .v_y = 3.0f});
    WMATH_TYPE(Vec2) v2_scaled = WMATH_SCALE(Vec2)(v2, 2.0f);

    if (fabsf(v2_scaled.v[0] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(v2_scaled.v[1] - 6.0f) > wcn_math_get_epsilon()) {
        printf("Vec2 scale test failed\n");
        return false;
    }

    // Test Vec3 scale
    WMATH_TYPE(Vec3) v3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});
    WMATH_TYPE(Vec3) v3_scaled = WMATH_SCALE(Vec3)(v3, 3.0f);

    if (fabsf(v3_scaled.v[0] - 6.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_scaled.v[1] - 9.0f) > wcn_math_get_epsilon() ||
        fabsf(v3_scaled.v[2] - 12.0f) > wcn_math_get_epsilon()) {
        printf("Vec3 scale test failed\n");
        return false;
    }

    // Test Quat scale
    WMATH_TYPE(Quat) q = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Quat) q_scaled = WMATH_SCALE(Quat)(q, 2.0f);

    if (fabsf(q_scaled.v[0] - 2.0f) > wcn_math_get_epsilon() ||
        fabsf(q_scaled.v[1] - 4.0f) > wcn_math_get_epsilon() ||
        fabsf(q_scaled.v[2] - 6.0f) > wcn_math_get_epsilon() ||
        fabsf(q_scaled.v[3] - 8.0f) > wcn_math_get_epsilon()) {
        printf("Quat scale test failed\n");
        return false;
    }

    printf("All Scale operation tests passed!\n");
    return true;
}

int main() {
    TestVec2();
    TestVec3();
    TestVec4();
    TestQuat();
    TestMat3();
    TestMat4();
    TestTransformations();
    TestScaleOperations();
    return 0;
}