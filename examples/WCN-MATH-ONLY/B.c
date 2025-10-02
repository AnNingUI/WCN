#include "WCN/WCN_Math.h"
#include <time.h>
#include <stdio.h>

/**
 *
 * 性能测试
 */
void TestVec2() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test vectors
    T$(Vec2) v1 = INIT$(Vec2, .v_x = 1.0f, .v_y = 2.0f);
    T$(Vec2) v2 = INIT$(Vec2, .v_x = 3.0f, .v_y = 4.0f);
    T$(Vec2) v3 = INIT$(Vec2, .v_x = 5.0f, .v_y = 6.0f);
    
    printf("Vec2 Performance Tests:\n");
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        T$(Vec2) v = INIT$(Vec2, .v_x = i, .v_y = i+1);
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test set
    start = clock();
    for (int i = 0; i < iterations; i++) {
        v1 = WMATH_SET(Vec2)(v1, (float)i, (float)(i+1));
    }
    end = clock();
    printf("  set: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_COPY(Vec2)(v1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_ADD(Vec2)(v1, v2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test sub
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_SUB(Vec2)(v1, v2);
    }
    end = clock();
    printf("  sub: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_MULTIPLY(Vec2)(v1, v2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiplyScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_MULTIPLY_SCALAR(Vec2)(v1, 2.0f);
    }
    end = clock();
    printf("  multiplyScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test dot
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DOT(Vec2)(v1, v2);
    }
    end = clock();
    printf("  dot: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test length
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH(Vec2)(v1);
    }
    end = clock();
    printf("  length: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test normalize
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_NORMALIZE(Vec2)(v1);
    }
    end = clock();
    printf("  normalize: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_LERP(Vec2)(v1, v2, 0.5f);
    }
    end = clock();
    printf("  lerp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test rotate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_ROTATE(Vec2)(v1, v2, 1.0f);
    }
    end = clock();
    printf("  rotate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test cross
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_CROSS(Vec2)(v1, v2);
    }
    end = clock();
    printf("  cross: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distance
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE(Vec2)(v1, v2);
    }
    end = clock();
    printf("  distance: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equals
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS(Vec2)(v1, v2);
    }
    end = clock();
    printf("  equals: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equalsApproximately
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS_APPROXIMATELY(Vec2)(v1, v2);
    }
    end = clock();
    printf("  equalsApproximately: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test negate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_NEGATE(Vec2)(v1);
    }
    end = clock();
    printf("  negate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test ceil
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_CEIL(Vec2)(v1);
    }
    end = clock();
    printf("  ceil: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test floor
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_FLOOR(Vec2)(v1);
    }
    end = clock();
    printf("  floor: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test round
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_ROUND(Vec2)(v1);
    }
    end = clock();
    printf("  round: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test clamp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_CLAMP(Vec2)(v1, 0.0f, 10.0f);
    }
    end = clock();
    printf("  clamp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test addScaled
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_ADD_SCALED(Vec2)(v1, v2, 2.0f);
    }
    end = clock();
    printf("  addScaled: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test angle
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_ANGLE(Vec2)(v1, v2);
    }
    end = clock();
    printf("  angle: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerpV
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_LERP_V(Vec2)(v1, v2, v3);
    }
    end = clock();
    printf("  lerpV: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmax
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_FMAX(Vec2)(v1, v2);
    }
    end = clock();
    printf("  fmax: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmin
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_FMIN(Vec2)(v1, v2);
    }
    end = clock();
    printf("  fmin: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test divScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_DIV_SCALAR(Vec2)(v1, 2.0f);
    }
    end = clock();
    printf("  divScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test inverse
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_INVERSE(Vec2)(v1);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test div
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_DIV(Vec2)(v1, v2);
    }
    end = clock();
    printf("  div: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distanceSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE_SQ(Vec2)(v1, v2);
    }
    end = clock();
    printf("  distanceSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lengthSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH_SQ(Vec2)(v1);
    }
    end = clock();
    printf("  lengthSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test random
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_RANDOM(Vec2)(1.0f);
    }
    end = clock();
    printf("  random: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_ZERO(Vec2)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test identity
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_IDENTITY(Vec2)();
    }
    end = clock();
    printf("  identity: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test setLength
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_SET_LENGTH(Vec2)(v1, 5.0f);
    }
    end = clock();
    printf("  setLength: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test truncate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_TRUNCATE(Vec2)(v1, 5.0f);
    }
    end = clock();
    printf("  truncate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test midpoint
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_MIDPOINT(Vec2)(v1, v2);
    }
    end = clock();
    printf("  midpoint: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

void TestVec3() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test vectors
    WMATH_TYPE(Vec3) v1 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Vec3) v2 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 4.0f, .v_y = 5.0f, .v_z = 6.0f});
    WMATH_TYPE(Vec3) v3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 7.0f, .v_y = 8.0f, .v_z = 9.0f});
    
    printf("Vec3 Performance Tests:\n");
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = (float)i, .v_y = (float)(i+1), .v_z = (float)(i+2)});
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test set
    start = clock();
    for (int i = 0; i < iterations; i++) {
        v1 = WMATH_SET(Vec3)(v1, (float)i, (float)(i+1), (float)(i+2));
    }
    end = clock();
    printf("  set: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_COPY(Vec3)(v1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_ADD(Vec3)(v1, v2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test sub
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_SUB(Vec3)(v1, v2);
    }
    end = clock();
    printf("  sub: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_MULTIPLY(Vec3)(v1, v2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiplyScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_MULTIPLY_SCALAR(Vec3)(v1, 2.0f);
    }
    end = clock();
    printf("  multiplyScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test dot
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DOT(Vec3)(v1, v2);
    }
    end = clock();
    printf("  dot: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test cross
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_CROSS(Vec3)(v1, v2);
    }
    end = clock();
    printf("  cross: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test length
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH(Vec3)(v1);
    }
    end = clock();
    printf("  length: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test normalize
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_NORMALIZE(Vec3)(v1);
    }
    end = clock();
    printf("  normalize: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_LERP(Vec3)(v1, v2, 0.5f);
    }
    end = clock();
    printf("  lerp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distance
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE(Vec3)(v1, v2);
    }
    end = clock();
    printf("  distance: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equals
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS(Vec3)(v1, v2);
    }
    end = clock();
    printf("  equals: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equalsApproximately
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS_APPROXIMATELY(Vec3)(v1, v2);
    }
    end = clock();
    printf("  equalsApproximately: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test negate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_NEGATE(Vec3)(v1);
    }
    end = clock();
    printf("  negate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test ceil
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_CEIL(Vec3)(v1);
    }
    end = clock();
    printf("  ceil: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test floor
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_FLOOR(Vec3)(v1);
    }
    end = clock();
    printf("  floor: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test round
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_ROUND(Vec3)(v1);
    }
    end = clock();
    printf("  round: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test clamp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_CLAMP(Vec3)(v1, 0.0f, 10.0f);
    }
    end = clock();
    printf("  clamp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test addScaled
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_ADD_SCALED(Vec3)(v1, v2, 2.0f);
    }
    end = clock();
    printf("  addScaled: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test angle
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_ANGLE(Vec3)(v1, v2);
    }
    end = clock();
    printf("  angle: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerpV
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_LERP_V(Vec3)(v1, v2, v3);
    }
    end = clock();
    printf("  lerpV: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmax
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_FMAX(Vec3)(v1, v2);
    }
    end = clock();
    printf("  fmax: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmin
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_FMIN(Vec3)(v1, v2);
    }
    end = clock();
    printf("  fmin: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test divScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_DIV_SCALAR(Vec3)(v1, 2.0f);
    }
    end = clock();
    printf("  divScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test inverse
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_INVERSE(Vec3)(v1);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test div
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_DIV(Vec3)(v1, v2);
    }
    end = clock();
    printf("  div: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distanceSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE_SQ(Vec3)(v1, v2);
    }
    end = clock();
    printf("  distanceSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lengthSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH_SQ(Vec3)(v1);
    }
    end = clock();
    printf("  lengthSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test random
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_RANDOM(Vec3)(1.0f);
    }
    end = clock();
    printf("  random: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_ZERO(Vec3)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test setLength
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_SET_LENGTH(Vec3)(v1, 5.0f);
    }
    end = clock();
    printf("  setLength: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test truncate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_TRUNCATE(Vec3)(v1, 5.0f);
    }
    end = clock();
    printf("  truncate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test midpoint
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_MIDPOINT(Vec3)(v1, v2);
    }
    end = clock();
    printf("  midpoint: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

void TestVec4() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test vectors
    WMATH_TYPE(Vec4) v1 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Vec4) v2 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 5.0f, .v_y = 6.0f, .v_z = 7.0f, .v_w = 8.0f});
    WMATH_TYPE(Vec4) v3 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 9.0f, .v_y = 10.0f, .v_z = 11.0f, .v_w = 12.0f});
    
    printf("Vec4 Performance Tests:\n");
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = (float)i, .v_y = (float)(i+1), .v_z = (float)(i+2), .v_w = (float)(i+3)});
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test set
    start = clock();
    for (int i = 0; i < iterations; i++) {
        v1 = WMATH_SET(Vec4)(v1, (float)i, (float)(i+1), (float)(i+2), (float)(i+3));
    }
    end = clock();
    printf("  set: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_COPY(Vec4)(v1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_ADD(Vec4)(v1, v2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test sub
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_SUB(Vec4)(v1, v2);
    }
    end = clock();
    printf("  sub: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_MULTIPLY(Vec4)(v1, v2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiplyScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_MULTIPLY_SCALAR(Vec4)(v1, 2.0f);
    }
    end = clock();
    printf("  multiplyScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test dot
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DOT(Vec4)(v1, v2);
    }
    end = clock();
    printf("  dot: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test length
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH(Vec4)(v1);
    }
    end = clock();
    printf("  length: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test normalize
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_NORMALIZE(Vec4)(v1);
    }
    end = clock();
    printf("  normalize: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_LERP(Vec4)(v1, v2, 0.5f);
    }
    end = clock();
    printf("  lerp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distance
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE(Vec4)(v1, v2);
    }
    end = clock();
    printf("  distance: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equals
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS(Vec4)(v1, v2);
    }
    end = clock();
    printf("  equals: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equalsApproximately
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS_APPROXIMATELY(Vec4)(v1, v2);
    }
    end = clock();
    printf("  equalsApproximately: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test negate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_NEGATE(Vec4)(v1);
    }
    end = clock();
    printf("  negate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test ceil
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_CEIL(Vec4)(v1);
    }
    end = clock();
    printf("  ceil: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test floor
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_FLOOR(Vec4)(v1);
    }
    end = clock();
    printf("  floor: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test round
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_ROUND(Vec4)(v1);
    }
    end = clock();
    printf("  round: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test clamp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_CLAMP(Vec4)(v1, 0.0f, 10.0f);
    }
    end = clock();
    printf("  clamp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test addScaled
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_ADD_SCALED(Vec4)(v1, v2, 2.0f);
    }
    end = clock();
    printf("  addScaled: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerpV
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_LERP_V(Vec4)(v1, v2, v3);
    }
    end = clock();
    printf("  lerpV: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmax
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_FMAX(Vec4)(v1, v2);
    }
    end = clock();
    printf("  fmax: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test fmin
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_FMIN(Vec4)(v1, v2);
    }
    end = clock();
    printf("  fmin: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test divScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_DIV_SCALAR(Vec4)(v1, 2.0f);
    }
    end = clock();
    printf("  divScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test inverse
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_INVERSE(Vec4)(v1);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test div
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_DIV(Vec4)(v1, v2);
    }
    end = clock();
    printf("  div: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test distanceSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DISTANCE_SQ(Vec4)(v1, v2);
    }
    end = clock();
    printf("  distanceSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lengthSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH_SQ(Vec4)(v1);
    }
    end = clock();
    printf("  lengthSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_ZERO(Vec4)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test identity
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_IDENTITY(Vec4)();
    }
    end = clock();
    printf("  identity: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test setLength
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_SET_LENGTH(Vec4)(v1, 5.0f);
    }
    end = clock();
    printf("  setLength: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test truncate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_TRUNCATE(Vec4)(v1, 5.0f);
    }
    end = clock();
    printf("  truncate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test midpoint
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) v = WMATH_MIDPOINT(Vec4)(v1, v2);
    }
    end = clock();
    printf("  midpoint: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

void TestQuat() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test quaternions
    WMATH_TYPE(Quat) q1 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 4.0f});
    WMATH_TYPE(Quat) q2 = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = 5.0f, .v_y = 6.0f, .v_z = 7.0f, .v_w = 8.0f});
    
    printf("Quat Performance Tests:\n");
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_ZERO(Quat)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test identity
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_IDENTITY(Quat)();
    }
    end = clock();
    printf("  identity: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CREATE(Quat)((WMATH_CREATE_TYPE(Quat)){.v_x = (float)i, .v_y = (float)(i+1), .v_z = (float)(i+2), .v_w = (float)(i+3)});
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test set
    start = clock();
    for (int i = 0; i < iterations; i++) {
        q1 = WMATH_SET(Quat)(q1, (float)i, (float)(i+1), (float)(i+2), (float)(i+3));
    }
    end = clock();
    printf("  set: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_COPY(Quat)(q1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test dot
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DOT(Quat)(q1, q2);
    }
    end = clock();
    printf("  dot: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test length
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH(Quat)(q1);
    }
    end = clock();
    printf("  length: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lengthSq
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result += WMATH_LENGTH_SQ(Quat)(q1);
    }
    end = clock();
    printf("  lengthSq: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test normalize
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_NORMALIZE(Quat)(q1);
    }
    end = clock();
    printf("  normalize: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equals
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS(Quat)(q1, q2);
    }
    end = clock();
    printf("  equals: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test equalsApproximately
    start = clock();
    for (int i = 0; i < iterations; i++) {
        bool eq = WMATH_EQUALS_APPROXIMATELY(Quat)(q1, q2);
    }
    end = clock();
    printf("  equalsApproximately: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_ADD(Quat)(q1, q2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test sub
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_SUB(Quat)(q1, q2);
    }
    end = clock();
    printf("  sub: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiplyScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_MULTIPLY_SCALAR(Quat)(q1, 2.0f);
    }
    end = clock();
    printf("  multiplyScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test divScalar
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_DIV_SCALAR(Quat)(q1, 2.0f);
    }
    end = clock();
    printf("  divScalar: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test conjugate
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CALL(Quat, conjugate)(q1);
    }
    end = clock();
    printf("  conjugate: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test inverse
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_INVERSE(Quat)(q1);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_MULTIPLY(Quat)(q1, q2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test lerp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_LERP(Quat)(q1, q2, 0.5f);
    }
    end = clock();
    printf("  lerp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test slerp
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CALL(Quat, slerp)(q1, q2, 0.5f);
    }
    end = clock();
    printf("  slerp: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test from_axis_angle
    start = clock();
    WMATH_TYPE(Vec3) axis = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 1.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CALL(Quat, from_axis_angle)(axis, WMATH_PI_2);
    }
    end = clock();
    printf("  from_axis_angle: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test to_axis_angle
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WCN_Math_Vec3_WithAngleAxis result = WMATH_CALL(Quat, to_axis_angle)(q1);
    }
    end = clock();
    printf("  to_axis_angle: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation_to
    start = clock();
    WMATH_TYPE(Vec3) from = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) to = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CALL(Quat, rotation_to)(from, to);
    }
    end = clock();
    printf("  rotation_to: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test angle
    start = clock();
    for (int i = 0; i < iterations; i++) {
        float angle = WMATH_ANGLE(Quat)(q1, q2);
    }
    end = clock();
    printf("  angle: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test from_euler
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Quat) q = WMATH_CALL(Quat, from_euler)(WMATH_PI_2, 0.0f, 0.0f, WCN_Math_RotationOrder_XYZ);
    }
    end = clock();
    printf("  from_euler: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

void TestMat3() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test matrices
    WMATH_TYPE(Mat3) m1 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 1.0f, .m_01 = 2.0f, .m_02 = 3.0f,
        .m_10 = 4.0f, .m_11 = 5.0f, .m_12 = 6.0f,
        .m_20 = 7.0f, .m_21 = 8.0f, .m_22 = 9.0f
    });
    WMATH_TYPE(Mat3) m2 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f
    });
    
    printf("Mat3 Performance Tests:\n");
    
    // Test identity
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_IDENTITY(Mat3)();
    }
    end = clock();
    printf("  identity: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_ZERO(Mat3)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
            .m_00 = (float)i, .m_01 = (float)(i+1), .m_02 = (float)(i+2),
            .m_10 = (float)(i+3), .m_11 = (float)(i+4), .m_12 = (float)(i+5),
            .m_20 = (float)(i+6), .m_21 = (float)(i+7), .m_22 = (float)(i+8)
        });
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_COPY(Mat3)(m1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_ADD(Mat3)(m1, m2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_MULTIPLY(Mat3)(m1, m2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test transpose
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_CALL(Mat3, transpose)(m1);
    }
    end = clock();
    printf("  transpose: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test determinant
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DETERMINANT(Mat3)(m1);
    }
    end = clock();
    printf("  determinant: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test inverse
    start = clock();
    WMATH_TYPE(Mat3) m3 = WMATH_CREATE(Mat3)((WMATH_CREATE_TYPE(Mat3)){
        .m_00 = 2.0f, .m_01 = 1.0f, .m_02 = 1.0f,
        .m_10 = 1.0f, .m_11 = 3.0f, .m_12 = 2.0f,
        .m_20 = 3.0f, .m_21 = 2.0f, .m_22 = 4.0f
    });
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_INVERSE(Mat3)(m3);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation functions
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_ROTATION(Mat3)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation_x
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_ROTATION_X(Mat3)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation_x: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation_z
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_ROTATION_Z(Mat3)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation_z: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test scaling
    start = clock();
    WMATH_TYPE(Vec2) scale2d = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 2.0f, .v_y = 3.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_CALL(Mat3, scaling)(scale2d);
    }
    end = clock();
    printf("  scaling: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test translation
    start = clock();
    WMATH_TYPE(Vec2) translation2d = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 5.0f, .v_y = 10.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) m = WMATH_TRANSLATION(Mat3)(translation2d);
    }
    end = clock();
    printf("  translation: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test get_translation
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) v = WMATH_GET_TRANSLATION(Mat3)(m1);
    }
    end = clock();
    printf("  get_translation: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}
void TestMat4() {
    const int iterations = 1000000;
    clock_t start, end;
    
    // Create test matrices
    WMATH_TYPE(Mat4) m1 = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 1.0f, .m_01 = 2.0f, .m_02 = 3.0f, .m_03 = 4.0f,
        .m_10 = 5.0f, .m_11 = 6.0f, .m_12 = 7.0f, .m_13 = 8.0f,
        .m_20 = 9.0f, .m_21 = 10.0f, .m_22 = 11.0f, .m_23 = 12.0f,
        .m_30 = 13.0f, .m_31 = 14.0f, .m_32 = 15.0f, .m_33 = 16.0f
    });
    WMATH_TYPE(Mat4) m2 = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f, .m_03 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f, .m_13 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f, .m_23 = 0.0f,
        .m_30 = 0.0f, .m_31 = 0.0f, .m_32 = 0.0f, .m_33 = 2.0f
    });
    
    printf("Mat4 Performance Tests:\n");
    
    // Test identity
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_IDENTITY(Mat4)();
    }
    end = clock();
    printf("  identity: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test zero
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_ZERO(Mat4)();
    }
    end = clock();
    printf("  zero: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test create
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
            .m_00 = (float)i, .m_01 = (float)(i+1), .m_02 = (float)(i+2), .m_03 = (float)(i+3),
            .m_10 = (float)(i+4), .m_11 = (float)(i+5), .m_12 = (float)(i+6), .m_13 = (float)(i+7),
            .m_20 = (float)(i+8), .m_21 = (float)(i+9), .m_22 = (float)(i+10), .m_23 = (float)(i+11),
            .m_30 = (float)(i+12), .m_31 = (float)(i+13), .m_32 = (float)(i+14), .m_33 = (float)(i+15)
        });
    }
    end = clock();
    printf("  create: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test copy
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_COPY(Mat4)(m1);
    }
    end = clock();
    printf("  copy: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test add
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_ADD(Mat4)(m1, m2);
    }
    end = clock();
    printf("  add: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test multiply
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_MULTIPLY(Mat4)(m1, m2);
    }
    end = clock();
    printf("  multiply: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test transpose
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CALL(Mat4, transpose)(m1);
    }
    end = clock();
    printf("  transpose: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    
    // Test determinant
    start = clock();
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += WMATH_DETERMINANT(Mat4)(m1);
    }
    end = clock();
    printf("  determinant: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test inverse
    start = clock();
    WMATH_TYPE(Mat4) simpleMatrix = WMATH_CREATE(Mat4)((WMATH_CREATE_TYPE(Mat4)){
        .m_00 = 2.0f, .m_01 = 0.0f, .m_02 = 0.0f, .m_03 = 0.0f,
        .m_10 = 0.0f, .m_11 = 2.0f, .m_12 = 0.0f, .m_13 = 0.0f,
        .m_20 = 0.0f, .m_21 = 0.0f, .m_22 = 2.0f, .m_23 = 0.0f,
        .m_30 = 0.0f, .m_31 = 0.0f, .m_32 = 0.0f, .m_33 = 1.0f
    });
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_INVERSE(Mat4)(simpleMatrix);
    }
    end = clock();
    printf("  inverse: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation functions
    start = clock();
    WMATH_TYPE(Vec3) axis = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 1.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_ROTATION_X(Mat4)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation_x: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation_y
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_ROTATION_Y(Mat4)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation_y: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test rotation_z
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_ROTATION_Z(Mat4)(WMATH_PI_2);
    }
    end = clock();
    printf("  rotation_z: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test scaling
    start = clock();
    WMATH_TYPE(Vec3) scale3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 2.0f, .v_y = 3.0f, .v_z = 4.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CALL(Mat4, scaling)(scale3d);
    }
    end = clock();
    printf("  scaling: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test translation
    start = clock();
    WMATH_TYPE(Vec3) translation3d = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 5.0f, .v_y = 10.0f, .v_z = 15.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_TRANSLATION(Mat4)(translation3d);
    }
    end = clock();
    printf("  translation: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test get_translation
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) v = WMATH_GET_TRANSLATION(Mat4)(m1);
    }
    end = clock();
    printf("  get_translation: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test perspective
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CALL(Mat4, perspective)(WMATH_PI / 4.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    }
    end = clock();
    printf("  perspective: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test ortho
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CALL(Mat4, ortho)(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    }
    end = clock();
    printf("  ortho: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test look_at
    start = clock();
    WMATH_TYPE(Vec3) eye = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 5.0f});
    WMATH_TYPE(Vec3) target = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 0.0f, .v_z = 0.0f});
    WMATH_TYPE(Vec3) up = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 0.0f, .v_y = 1.0f, .v_z = 0.0f});
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) m = WMATH_CALL(Mat4, look_at)(eye, target, up);
    }
    end = clock();
    printf("  look_at: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

void TestTransformations() {
    const int iterations = 1000000;
    clock_t start, end;

    printf("Transformation Performance Tests:\n");

    // Test Vec2 transformations
    WMATH_TYPE(Vec2) v2 = WMATH_CREATE(Vec2)((WMATH_CREATE_TYPE(Vec2)){.v_x = 1.0f, .v_y = 2.0f});
    WMATH_TYPE(Mat3) m3 = WMATH_IDENTITY(Mat3)();
    WMATH_TYPE(Mat4) m4 = WMATH_IDENTITY(Mat4)();

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) result = WMATH_CALL(Vec2, transform_mat3)(v2, m3);
    }
    end = clock();
    printf("  vec2_transform_mat3: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec2) result = WMATH_CALL(Vec2, transform_mat4)(v2, m4);
    }
    end = clock();
    printf("  vec2_transform_mat4: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test Vec3 transformations
    WMATH_TYPE(Vec3) v3 = WMATH_CREATE(Vec3)((WMATH_CREATE_TYPE(Vec3)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f});
    WMATH_TYPE(Quat) q = WMATH_IDENTITY(Quat)();

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) result = WMATH_CALL(Vec3, transform_mat3)(v3, m3);
    }
    end = clock();
    printf("  vec3_transform_mat3: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) result = WMATH_CALL(Vec3, transform_mat4)(v3, m4);
    }
    end = clock();
    printf("  vec3_transform_mat4: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec3) result = WMATH_CALL(Vec3, transform_quat)(v3, q);
    }
    end = clock();
    printf("  vec3_transform_quat: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test Vec4 transformations
    WMATH_TYPE(Vec4) v4 = WMATH_CREATE(Vec4)((WMATH_CREATE_TYPE(Vec4)){.v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f, .v_w = 1.0f});

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Vec4) result = WMATH_CALL(Vec4, transform_mat4)(v4, m4);
    }
    end = clock();
    printf("  vec4_transform_mat4: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Test conversions
    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat3) result = WMATH_CALL(Mat3, from_quat)(q);
    }
    end = clock();
    printf("  mat3_from_quat: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for (int i = 0; i < iterations; i++) {
        WMATH_TYPE(Mat4) result = WMATH_CALL(Mat4, from_quat)(q);
    }
    end = clock();
    printf("  mat4_from_quat: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    printf("\n");
}

int main() {
    TestVec2();
    TestVec3();
    TestVec4();
    TestQuat();
    TestMat3();
    TestMat4();
    TestTransformations();
}