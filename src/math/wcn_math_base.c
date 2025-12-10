#include "WCN/WCN_Math.h"
#include <stdlib.h>

#define _LERP (a + ((b) - (a)) * t)
#define _CLAMP (v < min ? min : v > max ? max : v)
// #define WMATH_NUM_LERP(a, b, t) ((a) + ((b) - (a)) * (t))
// Impl of lerp for float, double, int, and float_t
// ==================================================================
int WMATH_LERP(int)(const int a, const int b, const float t) {return _LERP;}
float WMATH_LERP(float)(const float a, const float b, const float t) {return _LERP;}
double WMATH_LERP(double)(const double a, const double b, const double t) {return _LERP;}
float_t WMATH_LERP(float_t)(const float_t a, const float_t b, const float_t t) {return _LERP;}
double_t WMATH_LERP(double_t)(const double_t a, const double_t b, const double_t t) {return _LERP;}
// ==================================================================

// Impl of random for float, double, int, and float_t
// ==================================================================
int WMATH_RANDOM(int)() { return rand(); }
float WMATH_RANDOM(float)() { return ((float)rand()) / RAND_MAX; }
double WMATH_RANDOM(double)() { return ((double)rand()) / RAND_MAX; }
float_t WMATH_RANDOM(float_t)() { return ((float_t)rand()) / RAND_MAX; }
double_t WMATH_RANDOM(double_t)() { return ((double_t)rand()) / RAND_MAX; }
// ==================================================================


// Impl of clamp for float, double, int, and float_t
int WMATH_CLAMP(int)(int v, int min, int max) {return _CLAMP;}
float WMATH_CLAMP(float)(float v, float min, float max) {return _CLAMP;}
double WMATH_CLAMP(double)(double v, double min, double max) {return _CLAMP;}
float_t WMATH_CLAMP(float_t)(float_t v, float_t min, float_t max) {return _CLAMP;}
double_t WMATH_CLAMP(double_t)(float_t v, float_t min, float_t max) {return _CLAMP;}