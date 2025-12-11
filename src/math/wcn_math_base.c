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
// ==================================================================

// Impl of random for float, double, int, and float_t
// ==================================================================
int WMATH_RANDOM(int)() { return rand(); }
float WMATH_RANDOM(float)() { return ((float)rand()) / RAND_MAX; }
double WMATH_RANDOM(double)() { return ((double)rand()) / RAND_MAX; }
// ==================================================================


// Impl of clamp for float, double, int, and float_t
int WMATH_CLAMP(int)(int v, int min, int max) {return _CLAMP;}
float WMATH_CLAMP(float)(float v, float min, float max) {return _CLAMP;}
double WMATH_CLAMP(double)(double v, double min, double max) {return _CLAMP;}