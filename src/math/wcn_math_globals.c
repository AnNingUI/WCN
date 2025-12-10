#include "wcn_math_internal.h"

// Define the global variables that were declared as extern in the header
const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4] = {
  { 1, -1,  1, -1}, // XYZ
  {-1, -1,  1,  1}, // XZY
  { 1, -1, -1,  1}, // YXZ
  { 1,  1, -1, -1}, // YZX
  {-1,  1,  1, -1}, // ZXY
  {-1,  1, -1,  1}  // ZYX
};

float EPSILON = 0.0f;

bool EPSILON_IS_SET = false;

float wcn_math_set_epsilon(const float epsilon) {
  const float old_epsilon = EPSILON_IS_SET ? EPSILON : 1e-6f;
  EPSILON = epsilon;
  EPSILON_IS_SET = true;
  return old_epsilon;
}

float wcn_math_get_epsilon() { return EPSILON_IS_SET ? EPSILON : 1e-6f; }