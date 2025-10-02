//
// Created by AnNingUI on 25-9-20.
//

#ifndef WCN_FP_H
#define WCN_FP_H

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------
// GNU C / Clang 支持 __typeof__
// -------------------------
#if defined(__GNUC__) || defined(__clang__)
#define CROSS_TYPEOF(expr) __typeof__(expr)
// MSVC C++ 支持 decltype
#elif defined(_MSC_VER) && defined(__cplusplus)
#define CROSS_TYPEOF(expr) decltype(expr)
// C11 支持 _Generic
#elif __STDC_VERSION__ >= 201112L
#define CROSS_TYPEOF(x)                                                        \
  _Generic((x),                                                                \
      char: char,                                                              \
      signed char: signed char,                                                \
      unsigned char: unsigned char,                                            \
      short: short,                                                            \
      unsigned short: unsigned short,                                          \
      int: int,                                                                \
      unsigned int: unsigned int,                                              \
      long: long,                                                              \
      unsigned long: unsigned long,                                            \
      long long: long long,                                                    \
      unsigned long long: unsigned long long,                                  \
      float: float,                                                            \
      double: double,                                                          \
      long double: long double,                                                \
      default: _Generic(&(x),                                                  \
          void *: void *,                                                      \
          default: __typeof__(*(0 ? &(x)                                       \
                                  : (x *)0)) /* 推导 struct/union/数组 */      \
                        ))

// 其他情况报错
#else
#error "Compiler does not support typeof. You need C11 or C++11 or GCC/Clang."
#endif

#define Fp_Send(arr, new_value) arr[$index] = new_value
#define Fp_ForIn(arr)                                                          \
  for (int $index = 0, $now_index = 0; $index < sizeof(arr) / sizeof(arr[0]);  \
       $index++)                                                               \
    for (CROSS_TYPEOF(arr[0]) $value = arr[$index]; $index - $now_index == 0;  \
         $now_index++)

#define Fp_Let(TYPE_VAL, NEXT_FUNC)                                            \
  TYPE_VAL;                                                                    \
  NEXT_FUNC

/// ==============================
/// Map
/// ==============================
#define Fp_Map(arr)                                                            \
  {0};                                                                         \
  Fp_ForIn(arr)

#define Fp_MapLet(TYPE, VAL_NAME, MAP_ARR, CODE_BODY)                          \
  TYPE VAL_NAME[sizeof(MAP_ARR) / sizeof(MAP_ARR[0])] = {0};                   \
  typedef TYPE $Self;                                                          \
  Fp_ForIn(MAP_ARR) {                                                          \
    TYPE $_NOW_VALUE;                                                          \
    CODE_BODY                                                                  \
    Fp_Send(VAL_NAME, $_NOW_VALUE);                                            \
  }

#define Fp_MapLet_Send(new_value) $_NOW_VALUE = ($Self)new_value

/// ==============================
/// Reduce
/// ==============================
#define Fp_Reduce_Send(bind_value, new_value)                                  \
  $acc = ($Self)new_value;                                                     \
  bind_value = $acc
#define Fp_Reduce(TYPE, SRC_ARR, INIT, CODE_BODY)                              \
  {0};                                                                         \
  {                                                                            \
    TYPE $acc = (TYPE)(INIT);                                                  \
    for (int $index = 0, $now_index = 0;                                       \
         $index < sizeof(arr) / sizeof(arr[0]); $index++) {                    \
      for (__typeof__(arr[0]) $value = arr[$index]; $index - $now_index == 0;  \
           $now_index++) {                                                     \
        typedef TYPE $Self;                                                    \
        CODE_BODY                                                              \
      }                                                                        \
    }                                                                          \
  }

#define Fp_ReduceLet(TYPE, VAL_NAME, SRC_ARR, INIT, CODE_BODY)                 \
  TYPE VAL_NAME;                                                               \
  {                                                                            \
    typedef TYPE $Self;                                                        \
    TYPE $acc = ($Self)INIT;                                                   \
    for (int $index = 0, $now_index = 0;                                       \
         $index < sizeof(arr) / sizeof(arr[0]); $index++) {                    \
      for (__typeof__(arr[0]) $value = arr[$index]; $index - $now_index == 0;  \
           $now_index++) {                                                     \
        CODE_BODY                                                              \
      }                                                                        \
    }                                                                          \
    VAL_NAME = $acc;                                                           \
  }

#define Fp_ReduceLet_Send(new_value) $acc = ($Self)new_value

/// ==============================
/// Filter
/// ==============================    
#define Fp_FilterLet(TYPE, VAL_NAME, SRC_ARR, CODE_BODY)                        \
TYPE VAL_NAME[sizeof(SRC_ARR) / sizeof(SRC_ARR[0])] = {0};                   \
{ \
    int $count = 0;                                                              \
    Fp_ForIn(SRC_ARR) {                                                          \
        typedef TYPE $Self;                                                        \
        if (CODE_BODY) {                                                           \
            VAL_NAME[$count++] = ($Self)$value;                                      \
        }                                                                          \
    } \
}
    
#ifdef __cplusplus
}
#endif

#endif // WCN_FP_H
