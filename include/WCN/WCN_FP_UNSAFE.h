// //
// // Created by AnNingUI on 25-9-20.
// //
#ifndef WCN_FP_UNSAFE_H
#define WCN_FP_UNSAFE_H

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================
// 场景 1: C23 标准 (MSVC /std:clatest, GCC -std=c2x)
// ===============================================================
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
    // 最完美的情况：使用标准去除 const
    #define CROSS_TYPE_UNQUAL(expr) typeof_unqual(expr)

// ===============================================================
// 场景 2: C++ (C++11 及以上)
// ===============================================================
#elif defined(__cplusplus)
    #include <type_traits>
    // 使用 std::remove_cv_t 去除 const/volatile
    #define CROSS_TYPE_UNQUAL(expr) std::remove_cv_t<decltype(expr)>

// ===============================================================
// 场景 3: GCC / Clang (非 C23 模式)
// 这里是难点：旧版 GCC 没有内置去除 const 的 typeof。
// 我们可以利用算术运算的一个特性：(expr) + 0 通常会进行类型提升并去除 const
// 但这不适用于结构体(struct)。
// ===============================================================
#elif defined(__GNUC__) || defined(__clang__)
    // 方案 A: 如果你只处理基本类型 (int, float)，可以用这个 trick
    // #define CROSS_TYPE_UNQUAL(expr) __typeof__((expr) + 0)
    
    // 方案 B: 诚实地使用 __typeof__，但警告用户不要传 const 数组
    // 这是目前在旧编译器上唯一的妥协
    #define CROSS_TYPE_UNQUAL(expr) __typeof__(expr)

// ===============================================================
// 场景 4: 旧版 MSVC 或其他不支持的编译器
// ===============================================================
#else
    #error "Current compiler implies strict standard C without typeof support. Please upgrade."
#endif

// 自动推导版本 (依赖 C23 或 C++)
#define FP_Map_Auto(VAR_NAME, SRC_ARR, DEST_ARR, LEN, EXPR) \
do { \
    CROSS_TYPE_UNQUAL((SRC_ARR)[0]) const* _fp_src = (SRC_ARR); \
    CROSS_TYPE_UNQUAL((DEST_ARR)[0])* _fp_dest = (DEST_ARR); \
    /* 注意：这里声明 VAR_NAME 时去除了 const，防止源是 const 时无法赋值 */ \
    CROSS_TYPE_UNQUAL((SRC_ARR)[0]) VAR_NAME; \
    const size_t _fp_len = (LEN); \
    for (size_t _i = 0; _i < _fp_len; ++_i) { \
        VAR_NAME = _fp_src[_i]; \
        _fp_dest[_i] = (EXPR); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // WCN_FP_UNSAFE_H
