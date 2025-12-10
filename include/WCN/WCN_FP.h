
#ifndef WCN_FP_H
#define WCN_FP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> /* for size_t */
#include <assert.h> /* for assert */

/**
 * 工业级 C FP 宏库设计规范：
 * 1. 所有的源数组 (SRC) 必须传入长度 (LEN)。
 * 2. 所有的目标数组 (DEST) 必须由用户预先分配好内存（栈或堆）。
 * 3. 宏内部使用 const 修饰源数据，保证不被篡改。
 */

// ==========================================
// 1. ForEach
// ==========================================
// 用法: FP_ForEach(int, x, arr, len, { printf("%d ", x); });
#define FP_ForEach(TYPE, VAR_NAME, SRC_ARR, LEN, CODE_BODY) \
do { \
    const TYPE* _fp_src = (SRC_ARR); \
    size_t _fp_len = (LEN); \
    assert(_fp_src != NULL && "Source array cannot be NULL"); \
    for (size_t _i = 0; _i < _fp_len; ++_i) { \
        const TYPE VAR_NAME = _fp_src[_i]; \
        CODE_BODY \
    } \
} while(0)

// ==========================================
// 2. Map
// ==========================================
// 将映射结果存入 DEST_ARR。DEST_ARR 大小必须 >= LEN。
// 用法: FP_Map(int, x, src, dest, len, (x * x));
#define FP_Map(TYPE, VAR_NAME, SRC_ARR, DEST_ARR, LEN, EXPR) \
do { \
    const TYPE* _fp_src = (SRC_ARR); \
    TYPE* _fp_dest = (DEST_ARR); \
    size_t _fp_len = (LEN); \
    assert(_fp_src != NULL && "Source array cannot be NULL"); \
    assert(_fp_dest != NULL && "Dest array cannot be NULL"); \
    for (size_t _i = 0; _i < _fp_len; ++_i) { \
        const TYPE VAR_NAME = _fp_src[_i]; \
        _fp_dest[_i] = (EXPR); \
    } \
} while(0)

// ==========================================
// 3. Filter
// ==========================================
// 将符合 CONDITION 的元素存入 DEST_ARR。
// OUT_LEN_PTR 是一个 size_t*，用于返回筛选后的数量。
// 注意：DEST_ARR 最好分配为与 SRC_ARR 等大，以防全选。
// 用法: FP_Filter(int, x, src, dest, len, (x > 10), &out_len);
#define FP_Filter(TYPE, VAR_NAME, SRC_ARR, DEST_ARR, LEN, CONDITION, OUT_LEN_PTR) \
do { \
    const TYPE* _fp_src = (SRC_ARR); \
    TYPE* _fp_dest = (DEST_ARR); \
    size_t _fp_len = (LEN); \
    size_t _fp_count = 0; \
    assert(_fp_src != NULL && "Source array cannot be NULL"); \
    assert(_fp_dest != NULL && "Dest array cannot be NULL"); \
    assert(OUT_LEN_PTR != NULL && "Output length pointer cannot be NULL"); \
    for (size_t _i = 0; _i < _fp_len; ++_i) { \
        TYPE VAR_NAME = _fp_src[_i]; \
        if (CONDITION) { \
            _fp_dest[_fp_count++] = VAR_NAME; \
        } \
    } \
    *(OUT_LEN_PTR) = _fp_count; \
} while(0)

// ==========================================
// 4. Reduce
// ==========================================
// 归约。ACC_VAR 是累加器变量名，INIT_VAL 是初始值。
// 结果会保留在 ACC_VAR 中 (注意 ACC_VAR 必须在宏外部定义)。
// 用法: int sum; FP_Reduce(int, x, sum, src, len, 0, (sum + x));
#define FP_Reduce(TYPE, VAR_NAME, ACC_VAR, SRC_ARR, LEN, INIT_VAL, EXPR) \
do { \
    const TYPE* _fp_src = (SRC_ARR); \
    size_t _fp_len = (LEN); \
    assert(_fp_src != NULL && "Source array cannot be NULL"); \
    ACC_VAR = (INIT_VAL); \
    for (size_t _i = 0; _i < _fp_len; ++_i) { \
        const TYPE VAR_NAME = _fp_src[_i]; \
        ACC_VAR = (EXPR); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif

#endif // WCN_FP_H
