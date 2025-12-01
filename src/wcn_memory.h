#ifndef WCN_MEMORY_H
#define WCN_MEMORY_H

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// 内存统计和监控功能
#ifdef WCN_DEBUG_MEMORY

extern size_t g_wcn_total_memory_allocated;
extern size_t g_wcn_total_memory_freed;

static inline void* wcn_debug_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) {
        g_wcn_total_memory_allocated += size;
    }
    return ptr;
}

static inline void* wcn_debug_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (ptr) {
        g_wcn_total_memory_allocated += num * size;
    }
    return ptr;
}

static inline void* wcn_debug_realloc(void* ptr, size_t size) {
    // 注意：这里简化处理，实际应该跟踪旧大小
    void* new_ptr = realloc(ptr, size);
    if (new_ptr) {
        // 简化：假设每次realloc都是净增加
        g_wcn_total_memory_allocated += size;
    }
    return new_ptr;
}

static inline void wcn_debug_free(void* ptr) {
    // 注意：这里简化处理，实际应该跟踪释放的大小
    if (ptr) {
        g_wcn_total_memory_freed += 1; // 简化：只计数，不跟踪大小
    }
    free(ptr);
}

#define malloc(size) wcn_debug_malloc(size)
#define calloc(num, size) wcn_debug_calloc(num, size)
#define realloc(ptr, size) wcn_debug_realloc(ptr, size)
#define free(ptr) wcn_debug_free(ptr)

void wcn_print_memory_stats(void);

#else

// 非调试模式使用标准函数
#define wcn_print_memory_stats() ((void)0)

#endif

// 内存管理最佳实践工具函数
static inline void wcn_safe_free(void** ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

#ifdef __cplusplus
}
#endif

#endif // WCN_MEMORY_H