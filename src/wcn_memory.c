#include "wcn_memory.h"

#ifdef WCN_DEBUG_MEMORY

size_t g_wcn_total_memory_allocated = 0;
size_t g_wcn_total_memory_freed = 0;

void wcn_print_memory_stats(void) {
    printf("=== WCN Memory Statistics ===\n");
    printf("Total Allocated: %zu bytes\n", g_wcn_total_memory_allocated);
    printf("Total Freed: %zu bytes\n", g_wcn_total_memory_freed);
    printf("Current Usage: %zu bytes\n", g_wcn_total_memory_allocated - g_wcn_total_memory_freed);
    printf("============================\n");
}

#endif