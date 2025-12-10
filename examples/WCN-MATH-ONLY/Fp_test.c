#include <stdio.h>
#include <stdlib.h>
#include "WCN/WCN_FP.h"

int main() {
    // -------------------------------------------------
    // 场景 1: 堆内存 (Heap) - 工业级常用场景
    // -------------------------------------------------
    const int len = 5;
    int* src = (int*)malloc(len * sizeof(int));
    for(int i=0; i<len; i++) src[i] = i + 1; // 1, 2, 3, 4, 5

    // Map: 将 src 的平方存入 dest
    int* dest_map = (int*)malloc(len * sizeof(int));
    
    // FP_Map(类型, 迭代变量名, 源地址, 目标地址, 长度, 表达式)
    FP_Map(int, v, src, dest_map, len, v * v);

    printf("Map Result (Heap): ");
    FP_ForEach(int, v, dest_map, len, { printf("%d ", v); });
    printf("\n");

    // -------------------------------------------------
    // 场景 2: 栈内存 (Stack) - 嵌入式常用场景
    // -------------------------------------------------
    const int stack_arr[] = {10, 20, 30, 40, 50};
    const int stack_len = 5;
    
    // Filter: 找出大于 25 的数
    // 注意：目标数组大小分配为最大可能长度（即源长度），防止溢出
    int dest_filter[5]; 
    size_t filter_count = 0;

    // FP_Filter(类型, 迭代变量名, 源, 目标, 长度, 条件, 输出数量指针)
    FP_Filter(int, v, stack_arr, dest_filter, stack_len, v > 25, &filter_count);

    printf("Filter Result (Stack): ");
    // 注意这里使用 filter_count 而不是 stack_len
    FP_ForEach(int, v, dest_filter, filter_count, { printf("%d ", v); });
    printf("\n");

    // -------------------------------------------------
    // 场景 3: Reduce
    // -------------------------------------------------
    int total_sum;
    // FP_Reduce(类型, 迭代变量名, 累加器变量, 源, 长度, 初始值, 累加表达式)
    FP_Reduce(int, v, total_sum, src, len, 0, total_sum + v);
    
    printf("Reduce Sum: %d\n", total_sum);

    // 清理堆内存
    free(src);
    free(dest_map);

    return 0;
}