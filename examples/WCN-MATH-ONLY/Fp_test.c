#include <stdio.h>
#include <WCN/WCN_FP.h>
#include <string.h>

typedef struct {
    int a;
} A;

int main() {
    const int arr[5] = {
        0,1,2,3,4
    };
    // A new_arr[5] = Fp_Map(arr) {
    //     Fp_Send(new_arr, (A){
    //       .a = $value + $index
    //     });
    // }
    Fp_MapLet(A, new_arr, arr, {
        Fp_MapLet_Send({
            .a = _value + _index
        });
    })
    // int sum = Fp_Reduce(int, arr, 0, {
    //     Fp_Reduce_Send(sum, $acc + $value);
    // });
    Fp_ReduceLet(A, sum, arr, { .a = 0 }, {
        Fp_ReduceLet_Send({
            .a = _acc.a + _value
        });
    })

    Fp_FilterLet(int, _filtered_arr, arr, _value >= 2);
    int filtered_arr[_filtered_arr_size];
    memcpy(filtered_arr, _filtered_arr, sizeof(int) * _filtered_arr_size);

    printf("Mapped values:\n");
    Fp_ForIn(new_arr) { printf("%d ", _value.a); }
    printf("\nSum: %d\n", sum.a);
    Fp_ForIn(filtered_arr) {
        printf("index: %d \n", _index);
        printf("f-value: %d \n", _value);
    }
}