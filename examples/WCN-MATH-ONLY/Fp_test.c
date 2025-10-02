#include <stdio.h>
#include <WCN/WCN_FP.h>

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
            .a = $value + $index
        });
    })
    // int sum = Fp_Reduce(int, arr, 0, {
    //     Fp_Reduce_Send(sum, $acc + $value);
    // });
    Fp_ReduceLet(A, sum, arr, { .a = 0 }, {
        Fp_ReduceLet_Send({
            .a = $acc.a + $value
        });
    })

    Fp_FilterLet(int, filtered_arr, arr, $value > 2);

    printf("Mapped values:\n");
    Fp_ForIn(new_arr) { printf("%d ", $value.a); }
    printf("\nSum: %d\n", sum.a);
    Fp_ForIn(filtered_arr) {
        printf("f-value: %d \n", $value);
    }
}