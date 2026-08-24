#include <stdio.h>
#include <stdint.h>

void conv1d(const int8_t *feature, const int8_t *weight, int32_t *output) {
    asm volatile(
        "vsetvli    t0, zero, e8, m1        \n\t"
        "vle8.v     v0, (%[A])              \n\t"
        "addi       %[A], %[A], 32          \n\t"
        "vle8.v     v1, (%[A])              \n\t"
        "addi       %[A], %[A], 32          \n\t"
        "vle8.v     v8, (%[B])              \n\t"
        "addi       %[B], %[B], 32          \n\t"
        "smt.vmadot     v16, v0, v8         \n\t"
        "smt.vmadot1    v16, v0, v8         \n\t"
        "smt.vmadot2    v16, v0, v8         \n\t"
        "vsetvli    t0, zero, e32, m2       \n\t"
        "vse32.v    v16, (%[C])             \n\t"
        : [A] "+r"(feature), [B] "+r"(weight), [C] "+r"(output)
        :
        : "t0", "v0", "v1", "v8", "v16", "v17", "memory"
    );
}

int main() {
    int8_t A[96] = {0};  // 3 blocos de 32 bytes
    // Preenche A com valores conhecidos (mesmo do exemplo da spec, adaptado)
    for (int i = 0; i < 96; i++) A[i] = i % 8;  // apenas para teste
    int8_t B[32] = {0};
    for (int i = 0; i < 32; i++) B[i] = i % 4;  // pesos

    int32_t C[16] = {0};

    conv1d(A, B, C);

    printf("Resultado C:\n");
    for (int i = 0; i < 16; i++) printf("%d ", C[i]);
    printf("\n");

    return 0;
}