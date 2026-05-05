#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define COMPRIMENTO (16 * 1024)

/*
 * vle8.v      : carrega int8         (e8, m1)
 * vsext.vf4   : sign-extend          int8 -> int32 em v8:v11 e v12:v15
 * vmul.vv     : multiplica int32     v8:v11 *= v12:v15
 * vredsum.vs  : redução escalar      v4[0] += sum(v8:v11)
 *
 * reusa t0 para trocar e8,m1 -> e32,m4 sem alterar VL.
 */
int32_t dot_product_int8(const int8_t *pointer1, const int8_t *pointer2, int comprimento)
{
    int32_t resultado = 0;
    asm volatile(
        "vmv.v.i v4, 0\n"                               // zera v4 (acumulador)
        "1:\n"                                          // inicia loop
        "vsetvli t0, %[comprimento], e8, m1, ta, ma\n"  // t0 = VL da iteracao
        "vle8.v v0, (%[pointer1])\n"                    // carrega pointer1 em v0
        "vle8.v v1, (%[pointer2])\n"                    // '' '' pointer2 '' v1
        "vsetvli zero, t0, e32, m4, ta, ma\n"           // troca vtype para e32,m4 (reusa t0)
        "vsext.vf4 v8, v0\n"                            // v8:v11 = int32(pointer1)
        "vsext.vf4 v12, v1\n"                           // v12:v15 = int32(pointer2)
        "vmul.vv v8, v8, v12\n"                         // v8:v11 = p1[i] * p2[i]
        "vredsum.vs v4, v8, v4\n"                       // v4[0] += sum(v8:v11)
        "add %[pointer1], %[pointer1], t0\n"            // avanca ponteiro 1
        "add %[pointer2], %[pointer2], t0\n"            // '' '' 2
        "sub %[comprimento], %[comprimento], t0\n"      // tira comprimento processado
        "bnez %[comprimento], 1b\n"                     // volta para '1' se ainda resta
        "vmv.x.s %[resultado], v4\n"                    // copia resultado p/ C
        : [resultado] "=r" (resultado), [pointer1] "+r" (pointer1), [pointer2] "+r" (pointer2), [comprimento] "+r" (comprimento)
        :
        : "v0", "v1", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "t0", "memory"
    );
    return resultado;
}

/*
 * vle32.v    : carrega int32         (e32, m1)
 * vmul.vv    : multiplica int32      v2[i] = v0[i] * v1[i]
 * vredsum.vs : redução escalar       v4[0] += sum(v2[i])
 * vmv.x.s    : extrai v4[0] gpr
 */
int32_t dot_product_int32(const int32_t *pointer1, const int32_t *pointer2, int comprimento)
{
    int32_t resultado = 0;
    asm volatile(
        "vmv.v.i v4, 0\n"                               // zera v4 (acumulador)
        "1:\n"                                          // inicia loop
        "vsetvli t0, %[comprimento], e32, m1, ta, ma\n" // t0 = VL da iteracao
        "vle32.v v0, (%[pointer1])\n"                   // carrega pointer1 em v0
        "vle32.v v1, (%[pointer2])\n"                   // '' '' pointer2 '' v1
        "vmul.vv v2, v0, v1\n"                          // v2[i] = v0[i] * v1[i]
        "vredsum.vs v4, v2, v4\n"                       // v4[0] += sum(v2[i])
        "slli t1, t0, 2\n"                              // t1 = t0 * 4 (byte offset)
        "add %[pointer1], %[pointer1], t1\n"            // avanca ponteiro 1
        "add %[pointer2], %[pointer2], t1\n"            // '' '' 2
        "sub %[comprimento], %[comprimento], t0\n"      // tira comprimento processado
        "bnez %[comprimento], 1b\n"                     // volta para '1' se ainda resta
        "vmv.x.s %[resultado], v4\n"                    // copia resultado p/ C
        : [resultado] "=r" (resultado), [pointer1] "+r" (pointer1), [pointer2] "+r" (pointer2), [comprimento] "+r" (comprimento)
        :
        : "v0", "v1", "v2", "v4", "t0", "t1", "memory"
    );
    return resultado;
}

/*
 * vle32.v      : carrega float32     (e32, m1)
 * vfmul.vv     : multiplica float    v2[i] = v0[i] * v1[i]
 * vfredusum.vs : redução escalar     v4[0] += sum(v2[i])
 * vfmv.f.s     : extrai v4[0] fpr
 */
float dot_product_float32(const float *pointer1, const float *pointer2, int comprimento)
{
    float resultado = 0.0f;
    asm volatile(
        "vmv.v.i v4, 0\n"                               // zera v4 
        "1:\n"                                          // inicia loop
        "vsetvli t0, %[comprimento], e32, m1, ta, ma\n" // t0 = VL da iteracao
        "vle32.v v0, (%[pointer1])\n"                   // carrega pointer1 em v0
        "vle32.v v1, (%[pointer2])\n"                   // '' '' pointer2 '' v1
        "vfmul.vv v2, v0, v1\n"                         // v2[i] = v0[i] * v1[i]
        "vfredusum.vs v4, v2, v4\n"                     // v4[0] += sum(v2[i])
        "slli t1, t0, 2\n"                              // t1 = t0 * 4 (byte offset)
        "add %[pointer1], %[pointer1], t1\n"            // avanca ponteiro 1
        "add %[pointer2], %[pointer2], t1\n"            // '' '' 2
        "sub %[comprimento], %[comprimento], t0\n"      // tira comprimento processado
        "bnez %[comprimento], 1b\n"                     // volta para '1' se ainda resta
        "vfmv.f.s %[resultado], v4\n"                   // copia resultado p/ C
        : [resultado] "=f" (resultado), [pointer1] "+r" (pointer1), [pointer2] "+r" (pointer2), [comprimento] "+r" (comprimento)
        :
        : "v0", "v1", "v2", "v4", "t0", "t1", "memory"
    );
    return resultado;
}

int main(int argc, char *argv[])
{
    int8_t  *ai8  = malloc(COMPRIMENTO);
    int8_t  *bi8  = malloc(COMPRIMENTO);
    int32_t *ai32 = malloc(COMPRIMENTO * sizeof(int32_t));
    int32_t *bi32 = malloc(COMPRIMENTO * sizeof(int32_t));
    int32_t doti8, doti32;
    float   *af = malloc(COMPRIMENTO * sizeof(float));
    float   *bf = malloc(COMPRIMENTO * sizeof(float));
    float   dotf;
    int i;

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) {
        perror("malloc");
        return 1;
    }

    srand(0); //time(NULL));
    for (i = 0; i < COMPRIMENTO; i++) {
        ai8[i]  = rand();
        bi8[i]  = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i]   = ai8[i];
        bf[i]   = bi8[i];
    }

    doti8  = dot_product_int8   (ai8,  bi8,  COMPRIMENTO);
    doti32 = dot_product_int32  (ai32, bi32, COMPRIMENTO);
    dotf   = dot_product_float32(af,   bf,   COMPRIMENTO);

    printf("%d %d %f\n", (int)doti8, doti32, dotf);

    free(ai8);  free(bi8);
    free(ai32); free(bi32);
    free(af);   free(bf);

    return 0;
}
