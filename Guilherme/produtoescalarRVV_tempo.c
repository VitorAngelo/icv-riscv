#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void core_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}

/*
 * vle8.v      : carrega int8         (e8, m1)
 * vsext.vf4   : sign-extend          int8 -> int32 em v8:v11 e v12:v15
 * vmul.vv     : multiplica int32     v8:v11 *= v12:v15
 * vredsum.vs  : redução escalar      v4[0] += sum(v8:v11)
 *
 * reusa t0 para trocar e8,m1 -> e32,m4 sem alterar VL.
 *
 * passo: 32 elementos int8/iteração, 32 bytes/iteração
 */
int32_t dot_product_int8(const int8_t *pointer1, const int8_t *pointer2, int comprimento)
{
    int32_t resultado = 0;
    asm volatile(
        "vsetvli t1, zero, e32, m1, ta, ma\n"
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
        "vsetvli t1, zero, e32, m1, ta, ma\n"           // reconfigura e32, m1
        "vmv.x.s %[resultado], v4\n"                    // copia resultado p/ C
        : [resultado] "=r" (resultado), [pointer1] "+r" (pointer1), [pointer2] "+r" (pointer2), [comprimento] "+r" (comprimento)
        :
        : "v0", "v1", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "t0", "t1", "memory"
    );
    return resultado;
}

/*
 * vle32.v    : carrega int32         (e32, m1)
 * vmul.vv    : multiplica int32      v2[i] = v0[i] * v1[i]
 * vredsum.vs : redução escalar       v4[0] += sum(v2[i])
 * vmv.x.s    : extrai v4[0] gpr
 *
 * VLEN = 256 bits (validado na CPU SpacemiT K1)
 * passo: 8 elementos int32/iteração, 32 bytes/iteração
 */
int32_t dot_product_int32(const int32_t *pointer1, const int32_t *pointer2, int comprimento)
{
    int32_t resultado = 0;
    asm volatile(
        "vsetvli t1, zero, e32, m1, ta, ma\n"
        "vmv.v.i v4, 0\n"                               // zera v4 (acumulador)
        "1:\n"                                          // inicia loop
        "vsetvli t0, %[comprimento], e32, m1, ta, ma\n" // t0 = VL da iteracao
        "vle32.v v0, (%[pointer1])\n"                   // carrega pointer1 em v0
        "vle32.v v1, (%[pointer2])\n"                   // '' '' pointer2 '' v1
        "vmul.vv v2, v0, v1\n"                          // v2[i] = v0[i] * v1[i]
        "vredsum.vs v4, v2, v4\n"                       // v4[0] += sum(v2[i])
        "slli t1, t0, 2\n"                              // t1 = t0 * 4 avança
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
 *
 * VLEN = 256 bits (validado na CPU SpacemiT K1)
 * passo: 8 elementos float32/iteração, 32 bytes/iteração
 */
float dot_product_float32(const float *pointer1, const float *pointer2, int comprimento)
{
    float resultado = 0.0f;
    asm volatile(
        "vsetvli t1, zero, e32, m1, ta, ma\n"
        "vmv.v.i v4, 0\n"                               // zera v4 
        "1:\n"                                          // inicia loop
        "vsetvli t0, %[comprimento], e32, m1, ta, ma\n" // t0 = VL da iteracao
        "vle32.v v0, (%[pointer1])\n"                   // carrega pointer1 em v0
        "vle32.v v1, (%[pointer2])\n"                   // '' '' pointer2 '' v1
        "vfmul.vv v2, v0, v1\n"                         // v2[i] = v0[i] * v1[i]
        "vfredusum.vs v4, v2, v4\n"                     // v4[0] += sum(v2[i])
        "slli t1, t0, 2\n"                              // t1 = t0 * 4 avança
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
    struct timespec t1, t2, t3, t4;
    long long ns2, ns3, ns4;
    double macs2, macs3, macs4;
    int i, rseed, core_id, klen;

    if ((argc < 3) || (sscanf(argv[1], "%d", &core_id) != 1) || (sscanf(argv[2], "%d", &klen) != 1)) {
        fprintf(stderr, "%s core_num vec_len(kb)\n", argv[0]);
        exit(0);
    }
    if ((argc != 4) || (sscanf(argv[3], "%d", &rseed) != 1)) {
        rseed = 0;
    }
    fprintf(stderr, "core_id=%d klen=%d rseed=%d\n", core_id, klen, rseed);
    core_affinity(core_id);
    klen *= 1024;
    srand(rseed);

    int8_t *ai8 = aligned_alloc(64, klen);
    int8_t *bi8 = aligned_alloc(64, klen);
    int32_t *ai32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t *bi32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t doti8, doti32;
    float *af = aligned_alloc(64, klen * sizeof(float));
    float *bf = aligned_alloc(64, klen * sizeof(float));
    float dotf;

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) {
        perror("malloc");
        return 1;
    }

    for (i = 0; i < klen; i++) {
        ai8[i] = rand();
        bi8[i] = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i] = ai8[i];
        bf[i] = bi8[i];
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    doti8 = dot_product_int8(ai8, bi8, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    doti32 = dot_product_int32(ai32, bi32, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t3);
    dotf = dot_product_float32(af, bf, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t4);

    ns2 = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
    ns3 = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);
    ns4 = (t4.tv_sec - t3.tv_sec) * 1000000000LL + (t4.tv_nsec - t3.tv_nsec);
    macs2 = (double)klen / ((double)ns2 * 1.0e-9);
    macs3 = (double)klen / ((double)ns3 * 1.0e-9);
    macs4 = (double)klen / ((double)ns4 * 1.0e-9);
    printf("core id ; MACS ; int8 (ns); int32 (ns) ; f32 (ns) ; int8 (MACs/s) ; int32 (MACs/s) ; f32 (MACs/s)\n");
    printf("%d ; %d ; %lld ; %lld ; %lld ; %.3e ; %.3e ; %.3e\n", core_id, klen, ns2, ns3, ns4, macs2, macs3, macs4);

    fprintf(stderr, "%d %d %f\n", (int)doti8, doti32, dotf);
    free(ai8); free(bi8);
    free(ai32); free(bi32);
    free(af); free(bf);
    return 0;
}
