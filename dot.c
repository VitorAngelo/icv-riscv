#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int32_t dot_product_int8(const int8_t *a, const int8_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

int32_t dot_product_int32(const int32_t *a, const int32_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

float dot_product_float32(const float *a, const float *b, int len)
{
    float ret = 0.0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

void core_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    struct timespec t1a, t2a, t3a;
    struct timespec t1b, t2b, t3b;
    long long ns1, ns2, ns3;
    double macs1, macs2, macs3;
    int i, rseed, core_id, klen, runs;

    if ((argc < 3) || (sscanf(argv[1], "%d", &core_id) != 1) || (sscanf(argv[2], "%d", &klen) != 1)) {
        fprintf(stderr, "%s core_num vec_len(kb) [rseed] [runs]\n", argv[0]);
        exit(0);
    }
    if ((argc < 4) || (sscanf(argv[3], "%d", &rseed) != 1)) {
        rseed = 0;
    }
    runs = 1;
    if (argc > 4) {
        if ((sscanf(argv[4], "%d", &runs) == 1) && (runs < 1)) {
            runs = 1;
        }
    }
    fprintf(stderr, "core_id=%d klen=%d rseed=%d\n", core_id, klen, rseed);
    core_affinity(core_id);
    klen *= 1024;
    srand(rseed);

    int8_t *ai8 = aligned_alloc(64, klen);
    int8_t *bi8 = aligned_alloc(64, klen);
    int32_t *ai32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t *bi32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t doti8 = 0, doti32 = 0;
    float *af = aligned_alloc(64, klen * sizeof(float));
    float *bf = aligned_alloc(64, klen * sizeof(float));
    float dotf = 0.0;

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

    if (runs == 1) doti8 = dot_product_int8(ai8, bi8, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1a);
    for (i = 0; i < runs; i++) {
        doti8 = dot_product_int8(ai8, bi8, klen);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1b);

    if (runs == 1) doti32 = dot_product_int32(ai32, bi32, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2a);
    for (i = 0; i < runs; i++) {
        doti32 = dot_product_int32(ai32, bi32, klen);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2b);
    
    if (runs == 1) dotf = dot_product_float32(af, bf, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t3a);
    for (i = 0; i < runs; i++) {
        dotf = dot_product_float32(af, bf, klen);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t3b);

    ns1 = ((t1b.tv_sec - t1a.tv_sec) * 1000000000LL + (t1b.tv_nsec - t1a.tv_nsec)) / (long long) runs;
    ns2 = ((t2b.tv_sec - t2a.tv_sec) * 1000000000LL + (t2b.tv_nsec - t2a.tv_nsec)) / (long long) runs;
    ns3 = ((t3b.tv_sec - t3a.tv_sec) * 1000000000LL + (t3b.tv_nsec - t3a.tv_nsec)) / (long long) runs;
    macs1 = (double)klen / ((double)ns1 * 1.0e-9);
    macs2 = (double)klen / ((double)ns2 * 1.0e-9);
    macs3 = (double)klen / ((double)ns3 * 1.0e-9);
    printf("core id ; MACS ; int8 (ns); int32 (ns) ; f32 (ns) ; int8 (MACs/s) ; int32 (MACs/s) ; f32 (MACs/s)\n");
    printf("%d ; %d ; %lld ; %lld ; %lld ; %.3e ; %.3e ; %.3e\n", core_id, klen, ns1, ns2, ns3, macs1, macs2, macs3);

    fprintf(stderr, "%d %d %f\n", (int)doti8, doti32, dotf);
    free(ai8); free(bi8);
    free(ai32); free(bi32);
    free(af); free(bf);
    return 0;
}
