#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "int32.h"

#include "q32_conv1_w.h"
#include "q32_conv1_b.h"
#include "q32_conv2_w.h"
#include "q32_conv2_b.h"
#include "q32_conv3_w.h"
#include "q32_conv3_b.h"
#include "q32_dense1_w.h"
#include "q32_dense1_b.h"
#include "q32_output_w.h"
#include "q32_output_b.h"

#define IMG_H 32
#define IMG_W 32
#define IMG_C 1

#define CONV1_F 6
#define CONV1_K 5
#define CONV1_OUT_H (IMG_H - CONV1_K + 1)
#define CONV1_OUT_W (IMG_W - CONV1_K + 1)

#define POOL1_H (CONV1_OUT_H / 2)
#define POOL1_W (CONV1_OUT_W / 2)

#define CONV2_F 16
#define CONV2_K 5
#define CONV2_OUT_H (POOL1_H - CONV2_K + 1)
#define CONV2_OUT_W (POOL1_W - CONV2_K + 1)

#define POOL2_H (CONV2_OUT_H / 2)
#define POOL2_W (CONV2_OUT_W / 2)

#define CONV3_F 120
#define CONV3_K 5
#define CONV3_OUT_H (POOL2_H - CONV3_K + 1)
#define CONV3_OUT_W (POOL2_W - CONV3_K + 1)

#define FLAT_SIZE (CONV3_F * CONV3_OUT_H * CONV3_OUT_W)

#define DENSE1_OUT 84
#define OUTPUT_SIZE 10

#define ESCALA_Q16 65536.0f

static double diff_us(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) * 1e6 +
           (fim.tv_nsec - inicio.tv_nsec) / 1e3;
}

static void convolucao_2d_int32(const int32_t *entrada, int H, int W, int C_in,
                                const int32_t *kernel, const int32_t *vies,
                                int K, int C_out, int64_t *saida) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int f = 0; f < C_out; f++) {
        int64_t bias_shift = (int64_t)vies[f] << 16;  
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int64_t soma = bias_shift;
                for (int c = 0; c < C_in; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            int idx_ent = c * (H * W) + (i + ki) * W + (j + kj);
                            int idx_kern = ((ki * K + kj) * C_in + c) * C_out + f;
                            soma += (int64_t)entrada[idx_ent] * kernel[idx_kern];
                        }
                    }
                }
                saida[f * (H_out * W_out) + i * W_out + j] = soma;
            }
        }
    }
}

static void camada_densa_int32(const int32_t *x, int dim_entrada,
                               const int32_t *W, const int32_t *b,
                               int dim_saida, int64_t *saida) {
    for (int j = 0; j < dim_saida; j++) {
        int64_t soma = (int64_t)b[j] << 16;   
        for (int i = 0; i < dim_entrada; i++) {
            soma += (int64_t)x[i] * W[i * dim_saida + j];
        }
        saida[j] = soma;
    }
}

static void relu_int64(int64_t *dados, int n) {
    for (int i = 0; i < n; i++)
        if (dados[i] < 0) dados[i] = 0;
}

static void average_pooling_2x2_int64(const int64_t *entrada, int H, int W, int C,
                                      int64_t *saida) {
    int H_out = H >> 1;
    int W_out = W >> 1;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int64_t soma = 0;
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++)
                        soma += entrada[c * (H * W) + (2*i + pi) * W + (2*j + pj)];
                saida[c * (H_out * W_out) + i * W_out + j] = (soma + 2) >> 2;
            }
        }
    }
}

static void quantizar_para_int32(const int64_t *entrada, int n, int32_t *saida) {
    for (int i = 0; i < n; i++) {
        saida[i] = (int32_t)((entrada[i] + (1LL << 15)) >> 16);   
    }
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float soma = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        soma += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= soma;
}

void inferencia_int32(const int32_t *img_q32, float *probs,
                      TemposCamadasInt32 *tempos, int aplicar_softmax) {
    int64_t conv1_out[CONV1_F * CONV1_OUT_H * CONV1_OUT_W];
    int64_t pool1_i64[CONV1_F * POOL1_H * POOL1_W];
    int32_t pool1_q32[CONV1_F * POOL1_H * POOL1_W];

    int64_t conv2_out[CONV2_F * CONV2_OUT_H * CONV2_OUT_W];
    int64_t pool2_i64[CONV2_F * POOL2_H * POOL2_W];
    int32_t pool2_q32[CONV2_F * POOL2_H * POOL2_W];

    int64_t conv3_out[CONV3_F * CONV3_OUT_H * CONV3_OUT_W];
    int32_t conv3_q32[FLAT_SIZE];

    int64_t dense1_out[DENSE1_OUT];
    int32_t dense1_q32[DENSE1_OUT];

    int64_t output_i64[OUTPUT_SIZE];
    float output_float[OUTPUT_SIZE];

    struct timespec t1, t2;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int32(img_q32, IMG_H, IMG_W, IMG_C,
                        q32_conv1_w, q32_conv1_b, CONV1_K, CONV1_F, conv1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv1_us = diff_us(t1, t2);

    relu_int64(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);
    average_pooling_2x2_int64(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_i64);
    quantizar_para_int32(pool1_i64, CONV1_F * POOL1_H * POOL1_W, pool1_q32);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int32(pool1_q32, POOL1_H, POOL1_W, CONV1_F,
                        q32_conv2_w, q32_conv2_b, CONV2_K, CONV2_F, conv2_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv2_us = diff_us(t1, t2);

    relu_int64(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    average_pooling_2x2_int64(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_i64);
    quantizar_para_int32(pool2_i64, CONV2_F * POOL2_H * POOL2_W, pool2_q32);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int32(pool2_q32, POOL2_H, POOL2_W, CONV2_F,
                        q32_conv3_w, q32_conv3_b, CONV3_K, CONV3_F, conv3_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv3_us = diff_us(t1, t2);

    relu_int64(conv3_out, CONV3_F * CONV3_OUT_H * CONV3_OUT_W);
    quantizar_para_int32(conv3_out, FLAT_SIZE, conv3_q32);
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    camada_densa_int32(conv3_q32, FLAT_SIZE,
                       q32_dense1_w, q32_dense1_b, DENSE1_OUT, dense1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->dense1_us = diff_us(t1, t2);

    relu_int64(dense1_out, DENSE1_OUT);
    quantizar_para_int32(dense1_out, DENSE1_OUT, dense1_q32);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    camada_densa_int32(dense1_q32, DENSE1_OUT,
                       q32_output_w, q32_output_b, OUTPUT_SIZE, output_i64);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->output_us = diff_us(t1, t2);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_float[i] = (float)output_i64[i] / (ESCALA_Q16 * ESCALA_Q16);
    }

    if (aplicar_softmax) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        softmax(output_float, OUTPUT_SIZE);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        tempos->softmax_us = diff_us(t1, t2);
    } else {
        tempos->softmax_us = 0.0;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) probs[i] = output_float[i];
}
