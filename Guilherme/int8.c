#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "int8.h"

#include "conv1_w_int8.h"
#include "conv2_w_int8.h"
#include "conv3_w_int8.h"
#include "dense1_w_int8.h"
#include "output_w_int8.h"

#include "conv1_b_int32.h"
#include "conv2_b_int32.h"
#include "conv3_b_int32.h"
#include "dense1_b_int32.h"
#include "output_b_int32.h"

#include "conv1_escala.h"
#include "conv2_escala.h"
#include "conv3_escala.h"
#include "dense1_escala.h"
#include "output_escala.h"

#include "conv1_escala_ativ.h"
#include "conv2_escala_ativ.h"
#include "conv3_escala_ativ.h"
#include "dense1_escala_ativ.h"

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

static double diff_us(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) * 1e6 +
           (fim.tv_nsec - inicio.tv_nsec) / 1e3;
}

static void convolucao_2d_int8(const uint8_t *entrada, int H, int W, int C_in,
                               const int8_t *kernel, const int32_t *vies,
                               int K, int C_out, int32_t *saida) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int f = 0; f < C_out; f++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int32_t soma = vies[f];
                for (int c = 0; c < C_in; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            int idx_ent = c * (H * W) + (i + ki) * W + (j + kj);
                            int idx_kern = ((ki * K + kj) * C_in + c) * C_out + f;
                            soma += (int32_t)entrada[idx_ent] * kernel[idx_kern];
                        }
                    }
                }
                saida[f * (H_out * W_out) + i * W_out + j] = soma;
            }
        }
    }
}

static void camada_densa_int8(const uint8_t *x, int dim_entrada,
                              const int8_t *W, const int32_t *b,
                              int dim_saida, int32_t *saida) {
    for (int j = 0; j < dim_saida; j++) {
        int32_t soma = b[j];
        for (int i = 0; i < dim_entrada; i++) {
            soma += (int32_t)x[i] * W[i * dim_saida + j];
        }
        saida[j] = soma;
    }
}

static void relu_int32(int32_t *dados, int n) {
    for (int i = 0; i < n; i++)
        if (dados[i] < 0) dados[i] = 0;
}

static void average_pooling_2x2_int32(const int32_t *entrada, int H, int W, int C,
                                      int32_t *saida) {
    int H_out = H / 2;
    int W_out = W / 2;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int32_t soma = 0;
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++)
                        soma += entrada[c * (H * W) + (2*i + pi) * W + (2*j + pj)];
                saida[c * (H_out * W_out) + i * W_out + j] = (soma + 2) / 4;
            }
        }
    }
}

static void requantizar_por_canal(const int32_t *entrada, int n,
                                  const float *escala_peso, int n_canais,
                                  float escala_entrada, float escala_ativ_saida,
                                  uint8_t *saida) {
    int elementos_por_canal = n / n_canais;
    for (int i = 0; i < n; i++) {
        int canal = i / elementos_por_canal;
        float fator = (escala_entrada * escala_peso[canal]) / escala_ativ_saida;
        float val = (float)entrada[i] * fator;
        int32_t arred = (int32_t)(val + 0.5f);
        if (arred < 0) arred = 0;
        if (arred > 255) arred = 255;
        saida[i] = (uint8_t)arred;
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

void inferencia_int8(const uint8_t *img_uint8, float *probs,
                     TemposCamadasInt8 *tempos, int aplicar_softmax) {
    uint8_t img_u8[IMG_H * IMG_W];
    int32_t conv1_out[CONV1_F * CONV1_OUT_H * CONV1_OUT_W];
    int32_t pool1_i32[CONV1_F * POOL1_H * POOL1_W];
    uint8_t pool1_u8[CONV1_F * POOL1_H * POOL1_W];

    int32_t conv2_out[CONV2_F * CONV2_OUT_H * CONV2_OUT_W];
    int32_t pool2_i32[CONV2_F * POOL2_H * POOL2_W];
    uint8_t pool2_u8[CONV2_F * POOL2_H * POOL2_W];

    int32_t conv3_out[CONV3_F * CONV3_OUT_H * CONV3_OUT_W];
    uint8_t conv3_u8[FLAT_SIZE];

    int32_t dense1_out[DENSE1_OUT];
    uint8_t dense1_u8[DENSE1_OUT];

    int32_t output_i32[OUTPUT_SIZE];
    float output_float[OUTPUT_SIZE];

    struct timespec t1, t2;

    memcpy(img_u8, img_uint8, IMG_H * IMG_W);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(img_u8, IMG_H, IMG_W, IMG_C,
                       conv1_w_int8, conv1_b_int32, CONV1_K, CONV1_F, conv1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv1_us = diff_us(t1, t2);

    relu_int32(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);
    average_pooling_2x2_int32(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_i32);
    requantizar_por_canal(pool1_i32, CONV1_F * POOL1_H * POOL1_W,
                          conv1_escala, CONV1_F,
                          1.0f/255.0f, CONV1_ESCALA_ATIV_ESCALA, pool1_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(pool1_u8, POOL1_H, POOL1_W, CONV1_F,
                       conv2_w_int8, conv2_b_int32, CONV2_K, CONV2_F, conv2_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv2_us = diff_us(t1, t2);

    relu_int32(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    average_pooling_2x2_int32(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_i32);
    requantizar_por_canal(pool2_i32, CONV2_F * POOL2_H * POOL2_W,
                          conv2_escala, CONV2_F,
                          CONV1_ESCALA_ATIV_ESCALA, CONV2_ESCALA_ATIV_ESCALA, pool2_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(pool2_u8, POOL2_H, POOL2_W, CONV2_F,
                       conv3_w_int8, conv3_b_int32, CONV3_K, CONV3_F, conv3_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv3_us = diff_us(t1, t2);

    relu_int32(conv3_out, CONV3_F * CONV3_OUT_H * CONV3_OUT_W);
    requantizar_por_canal(conv3_out, FLAT_SIZE,
                          conv3_escala, CONV3_F,
                          CONV2_ESCALA_ATIV_ESCALA, CONV3_ESCALA_ATIV_ESCALA, conv3_u8);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    camada_densa_int8(conv3_u8, FLAT_SIZE, dense1_w_int8, dense1_b_int32,
                      DENSE1_OUT, dense1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->dense1_us = diff_us(t1, t2);

    relu_int32(dense1_out, DENSE1_OUT);
    requantizar_por_canal(dense1_out, DENSE1_OUT,
                          dense1_escala, DENSE1_OUT,
                          CONV3_ESCALA_ATIV_ESCALA, DENSE1_ESCALA_ATIV_ESCALA, dense1_u8);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

    camada_densa_int8(dense1_u8, DENSE1_OUT, output_w_int8, output_b_int32,
                      OUTPUT_SIZE, output_i32);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->output_us = diff_us(t1, t2);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_float[i] = (float)output_i32[i] *
                          (DENSE1_ESCALA_ATIV_ESCALA * output_escala[i]);
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
