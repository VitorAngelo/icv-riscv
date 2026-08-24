#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>  

#include "int8.h"

#include "conv1_w_int8.h"
#include "conv1_escala.h"
#include "conv1_b_float.h"
#include "conv2_w_int8.h"
#include "conv2_escala.h"
#include "conv2_b_float.h"
#include "conv3_w_int8.h"
#include "conv3_escala.h"
#include "conv3_b_float.h"
#include "dense1_w_int8.h"
#include "dense1_escala.h"
#include "dense1_b_float.h"
#include "output_w_int8.h"
#include "output_escala.h"
#include "output_b_float.h"
#include "dense1_escala_ativ.h" 
#include "conv3_escala_ativ.h"

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
    return (fim.tv_sec - inicio.tv_sec) * 1e6 + (fim.tv_nsec - inicio.tv_nsec) / 1e3;
}

static void convolucao_2d_int8(const float *entrada, int H, int W, int C_in,
                               const int8_t *kernel, const float *escala, const float *vies,
                               int K, int C_out, float *saida) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int f = 0; f < C_out; f++) {
        float esc = escala[f];
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                float soma = 0.0f;
                for (int c = 0; c < C_in; c++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            int idx_ent = c * (H * W) + (i + ki) * W + (j + kj);
                            int idx_kern = ((ki * K + kj) * C_in + c) * C_out + f;
                            float w = kernel[idx_kern] * esc;
                            soma += entrada[idx_ent] * w;
                        }
                    }
                }
                soma += vies[f];
                saida[f * (H_out * W_out) + i * W_out + j] = soma;
            }
        }
    }
}

static void average_pooling_2x2(const float *entrada, int H, int W, int C,
                                float *saida) {
    int H_out = H / 2;
    int W_out = W / 2;

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                float soma = 0.0f;
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++)
                        soma += entrada[c * (H * W) + (2*i + pi) * W + (2*j + pj)];
                saida[c * (H_out * W_out) + i * W_out + j] = soma * 0.25f;
            }
        }
    }
}

static void relu(float *dados, int n) {
    for (int i = 0; i < n; i++)
        if (dados[i] < 0.0f) dados[i] = 0.0f;
}

static void camada_densa_int8(const float *x, int dim_entrada,
                              const int8_t *W, const float *escala, const float *b,
                              int dim_saida, float *y) {
    for (int j = 0; j < dim_saida; j++) {
        float esc = escala[j];
        float soma = 0.0f;
        for (int i = 0; i < dim_entrada; i++) {
            float w = W[i * dim_saida + j] * esc;
            soma += x[i] * w;
        }
        y[j] = soma + b[j];
    }
}

static void camada_densa_ime(const float *x_float, int dim_entrada,
                             const int8_t *W, const float *escala_peso, const float *b,
                             int dim_saida, float escala_ativ_entrada, float *y) {
    // converte entrada float -> uint8 usando a escala de ativação
    uint8_t *x_u8 = malloc(dim_entrada * sizeof(uint8_t));
    for (int i = 0; i < dim_entrada; i++) {
        float v = x_float[i] / escala_ativ_entrada;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        x_u8[i] = (uint8_t)(v + 0.5f);
    }

    // processa em blocos de 4 saídas
    for (int jj = 0; jj < dim_saida; jj += 4) {
        // zera v2:v3 (acumulador) antes de iniciar os blocos de entrada
        asm volatile(
            "vsetivli t0, 16, e32, m2, ta, ma\n"
            "vmv.v.i v2, 0\n"
            :
            :
            : "t0", "v2", "v3", "memory"
        );

        // percorre a dimensão de entrada em blocos de 8
        for (int ii = 0; ii < dim_entrada; ii += 8) {
            uint8_t A[32] = {0};
            for (int k = 0; k < 8; k++) {
                int idx = ii + k;
                if (idx < dim_entrada) A[k] = x_u8[idx];
            }

            int8_t B[32] = {0};
            for (int q = 0; q < 4; q++) {
                int j = jj + q;
                if (j >= dim_saida) break;
                for (int k = 0; k < 8; k++) {
                    int idx = ii + k;
                    if (idx < dim_entrada) {
                        B[q * 8 + k] = W[idx * dim_saida + j];
                    }
                }
            }

            asm volatile(
                "li t1, 32\n"
                "vsetvli t0, t1, e8, m1, ta, ma\n"
                "vle8.v v0, (%[A])\n"
                "vle8.v v1, (%[B])\n"
                "smt.vmadotus v2, v0, v1\n"   // acumula em v2:v3
                :
                : [A] "r" (A), [B] "r" (B)
                : "t0", "t1", "v0", "v1", "v2", "v3", "memory"
            );
        }

        // extrai o resultado final (v2:v3) após todos os blocos de entrada
        int32_t C[16] = {0};
        asm volatile(
            "vsetivli t0, 16, e32, m2, ta, ma\n"
            "vse32.v v2, (%[C])\n"
            :
            : [C] "r" (C)
            : "t0", "v2", "v3", "memory"
        );

        // desquantiza e soma bias
        for (int q = 0; q < 4; q++) {
            int j = jj + q;
            if (j < dim_saida) {
                y[j] = (float)C[q] * (escala_ativ_entrada * escala_peso[j]) + b[j];
            }
        }
    }

    free(x_u8);
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
    float img[IMG_H * IMG_W];
    for (int i = 0; i < IMG_H * IMG_W; i++)
        img[i] = (float)img_uint8[i] / 255.0f;

    float conv1_out[CONV1_F * CONV1_OUT_H * CONV1_OUT_W];
    float pool1_out[CONV1_F * POOL1_H * POOL1_W];
    float conv2_out[CONV2_F * CONV2_OUT_H * CONV2_OUT_W];
    float pool2_out[CONV2_F * POOL2_H * POOL2_W];
    float conv3_out[CONV3_F * CONV3_OUT_H * CONV3_OUT_W];
    float dense1_out[DENSE1_OUT];
    float output[OUTPUT_SIZE];

    struct timespec t1, t2;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(img, IMG_H, IMG_W, IMG_C,
                       conv1_w_int8, conv1_escala, conv1_b_float, CONV1_K, CONV1_F, conv1_out);
    relu(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);
    average_pooling_2x2(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv1_us = diff_us(t1, t2);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(pool1_out, POOL1_H, POOL1_W, CONV1_F,
                       conv2_w_int8, conv2_escala, conv2_b_float, CONV2_K, CONV2_F, conv2_out);
    relu(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    average_pooling_2x2(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv2_us = diff_us(t1, t2);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    convolucao_2d_int8(pool2_out, POOL2_H, POOL2_W, CONV2_F,
                       conv3_w_int8, conv3_escala, conv3_b_float, CONV3_K, CONV3_F, conv3_out);
    relu(conv3_out, CONV3_F * CONV3_OUT_H * CONV3_OUT_W);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv3_us = diff_us(t1, t2);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    camada_densa_ime(conv3_out, FLAT_SIZE, dense1_w_int8, dense1_escala, dense1_b_float,
                 DENSE1_OUT, CONV3_ESCALA_ATIV_ESCALA, dense1_out);
    relu(dense1_out, DENSE1_OUT);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->dense1_us = diff_us(t1, t2);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    camada_densa_ime(dense1_out, DENSE1_OUT, output_w_int8, output_escala, output_b_float,
                     OUTPUT_SIZE, DENSE1_ESCALA_ATIV_ESCALA, output);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->output_us = diff_us(t1, t2);

    if (aplicar_softmax) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        softmax(output, OUTPUT_SIZE);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        tempos->softmax_us = diff_us(t1, t2);
    } else {
        tempos->softmax_us = 0.0;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) probs[i] = output[i];
}