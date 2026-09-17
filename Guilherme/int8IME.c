#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "int8.h"

//pesos que carregam Btile diretamente
#include "int8_conv1_w_ime_tiles.h"
#include "int8_conv1_b_ime.h"
#include "int8_conv2_w_ime_tiles.h"
#include "int8_conv2_b_ime.h"
#include "int8_conv3_w_ime_tiles.h"
#include "int8_conv3_b_ime.h"
#include "int8_dense1_w_ime_tiles.h"
#include "int8_dense1_b_ime.h"

#include "int8_output_w.h"
#include "int8_output_b.h"

//escalas por canal pra requantizar
#include "int8_conv1_escala.h"
#include "int8_conv2_escala.h"
#include "int8_conv3_escala.h"
#include "int8_dense1_escala.h"
#include "int8_output_escala.h"

//escalas pra ativação
#include "int8_conv1_escala_ativ.h"
#include "int8_conv2_escala_ativ.h"
#include "int8_conv3_escala_ativ.h"
#include "int8_dense1_escala_ativ.h"

//dimensões
#define IMG_H 32
#define IMG_W 32
#define IMG_C 1

#define CONV1_F 6
#define CONV1_K 5
#define CONV1_OUT_H (IMG_H - CONV1_K + 1)
#define CONV1_OUT_W (IMG_W - CONV1_K + 1)

#define POOL1_H (CONV1_OUT_H >> 1)
#define POOL1_W (CONV1_OUT_W >> 1)

#define CONV2_F 16
#define CONV2_K 5
#define CONV2_OUT_H (POOL1_H - CONV2_K + 1)
#define CONV2_OUT_W (POOL1_W - CONV2_K + 1)

#define POOL2_H (CONV2_OUT_H >> 1)
#define POOL2_W (CONV2_OUT_W >> 1)

#define CONV3_F 120
#define CONV3_K 5
#define CONV3_OUT_H (POOL2_H - CONV3_K + 1)
#define CONV3_OUT_W (POOL2_W - CONV3_K + 1)

#define FLAT_SIZE (CONV3_F * CONV3_OUT_H * CONV3_OUT_W)

#define DENSE1_OUT 84
#define OUTPUT_SIZE 10

#define CONV1_K_REAL (IMG_C * CONV1_K * CONV1_K)  
#define CONV1_K_PAD  32
#define CONV1_F_PAD  8

#define CONV2_K_REAL (CONV1_F * CONV2_K * CONV2_K)  
#define CONV2_K_PAD  152

#define CONV3_K_REAL (CONV2_F * CONV3_K * CONV3_K) 

static double diff_us(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) * 1e6 +
           (fim.tv_nsec - inicio.tv_nsec) / 1e3;
}

//C += A * B^T para matrizes 4x8 (A) e 8x4 (B), resultando em C 4x4.
//Ae B em int8, C int32.
static inline void tile4x8x4_us(const uint8_t A[32], const int8_t B[32], int32_t Cacc[16]) {
    __asm__ volatile(
        "vsetvli      t0, zero, e32, m2   \n\t"  //configura 16 acumuladores int32
        "vle32.v      v16, (%[C])         \n\t"  //carrega Cacc
        "vsetvli      t0, zero, e8, m1    \n\t"  //configura vetores int8 de 32 bytes
        "vle8.v       v0, (%[A])          \n\t"  //carrega A
        "vle8.v       v1, (%[B])          \n\t"  //carrega B
        "smt.vmadotus v16, v0, v1         \n\t"  //acumula
        "vsetvli      t0, zero, e32, m2   \n\t"
        "vse32.v      v16, (%[C])         \n\t"  // salva Cacc
        :
        : [A] "r"(A), [B] "r"(B), [C] "r"(Cacc)
        : "memory", "t0", "v0", "v1", "v16", "v17");
}

static void conv2d_ime(const uint8_t *entrada, int H, int W,
                       int K, int C_out_real, int C_out_pad,
                       int K_real, int K_pad,
                       const int8_t *pesos_tiles, const int32_t *bias_pad,
                       int32_t *saida) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int HW = H * W;
    int num_tiles_k = K_pad >> 3;

    for (int f = 0; f < C_out_pad; f += 4) {
        int tile_n = f >> 2;

        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j += 4) {
                int jj[4];
                for (int r = 0; r < 4; r++)
                    jj[r] = (j + r < W_out) ? (j + r) : (W_out - 1);

                int32_t acc[16];
                for (int r = 0; r < 4; r++)
                    for (int c4 = 0; c4 < 4; c4++)
                        acc[r * 4 + c4] = bias_pad[f + c4];

                int c = 0, ki = 0, kj = 0;

                for (int kg0 = 0; kg0 < K_pad; kg0 += 8) {
                    uint8_t Atile[32] = {0};

                    for (int kk = 0; kk < 8; kk++) {
                        int kg = kg0 + kk;
                        if (kg < K_real) {
                            for (int r = 0; r < 4; r++)
                                Atile[r * 8 + kk] = entrada[c * HW + (i + ki) * W + (jj[r] + kj)];
                        }
                        kj++;
                        if (kj == K) { kj = 0; ki++; if (ki == K) { ki = 0; c++; } }
                    }

                    int tile_k = kg0 >> 3;
                    const int8_t *B = pesos_tiles + ((tile_n * num_tiles_k + tile_k) * 32);

                    tile4x8x4_us(Atile, B, acc);
                }

                for (int r = 0; r < 4; r++) {
                    if (j + r >= W_out) break;
                    int mg = i * W_out + (j + r);
                    for (int c4 = 0; c4 < 4; c4++) {
                        if (f + c4 < C_out_real)
                            saida[(f + c4) * (H_out * W_out) + mg] = acc[r * 4 + c4];
                    }
                }
            }
        }
    }
}

static void dense_ime(const uint8_t *x, int Ein, int Eout,
                      const int8_t *pesos_tiles, const int32_t *bias,
                      int32_t *saida) {
    int num_tiles_k = Ein >> 3;

    for (int n = 0; n < Eout; n += 4) {
        int tile_n = n >> 2;

        int32_t acc[16] = {0};
        acc[0] = bias[n];
        acc[1] = bias[n + 1];
        acc[2] = bias[n + 2];
        acc[3] = bias[n + 3];

        uint8_t Atile[32];

        for (int k = 0; k < Ein; k += 8) {
            memcpy(Atile,      x + k, 8);
            memcpy(Atile + 8,  x + k, 8);
            memcpy(Atile + 16, x + k, 8);
            memcpy(Atile + 24, x + k, 8);

            int tile_k = k >> 3;
            const int8_t *B = pesos_tiles + ((tile_n * num_tiles_k + tile_k) * 32);

            tile4x8x4_us(Atile, B, acc);
        }

        saida[n]     = acc[0];
        saida[n + 1] = acc[1];
        saida[n + 2] = acc[2];
        saida[n + 3] = acc[3];
    }
}

static void average_pooling_2x2_int32(const int32_t *entrada, int H, int W, int C,
                                      int32_t *saida) {
    int H_out = H >> 1, W_out = W >> 1;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int32_t soma = 0;
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++)
                        soma += entrada[c * (H * W) + (2*i + pi) * W + (2*j + pj)];
                saida[c * (H_out * W_out) + i * W_out + j] = (soma + 2) >> 2;
            }
        }
    }
}

static void relu_int32(int32_t *dados, int n) {
    for (int i = 0; i < n; i++)
        if (dados[i] < 0) dados[i] = 0;
}

static void requantizar_por_canal(const int32_t *entrada, int H, int W,
                                  const float *escala_peso, int n_canais,
                                  float escala_entrada, float escala_ativ_saida,
                                  uint8_t *saida) {
    int plano = H * W;
    for (int c = 0; c < n_canais; c++) {
        float fator = (escala_entrada * escala_peso[c]) / escala_ativ_saida;
        int offset = c * plano;
        for (int i = 0; i < plano; i++) {
            int idx = offset + i;
            float val = (float)entrada[idx] * fator;
            int32_t arred = (int32_t)(val + 0.5f);
            if (arred < 0) arred = 0;
            if (arred > 255) arred = 255;
            saida[idx] = (uint8_t)arred;
        }
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

    struct timespec t1, t2;

    //conv1
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    conv2d_ime(img_uint8, IMG_H, IMG_W, CONV1_K,
               CONV1_F, CONV1_F_PAD, CONV1_K_REAL, CONV1_K_PAD,
               int8_conv1_w_ime_tiles, int8_conv1_b_ime, conv1_out);
    relu_int32(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv1_us = diff_us(t1, t2);

    average_pooling_2x2_int32(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_i32);
    requantizar_por_canal(pool1_i32, POOL1_H, POOL1_W,
                          int8_conv1_escala, CONV1_F,
                          1.0f/255.0f, INT8_CONV1_ESCALA_ATIV, pool1_u8);

    //conv2
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    conv2d_ime(pool1_u8, POOL1_H, POOL1_W, CONV2_K,
               CONV2_F, CONV2_F, CONV2_K_REAL, CONV2_K_PAD,
               int8_conv2_w_ime_tiles, int8_conv2_b_ime, conv2_out);
    relu_int32(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv2_us = diff_us(t1, t2);

    average_pooling_2x2_int32(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_i32);
    requantizar_por_canal(pool2_i32, POOL2_H, POOL2_W,
                          int8_conv2_escala, CONV2_F,
                          INT8_CONV1_ESCALA_ATIV, INT8_CONV2_ESCALA_ATIV, pool2_u8);

    //conv3
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    conv2d_ime(pool2_u8, POOL2_H, POOL2_W, CONV3_K,
               CONV3_F, CONV3_F, CONV3_K_REAL, CONV3_K_REAL,
               int8_conv3_w_ime_tiles, int8_conv3_b_ime, conv3_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv3_us = diff_us(t1, t2);

    requantizar_por_canal(conv3_out, CONV3_OUT_H, CONV3_OUT_W,
                          int8_conv3_escala, CONV3_F,
                          INT8_CONV2_ESCALA_ATIV, INT8_CONV3_ESCALA_ATIV, conv3_u8);

    //dense1
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    dense_ime(conv3_u8, FLAT_SIZE, DENSE1_OUT,
              int8_dense1_w_ime_tiles, int8_dense1_b_ime, dense1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->dense1_us = diff_us(t1, t2);

    requantizar_por_canal(dense1_out, 1, 1,
                          int8_dense1_escala, DENSE1_OUT,
                          INT8_CONV3_ESCALA_ATIV, INT8_DENSE1_ESCALA_ATIV, dense1_u8);

    //output
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        int32_t soma = int8_output_b[j];
        for (int i = 0; i < DENSE1_OUT; i++)
            soma += (int32_t)dense1_u8[i] * int8_output_w[i * OUTPUT_SIZE + j];
        output_i32[j] = soma;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->output_us = diff_us(t1, t2);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        probs[i] = (float)output_i32[i] * (INT8_DENSE1_ESCALA_ATIV * int8_output_escala[i]);

    if (aplicar_softmax) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        softmax(probs, OUTPUT_SIZE);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        tempos->softmax_us = diff_us(t1, t2);
    } else {
        tempos->softmax_us = 0.0;
    }
}
