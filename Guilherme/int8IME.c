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

#define CONV1_K_REAL (IMG_C * CONV1_K * CONV1_K) 
#define CONV1_K_PAD  32
#define CONV1_F_PAD  8

#define CONV2_K_REAL (CONV1_F * CONV2_K * CONV2_K)  
#define CONV2_K_PAD  152

#define CONV3_K_REAL (CONV2_F * CONV3_K * CONV3_K)  

#define OUTPUT_EIN_PAD  88
#define OUTPUT_EOUT_PAD 12 

static double diff_us(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) * 1e6 +
           (fim.tv_nsec - inicio.tv_nsec) / 1e3;
}

// Executa um tile 4x8x4 usando smt.vmadotus (A uint8, B int8, C int32)
static inline void tile4x8x4_us(const uint8_t A[32], const int8_t B[32], int32_t Cacc[16]) {
    __asm__ volatile(
        "vsetvli      t0, zero, e32, m2   \n\t"   // configura vd/vd+1 como int32
        "vle32.v      v16, (%[C])         \n\t"   // carrega acumulador atual
        "vsetvli      t0, zero, e8, m1    \n\t"   // configura entrada int8
        "vle8.v       v0, (%[A])          \n\t"
        "vle8.v       v1, (%[B])          \n\t"
        "smt.vmadotus v16, v0, v1         \n\t"   // C += A * B^T
        "vsetvli      t0, zero, e32, m2   \n\t"
        "vse32.v      v16, (%[C])         \n\t"   // armazena acumulador
        :
        : [A] "r"(A), [B] "r"(B), [C] "r"(Cacc)
        : "cc", "memory", "t0");
}

static void conv2d_ime_padded(const uint8_t *entrada, int H, int W,
                               int K, int C_out_real, int C_out_pad,
                               int K_real, int K_pad,
                               const int8_t *pesos_pad, const int32_t *bias_pad,
                               int32_t *saida) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int M_total = H_out * W_out;
    int KK = K * K, HW = H * W;

    for (int f = 0; f < C_out_pad; f += 4) {        
        for (int m_start = 0; m_start < M_total; m_start += 4) { 
            int mg[4], ii[4], jj[4];
            for (int r = 0; r < 4; r++) {
                int m = m_start + r;
                if (m >= M_total) m = M_total - 1;   
                mg[r] = m; ii[r] = m / W_out; jj[r] = m % W_out;
            }

            int32_t acc[16];
            for (int r = 0; r < 4; r++)
                for (int c4 = 0; c4 < 4; c4++)
                    acc[r * 4 + c4] = bias_pad[f + c4]; 
            for (int kg0 = 0; kg0 < K_pad; kg0 += 8) {   
                uint8_t Atile[32];
                int8_t  Btile[32];
                for (int kk = 0; kk < 8; kk++) {
                    int kg = kg0 + kk;
                    int kg_ent = (kg < K_real) ? kg : (K_real - 1); 
                    int c = kg_ent / KK, rm = kg_ent % KK, ki = rm / K, kj = rm % K;
                    for (int r = 0; r < 4; r++)
                        Atile[r * 8 + kk] = entrada[c * HW + (ii[r] + ki) * W + (jj[r] + kj)];
                    for (int c4 = 0; c4 < 4; c4++)
                        Btile[c4 * 8 + kk] = pesos_pad[kg * C_out_pad + (f + c4)];
                }
                tile4x8x4_us(Atile, Btile, acc);
            }

            for (int r = 0; r < 4; r++) {
                if (m_start + r >= M_total) break;    
                for (int c4 = 0; c4 < 4; c4++) {
                    int fc = f + c4;
                    if (fc < C_out_real)             
                        saida[fc * (H_out * W_out) + mg[r]] = acc[r * 4 + c4];
                }
            }
        }
    }
}

static void densa_ime_padded(const uint8_t *x, int Ein_real, int Ein_pad,
                              int Eout_real, int Eout_pad,
                              const int8_t *pesos_pad, const int32_t *bias_pad,
                              int32_t *saida) {
    for (int n = 0; n < Eout_pad; n += 4) {          
        int32_t acc[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                acc[r * 4 + c] = bias_pad[n + c];

        for (int k = 0; k < Ein_pad; k += 8) {       
            uint8_t Atile[32];
            int8_t  Btile[32];
            for (int kk = 0; kk < 8; kk++) {
                int kg = k + kk;
                uint8_t a = (kg < Ein_real) ? x[kg] : x[Ein_real - 1];
                for (int r = 0; r < 4; r++)
                    Atile[r * 8 + kk] = a;            
            }
            for (int c = 0; c < 4; c++)
                for (int kk = 0; kk < 8; kk++)
                    Btile[c * 8 + kk] = pesos_pad[(k + kk) * Eout_pad + (n + c)];
            tile4x8x4_us(Atile, Btile, acc);
        }

        for (int c = 0; c < 4; c++) {
            int nc = n + c;
            if (nc < Eout_real) saida[nc] = acc[c];
        }
    }
}

static int8_t  conv1_w_pad[CONV1_K_PAD * CONV1_F_PAD];
static int32_t conv1_b_pad[CONV1_F_PAD];
static int conv1_pad_ok = 0;
static void conv1_preparar_pad(void) {
    if (conv1_pad_ok) return;
    memset(conv1_w_pad, 0, sizeof(conv1_w_pad));
    memset(conv1_b_pad, 0, sizeof(conv1_b_pad));
    for (int c = 0; c < IMG_C; c++)
        for (int ki = 0; ki < CONV1_K; ki++)
            for (int kj = 0; kj < CONV1_K; kj++) {
                int kg = c * (CONV1_K * CONV1_K) + ki * CONV1_K + kj;
                int base = ((ki * CONV1_K + kj) * IMG_C + c) * CONV1_F;
                for (int f = 0; f < CONV1_F; f++)
                    conv1_w_pad[kg * CONV1_F_PAD + f] = conv1_w_int8[base + f];
            }
    for (int f = 0; f < CONV1_F; f++) conv1_b_pad[f] = conv1_b_int32[f];
    conv1_pad_ok = 1;
}

static int8_t conv2_w_pad[CONV2_K_PAD * CONV2_F];
static int conv2_pad_ok = 0;
static void conv2_preparar_pad(void) {
    if (conv2_pad_ok) return;
    memset(conv2_w_pad, 0, sizeof(conv2_w_pad));
    for (int c = 0; c < CONV1_F; c++)
        for (int ki = 0; ki < CONV2_K; ki++)
            for (int kj = 0; kj < CONV2_K; kj++) {
                int kg = c * (CONV2_K * CONV2_K) + ki * CONV2_K + kj;
                int base = ((ki * CONV2_K + kj) * CONV1_F + c) * CONV2_F;
                for (int f = 0; f < CONV2_F; f++)
                    conv2_w_pad[kg * CONV2_F + f] = conv2_w_int8[base + f];
            }
    conv2_pad_ok = 1;
}

static int8_t conv3_w_pad[CONV3_K_REAL * CONV3_F];
static int conv3_pad_ok = 0;
static void conv3_preparar_pad(void) {
    if (conv3_pad_ok) return;
    for (int c = 0; c < CONV2_F; c++)
        for (int ki = 0; ki < CONV3_K; ki++)
            for (int kj = 0; kj < CONV3_K; kj++) {
                int kg = c * (CONV3_K * CONV3_K) + ki * CONV3_K + kj;
                int base = ((ki * CONV3_K + kj) * CONV2_F + c) * CONV3_F;
                for (int f = 0; f < CONV3_F; f++)
                    conv3_w_pad[kg * CONV3_F + f] = conv3_w_int8[base + f];
            }
    conv3_pad_ok = 1;
}

static int8_t  output_w_pad[OUTPUT_EIN_PAD * OUTPUT_EOUT_PAD];
static int32_t output_b_pad[OUTPUT_EOUT_PAD];
static int output_pad_ok = 0;
static void output_preparar_pad(void) {
    if (output_pad_ok) return;
    memset(output_w_pad, 0, sizeof(output_w_pad));
    memset(output_b_pad, 0, sizeof(output_b_pad));
    for (int k = 0; k < DENSE1_OUT; k++)
        for (int n = 0; n < OUTPUT_SIZE; n++)
            output_w_pad[k * OUTPUT_EOUT_PAD + n] = output_w_int8[k * OUTPUT_SIZE + n];
    for (int n = 0; n < OUTPUT_SIZE; n++) output_b_pad[n] = output_b_int32[n];
    output_pad_ok = 1;
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
    conv1_preparar_pad();
    conv2_preparar_pad();
    conv3_preparar_pad();
    output_preparar_pad();

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
    conv2d_ime_padded(img_u8, IMG_H, IMG_W, CONV1_K,
                      CONV1_F, CONV1_F_PAD, CONV1_K_REAL, CONV1_K_PAD,
                      conv1_w_pad, conv1_b_pad, conv1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv1_us = diff_us(t1, t2);

    relu_int32(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);
    average_pooling_2x2_int32(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_i32);
    requantizar_por_canal(pool1_i32, CONV1_F * POOL1_H * POOL1_W,
                          conv1_escala, CONV1_F,
                          1.0f/255.0f, CONV1_ESCALA_ATIV_ESCALA, pool1_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    conv2d_ime_padded(pool1_u8, POOL1_H, POOL1_W, CONV2_K,
                      CONV2_F, CONV2_F, CONV2_K_REAL, CONV2_K_PAD,
                      conv2_w_pad, conv2_b_int32, conv2_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv2_us = diff_us(t1, t2);

    relu_int32(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    average_pooling_2x2_int32(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_i32);
    requantizar_por_canal(pool2_i32, CONV2_F * POOL2_H * POOL2_W,
                          conv2_escala, CONV2_F,
                          CONV1_ESCALA_ATIV_ESCALA, CONV2_ESCALA_ATIV_ESCALA, pool2_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    conv2d_ime_padded(pool2_u8, POOL2_H, POOL2_W, CONV3_K,
                      CONV3_F, CONV3_F, CONV3_K_REAL, CONV3_K_REAL,
                      conv3_w_pad, conv3_b_int32, conv3_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->conv3_us = diff_us(t1, t2);

    relu_int32(conv3_out, CONV3_F * CONV3_OUT_H * CONV3_OUT_W);
    requantizar_por_canal(conv3_out, FLAT_SIZE,
                          conv3_escala, CONV3_F,
                          CONV2_ESCALA_ATIV_ESCALA, CONV3_ESCALA_ATIV_ESCALA, conv3_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    densa_ime_padded(conv3_u8, FLAT_SIZE, FLAT_SIZE, DENSE1_OUT, DENSE1_OUT,
                     dense1_w_int8, dense1_b_int32, dense1_out);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    tempos->dense1_us = diff_us(t1, t2);

    relu_int32(dense1_out, DENSE1_OUT);
    requantizar_por_canal(dense1_out, DENSE1_OUT,
                          dense1_escala, DENSE1_OUT,
                          CONV3_ESCALA_ATIV_ESCALA, DENSE1_ESCALA_ATIV_ESCALA, dense1_u8);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    densa_ime_padded(dense1_u8, DENSE1_OUT, OUTPUT_EIN_PAD, OUTPUT_SIZE, OUTPUT_EOUT_PAD,
                     output_w_pad, output_b_pad, output_i32);
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
