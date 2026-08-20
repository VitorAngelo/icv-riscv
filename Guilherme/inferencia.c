#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

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
#include "x_test_uint8.h"
#include "y_test_uint8.h"

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

void core_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}

void convolucao_2d(const float *entrada, int H, int W, int C_in,
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

void average_pooling_2x2(const float *entrada, int H, int W, int C,
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

void relu(float *dados, int n) {
    for (int i = 0; i < n; i++)
        if (dados[i] < 0.0f) dados[i] = 0.0f;
}

void camada_densa(const float *x, int dim_entrada,
                  const int8_t *W, const float *escala, const float *b, int dim_saida,
                  float *y) {
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

void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];

    float soma = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        soma += x[i];
    }

    for (int i = 0; i < n; i++) x[i] /= soma;
}

void forward(const float *img, float *probs) {
    float conv1_out[CONV1_F * CONV1_OUT_H * CONV1_OUT_W];
    float pool1_out[CONV1_F * POOL1_H * POOL1_W];
    float conv2_out[CONV2_F * CONV2_OUT_H * CONV2_OUT_W];
    float pool2_out[CONV2_F * POOL2_H * POOL2_W];
    float conv3_out[CONV3_F * CONV3_OUT_H * CONV3_OUT_W];
    float dense1_out[DENSE1_OUT];
    float output[OUTPUT_SIZE];

    convolucao_2d(img, IMG_H, IMG_W, IMG_C,
                  conv1_w_int8, conv1_escala, conv1_b_float, CONV1_K, CONV1_F, conv1_out);
    relu(conv1_out, CONV1_F * CONV1_OUT_H * CONV1_OUT_W);
    average_pooling_2x2(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_F, pool1_out);

    convolucao_2d(pool1_out, POOL1_H, POOL1_W, CONV1_F,
                  conv2_w_int8, conv2_escala, conv2_b_float, CONV2_K, CONV2_F, conv2_out);
    relu(conv2_out, CONV2_F * CONV2_OUT_H * CONV2_OUT_W);
    average_pooling_2x2(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_F, pool2_out);

    convolucao_2d(pool2_out, POOL2_H, POOL2_W, CONV2_F,
                  conv3_w_int8, conv3_escala, conv3_b_float, CONV3_K, CONV3_F, conv3_out);
    relu(conv3_out, CONV3_F * CONV3_OUT_H * CONV3_OUT_W);

    camada_densa(conv3_out, FLAT_SIZE, dense1_w_int8, dense1_escala, dense1_b_float, DENSE1_OUT, dense1_out);
    relu(dense1_out, DENSE1_OUT);

    camada_densa(dense1_out, DENSE1_OUT, output_w_int8, output_escala, output_b_float, OUTPUT_SIZE, output);
    softmax(output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) probs[i] = output[i];
}

int main(int argc, char **argv) { // essa e core_affinity no arquivo principal
    int core_id = 0;
    if (argc > 1) core_id = atoi(argv[1]);

    core_affinity(core_id);

    float img[IMG_H * IMG_W];
    float probs[OUTPUT_SIZE];

    struct timespec t1, t2;
    long long ns_total = 0;
    int acertos = 0;
    int num_imagens = x_test_uint8_NUM_IMAGENS;

    for (int n = 0; n < num_imagens; n++) {
        const uint8_t *img_ptr = &x_test_uint8[n * IMG_H * IMG_W];
        for (int i = 0; i < IMG_H * IMG_W; i++) // vai para dentro da funcao
            img[i] = (float)img_ptr[i] / 255.0f;

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        forward(img, probs); // mudar img_ptr para a assinatura da funcao
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        long long ns = (t2.tv_sec - t1.tv_sec) * 1000000000LL +
                       (t2.tv_nsec - t1.tv_nsec);
        ns_total += ns;

        int classe = 0;
        float max_p = probs[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (probs[i] > max_p) {
                max_p = probs[i];
                classe = i;
            }
        }
        if (classe == y_test_uint8[n]) acertos++;
    }

    double acuracia = 100.0 * (double)acertos / (double)num_imagens;
    double tempo_medio_ns = (double)ns_total / (double)num_imagens;

    printf("Core: %d\n", core_id);
    printf("Imagens avaliadas: %d\n", num_imagens);
    printf("Acertos: %d\n", acertos);
    printf("Acurácia: %.2f%%\n", acuracia);
    printf("Tempo total: %.3f ms\n", ns_total / 1e6);
    printf("Tempo médio por inferência: %.2f us\n", tempo_medio_ns / 1e3);

    return 0;
}
