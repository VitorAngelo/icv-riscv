#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#include "int8.h"
#include "int32.h"
#include "float32.h"
#include "x_test_uint8.h"
#include "y_test_uint8.h"

void core_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}

static void testar_int8(int core_id, int aplicar_softmax, int num_imagens) {
    float probs[10];
    TemposCamadasInt8 tempos_total = {0};
    struct timespec t1, t2;
    long long ns_total = 0;
    int acertos = 0;

    for (int n = 0; n < num_imagens; n++) {
        const uint8_t *img_ptr = &x_test_uint8[n * 32 * 32];
        TemposCamadasInt8 tempos_imagem;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        inferencia_int8(img_ptr, probs, &tempos_imagem, aplicar_softmax);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        tempos_total.conv1_us += tempos_imagem.conv1_us;
        tempos_total.conv2_us += tempos_imagem.conv2_us;
        tempos_total.conv3_us += tempos_imagem.conv3_us;
        tempos_total.dense1_us += tempos_imagem.dense1_us;
        tempos_total.output_us += tempos_imagem.output_us;
        tempos_total.softmax_us += tempos_imagem.softmax_us;

        ns_total += (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);

        int classe = 0;
        float max_p = probs[0];
        for (int i = 1; i < 10; i++) {
            if (probs[i] > max_p) { max_p = probs[i]; classe = i; }
        }
        if (classe == y_test_uint8[n]) acertos++;
    }

    double acuracia = 100.0 * acertos / num_imagens;
    double tempo_medio_us = ns_total / 1e3 / num_imagens;

    printf("=== Versão int8 ===\n");
    printf("Core: %d\n", core_id);
    printf("Softmax: %s\n", aplicar_softmax ? "sim" : "não");
    printf("Imagens avaliadas: %d\n", num_imagens);
    printf("Acertos: %d\n", acertos);
    printf("Acurácia: %.2f%%\n", acuracia);
    printf("Tempo médio total: %.2f us\n", tempo_medio_us);
    printf("Tempos médios por camada (us):\n");
    printf("  conv1: %.2f\n", tempos_total.conv1_us / num_imagens);
    printf("  conv2: %.2f\n", tempos_total.conv2_us / num_imagens);
    printf("  conv3: %.2f\n", tempos_total.conv3_us / num_imagens);
    printf("  dense1: %.2f\n", tempos_total.dense1_us / num_imagens);
    printf("  output: %.2f\n", tempos_total.output_us / num_imagens);
    printf("  softmax: %.2f\n", tempos_total.softmax_us / num_imagens);
}

static void testar_int32(int core_id, int aplicar_softmax, int num_imagens) {
    float probs[10];
    TemposCamadasInt32 tempos_total = {0};
    struct timespec t1, t2;
    long long ns_total = 0;
    int acertos = 0;

    for (int n = 0; n < num_imagens; n++) {
        const uint8_t *img_ptr = &x_test_uint8[n * 32 * 32];
        TemposCamadasInt32 tempos_imagem;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        inferencia_int32(img_ptr, probs, &tempos_imagem, aplicar_softmax);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        tempos_total.conv1_us += tempos_imagem.conv1_us;
        tempos_total.conv2_us += tempos_imagem.conv2_us;
        tempos_total.conv3_us += tempos_imagem.conv3_us;
        tempos_total.dense1_us += tempos_imagem.dense1_us;
        tempos_total.output_us += tempos_imagem.output_us;
        tempos_total.softmax_us += tempos_imagem.softmax_us;

        ns_total += (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);

        int classe = 0;
        float max_p = probs[0];
        for (int i = 1; i < 10; i++) {
            if (probs[i] > max_p) { max_p = probs[i]; classe = i; }
        }
        if (classe == y_test_uint8[n]) acertos++;
    }

    double acuracia = 100.0 * acertos / num_imagens;
    double tempo_medio_us = ns_total / 1e3 / num_imagens;

    printf("=== Versão int32 ===\n");
    printf("Core: %d\n", core_id);
    printf("Softmax: %s\n", aplicar_softmax ? "sim" : "não");
    printf("Imagens avaliadas: %d\n", num_imagens);
    printf("Acertos: %d\n", acertos);
    printf("Acurácia: %.2f%%\n", acuracia);
    printf("Tempo médio total: %.2f us\n", tempo_medio_us);
    printf("Tempos médios por camada (us):\n");
    printf("  conv1: %.2f\n", tempos_total.conv1_us / num_imagens);
    printf("  conv2: %.2f\n", tempos_total.conv2_us / num_imagens);
    printf("  conv3: %.2f\n", tempos_total.conv3_us / num_imagens);
    printf("  dense1: %.2f\n", tempos_total.dense1_us / num_imagens);
    printf("  output: %.2f\n", tempos_total.output_us / num_imagens);
    printf("  softmax: %.2f\n", tempos_total.softmax_us / num_imagens);
}

static void testar_float32(int core_id, int aplicar_softmax, int num_imagens) {
    float probs[10];
    TemposCamadasFloat32 tempos_total = {0};
    struct timespec t1, t2;
    long long ns_total = 0;
    int acertos = 0;

    for (int n = 0; n < num_imagens; n++) {
        const uint8_t *img_ptr = &x_test_uint8[n * 32 * 32];
        TemposCamadasFloat32 tempos_imagem;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        inferencia_float32(img_ptr, probs, &tempos_imagem, aplicar_softmax);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        tempos_total.conv1_us += tempos_imagem.conv1_us;
        tempos_total.conv2_us += tempos_imagem.conv2_us;
        tempos_total.conv3_us += tempos_imagem.conv3_us;
        tempos_total.dense1_us += tempos_imagem.dense1_us;
        tempos_total.output_us += tempos_imagem.output_us;
        tempos_total.softmax_us += tempos_imagem.softmax_us;

        ns_total += (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);

        int classe = 0;
        float max_p = probs[0];
        for (int i = 1; i < 10; i++) {
            if (probs[i] > max_p) { max_p = probs[i]; classe = i; }
        }
        if (classe == y_test_uint8[n]) acertos++;
    }

    double acuracia = 100.0 * acertos / num_imagens;
    double tempo_medio_us = ns_total / 1e3 / num_imagens;

    printf("=== Versão float32 ===\n");
    printf("Core: %d\n", core_id);
    printf("Softmax: %s\n", aplicar_softmax ? "sim" : "não");
    printf("Imagens avaliadas: %d\n", num_imagens);
    printf("Acertos: %d\n", acertos);
    printf("Acurácia: %.2f%%\n", acuracia);
    printf("Tempo médio total: %.2f us\n", tempo_medio_us);
    printf("Tempos médios por camada (us):\n");
    printf("  conv1: %.2f\n", tempos_total.conv1_us / num_imagens);
    printf("  conv2: %.2f\n", tempos_total.conv2_us / num_imagens);
    printf("  conv3: %.2f\n", tempos_total.conv3_us / num_imagens);
    printf("  dense1: %.2f\n", tempos_total.dense1_us / num_imagens);
    printf("  output: %.2f\n", tempos_total.output_us / num_imagens);
    printf("  softmax: %.2f\n", tempos_total.softmax_us / num_imagens);
}

int main(int argc, char **argv) {
    int core_id = 0;
    int aplicar_softmax = 0;
    int tipo = 0;         // 0=int8, 1=int32, 2=float32
    int num_imagens = 10000;

    if (argc > 1) core_id = atoi(argv[1]);
    if (argc > 2) aplicar_softmax = atoi(argv[2]);
    if (argc > 3) tipo = atoi(argv[3]);
    if (argc > 4) num_imagens = atoi(argv[4]);

    core_affinity(core_id);

    switch (tipo) {
        case 0:
            testar_int8(core_id, aplicar_softmax, num_imagens);
            break;
        case 1:
            testar_int32(core_id, aplicar_softmax, num_imagens);
            break;
        case 2:
            testar_float32(core_id, aplicar_softmax, num_imagens);
            break;
        default:
            fprintf(stderr, "Tipo inválido. Use 0 (int8), 1 (int32), 2 (float32).\n");
            return 1;
    }

    return 0;
}
