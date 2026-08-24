#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "float32.h"

// Incluir futuramente:
// #include "conv1_w_float32.h"
// ...

void inferencia_float32(const float *img_float, float *probs,
                        TemposCamadasFloat32 *tempos, int aplicar_softmax) {
    (void)img_float;
    (void)probs;
    (void)tempos;
    (void)aplicar_softmax;
    // Implementar quando tiver pesos float32
}