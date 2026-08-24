#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "int32.h"

// Incluir futuramente:
// #include "conv1_w_int32.h"
// ...

void inferencia_int32(const int32_t *img_int32, float *probs,
                      TemposCamadasInt32 *tempos, int aplicar_softmax) {
    (void)img_int32;
    (void)probs;
    (void)tempos;
    (void)aplicar_softmax;
}