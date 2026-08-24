#ifndef INT32_H
#define INT32_H

#include <stdint.h>

typedef struct {
    double conv1_us;
    double conv2_us;
    double conv3_us;
    double dense1_us;
    double output_us;
    double softmax_us;
} TemposCamadasInt32;

void inferencia_int32(const int32_t *img_int32, float *probs,
                      TemposCamadasInt32 *tempos, int aplicar_softmax);

#endif