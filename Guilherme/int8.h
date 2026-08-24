#ifndef INT8_H
#define INT8_H

#include <stdint.h>

typedef struct {
    double conv1_us;
    double conv2_us;
    double conv3_us;
    double dense1_us;
    double output_us;
    double softmax_us;
} TemposCamadasInt8;

void inferencia_int8(const uint8_t *img_uint8, float *probs,
                     TemposCamadasInt8 *tempos, int aplicar_softmax);

#endif