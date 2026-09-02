#ifndef FLOAT32_H
#define FLOAT32_H

#include <stdint.h>

typedef struct {
    double conv1_us;
    double conv2_us;
    double conv3_us;
    double dense1_us;
    double output_us;
    double softmax_us;
} TemposCamadasFloat32;

void inferencia_float32(const float *img_f32, float *probs,
                        TemposCamadasFloat32 *tempos, int aplicar_softmax);

#endif
