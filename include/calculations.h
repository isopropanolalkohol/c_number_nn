//
// Created by joshua on 08.06.25.
//

#ifndef CALCULATIONS_H
#define CALCULATIONS_H
#include "neural_net.h"
#include <stdint.h>
double sigmoid(const double x);
double sigmoid_derivative(const double x);

gradCf* derivative(const NeuralNet* neuralNet, int expected_value);

double* z_l(NeuronLayer* lhs, ConnectionLayer* conn, NeuronLayer* rhs);

uint32_t reverse_uint32(uint32_t val);
float rand_uniform(float min, float max);

#endif //CALCULATIONS_H
