//
// Created by joshua on 08.06.25.
//
#include "calculations.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

double sigmoid(const double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(const double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// This has got to be the worst fucking shit code ever written. Oscar for boilerplate
gradCf* derivative(const NeuralNet* neuralNet, const int expected_value)
{
    gradCf* gradient = malloc(sizeof(gradCf));
    gradient->gradBiases_l1 = NULL;
    gradient->gradBiases_l2 = NULL;
    gradient->gradBiases_l3 = NULL;
    gradient->gradWeights_l1 = NULL;
    gradient->gradWeights_l2 = NULL;
    gradient->gradWeights_l3 = NULL;

    double* delta_l3 = malloc(sizeof(double)*LAYER_4);
    double* zl3 = z_l(neuralNet->layer3, neuralNet->connection3_4, neuralNet->layer4);

    for (int i = 0; i < LAYER_4; i++)
    {
        double z = zl3[i];
        double a = 1.0 / (1.0 + exp(-z));
        double y = (i == expected_value) ? 1.0 : 0.0;
        delta_l3[i] = 2.0 * (a - y) * a * (1.0 - a);
    }
    //printf("delta3: %03f, %03f, %03f\n", delta_l3[0], delta_l3[1], delta_l3[2]);
    free(zl3);
    gradient->gradBiases_l3 = delta_l3;
    double* del_w_jk_3 = malloc(LAYER_3*LAYER_4*sizeof(double));
    for (int j = 0; j < LAYER_4; j++)
    {
        for (int k = 0; k < LAYER_3; k++)
        {
            del_w_jk_3[j*LAYER_3+k] = delta_l3[j]*neuralNet->layer3->neuronArray[k]->activation;
        }
    }
    gradient->gradWeights_l3 = del_w_jk_3;

    double* delta_l2 = malloc(sizeof(double)*LAYER_3);
    double* zl2 = z_l(neuralNet->layer2, neuralNet->connection2_3, neuralNet->layer3);
    double* w_3_T_delta_l3 = malloc(sizeof(double)*LAYER_3);
    for (int i = 0; i < LAYER_3; i++)
    {
        double sum = 0;
        for (int j = 0; j < LAYER_4; j++)
        {
            sum += neuralNet->connection3_4->connectionArray[j * LAYER_3 + i]->weight * delta_l3[j];
        }
        w_3_T_delta_l3[i] = sum;
    }
    for (int i = 0; i < LAYER_3; i++)
    {
        delta_l2[i] = w_3_T_delta_l3[i] * sigmoid_derivative(zl2[i]);
    }
    //printf("delta2: %03f, %03f, %03f\n", delta_l2[0], delta_l2[1], delta_l2[2]);
    free(zl2);
    free(w_3_T_delta_l3);
    gradient->gradBiases_l2 = delta_l2;

    double* del_w_jk_2 = malloc(sizeof(double)*LAYER_3*LAYER_2);
    for (int j = 0; j < LAYER_3; j++)
    {
        for (int k = 0; k < LAYER_2; k++)
        {
            del_w_jk_2[j*LAYER_2+k] = delta_l2[j]*neuralNet->layer2->neuronArray[k]->activation;
        }
    }
    gradient->gradWeights_l2 = del_w_jk_2;

    double* delta_l1 = malloc(sizeof(double)*LAYER_2);

    double* zl1 = z_l(neuralNet->layer1, neuralNet->connection1_2, neuralNet->layer2);
    double* w_2_T_deltal2 = malloc(sizeof(double)*LAYER_2);
    for (int i = 0; i < LAYER_2; i++)
    {
        double sum = 0;
        for (int j = 0; j < LAYER_3; j++)
        {
            sum += neuralNet->connection2_3->connectionArray[i * LAYER_3 + j]->weight * delta_l2[j];
        }
        w_2_T_deltal2[i] = sum;
    }
    //printf("w_2_T_deltal2: %03f, %03f, %03f\n", w_2_T_deltal2[0], w_2_T_deltal2[1], w_2_T_deltal2[2]);
    for (int i = 0; i < LAYER_2; i++)
    {
        delta_l1[i] = w_2_T_deltal2[i] * sigmoid_derivative(zl1[i]);
    }
    //printf("delta1: %03f, %03f, %03f\n", delta_l1[0], delta_l1[1], delta_l1[2]);
    free(zl1);
    free(w_2_T_deltal2);
    gradient->gradBiases_l1 = delta_l1;
    double* del_w_jk_1 = malloc(sizeof(double)*LAYER_1*LAYER_2);
    for (int j = 0; j < LAYER_2; j++)
    {
        for (int k = 0; k < LAYER_1; k++)
        {
            del_w_jk_1[j*LAYER_1 + k] = delta_l1[j]*neuralNet->layer1->neuronArray[k]->activation;
        }
    }
    gradient->gradWeights_l1 = del_w_jk_1;

    return gradient;
}

double* z_l(NeuronLayer* lhs, ConnectionLayer* conn, NeuronLayer* rhs)
{
    // MATRIX (weights) * activations + bias -> sigmoid (holy fucking shit is this exhausting)
    double* activation_vector = malloc(sizeof(double) * lhs->size);
    for (int i = 0; i < lhs->size; i++)
    {
        activation_vector[i] = lhs->neuronArray[i]->activation;
    }

    double* weight_matrix = malloc(sizeof(double) * conn->size);
    for (int i = 0; i < conn->size; i++)
    {
        weight_matrix[i] = conn->connectionArray[i]->weight;
    }
    double* result_vector = malloc(sizeof(double) * rhs->size);

    for (int i = 0; i < rhs->size; i++)
    {
        double sum = 0;
        for (int j = 0; j < lhs->size; j++)
        {
            sum += activation_vector[j] * weight_matrix[lhs->size*i + j];
        }
        result_vector[i] = sum + rhs->neuronArray[i]->bias;
    }
    free(weight_matrix);
    free(activation_vector);
    return result_vector;
}
uint32_t reverse_uint32(uint32_t val)
{
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8 ) |
           ((val & 0x0000FF00) << 8 ) |
           ((val & 0x000000FF) << 24);
}
float rand_uniform(float min, float max)
{
    return min + ((float)rand() / RAND_MAX) * (max - min);
}