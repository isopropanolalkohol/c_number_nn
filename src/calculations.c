//
// Created by joshua on 08.06.25.
//
#include "../include/calculations.h"

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}