//
// Created by joshua on 08.06.25.
//
#include "../include/calculations.h"
#include <math.h>

double sigmoid(const double x)
{
    return 1 / (1 + exp(-x));
}