#pragma once
#include "Matrice.h"

float MeanSquarredError(const Matrice& output, const Matrice& expected);
Matrice MeanSquarredError_Derivative(const Matrice& output, const Matrice& expected);