#pragma once
#include "Matrice.h"

void sigmoig(Matrice & mat);
float sigmoid(float x);
void sigmoid_derivative(Matrice & mat);
float sigmoid_derivative(float x);

void tanh(Matrice & mat);
float tanh_derivative(float x);
void tanh_derivative(Matrice & mat);

void softmax(Matrice & mat);
void softmax_derivative(Matrice & mat);