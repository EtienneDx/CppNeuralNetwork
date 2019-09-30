#include "stdafx.h"
#include "Activations.h"

float sigmoid(float x)
{
	return 1.0 / (1.0 + expf(-x));
}

float sigmoid_derivative(float x)
{
	float s = sigmoid(x);
	return s * (1.0 - s);
}

float tanh_derivative(float x)
{
	float t = tanhf(x);
	return 1.0 - t*t;
}
