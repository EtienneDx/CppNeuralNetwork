#include "stdafx.h"
#include "Activations.h"

void sigmoig(Matrice & mat)
{
	float(*s)(float f) = sigmoid;
	mat.Apply(s);
}

float sigmoid(float x)
{
	return 1.0 / (1.0 + expf(-x));
}

void sigmoid_derivative(Matrice & mat)
{
	float(*s)(float f) = sigmoid_derivative;
	mat.Apply(s);
}

float sigmoid_derivative(float x)
{
	float s = sigmoid(x);
	return s * (1.0 - s);
}

void tanh(Matrice & mat)
{
	float (*s)(float f) = tanh;
	mat.Apply(s);
}

float tanh_derivative(float x)
{
	float t = tanhf(x);
	return 1.0 - t*t;
}

void tanh_derivative(Matrice & mat)
{
	float(*s)(float f) = tanh_derivative;
	mat.Apply(s);
}

void softmax(Matrice & mat)
{
	float tot = 0;
	for (size_t i = 0; i < mat.GetWidth(); i++)
	{
		tot += exp(mat.Get(0, i));
	}
	for (size_t i = 0; i < mat.GetWidth(); i++)
	{
		float & f = mat.Get(0, i);
		f = exp(f) / tot;
	}
}

void softmax_derivative(Matrice & mat)
{
}
