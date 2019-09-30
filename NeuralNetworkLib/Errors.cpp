#include "stdafx.h"
#include "Errors.h"

float MeanSquarredError(const Matrice & output, const Matrice & expected)
{
	float f = 0;
	for (size_t i = 0; i < output.GetWidth(); i++)
	{
		f += pow(expected.Get(0, i) - output.Get(0, i), 2);
	}
	return f / output.GetWidth();

}

Matrice * MeanSquarredError_Derivative(const Matrice & output, const Matrice & expected)
{
	int count = output.GetWidth();
	Matrice * ret = new Matrice(count, 1);
	float f = 2.0 / count;

	for (size_t i = 0; i < count; i++)
	{
		ret->Set(0, i, f * (output.Get(0, i) - expected.Get(0, i)));
	}

	return ret;
}
