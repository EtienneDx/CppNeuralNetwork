#include "stdafx.h"
#include "InputLayer.h"


InputLayer::InputLayer(int neuronCount) : Layer(neuronCount)
{
}

void InputLayer::Input(float * values)
{
	for (size_t i = 0; i < this->GetNeurons().GetWidth(); i++)
	{
		this->GetProtectedNeurons().Set(0, i, values[i]);
	}
}