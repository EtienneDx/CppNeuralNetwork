#pragma once
#include "Layer.h"

class InputLayer : Layer
{
public:
	InputLayer(int neuronCount);
	void Input(float* values);
};


