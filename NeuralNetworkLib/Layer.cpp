#include "stdafx.h"
#include "Layer.h"

Layer::Layer(int neuronCount) : neurons_(neuronCount, 1), prevLayer_(nullptr)
{
}

Matrice & Layer::GetProtectedNeurons()
{
	return this->neurons_;
}

const Matrice & Layer::GetNeurons() const
{
	return this->neurons_;
}

void Layer::SetPreviousLayer(Layer * layer)
{
	if (this->GetPreviousLayer())
	{
		this->SetPreviousLayer(layer);
		return;
	}
	this->prevLayer_ = layer;
}

Layer * Layer::GetPreviousLayer()
{
	return this->prevLayer_;
}
