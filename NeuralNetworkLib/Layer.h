#pragma once
#include "Matrice.h"

class Layer
{
private:
	Matrice neurons_;
	Layer* prevLayer_;
protected:
	Matrice & GetProtectedNeurons();
public:
	Layer(int neuronCount);
	const Matrice& GetNeurons() const;
	virtual void SetPreviousLayer(Layer *layer);
	Layer* GetPreviousLayer();
};

