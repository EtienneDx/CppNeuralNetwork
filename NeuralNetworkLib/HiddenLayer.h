#pragma once
#include "Matrice.h"
#include "Layer.h"

class HiddenLayer : public Layer
{
private:
	Matrice preActivationNeurons_;
	Matrice * weights_;
	Matrice * deltas_;
	Matrice bias_;
	Matrice deltaBias_;
	float(*activation_)(float);
	float(*derived_activation_)(float);
public:
	HiddenLayer(int neuronCount, float(*activation)(float), float(*derived_activation)(float));
	~HiddenLayer();

	void SetPreviousLayer(Layer* layer) override;
	void FeedForward();
	Matrice* GetWeights();
	void Train(Matrice dE_dY, float trainSpeed);
	void CalculateDelta(Matrice dE_dY);
	void ApplyDelta(float trainSpeed);
	const Matrice& GetPreActivationNeurons() const;
};

