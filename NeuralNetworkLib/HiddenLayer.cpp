#include "stdafx.h"
#include "HiddenLayer.h"

#include <iostream>


HiddenLayer::HiddenLayer(int neuronCount, float(*activation)(float), float(*derived_activation)(float)) :
	Layer(neuronCount), weights_(nullptr), 
	activation_(activation), derived_activation_(derived_activation), bias_(neuronCount, 1, true), preActivationNeurons_(neuronCount, 1)
{
}

HiddenLayer::~HiddenLayer()
{
	delete this->weights_;
}

void HiddenLayer::SetPreviousLayer(Layer * layer)
{
	Layer::SetPreviousLayer(layer);
	if (this->GetWeights() == nullptr)
	{
		this->weights_ = new Matrice(this->GetNeurons().GetWidth(), layer->GetNeurons().GetWidth(), true);
	}
}

void HiddenLayer::FeedForward()
{
	if (this->GetPreviousLayer() == nullptr)
	{
		throw "No previous layer detected, unable to calculate output!";
	}

	if (HiddenLayer* v = dynamic_cast<HiddenLayer*>(this->GetPreviousLayer()))
	{
		v->FeedForward();
	}

	this->GetPreviousLayer()->GetNeurons().Dot(*this->GetWeights(), this->preActivationNeurons_);
	this->preActivationNeurons_ += this->bias_;
	//std::cout << "weights : " << *this->GetWeights() << std::endl;
	//std::cout << "neurons : " << this->GetProtectedNeurons() << std::endl;
	this->GetProtectedNeurons() = this->preActivationNeurons_;
	this->GetProtectedNeurons().Apply(this->activation_);
}

Matrice * HiddenLayer::GetWeights()
{
	return this->weights_;
}

void HiddenLayer::Train(Matrice * dE_dY, float trainSpeed)
{
	//std::cout << *dE_dY << std::endl;
	//dE_dY *= (*this->derived_activation_)(this->GetNeurons());
	Matrice * d_acti_neurons = new Matrice(this->GetPreActivationNeurons());
	d_acti_neurons->Apply(this->derived_activation_);
	dE_dY->Hadamard(*d_acti_neurons);
	delete d_acti_neurons;
	//std::cout << "dE/dY : " << *dE_dY << std::endl;

	Matrice * transWeight = this->GetWeights()->Transposed();
	Matrice * dE_dX = dE_dY->Dot(*transWeight);// dE/dX is the error at the output of the previous layer
	delete transWeight;

	Matrice * transNeurons = this->GetPreviousLayer()->GetNeurons().Transposed();
	//std::cout << "trans neurons : " << *transNeurons << std::endl;
	Matrice * dE_dW = transNeurons->Dot(*dE_dY);
	//std::cout << "dE/dW : " << *dE_dW << std::endl;
	delete transNeurons;

	Matrice * dBias = (*dE_dY * trainSpeed);
	this->bias_ += *dBias;
	delete dBias;

	Matrice * dWeight = (*dE_dW * trainSpeed);
	//std::cout << *dE_dY << std::endl << *dE_dW << std::endl;
	*this->weights_ += *dWeight;
	delete dWeight;

	delete dE_dW;

	if (HiddenLayer* v = dynamic_cast<HiddenLayer*>(this->GetPreviousLayer()))
	{
		v->Train(dE_dX, trainSpeed);
	}

	delete dE_dX;
}

const Matrice & HiddenLayer::GetPreActivationNeurons() const
{
	return this->preActivationNeurons_;
}
