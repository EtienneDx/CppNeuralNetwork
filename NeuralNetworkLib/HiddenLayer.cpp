#include "stdafx.h"
#include "HiddenLayer.h"

#include <iostream>


HiddenLayer::HiddenLayer(int neuronCount, float(*activation)(float), float(*derived_activation)(float)) :
	Layer(neuronCount), weights_(nullptr), deltas_(nullptr),
	activation_(activation), derived_activation_(derived_activation), bias_(neuronCount, 1, true), deltaBias_(neuronCount, 1), preActivationNeurons_(neuronCount, 1)
{
}

HiddenLayer::~HiddenLayer()
{
	delete this->weights_;
	delete this->deltas_;
}

void HiddenLayer::SetPreviousLayer(Layer * layer)
{
	Layer::SetPreviousLayer(layer);
	if (this->GetWeights() == nullptr)
	{
		this->weights_ = new Matrice(this->GetNeurons().GetWidth(), layer->GetNeurons().GetWidth(), true);
		this->deltas_ = new Matrice(this->GetNeurons().GetWidth(), layer->GetNeurons().GetWidth());
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

void HiddenLayer::Train(Matrice dE_dY, float trainSpeed)
{
	//std::cout << *dE_dY << std::endl;
	Matrice d_acti_neurons(this->GetPreActivationNeurons());
	d_acti_neurons.Apply(this->derived_activation_);
	dE_dY.Hadamard(d_acti_neurons);
	//std::cout << "dE/dY : " << *dE_dY << std::endl;

	Matrice dE_dX = dE_dY.Dot(this->GetWeights()->Transposed());// dE/dX is the error at the output of the previous layer

	//std::cout << "trans neurons : " << *transNeurons << std::endl;
	Matrice dE_dW = this->GetPreviousLayer()->GetNeurons().Transposed().Dot(dE_dY);
	//std::cout << "dE/dW : " << *dE_dW << std::endl;

	this->bias_ -= dE_dY * trainSpeed;

	//std::cout << *dWeight << std::endl;
	*this->weights_ -= dE_dW * trainSpeed;

	if (HiddenLayer* v = dynamic_cast<HiddenLayer*>(this->GetPreviousLayer()))
	{
		v->Train(dE_dX, trainSpeed);
	}
}

void HiddenLayer::CalculateDelta(Matrice dE_dY)
{
	Matrice d_acti_neurons(this->GetPreActivationNeurons());
	d_acti_neurons.Apply(this->derived_activation_);
	dE_dY.Hadamard(d_acti_neurons);

	Matrice dE_dX = dE_dY.Dot(this->GetWeights()->Transposed());// dE/dX is the error at the output of the previous layer

	Matrice dE_dW = this->GetPreviousLayer()->GetNeurons().Transposed().Dot(dE_dY);

	this->deltaBias_ += dE_dY;

	*this->deltas_ += dE_dW;

	if (HiddenLayer* v = dynamic_cast<HiddenLayer*>(this->GetPreviousLayer()))
	{
		v->CalculateDelta(dE_dX);
	}
}



void HiddenLayer::ApplyDelta(float trainSpeed)
{
	*this->weights_ -= *this->deltas_ * trainSpeed;
	this->bias_ -= this->deltaBias_ * trainSpeed;

	if (HiddenLayer* v = dynamic_cast<HiddenLayer*>(this->GetPreviousLayer()))
	{
		v->ApplyDelta(trainSpeed);
	}

	// reset deltas
	for (size_t i = 0; i < this->deltas_->GetHeight(); i++)
	{
		for (size_t j = 0; j < this->deltas_->GetWidth(); j++)
		{
			this->deltas_->Set(i, j);
		}
	}

	for (size_t i = 0; i < this->bias_.GetWidth(); i++)
	{
		this->bias_.Set(0, i);
	}
}

const Matrice & HiddenLayer::GetPreActivationNeurons() const
{
	return this->preActivationNeurons_;
}
