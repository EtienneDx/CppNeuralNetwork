#include "stdafx.h"
#include "Network.h"

Network::Network(int inputCount, 
	float(*error)(const Matrice &output, const Matrice &expected), 
	Matrice *(*errorDerivative)(const Matrice &output, const Matrice &expected)) :
	inputLayer_(inputCount), error_(error), derivedError_(errorDerivative)
{
}

Network::~Network()
{
	if (this->outputLayer_)
	{
		Layer * l = this->outputLayer_, *l2;
		while ((l2 = l->GetPreviousLayer()) != nullptr)// first layer doesn't need to be deleted, it's normal allocation
		{
			delete l;
			l = l2;
		}
	}
}

void Network::AddHiddenLayer(int neuronCount, float(*activation)(float), float(*derived_activation)(float))
{
	HiddenLayer * layer = new HiddenLayer(neuronCount, activation, derived_activation);
	layer->SetPreviousLayer(this->outputLayer_ ? (Layer*)this->outputLayer_ : (Layer*)&this->inputLayer_);
	this->outputLayer_ = layer;
}

const Matrice & Network::GetOutput() const
{
	return this->outputLayer_->GetNeurons();
}

void Network::FeedForward(float * values)
{
	this->inputLayer_.Input(values);
	this->outputLayer_->FeedForward();
}

float Network::Train(float * input, float * expected, float train_speed)
{
	this->inputLayer_.Input(input);
	this->outputLayer_->FeedForward();
	Matrice * expectedMat = new Matrice(this->outputLayer_->GetNeurons().GetWidth(), expected);
	float err = (*this->error_)(this->outputLayer_->GetNeurons(), *expectedMat);

	Matrice * mat_err = (*this->derivedError_)(this->outputLayer_->GetNeurons(), *expectedMat);
	//std::cout << *mat_err << std::endl;
	delete expectedMat;
	//std::cout << *this->outputLayer_->GetWeights() << std::endl << *((HiddenLayer*)(this->outputLayer_->GetPreviousLayer()))->GetWeights() << std::endl;
	this->outputLayer_->Train(mat_err, train_speed);

	delete mat_err;

	return err;
}
