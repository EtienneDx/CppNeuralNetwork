#include "stdafx.h"
#include "Network.h"

Network::Network(int inputCount, 
	float(*error)(const Matrice &output, const Matrice &expected), 
	Matrice (*errorDerivative)(const Matrice &output, const Matrice &expected)) :
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
	Matrice expectedMat(this->outputLayer_->GetNeurons().GetWidth(), expected);
	float err = (*this->error_)(this->outputLayer_->GetNeurons(), expectedMat);

	Matrice mat_err = (*this->derivedError_)(this->outputLayer_->GetNeurons(), expectedMat);

	this->outputLayer_->Train(mat_err, train_speed);

	return err;
}

float Network::Train(std::vector<TrainData*> data, int mini_batch_size, float train_speed)
{
	float err = 0;

	for (size_t i = 0; i < data.size(); i++)
	{
		this->inputLayer_.Input(data[i]->GetInput());
		this->outputLayer_->FeedForward();

		Matrice expectedMat(this->outputLayer_->GetNeurons().GetWidth(), data[i]->GetOutput());
		err += (*this->error_)(this->outputLayer_->GetNeurons(), expectedMat);

		Matrice mat_err = (*this->derivedError_)(this->outputLayer_->GetNeurons(), expectedMat);

		this->outputLayer_->CalculateDelta(mat_err);

		if ((i + 1) % mini_batch_size == 0)// avoid first mini-batch containing a lonely element
		{
			this->outputLayer_->ApplyDelta(train_speed);
		}
	}
	this->outputLayer_->ApplyDelta(train_speed);// last items go together in a mini-mini-batch

	return err / data.size();
}
