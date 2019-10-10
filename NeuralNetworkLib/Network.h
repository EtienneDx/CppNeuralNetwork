#pragma once
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "Errors.h"
#include "TrainData.h"

#include <vector>

class Network
{
private:
	InputLayer inputLayer_;
	HiddenLayer * outputLayer_;
	float(*error_)(const Matrice& output, const Matrice& expected);
	Matrice(*derivedError_)(const Matrice& output, const Matrice& expected);
public:
	Network(int inputCount, 
		float(*error)(const Matrice& output, const Matrice& expected) = MeanSquarredError, 
		Matrice(*errorDerivative)(const Matrice& output, const Matrice& expected) = MeanSquarredError_Derivative);
	~Network();
	void AddHiddenLayer(int neuronCount, float(*activation)(float), float(*derived_activation)(float));
	const Matrice& GetOutput() const;
	void FeedForward(float* values);
	float Train(float* input, float* expected, float train_speed = 0.05);
	float Train(std::vector<TrainData*> data, int mini_batch_size = 32, float train_speed = 0.05);
};

