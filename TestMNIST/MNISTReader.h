#pragma once
#include "TrainData.h"

#include <vector>

class Data : public TrainData
{
private:
	float input[392];
	float output[10];
public:
	float* GetInput() override;
	float* GetOutput() override;
};

int reverseInt(int i);
std::vector<Data*> read_mnist(std::string path, std::string labels = "");