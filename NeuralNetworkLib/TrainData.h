#pragma once

class TrainData
{
public:
	virtual float* GetInput() = 0;
	virtual float* GetOutput() = 0;
};