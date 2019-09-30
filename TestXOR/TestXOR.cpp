// TestXOR.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"

using namespace std;

struct TrainData
{
	float input[2];
	float output[1];
};

int main()
{
	srand(time(0));
	Network net(2);

	net.AddHiddenLayer(2, sigmoid, sigmoid_derivative);
	net.AddHiddenLayer(1, sigmoid, sigmoid_derivative);

	vector<TrainData> trainData{
		{ { 0, 0 },{ 1 } },
		{ { 1, 0 },{ 0 } },
		{ { 0, 1 },{ 0 } },
		{ { 1, 1 },{ 1 } }
	};

	for (size_t i = 0; i < 20000; i++)
	{
		auto data = trainData[i % 4];
		auto f = net.Train(data.input, data.output, 0.1);
		if (i % 10 == 0)
			cout << "Output : " << net.GetOutput() << "\t - Error : " << f << endl;
	}

	cout << endl << "Press any key to exit";

	cin.get();

    return 0;
}

