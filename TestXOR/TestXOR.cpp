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
		{ { 0, 0 },{ 0 } },
		{ { 1, 0 },{ 1 } },
		{ { 0, 1 },{ 1 } },
		{ { 1, 1 },{ 0 } }
	};

	for (size_t i = 0; i < 100000; i++)
	{
		auto data = trainData[rand() % 4];
		auto f = net.Train(data.input, data.output, 0.5);
		/*if (i % 100 == 0)
			cout << "Output : " << net.GetOutput() << "\t - Error : " << f << endl;*/
	}

	float f[2];
	do
	{
		cout << "Input i (or < 0 to exit) : ";
		cin >> f[0];
		if (f[0] >= 0)
		{
			cout << "Input j : ";
			cin >> f[1];
			net.FeedForward(f);
			cout << net.GetOutput().Get(0, 0) << endl;
		}
	} while (f[0] >= 0);

	cout << endl << "Press enter key to exit";

	cin.get();

    return 0;
}

