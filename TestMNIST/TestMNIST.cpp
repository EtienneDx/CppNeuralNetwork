// TestMNIST.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include "MNISTReader.h"

#define TRAIN_DATA		"train-images.idx3-ubyte"
#define TRAIN_LABELS	"train-labels.idx1-ubyte"

#define TEST_DATA		"t10k-images.idx3-ubyte"
#define TEST_LABELS		"t10k-labels.idx1-ubyte"

using namespace std;

int main()
{
	srand(time(0));
	vector<Data*> train = read_mnist(TRAIN_DATA, TRAIN_LABELS);
	vector<Data*> test = read_mnist(TEST_DATA, TEST_LABELS);
	random_shuffle(train.begin(), train.end());
	random_shuffle(test.begin(), test.end());

	Network net(392);

	net.AddHiddenLayer(16, sigmoid, sigmoid_derivative);
	net.AddHiddenLayer(10, sigmoid, sigmoid_derivative);

	/*for (size_t i = 0; i < 50; i++)
	{
		double total_error = 0;
		for (size_t j = 0; j < 10000; j++)
		{
			auto data = train[j % train.size()];
			total_error += net.Train(data->GetInput(), data->GetOutput(), 0.05);
			//std::cout << "Test, Expected value : " << distance(data->output, max_element(data->output, data->output + 10)) << endl;
		}
		std::cout << "Epoch " << i << "\t\tError : " << total_error / 200 << endl;
	}*/
	for (size_t i = 0; i < 50; i++)
	{
		int j = (i % 12) * 5000;
		vector<TrainData*> trainData(train.begin() + j, train.begin() + j + 5000);
		/*for(size_t k = 0; k < 100; k++)
			std::cout << distance(trainData[k]->GetOutput(), max_element(trainData[k]->GetOutput(), trainData[k]->GetOutput() + 10)) << endl;*/
		float err = net.Train(trainData, 200);
		std::cout << "Epoch " << i << "\t\tError : " << err << endl;

	}

	for (size_t i = 0; i < 1000; i++)
	{
		net.FeedForward(test[i]->GetInput());
		auto o = net.GetOutput()[0];
		std:cout << "Test, Expected value : " << distance(test[i]->GetOutput(), max_element(test[i]->GetOutput(), test[i]->GetOutput() + 10)) << 
		" and received " << distance(o, max_element(o, o + 10)) << endl;
	}

	cout << endl << "Press enter key to exit";

	cin.get();

    return 0;
}

