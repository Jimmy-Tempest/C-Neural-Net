#include <iostream>
#include<vector>
using namespace std;

#include "./MLP.h"

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main(int argc, char** argv) {
bool data;
int hiddenSize[4] = {4,4,2,2};

#include "./TrainingData.cpp"
	MLP *model = new MLP(4, 4, hiddenSize, 2, 0.5);
    for (int i=0;i<numTrainingSets;i++){
		data = model->Training(training_inputs[0], training_outputs[0]);
    	// for (int j=0;j<numInputs;j++)
    	// 	cout << training_inputs[i][j] << endl;
	}
	if(data)
	model->SaveWeight("savedWeight.txt");
	model->LoadWeight("savedWeight.txt");
	// model->SaveWeight("loadedSavedWeight.txt");
	
	return 0;
}
