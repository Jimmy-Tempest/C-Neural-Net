#ifndef MLP_H
#define MLP_H

#include <math.h>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

class MLPCell // Neuron
{
		double delta; // used in backpropagation
		double bias;  // bias delta * lr (added when sum weights)
        double sigmoid(double x) { return 1/(1 + exp(-x)); }  // squish function
        double dSigmoid(double x) { return x*(1-x); }  //  sigmoid but backpropagation
	public:
		vector<double> input; // activated data from last layer
		vector<double> inerr;  // loss
		vector<double> weight;  // adjustable value signify how the network think
		double output;  // output
		MLPCell(int inputNum);
		void FeedForward();  // train forward
		void BackPropagate(double derr);  // 
		void AdjustWeight(double lr);  // adjust all weights
			
};

class MLP
{
		vector<MLPCell> hiddenLayer;  // hidden layer
		vector<MLPCell> hiddenLayer2; // hidden layer2
		vector<MLPCell> outputLayer;  // output layer
		double myThreshold;  // value which classify certainty
		double Step(double value) {if(value<myThreshold) return 0.0; else return 1.0;}
	public:
		vector<double> input;  // vector of input
		vector<double> output;  // vector of output
		MLP(int inputNum,int hiddenNum, int hiddenNum2,int outputNum,double threshold);  
		void Testing();
		bool Training(double trainingInput[], double trainingOutput[]);	
		void SaveWeight(string FileName);                                 //Assignment1
		void LoadWeight(string FileName);                                 //Assignment2    	
};

#endif

// ============== Methods ==============

MLPCell::MLPCell(int inputNum){
	srand((unsigned)time(NULL));
	bias = 0;
	input.resize(inputNum);
	inerr.resize(inputNum);
	for (int i=0;i<inputNum;i++)
		weight.push_back(rand()/RAND_MAX);		
}


void MLPCell::FeedForward(){
	double sum = bias;
	for (int i=0;i<input.size();i++)
			sum += input[i]*weight[i]; // dot product
	output = sigmoid(sum);
}

void MLPCell::BackPropagate(double derr){
	delta = derr*dSigmoid(output);
	for (int i=0;i<inerr.size();i++)
		inerr[i] = delta * weight[i];
}

void MLPCell::AdjustWeight(double lr){
	bias += delta*lr;
	for (int i=0;i<weight.size();i++)
		weight[i] += input[i]*delta*lr;
}

//=================== MLP ========================

MLP::MLP(int inputNum,int hiddenNum, int hiddenNum2, int outputNum,double threshold){
	hiddenLayer.resize(hiddenNum,MLPCell(inputNum));
	hiddenLayer2.resize(hiddenNum2,MLPCell(hiddenNum));
	outputLayer.resize(outputNum,MLPCell(hiddenNum2));
	input.resize(inputNum);
	output.resize(outputNum);
	myThreshold = threshold;
}


bool MLP::Training(double trainingInput[], double trainingOutput[]){
	// cout << *(&trainingInput + 1) - trainingInput;
	// if (sizeof(trainingInput)/sizeof(trainingInput[0]) != input.size() || 
	//     sizeof(trainingOutput)/sizeof(trainingOutput[0]) != output.size()) {
	//     	cout << "Training data range not match!!" << endl;
	//     	return false;
	// } 
	for (int i=0;i<input.size();i++)
		input[i]=trainingInput[i];	
	do{
		Testing();
		double sumerr=0;		
		for (int i=0;i<outputLayer.size();i++)	
			sumerr+=abs(trainingOutput[i]-output[i]);
		if(sumerr==0) break;
		for(int i=0;i<outputLayer.size();i++){
			outputLayer[i].BackPropagate(trainingOutput[i]-outputLayer[i].output);
			outputLayer[i].AdjustWeight(0.01);			
		}
		for(int i=0;i<hiddenLayer2.size();i++){
			double sumInerr=0;
			for(int j=0;j<outputLayer.size();j++)
				sumInerr+=outputLayer[j].inerr[i];
			hiddenLayer2[i].BackPropagate(sumInerr);	
			hiddenLayer2[i].AdjustWeight(0.01);
		}	
		for(int i=0;i<hiddenLayer.size();i++){
			double sumInerr=0;
			for(int j=0;j<hiddenLayer2.size();j++)
				sumInerr+=hiddenLayer2[j].inerr[i];
			hiddenLayer[i].BackPropagate(sumInerr);	
			hiddenLayer[i].AdjustWeight(0.01);
		}	
	}while(true);		
	return true;	
}

void MLP::Testing(){	
	// -------------- testing hedden layer -------------------
	for (int i=0;i<hiddenLayer.size();i++){
		for(int j=0;j<input.size();j++) 
			hiddenLayer[i].input[j]=input[j];	
		hiddenLayer[i].FeedForward();	
	}
	// -------------- testing hedden layer 2 -------------------
	for (int i=0;i<hiddenLayer2.size();i++){
		for(int j=0;j<hiddenLayer.size();j++) 
			hiddenLayer2[i].input[j]=hiddenLayer[j].output;	
		hiddenLayer2[i].FeedForward();	
		// output[i] = Step(outputLayer[i].output);
	}
	//--------------- testing output layer -------------------
	for (int i=0;i<outputLayer.size();i++){
		for(int j=0;j<hiddenLayer2.size();j++) 
			outputLayer[i].input[j]=hiddenLayer2[j].output;	
		outputLayer[i].FeedForward();	
		output[i] = Step(outputLayer[i].output);
	}
}

void MLP::SaveWeight(string FileName) {
	// write file signature
	// write shapes of Input, Hidden, Output Layer
	ofstream saveFile(FileName);
	saveFile << "=== MLP Weight ===" << endl;

	for (int i=0;i<hiddenLayer.size();i++){
		for(int j=0;j<hiddenLayer[i].weight.size();j++) 
			saveFile << hiddenLayer[i].weight[j] << " ";
		saveFile << endl;
	}
	for (int i=0;i<hiddenLayer2.size();i++){
		for(int j=0;j<hiddenLayer2[i].weight.size();j++) 
			saveFile << hiddenLayer2[i].weight[j] << " ";
		saveFile << endl;
	}
	for (int i=0;i<outputLayer.size();i++){
		for(int j=0;j<outputLayer[i].weight.size();j++) 
			saveFile << outputLayer[i].weight[j] << " ";
		saveFile << endl;
	}
	saveFile << "=== endfile ===" << endl;
	
	// loop layer in layers
		// loop neuron in layer
			// loop weight in neuron
	saveFile.close();
	cout << "Saved" << endl;

}

void MLP::LoadWeight(string FileName){
	string line;
	ifstream saveFile(FileName);
	
	getline (saveFile, line);
	if (line != "=== MLP Weight ===") cout << "This is not a Weight";
	for (int i=0;i<hiddenLayer.size();i++){
		getline (saveFile, line);
		size_t pos = 0;
		string token;
		double value;
		string delimiter = " ";
			for(int j=0;j<hiddenLayer[i].weight.size();j++){
				if ((pos = line.find(delimiter)) != string::npos && line != "=== endfile ==="){
					token = line.substr(0, pos);
					// cout << token << endl;
					value = stod(token);
					hiddenLayer[i].weight[j]=value;
					line.erase(0, pos + delimiter.length());
					// cout << i << " and " << j << endl;
				}
			}
	}
	for (int i=0;i<outputLayer.size();i++){
		getline (saveFile, line);
		size_t pos = 0;
		string token;
		double value;
		string delimiter = " ";
			for(int j=0;j<outputLayer[i].weight.size();j++){
				if ((pos = line.find(delimiter)) != string::npos && line != "=== endfile ==="){
					token = line.substr(0, pos);
					// cout << token << endl;
					value = stod(token);
					outputLayer[i].weight[j]=value;
					line.erase(0, pos + delimiter.length());
					// cout << i << " and " << j << endl;
				}
			}
		}
}



// note: things after // are comments that won't be in the file
/*
=== MLP Weight ===
0.3 0.4 // hidden neuron 1
0.3 0.4 // hidden neuron 2
0.3 0.4 // hidden neuron 3
0.3 0.4 // hidden neuron 4
0.3 0.4 // output neuron 1
0.3 0.4 // output neuron 2
=== endfile ===
*/