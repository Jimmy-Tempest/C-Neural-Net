#include "MLP.h"
#include <time.h>
#include <stdlib.h>
#include <iostream>

#define LearningRate 0.1

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
			sum += input[i]*weight[i];
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

MLP::MLP(int inputNum,int hiddenNum,int outputNum,double threshold){
	hiddenLayer.resize(hiddenNum,MLPCell(inputNum));
	outputLayer.resize(outputNum,MLPCell(hiddenNum));
	input.resize(inputNum);
	output.resize(outputNum);
	myThreshold = threshold;
}


bool MLP::Training(double trainingInput[], double trainingOutput[]){
	if (sizeof(trainingInput)/sizeof(double) != input.size() || 
	    sizeof(trainingOutput)/sizeof(double) != output.size()) {
	    	cout << "Training data range not match!!" << endl;
	    	return false;
	} 
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
		for(int i=0;i<hiddenLayer.size();i++){
			double sumInerr=0;
			for(int j=0;j<outputLayer.size();j++)
				sumInerr+=outputLayer[j].inerr[i];
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
	//--------------- testing output layer -------------------
	for (int i=0;i<outputLayer.size();i++){
		for(int j=0;j<hiddenLayer.size();j++) 
			outputLayer[i].input[j]=hiddenLayer[j].output;	
		outputLayer[i].FeedForward();	
		output[i] = Step(outputLayer[i].output);
	}
}
