#include "MLP.h"
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

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
/*	if (sizeof(trainingInput)/sizeof(double) != input.size() || 
	    sizeof(trainingOutput)/sizeof(double) != output.size()) {
	    	cout << "input=" << sizeof(trainingInput)/sizeof(double) <<endl;
	    	cout << "output=" << sizeof(trainingOutput)/sizeof(double) <<endl;
	    	cout << "Training data range not match!!" << endl;
	    	return false;
	} */
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

void MLP::SaveWeight(string FileName){
	
} 

void MLP::LoadWeight(string FileName){
	
}


/*
bool Graph::SaveGraph(string FileName){
	bool result = true;
	try{
		ofstream Datafile(FileName.data());
		if (!Datafile.is_open())
			throw(201);
		else {
			Datafile << "#== Graph Link Data File ==#" << endl;
			for (int i=0; i<nodeList.size();i++) {
				Datafile << nodeList[i].Name() << endl;
				for (int j=0;j<nodeList[i].linkTo.size() ;j++)
					Datafile << nodeList[i].linkTo[j].index << " " << nodeList[i].linkTo[j].weight << endl;	
				Datafile <<	"#----------#" << endl;		
			}
		}
	} catch(int e) {
		cout <<  "exception: " << ErrorCase(e) << '\n';
		result =false;		
	}
	
	return result;
}

bool Graph::LoadGraph(string FileName){
	bool result = true;
	try	{
		ifstream Datafile(FileName.data());
   		if (!Datafile.is_open())
   			throw (101);
		else {   
      		string tp;
      		getline(Datafile, tp);
      		if(tp.compare("#== Graph Link Data File ==#")!=0) throw(102);
      		int index;
      		double weight;
      		while(getline(Datafile, tp)){
      			nodeList.push_back(node(tp));
				while(getline(Datafile, tp)){ 
         			if(tp.compare("#----------#")!=0){
         				stringstream LinkInfo(tp);
         				LinkInfo >> index;
						LinkInfo >> weight;  
						nodeList.back().SetLink(index,weight);
				 	}
         			else break;
            	}	
			}
      	}
      	Datafile.close(); 
	} catch(int e) {
		cout <<  "exception: " << ErrorCase(e) << '\n';
		nodeList.clear();
		result =false;
	}
	return result;
}*/
