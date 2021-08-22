static const int numTrainingSets = 4;
static const int numInputs=4;
static const int numOutputs=2;

double training_inputs[numTrainingSets][numInputs] = { 
    {1.0,0.0,0.0,0.0},
    {0.0,1.0,0.0,0.0},
    {0.0,0.0,1.0,0.0},
    {0.0,0.0,0.0,1.0} 
};
double training_outputs[numTrainingSets][numOutputs] = { 
    {0.0,0.0},
    {0.0,1.0},
    {1.0,0.0},
    {1.0,1.0} 
};
