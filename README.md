# ECHO
This project aims to:
1. Write a solidity contract generator for a given machine learning classifier description
2. Write a Pytorch -> Solidity Interpreter (Init + Forward)
3. Find some mathematical optimization in a basic classifier architecture to preserve prediction accuracy on a smart contract
4. Find a use case for classifiers in DeFi

## Sol classifiers
current available sol classifiers includes KNN, log_regression, naive_bayes, perceptron, simpleMLP, and SVM

## Interpreter 
The current version of the interpreter is capable of translating logistic regression and perceptron 

## Model deployment 

## Running Python Model
The Python implementation of the model is located in the ML_Models folder. Run the model with the following command
```bash
python train.py
```
It will train the models with provided datasets and save the weights into json files

## Translating Python file to Sol
```bash
python translate.py
```

## Floating Point Computations
used ABDK64x64 int128 due to sol's lack of native floating-point computation support 
