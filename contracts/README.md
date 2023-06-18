## Contracts 

## Goals
1. Write a solidity contract generator for a given machine learning classifier description
2. Write a Pytorch -> Solidity Interpreter (Init + Forward)
3. Find some mathematical optimization in a basic classifier architecture to preserve prediction accuracy on a smart contract
4. Find a use case for classifiers in DeFi

## Classifier
- used ABDK64x64 int128 to get around floating point computations

## Classifiers
- CNN
- log_regression
- perceptron (SVM and perceptron have essentially the same forward pass while they only differ in training (kernel trick)) 
- (knn, centroid, etc. classifiers are too simple)
- adaboost, xgboost have basic classifier architectures but perform different steps in fitting to reduce errors on bad cases/overfitting
- We can use transformers/ViTs for zero-shot text/image classification but quality models are too large for smart contracts
