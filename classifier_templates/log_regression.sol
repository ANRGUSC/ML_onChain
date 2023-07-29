// SPDX-License-Identifier: UNLICENSED 
//single feature binary logistic regression 

pragma solidity ^0.8.0;

import "../libraries/ABDKMath64x64.sol";

contract LogisticRegression {
    using ABDKMath64x64 for int128;

    int128 public weight;
    int128 public bias;

    int128 private lr;
    uint256 private epochs;

    constructor(int128 _learningRate, uint256 _epochs) {
        lr = _learningRate;
        epochs = _epochs;
        weight = ABDKMath64x64.fromInt(0);
        bias = ABDKMath64x64.fromInt(0);
    }

    function fit(int128[] memory X, int128[] memory y) public {
        require(X.length == y.length, "Input dimensions do not match.");

        for (uint256 epoch = 0; epoch < epochs; epoch++) {
            for (uint256 i = 0; i < X.length; i++) {
                int128 predicted = predict(X[i]);
                int128 error = y[i].sub(predicted);
                
                weight = weight.add(lr.mul(X[i].mul(error)));
                bias = bias.add(lr.mul(error));
            }
        }
    }

    function predict(int128 X) public view returns (int128) {
        int128 linear_model = X.mul(weight).add(bias);
        return ABDKMath64x64.div(ABDKMath64x64.fromInt(1), ABDKMath64x64.exp(ABDKMath64x64.neg(linear_model)));
    }
}
