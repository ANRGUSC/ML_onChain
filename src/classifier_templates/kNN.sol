// SPDX-License-Identifier: UNLICENSED 
pragma solidity ^0.8.0;

import "../libraries/ABDKMath64x64.sol";

contract KNN {
    using ABDKMath64x64 for int128;

    int128[][] public trainingDataX;
    int128[] public trainingDataY;

    function train(int128[][] memory X, int128[] memory y) public {
        require(X.length == y.length, "Input dimensions do not match.");

        for (uint i = 0; i < X.length; i++) {
            trainingDataX.push(X[i]);
            trainingDataY.push(y[i]);
        }
    }

    function predict(int128[] memory x) public view returns (int128) {
        require(trainingDataX.length > 0, "No training data available.");

        uint bestIndex = 0;
        int128 bestDistance = computeDistance(x, trainingDataX[0]);

        for (uint i = 1; i < trainingDataX.length; i++) {
            int128 distance = computeDistance(x, trainingDataX[i]);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return trainingDataY[bestIndex];
    }

    function computeDistance(int128[] memory x1, int128[] memory x2) private pure returns (int128) {
        require(x1.length == x2.length, "Input dimensions do not match.");

        int128 distance = ABDKMath64x64.fromInt(0);

        for (uint i = 0; i < x1.length; i++) {
            int128 diff = x1[i].sub(x2[i]);
            distance = distance.add(diff.mul(diff));
        }

        return distance.sqrt();
    }
}
