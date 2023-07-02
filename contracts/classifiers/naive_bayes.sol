// SPDX-License-Identifier: UNLICENSED 
pragma solidity ^0.8.0;

import "../libraries/ABDKMath64x64.sol";

contract NaiveBayes {
    using ABDKMath64x64 for int128;

    int128 public probPos;
    int128 public probNeg;
    int128[] public probFeaturePos;
    int128[] public probFeatureNeg;

    function train(int128[] memory X, int128[] memory y) public {
        require(X.length == y.length, "Input dimensions do not match.");

        int128 countPos = ABDKMath64x64.fromInt(0);
        int128 countNeg = ABDKMath64x64.fromInt(0);

        for (uint256 i = 0; i < X.length; i++) {
            if (y[i] == ABDKMath64x64.fromInt(1)) {
                countPos = countPos.add(ABDKMath64x64.fromInt(1));
                probFeaturePos[i] = probFeaturePos[i].add(X[i]);
            } else {
                countNeg = countNeg.add(ABDKMath64x64.fromInt(1));
                probFeatureNeg[i] = probFeatureNeg[i].add(X[i]);
            }
        }

        probPos = countPos.div(ABDKMath64x64.fromInt(int256(X.length)));
        probNeg = countNeg.div(ABDKMath64x64.fromInt(int256(X.length)));

        for (uint256 i = 0; i < X.length; i++) {
            probFeaturePos[i] = probFeaturePos[i].div(countPos);
            probFeatureNeg[i] = probFeatureNeg[i].div(countNeg);
        }
    }

    function predict(int128[] memory X) public view returns (int128) {
        int128 logProbPos = probPos.ln();
        int128 logProbNeg = probNeg.ln();

        for (uint256 i = 0; i < X.length; i++) {
            logProbPos = logProbPos.add(X[i].mul(probFeaturePos[i].ln()));
            logProbNeg = logProbNeg.add(X[i].mul(probFeatureNeg[i].ln()));
        }

        return logProbPos > logProbNeg ? ABDKMath64x64.fromInt(1) : ABDKMath64x64.fromInt(0);
    }
}
