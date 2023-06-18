pragma solidity ^0.8.0;
import "../libraries/ABDKMath64x64.sol";

contract SimpleSVM {
    using ABDKMath64x64 for int128;

    int128[] public weights;
    int128 public bias;

    int128 private lr;
    int128 private lambda;
    uint256 private epochs;

    constructor(int128 _learningRate, int128 _lambdaParam, uint256 _epochs) {
        lr = _learningRate;
        lambda = _lambdaParam;
        epochs = _epochs;
        bias = ABDKMath64x64.fromInt(0);
    }

    function fit(int128[] memory X, int128[] memory y) public {
        require(X.length == y.length, "Input dimensions do not match.");
        
        // Initialize weights to zero
        for (uint i = 0; i < X.length; i++) {
            weights.push(ABDKMath64x64.fromInt(0));
        }

        for (uint256 epoch = 0; epoch < epochs; epoch++) {
            for (uint256 idx = 0; idx < X.length; idx++) {
                int128 condition = y[idx].mul(X[idx].mul(weights[idx]).sub(bias));
                
                if (condition >= ABDKMath64x64.fromInt(1)) {
                    weights[idx] = weights[idx].sub(lr.mul(weights[idx].mul(lambda).mul(ABDKMath64x64.fromInt(2))));
                } else {
                    weights[idx] = weights[idx].sub(
                        lr.mul(weights[idx].mul(lambda).mul(ABDKMath64x64.fromInt(2)).sub(X[idx].mul(y[idx])))
                    );
                    bias = bias.sub(lr.mul(y[idx]));
                }
            }
        }
    }

    function predict(int128 X) public view returns (int8) {
        int128 result = X.mul(weights[0]).sub(bias);
        return result > ABDKMath64x64.fromInt(0) ? int8(1) : int8(-1);
    }
}