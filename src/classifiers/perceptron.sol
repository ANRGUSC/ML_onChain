pragma solidity >=0.5.0 <0.9.0;

contract Perceptron {
    int256 public threshold;
    int256[] public weights;

    constructor(int256 _threshold, int256[] memory _weights) public {
        threshold = _threshold;
        weights = _weights;
    }

    function predict(int256[] memory inputs) public view returns (int256) {
        require(inputs.length == weights.length, "Inputs and weights must have the same length");

        int256 sum = 0;
        for(uint256 i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        if(sum > threshold) {
            return 1;
        } else {
            return -1;
        }
    }

    function updateWeights(int256[] memory newWeights) public {
        require(newWeights.length == weights.length, "New weights must have the same length as current weights");
        weights = newWeights;
    }

    function updateThreshold(int256 newThreshold) public {
        threshold = newThreshold;
    }
}
