pragma solidity >=0.4.22 <0.9.0;

contract Perceptron {

    int[] public weights;
    int public bias;

    constructor(int[5] memory initial_weights, int initial_bias) public {
        weights = initial_weights;
        bias = initial_bias;
    }

    function predict(int[5] memory inputs) public view returns (int8) {
        require(inputs.length == weights.length, "Length of inputs does not match length of weights");

        int sum = bias;
        for(uint i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }

        return (sum >= 0) ? int8(1) : -1;
    }

    function updateWeights(int[5] memory new_weights) public {
        require(new_weights.length == weights.length, "Length of new_weights does not match current weights");
        weights = new_weights;
    }

    function updateBias(int new_bias) public {
        bias = new_bias;
    }
}
