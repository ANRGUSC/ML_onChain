//SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;
import {SD59x18, convert, sd} from "../lib/prb-math/src/SD59x18.sol";

contract MLP_2L_3N {
    int256[][] public weights_layer1; // 2D array for weights
    int256[][] public weights_layer2;
    int256[][] public biases; // 2D array for biases
    int256[][] public training_data;
    int public correct_Count;

    constructor(uint256 neurons) {
        biases = new int256[][](neurons);
        biases[0] = new int256[](3); // 1 neuron in the first layer
        biases[1] = new int256[](1); // 1 neuron in the second layer
    }

    function set_Biases(uint256 layer, int256[] calldata b) external {
        require(
            b.length == biases[layer].length,
            "Size of input biases does not match neuron number"
        );
        biases[layer] = b;
    }

    function set_Weights(uint256 layer, int256[] calldata w) external {
        require(layer < 2, "Layer index out of bounds");
        int256[] memory temp_w = new int256[](w.length);
        for (uint256 i = 0; i < w.length; i++) {
            temp_w[i] = w[i];
        }
        if (layer == 0) {
            weights_layer1.push(temp_w);
        } else if (layer == 1) {
            weights_layer2.push(temp_w);
        }
    }

    function view_dataset_size() external view returns (uint256 size) {
        size = training_data.length;
    }

    function set_TrainingData(int256[] calldata d) external {
        int256[] memory temp_d = new int256[](d.length);
        for (uint256 i = 0; i < d.length; i++) {
            temp_d[i] = d[i];
        }
        training_data.push(temp_d);
    }

    // calculate the sigmoid of x
    function sigmoid(SD59x18 x) public pure returns (SD59x18) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((-x).exp()));
    }

    //relu activation function
    function relu(SD59x18 x) public pure returns (SD59x18) {
        int256 zero = 0;
        SD59x18 zero_cvt = convert(zero);
        if (x.gte(zero_cvt)) {
            return x;
        }
        return zero_cvt;
    }

    function classify() public view returns (int) {
        int correct = 0;

        for (uint256 j = 0; j < 50; j++) {
            int256[] memory data = training_data[j];
            int256 label = data[0];

            // Neuron results for Layer 1
            SD59x18[] memory neuronResultsLayer1 = new SD59x18[](3);

            for (uint256 n = 0; n < 3; n++) {
                neuronResultsLayer1[n] = SD59x18.wrap(biases[0][n]);
                for (uint256 i = 1; i < data.length; i++) {
                    neuronResultsLayer1[n] = neuronResultsLayer1[n].add(
                        SD59x18.wrap(data[i]).mul(
                            SD59x18.wrap(weights_layer1[n][i - 1])
                        )
                    );
                }
                neuronResultsLayer1[n] = relu(neuronResultsLayer1[n]);
            }
            //-------------------------------------------------------------------------
            // Neuron results for Layer 2
            SD59x18 neuronResultLayer2 = SD59x18.wrap(biases[1][0]);
            for (uint256 n = 0; n < 3; n++) {
                neuronResultLayer2 = neuronResultLayer2.add(
                    neuronResultsLayer1[n].mul(
                        SD59x18.wrap(weights_layer2[0][n])
                    )
                );
            }
            neuronResultLayer2 = sigmoid(neuronResultLayer2);
            //-------------------------------------------------------------------------
            int256 classification;
            SD59x18 point_five = sd(0.5e18);
            if (neuronResultLayer2.gte(point_five)) {
                classification = int256(1e18);
            } else {
                classification = int256(0e18);
            }

            if (label == classification) {
                correct++;
            }
        }
        return correct;
    }
}
