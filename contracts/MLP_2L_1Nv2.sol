// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;
import {SD59x18, convert, sd} from "../lib/prb-math/src/SD59x18.sol";

contract MLP_2L_1n {
    int256[][] public weights_layer1;
    int256[][] public weights_layer2;
    int256[][] public biases;
    int256[][] public training_data;
    int public correct_Count;

    function setfc1(int[] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            fc1[i] = value[i];
        }
    }

    function setfc2(int[] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            fc2[i] = value[i];
        }
    }

    function sigmoid(SD59x18 x) public pure returns (SD59x18) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((-x).exp()));
    }

    constructor(uint256 input_dim) {
        biases = new int256[][](input_dim);
        biases[0] = new int256[](1);
        biases[1] = new int256[](1);
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

    // function classify(int[] memory x) public view returns (int[] memory) {
    //       int[] memory res1 = new int[](1);
    //       int c = 0;
    //       for (uint256 i = 0; i < i; ++i) {
    //           c += fc1[i] * x[i];
    //       }
    //       res1[0] = c;
    //       //relu activation function
    //       function relu(SD59x18 x) public pure returns (SD59x18) {
    //           int256 zero = 0;
    //           SD59x18 zero_cvt = convert(zero);
    //           if (x.gte(zero_cvt)) {
    //               return x;
    //           }
    //           return zero_cvt;
    //       }

    //       int[] memory res2 = new int[](1);
    //       int c = 0;
    //       for (uint256 i = 0; i < 1; ++i) {
    //           c += fc2[i] * res1[i];
    //       }
    //       res2[0] = c;
    //       for (uint256 i = 0; i < res.length; ++i) {
    //           res[i] = sigmoid(res[i]);
    //       }
    //       return res;

    // }
}
