// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import { SD59x18, convert } from "../../lib/prb-math/src/SD59x18.sol";



contract MLP_Test {

    function multiply(SD59x18 x,SD59x18 y) external pure returns (SD59x18 result) {
    result = x.mul(y);
    }

    function test_convert() external pure returns (SD59x18 result) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        result = one_cvt;
    }

    function test_cast() external pure returns (SD59x18 result) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        result = one_cvt;
    }

    function sigmoid(int x) public pure returns (SD59x18) {
        SD59x18 y = convert(x);
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((-y).exp()));
    }

    function array_ops() public pure returns (SD59x18 result) {
        int256[9] memory data;
        int256[9] memory weights;
        result = convert(0);
        for (uint256 i = 0; i < data.length; i++) {
            data[i] = int256(i*1e18);
        }
        for (uint256 i = 0; i < data.length; i++) {
            weights[i] = int256(i*1e18);
        }

        for (uint256 i = 0; i < data.length; i++) {
            SD59x18 a = SD59x18.wrap(data[i]);
            SD59x18 b = SD59x18.wrap(weights[i]); // Subtract 1 from i to match weights index
            result = result.add(a.mul(b)); // Subtract 1 from i to match weights index
        }
        return result;
        /*
        result = result.add(SD59x18.wrap(biases[0]));
        int classification;
        SD59x18 point_five = sd(0.5e18);
        if (sigmoid(result).gte(point_five)) { // If sigmoid(result) >= 0.5
            classification = 1;
        } else {
            classification = 0;
        }
        if (label == classification) {
           correct++;  // Correctly classified
        }
        correct_Count = correct;
        */
    }
}

