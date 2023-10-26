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

    function array_ops() public pure returns (int256[10] memory result) {
        int256[] storage data = [1,2,3,4,5,6,7,8,9,10];
        int256[10] memory prb_data = [data.length];
        for (uint256 i = 0; i < data.length; i++) {
            prb_data[i] =convert(data[i]);
        }
        result = prb_data;
    }
}

