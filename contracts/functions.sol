// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

import {SD59x18, convert, sd} from "../lib/prb-math/src/SD59x18.sol";

contract functions {

    function sigmoid(SD59x18 x) public pure returns (SD59x18) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((- x).exp()));
    }

    function relu(SD59x18 x) public pure returns (SD59x18) {
        int256 zero = 0;
        SD59x18 zero_cvt = convert(zero);
        if (x.gte(zero_cvt)) {
            return x;
        }
        return zero_cvt;
    }

    // sol native signed add
    function add_1(int x, int y) public pure returns (int) {
        return x + y;
    }
    function add_2(int x, int y) public pure returns (int) {
        return x + y+ y;
    }

    // PRBMath add
    function add_prb_1(SD59x18 x, SD59x18 y) public pure returns(SD59x18){
        return x.add(y);
    }
    function add_prb_2(SD59x18 x, SD59x18 y) public pure returns(SD59x18){
        return x.add(y).add(y);
    }

    // sol native mul
    function mul_1(int x, int y) public pure returns (int){
        return x * y;
    }
    function mul_2(int x, int y) public pure returns (int){
        return x * y*y;
    }
    // PRBMath mul
    function mul_prb_1(SD59x18 x, SD59x18 y) public pure returns(SD59x18){
        return x.mul(y);
    }

    function mul_prb_2(SD59x18 x, SD59x18 y) public pure returns(SD59x18){
        return x.mul(y).mul(y);
    }

}