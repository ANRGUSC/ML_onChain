// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import { SD59x18 } from "../../lib/prb-math/src/SD59x18.sol";

contract MLP_Test {

    function multiply(SD59x18 x,SD59x18 y) external pure returns (SD59x18 result) {
    result = x.mul(y);
  }

}

