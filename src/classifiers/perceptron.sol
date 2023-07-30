// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

contract Perceptron {
	int[] public fc;

	
    function setfc(int[] memory value) public {
        for (uint i = 0; i < value.length; ++i) {
            fc[i] = value[i];
        }
    }

	constructor(uint input_dim) {
 		fc = new int[](input_dim);
	}

	function predict(int[] memory x) public view returns (int[] memory) {	
        int[] memory res1 = new int[](1);
        int c = 0;
        for (uint i = 0; i < i; ++i) {
            c += fc[i] * x[i];
        }
        res1[0] = c;
        for (uint i = 0; i < res1.length; ++i) {
            res1[i] = ((res1[i] >= 0) ? ((res1[i] == 0) ? int(0) : int(1)) : -1);
        }
        return res1;
        
	}
 }