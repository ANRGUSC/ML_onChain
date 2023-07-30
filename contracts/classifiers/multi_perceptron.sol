// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

contract MultiPerceptron {
	int[][] public fc;
	int[][] public fc2;
	int[] public fc3;

	function setfc(int[][] memory value) public {
        for (uint i = 0; i < value.length; ++i) {
            for (uint j = 0; j < value[0].length; ++j) {
                fc[i][j] = value[i][j];
            }
        }
    }

	function setfc2(int[][] memory value) public {
        for (uint i = 0; i < value.length; ++i) {
            for (uint j = 0; j < value[0].length; ++j) {
                fc2[i][j] = value[i][j];
            }
        }
    }

	
    function setfc3(int[] memory value) public {
        for (uint i = 0; i < value.length; ++i) {
            fc3[i] = value[i];
        }
    }

	constructor(uint input_dim) {
 		fc = new int[][](input_dim);
		for (uint i = 0; i < input_dim; i++) {
			fc[i] = new int[](2);
		}

		fc2 = new int[][](2);
		for (uint i = 0; i < 2; i++) {
			fc2[i] = new int[](2);
		}

		fc3 = new int[](2);
	}

	function predict(int[] memory x) public view returns (int[] memory) {	
        int[] memory res1 = new int[](2);
        int c;
        for (uint i = 0; i < 2; ++i) {
            c = 0;
            for (uint j = 0; j < x.length; ++j) {
                c += fc[i][j] * x[j];
            }
            res1[i] = c;
        }
        int[] memory res2 = new int[](2);
        c;
        for (uint i = 0; i < 2; ++i) {
            c = 0;
            for (uint j = 0; j < x.length; ++j) {
                c += fc2[i][j] * res1[j];
            }
            res2[i] = c;
        }
        int[] memory res3 = new int[](1);
        c = 0;
        for (uint i = 0; i < 2; ++i) {
            c += fc3[i] * res2[i];
        }
        res3[0] = c;
        for (uint i = 0; i < res3.length; ++i) {
            res3[i] = ((res3[i] >= 0) ? ((res3[i] == 0) ? int(0) : int(1)) : -1);
        }
        return res3;
        
	}
 }