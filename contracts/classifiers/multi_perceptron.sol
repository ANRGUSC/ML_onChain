// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

contract MultiPerceptron {
	int[][] public fc;
	int[][] public fc2;
	int[][] public fc3;
	int[] public fc4;

	function setfc(int[][] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            for (uint256 j = 0; j < value[0].length; ++j) {
                fc[i][j] = value[i][j];
            }
        }
    }

	function setfc2(int[][] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            for (uint256 j = 0; j < value[0].length; ++j) {
                fc2[i][j] = value[i][j];
            }
        }
    }

	function setfc3(int[][] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            for (uint256 j = 0; j < value[0].length; ++j) {
                fc3[i][j] = value[i][j];
            }
        }
    }

	
    function setfc4(int[] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            fc4[i] = value[i];
        }
    }

	constructor(uint256 input_dim) {
 		fc = new int[][](input_dim);
		for (uint256 i = 0; i < input_dim; i++) {
			fc[i] = new int[](2);
		}

		fc2 = new int[][](2);
		for (uint256 i = 0; i < 2; i++) {
			fc2[i] = new int[](2);
		}

		fc3 = new int[][](2);
		for (uint256 i = 0; i < 2; i++) {
			fc3[i] = new int[](2);
		}

		fc4 = new int[](2);
	}

	function predict(int[] memory x) public view returns (int[] memory) {	
        int[] memory res1 = new int[](2);
        int c;
        for (uint256 i = 0; i < 2; ++i) {
            c = 0;
            for (uint256 j = 0; j < x.length; ++j) {
                c += fc[i][j] * x[j];
            }
            res1[i] = c;
        }
        int[] memory res2 = new int[](2);
        c;
        for (uint256 i = 0; i < 2; ++i) {
            c = 0;
            for (uint256 j = 0; j < x.length; ++j) {
                c += fc2[i][j] * res1[j];
            }
            res2[i] = c;
        }
        int[] memory res3 = new int[](2);
        c;
        for (uint256 i = 0; i < 2; ++i) {
            c = 0;
            for (uint256 j = 0; j < x.length; ++j) {
                c += fc3[i][j] * res2[j];
            }
            res3[i] = c;
        }
        int[] memory res4 = new int[](1);
        c = 0;
        for (uint256 i = 0; i < 2; ++i) {
            c += fc4[i] * res3[i];
        }
        res4[0] = c;
        for (uint256 i = 0; i < res4.length; ++i) {
            if (res4[i] < 0) {
                res4[i] = 0;
            }
        }
        return res4;
        
	}
 }