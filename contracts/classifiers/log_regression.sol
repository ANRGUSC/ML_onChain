// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

contract LogisticRegressionModel {
	int[] public linear;

	
    function setlinear(int[] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            linear[i] = value[i];
        }
    }

	
    function sigmoid(int x) public pure returns (int64) {
        int64 x64 = ABDKMath64x64.fromInt(x);

        // Now, we compute the negative of x64.
        int64 negX64 = ABDKMath64x64.neg(x64);

        // Then, we compute e^(negX64).
        int64 expNegX64 = ABDKMath64x64.exp(negX64);

        // Next, we add 1 to expNegX64. 
        int64 onePlusExpNegX64 = ABDKMath64x64.add(ABDKMath64x64.fromInt(1), expNegX64);

        // Finally, we compute the reciprocal of onePlusExpNegX64, which gives us the result of the sigmoid function.
        int64 sigmoidResult = ABDKMath64x64.inv(onePlusExpNegX64);

        return sigmoidResult;
    }
            

	constructor(uint256 n_features) {
 		linear = new int[](n_features);
	}

	function predict(int[] memory x) public view returns (int[] memory) {	
        int[] memory res1 = new int[](1);
        int c = 0;
        for (uint256 i = 0; i < n; ++i) {
            c += linear[i] * x[i];
        }
        res1[0] = c;
        for (uint256 i = 0; i < res.length; ++i) {
            res[i] = sigmoid(res[i]);
        }
        return res;
        
	}
 }