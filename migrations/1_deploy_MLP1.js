const MLP_1 = artifacts.require("MLP_1L_1N.sol");

module.exports = function(deployer) {
   deployer.deploy(MLP_1,1);
};
