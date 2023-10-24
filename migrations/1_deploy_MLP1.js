const MLP_1 = artifacts.require("MLP_1");

module.exports = function(deployer) {
   deployer.deploy(MLP_1,30,1);
};
