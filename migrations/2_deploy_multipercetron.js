var MultiPerceptron = artifacts.require("MultiPerceptron");

module.exports = function(deployer) {
  // Pass 10 as the constructor argument for `input_dim`
  deployer.deploy(MultiPerceptron, 10);
};
