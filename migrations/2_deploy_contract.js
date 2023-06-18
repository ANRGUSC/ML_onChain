var Hero = artifacts.require("Hero");
module.exports = function(deployer) {
  deployer.deploy(Hero,"Hulk");
};