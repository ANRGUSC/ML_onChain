const MLP_1 = artifacts.require("MLP_1");
const ABDKMath64x64 = artifacts.require("ABDKMath64x64");

module.exports = async function(deployer) {
  // Deploy the ABDK library
  await deployer.deploy(ABDKMath64x64);

  // Link the ABDK library to your contract
  deployer.link(ABDKMath64x64, MLP_1);

  // Deploy your contract
  await deployer.deploy(MLP_1,33);
};
