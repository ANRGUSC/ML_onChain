var MyContract = artifacts.require("HelloWorld");

module.exports = function(deployer) {
  deployer.deploy(MyContract, "Hello world!")
    .then(() => console.log("Contract deployed successfully!"))
    .catch(err => console.error("Contract deployment failed:", err));
};
