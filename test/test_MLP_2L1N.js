const fs = require('fs');
const MLP_2L_1N = artifacts.require("MLP_2L_1N.sol");
const fsPromises = fs.promises;

// gas cost of diff functions
let gas_classify = 0;
let gas_upload_weightBias = 0;
let gas_deployment = 0;
let gas_upload_testData = 0;

const {array_to_PRB, array_from_PRB, classify,upload_trainingData, upload_weightsBiases} = require('./util_functions.js');

//saves log into a file
const originalConsoleLog = console.log;
console.log = function (...args) {
    originalConsoleLog.apply(console, args);  // This will ensure the logs still display in the console
    fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
};

contract("MLP_2L_1N.sol", accounts => {
    let instance;

    before(async () => {
        instance = await MLP_2L_1N.new(2);
        gas_deployment += await MLP_2L_1N.new.estimateGas(2)
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });


    it("Upload weights and biases", async () => {
        gas_upload_weightBias += await upload_weightsBiases(instance, fsPromises,'MLP_2L1.json',2);
    });

    it("Upload training data", async () => {
        gas_upload_testData += await upload_trainingData(instance, fsPromises)
    });

    it("Classify", async () => {
        gas_classify += await classify(instance);
    });


    after(() => {
        console.log('Name: MLP_2L_1n');
        console.log(`Deployment Gas: ${gas_deployment}`);
        console.log(`Test data upload gas: ${gas_upload_testData}`);
        console.log(`Weights and biases upload gas: ${gas_upload_weightBias}`);
        console.log(`Classify gas: ${gas_classify}\n`);
   });

});

