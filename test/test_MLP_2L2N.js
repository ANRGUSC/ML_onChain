const fs = require('fs');
const MLP_2L_2n = artifacts.require("MLP_2L_2n.sol");
const fsPromises = fs.promises;

let gas_classify = 0;
let gas_upload_weightBias = 0;
let gas_deployment = 0;
let gas_upload_testData = 0;
const {array_to_PRB,upload_weight_biases} = require('./util_functions.js');

//saves log into a file
const originalConsoleLog = console.log;
console.log = function (...args) {
    originalConsoleLog.apply(console, args);  // This will ensure the logs still display in the console
    fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
};

contract("MLP_2L_2n.sol", accounts => {
    let instance;

    before(async () => {
        instance = await MLP_2L_2n.new(3);
        gas_deployment += await MLP_2L_2n.new.estimateGas(3)
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });


    it("Upload weights and biases", async () => {
        gas_upload_weightBias += upload_weight_biases(instance,2,'MLP_2L2.json');
    });


    it("Upload training data", async () => {
        try {
            const data = await fsPromises.readFile('./src/data/processed_data.csv', 'utf8');
            const lines = data.split('\n');
            for (let i = 1; i <= 50 && i < lines.length; i++) { // Starting from 1 to skip header
                const line = lines[i];
                const splitData = line.split(',');

                // Convert "diagnosis" column to binary
                const diagnosisBinary = +splitData[1];

                // Drop the first column (ID) and replace the "diagnosis" column with its binary value
                const features = [diagnosisBinary].concat(splitData.slice(2).map(num => parseFloat(num)));

                const prb_features = array_to_PRB(features);
                //console.log("The training data is", prb_features);

                // Send the features to the contract
                await instance.set_TrainingData(prb_features);
                gas_upload_testData += await instance.set_TrainingData.estimateGas(prb_features);
            }
            console.log("Finished sending training data.");
        } catch (err) {
            console.error("Error reading or processing the file:", err);
        }
    });

    it("Classify", async () => {
        const result = await instance.classify();
        console.log('Accuracy is', Number(result / 50 * 100).toFixed(2), "%");
        gas_classify += await instance.classify.estimateGas();
    });

    after(() => {
        console.log(`Deployment Gas: ${gas_deployment}`);
        console.log(`Test data upload gas: ${gas_upload_testData}`);
        console.log(`Weights and biases upload gas: ${gas_upload_weightBias}`);
        console.log(`Classify gas: ${gas_classify}`);
    });

});

