const fs = require('fs');
const MLP_1L_1n = artifacts.require("MLP_1L_1n.sol");
const fsPromises = fs.promises;

let totalGasUsed = 0;

//const {array_from_PRB, array_to_PRB, num_from_prb, num_to_prb} = require ('./util_functions.js');
const {array_from_PRB, array_to_PRB,upload_weight_biases} = require('./util_functions.js');
//saves log into a file
const originalConsoleLog = console.log;
console.log = function (...args) {
    originalConsoleLog.apply(console, args);  // This will ensure the logs still display in the console
    fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
};

contract("MLP_1L_1n.sol", accounts => {
    let instance;

    before(async () => {
        instance = await MLP_1L_1n.new(1);
        totalGasUsed += await MLP_1L_1n.new.estimateGas(1)
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });


    it("Upload weights and biases", async () => {
        upload_weight_biases(instance,1,'MLP_1L1.json');
        totalGasUsed += await upload_weight_biases.estimateGas(instance,1,'MLP_1L1.json');
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
                totalGasUsed += await instance.set_TrainingData.estimateGas(prb_features);
            }
            console.log("Finished sending training data.");
        } catch (err) {
            console.error("Error reading or processing the file:", err);
        }
    });

    it("Get dataset size", async () => {
        const size = await instance.view_dataset_size();
        console.log('Size of the dataset is', Number(size));
    });


    it("Classify", async () => {
        const result = await instance.classify();
        console.log('Accuracy is', Number(result / 50 * 100).toFixed(2), "%");
        totalGasUsed += await instance.classify.estimateGas();
    });

    after(() => {
        console.log(`Total Estimated Gas: ${totalGasUsed}`);
    });

});

