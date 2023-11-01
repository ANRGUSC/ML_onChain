const fs = require('fs');
const MLP_1L_1N = artifacts.require("MLP_1L_1N.sol");
const fsPromises = fs.promises;

//saves log into a file
const originalConsoleLog = console.log;
console.log = function(...args) {
    originalConsoleLog.apply(console, args);  // This will ensure the logs still display in the console
    fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
};

function num_to_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value * 1e18));
}

function array_to_PRB(array) {
    return array.map(value => num_to_PRB(value));
}

function num_from_PRB(value) {
    return Number(value)/1e18;
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}

contract("MLP_1L_1N.sol", accounts => {
    let instance;

    before(async () => {
        instance = await MLP_1L_1N.new(1);
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });


    it("Upload weights and biases", async()=>{
        fs.readFile('./src/weights_biases/MLP_1L1.json', 'utf8', async (err, data) => {
            if (err) {
                console.error("Error reading the file:", err);
                return;
            }
            const content = JSON.parse(data);
            let weights = content["fc1.weight"];
            let biases = content["fc1.bias"];

            let prb_biases = array_to_PRB(biases);

            // Send the biases to the contract
            console.log("The Biases are:",array_from_PRB(prb_biases));
            await instance.set_Biases(prb_biases);
            // Send each row of the 2D weight array to the contract
            for (let weightRow of weights) {
                let prb_weightRow = array_to_PRB(weightRow);
                await instance.set_Weights(prb_weightRow);
            }
        });
    });

    it("Upload training data", async()=>{
         try {
        const data = await fsPromises.readFile('./src/data/processed_data.csv', 'utf8');
        const lines = data.split('\n');
        for(let i = 1; i <= 100 && i < lines.length; i++) { // Starting from 1 to skip header
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
        }
        console.log("Finished sending training data.");
    } catch (err) {
        console.error("Error reading or processing the file:", err);
    }
    });

    it("Get dataset size", async()=> {
        const size = await instance.view_dataset_size();
        console.log('Size of the dataset is',Number(size));
    });

    it ("Classify", async()  => {
        const result = await instance.classify();
        console.log('Accuracy is',Number(result),"%");
    });

});

