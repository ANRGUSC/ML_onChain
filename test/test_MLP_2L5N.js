const fs = require('fs');
const MLP_2L_5N = artifacts.require("MLP_2L_5N.sol");
const fsPromises = fs.promises;
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

contract("MLP_2L_5N.sol", accounts => {
    let instance;

    before(async () => {
        instance = await MLP_2L_3N.new(6);
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });


    it("Upload weights and biases", async()=>{
        fs.readFile('./src/weights_biases/MLP_2L_5N.json', 'utf8', async (err, data) => {
            if (err) {
                console.error("Error reading the file:", err);
                return;
            }
            const content = JSON.parse(data);

            // Layer 1
            let weights1 = content["fc1.weight"];
            let biases1 = content["fc1.bias"];    // Expected to be an array of size 2

            let prb_biases1 = array_to_PRB(biases1);
            console.log("The Layer 1 Biases are:", array_from_PRB(prb_biases1));
            await instance.set_Biases(0, prb_biases1);  // 0 indicates the first layer

            for (let i = 0; i < 3; i++) {
                let prb_weightRow = array_to_PRB(weights1[i]);
                await instance.set_Weights(0, prb_weightRow);  // 0 indicates the first layer
            }


            // Layer 2
            let weights2 = content["fc2.weight"];
            let biases2 = content["fc2.bias"];

            let prb_biases2 = array_to_PRB(biases2);
            console.log("The Layer 2 Biases are:", array_from_PRB(prb_biases2));
            await instance.set_Biases(1, prb_biases2);  // 1 indicates the second layer

            for (let weightRow of weights2) {
                let prb_weightRow = array_to_PRB(weightRow);
                await instance.set_Weights(1, prb_weightRow);  // 1 indicates the second layer
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
