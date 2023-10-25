const { Web3 } = require('web3');
const fs = require('fs');
const contractData = require("../build/contracts/MLP_1.json");
const web3 = new Web3('HTTP://127.0.0.1:7545');
const contract = new web3.eth.Contract(contractData.abi, '0xC229D03B99c0fB862e1c35C166e650a92488d8bE');

function num_to_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value * 10**18));
}

function array_to_PRB(array) {
    return array.map(value => num_to_PRB(value));
}

function num_from_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value / 10**18));
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}

async function upload_weights_biases() {
    const accounts = await web3.eth.getAccounts();

    fs.readFile('../src/dict/MLP_dict_1.json', 'utf8', async (err, data) => {
        if (err) {
            console.error("Error reading the file:", err);
            return;
        }

        const content = JSON.parse(data);
        let weights = content["fc1.weight"];
        let biases = content["fc1.bias"];

        let prb_biases = array_to_PRB(biases);

        // Send the biases to the contract
        console.log(prb_biases);
        await contract.methods.set_Biases(prb_biases).send({ from: accounts[0], gas: 1000000 });
        // Send each row of the 2D weight array to the contract
        for (let weightRow of weights) {
            let prb_weightRow = array_to_PRB(weightRow);
            await contract.methods.set_Weights(prb_weightRow).send({ from: accounts[0],gas: 1000000 });
        }
    });
}

async function upload_TrainingData() {
    const accounts = await web3.eth.getAccounts();

    fs.readFile('../src/binary_classification.csv', 'utf8', async (err, data) => {
        if (err) {
            console.error("Error reading the file:", err);
            return;
        }

        const lines = data.split('\n');
        for(let i = 1; i <= 100 && i < lines.length; i++) { // Starting from 1 to skip header
            const line = lines[i];

            const splitData = line.split(',');

            // Convert "diagnosis" column to binary
            const diagnosisBinary = splitData[1] === 'M' ? 1 : 0;

            // Drop the first column (ID) and replace the "diagnosis" column with its binary value
            const features = [diagnosisBinary].concat(splitData.slice(2).map(num => parseFloat(num)));

            const prb_features = array_to_PRB(features);
            console.log(features)
            // Send the features to the contract
            //await contract.methods.set_TrainingData(prb_features).send({ from: accounts[0], gas: 1000000 });
        }

        console.log("Finished sending training data.");
    });
}

upload_weights_biases();
upload_TrainingData()

