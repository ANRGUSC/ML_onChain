const { Web3 } = require('web3');
const contractData_1 = require('../build/contracts/MLP_1.json');
const fs = require('fs');

const web3 = new Web3(new Web3.providers.HttpProvider('HTTP://127.0.0.1:7545'));
const abi_1 = contractData_1.abi;
const contractAddress_1 = '0x11B138FF251941D355b0459dC41eA38459f1C72D';
const contract_1 = new web3.eth.Contract(abi_1, contractAddress_1);

function toPRBFixedPoint(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);  // or handle this differently based on your needs
    }
    return BigInt(Math.round(value * 10**18));
}

function convertArrayToPRBFormat(array) {
    if (Array.isArray(array[0])) {  // Handling for 2D arrays
        return array.map(subArray => {
            if (!Array.isArray(subArray)) {
                console.error('Invalid subArray:', subArray);
                return [];
            }
            return subArray.map(value => toPRBFixedPoint(value));
        });
    } else {  // Handling for 1D arrays
        return array.map(value => toPRBFixedPoint(value));
    }
}

async function update_MLP1() {
    const accounts = await web3.eth.getAccounts();

    fs.readFile('../src/dict/MLP_dict_1.json', 'utf8', async (err, data) => {
        if (err) {
            console.error("Error reading the file:", err);
            return;
        }

        // Parse the JSON content
        const content = JSON.parse(data);

        // Extract weights and biases
        //const weights = content["fc1.weight"];
        //const biases = content["fc1.bias"];
        const weights = [[2,3,4,5,6,7,8]];
        const biases = [1];

        // Convert weights and biases to PRB format
        let convertedWeights = convertArrayToPRBFormat(weights);
        let convertedBiases = convertArrayToPRBFormat(biases);
        console.log("convertedWeights:", convertedWeights);
        console.log("convertedBiases:", convertedBiases);

        try {
            const estimatedGas = await contract_1.methods.setfc(convertedWeights, convertedBiases).estimateGas({ from: accounts[0] });
            console.log("Estimated gas:", estimatedGas);
            await contract_1.methods.setfc(convertedWeights, convertedBiases).send({ from: accounts[0], gas: estimatedGas  });
        } catch (error) {
            console.error('Error sending transaction:', error);
        }
    });
}

update_MLP1();
