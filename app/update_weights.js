const { Web3 } = require('web3');
const contractData_1 = require('../build/contracts/MLP_1.json');
const fs = require('fs');

const web3 = new Web3(new Web3.providers.HttpProvider('HTTP://127.0.0.1:7545'));
const abi_1 = contractData_1.abi;
const contractAddress_1 = '0xB943Ed23B004eB6299658fEdE1C19DAcD9893E15';
const contract_1 = new web3.eth.Contract(abi_1, contractAddress_1);

function toABDKFixedPoint64x64(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);  // or handle this differently based on your needs
    }
    return BigInt(Math.round(value * 2**64));
}

function convertArrayToABDKFormat(array) {
    if (Array.isArray(array[0])) {  // Handling for 2D arrays
        return array.map(subArray => {
            if (!Array.isArray(subArray)) {
                console.error('Invalid subArray:', subArray);
                return [];
            }
            return subArray.map(value => toABDKFixedPoint64x64(value));
        });
    } else {  // Handling for 1D arrays
        return array.map(value => toABDKFixedPoint64x64(value));
    }
}

async function update_MLP1() {
    const accounts = await web3.eth.getAccounts();
    let weights = JSON.parse(fs.readFileSync('../src/dict/MLP_dict_1.json', 'utf8'));
    let newFc = convertArrayToABDKFormat(weights["fc1.weight"]);
    let newBiases = convertArrayToABDKFormat(weights["fc1.bias"]);

    console.log("newFc:", newFc);
    console.log("newBiases:", newBiases);

    try {
        await contract_1.methods.setfc(newFc, newBiases).send({ from: accounts[0], gas: 5000000 });
    } catch (error) {
        console.error('Error sending transaction:', error);
    }
}

update_MLP1();
