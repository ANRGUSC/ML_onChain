const {Web3} = require('web3');
const contractData_1 = require('../build/contracts/MLP_1.json');  // Update the path if necessary
const contractData_2 = require('../build/contracts/MLP_2.json');  // Update the path if necessary
const contractData_3 = require('../build/contracts/MLP_3.json');  // Update the path if necessary
const fs = require('fs');

// Instantiate web3 with Ganache as provider
const web3 = new Web3(new Web3.providers.HttpProvider('HTTP://127.0.0.1:7545'));

// Get the ABI from the contractData
const abi_1 = contractData_1.abi;
const abi_2 = contractData_2.abi;
const abi_3 = contractData_3.abi;

// Replace this with your contract's address
const contractAddress_1 = '0xb1189700C19b89dd2D00Cc214985B786d1c9d5bb';
const contractAddress_2 = '0xB003c2582Fa238B0b7a93E890ab599F036aD5460';
const contractAddress_3 = '0x4dd716231846b9f48A7D4dE1F62A463729EFf673';

// Create contract instance
const contract_1 = new web3.eth.Contract(abi_1, contractAddress_1);
const contract_2 = new web3.eth.Contract(abi_2, contractAddress_2);
const contract_3 = new web3.eth.Contract(abi_3, contractAddress_3);
function toABDKFixedPoint64x64(value) {
    return BigInt(Math.round(value * 2**64));
}

function convertArrayToABDKFormat(array) {
    return array.map(subArray =>
        subArray.map(value =>
            toABDKFixedPoint64x64(value)
        )
    );
}

async function update_MLP1() {
    const accounts = await web3.eth.getAccounts();
    let weights = JSON.parse(fs.readFileSync('../src/dict/MLP_dict_1.json', 'utf8'));

    let newFc = convertArrayToABDKFormat(weights["fc1.weight"]);

    await contract_1.methods.setfc(newFc).send({ from: accounts[0], gas: 5000000 });
}

async function update_MLP2() {
    const accounts = await web3.eth.getAccounts();
    let weights = JSON.parse(fs.readFileSync('../src/dict/MLP_dict_2.json', 'utf8'));

    let newFc = convertArrayToABDKFormat(weights["fc1.weight"]);
    let newFc2 = convertArrayToABDKFormat(weights["fc2.weight"]);

    await contract_2.methods.setfc(newFc).send({ from: accounts[0], gas: 5000000000000 });
    await contract_2.methods.setfc2(newFc2).send({ from: accounts[0], gas: 5000000000000 });
}

async function update_MLP3() {
    const accounts = await web3.eth.getAccounts();
    let weights = JSON.parse(fs.readFileSync('../src/dict/MLP_dict_3.json', 'utf8'));

    let newFc = convertArrayToABDKFormat(weights["fc1.weight"]);
    let newFc2 = convertArrayToABDKFormat(weights["fc2.weight"]);
    let newFc3 = convertArrayToABDKFormat(weights["fc3.weight"]);

    await contract_3.methods.setfc(newFc).send({ from: accounts[0], gas: 5000000000000 });
    await contract_3.methods.setfc2(newFc2).send({ from: accounts[0], gas: 5000000000000 });
    await contract_3.methods.setfc3(newFc3).send({ from: accounts[0], gas: 5000000000000 });
}

update_MLP1();
//update_MLP2();
//update_MLP3();