const { Web3 } = require('web3');
const fs = require('fs');
const contractData = require("../build/contracts/MLP_1.json");
const web3 = new Web3('HTTP://127.0.0.1:7545');
const contract = new web3.eth.Contract(contractData.abi, '0x539FcF3eF2e0E2c63978C34882FFa6BFA1d0b570');

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

async function update_MLP1() {
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
            console.log(prb_weightRow);
            await contract.methods.set_Weights(prb_weightRow).send({ from: accounts[0],gas: 1000000 });
        }
    });
}

update_MLP1();
