const {Web3} = require('web3');
const contractData = require('../build/contracts/MultiPerceptron.json');  // Update the path if necessary

// Instantiate web3 with Ganache as provider
const web3 = new Web3(new Web3.providers.HttpProvider('HTTP://127.0.0.1:7545'));

// Get the ABI from the contractData
const abi = contractData.abi;

// Replace this with your contract's address
const contractAddress = '0x98B487e45a127e220F801D2a93775B2ADeDf5294';

// Create contract instance
const contract = new web3.eth.Contract(abi, contractAddress);

async function updateFcFc2AndFc3() {
    const accounts = await web3.eth.getAccounts();

    // Create new values for fc, fc2 and fc3
    let newFc = [[1, 2], [3, 4]];
    let newFc2 = [[5, 6], [7, 8]];
    let newFc3 = [9, 10];

    // Call the setfc, setfc2 and setfc3 functions with the new values
    //await contract.methods.setfc(newFc).send({ from: accounts[0] });
    //await contract.methods.setfc2(newFc2).send({ from: accounts[0] });
    //await contract.methods.setfc3(newFc3).send({ from: accounts[0] });

    // removes gas limit
    await contract.methods.setfc(newFc).send({ from: accounts[0], gas: 5000000 });
    await contract.methods.setfc2(newFc2).send({ from: accounts[0], gas: 5000000 });
    await contract.methods.setfc3(newFc3).send({ from: accounts[0], gas: 5000000 });

}

updateFcFc2AndFc3();
