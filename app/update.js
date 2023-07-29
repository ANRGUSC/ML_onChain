const Web3 = require('web3');

// Instantiate web3 with Ganache
const web3 = new Web3('http://localhost:7545');

//replace ABI with contract and contnractAddress with contract address
const contract = new web3.eth.Contract(abi, contractAddress);

async function updateFcAndFc2() {
    const accounts = await web3.eth.getAccounts();

    // Create new values for fc and fc2
    let newFc = [[1, 2], [3, 4]];
    let newFc2 = [5];

    // Call the setfc and setfc2 functions with the new values
    await contract.methods.setfc(newFc).send({ from: accounts[0] });
    await contract.methods.setfc2(newFc2).send({ from: accounts[0] });
}

updateFcAndFc2();
