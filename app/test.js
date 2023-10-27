const {Web3} = require('web3');
const contractData = require('../build/contracts/MLP_Test.json');  // Replace with the path to your ABI

const web3 = new Web3('HTTP://127.0.0.1:7545');  // Replace with your provider
const contract = new web3.eth.Contract(contractData.abi, '0x494BcBda3aa964C567F9281440aF4C25Ac18269C');  // Replace with your contract's address

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
    return Number(value) / 10**18;
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}
async function test_convert() {
    const result = await contract.methods.test_convert().call();
    console.log(num_from_PRB(result));
}

async function test_sigmoid(){
    const result = await contract.methods.sigmoid(5).call();
    console.log(num_from_PRB(result));
}

test_convert(); // passed
test_sigmoid(); //passed

