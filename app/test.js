const {Web3} = require('web3');
const contractData = require('../build/contracts/MLP_Test.json');  // Replace with the path to your ABI

const web3 = new Web3('HTTP://127.0.0.1:7545');  // Replace with your provider
const contract = new web3.eth.Contract(contractData.abi, '0x494BcBda3aa964C567F9281440aF4C25Ac18269C');  // Replace with your contract's address

function toPRBMathFormat(value) {
    return BigInt(Math.round(value * 10**18));
}
function fromPRBMathFormat(value) {
    return Number(value) / 10**18;
}
async function test_convert() {
    const result = await contract.methods.test_convert().call();
    console.log(fromPRBMathFormat(result));
}

async function test_sigmoid(){
    const result = await contract.methods.sigmoid(5).call();
    console.log(fromPRBMathFormat(result));
}

test_convert(); // passed
test_sigmoid(); //passed
