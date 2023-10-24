const {Web3} = require('web3');
const contractData = require('../build/contracts/MLP_Test.json');  // Replace with the path to your ABI

const web3 = new Web3('HTTP://127.0.0.1:7545');  // Replace with your provider
const contract = new web3.eth.Contract(contractData.abi, '0xaCCcB71580008a65c1951AA545F703Eb38a43081');  // Replace with your contract's address

function toPRBMathFormat(value) {
    return BigInt(Math.round(value * 10**18));
}
function fromPRBMathFormat(value) {
    return Number(value) / 10**18;
}
async function multiply(value1, value2) {
    const scaledValue1 = toPRBMathFormat(value1);
    const scaledValue2 = toPRBMathFormat(value2);
    console.log(scaledValue1);
    console.log(scaledValue2);
    const result = await contract.methods.multiply(scaledValue1, scaledValue2).call();
    console.log(fromPRBMathFormat(result));
}

multiply(2.5, 1.1);

