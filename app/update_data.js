const { Web3 } = require('web3');
const fs = require('fs');
const csv = require('csv-parser');
const contractData = require('../build/contracts/MLP_1.json'); // Update the path if necessary

const web3 = new Web3(new Web3.providers.HttpProvider('HTTP://127.0.0.1:7545'));
const abi = contractData.abi;
const contractAddress = '0xbA61d89c239d85900d5ef7073a53F5Ecf8fe64C0';
const contract = new web3.eth.Contract(abi, contractAddress);

async function inputDataFromCsv() {
    const accounts = await web3.eth.getAccounts();

    // Read data from CSV
    let dataArr = [];
    fs.createReadStream('../src/synthetic_data.csv') // Make sure you have a file named 'data.csv'
        .pipe(csv())
        .on('data', (data) => dataArr.push(Object.values(data)))
        .on('end', async () => {
            // Assuming each row in the CSV is an array of values you want to feed into the contract
            for (let i = 0; i < dataArr.length; i++) {
                // Convert string data to integers
                let intArray = dataArr[i].map(item => parseInt(item));

                // You can now call the contract method that you wish to feed the data into.
                // For this example, I'm using the predict function. Update as necessary.
                let result = await contract.methods.predict(intArray).call({ from: accounts[0] });
                console.log(`Result for row ${i + 1}: ${result}`);
            }
        });
}

inputDataFromCsv();
