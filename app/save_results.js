const Web3 = require('web3');
const fs = require('fs');  // Import the File System module

const web3 = new Web3('HTTP://127.0.0.1:7545');

const contractABI = [
    // ... [Your contract's ABI]
];

const contractAddress = '0xbA61d89c239d85900d5ef7073a53F5Ecf8fe64C0';
const perceptronContract = new web3.eth.Contract(contractABI, contractAddress);

perceptronContract.methods.classifiedResults().call()
.then(results => {
    console.log("Classified Results:", results);

    // Convert the results to CSV format
    let csvContent = "data\n";  // Header for the CSV
    results.forEach(result => {
        csvContent += result + "\n";
    });

    // Save to CSV
    fs.writeFile("classifiedResults.csv", csvContent, 'utf8', function (err) {
        if (err) {
            console.error("Error saving to CSV:", err);
        } else {
            console.log("classifiedResults.csv was saved successfully.");
        }
    });
})
.catch(error => {
    console.error("Error fetching results:", error);
});
