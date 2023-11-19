const fs = require("fs");

function num_to_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value * 1e18));
}

function array_to_PRB(array) {
    return array.map(value => num_to_PRB(value));
}

function num_from_PRB(value) {
    return Number(value) / 1e18;
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}

async function classify(instance){
    gas_classify = 0
    const result = await instance.classify();
    console.log('Accuracy:', Number(result / 50 * 100).toFixed(2), "%");
    gas_classify += await instance.classify.estimateGas();
    return gas_classify
}

async function upload_trainingData(instance,fsPromises){
    let gas_upload_testData = 0
    try {
        const data = await fsPromises.readFile('./src/data/processed_data.csv', 'utf8');
        const lines = data.split('\n');
        for (let i = 1; i <= 50 && i < lines.length; i++) { // Starting from 1 to skip header
            const line = lines[i];
            const splitData = line.split(',');
            // Convert "diagnosis" column to binary
            const diagnosisBinary = +splitData[1];
            // Drop the first column (ID) and replace the "diagnosis" column with its binary value
            const features = [diagnosisBinary].concat(splitData.slice(2).map(num => parseFloat(num)));
            const prb_features = array_to_PRB(features);
            //console.log("The training data is", prb_features);
            // Send the features to the contract
            await instance.set_TrainingData(prb_features);
            gas_upload_testData += await instance.set_TrainingData.estimateGas(prb_features);
        }
    } catch (err) {
        console.error("Error reading or processing the file:", err);
    }
    return gas_upload_testData;
}

async function upload_weightsBiases(instance, fsPromises, filename, num_layers, debug=false){
    let gas_upload_weightBias = 0
    try {
        const data = await fsPromises.readFile('./src/weights_biases/' + filename, 'utf8');
        const content = JSON.parse(data);
        let weights1 = content["fc1.weight"];
        let biases = content["fc1.bias"];

        let prb_biases1 = array_to_PRB(biases);
        //---------------------------------------------------------------------------
        // single layer models
        //---------------------------------------------------------------------------
        if (num_layers === 1) {
            // Send the biases to the contract
            if(debug){
                console.log("The Biases are:", array_from_PRB(prb_biases1));
            }
            gas_upload_weightBias += await instance.set_Biases.estimateGas(0, prb_biases1);
            await instance.set_Biases(0, prb_biases1);
            // Send each row of the 2D weight array to the contract
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(0, prb_weightRow);
                await instance.set_Weights(0, prb_weightRow);
            }
        }
        //---------------------------------------------------------------------------
        // two layer models
        //---------------------------------------------------------------------------
        if (num_layers === 2) {
            if(debug){
                console.log("The Layer 1 Biases are:", array_from_PRB(prb_biases1));
            }
            await instance.set_Biases(0, prb_biases1);  // 0 indicates the first layer
            gas_upload_weightBias += await instance.set_Biases.estimateGas(0, prb_biases1);
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(0, prb_weightRow);
                await instance.set_Weights(0, prb_weightRow);  // 0 indicates the first layer
            }
            // Layer 2
            let weights2 = content["fc2.weight"];
            let biases2 = content["fc2.bias"];
            let prb_biases2 = array_to_PRB(biases2);
            if(debug){
                console.log("The Layer 2 Biases are:", array_from_PRB(prb_biases2));
            }
            await instance.set_Biases(1, prb_biases2);  // 1 indicates the second layer
            gas_upload_weightBias += await instance.set_Biases.estimateGas(1, prb_biases2);
            for (let weightRow of weights2) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(1, prb_weightRow);
                await instance.set_Weights(1, prb_weightRow);  // 1 indicates the second layer
            }
        }
        //---------------------------------------------------------------------------
        // three layer models
        //---------------------------------------------------------------------------
        if (num_layers === 3) {
            if(debug){
                console.log("The Layer 1 Biases are:", array_from_PRB(prb_biases1));
            }
            await instance.set_Biases(0, prb_biases1);  // 0 indicates the first layer
            gas_upload_weightBias += await instance.set_Biases.estimateGas(0, prb_biases1);
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(0, prb_weightRow);
                await instance.set_Weights(0, prb_weightRow);  // 0 indicates the first layer
            }
            // Layer 2
            let weights2 = content["fc2.weight"];
            let biases2 = content["fc2.bias"];
            let prb_biases2 = array_to_PRB(biases2);
            if(debug){
                console.log("The Layer 2 Biases are:", array_from_PRB(prb_biases2));
            }
            await instance.set_Biases(1, prb_biases2);  // 1 indicates the second layer
            gas_upload_weightBias += await instance.set_Biases.estimateGas(1, prb_biases2);
            for (let weightRow of weights2) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(1, prb_weightRow);
                await instance.set_Weights(1, prb_weightRow);  // 1 indicates the second layer
            }
            // Layer 2
            let weights3 = content["fc3.weight"];
            let biases3 = content["fc3.bias"];
            let prb_biases3 = array_to_PRB(biases3);
            if(debug){
                console.log("The Layer 3 Biases are:", array_from_PRB(prb_biases3));
            }
            await instance.set_Biases(2, prb_biases3);  // 2 indicates the third layer
            gas_upload_weightBias += await instance.set_Biases.estimateGas(2, prb_biases3);
            for (let weightRow of weights3) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += await instance.set_Weights.estimateGas(2, prb_weightRow);
                await instance.set_Weights(2, prb_weightRow);  // 2 indicates the third layer
            }
        }
    } catch (err) {
        console.error("Error reading or processing the file:", err);
    }
    return gas_upload_weightBias;
}

module.exports = {array_from_PRB, array_to_PRB, num_from_PRB, num_to_PRB, classify,
    upload_weightsBiases, upload_trainingData};