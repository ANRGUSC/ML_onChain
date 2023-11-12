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

function upload_weight_biases(instance, num_layers, filename) {
    fs.readFile('./src/weights_biases/' + filename, 'utf8', async (err, data) => {
        if (err) {
            console.error("Error reading the file:", err);
            return;
        }
        const content = JSON.parse(data);
        let weights1 = content["fc1.weight"];
        let biases = content["fc1.bias"];

        let prb_biases1 = array_to_PRB(biases);
        //---------------------------------------------------------------------------
        // single layer models
        //---------------------------------------------------------------------------
        if (num_layers === 1) {
            // Send the biases to the contract
            console.log("The Biases are:", array_from_PRB(prb_biases1));
            await instance.set_Biases(0, prb_biases1);
            // Send each row of the 2D weight array to the contract
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                await instance.set_Weights(0, prb_weightRow);
            }
        }
        //---------------------------------------------------------------------------
        // two layer models
        //---------------------------------------------------------------------------
        if (num_layers === 2) {
            console.log("The Layer 1 Biases are:", array_from_PRB(prb_biases1));
            await instance.set_Biases(0, prb_biases1);  // 0 indicates the first layer

            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                await instance.set_Weights(0, prb_weightRow);  // 0 indicates the first layer
            }

            // Layer 2
            let weights2 = content["fc2.weight"];
            let biases2 = content["fc2.bias"];

            let prb_biases2 = array_to_PRB(biases2);

            console.log("The Layer 2 Biases are:", array_from_PRB(prb_biases2));
            await instance.set_Biases(1, prb_biases2);  // 1 indicates the second layer

            for (let weightRow of weights2) {
                let prb_weightRow = array_to_PRB(weightRow);
                await instance.set_Weights(1, prb_weightRow);  // 1 indicates the second layer
            }
        }

    });
}

module.exports = {array_from_PRB, array_to_PRB, num_from_PRB, num_to_PRB, upload_weight_biases};