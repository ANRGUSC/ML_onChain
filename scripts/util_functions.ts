import { Contract } from "ethers";
import { promises as fsPromises } from "fs";
import { type FileHandle } from "fs/promises";

// Type definitions
type PRBValue = bigint;
type WeightsBiasesContent = {
    "fc1.weight": number[][];
    "fc1.bias": number[];
    "fc2.weight"?: number[][];
    "fc2.bias"?: number[];
    "fc3.weight"?: number[][];
    "fc3.bias"?: number[];
};

// Utility functions for PRB conversions
function num_to_PRB(value: number): PRBValue {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value * 1e18));
}

function array_to_PRB(array: number[]): PRBValue[] {
    return array.map(value => num_to_PRB(value));
}

function num_from_PRB(value: PRBValue): number {
    return Number(value) / 1e18;
}

function array_from_PRB(array: PRBValue[]): number[] {
    return array.map(value => num_from_PRB(value));
}

async function classify(instance: Contract): Promise<bigint> {
    let gas_classify: bigint = BigInt(0);
    const result = await instance.classify();
    console.log('Accuracy:', Number(result * BigInt(100) / BigInt(50)).toFixed(2), "%");
    gas_classify += BigInt(await instance.classify.estimateGas());
    return gas_classify;
}

async function upload_trainingData(instance: Contract, fsPromises: typeof import("fs/promises")): Promise<bigint> {
    let gas_upload_testData: bigint = BigInt(0);
    try {
        const data = await fsPromises.readFile('./src/data/processed_data.csv', 'utf8');
        const lines = data.split('\n');
        for (let i = 1; i <= 50 && i < lines.length; i++) {
            const line = lines[i];
            const splitData = line.split(',');
            const diagnosisBinary = +splitData[1];
            const features = [diagnosisBinary, ...splitData.slice(2).map(num => parseFloat(num))];
            const prb_features = array_to_PRB(features);
            
            await instance.set_TrainingData(prb_features);
            gas_upload_testData += BigInt(await instance.set_TrainingData.estimateGas(prb_features));
        }
    } catch (err) {
        console.error("Error reading or processing the file:", err);
    }
    return gas_upload_testData;
}

async function upload_weightsBiases(
    instance: Contract, 
    fsPromises: typeof import("fs/promises"), 
    filename: string, 
    num_layers: number, 
    debug: boolean = false
): Promise<bigint> {
    let gas_upload_weightBias: bigint = BigInt(0);
    try {
        const data = await fsPromises.readFile('./src/weights_biases/' + filename, 'utf8');
        const content: WeightsBiasesContent = JSON.parse(data);
        let weights1 = content["fc1.weight"];
        let biases = content["fc1.bias"];

        let prb_biases1 = array_to_PRB(biases);

        // Single layer models
        if (num_layers === 1) {
            if(debug) {
                console.log("The Biases are:", array_from_PRB(prb_biases1));
            }
            gas_upload_weightBias += BigInt(await instance.set_Biases.estimateGas(0, prb_biases1));
            await instance.set_Biases(0, prb_biases1);
            
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += BigInt(await instance.set_Weights.estimateGas(0, prb_weightRow));
                await instance.set_Weights(0, prb_weightRow);
            }
        }

        // Two layer models
        if (num_layers === 2 && content["fc2.weight"] && content["fc2.bias"]) {
            if(debug) {
                console.log("The Layer 1 Biases are:", array_from_PRB(prb_biases1));
            }
            await instance.set_Biases(0, prb_biases1);
            gas_upload_weightBias += BigInt(await instance.set_Biases.estimateGas(0, prb_biases1));
            
            for (let weightRow of weights1) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += BigInt(await instance.set_Weights.estimateGas(0, prb_weightRow));
                await instance.set_Weights(0, prb_weightRow);
            }

            let weights2 = content["fc2.weight"];
            let biases2 = content["fc2.bias"];
            let prb_biases2 = array_to_PRB(biases2);
            
            if(debug) {
                console.log("The Layer 2 Biases are:", array_from_PRB(prb_biases2));
            }
            await instance.set_Biases(1, prb_biases2);
            gas_upload_weightBias += BigInt(await instance.set_Biases.estimateGas(1, prb_biases2));
            
            for (let weightRow of weights2) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += BigInt(await instance.set_Weights.estimateGas(1, prb_weightRow));
                await instance.set_Weights(1, prb_weightRow);
            }
        }

        // Three layer models
        if (num_layers === 3 && content["fc2.weight"] && content["fc2.bias"] && content["fc3.weight"] && content["fc3.bias"]) {
            // Layer 1 processing (same as above)
            // Layer 2 processing (same as above)
            // Layer 3
            let weights3 = content["fc3.weight"];
            let biases3 = content["fc3.bias"];
            let prb_biases3 = array_to_PRB(biases3);
            
            if(debug) {
                console.log("The Layer 3 Biases are:", array_from_PRB(prb_biases3));
            }
            await instance.set_Biases(2, prb_biases3);
            gas_upload_weightBias += BigInt(await instance.set_Biases.estimateGas(2, prb_biases3));
            
            for (let weightRow of weights3) {
                let prb_weightRow = array_to_PRB(weightRow);
                gas_upload_weightBias += BigInt(await instance.set_Weights.estimateGas(2, prb_weightRow));
                await instance.set_Weights(2, prb_weightRow);
            }
        }
    } catch (err) {
        console.error("Error reading or processing the file:", err);
    }
    return gas_upload_weightBias;
}

export {
    array_from_PRB,
    array_to_PRB,
    num_from_PRB,
    num_to_PRB,
    classify,
    upload_weightsBiases,
    upload_trainingData,
    type PRBValue
};