import { expect } from "chai";
import { ethers } from "hardhat";
import fs from 'fs';
import { Contract } from "ethers";
// @ts-ignore
import { classify, upload_trainingData, upload_weightsBiases } from '../scripts/util_functions';

interface MLPConfig {
    contractName: string;     // e.g., "MLP_2L_2n"
    displayName: string;      // e.g., "MLP_2L_2n" (for logging)
    numLayers: number;        // e.g., 2 or 3
    weightsFile: string;      // e.g., "MLP_2L2.json"
}

async function runMLPTest(config: MLPConfig) {
    let gas_classify: bigint = BigInt(0);
    let gas_upload_weightBias: bigint = BigInt(0);
    let gas_deployment: bigint = BigInt(0);
    let gas_upload_testData: bigint = BigInt(0);

    describe(config.contractName, function() {
        let instance: Contract;

        before(async function() {
            try {
                const [signer] = await ethers.getSigners();
                const MLP = await ethers.getContractFactory(config.contractName);
                
                const deployTx = await MLP.getDeployTransaction(config.numLayers);
                gas_deployment = BigInt(await signer.estimateGas(deployTx));
                
                instance = await MLP.deploy(config.numLayers);
                await instance.waitForDeployment();
            } catch (error) {
                console.error("Deployment error:", error);
                throw error;
            }
        });

        it("deployment", async function() {
            const address = await instance.getAddress();
            expect(address).to.be.properAddress;
            expect(address).to.not.equal(ethers.ZeroAddress);
        });

        it("Upload weights and biases", async function() {
            try {
                gas_upload_weightBias = await upload_weightsBiases(
                    instance, 
                    fs.promises, 
                    config.weightsFile, 
                    config.numLayers
                );
            } catch (error) {
                console.error("Upload weights error:", error);
                throw error;
            }
        });

        it("Upload training data", async function() {
            try {
                const uploadGas = await upload_trainingData(instance, fs.promises);
                gas_upload_testData = BigInt(uploadGas);
            } catch (error) {
                console.error("Upload training data error:", error);
                throw error;
            }
        });

        it("Classify", async function() {
            try {
                const classifyGas = await classify(instance);
                gas_classify = BigInt(classifyGas);
            } catch (error) {
                console.error("Classification error:", error);
                throw error;
            }
        });

        after(function() {
            console.log(`Name: ${config.displayName}`);
            console.log(`Deployment Gas: ${gas_deployment.toString()}`);
            console.log(`Test data upload gas: ${gas_upload_testData.toString()}`);
            console.log(`Weights and biases upload gas: ${gas_upload_weightBias.toString()}`);
            console.log(`Classify gas: ${gas_classify.toString()}\n`);
        });
    });
}

// Configuration for different MLP models
const mlpConfigs: MLPConfig[] = [
    {
        contractName: "MLP_1L_1n",
        displayName: "MLP_1L_1n",
        numLayers: 1,
        weightsFile: "MLP_1L1.json"
    },
    {
        contractName: "MLP_2L_1n",
        displayName: "MLP_2L_1n",
        numLayers: 2,
        weightsFile: "MLP_2L1.json"
    },
    {
        contractName: "MLP_2L_2n",
        displayName: "MLP_2L_2n",
        numLayers: 2,
        weightsFile: "MLP_2L2.json"
    },
    {
        contractName: "MLP_2L_3n",
        displayName: "MLP_2L_3n",
        numLayers: 2,
        weightsFile: "MLP_2L3.json"
    },
    {
        contractName: "MLP_2L_4n",
        displayName: "MLP_2L_4n",
        numLayers: 2,
        weightsFile: "MLP_2L4.json"
    },
    {
        contractName: "MLP_3L_1n1n",
        displayName: "MLP_3L_1n",
        numLayers: 3,
        weightsFile: "MLP_3L1.json"
    },
    {
        contractName: "MLP_3L_2n1n",
        displayName: "MLP_3L_2n",
        numLayers: 3,
        weightsFile: "MLP_3L2.json"
    },
    {
        contractName: "MLP_3L_3n1n",
        displayName: "MLP_3L_3n",
        numLayers: 3,
        weightsFile: "MLP_3L3.json"
    },
    {
        contractName: "MLP_3L_4n1n",
        displayName: "MLP_3L_4n",
        numLayers: 3,
        weightsFile: "MLP_3L4.json"
    },
];

// Run tests for all configurations
describe("MLP Tests", function() {
    // Save console.log output to file
    before(function() {
        const originalConsoleLog = console.log;
        console.log = function (...args: any[]) {
            originalConsoleLog.apply(console, args);
            fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
        };
    });

    for (const config of mlpConfigs) {
        runMLPTest(config);
    }
});