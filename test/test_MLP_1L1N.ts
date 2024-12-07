import { expect } from "chai";
import { ethers } from "hardhat";
import fs from 'fs';
import { Contract } from "ethers";
import { classify, upload_trainingData, upload_weightsBiases } from '../scripts/util_functions';

// gas cost of diff functions
let gas_classify: bigint = BigInt(0);
let gas_upload_weightBias: bigint = BigInt(0);
let gas_deployment: bigint = BigInt(0);
let gas_upload_testData: bigint = BigInt(0);

// saves log into a file
const originalConsoleLog = console.log;
console.log = function (...args: any[]) {
    originalConsoleLog.apply(console, args);
    fs.appendFileSync('./results/OnChain_accuracy', args.join(' ') + '\n');
};

describe("MLP_1L_1n.sol", function() {
    let instance: Contract;

    before(async function() {
        try {
            const [signer] = await ethers.getSigners();
            const MLP = await ethers.getContractFactory("MLP_1L_1n");
            
            // Get deployment transaction
            const deployTx = await MLP.getDeployTransaction(1);
            
            // Estimate gas
            gas_deployment = BigInt(await signer.estimateGas(deployTx));
            
            // Deploy the contract
            instance = await MLP.deploy(1);
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
            gas_upload_weightBias = await upload_weightsBiases(instance, fs.promises, 'MLP_1L1.json', 1);
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
        console.log('Name: MLP_1L_1n');
        console.log(`Deployment Gas: ${gas_deployment.toString()}`);
        console.log(`Test data upload gas: ${gas_upload_testData.toString()}`);
        console.log(`Weights and biases upload gas: ${gas_upload_weightBias.toString()}`);
        console.log(`Classify gas: ${gas_classify.toString()}\n`);
    });
});