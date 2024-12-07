import { expect } from "chai";
import { ethers } from "hardhat";
import fs from 'fs';
import { Contract } from "ethers";

// Import your util functions
import { num_to_PRB, upload_trainingData, upload_weightsBiases } from '../scripts/util_functions';

// gas cost variables
let gas_sigmoid: number = 0;
let gas_relu: number = 0;
let gas_add: number = 0;
let gas_add_prb: number = 0;
let gas_mul: number = 0;
let gas_mul_prb: number = 0;
let gas_div: number = 0;
let gas_div_prb: number = 0;

// Console log override
const originalConsoleLog = console.log;
console.log = function (...args: any[]) {
    originalConsoleLog.apply(console, args);
    fs.appendFileSync('./results/gas_function', args.join(' ') + '\n');
};

describe("functions.sol", function () {
    let functions: Contract;
    let owner: any;
    let instance: Contract;

    before(async function () {
        [owner] = await ethers.getSigners();
        const Functions = await ethers.getContractFactory("functions");
        instance = await Functions.deploy();
        // Remove .deployed() and wait for the deployment transaction
        await instance.waitForDeployment();
    });

    it("deployment", async function () {
        expect(await instance.getAddress()).to.not.equal("");
    });

    it("test sigmoid relu", async function () {
        const sigmoidTx = await instance.sigmoid.estimateGas(num_to_PRB(0.6));
        gas_sigmoid = Number(sigmoidTx);

        const reluTx = await instance.relu.estimateGas(num_to_PRB(0.6));
        gas_relu = Number(reluTx);
    });

    it("test operations", async function () {
        gas_add = Number(await instance.add_1.estimateGas(4, 2));
        const x = Number(await instance.add_2.estimateGas(4, 2));
        gas_add = x - gas_add;

        gas_add_prb = Number(await instance.add_prb_1.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        const a = Number(await instance.add_prb_2.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        gas_add_prb = a - gas_add_prb;

        gas_mul = Number(await instance.mul_1.estimateGas(4, 2));
        const y = Number(await instance.mul_2.estimateGas(4, 2));
        gas_mul = y - gas_mul;

        gas_mul_prb = Number(await instance.mul_prb_1.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        const b = Number(await instance.mul_prb_2.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        gas_mul_prb = b - gas_mul_prb;

        gas_div = Number(await instance.div_1.estimateGas(4, 2));
        const z = Number(await instance.div_2.estimateGas(4, 2));
        gas_div = z - gas_div;

        gas_div_prb = Number(await instance.div_prb_1.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        const c = Number(await instance.div_prb_2.estimateGas(num_to_PRB(4), num_to_PRB(2)));
        gas_div_prb = c - gas_div_prb;
    });

    after(function () {
        console.log('functions');
        console.log(`relu: ${gas_relu}`);
        console.log(`sigmoid: ${gas_sigmoid}`);
        console.log(`add: ${gas_add}`);
        console.log(`add_prb: ${gas_add_prb}`);
        console.log(`mul: ${gas_mul}`);
        console.log(`mul_prb: ${gas_mul_prb}`);
        console.log(`div: ${gas_div}`);
        console.log(`div_prb: ${gas_div_prb}`);
    });
});