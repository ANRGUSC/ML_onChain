const fs = require('fs');
const functions = artifacts.require("functions.sol");
const fsPromises = fs.promises;

// gas cost of diff functions
let gas_sigmoid = 0;
let gas_relu = 0;

// gas cost of add
let gas_add = 0;
let gas_add_prb = 0;

//gas cost of mul
let gas_mul = 0;
let gas_mul_prb = 0;

// gas cost of div
let gas_div = 0;
let gas_div_prb = 0;


const {num_to_PRB, upload_trainingData, upload_weightsBiases} = require('./util_functions.js');
//saves log into a file
const originalConsoleLog = console.log;
console.log = function (...args) {
    originalConsoleLog.apply(console, args);  // This will ensure the logs still display in the console
    fs.appendFileSync('./results/gas_function', args.join(' ') + '\n');
};

contract("functions.sol", accounts => {
    let instance;

    before(async () => {
        instance = await functions.new();
    });

    it("deployment", async () => {
        assert(instance.address !== "");
    });

    it("test sigmoid relu", async () => {
         gas_sigmoid = await instance.sigmoid.estimateGas(num_to_PRB(0.6));
         gas_relu = await instance.relu.estimateGas(num_to_PRB(0.6));
    });

    it("test operations", async () => {
        gas_add = await instance.add_1.estimateGas(4,2);
        let x = await instance.add_2.estimateGas(4,2);
        gas_add = x- gas_add;

        gas_add_prb = await instance.add_prb_1.estimateGas(num_to_PRB(4),num_to_PRB(2));
        let a = await instance.add_prb_2.estimateGas(num_to_PRB(4),num_to_PRB(2));
        gas_add_prb = a- gas_add_prb;

        gas_mul = await instance.mul_1.estimateGas(4,2);
        let y = await instance.mul_2.estimateGas(4,2);
        gas_mul = y- gas_mul;

        gas_mul_prb = await instance.mul_prb_1.estimateGas(num_to_PRB(4),num_to_PRB(2));
        let b = await instance.mul_prb_2.estimateGas(num_to_PRB(4),num_to_PRB(2));
        gas_mul_prb = b- gas_mul_prb;

        gas_div = await instance.div_1.estimateGas(4,2);
        let z = await instance.div_2.estimateGas(4,2);
        gas_div = z- gas_div;

        gas_div_prb = await instance.div_prb_1.estimateGas(num_to_PRB(4),num_to_PRB(2));
        let c = await instance.div_prb_2.estimateGas(num_to_PRB(4),num_to_PRB(2));
        gas_div_prb = c- gas_div_prb;


    });

    after(() => {
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

