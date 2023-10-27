const MLP_Test = artifacts.require("MLP_Test");

function num_to_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value * 10**18));
}

function array_to_PRB(array) {
    return array.map(value => num_to_PRB(value));
}

function num_from_PRB(value) {
    if (isNaN(value)) {
        console.error('Invalid value encountered:', value);
        return BigInt(0);
    }
    return BigInt(Math.round(value / 10**18));
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}

function approximatelyEqual(num1, num2, precision = 1e-6) {
    return Math.abs(num1 - num2) < precision;
}
contract("MLP_Test", accounts => {
    let instance;

    beforeEach(async () => {
        instance = await MLP_Test.new();
    });

    // test deployment
    it("deployment", async () => {
        assert(instance.address !== "");
    });

    // test sigmoid calculation
    it("sigmoid calculation", async () => {
        const result = await instance.sigmoid(5);
        let correct_answer = 1/(1+Math.exp(-5));
        console.log("The result is:",num_from_PRB(result));
        console.log("The correct answer is:",correct_answer);
        assert(approximatelyEqual(num_from_PRB(result),correct_answer), "Wrong sigmoid result")
    });

    // test conversion
    it("conversion", async () => {
        const result = await instance.test_convert();
        console.log("The result is:",num_from_PRB(result));
        assert(num_from_PRB(result) === 1, "Wrong conversion result");

    });

    // test array operations
    it("array operations", async () => {
        const result = await instance.array_ops();
        console.log("The result is:",num_from_PRB(result));
        //assert(array_from_PRB(result)[0] === 1, "Wrong array result");

    });

});

