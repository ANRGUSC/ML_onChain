const MLP_1 = artifacts.require("MLP_1");

function toABDKFixedPoint64x64(value) {
    return BigInt(Math.round(value * 2**64));
}

contract("MLP_1", accounts => {
    let instance;

    beforeEach(async () => {
        instance = await MLP_1.new(14, 1);
    });

    it("should be deployed", async () => {
        assert(instance.address !== "");
    });

    it("should set weights and biases", async () => {
        let weights = [[-0.1610087752342224, 0.049620844423770905, 0.08205176889896393, 0.39892637729644775]];
        let biases = [-0.09638084471225739];

        // Convert weights and biases to ABDK format
        let convertedWeights = weights.map(subArray => subArray.map(value => toABDKFixedPoint64x64(value)));
        let convertedBiases = biases.map(value => toABDKFixedPoint64x64(value));

        await instance.setfc(convertedWeights, convertedBiases, { from: accounts[0] });

        // Now you should retrieve the weights and biases from the contract and validate them
        // In this test, I'll simply check if they have been set (non-zero values)
        let storedWeight = await instance.fc_weights(0, 0);
        let storedBias = await instance.fc_biases(0);

        assert(storedWeight.toString() !== "0", "Weight was not stored correctly");
        assert(storedBias.toString() !== "0", "Bias was not stored correctly");
    });

    // Add more tests as needed
});

