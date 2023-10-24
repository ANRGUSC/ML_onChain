const MLP_1 = artifacts.require("MLP_1");

function toABDKFixedPoint64x64(value) {
    return BigInt(Math.round(value * 2**64));
}

contract("MLP_1", accounts => {
    let instance;

    beforeEach(async () => {
        instance = await MLP_1.new(30, 1);
    });

    it("should be deployed", async () => {
        assert(instance.address !== "");
    });

    it("should set weights and biases", async () => {
        let weights = [[-0.1610087752342224, 0.049620844423770905, 0.08205176889896393, 0.39892637729644775, -0.04140500724315643, 0.11601730436086655, -0.12026197463274002, -0.021909520030021667, 0.04891527444124222, -0.06818708777427673, -0.08882831782102585, -0.1374446600675583, -0.022840706631541252, 0.04143739491701126, 0.04935785382986069, 0.16009245812892914, 0.021886542439460754, 0.004863104782998562, 0.040804751217365265, 0.007020716555416584, 0.09541811048984528, -0.12490775436162949, 0.10441724956035614, 1.2932366132736206, -0.08082078397274017, 0.1273333728313446, -0.144283264875412, 0.07513242959976196, 0.1578405648469925, 0.12714360654354095]];
        let biases = [-0.09638084471225739];

        // Convert weights and biases to ABDK format
        let convertedWeights = weights.map(subArray => subArray.map(value => toABDKFixedPoint64x64(value)));
        let convertedBiases = biases.map(value => toABDKFixedPoint64x64(value));
        console.log("convertedWeights:", convertedWeights);
        console.log("convertedBiases:", convertedBiases);
        await instance.setfc(convertedWeights, convertedBiases, { from: accounts[0] });

        // Now you should retrieve the weights and biases from the contract and validate them
        // In this test, I'll simply check if they have been set (non-zero values)
        let storedWeight = await instance.fc_weights(0, 0);
        let storedBias = await instance.fc_biases(0);

        console.log(storedBias.toString());
        assert(storedWeight.toString() !== "0", "Weight was not stored correctly");
        assert(storedBias.toString() !== "0", "Bias was not stored correctly");
    });

    // Add more tests as needed
});

