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
    return Number(value)/1e18;
}

function array_from_PRB(array) {
    return array.map(value => num_from_PRB(value));
}

module.exports = {array_from_PRB, array_to_PRB, num_from_PRB, num_to_PRB};