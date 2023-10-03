const Checks = require('./checks.js');
const math = require('mathjs');

function constructorTests() {
    console.log('Running constructor tests:');

    console.log('----- TEST ONE -----');
    let first = new Checks();
    console.log(math.matrix(first.inputWeightsSize ) + ', ' + math.size(first.inputWeights));
    console.log(math.matrix(first.layersWeightsSize) + ', ' + math.size(first.layersWeights));
    console.log(math.matrix(first.layersBiasSize   ) + ', ' + math.size(first.layersBias));
    console.log(math.matrix(first.outputWeightsSize) + ', ' + math.size(first.outputWeights));
    console.log(math.matrix(first.outputBiasSize   ) + ', ' + math.size(first.outputBias));
    if (
        !math.deepEqual(math.matrix(first.inputWeightsSize ), math.size(first.inputWeights) )   ||
        !math.deepEqual(math.matrix(first.layersWeightsSize), math.size(first.layersWeights))   ||
        !math.deepEqual(math.matrix(first.layersBiasSize   ), math.size(first.layersBias)   )   ||
        !math.deepEqual(math.matrix(first.outputWeightsSize), math.size(first.outputWeights))   ||
        !math.deepEqual(math.matrix(first.outputBiasSize   ), math.size(first.outputBias)   )
    ) {
        console.log('Failed constructor test: the array sizes are incorrect for a new AI');
        return -1;
    }

    console.log('----- TEST TWO -----');
    let firstSeed = first.seed;
    let second = new Checks(firstSeed);
    console.log(math.matrix(second.inputWeightsSize ) + ', ' + math.size(second.inputWeights));
    console.log(math.matrix(second.layersWeightsSize) + ', ' + math.size(second.layersWeights));
    console.log(math.matrix(second.layersBiasSize   ) + ', ' + math.size(second.layersBias));
    console.log(math.matrix(second.outputWeightsSize) + ', ' + math.size(second.outputWeights));
    console.log(math.matrix(second.outputBiasSize   ) + ', ' + math.size(second.outputBias));
    if (
        !math.deepEqual(math.matrix(second.inputWeightsSize ), math.size(second.inputWeights) )   ||
        !math.deepEqual(math.matrix(second.layersWeightsSize), math.size(second.layersWeights))   ||
        !math.deepEqual(math.matrix(second.layersBiasSize   ), math.size(second.layersBias)   )   ||
        !math.deepEqual(math.matrix(second.outputWeightsSize), math.size(second.outputWeights))   ||
        !math.deepEqual(math.matrix(second.outputBiasSize   ), math.size(second.outputBias)   )
    ) {
        console.log('Failed constructor test: the array sizes are incorrect for a new AI from a seed');
        return -1;
    }

    console.log('----- TEST THREE -----');
    console.log(math.deepEqual(first.inputWeights , second.inputWeights ));
    console.log(math.deepEqual(first.layersWeights, second.layersWeights));
    console.log(math.deepEqual(first.layersBias   , second.layersBias   ));
    console.log(math.deepEqual(first.outputWeights, second.outputWeights));
    console.log(math.deepEqual(first.outputBias   , second.outputBias   ));
    if (
        !math.deepEqual(first.inputWeights , second.inputWeights )   ||
        !math.deepEqual(first.layersWeights, second.layersWeights)   ||
        !math.deepEqual(first.layersBias   , second.layersBias   )   ||
        !math.deepEqual(first.outputWeights, second.outputWeights)   ||
        !math.deepEqual(first.outputBias   , second.outputBias   )
    ) {
        console.log('Failed constructor test: the new AI formed from the seed is not equivalent to the original');
        return -1;
    }

    console.log('Constructor tests passed. All functions are functional :)\n');
    return 1;
}

function evaluateTests() {
    console.log('Running evaluate tests:');
    console.log('----- TEST ONE -----');
    let first = new Checks();
    let firstInputs = math.random(math.matrix([2427]), -1, 1);;
    let firstRow = Checks.inputsToRow(firstInputs, first, true);
    console.log(math.size(firstRow.weightsXInputs));
    console.log(math.size(firstRow.withBias));
    console.log(math.size(firstRow.bounded));
    if (!math.deepEqual(math.size(firstRow.weightsXInputs), math.size(firstRow.withBias)) ||
        !math.deepEqual(math.size(firstRow.weightsXInputs), math.size(firstRow.bounded))) {
        console.log('Failed evaluate test: size changed during input to row conversion');
        return -1;
    }

    console.log('----- TEST TWO -----');
    let secondRow = Checks.advanceRow(firstRow.bounded, 0, first, true);
    console.log(math.size(secondRow.currentWeights));
    console.log(math.size(secondRow.weightsXRow   ));
    console.log(math.size(secondRow.currentBias   ));
    console.log(math.size(secondRow.withBias      ));
    console.log(math.size(secondRow.bounded       ));
    if (
        !math.deepEqual(math.size(firstRow.bounded), math.size(secondRow.weightsXRow   )) ||
        !math.deepEqual(math.size(firstRow.bounded), math.size(secondRow.currentBias   )) ||
        !math.deepEqual(math.size(firstRow.bounded), math.size(secondRow.withBias      )) ||
        !math.deepEqual(math.size(firstRow.bounded), math.size(secondRow.bounded       ))
    ) {
        console.log('Failed evaluate test: size changed during row advancement');
        return -1;
    }

    console.log('----- TEST THREE -----');
    let thirdRow = Checks.evaluateResult(secondRow.bounded, first, 0, true);
    console.log(math.size(thirdRow.outputWeight));
    console.log(thirdRow.totalWeights);
    console.log(thirdRow.outputBias);
    console.log(thirdRow.withBias);
    console.log(thirdRow.bounded);
    if (!math.deepEqual(math.size(firstRow.bounded), math.size(thirdRow.outputWeight))) {
         console.log('Failed evaluate test: size changed during output calculation');
         return -1;
    }

    console.log('----- TEST FOUR -----');
    let finalEvaluation = first.evaluate(firstInputs, 0);
    console.log(finalEvaluation);

    console.log('Evaluate tests passed. All functions are functional :)\n');
}

function helperFunctionTests() {
    console.log('Running helper function tests:');
    console.log('----- TEST ONE -----');
    let first = math.random(math.matrix([10]), -1, 0);
    let firstResponse =
    Checks.sigmoidMatrix(first, true);
    console.log(first);
    console.log(firstResponse.inverseMultiplicative);
    if (math.filter(math.isNegative(firstResponse.inverseMultiplicative), (e) => e).length) {
        console.log('Failed helper function test: An item is negative after passing through the sigmoid function');
        return -1;
    }

    console.log('----- TEST TWO -----');
    console.log(math.size(first));
    console.log(math.size(firstResponse.inverseSign));
    console.log(math.size(firstResponse.eToIt));
    console.log(math.size(firstResponse.with1));
    console.log(math.size(firstResponse.inverseMultiplicative));
    if (
        !math.deepEqual(math.size(first), math.size(firstResponse.inverseSign))   ||
        !math.deepEqual(math.size(first), math.size(firstResponse.eToIt))         ||
        !math.deepEqual(math.size(first), math.size(firstResponse.with1))         ||
        !math.deepEqual(math.size(first), math.size(firstResponse.inverseMultiplicative))
    ) {
        console.log('Failed helper function test: the array changed in size during an operation');
        return -1;
    }
    console.log('Helper function tests passed. All functions are functional :)\n');
}

//Running...
let results = {};
results.constructorTests    = constructorTests();
results['evaluateTests\t']  = evaluateTests();
results.helperFunctionTests = helperFunctionTests();

for (let i in results) {
    if (results[i] == -1) {
        console.log(i.toUpperCase() + ': \tFAIL');
    } else {
        console.log(i.toUpperCase() + ': \tPASS');
    }
}
