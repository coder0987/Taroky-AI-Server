//Something is wrong with the original code in AI.js, and it's too complicated to figure out what
//So, instead of debugging, I'm rewriting the program in smaller increments and using tests to check each function

const math = require('mathjs');
const fs = require('fs');
const { SUIT,
    SUIT_REVERSE,
    RED_VALUE,
    BLACK_VALUE,
    TRUMP_VALUE,
    VALUE_REVERSE,
    DIFFICULTY,
    DIFFICULTY_TABLE,
    MESSAGE_TYPE,
    PLAYER_TYPE } = require('./enums.js');
let h5wasm = null;
async function importH5wasm() {
    h5wasm = await import("h5wasm");
    await h5wasm.ready;
    //Do stuff
}

class Checks {
    constructor (seed, mutate){
        this.inputWeightsSize  = [2427,100];
        this.layersWeightsSize = [6,100,100];
        this.layersBiasSize    = [7,100];
        this.outputWeightsSize = [100,14];
        this.outputBiasSize    = [14];
        if (seed) {
            this.inputWeights = seed[0];
            this.layersWeights = seed[1];
            this.layersBias = seed[2];
            this.outputWeights = seed[3];
            this.outputBias = seed[4];
            if (mutate) {
                //Iterate over each and every weight and bias and add mutate * Math.random() to each
                this.inputWeights  = math.add(this.inputWeights,  math.random(this.inputWeightsSize , -mutate, mutate));
                this.layersWeights = math.add(this.layersWeights, math.random(this.layersWeightsSize, -mutate, mutate));
                this.layersBias    = math.add(this.layersBias,    math.random(this.layersBiasSize   , -mutate, mutate));
                this.outputWeights = math.add(this.outputWeights, math.random(this.outputWeightsSize, -mutate, mutate));
                this.outputBias    = math.add(this.outputBias,    math.random(this.outputBiasSize   , -mutate, mutate));
            }
        } else {
            //CPU-based mathjs style
            this.inputWeights   = math.random(math.matrix([this.inputWeightsSize[0], this.inputWeightsSize[1]]), -1, 1); // 2k x 1k
            this.layersWeights  = math.random(math.matrix([this.layersWeightsSize[0], this.layersWeightsSize[1], this.layersWeightsSize[2]]), -1, 1); // 20 x 1k x 1k
            this.layersBias     = math.random(math.matrix([this.layersBiasSize[0], this.layersBiasSize[1]]), -1, 1); // 20 x 1k x 1
            this.outputWeights  = math.random(math.matrix([this.outputWeightsSize[0], this.outputWeightsSize[1]]), -1, 1); // 14 x 1k
            this.outputBias     = math.random(math.matrix([this.outputBiasSize[0]]), -1, 1); // 14 x 1
        }
    }

    evaluate(inputs, output) {
        let currentRow = Checks.inputsToRow(inputs, this);

        for (let i=0; i<this.layersWeightsSize[0]; i++) {
            currentRow = Checks.advanceRow(currentRow, i, this);
        }

        let result = Checks.evaluateResult(currentRow, this, output);

        return result;
    }

    static inputsToRow(inputs, ai, testing) {
        //Step 1: Get the inputs and multiply them by the weights
        //  Size 1B        =                 1A   x   AB
        let weightsXInputs = math.multiply(math.squeeze(inputs), ai.inputWeights);


        //Step 2: Retrieve a subset of layersBias because layersBias is an array of bias vectors and the inputs only needs the first vector
        let currentBias = math.squeeze(ai.layersBias.subset(math.index(math.range(0,1),math.range(0,ai.layersBiasSize[1]))))

        //Step 3: add bias to each node on the newly formed row of size B
        // Size 1B   =   1B + 1B
        let withBias = math.add(weightsXInputs, currentBias);

        //Step 4: use the sigmoid function to bound the result between 0 and 1
        let bounded = Checks.sigmoidMatrix(withBias);

        if (testing) {
            return {'weightsXInputs': weightsXInputs, 'withBias': withBias, 'bounded':bounded};
        }

        return bounded;
    }

    static advanceRow(currentRow, i, ai, testing) {
        //Step 1: Retrieve the layer weights from the layer weights array
        let currentWeights = math.squeeze(
                                 math.subset(
                                     ai.layersWeights, math.index(
                                         i,math.range(0,ai.layersWeightsSize[1]),math.range(0,ai.layersWeightsSize[2])
                                     )
                             ));
        //Step 2: Multiply the weights by the current row
        let weightsXRow = math.multiply(currentRow,currentWeights);

        //Step 3: Retrieve the bias from the layer bias array
        let currentBias = math.squeeze(
                              math.subset(ai.layersBias, math.index(
                                  i+1,math.range(0,ai.layersBiasSize[1])
                          )))

        //Step 4: Add the bias to the current row
        let withBias = math.add(weightsXRow, currentRow)

        //Step 5: Apply the sigmoid function to bound the results between 0 and 1
        let bounded = Checks.sigmoidMatrix(withBias);

        if (testing) {
            return {
                'currentWeights': currentWeights,
                'weightsXRow'   : weightsXRow,
                'currentBias'   : currentBias,
                'withBias'      : withBias,
                'bounded'       : bounded
            }
        }

        return bounded;
    }

    static evaluateResult(currentRow, ai, output, testing) {
        //Step 1: get the output weights associated with the required output
        let outputWeight = math.squeeze(
                               math.subset(
                                   ai.outputWeights,
                                   math.index(math.range(0,ai.outputWeightsSize[0]),+output)
                           ));

        //Step 2: Multiply the current row by each weight and sum the total
        let totalWeights = math.multiply(currentRow,outputWeight);

        //Step 3: Get the associated bias
        let outputBias = math.subset(ai.outputBias,math.index(+output));

        //Step 4: Add the bias to the previous sum
        let withBias = math.add(totalWeights, outputBias);

        //Step 5: Bound the result between 0 and 1
        let bounded = Checks.sigmoid(withBias);

        if (testing) {
            return {
                'outputWeight': outputWeight,
                'totalWeights': totalWeights,
                'outputBias': outputBias,
                'withBias': withBias,
                'bounded': bounded
            }
        }

        return bounded;
    }

    static sigmoid(z) {
        if (z == null) {
            console.trace('null');
        }
        if (z<-10) {return 0;}
        else if (z>10) {return 1;}
        return 1 / (1 + Math.exp(-z));
    }

    static sigmoidMatrix(m, testing) {
        //Assumes input to be an N x 1 matrix
        //Step 1: Multiply each item by -1
        let inverseSign = math.subtract(0,m);

        //Step 2: raise e to the power of each item, individually
        let eToIt = math.map(inverseSign, math.exp);

        //Step 3: Add 1
        let with1 = math.add(1, eToIt);

        //Step 4: invert each item
        let inverseMultiplicative = math.map(with1, math.inv);

        if (testing) {
            return {
                'inverseSign': inverseSign,
                'eToIt': eToIt,
                'with1': with1,
                'inverseMultiplicative': inverseMultiplicative
            };
        }

        return inverseMultiplicative;
    }

    get seed() {
        let theSeed = [];
        theSeed.push(this.inputWeights);
        theSeed.push(this.layersWeights);
        theSeed.push(this.layersBias);
        theSeed.push(this.outputWeights);
        theSeed.push(this.outputBias);
        return theSeed;
    }
}

module.exports = Checks;