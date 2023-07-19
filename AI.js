/*
    IMPORTANT NOTE
    Currently, this is the AI file exactly as copied from MachTarok and without GPU acceleration
*/


const math = require('mathjs');
const Deck = require('./deck.js');
const Interface = require('./interface.js');
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
    //TODO: uncomment. For some reason H5 really likes to spam the console when the server crashes
    //if (!DEBUG_MODE) {
        return;
    //}
    h5wasm = await import("h5wasm");
    await h5wasm.ready;
    aiFromFile('latest.h5');
}
importH5wasm();

class AI {
    constructor (seed, mutate){
        if (seed) {
            //Matrix multiplication: Size[A,B] x Size[B,C] = Size[A,C]
            this.inputWeights = seed[0]; // 2k x 1k
            this.layersWeights = seed[1]; // 20 x 1k x 1k
            this.layersBias = seed[2]; // 20 x 1k x 1
            this.outputWeights = seed[3]; // 14 x 1k
            this.outputBias = seed[4]; // 14 x 1
        } else {
            this.inputWeightsSize = [2523,1000];
            this.layersWeightsSize = [20,1000,1000];
            this.layersBiasSize = [21,1000];
            this.outputWeightsSize = [1000,14];
            this.outputBiasSize = [14];

            /* CPU-based mathjs style
            this.inputWeights   = math.random(math.matrix([this.inputWeightsSize[0], this.inputWeightsSize[1]])); // 2k x 1k
            this.layersWeights  = math.random(math.matrix([this.layersWeightsSize[0], this.layersWeightsSize[1], this.layersWeightsSize[2]])); // 20 x 1k x 1k
            this.layersBias     = math.random(math.matrix([this.layersBiasSize[0], this.layersBiasSize[1]])); // 20 x 1k x 1
            this.outputWeights  = math.random(math.matrix([this.outputWeightsSize[1], this.outputWeightsSize[0]])); // 14 x 1k
            this.outputBias     = math.random(math.matrix([this.outputBiasSize[0]])); // 14 x 1
            */
            this.inputWeights   = Interface.createRandomMatrix(this.inputWeightsSize[0], this.inputWeightsSize[1]);
            this.layersWeights  = [];
            for (let i=0; i<this.layersWeightsSize[0]; i++) {
                this.layersWeights[i] = Interface.createRandomMatrix(this.layersWeightsSize[1], this.layersWeightsSize[2]);
            }
            this.layersBias     = [];
            for (let i=0; i<this.layersBiasSize[0]; i++) {
                Interface.createRandomMatrix(this.layersBiasSize[1], 1);
            }
            this.outputWeights  = Interface.createRandomMatrix(this.outputWeightsSize[1], this.outputWeightsSize[0]);
            this.outputBias     = Interface.createRandomMatrix(this.outputBiasSize[0], 1);

            mutate = 0;
        }
        if (mutate) {
            //Iterate over each and every weight and bias and add mutate * Math.random() to each
            /* old mathjs CPU randomization
            this.inputWeights  = math.add(this.inputWeights,  math.random([2523, 1000],     -mutate, mutate));
            this.layersWeights = math.add(this.layersWeights, math.random([20, 1000, 2523], -mutate, mutate));
            this.layersBias    = math.add(this.layersBias,    math.random([21, 1000],       -mutate, mutate));
            this.outputWeights = math.add(this.outputWeights, math.random([14, 1000],       -mutate, mutate));
            this.outputBias    = math.add(this.outputBias,    math.random([14],             -mutate, mutate));
            */
            this.inputWeights  = mutateMatrix(this.inputWeights , mutate);
            this.layersBias    = mutateMatrix(this.layersBias   , mutate);
            this.outputWeights = mutateMatrix(this.outputWeights, mutate);
            this.outputBias    = mutateMatrix(this.outputBias   , mutate);
            for (let i in this.layersWeights) {
                this.layersWeights[i] = mutateMatrix(this.layersWeights[i], mutate);
            }
            for (let i in this.layersBias) {
                this.layersBias[i] = mutateMatrix(this.layersBias[i], mutate);
            }
        }
    }

    evaluate(inputs, output) {
        let result = 0;

        //let currentRow = math.add(math.multiply(inputs, this.inputWeights), this.layersBias.subset(math.index(math.range(0,1),math.range(0,1000))));
        let currentRow = Interface.multiplyAndAddMatrix(inputs,this.inputWeights,this.layersBias[0]);
        currentRow = Interface.sigmoidMatrix(currentRow);

        for (let i=0; i<this.layersWeightsSize[0]; i++) {
            /*currentRow = AI.sigmoidMatrix(
                math.add(
                    math.multiply(
                        currentRow,
                        math.squeeze(
                            math.subset(
                                this.layersWeights, math.index(
                                    i,math.range(0,1000),math.range(0,1000)
                                )
                        ))),
                    math.squeeze(
                        math.subset(this.layersBias, math.index(
                            i+1,math.range(0,1000)
                    )))
                )
            );*/
            currentRow = Interface.sigmoidMatrix(
                Interface.multiplyAndAddMatrix(
                    currentRow,
                    this.layersWeights[i],
                    this.layersBias[i+1]
                )
            );
        }
        /*result = AI.sigmoid(
            math.add(
                math.multiply(
                    currentRow,
                    math.squeeze(
                        math.subset(
                            this.outputWeights,
                            math.index(output, math.range(0,1000))
                    ))
                ),
                math.subset(
                    this.outputBias,
                    math.index(output)
                )
            )
        );*/
        result = AI.sigmoid(
            Interface.multiplyAndAddMatrix(
                currentRow,
                this.outputWeights,
                this.outputBias
            )[output];
        );
        return result;
    }


    static sigmoid(z) {
        if (z<-10) {return 0;}
        else if (z>10) {return 1;}
        return 1 / (1 + Math.exp(-z));
    }

    static sigmoidMatrix(m) {
        //Assumes input to be an N x 1 matrix
        console.trace('Deprecated. Use Interface.sigmoidMatrix() instead')
        return math.map(math.add(1, math.map(math.subtract(0,m), math.exp)), math.inv);
    }

    static aiFromFile(file) {
        //Note: file is a location, not an actual file
        let f;
        let latestAI;
        try {
            const seed = [];
            f = new h5wasm.File(file, "r");
            seed[0] = f.get('/ai/inputWeights', 'r').to_array();
            seed[1] = f.get('/ai/layersWeights', 'r').to_array();
            seed[2] = f.get('/ai/layersBias', 'r').to_array();
            seed[3] = f.get('/ai/outputWeights', 'r').to_array();
            seed[4] = f.get('/ai/outputBias', 'r').to_array();
            latestAI = new AI(seed, 0);
            SERVER.log('AI loaded successfully');
        } catch (err) {
            SERVER.error('Error reading file from disk: ' + err);
            //latestAI = new AI(false, 0);
        } finally {
            if (f) {f.close();}
        }
        return latestAI;
    }

    static aiToFile(ai, fileName) {
        let saveFile;
        try {
            saveFile = new h5wasm.File(fileName,'w');
            saveFile.create_group('ai');

            let tempInputWeights = [];
            //todo change shape to be a flat array always
            let inputWeightsShape = ai.inputWeightsSize;
            ai.inputWeights.forEach(function (value, index, matrix) {
                tempInputWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset('inputWeights', tempInputWeights, inputWeightsShape, '<f');

            let tempLayersWeights = [];
            let layersWeightsShape = ai.layersWeightsSize;
            ai.layersWeights.forEach(function (value, index, matrix) {
                tempLayersWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset('layersWeights', tempLayersWeights, layersWeightsShape, '<f');

            let tempLayersBias = [];
            let layersBiasShape = ai.layersBiasSize;
            ai.layersBias.forEach(function (value, index, matrix) {
                tempLayersBias.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset('layersBias', tempLayersBias, layersBiasShape, '<f');

            let tempOutputWeights = [];
            let outputWeightsShape = ai.outputWeightsSize;
            ai.outputWeights.forEach(function (value, index, matrix) {
                tempOutputWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset('outputWeights', tempOutputWeights, outputWeightsShape, '<f');

            let tempOutputBias = [];
            let outputBiasShape = ai.outputBiasSize;
            ai.outputBias.forEach(function (value, index, matrix) {
                tempOutputBias.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset('outputBias', tempOutputBias, outputBiasShape, '<f');

            SERVER.log('Saved the latest ai ' + Date.now());
        } catch (err) {
            SERVER.error('Error writing file: ' + err);
        } finally {
            if (saveFile) {saveFile.close();}
        }
    }

//AI

/*
    AI theory
    Taroky is a complicated game. There are two main ways we could possibly implement AI
    1. Action-base. Give the AI what action is currently happening and let it select a choice
    2. Prompt-based. Look at all possible options and prompt the AI with each one. The AI will rank each option from 0-1 and the highest rank wins
    In this project, I'm going to use the second model.
    Here's how it looks (note that player numbers are using 0 as current player, +1 as player to the right, etc.):
    Every single value is between 0 and 1. S is for sigmoid
    INPUTS  (2k x 1)            HIDDEN LAYERS  (1k x 20)        OUTPUTS (14 x 1)
    0-3   S(chips/100)          Inputs is already the           Discard this
    4-7   isPovinnost           absolutely ludicrous            Play this
    8-11  isPrever              2k parameters. How many         Keep talon
    12    preverTalon           Hidden layers do we need?       Keep talon bottom
    13-44 8 types of            Honestly 20 should be           Keep talon top
          moneyCards            More than plenty.               Contra
    45-48 valat                 If hidden layers are 2kx20      Rhea-contra
    49-52 iote                  That means each layer has       Supra-contra
    53-56 contra                2k x 2k = 4m parameters,        Prever
    57    someoneIsValat        x20 layers = 80m parameters     Valat
    58    someoneIsContra       Which is insane.                IOTE
    59    someoneIsIOTE                                         povinnost b/u choice
    60    IAmValat              I'll test it out.               Play alone (call XIX)
    61    IAmIOTE               If it seems really slow         Play together (call lowest)
    62    IAmPrever             then I'll chop off 1k           Total: 14
    63-68 PartnerCard           1k x 1k = 1m parameters
          XIX, then XVIII       x20 = 20m total, much
          etc                   nicer on my computer.

    CURRENT TRICK INFORMATION
    69-72 TrickLeader
    73-76 myPositionInTrick
    +28   firstCard
    +28   secondCard
    +28   thirdCard

    TRICK HISTORY
    +1    hasBeenPlayed
    +4    whoLead
    +4    myPosition
    +4    whoWon
    +28   firstCard
    +28   secondCard
    +28   thirdCard
    +28   fourthCard
    x11   tricks

    MY HAND
    +28   card
    x16   Max num cards in hand

    PREVER TALON
    +28   Card
    x3    num cards in talon

    PARTNER INFORMATION
    -Only information the AI should know-
    +3    isMyPartner

    TRUMP DISCARD
    +28   card
    x4    max

    CURRENT CARD/ACTION
    +28   card
    +25   number of actions

    Roughly 2k inputs total

    Matrix layout

    Matrix {
        INPUTS {0,1,0,0... 2k}
        INPUT_ROW {
            w, w, w, w... 2k
            w, w, w, w... 2k
            ...
            2k
        }
        INPUT_BIAS {b, b, b... 2k}
        LAYER 1 {0.75, 0.23, 0.01... 2k}
        L1 = I x IR + IB;
        2kx1 = 2kx1 x 2kx2k + 2kx1; //Checks out
        HIDDEN_LAYER_1_ROW {
            w, w, w, w... 2k
            w, w, w, w... 2k
            ...
            2k
        }... 20x

        LAYER 20 {0.75, 0.23, 0.01... 2k}

        HIDDEN_LAYER_20_ROW {
            -------> 2k
            -------> 2k
            ...
            14
        }
        L20B = {b, b, b... 14}
        OUTPUT ROW = L20 x L20R + L20B;
        1x14 = 1x2k x 2kx14 + 1x14; //Checks out

        Actual matrix:
        in      = new matrix [2k, 1k]
        layers  = new matrix [20, 1k, 1k]
        layersB = new matrix [20, 1k]
        out     = new matrix [1k, 14]
        outB    = new matrix [14]

        Note that layers has an extra dimension because there are n layers (n=20) and always 1 in and 1 out
        B has 1 less layer than w because N = O x W + B means that preservation of matrix size requires W to be O-width wide and high, whereas B needs to be O-width wide but only 1 tall

        Also, only the required output must be calculated
    }
*/

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

module.exports = AI;