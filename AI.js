const math = require('mathjs');
const fs = require('fs');
//const Interface = require('./interface.js');
//GPUjs was a TON slower than mathjs (25s vs 4s, 2.5s vs 7ms)
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
    //AI.leader = new AI();
    AI.leader = AI.aiToFile(new AI(), 'latest.h5')
    //AI.leader = AI.aiFromFile('latest.h5');
}

class AI {
    constructor (seed, mutate){
        this.inputWeightsSize  = [2427,100];
        this.layersWeightsSize = [6,100,100];
        this.layersBiasSize    = [7,100,1];
        this.outputWeightsSize = [100,14];
        this.outputBiasSize    = [14,1];
        this.vIWS = [2427,100];
        this.vLWS = [6,100,100];
        this.vLBS = [7,100];
        this.vOWS = [14,100];
        this.vOBS = [14];
        if (seed) {
            //Matrix multiplication: Size[A,B] x Size[B,C] = Size[A,C]
            /*
            seed[0]'/ai/inputWeights
            seed[1] /ai/layersWeights
            seed[2] /ai/layersBias
            seed[3] /ai/outputWeights
            seed[4] /ai/outputBias
            */
            this.inputWeights = seed[0];
            this.layersWeights = seed[1];
            this.layersBias = seed[2];
            this.outputWeights = seed[3];
            this.outputBias = seed[4];
        }
        if (!seed) {
            //CPU-based mathjs style
            this.inputWeights   = math.random(math.matrix([this.inputWeightsSize[0], this.inputWeightsSize[1]]), -1, 1); // 2k x 1k
            this.layersWeights  = math.random(math.matrix([this.layersWeightsSize[0], this.layersWeightsSize[1], this.layersWeightsSize[2]]), -1, 1); // 20 x 1k x 1k
            this.layersBias     = math.random(math.matrix([this.layersBiasSize[0], this.layersBiasSize[1]]), -1, 1); // 20 x 1k x 1
            this.outputWeights  = math.random(math.matrix([this.outputWeightsSize[0], this.outputWeightsSize[1]]), -1, 1); // 14 x 1k
            this.outputBias     = math.random(math.matrix([this.outputBiasSize[0]]), -1, 1); // 14 x 1
            /*
            this.inputWeights   = Interface.createRandomMatrix(this.inputWeightsSize[0], this.inputWeightsSize[1]);
            this.layersWeights  = [];
            for (let i=0; i<this.layersWeightsSize[0]; i++) {
                this.layersWeights[i] = Interface.createRandomMatrix(this.layersWeightsSize[1], this.layersWeightsSize[2]);
            }
            this.layersBias     = [];
            for (let i=0; i<this.layersBiasSize[0]; i++) {
                this.layersBias[i] = Interface.createRandomMatrix(this.layersBiasSize[1], this.layersBiasSize[2]);
            }
            this.outputWeights  = Interface.createRandomMatrix(this.outputWeightsSize[1], this.outputWeightsSize[0]);
            this.outputBias     = Interface.createRandomMatrix(this.outputBiasSize[0], 1);
            */
            mutate = 0;
        }
        if (mutate) {
            //Iterate over each and every weight and bias and add mutate * Math.random() to each
            //old mathjs CPU randomization
            this.inputWeights  = math.add(this.inputWeights,  math.random(this.inputWeightsSize , -mutate, mutate));
            this.layersWeights = math.add(this.layersWeights, math.random(this.layersWeightsSize, -mutate, mutate));
            this.layersBias    = math.add(this.layersBias,    math.random(this.layersBiasSize   , -mutate, mutate));
            this.outputWeights = math.add(this.outputWeights, math.random(this.outputWeightsSize, -mutate, mutate));
            this.outputBias    = math.add(this.outputBias,    math.random(this.outputBiasSize   , -mutate, mutate));
            /*
            this.inputWeights  = mutateMatrix(this.inputWeights , mutate);
            this.outputWeights = mutateMatrix(this.outputWeights, mutate);
            this.outputBias    = mutateMatrix(this.outputBias   , mutate);
            for (let i in this.layersWeights) {
                this.layersWeights[i] = mutateMatrix(this.layersWeights[i], mutate);
            }
            for (let i in this.layersBias) {
                this.layersBias[i] = mutateMatrix(this.layersBias[i], mutate);
            }*/
        }
    }

    evaluate(inputs, output) {
        let result = 0;

        let currentRow = math.add(math.multiply(math.squeeze(inputs), this.inputWeights), this.layersBias.subset(math.index(math.range(0,1),math.range(0,this.layersBiasSize[1]))));
        //let currentRow = Interface.multiplyAndAddMatrix(inputs,this.inputWeights,this.layersBias[0]);
        //currentRow = Interface.sigmoidMatrix(currentRow);

        for (let i=0; i<this.layersWeightsSize[0]; i++) {
            currentRow = AI.sigmoidMatrix(
                math.add(
                    math.multiply(
                        currentRow,
                        math.squeeze(
                            math.subset(
                                this.layersWeights, math.index(
                                    i,math.range(0,this.layersWeightsSize[1]),math.range(0,this.layersWeightsSize[2])
                                )
                        ))),
                    math.squeeze(
                        math.subset(this.layersBias, math.index(
                            i+1,math.range(0,this.layersBiasSize[1])
                    )))
                )
            );/*
            currentRow = Interface.sigmoidMatrix(
                Interface.multiplyAndAddMatrix(
                    currentRow,
                    this.layersWeights[i],
                    this.layersBias[i+1]
                )
            );*/
        }
        result = AI.sigmoid(
            math.add(
                math.multiply(
                    currentRow,
                    math.squeeze(
                        math.subset(
                            this.outputWeights,
                            math.index(math.range(0,this.outputWeightsSize[0]),+output)
                    ))
                ),
                math.subset(
                    this.outputBias,
                    math.index(+output)
                )
            ).get([0])
        );
        /*
        result = AI.sigmoid(
            Interface.multiplyAndAddMatrix(
                currentRow,
                this.outputWeights,
                this.outputBias
            )[0][output]
        );*/
        return result;
    }

    backpropagation(inputs, output, value) {
        output = +output;
        value = +value;
        //console.log('backpropagation called with values: ' + inputs + ' ' + output + ' ' + value);
        //Start by caching values during evaluation
        const a = [];//a is the activation values of each layer of neurons
        let currentRow = math.add(math.multiply(math.squeeze(inputs), this.inputWeights), this.layersBias.subset(math.index(math.range(0,1),math.range(0,this.layersBiasSize[1]))));
        a[0] = math.matrix(currentRow);

        for (let i=0; i<this.layersWeightsSize[0]; i++) {
            currentRow = AI.sigmoidMatrix(
                math.add(
                    math.multiply(
                        currentRow,
                        math.squeeze(
                            math.subset(
                                this.layersWeights, math.index(
                                    i,math.range(0,this.layersWeightsSize[1]),math.range(0,this.layersWeightsSize[2])
                                )
                        ))),
                    math.squeeze(
                        math.subset(this.layersBias, math.index(
                            i+1,math.range(0,this.layersBiasSize[1])
                    )))
                )
            );
            a[i+1] = math.matrix(currentRow);
        }
        currentRow = AI.sigmoidMatrix(
            math.add(
                math.multiply(
                    currentRow,
                    math.squeeze(
                        math.subset(
                            this.outputWeights,
                            math.index(math.range(0,this.outputWeightsSize[0]),0)
                        )
                    )
                ),
                math.squeeze(
                    this.outputBias
                )
            )
        );

        a[a.length] = math.matrix(currentRow);

        let L = a.length - 1;
        let cost = Math.pow(value - currentRow.get([output]),2);
        //console.log(cost);

        let outputSigCost = math.dotMultiply(
                a[L].get([output]),
                math.dotMultiply(
                    math.subtract(
                        a[L].get([output]),
                        value
                    ),
                    math.dotMultiply(
                        2,
                        math.subtract(
                            1,
                            a[L].get([output])
                        )
                    )
                )
            );
        let outputWeightsCost = math.multiply(
            a[L - 1], outputSigCost
        );
        /*
        math.subset(
        this.layersWeights, math.index(
            i,math.range(0,this.layersWeightsSize[1]),math.range(0,this.layersWeightsSize[2])
        )
        */

        let previousCost = math.squeeze(math.dotMultiply(
            math.subset(this.outputWeights, math.index(
                math.range(0,this.outputWeightsSize[0]),output
            )), outputSigCost
        ));
        math.subset(
            this.outputWeights,
            math.index(math.range(0,this.outputWeightsSize[0]),output),
            math.add(
                math.squeeze(
                    math.subset(
                        this.outputWeights,
                        math.index(math.range(0,this.outputWeightsSize[0]),output)
                    )),
                math.squeeze(outputWeightsCost)
            )
        );
        //outputBiasCost is outputSigCost, since it's just that times 1
        math.subset(this.outputBias, math.index(
                        output
                    ), math.add(math.subset(this.outputBias, math.index(
                        output
                    )), outputSigCost));

        //console.log(a);
        //console.log(math.size(previousCost));//should be array [100]

        for (let i = this.layersWeightsSize[0]; i > 0; i--) {
            //console.log('previouscost: ' + math.size(previousCost));//should be 100
            //start in the last layer, with the output
            //in the evaluate function, we've already calculated Z so it should be cached
            let sigCost = math.squeeze(math.dotMultiply(
                a[i],
                math.squeeze(
                    math.dotMultiply(
                        previousCost,
                        math.squeeze(
                            math.map(a[i], function(value) {
                                return (1 - value)
                            })
                        )
                    )
                )
            ));
            //console.log('sigcost: ' + math.size(sigCost));//should be array 100
            let layerWeightsCost = math.dotMultiply(
                a[i - 1], math.squeeze(sigCost)
            );
            previousCost = math.dotMultiply(
                    math.squeeze(math.subset(this.layersWeights, math.index(i - 1, 0, math.range(0,this.layersWeightsSize[2])))),
                math.squeeze(sigCost)
            );
            for (let j=1; j<this.layersWeightsSize[1]; j++) {
                previousCost = math.add(
                    previousCost,
                    math.dotMultiply(
                        math.squeeze(math.subset(this.layersWeights, math.index(i - 1, j, math.range(0,this.layersWeightsSize[2])))),
                        math.squeeze(sigCost)
                    )
                );
            }
            math.subset(
                this.layersWeights,
                math.index(i - 1, math.range(0,this.layersWeightsSize[1]), math.range(0,this.layersWeightsSize[2])),
                math.add(math.squeeze(math.subset(this.layersWeights, math.index(i - 1, math.range(0,this.layersWeightsSize[1]), math.range(0,this.layersWeightsSize[2])))), layerWeightsCost)
            );
            math.subset(
                this.layersBias,
                math.index(i, math.range(0,this.layersBiasSize[1])),
                math.add(math.squeeze(math.subset(this.layersBias,math.index(i, math.range(0,this.layersBiasSize[1])))), sigCost)
            );
            //let delW1 = a[L - 1][1] * sigmoid(z) * (1 - sigmoid(z)) * 2 * (a[L][output] - value);
        }
        //console.log(math.size(previousCost));//100 x 100
        //console.log(math.size(a[0]));//100
        //input weights and biases
        let inputSigCost = math.map(
            math.dotMultiply(
                a[0],
                math.dotMultiply(
                    math.subtract(
                        a[0],
                        previousCost
                    ),
                    math.squeeze(
                        math.map(a[0], function(value) {
                            return (1 - value)
                        })
                    )
                )
            ), function(v) {return v*2}
        );
        //console.log(math.size(inputs));//2427
        //console.log(math.size(inputSigCost));//1?
        //console.log(inputSigCost);
        let inputWeightsCost = math.multiply(
            math.transpose(inputs), inputSigCost
        );
        //console.log(math.size(inputWeightsCost));//2427
        //console.log(math.size(this.inputWeights));//2427 x 100
        this.inputWeights = math.add(this.inputWeights, inputWeightsCost);
        math.subset(
            this.layersBias,
            math.index(0, math.range(0,this.layersBiasSize[1])),
            math.add(math.squeeze(math.subset(this.layersBias,math.index(0, math.range(0,this.layersBiasSize[1])))), inputSigCost)
        );
        return cost;
    }

    static winner(ai) {
        AI.leader = new AI(ai.seed, 0);
        aiToFile(ai, 'latest.h5');
    }

    static sigmoid(z) {
        if (z == null) {
            console.trace('null');
        }
        if (z<-10) {return 0;}
        else if (z>10) {return 1;}
        return 1 / (1 + Math.exp(-z));
    }

    static sigmoidMatrix(m) {
        //Assumes input to be an N x 1 matrix
        //console.trace('Deprecated. Use Interface.sigmoidMatrix() instead')
        return math.map(math.add(1, math.map(math.subtract(0,m), math.exp)), math.inv);
    }

    static aiFromFile(file) {
        //Note: file is a location, not an actual file
        let f;
        let latestAI;
        try {
            if (!fs.existsSync(file)) {
                throw "File does not yet exist. Creating a new AI...";
            }
            const seed = [];
            f = new h5wasm.File(file, "r");
            seed[0] = math.squeeze(math.matrix(f.get('/ai/inputWeights', 'r').to_array()));
            seed[1] = math.squeeze(math.matrix(f.get('/ai/layersWeights', 'r').to_array()));
            seed[2] = math.squeeze(math.matrix(f.get('/ai/layersBias', 'r').to_array()));
            seed[3] = math.squeeze(math.matrix(f.get('/ai/outputWeights', 'r').to_array()));
            seed[4] = math.squeeze(math.matrix(f.get('/ai/outputBias', 'r').to_array()));
            latestAI = new AI(seed, 0);
            console.log('AI loaded successfully');
        } catch (err) {
            console.error('Error reading file from disk: ' + err);
            if (AI.leader) {
                //personalized AI start out as the best AI that exists so far
                latestAI = new AI(AI.leader.seed, 0);
            } else {
                latestAI = new AI(false, 1);
            }
            console.log('AI loaded');
        } finally {
            if (f) {f.close();}
        }
        return latestAI;
    }

    static aiToFile(ai, fileName) {
        let saveFile;
        try {
            if (!ai) {
                throw "No ai input";
            }
            saveFile = new h5wasm.File(fileName,'w');
            saveFile.create_group('ai');

            let inputWeightsShape = ai.vIWS;
            let tempInputWeights = [];
            ai.inputWeights.forEach(function (value, index, matrix) {
                tempInputWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset({name:'inputWeights', data:tempInputWeights, shape:inputWeightsShape, dtype:'<f'});

            //saveFile.get('ai').create_dataset('inputWeights', math.flatten(ai.inputWeights), inputWeightsShape, '<f');

            //TODO: use math.flatten for the save file

            let tempLayersWeights = [];
            let layersWeightsShape = ai.vLWS;
            ai.layersWeights.forEach(function (value, index, matrix) {
                tempLayersWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset({name:'layersWeights', data:tempLayersWeights, shape:layersWeightsShape, dtype:'<d'});

            let tempLayersBias = [];
            let layersBiasShape = ai.vLBS;
            ai.layersBias.forEach(function (value, index, matrix) {
                tempLayersBias.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset({name:'layersBias', data:tempLayersBias, shape:layersBiasShape, dtype:'<f'});

            let tempOutputWeights = [];
            let outputWeightsShape = ai.vOWS;
            ai.outputWeights.forEach(function (value, index, matrix) {
                tempOutputWeights.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset({name:'outputWeights', data:tempOutputWeights, shape:outputWeightsShape, dtype:'<f'});

            let tempOutputBias = [];
            let outputBiasShape = ai.vOBS;
            ai.outputBias.forEach(function (value, index, matrix) {
                tempOutputBias.push(value);//Lines up the 2d array into 1 dimension
            });
            saveFile.get('ai').create_dataset({name:'outputBias', data:tempOutputBias, shape:outputBiasShape, dtype:'<f'});

            console.log('Saved the latest ai ' + Date.now());
        } catch (err) {
            console.error('Error writing file: ' + err);
        } finally {
            if (saveFile) {saveFile.close();}
        }
        return ai;
    }
    static leader;

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

importH5wasm();
//AI.leader = new AI();
module.exports = AI;