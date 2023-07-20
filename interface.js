const { GPU, input } = require('gpu.js');
const gpu = new GPU();

class Interface {
    // WH x W2H2 = HW2 Original height and 2nd width are preserved. Others must be =

    static createMatrixMultiplier(resultWidth, resultHeight, inner) {
        return gpu.createKernel(function(a, b) {
            let sum = 0;
            for (let i = 0; i < this.constants.size; i++) {
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
        }).setOutput([resultWidth, resultHeight]).setConstants({size: inner});
    }
    static createMatrixAdder(width, height) {
        return gpu.createKernel(function(a, b) {
            return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
        }).setOutput([width, height]);
    }
    static createMatrixMultiplierAndAdder(firstWidth, firstHeight, secondWidth, secondHeight, thirdWidth, thirdHeight) {
        if (firstWidth != secondHeight) {
            console.log(firstWidth + ' != ' + secondHeight);
            throw 'Invalid dimensions for matrix multiplication';
        }
        if (firstHeight != thirdHeight || secondWidth != thirdWidth) {
            throw 'Invalid dimensions for matrix addition';
        }
        return (inputs, weights, bias) => {
            let add = Interface.createMatrixAdder(thirdWidth, thirdHeight);
            let multiply = Interface.createMatrixMultiplier(secondWidth, firstHeight, firstWidth);
            //inputs * weights + bias
            return add(multiply(inputs, weights), bias);
        }
    }
    static multiplyAndAddMatrix(inputs, weights, biases) {
        const multiplyAndAddKernel = Interface.createMatrixMultiplierAndAdder(
            inputs[0].length, inputs.length,
            weights[0].length, weights.length,
            biases[0].length, biases.length
        );
        return multiplyAndAddKernel(inputs, weights, biases);
    }
    static createRandomMatrix(width, height) {
        return ((w, h) => {
            const matrix = [];
            for (let y = 0; y < h; y++){
                matrix.push([]);
                for (let x = 0; x < w; x++){
                  matrix[y].push(Math.random()*2-1);
                }
            }
            return matrix;
        })(width, height);
    }
    static mutateMatrix(matrix, mutate) {
        return createMatrixAdder(matrix[0].length, matrix.length)(matrix, createRandomMatrix(matrix[0].length, matrix.length));
    }
    static sigmoidMatrix(matrix) {
        let width = matrix[0].length;
        let height = matrix.length;
        const sigmoidKernel = gpu.createKernel(function(a) {
                let z = a[this.thread.y][this.thread.x];
                if (z<-10) {return 0;}
                else if (z>10) {return 1;}
                return 1 / (1 + Math.exp(-z));
            }).setOutput([width, height]);
        return sigmoidKernel(matrix);
    }
}

module.exports = Interface;


/*
[ First number = y coordinate
[1,2,3,4,5... second number = x coordinate]
[1,2,3,4,5...]
[1,2,3,4,5...]
...
]
Third number = z coordinate if applicable

ex: matrix[y][x][z]

Notes:
All inputs are expected to be 2d matrices, including biases

Bias example (vector but 2d):
[
[1],
[2],
[3]
]

*/