/*
This is the Taroky AI for MachTarok.com

This AI server is only meant to be accessible BACK-END
Effectively, this means that there is no way to access it externally or via the website
Instead, it must be accessed locally through the MachTarok server, which extends the functionality to the front end

There are several interface methods, all of which prioritize transmitting as little data as possible

Additionally, this AI server relies heavily on NVIDIA's RAPIDS architecture, so that must be installed before running

*/

const http = require('http');
const url = require('url');
const fs = require('fs');
const path = require('path');
const express = require('express');
const math = require('mathjs');

const app = express();

const AI = require('./AI.js');

//AI variables
let leader;
let followers = [];

let players = {};
let personalized;

//Interface set up
const server = http.createServer((req, res) => {
    let q = url.parse(req.url, true);
    if (req.method == 'GET') {
        //Unlike standard HTTP file-serving, GET in this context is not to get files, but a JSON AI response
        //The headers contain information on the room information, while the path contains which AI should be accessed
        if (!req.headers.inputs || !req.headers.output) {
            res.writeHead(400);
            return res.end();
        }

        switch (q.pathname.split('/')[1]) {
            case 'standard':
                //Standard AI, no deep learning
                res.write(standardAI(req.headers.inputs, req.headers.output));
                res.writeHead(200);
                return res.end();
            case 'personalized':
                //Deep learning, based on human interactions
                res.write(personalizedAI(req.headers.inputs, req.headers.output, req.headers.id, q.pathname.split('/')[2]));
                res.writeHead(200);
                return res.end();
        }
    }
}

//Standard self-battling
function standardAI(inputs, output) {
    //Inputs should be a several-thousand long array of boolean values (false represents 0, true 1)
    return {answer:leader.evaluate(inputs, output)};
    //Returns an object so that other parameters can be added in the future
}

function personalizedAI(inputs, output, id, name) {
    //Again, inputs are several-thousand long. This time, however, deep learning is used to retrieve the answer
    return {answer:players[name].evaluate(inputs, output)};
    //Returns an object so that other parameters can be added in the future
}