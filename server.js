#!/usr/bin/env node

/*
This is the Taroky AI for MachTarok.com

This AI server is only meant to be accessible BACK-END
Effectively, this means that there is no way to access it externally or via the website
Instead, it must be accessed locally through the MachTarok server, which extends the functionality to the front end

There are several interface methods, all of which prioritize transmitting as little data as possible

Additionally, this AI server relies heavily on GPU JS, so that must be installed before running
*/

const http = require('http');
const url = require('url');
const fs = require('fs');
const path = require('path');
const express = require('express');
const math = require('mathjs');
const { Buffer } = require('node:buffer');
function bufferToArray(buf){
    if (buf.length > 0) {
        const data = new Array(buf.length);
        for (let i = 0; i < buf.length; i=i+1)
            data[i] = buf[i];
        return data;
    }
    return [];
}


const app = express();

const AI = require('./AI.js');

//AI variables
//AI.leader
let followers = [];

let players = {};
let personalized = {};

//Interface set up
const server = http.createServer((req, res) => {
    let q = url.parse(req.url, true);
    if (req.method == 'POST') {
        //Unlike standard HTTP file-serving, POST in this context is not to get files, but a JSON AI response
        //The headers contain information on the room information, while the path contains which AI should be accessed
        let body = [];
        req.on('data', (chunk) => {
            body.push(chunk);
        }).on('end', () => {
            body = Buffer.concat(body);
            body = bufferToArray(body);

            if (body) {
                postDataComplete(body, req, res);
            } else {
                res.writeHead(400);
                return res.end();
            }
        });
    } else if (req.method == 'GET') {
        //For anything that does not require inputs
        switch (q.pathname.split('/')[1]) {
            case 'trainAI':
                switch (q.pathname.split('/')[3]) {
                    case 'win':
                        if (players[q.pathname.split('/')[2]]) {
                            AI.leader = players[q.pathname.split('/')[2]];
                            removeAITrainers(req.headers.ids);
                        } else {
                            res.writeHead(400);
                            return res.end();
                        }
                        break;
                    case 'create':
                        createAITrainers(q.pathname.split('/')[2]);
                        break;
                }
                break;
                res.writeHead(200);
                return res.end();
        }
    }
});

function postDataComplete(postData, req, res) {
    let q = url.parse(req.url, true);
    if (!postData || postData == '' || !req.headers.output) {
        res.writeHead(400);
        return res.end();
    }

    if (postData.length != 2427) {
        //something went wrong
        console.log(postData);
    }
    postData = [postData];//Interface expects a 2D array

    let st = Date.now();

    switch (q.pathname.split('/')[1]) {
        case 'standard':
            //Standard AI, no deep learning
            let msg = standardAI(postData, req.headers.output);
            res.write(msg, 'utf8');
            res.writeHead(200);
            console.log('Calculation finished in ' + (Date.now() - st) + 'ms');
            return res.end();
        case 'personalized':
            //Deep learning, based on human interactions
            res.write(personalizedAI(postData, req.headers.output, q.pathname.split('/')[2].toLowerCase()));
            res.writeHead(200);
            console.log('Calculation finished in ' + (Date.now() - st) + 'ms');
            return res.end();
        case 'trainPlayer':
            //For personalized ai meant to imitate a player
            let w = trainPersonalizedAI(postData, req.headers.output, q.pathname.split('/')[2].toLowerCase(), req.headers.value);
            if (w==200 && req.headers.save) {
                AI.aiToFile(personalized[q.pathname.split('/')[2].toLowerCase()],q.pathname.split('/')[2].toLowerCase() + '.h5');
            }
            res.writeHead(w);
            console.log('Calculation finished in ' + (Date.now() - st) + 'ms');
            return res.end();
        case 'trainAI':
            //For standard ai trying to be better against itself
            switch (q.pathname.split('/')[3]) {
                case 'play':
                    let msg = trainAI(q.pathname.split('/')[2], postData, req.headers.output);
                    res.write(msg, 'utf8');
                    break;
            }

            res.writeHead(200);
            console.log('Calculation finished in ' + (Date.now() - st) + 'ms');
            return res.end();
    }
}

//Standard self-battling
function standardAI(inputs, output) {
    //Inputs should be a several-thousand long array of boolean values (false represents 0, true 1)

    const response = {answer:AI.leader.evaluate(inputs, output)};
    const stringResponse = JSON.stringify(response)
    console.log('AI response sent: ' + stringResponse);

    return stringResponse;
    //Returns an object so that other parameters can be added in the future
}

function trainAI(id, inputs, output) {
    //Inputs should be a several-thousand long array of boolean values (false represents 0, true 1)
    const response = {answer:players[id].evaluate(inputs, output)};
    const stringResponse = JSON.stringify(response);
    //console.log('AI response sent: ' + stringResponse);
    return stringResponse;
    //Returns an object so that other parameters can be added in the future
}

function createAITrainers(id) {
    if (id == 'latest') {
        players[id] = new AI(AI.leader.seed, 0);
    } else {
        players[id] = new AI(AI.leader.seed, 0.5);
    }
}

function removeAITrainers(ids) {
    for (let id in ids) {
        delete players[id];
    }
}

function personalizedAI(inputs, output, name) {
    //Again, inputs are several-thousand long. This time, however, deep learning is used to retrieve the answer
    const response = {answer:personalized[name].evaluate(inputs, output)}
    return JSON.stringify(response);
    //Returns an object so that other parameters can be added in the future
}

function trainPersonalizedAI(inputs, output, name, value) {
    try {
        if (!personalized[name]) {
            personalized[name] = AI.aiFromFile(name + '.h5');
            //generates a new AI automatically if one doesn't exist
        }
        personalized[name].backpropagation(inputs, output, value);
    } catch (error) {
        console.trace(error);
        return 400;
    }
    return 200;
}

console.log("Listening on port 8441 (Accessible at http://localhost:8441/ )");
server.listen(8441);