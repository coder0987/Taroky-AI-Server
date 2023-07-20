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
let personalized;

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
            res.write(personalizedAI(postData, req.headers.output, req.headers.id, q.pathname.split('/')[2]));
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

function personalizedAI(inputs, output, id, name) {
    //Again, inputs are several-thousand long. This time, however, deep learning is used to retrieve the answer
    const response = {answer:players[name].evaluate(inputs, output)}
    return JSON.stringify(response);
    //Returns an object so that other parameters can be added in the future
}

console.log("Listening on port 8441 (Accessible at http://localhost:8441/ )");
server.listen(8441);