// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
// require("@tensorflow/tfjs-node");
// var version = process.versions.napi;
// console.log(version);

const faceapi = require("face-api.js");

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
const canvas = require("canvas");

// const MODEL_URL = `${__dirname}/face-models/`;
// async function setup() {
//   faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
// }

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

module.exports = { canvas, faceapi };
