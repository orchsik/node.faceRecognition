const { canvas, faceapi } = require("./env");
const { faceDetectionNet, faceDetectionOptions } = require("./faceDetection");
const { saveFile } = require("./saveFile");

module.exports = {
  canvas,
  faceapi,
  faceDetectionNet,
  faceDetectionOptions,
  saveFile,
};
