const fs = require("fs");
const path = require("path");
const faceapi = require("face-api.js");
const canvas = require("canvas");

const MODEL_URL = `${__dirname}/face-models/`;
const IMG_DIR = path.join(__dirname, "img");

const fullFaceDescriptionFor = async ({ filename, label = "label" }) => {
  const imgBuffer = fs.readFileSync(`${IMG_DIR}/${filename}`);
  // const input = await canvas.loadImage("./img/001.jpg"); // `src :string`으로 주면 한글 못 읽음.
  const img = await canvas.loadImage(imgBuffer); // `src :string`으로 주면 한글 못 읽음.

  // detect the face with the highest score in the image and compute it's landmarks and face descriptor
  const fullFaceDescription = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!fullFaceDescription) {
    throw new Error(`no faces detected for ${filename}`);
  }

  return {
    ...fullFaceDescription,
    filename,
    label,
  };
};

async function run() {
  const label = "T00000001";
  const fullFaceDescription_label = await fullFaceDescriptionFor({
    filename: "진모1.jpg",
    label: "T00000001",
  });
  const labeledFaceDescriptors = new faceapi.LabeledFaceDescriptors("PIVOT", [
    fullFaceDescription_label.descriptor,
  ]);

  const maxDescriptorDistance = 0.4;
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    maxDescriptorDistance
  );

  const filenames = fs.readdirSync(IMG_DIR);
  const fullFaceDescriptions = await Promise.all(
    filenames.map((filename) => fullFaceDescriptionFor({ filename }))
  );

  const result = fullFaceDescriptions.map((fd) => {
    const faceMatch = faceMatcher.findBestMatch(fd.descriptor);
    return {
      label: faceMatch._label,
      distance: faceMatch._distance,
      filename: fd.filename,
    };
  });
  console.log({ result });
}

async function setup() {
  try {
    const { Canvas, Image, ImageData } = canvas;
    faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
    console.log("Setup success\n============================");
  } catch (error) {
    console.error("setup", error);
  }
}
setup().then(() => {
  run();
});

// Hi there 👋. Looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, which binds to TensorFlow C++, by running npm i
// @tensorflow/tfjs-node, or npm i @tensorflow/tfjs-node-gpu if you have CUDA. Then call require('@tensorflow/tfjs-node'); (-gpu suffix for CUDA) at the start of your program.
// Visit https://github.com/tensorflow/tfjs-node for more details.
