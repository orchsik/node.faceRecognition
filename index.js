if (process.platform === "linux") {
  // https://github.com/tensorflow/tfjs/tree/master/tfjs-node
  // Windows Requires Python 2.7
  // Mac OS X Requires Python 2.7 & Xcode
  require("@tensorflow/tfjs-node");
}

const fs = require("fs");
const path = require("path");
const canvas = require("canvas");
const faceapi = require("face-api.js");

const MODEL_URL = `${__dirname}/face-models/`;
const IMG_DIR = path.join(__dirname, "img");

const fullFaceDescriptionFor = async ({ filename, label = "" }) => {
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

const getFaceMatcher = async () => {
  const labelInfo = [
    { filename: "진모1.jpg", label: "진모1" },
    { filename: "진모2.jpg", label: "진모2" },
  ];

  const fullFaceDescriptions = await Promise.all(
    labelInfo.map(({ filename, label }) =>
      fullFaceDescriptionFor({ filename, label })
    )
  );
  const labeledFaceDescriptors = fullFaceDescriptions.map((fd) => {
    return new faceapi.LabeledFaceDescriptors(fd.label, [fd.descriptor]);
  });

  const maxDescriptorDistance = 0.4;
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    maxDescriptorDistance
  );
  return faceMatcher;
};

async function run() {
  const faceMatcher = await getFaceMatcher();

  // const filenames = ['T00000001.jpg']
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
