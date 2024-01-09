import path from "path";

const {
  getSystemInfo,
  loadModelFromFile,
  getModelDesc,
}: typeof import("llama-bindings") = require("bindings")("llama-bindings");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

const systemInfo = getSystemInfo();
const model = loadModelFromFile(modelPath, {});
const modelDesc = getModelDesc(model);

console.dir(systemInfo, { depth: null });
console.dir(`model description: ${modelDesc}`);
