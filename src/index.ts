import path from "path";

let addon = require("bindings")("llama-addon");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

const systemInfo = addon.getSystemInfo();
const model = new addon.LLAMAModel(modelPath);

console.dir(systemInfo, { depth: null });
console.dir(model, { depth: null });
