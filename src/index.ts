import path from "path";

var addon = require("bindings")("llama-addon");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

const model = new addon.LLAMAModel(modelPath);
console.dir(model, { depth: null });
