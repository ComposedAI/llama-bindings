import path from "path";

type LLAMABindings = typeof import("llama-bindings");

const { LLAMAContext }: LLAMABindings = require("bindings")("llama-bindings");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

const ctx = new LLAMAContext(modelPath);

const tokens = ctx.encode("hello world");
const decoded = ctx.decode(tokens);

console.log(decoded);

const results = ctx.evaluate("how are you doing?");
console.log(results);
