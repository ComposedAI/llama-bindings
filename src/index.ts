import path from "path";

const {
  llamaBackendInit,
  llamaLoadModelFromFile,
  llamaModelDesc,
  llamaModelDefaultParams,
  llamaContextDefaultParams,
  llamaNewContextWithModel,
  llamaTokenize,
  llamaDecode,
  LLAMAContext,
}: typeof import("llama-bindings") = require("bindings")("llama-bindings");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

const context = new LLAMAContext(modelPath);

// init LLM backend
// llamaBackendInit(false);

// initialize the model
// const params = llamaModelDefaultParams();
// const model = llamaLoadModelFromFile(modelPath, params);

// if (!model) {
//   console.error(`error: unable to load model`);
//   process.exit(1);
// }

// const modelDesc = llamaModelDesc(model);
// console.dir(`model description: ${modelDesc}`);

// initialize the context
// const ctx_params = llamaContextDefaultParams();

// const ctx = llamaNewContextWithModel(model, ctx_params);
// const tokens = llamaTokenize(ctx, "hello world", true);
// const decoded = llamaDecode(ctx, tokens);

console.dir(context, { depth: null });
