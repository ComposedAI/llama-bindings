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
}: typeof import("llama-bindings") = require("bindings")("llama-bindings");

const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");

// init LLM backend
llamaBackendInit(false);

// initialize the model
const params = llamaModelDefaultParams();
const model = llamaLoadModelFromFile(modelPath, params);

if (!model) {
  console.error(`error: unable to load model`);
  process.exit(1);
}

const modelDesc = llamaModelDesc(model);
console.dir(`model description: ${modelDesc}`);

// initialize the context
const ctx_params = llamaContextDefaultParams();

// ctx_params.seed = 1234;
// ctx_params.n_ctx = 2048;
// ctx_params.n_threads = params.n_threads;
// ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

const ctx = llamaNewContextWithModel(model, ctx_params);
const tokens = llamaTokenize(ctx, "hello world", true);
const decoded = llamaDecode(ctx, tokens);

console.dir(decoded);
