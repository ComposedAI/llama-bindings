import path from "path";
import { stdout } from "process";

type LLAMABindings = typeof import("llama-bindings");

const { LLAMAContext }: LLAMABindings = require("bindings")("llama-bindings");

// const modelPath = path.resolve("models/mistral-7b-instruct-v0.2.Q4_0.gguf");
const modelPath = path.resolve("models/llama-2-7b-chat.Q4_0.gguf");

const ctx = new LLAMAContext(modelPath);

const tokens = ctx.encode("hello world");
const decoded = ctx.decode(tokens);

console.log(decoded);

async function* evalGenerator(
  ctx: InstanceType<typeof LLAMAContext>,
  tokens: Uint32Array
) {
  let evalTokens = tokens;
  const tokenEos = ctx.tokenEos();

  while (true) {
    // Evaluate to get the next token.
    const nextToken = await ctx.evaluate(evalTokens);

    // the assistant finished answering
    if (nextToken === tokenEos) break;

    yield nextToken;

    // Create tokens for the next eval.
    evalTokens = Uint32Array.from([nextToken]);
  }
}

const prompt = "Once upon a time";

// Evaluate the prompt.
async function main() {
  const evalIterator = evalGenerator(ctx, ctx.encode(prompt));

  for await (const chunk of evalIterator) {
    const text = ctx.decode(Uint32Array.from([chunk]));
    stdout.write(text);
  }
}
main();
