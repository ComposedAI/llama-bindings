# llama-bindings

Node bindings for llama.cpp

The llama-bindings module is a wrapper around the llama.cpp library. It provides a simple interface for loading and running language models. The module is written in C++ and uses the [N-API](https://nodejs.org/api/n-api.html) interface to Node.js. It has minimal dependencies and should work on most platforms.

Much of the inspiration for this module comes from the [node-llama-cpp](https://github.com/withcatai/node-llama-cpp.git). The major difference is that this module has minimal dependencies and exposes the llama.cpp API directly.

## Installation

```bash
npm install llama-bindings
```

## Usage

```javascript
let addon = require("bindings")("llama-addon");

// Load the model
const model = llama.loadModel("path/to/model");

// Run the model
const result = model.run("This is a test");
// Print the result
```

## References and Acknowledgements

### References

- [Node API](https://nodejs.github.io/node-addon-examples/)
- [Simple Guide to Node.js C++ Addons](https://medium.com/jspoint/a-simple-guide-to-load-c-c-code-into-node-js-javascript-applications-3fcccf54fd32)
- [LLM to APIs](https://gorilla.cs.berkeley.edu/index.html)
- [LLM for Function Calling](https://github.com/nexusflowai/NexusRaven-V2)
- [Gorrila Open Functions](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v1)

### Acknowledgements

- [node-llama-cpp](https://github.com/withcatai/node-llama-cpp.git)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ollama](https://github.com/jmorganca/ollama)
