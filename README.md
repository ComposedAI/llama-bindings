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
const llama = require("llama-bindings");

// Load the model
const model = llama.loadModel("path/to/model");

// Run the model
const result = model.run("This is a test");
// Print the result
```
