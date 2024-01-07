var addon = require("bindings")("addon");

const helloAddon = new addon.HelloAddon();
console.log(helloAddon.hello()); // 'world'
