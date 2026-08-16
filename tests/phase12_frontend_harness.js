"use strict";

const fs = require("node:fs");
const vm = require("node:vm");

class FakeClassList {
  constructor(initial = []) {
    this.values = new Set(initial);
  }

  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  contains(value) { return this.values.has(value); }
  toggle(value, force) {
    if (force === undefined ? !this.values.has(value) : force) this.values.add(value);
    else this.values.delete(value);
  }
}

class FakeNode {
  constructor(id = "") {
    this.id = id;
    this.children = [];
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.listeners = new Map();
    this.classList = new FakeClassList(id === "replay" ? ["hidden"] : []);
    this._textContent = "";
    this.className = "";
    this.type = "";
    this.value = "900";
  }

  get firstChild() { return this.children[0] || null; }
  get textContent() { return this._textContent + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
  }
  append(...children) { this.children.push(...children); }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  dispatch(name) {
    const callback = this.listeners.get(name);
    if (callback) callback();
  }
}

const ids = [
  "replay-file", "empty-state", "replay", "run-status", "run-id", "run-facts",
  "step-list", "previous", "play", "next", "speed", "frame-count", "frame-kind",
  "frame-title", "logical-time", "frame-description", "frame-content",
  "retrieval-panel", "memory-panel", "overlay-panel", "empty-panel-template",
];
const nodes = new Map(ids.map((id) => [id, new FakeNode(id)]));

const document = {
  getElementById(id) { return nodes.get(id); },
  createElement() { return new FakeNode(); },
  querySelectorAll() { return []; },
};

class FakeFileReader {
  constructor() {
    this.listeners = new Map();
    this.result = null;
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  readAsText(file) {
    this.result = file.content;
    const callback = this.listeners.get("load");
    if (callback) callback();
  }
}

const window = {
  setInterval() { return 1; },
  clearInterval() {},
};

const context = vm.createContext({
  console,
  document,
  FileReader: FakeFileReader,
  window,
  setInterval: window.setInterval,
  clearInterval: window.clearInterval,
});
vm.runInContext(fs.readFileSync("visualizer/app.js", "utf8"), context, {
  filename: "visualizer/app.js",
});

const content = fs.readFileSync(0, "utf8");
const fileNode = nodes.get("replay-file");
fileNode.files = [{ content, size: Buffer.byteLength(content, "utf8") }];
fileNode.dispatch("change");

const allText = [...nodes.values()].map((node) => node.textContent).join("\n");
process.stdout.write(JSON.stringify({
  rendered: !nodes.get("replay").classList.contains("hidden"),
  error_text: nodes.get("empty-state").textContent,
  run_id: nodes.get("run-id").textContent,
  all_text: allText,
}));
