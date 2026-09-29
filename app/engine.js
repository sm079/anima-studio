// Page-side handle on the engine. Runs the pipeline in a Web Worker (worker.js) so the UI stays
// responsive; falls back to running it in the page when WebGPU isn't exposed to workers.
// Same shape as the pipeline methods main.js needs.

import { AnimaPipeline, resolveFiles } from "./pipeline.js";
import { requestPersistence, loraFile } from "./store.js";

class WorkerEngine {
  constructor(worker) {
    this.worker = worker;
    this.next = 1;
    this.calls = new Map();
    this.loaded = {};
    this.selection = null;
    this.gpu = { name: "", peak: 0 };
    worker.onmessage = (e) => {
      const m = e.data;
      const call = this.calls.get(m.id);
      if (!call) return;
      if (m.type === "status") call.onStatus?.(m.status);
      else if (m.type === "progress") call.onProgress?.(m.progress);
      else if (m.type === "preview") call.onPreview?.({ width: m.width, height: m.height, data: m.data }, m.step);
      else {
        this.calls.delete(m.id);
        if (m.type === "error") {
          const err = new Error(m.message);
          err.name = m.name;
          err.stack = m.stack;
          call.reject(err);
        } else call.resolve(m);
      }
    };
  }

  call(msg, hooks = {}) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.calls.set(id, { resolve, reject, ...hooks });
      if (hooks.signal) {
        if (hooks.signal.aborted) this.worker.postMessage({ type: "cancel", target: id });
        hooks.signal.addEventListener("abort", () => this.worker.postMessage({ type: "cancel", target: id }), { once: true });
      }
      this.worker.postMessage({ id, ...msg });
    });
  }

  isLoaded(files, backend = "webgpu") {
    return ["dit", "te", "vae"].every((k) => this.loaded[k] === (k === "dit" && backend === "webnn" ? files.dit.path + "#webnn" : files[k].path));
  }

  get model() {
    return this._model || null;
  }

  async load(base, manifest, selection, { onStatus, signal, token } = {}) {
    const r = await this.call({ type: "load", base: base.href, manifest, selection, token }, { onStatus, signal });
    this.loaded = r.loaded;
    this.selection = r.selection;
    this._model = resolveFiles(manifest, selection).model;
    this.gpu = r.gpu;
  }

  // list: [{ file (name in the OPFS LoRA folder), strength }] -> per-file report
  async setLoras(list) {
    return (await this.call({ type: "loras", list })).report;
  }

  async generate({ onProgress, onPreview, signal, ...opts }) {
    const r = await this.call({ type: "generate", opts }, { onProgress, onPreview, signal });
    this.gpu = r.gpu;
    return { image: { width: r.width, height: r.height, data: r.data }, timings: r.timings };
  }

  unload() {
    this.loaded = {};
    this._model = null;
    return this.call({ type: "unload" });
  }
}

class LocalEngine {
  constructor() {
    this.pipe = new AnimaPipeline();
  }
  isLoaded(files, backend) { return this.pipe.isLoaded(files, backend); }
  get model() { return this.pipe.model; }
  get selection() { return this.pipe.selection; }
  get loaded() { return this.pipe.loaded; }
  get gpu() {
    const i = this.pipe.gpu?.info || {};
    return { name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "), peak: this.pipe.gpu?.pool.peak || 0 };
  }
  load(base, manifest, selection, opts) { return this.pipe.load(base, manifest, selection, opts); }
  async setLoras(list) {
    const specs = await Promise.all(list.map(async (l) => ({ key: l.file, strength: l.strength, blob: await loraFile(l.file) })));
    return this.pipe.setLoras(specs);
  }
  generate(opts) { return this.pipe.generate(opts); }
  unload() { this.pipe.unload(); }
}

// inPage: force the in-page engine (debugging / comparison; ?engine=page)
export async function createEngine({ inPage = false } = {}) {
  requestPersistence(); // Window-only API; the worker can't ask for it
  if (inPage) return new LocalEngine();
  try {
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const engine = new WorkerEngine(worker);
    const { ok } = await Promise.race([
      engine.call({ type: "init" }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("worker did not start")), 10000)),
    ]);
    if (ok) {
      engine.inWorker = true;
      return engine;
    }
    worker.terminate();
    console.warn("WebGPU is not available in workers here; running the engine on the page");
  } catch (e) {
    console.warn("engine worker unavailable, running on the page:", e.message);
  }
  return new LocalEngine();
}
