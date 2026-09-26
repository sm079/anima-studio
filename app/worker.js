// Engine worker: hosts the pipeline so weight loading, command encoding and CPU-side sampler
// work never block the page. Protocol (see engine.js):
//   in : { id, type: "init" | "load" | "generate" | "cancel" | "unload", ... }
//   out: { id, type: "status" | "progress" | "preview" | "result" | "error", ... }

import { AnimaPipeline } from "./pipeline.js";
import { loraFile } from "./store.js";

let pipe = null;
let aborts = new Map();

const gpuInfo = () => {
  const i = pipe?.gpu?.info || {};
  return { name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "), peak: pipe?.gpu?.pool.peak || 0 };
};

async function handle(msg) {
  const { id, type } = msg;
  const reply = (m, transfer = []) => self.postMessage({ id, ...m }, transfer);
  if (type === "init") {
    reply({ type: "result", ok: !!self.navigator.gpu });
    return;
  }
  if (type === "cancel") {
    aborts.get(msg.target)?.abort();
    return;
  }
  if (type === "unload") {
    pipe?.unload();
    reply({ type: "result" });
    return;
  }
  const ac = new AbortController();
  aborts.set(id, ac);
  try {
    if (type === "load") {
      pipe = pipe || new AnimaPipeline();
      await pipe.load(new URL(msg.base), msg.manifest, msg.selection, { signal: ac.signal, token: msg.token, onStatus: (s) => reply({ type: "status", status: s }) });
      reply({ type: "result", loaded: { ...pipe.loaded }, selection: pipe.selection, gpu: gpuInfo() });
    } else if (type === "loras") {
      pipe = pipe || new AnimaPipeline();
      const specs = await Promise.all(msg.list.map(async (l) => ({ key: l.file, strength: l.strength, blob: await loraFile(l.file) })));
      reply({ type: "result", report: await pipe.setLoras(specs) });
    } else if (type === "generate") {
      const res = await pipe.generate({
        ...msg.opts,
        signal: ac.signal,
        onProgress: (p) => reply({ type: "progress", progress: p }),
        onPreview: (img, step) => reply({ type: "preview", step, width: img.width, height: img.height, data: img.data }, [img.data.buffer]),
      });
      const { image } = res;
      reply({ type: "result", width: image.width, height: image.height, data: image.data, timings: res.timings, gpu: gpuInfo() }, [image.data.buffer]);
    }
  } catch (e) {
    reply({ type: "error", name: e.name, message: e.message, stack: e.stack });
  } finally {
    aborts.delete(id);
  }
}

self.onmessage = (e) => { handle(e.data); };
