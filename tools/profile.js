import { AnimaPipeline, fetchManifest } from "../app/pipeline.js";
import { TorchGenerator } from "../app/rng.js";

const q = new URLSearchParams(location.search);
const base = new URL(q.get("models") || "../models/", location.href);
const size = +(q.get("size") || 1024);
const selection = { model: q.get("model") || "turbo-v1.1", dit: q.get("dit") || "int8", te: q.get("te") || "int8" };
const logEl = document.getElementById("log");
const log = (s) => { logEl.textContent += s + "\n"; console.log(s); };
window.__profileDone = false;

function table(title, rows, wallMs) {
  const total = rows.reduce((a, r) => a + r.ms, 0);
  const max = Math.max(...rows.map((r) => r.ms));
  let html = `<h2>${title}</h2><p class="muted">GPU total ${total.toFixed(1)} ms${wallMs ? `, wall (profiled) ${wallMs.toFixed(0)} ms` : ""}</p>`;
  html += "<table><tr><th>kernel</th><th>calls</th><th>ms</th><th>%</th><th>TFLOP/s</th><th></th></tr>";
  for (const r of rows) {
    const tf = r.flops ? (r.flops / (r.ms / 1000) / 1e12).toFixed(2) : "";
    html += `<tr><td>${r.name}</td><td>${r.count}</td><td>${r.ms.toFixed(1)}</td><td>${((100 * r.ms) / total).toFixed(1)}</td><td>${tf}</td><td><span class="bar" style="width:${(200 * r.ms) / max}px"></span></td></tr>`;
  }
  document.getElementById("out").insertAdjacentHTML("beforeend", html + "</table>");
  return { total, rows: rows.map((r) => ({ ...r, tflops: r.flops ? r.flops / (r.ms / 1000) / 1e12 : null })) };
}

try {
  const manifest = await fetchManifest(base);
  const pipe = new AnimaPipeline({ profile: true });
  log(`loading ${selection.model} dit ${selection.dit} te ${selection.te}`);
  await pipe.load(base, manifest, selection);
  const gpu = pipe.gpu;
  if (!gpu.profiler) throw new Error("timestamp-query unavailable in this browser");
  const h = size / 8, w = size / 8;
  const ctx = await pipe.encode("masterpiece, best quality, 1girl, solo, cherry blossoms");
  const x = new TorchGenerator(1).randn(16 * h * w);

  // warm-up: shader compilation
  let tc = pipe.dit.timeCond(1.0);
  await pipe.dit.forward(x, h, w, ctx, tc);
  tc.release();
  await gpu.sync();
  await gpu.profiler.report();
  gpu.profiler.reset();

  let t0 = performance.now();
  tc = pipe.dit.timeCond(0.9);
  await pipe.dit.forward(x, h, w, ctx, tc);
  tc.release();
  await gpu.sync();
  const dit = table(`DiT forward ${size}×${size}`, await gpu.profiler.report(), performance.now() - t0);
  gpu.profiler.reset();

  let vae = null;
  if (q.get("vae") !== "0") {
    t0 = performance.now();
    await pipe.vae.decode(x, h, w);
    await gpu.sync();
    vae = table(`VAE decode ${size}×${size}`, await gpu.profiler.report(), performance.now() - t0);
  }
  window.__profile = { size, selection, dit, vae };
  log("done");
} catch (e) {
  log("ERROR " + (e.stack || e));
  window.__profileError = String(e.stack || e);
}
window.__profileDone = true;
