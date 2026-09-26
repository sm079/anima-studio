// One-time model download into the Origin Private File System (OPFS).
//
// Files are written to "<name>.part" in place through a sync access handle (download.js), so
// an interrupted multi-GB download resumes with an HTTP Range request. Inside a worker (the
// engine worker) that runs inline; from a page it runs in download-worker.js. A finished file
// is renamed to its final name and reused on later visits.

import { downloadToOPFS, OPFS_DIR as DIR } from "./download.js";

const IN_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;

async function dir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

async function tryFile(d, name) {
  try {
    return await (await d.getFileHandle(name)).getFile();
  } catch {
    return null;
  }
}

export async function requestPersistence() {
  try {
    return await navigator.storage.persist?.();
  } catch {
    return false;
  }
}

function downloadInWorker(url, part, size, headers, onProgress, signal) {
  if (IN_WORKER) return downloadToOPFS({ url, part, size, headers, signal, onProgress: (done) => onProgress?.(done, size) });
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./download-worker.js", import.meta.url), { type: "module" });
    const stop = () => { w.terminate(); reject(new DOMException("Download cancelled", "AbortError")); };
    signal?.addEventListener("abort", stop, { once: true });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") onProgress?.(m.done, size);
      else {
        signal?.removeEventListener("abort", stop);
        w.terminate();
        if (m.type === "done") resolve();
        else reject(new Error(m.message));
      }
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "download worker failed")); };
    w.postMessage({ url, part, size, headers });
  });
}

// Fallback without sync access handles: one streaming pass through a writable.
async function downloadOnMainThread(d, url, part, size, headers, onProgress, signal) {
  const ph = await d.getFileHandle(part, { create: true });
  const res = await fetch(url, { headers, signal, cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  const w = await ph.createWritable();
  let have = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    await w.write(value);
    have += value.byteLength;
    onProgress?.(have, size);
  }
  await w.close();
}

// Returns a File for `name`, downloading it from `url` first if needed.
// onProgress(bytesDone, bytesTotal); headers go with every download request (e.g. a token)
export async function cachedFile(url, name, size, onProgress, signal, headers = {}) {
  const d = await dir();
  const done = await tryFile(d, name);
  if (done && done.size === size) {
    onProgress?.(size, size);
    return done;
  }
  if (done) await d.removeEntry(name);

  const part = name + ".part";
  const partFile = await tryFile(d, part);
  if (!(partFile && partFile.size === size)) {
    const workerOK = typeof FileSystemSyncAccessHandle !== "undefined" || "createSyncAccessHandle" in (globalThis.FileSystemFileHandle?.prototype || {});
    if (workerOK) await downloadInWorker(url, part, size, headers, onProgress, signal);
    else await downloadOnMainThread(d, url, part, size, headers, onProgress, signal);
  }

  const ph = await d.getFileHandle(part);
  const got = (await ph.getFile()).size;
  if (got !== size) throw new Error(`size mismatch for ${name}: got ${got}, expected ${size}`);
  if (ph.move) {
    await ph.move(name);
    return (await d.getFileHandle(name)).getFile();
  }
  return ph.getFile(); // no rename support: the complete .part is used as is
}

export async function listCached() {
  const d = await dir();
  const out = [];
  for await (const [name, h] of d.entries()) {
    if (h.kind !== "file") continue;
    const size = (await h.getFile()).size;
    out.push({ name: name.replace(/\.part$/, ""), size, partial: name.endsWith(".part") });
  }
  return out;
}

export async function clearCache() {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(DIR, { recursive: true });
}

// ------------------------------------------------------------------ LoRA files (separate folder)

const LORA_DIR = "anima-loras";

async function loraDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(LORA_DIR, { create: true });
}

// source: Blob/File, or a Response whose body is streamed to disk. onProgress(bytes)
export async function saveLoraFile(name, source, onProgress) {
  const d = await loraDir();
  const h = await d.getFileHandle(name, { create: true });
  const w = await h.createWritable();
  try {
    if (source instanceof Blob) {
      await w.write(source);
    } else {
      const reader = source.body.getReader();
      let done = 0;
      for (;;) {
        const { done: end, value } = await reader.read();
        if (end) break;
        await w.write(value);
        done += value.byteLength;
        onProgress?.(done);
      }
    }
    await w.close();
  } catch (e) {
    try { await w.abort(); } catch { /* ignore */ }
    try { await d.removeEntry(name); } catch { /* ignore */ }
    throw e;
  }
  return h.getFile();
}

export async function loraFile(name) {
  const d = await loraDir();
  return (await d.getFileHandle(name)).getFile();
}

export async function removeLoraFile(name) {
  const d = await loraDir();
  try { await d.removeEntry(name); } catch { /* already gone */ }
}

export async function loraStorageBytes() {
  const d = await loraDir();
  let n = 0;
  for await (const [, h] of d.entries()) if (h.kind === "file") n += (await h.getFile()).size;
  return n;
}

export async function clearLoras() {
  const root = await navigator.storage.getDirectory();
  try { await root.removeEntry(LORA_DIR, { recursive: true }); } catch { /* none */ }
}
