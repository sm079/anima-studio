// LoRAs from Hugging Face. Hugging Face serves repo files and its API with CORS headers (and
// supports Range requests), so the page downloads directly: no helper server.
//
// Input is a direct link to a .safetensors file:
//   https://huggingface.co/<owner>/<repo>/blob/<rev>/<path>.safetensors  (the file's page)
//   https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>.safetensors  (the download link)
// An access token (settings) is sent only to huggingface.co, for gated or private repos.

import { inspectLoraKeys } from "./lora.js";

const HOST = "https://huggingface.co";

// -> { repo, rev, path } or null
export function parseHfUrl(text) {
  let u;
  try { u = new URL(String(text || "").trim()); } catch { return null; }
  if (!/^(www\.)?(huggingface\.co|hf\.co)$/.test(u.hostname)) return null;
  const m = u.pathname.match(/^\/([^/]+)\/([^/]+)\/(?:blob|resolve)\/([^/]+)\/(.+\.safetensors)$/i);
  if (!m) return null;
  return { repo: `${m[1]}/${m[2]}`, rev: decodeURIComponent(m[3]), path: m[4].split("/").map(decodeURIComponent).join("/") };
}

const enc = (p) => p.split("/").map(encodeURIComponent).join("/");
const fileUrl = (ref) => `${HOST}/${ref.repo}/resolve/${encodeURIComponent(ref.rev)}/${enc(ref.path)}`;
const auth = (token) => (token ? { Authorization: `Bearer ${token}` } : {});

function httpError(status, what) {
  if (status === 401 || status === 403) {
    // Hugging Face answers 401 for missing repos too, so this can't tell "private" from "doesn't exist"
    return Object.assign(new Error(`This ${what} is gated, private or doesn't exist.`), { needsKey: true });
  }
  if (status === 404) return new Error(`That ${what} doesn't exist on Hugging Face.`);
  return new Error(`Hugging Face returned ${status}.`);
}

// Reads just the safetensors header with Range requests: file size and tensor names.
async function readHeader(url, token, signal) {
  const get = async (from, to) => {
    const res = await fetch(url, { headers: { ...auth(token), Range: `bytes=${from}-${to}` }, signal, cache: "no-store" });
    if (!res.ok) throw httpError(res.status, "file");
    const total = +(res.headers.get("content-range") || "").split("/")[1] || 0;
    const buf = new Uint8Array(await res.arrayBuffer());
    if (res.status !== 206) return { buf: buf.subarray(from, to + 1), total: buf.byteLength };
    return { buf, total };
  };
  const first = await get(0, 256 * 1024 - 1);
  const n = Number(new DataView(first.buf.buffer, first.buf.byteOffset, 8).getBigUint64(0, true));
  if (!n || n > 64 * 2 ** 20) throw new Error("That file isn't a safetensors file.");
  const bytes = 8 + n <= first.buf.byteLength ? first.buf.subarray(8, 8 + n) : (await get(8, 8 + n - 1)).buf;
  let header;
  try { header = JSON.parse(new TextDecoder().decode(bytes)); } catch { throw new Error("That file isn't a safetensors file."); }
  delete header.__metadata__;
  return { size: first.total, keys: Object.keys(header) };
}

// Repo card: trigger words and a preview image. Optional: failures are ignored.
async function readCard(ref, token, signal) {
  try {
    const res = await fetch(`${HOST}/api/models/${ref.repo}`, { headers: auth(token), signal });
    if (!res.ok) return {};
    const j = await res.json();
    const card = j.cardData || {};
    const words = String(card.instance_prompt || "").split(",").map((w) => w.trim()).filter(Boolean);
    let image = null;
    const w = (card.widget || []).find((x) => x?.output?.url);
    const imgPath = w?.output?.url || j.siblings?.map((s) => s.rfilename).find((f) => /\.(png|jpe?g|webp)$/i.test(f));
    if (imgPath && !j.gated && !j.private) image = /^https?:/.test(imgPath) ? imgPath : `${HOST}/${ref.repo}/resolve/${encodeURIComponent(ref.rev)}/${enc(imgPath)}`;
    return { words, image };
  } catch (e) {
    if (e.name === "AbortError") throw e;
    return {};
  }
}

// -> { id, repo, rev, path, name, fileName, size, words, image, pageUrl, url, check, madeFor }
// check: { lora, modules, anima, unsupported } from the header; madeFor: "anima" | "other" | ""
export async function fetchLoraInfo(ref, token, signal) {
  const url = fileUrl(ref);
  const [head, card] = await Promise.all([readHeader(url, token, signal), readCard(ref, token, signal)]);
  const check = inspectLoraKeys(head.keys);
  const fileName = ref.path.split("/").pop();
  const repoName = ref.repo.split("/")[1];
  return {
    id: `hf:${ref.repo}@${ref.rev}/${ref.path}`,
    ...ref,
    name: repoName.replace(/[-_]+/g, " "),
    fileName,
    size: head.size,
    words: card.words || [],
    image: card.image || null,
    pageUrl: `${HOST}/${ref.repo}/blob/${encodeURIComponent(ref.rev)}/${enc(ref.path)}`,
    url,
    check,
    madeFor: check.anima > 0 ? "anima" : check.lora ? "other" : "",
  };
}

// Resolves to a Response with a readable body.
export async function openDownload(info, token, signal) {
  const res = await fetch(info.url, { headers: auth(token), signal, cache: "no-store" });
  if (!res.ok) throw httpError(res.status, "file");
  return res;
}

// -> { ok: true, name } for a working token, { ok: false } for a rejected one; throws on network errors
export async function checkToken(token) {
  const res = await fetch(`${HOST}/api/whoami-v2`, { headers: auth(token) });
  if (res.status === 401) return { ok: false };
  if (!res.ok) throw new Error(`Hugging Face returned ${res.status}.`);
  const j = await res.json();
  return { ok: true, name: j.name };
}
