import { fetchManifest, resolveFiles } from "./pipeline.js";
import { createEngine } from "./engine.js";
import { listCached, clearCache, saveLoraFile, removeLoraFile, loraStorageBytes, clearLoras } from "./store.js";
import { parseHfUrl, fetchLoraInfo, openDownload, checkToken } from "./hf.js";
import { SafeTensors } from "./weights.js";
import { hasWebNN, probeWebNN, WEBNN_FLAG } from "./webnn/support.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Model files live on Hugging Face, pinned to one commit: browsers cache files by name, so a new
// upload only reaches visitors when this points at the new commit. ?models=./models/ uses a local build.
const MODELS_URL = "https://huggingface.co/sm079/anima-studio/resolve/1abd3f6530cdb5c497e9d3e740458400b6725644/";
const BASE = new URL(params.get("models") || MODELS_URL, location.href);
if (!BASE.pathname.endsWith("/")) BASE.pathname += "/";

// ------------------------------------------------------------------ choices (plain language)

// Widest to tallest. Medium sizes are the ~1 megapixel buckets the model was trained on.
const SHAPES = [
  { id: "wide", label: "Wide 3:2", w: 1216, h: 832 },
  { id: "landscape", label: "Landscape 4:3", w: 1152, h: 896 },
  { id: "square", label: "Square 1:1", w: 1024, h: 1024 },
  { id: "portrait", label: "Portrait 3:4", w: 896, h: 1152 },
  { id: "tall", label: "Tall 2:3", w: 832, h: 1216 },
];
const SIZES = [
  { id: "small", label: "Small", scale: 0.75, tip: "About 0.6 megapixels, fastest" },
  { id: "medium", label: "Medium", scale: 1, tip: "About 1 megapixel, what the model was trained on" },
  { id: "large", label: "Large", scale: 1.25, tip: "About 1.5 megapixels, slowest" },
];
const DETAIL = [
  { id: "fast", label: "Fast", steps: 8, tip: "Quickest (8 steps)" },
  { id: "better", label: "Better", steps: 10, tip: "A bit more refined (10 steps)" },
  { id: "best", label: "Best", steps: 12, tip: "Most refined, slowest (12 steps)" },
];
// Sampler/scheduler pairs that suit the Turbo model at 8-12 steps.
const STYLES = [
  { id: "crisp", label: "Crisp", sampler: "er_sde", scheduler: "beta", tip: "Clean lines and flat colors" },
  { id: "soft", label: "Soft", sampler: "euler_a", scheduler: "simple", tip: "Softer, thinner lines" },
  { id: "plain", label: "Plain", sampler: "euler", scheduler: "simple", tip: "Neutral, straightforward look" },
];
const SAMPLER_INFO = {
  er_sde: ["ER-SDE", "Sharp lines, flat colors (recommended)"],
  euler_a: ["Euler A", "Softer lines, a little random"],
  euler: ["Euler", "Neutral and stable"],
  dpmpp_2m: ["DPM++ 2M", "Smooth, detailed"],
};
const SCHED_INFO = {
  beta: ["Beta", "More care at the start and end (recommended)"],
  simple: ["Simple", "Evenly spaced steps"],
};
const PRESETS = [
  { id: "best", label: "Best quality", dit: "bf16", te: "bf16", desc: "The original model." },
  { id: "balanced", label: "Balanced", dit: "int8", te: "int8", desc: "Looks nearly identical to Best.", tag: "Recommended" },
  { id: "compact", label: "Compact", dit: "w4a8", te: "w4a8", desc: "Smallest. May read prompts a little differently." },
];
const PART_LABELS = { bf16: "Original", int8: "Compressed", w4a8: "Extra compressed" };
const BOOST = "masterpiece, best quality, score_7, safe, ";
const EXAMPLES = [
  "1girl, solo, silver hair, long hair, blue eyes, school uniform, cherry blossoms, petals, smile, looking at viewer, upper body",
  "1boy, knight, silver armor, holding sword, castle ruins, sunset, dramatic lighting, wind, flowing cape, wide shot",
  "no humans, scenery, cozy cafe interior, rain on window, warm lighting, plants, bookshelf, cat sleeping on a chair",
  "1girl, witch hat, purple cape, riding a broom, night sky, full moon, city lights below, stars, from side",
  "chibi, 2girls, eating ice cream, summer, beach, bright colors, simple background, happy",
  "a quiet mountain village in autumn, red maple leaves, stone path, paper lanterns, morning mist",
];
const ICONS = {
  dl: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  reuse: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>',
  ext: '<svg viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  words: '<svg viewBox="0 0 24 24"><path d="M4 7V5h16v2M9 19h6M12 5v14"/></svg>',
};

// ------------------------------------------------------------------ state

const store = {
  get(k, d) { try { const v = localStorage.getItem("anima-studio." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("anima-studio." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const ui = {
  w: 1024, h: 1024, steps: 8, sampler: "er_sde", scheduler: "beta", shift: 3,
  ...store.get("ui", {}),
};
let loras = store.get("loras", []); // { id, name, version, image, page, words, file, size, strength, on }
let manifest = null;
let pipe = null;
let sel = store.get("selection", null); // { model, dit, te }
// Engine for the image model: "webgpu" (default) or "webnn" (optional, experimental; ?backend=webnn)
let backend = params.get("backend") || store.get("backend", "webgpu");
let phase = "starting"; // starting | welcome | loading | ready | generating (queue running)
let loadAbort = null;
let exampleIdx = 0;
let current = null; // the finished image being viewed
let live = false; // viewer follows the image being made
let pinned = false; // user picked an older image during this queue run; don't jump to new ones
let galleryOpen = false;
const loraReport = new Map(); // file -> { matched, total, unsupported }
let appliedKey = null; // LoRA set currently applied in the engine
const gallery = [];
// queue
const jobs = []; // waiting
let running = null; // { ...job, abort, frac, thumb, hasPreview }

const saveUi = () => store.set("ui", ui);
const saveLoras = () => store.set("loras", loras.filter((l) => !l.pending).map(({ pending, progress, ...l }) => l));
const fmtGB = (n) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(n >= 100 * 2 ** 20 ? 0 : 1)} MB` : n > 0 ? `${Math.max(1, Math.round(n / 1024))} KB` : "0 KB");
const fmtTime = (ms) => {
  const s = Math.max(1, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60 ? `${s % 60} s` : ""}`.trim() : `${s} s`;
};
// Download speed averaged over the last 20 s, so the time-left estimate doesn't swing with bursty progress.
function rateMeter(windowMs = 20000) {
  const samples = [];
  return (done) => {
    const now = performance.now();
    samples.push([now, done]);
    while (samples.length > 2 && now - samples[0][0] > windowMs) samples.shift();
    const [t0, d0] = samples[0];
    return now - t0 > 1500 ? ((done - d0) / (now - t0)) * 1000 : 0;
  };
}
const round16 = (v) => Math.min(1536, Math.max(256, Math.round(v / 16) * 16));
const newSeed = () => Math.floor(Math.random() * 2 ** 32);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pressed = (id) => $(id).getAttribute("aria-pressed") === "true";

// ------------------------------------------------------------------ small UI helpers

function setStatus(text, kind = "idle") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot " + ({ ok: "ok", busy: "busy", err: "err" }[kind] || "");
}

function showPanel(which) {
  for (const id of ["welcome", "loading", "empty"]) $(id).hidden = id !== which;
  $("canvas").hidden = which !== "canvas";
  $("stageFoot").hidden = which === "welcome" || which === "loading";
  if (which === "welcome" || which === "loading") openGallery(false);
}

function showError(msg) {
  const el = $("errorCard");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => { el.hidden = true; }, 9000);
}

function friendlyError(e) {
  const m = String(e?.message || e);
  if (/HTTP 401/.test(m) && store.get("hfToken", "")) return "Hugging Face rejected your token. Fix or remove it in settings, then try again.";
  if (/device lost|out of memory|OOM|allocation/i.test(m)) return "Your graphics card ran out of memory. Try a smaller canvas, fewer LoRAs, or the Compact download in settings.";
  if (/HTTP|fetch|network|Failed to fetch/i.test(m)) return "The download was interrupted. Check your connection and try again.";
  if (/quota|storage|space/i.test(m)) return "Not enough storage space in this browser.";
  if (backend === "webnn") return `WebNN error: ${m}. The WebNN engine is experimental: switch back to WebGPU in settings if this keeps happening.`;
  return "Something went wrong: " + m;
}

function radioGroup(el, items, isOn, onPick, render) {
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(isOn(it)));
    if (it.tip) b.title = it.tip;
    if (render) render(b, it); else b.textContent = it.label;
    b.onclick = () => onPick(it);
    el.append(b);
  }
}

function showView(which) {
  const [show, hide] = which === "loras" ? ["loraView", "mainView"] : ["mainView", "loraView"];
  $(hide).hidden = true;
  $(show).hidden = false;
  $(show).classList.remove("enter");
  void $(show).offsetWidth;
  $(show).classList.add("enter");
  closePopover();
  if (which === "loras") $("loraUrl").focus();
}

// ------------------------------------------------------------------ canvas & sampling controls

const shapeOf = () => SHAPES.find((s) => SIZES.some((z) => round16(s.w * z.scale) === ui.w && round16(s.h * z.scale) === ui.h));
const sizeOf = () => SIZES.find((z) => SHAPES.some((s) => round16(s.w * z.scale) === ui.w && round16(s.h * z.scale) === ui.h));

function setDims(w, h) {
  ui.w = round16(w);
  ui.h = round16(h);
  saveUi();
  renderControls();
}

function renderControls() {
  const shape = shapeOf();
  const size = sizeOf();
  radioGroup($("shapes"), SHAPES, (s) => s === shape, (s) => {
    const z = size || SIZES[1];
    setDims(s.w * z.scale, s.h * z.scale);
  }, (b, s) => {
    b.className = "shape";
    b.title = s.label;
    b.setAttribute("aria-label", s.label);
    const icon = document.createElement("i");
    const r = s.w / s.h;
    icon.style.width = `${Math.round(r >= 1 ? 18 : 18 * r)}px`;
    icon.style.height = `${Math.round(r >= 1 ? 18 / r : 18)}px`;
    b.append(icon);
  });
  radioGroup($("sizes"), SIZES, (z) => z === size, (z) => {
    const s = shape || SHAPES[2];
    setDims(s.w * z.scale, s.h * z.scale);
  });
  $("wRange").value = ui.w;
  $("hRange").value = ui.h;
  $("wOut").textContent = ui.w;
  $("hOut").textContent = ui.h;
  const mp = (ui.w * ui.h) / 1e6;
  $("sizeHint").textContent = `${shape ? shape.label.split(" ")[1] : "custom"} · ${mp.toFixed(1)} MP${mp > 1.8 ? " · very large" : ""}`;

  radioGroup($("detail"), DETAIL, (d) => d.steps === ui.steps, (d) => { ui.steps = d.steps; saveUi(); renderControls(); });
  radioGroup($("style"), STYLES, (s) => s.sampler === ui.sampler && s.scheduler === ui.scheduler, (s) => {
    ui.sampler = s.sampler;
    ui.scheduler = s.scheduler;
    saveUi();
    renderControls();
  });
  renderChips();
  renderGo();
}

function renderChips() {
  const el = $("advChips");
  const openKey = $("advPop").hidden ? null : $("advPop").dataset.key;
  el.innerHTML = "";
  const chips = [
    ["steps", `<b>${ui.steps}</b> steps`],
    ["sampler", `<b>${SAMPLER_INFO[ui.sampler][0]}</b>`],
    ["scheduler", `<b>${SCHED_INFO[ui.scheduler][0]}</b> schedule`],
    ["shift", `shift <b>${ui.shift}</b>`],
  ];
  for (const [key, html] of chips) {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = html;
    b.title = "Change";
    b.dataset.key = key;
    b.setAttribute("aria-expanded", String(key === openKey));
    b.onclick = (e) => { e.stopPropagation(); openPopover(key, b); };
    el.append(b);
  }
}

function openPopover(key, anchor) {
  const pop = $("advPop");
  if (!pop.hidden && pop.dataset.key === key) { closePopover(); return; }
  pop.dataset.key = key;
  pop.innerHTML = "";
  const title = document.createElement("span");
  title.className = "label";
  const note = document.createElement("p");
  note.className = "help";
  const range = (min, max, step, value, onInput) => {
    const row = document.createElement("div");
    row.className = "row";
    const r = document.createElement("input");
    Object.assign(r, { type: "range", min, max, step, value });
    const o = document.createElement("output");
    o.textContent = value;
    r.oninput = () => { o.textContent = r.value; onInput(+r.value); };
    row.append(r, o);
    return row;
  };
  const options = (info, value, onPick) => {
    const box = document.createElement("div");
    box.className = "opts";
    for (const [id, [label, desc]] of Object.entries(info)) {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(id === value));
      b.innerHTML = `${esc(label)}<small>${esc(desc)}</small>`;
      b.onclick = () => { onPick(id); closePopover(); };
      box.append(b);
    }
    return box;
  };
  if (key === "steps") {
    title.textContent = "Steps";
    note.textContent = "More steps refine the image but take longer. This model is tuned for 8–12.";
    pop.append(title, range(1, 30, 1, ui.steps, (v) => { ui.steps = v; saveUi(); renderControls(); }), note);
  } else if (key === "shift") {
    title.textContent = "Shift";
    note.textContent = "Higher values spend more effort on the overall layout, lower on fine detail. Default 3.";
    pop.append(title, range(1, 6, 0.5, ui.shift, (v) => { ui.shift = v; saveUi(); renderControls(); }), note);
  } else if (key === "sampler") {
    title.textContent = "Sampler";
    pop.append(title, options(SAMPLER_INFO, ui.sampler, (id) => { ui.sampler = id; saveUi(); renderControls(); }));
  } else {
    title.textContent = "Schedule";
    pop.append(title, options(SCHED_INFO, ui.scheduler, (id) => { ui.scheduler = id; saveUi(); renderControls(); }));
  }
  pop.hidden = false;
  placePopover(anchor);
  anchor.setAttribute("aria-expanded", "true");
}

function placePopover(anchor) {
  const pop = $("advPop");
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  const left = Math.min(Math.max(8, r.left), innerWidth - pw - 8);
  let top = r.bottom + 6;
  if (top + ph > innerHeight - 8) top = r.top - ph - 6;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(8, top)}px`;
}

function closePopover() {
  $("advPop").hidden = true;
  $("advPop").dataset.key = "";
  for (const c of $("advChips").children) c.setAttribute("aria-expanded", "false");
}

// per-step time measured on this device, scaled by pixel count for other sizes
function estimateMs(w = ui.w, h = ui.h, steps = ui.steps) {
  const px = w * h;
  const runs = store.get("speed", []);
  if (!runs.length) return null;
  const ref = runs.reduce((a, r) => (Math.abs(Math.log(r.px / px)) < Math.abs(Math.log(a.px / px)) ? r : a));
  const loraFactor = loras.some((l) => l.on) ? 1.15 : 1;
  return ref.step * (px / ref.px) ** 1.2 * steps * loraFactor + ref.decode * (px / ref.px);
}

function renderGo() {
  const busy = phase === "generating";
  const n = +$("batch").value;
  $("goLabel").textContent = busy ? (n > 1 ? `Queue ${n} more` : "Add to queue") : n > 1 ? `Generate ${n}` : "Generate";
  const est = phase === "ready" || busy ? estimateMs() : null;
  $("estimate").textContent = est ? `about ${fmtTime(est * n)}` : "";
  $("stopBtn").hidden = !busy;
  $("goBtn").disabled = !(phase === "ready" || busy);
  if (busy) setStatus(jobs.length ? `Creating image… (${jobs.length} queued)` : "Creating image…", "busy");
}

function setRandom(on) {
  $("randomBtn").setAttribute("aria-pressed", String(on));
  $("randomBtn").title = on ? "A new random seed for every image (click to keep the seed)" : "Keeping this seed (click for a new random seed every image)";
  store.set("randomSeed", on);
}

function setBoost(on) {
  $("boostBtn").setAttribute("aria-pressed", String(on));
  store.set("boost", on);
}

// ------------------------------------------------------------------ LoRAs

function renderLoraEntry() {
  const on = loras.filter((l) => l.on && !l.pending);
  const stack = $("loraStack");
  stack.innerHTML = on.slice(0, 4).map((l) => (l.image ? `<img alt="" src="${esc(l.image)}">` : "<i></i>")).join("");
  $("loraSum").textContent = !loras.length ? "Add styles & characters" : on.length ? (on.length === 1 ? on[0].name : `${on.length} active`) : `${loras.length} off`;
  $("loraCount").textContent = loras.length ? `${on.length} of ${loras.length} on` : "";
}

function renderLoras() {
  renderLoraEntry();
  const el = $("loraList");
  el.innerHTML = "";
  for (const l of loras) {
    const rep = loraReport.get(l.file);
    const bad = rep && rep.matched === 0;
    const row = document.createElement("div");
    row.className = "lora" + (l.on ? "" : " off") + (bad ? " bad" : "") + (l.pending ? " pending" : "");
    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = "thumb";
    thumb.title = l.on ? "Turn off" : "Turn on";
    thumb.setAttribute("aria-pressed", String(l.on));
    thumb.innerHTML = l.image ? `<img alt="" src="${esc(l.image)}">` : "LoRA";
    thumb.disabled = !!l.pending;
    thumb.onclick = () => { l.on = !l.on; saveLoras(); renderLoras(); renderGo(); };
    const mid = document.createElement("div");
    mid.className = "mid";
    mid.innerHTML = `<span class="title" title="${esc(l.name)}${l.version ? " · " + esc(l.version) : ""}">${esc(l.name)} ${l.version ? `<small>${esc(l.version)}</small>` : ""}</span>`;
    if (l.pending) {
      mid.insertAdjacentHTML("beforeend", `<div class="bar-progress"><div style="width:${(l.progress || 0) * 100}%"></div></div>`);
    } else if (bad) {
      mid.insertAdjacentHTML("beforeend", `<span class="msg">Doesn't fit this model. Nothing applied.</span>`);
    } else {
      const str = document.createElement("div");
      str.className = "str";
      const r = document.createElement("input");
      Object.assign(r, { type: "range", min: -1, max: 2, step: 0.05, value: l.strength });
      r.setAttribute("aria-label", `${l.name} strength`);
      r.title = "Strength (double-click to reset)";
      const o = document.createElement("output");
      o.textContent = (+l.strength).toFixed(2);
      r.oninput = () => { l.strength = +r.value; o.textContent = l.strength.toFixed(2); saveLoras(); };
      r.ondblclick = () => { l.strength = 1; r.value = 1; o.textContent = "1.00"; saveLoras(); };
      str.append(r, o);
      mid.append(str);
    }
    const btns = document.createElement("div");
    btns.className = "btns";
    if (l.words?.length && !l.pending) {
      const w = document.createElement("button");
      w.type = "button";
      w.title = `Insert trigger words: ${l.words.join(", ")}`;
      w.innerHTML = ICONS.words;
      w.onclick = () => insertWords(l.words);
      btns.append(w);
    }
    if (l.page) {
      const a = document.createElement("a");
      a.href = l.page;
      a.target = "_blank";
      a.rel = "noopener";
      a.title = "Open its page";
      a.innerHTML = ICONS.ext;
      btns.append(a);
    }
    if (!l.pending) {
      const x = document.createElement("button");
      x.type = "button";
      x.title = "Remove";
      x.innerHTML = ICONS.x;
      x.onclick = async () => {
        loras = loras.filter((o) => o !== l);
        saveLoras();
        renderLoras();
        renderGo();
        if (!loras.some((o) => o.file === l.file)) await removeLoraFile(l.file);
      };
      btns.append(x);
    }
    row.append(thumb, mid, btns);
    el.append(row);
  }
}

function insertWords(words) {
  const p = $("prompt");
  const add = words.filter((w) => !p.value.toLowerCase().includes(w.toLowerCase()));
  if (!add.length) return;
  p.value = p.value.trim() ? `${p.value.trim().replace(/,\s*$/, "")}, ${add.join(", ")}` : add.join(", ");
  store.set("prompt", p.value);
}

const activeLoras = () => loras.filter((l) => l.on && !l.pending && loraReport.get(l.file)?.matched !== 0).map((l) => ({ file: l.file, strength: l.strength }));

// Make the engine's LoRA set match `list` (skips when unchanged; strength-only changes are cheap).
async function syncLoras(list) {
  const key = JSON.stringify(list);
  if (key === appliedKey) return;
  const report = await pipe.setLoras(list);
  appliedKey = key;
  let changed = false;
  for (const r of report) {
    if (loraReport.get(r.key)?.matched !== r.matched) changed = true;
    loraReport.set(r.key, r);
  }
  if (changed) renderLoras();
}

// ---- inline add (LoRA view)

let lookup = null;

function loraStatus(text, kind = "") {
  const el = $("loraStatus");
  el.innerHTML = text;
  el.style.color = kind === "err" ? "var(--bad)" : kind === "ok" ? "var(--good)" : "";
}

function resetLookup() {
  lookup = null;
  $("loraCard").hidden = true;
  $("loraCard").innerHTML = "";
  loraStatus("");
}

let lookupAbort = null;

async function doLookup() {
  const text = $("loraUrl").value.trim();
  lookupAbort?.abort();
  if (!text) { resetLookup(); return; }
  const ref = parseHfUrl(text);
  if (!ref) {
    resetLookup();
    loraStatus(/huggingface\.co|hf\.co/.test(text)
      ? "Link to the LoRA file itself: open the repo's <b>Files</b> tab, click the <b>.safetensors</b> file and copy that page's address."
      : "Paste a Hugging Face link to a <b>.safetensors</b> file.", "err");
    return;
  }
  loraStatus("Looking it up…");
  const ac = (lookupAbort = new AbortController());
  try {
    lookup = await fetchLoraInfo(ref, store.get("hfToken", ""), ac.signal);
  } catch (e) {
    if (e.name === "AbortError") return;
    resetLookup();
    showLoraError(e);
    return;
  }
  renderLookup();
}

function showLoraError(e) {
  if (e.needsKey) {
    loraStatus(`Couldn't open this file: check the link. If the LoRA is gated or private, accept its terms on Hugging Face and <a href="#" id="openKey">add an access token in settings</a>.`, "err");
    $("openKey").onclick = (ev) => { ev.preventDefault(); openSettings(true); };
  } else {
    loraStatus(esc(/^(That|Hugging Face)/.test(e.message) ? e.message : friendlyError(e)), "err");
  }
}

function renderLookup() {
  const i = lookup;
  const card = $("loraCard");
  const dup = loras.find((l) => l.id === i.id);
  let problem = "";
  if (!i.check.lora) problem = i.check.unsupported ? "This LoRA uses a format that isn't supported yet (LoKr, LoHa or DoRA)." : "This file isn't a LoRA.";
  else if (dup) problem = "Already in your list.";
  const fits = i.madeFor === "anima";
  card.innerHTML = `${i.image ? `<img alt="" src="${esc(i.image)}">` : '<div class="noimg"></div>'}
    <div>
      <h3 title="${esc(i.repo)}">${esc(i.name)}</h3>
      <div class="meta">
        <span title="${esc(i.path)}">${esc(i.fileName.replace(/\.safetensors$/i, ""))}</span>
        ${i.check.lora ? `<span class="badge ${fits ? "" : "warn"}">${fits ? "Fits Anima" : "Made for another model"}</span>` : ""}
        <span>${fmtGB(i.size)}</span>
        <a href="${esc(i.pageUrl)}" target="_blank" rel="noopener">Page ↗</a>
      </div>
      ${i.words.length ? `<div class="words" title="Trigger words">${i.words.slice(0, 6).map((w) => `<span>${esc(w.length > 40 ? w.slice(0, 40) + "…" : w)}</span>`).join("")}</div>` : ""}
    </div>
    <div class="card-actions">
      <button class="ghost" type="button" id="cardCancel">Cancel</button>
      <button class="primary slim" type="button" id="cardAdd" ${problem ? "disabled" : ""}>${fits ? "Download & add" : "Add anyway"}</button>
    </div>`;
  card.hidden = false;
  loraStatus(problem || (fits ? "" : "Its layers don't match Anima, so it probably won't do anything."), problem ? "err" : "");
  $("cardCancel").onclick = () => { $("loraUrl").value = ""; resetLookup(); };
  $("cardAdd").onclick = doAdd;
}

function showDownloadProgress(done, total, rate) {
  let dl = $("loraCard").querySelector(".dl");
  if (!dl) {
    $("loraCard").querySelector(".card-actions")?.remove();
    dl = document.createElement("div");
    dl.className = "dl";
    dl.innerHTML = `<div class="bar-progress"><div></div></div><div class="dl-meta"><span class="dl-done"></span><span class="dl-rate"></span></div>`;
    $("loraCard").append(dl);
  }
  const frac = total ? done / total : 0;
  dl.querySelector(".bar-progress > div").style.width = `${frac * 100}%`;
  dl.querySelector(".dl-done").textContent = `Downloading ${fmtGB(done)} of ${fmtGB(total)}`;
  dl.querySelector(".dl-rate").textContent = rate > 0 ? `${fmtGB(rate)}/s · ${fmtTime(((total - done) / rate) * 1000)} left` : `${Math.round(frac * 100)}%`;
}

async function doAdd() {
  const i = lookup;
  if (!i) return;
  const file = `hf-${Date.now()}-${i.fileName.replace(/[^\w.-]+/g, "_")}`;
  const meta = { id: i.id, name: i.name, version: i.fileName.replace(/\.safetensors$/i, ""), image: i.image, page: i.pageUrl, words: i.words, file, size: i.size };
  loraStatus("");
  const entry = { strength: 1, on: true, ...meta, pending: true, progress: 0 };
  loras.unshift(entry);
  renderLoras();
  const meter = rateMeter();
  showDownloadProgress(0, i.size, 0);
  try {
    const res = await openDownload(i, store.get("hfToken", ""));
    await saveLoraFile(file, res, (n) => {
      entry.progress = n / i.size;
      showDownloadProgress(n, i.size, meter(n));
      const bar = $("loraList").querySelector(".lora.pending .bar-progress > div");
      if (bar) bar.style.width = `${entry.progress * 100}%`;
    });
    delete entry.pending;
    delete entry.progress;
    saveLoras();
    renderLoras();
    renderGo();
    $("loraUrl").value = "";
    resetLookup();
    loraStatus(`Added <b>${esc(i.name)}</b>.`, "ok");
    validateInEngine();
  } catch (e) {
    loras = loras.filter((l) => l !== entry);
    renderLoras();
    renderLookup();
    showLoraError(e);
    removeLoraFile(file).catch(() => {});
  }
}

async function validateLoraFile(blob) {
  try {
    const st = await SafeTensors.open(blob);
    return Object.keys(st.header).some((k) => /lora_(A|B|down|up)\.weight$/.test(k));
  } catch {
    return false;
  }
}

async function importFile(f) {
  if (!f) return;
  loraStatus("Checking the file…");
  if (!(await validateLoraFile(f))) { loraStatus("That file isn't a LoRA in a supported format.", "err"); return; }
  const m = { id: `local-${Date.now()}`, name: f.name.replace(/\.safetensors$/i, ""), version: "", image: null, page: null, words: [] };
  const file = `local-${Date.now()}-${f.name.replace(/[^\w.-]+/g, "_")}`;
  await saveLoraFile(file, f);
  loras.unshift({ strength: 1, on: true, ...m, file, size: f.size });
  saveLoras();
  renderLoras();
  renderGo();
  $("loraUrl").value = "";
  resetLookup();
  loraStatus(`Added <b>${esc(m.name)}</b>.`, "ok");
  validateInEngine();
}

// check a newly added LoRA against the model (flags files that don't fit) when idle
async function validateInEngine() {
  if (phase !== "ready") return;
  try { await syncLoras(activeLoras()); } catch (e) { console.error(e); }
}

// ------------------------------------------------------------------ model selection

function currentPreset() {
  return PRESETS.find((p) => p.dit === sel.dit && p.te === sel.te) || null;
}

function normalizeSelection() {
  const m = manifest.models.find((x) => x.id === sel?.model) || manifest.models[0];
  const pick = (avail, want) => (avail.includes(want) ? want : ["int8", "w4a8", "bf16"].find((p) => avail.includes(p)) || avail[0]);
  sel = { model: m.id, dit: pick(Object.keys(m.dit), sel?.dit ?? "int8"), te: pick(Object.keys(manifest.te), sel?.te ?? "int8") };
  store.set("selection", sel);
}

function presetBytes(p) {
  const m = manifest.models.find((x) => x.id === sel.model);
  if (!m.dit[p.dit] || !manifest.te[p.te]) return null;
  return m.dit[p.dit].size + manifest.te[p.te].size + manifest.vae.size;
}

async function cachedMap() {
  return new Map((await listCached()).map((f) => [f.name, f.size]));
}

function isSaved(cached, p) {
  const { files } = resolveFiles(manifest, { ...sel, dit: p.dit, te: p.te });
  return ["dit", "te", "vae"].every((k) => cached.get(files[k].path) === files[k].size);
}

function renderPresets(el, cached, onPick) {
  el.innerHTML = "";
  for (const p of PRESETS) {
    const bytes = presetBytes(p);
    if (bytes == null) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(sel.dit === p.dit && sel.te === p.te));
    const vram = bytes + 1.5 * 2 ** 30;
    b.innerHTML = `${p.tag ? `<span class="tag">${p.tag}</span>` : ""}<b>${esc(p.label)}</b><span class="size">${fmtGB(bytes)} download</span><span class="desc">${esc(p.desc)} Needs about ${fmtGB(vram)} of graphics memory.</span>${isSaved(cached, p) ? '<span class="saved">✓ Downloaded</span>' : ""}`;
    b.onclick = () => onPick(p);
    el.append(b);
  }
}

function chipText() {
  const p = currentPreset();
  $("gpuChip").textContent = (p ? p.label : "Custom") + (backend === "webnn" ? " · WebNN" : "");
  $("gpuChip").hidden = false;
}

// ------------------------------------------------------------------ loading

async function ensureModel() {
  const { files } = resolveFiles(manifest, sel);
  if (pipe.isLoaded(files, backend)) { toReady(); return; }
  const cached = await cachedMap();
  const allSaved = ["dit", "te", "vae"].every((k) => cached.get(files[k].path) === files[k].size);
  if (allSaved) return loadModel();
  showWelcome(cached);
}

function showWelcome(cached) {
  phase = "welcome";
  showPanel("welcome");
  renderGo();
  setStatus("Choose a download to get started");
  $("welcome").querySelector("h1").textContent = cached.size === 0 ? "Make anime art on your own device" : "Download this version";
  renderPresets($("welcomePresets"), cached, (p) => {
    sel = { ...sel, dit: p.dit, te: p.te };
    store.set("selection", sel);
    chipText();
    ensureModel();
  });
  const { files } = resolveFiles(manifest, sel);
  const need = ["dit", "te", "vae"].filter((k) => cached.get(files[k].path) !== files[k].size).reduce((a, k) => a + files[k].size, 0);
  $("welcomeGo").textContent = need ? `Download ${fmtGB(need)} & start` : "Start";
  $("welcomeNote").textContent = "Downloads resume if interrupted. You can change this later in settings.";
  $("welcomeToken").value = store.get("hfToken", "");
  showTokenStatus();
}

// Optional Hugging Face token (shared by the first-run screen and settings): Hugging Face gives
// signed-in downloads higher limits, and gated or private LoRAs need one.
const TOKEN_HELP = "Signed-in downloads get higher limits on Hugging Face. A read token from a free account is enough. It's stored only in this browser and sent only to Hugging Face.";
let tokenCheck = 0;
async function showTokenStatus() {
  const el = $("welcomeTokenStatus");
  const token = store.get("hfToken", "");
  const run = ++tokenCheck;
  el.className = "help";
  el.textContent = token ? "Checking the token…" : TOKEN_HELP;
  if (!token) return;
  let r;
  try { r = await checkToken(token); } catch { r = null; }
  if (run !== tokenCheck) return;
  if (!r) { el.textContent = TOKEN_HELP; return; }
  el.className = r.ok ? "help ok" : "help err";
  el.textContent = r.ok ? `Signed in as ${r.name}. Downloads will use your account.` : "Hugging Face doesn't accept this token. Check it, or leave the field empty.";
}

function setToken(v) {
  store.set("hfToken", v.trim());
  clearTimeout(setToken.t);
  setToken.t = setTimeout(showTokenStatus, 500);
}

async function loadModel() {
  phase = "loading";
  showPanel("loading");
  renderGo();
  $("loadingCancel").hidden = true;
  $("loadingTitle").textContent = "Getting ready…";
  $("loadingText").textContent = "";
  $("loadingBar").style.width = "0%";
  loadAbort = new AbortController();
  appliedKey = null;
  const meter = rateMeter();
  try {
    await pipe.load(BASE, manifest, { ...sel, backend }, {
      token: store.get("hfToken", ""),
      signal: loadAbort.signal,
      onStatus: (s) => {
        if (s.phase === "download") {
          $("loadingCancel").hidden = false;
          const rate = meter(s.done);
          $("loadingTitle").textContent = "Downloading the model";
          $("loadingBar").style.width = `${(100 * s.done) / s.total}%`;
          const left = rate > 0 ? ` · about ${fmtTime(((s.total - s.done) / rate) * 1000)} left` : "";
          $("loadingText").textContent = `${fmtGB(s.done)} of ${fmtGB(s.total)}${left}`;
          setStatus(`Downloading ${Math.floor((100 * s.done) / s.total)}%`, "busy");
        } else if (s.phase === "load") {
          $("loadingCancel").hidden = true;
          $("loadingTitle").textContent = "Loading onto your graphics card";
          $("loadingBar").style.width = `${s.frac * 100}%`;
          const what = { "text encoder": "text understanding", VAE: "image decoder", LoRAs: "LoRAs" }[s.what] || "image model";
          $("loadingText").textContent = `Preparing the ${what}…`;
          setStatus("Loading model…", "busy");
        }
      },
    });
    await syncLoras(activeLoras());
    toReady();
  } catch (e) {
    phase = "welcome";
    if (e.name === "AbortError") { ensureModel(); return; }
    console.error(e);
    if (backend === "webnn") {
      // the experimental engine couldn't take this model: fall back instead of leaving the app stuck
      showError(`WebNN couldn't load the model (${e.message}). Switched back to the WebGPU engine.`);
      setBackend("webgpu");
      return;
    }
    setStatus("Couldn't load the model", "err");
    showWelcome(await cachedMap());
    showError(friendlyError(e));
  }
}

function toReady() {
  phase = "ready";
  setStatus("Ready", "ok");
  chipText();
  if (current) showItem(current); else showPanel("empty");
  renderGo();
  if (jobs.length) runQueue();
}

// ------------------------------------------------------------------ queue & generation

// The viewer shows either a finished image (`current`) or, while `live`, the image being made.
// Every draw bumps the token so a slow image decode can't paint over a newer choice.
let drawToken = 0;

function draw(img) {
  drawToken++;
  const c = $("canvas");
  if (c.width !== img.width || c.height !== img.height) { c.width = img.width; c.height = img.height; }
  c.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  showPanel("canvas");
}

async function showItem(item) {
  const tok = ++drawToken;
  const im = new Image();
  im.src = item.url;
  try { await im.decode(); } catch { return; }
  if (tok !== drawToken) return;
  const c = $("canvas");
  c.width = im.naturalWidth;
  c.height = im.naturalHeight;
  c.getContext("2d").drawImage(im, 0, 0);
  showPanel("canvas");
}

// Scale the small preview (kept on the running job) up to the full canvas.
function drawLive() {
  if (!running?.hasPreview) return;
  drawToken++;
  const c = $("canvas");
  const { w, h } = running.ui;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(running.thumb, 0, 0, w, h);
  showPanel("canvas");
}

function onPreview(job, img) {
  const t = job.thumb;
  if (t.width !== img.width || t.height !== img.height) { t.width = img.width; t.height = img.height; }
  t.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  job.hasPreview = true;
  for (const c of document.querySelectorAll(".tile.running canvas")) {
    if (c.width !== t.width || c.height !== t.height) { c.width = t.width; c.height = t.height; }
    c.getContext("2d").drawImage(t, 0, 0);
    c.parentElement.classList.add("has-preview");
  }
  if (live) drawLive();
}

// Snapshot everything at click time, so the settings can keep changing while jobs wait.
function enqueue() {
  if (!(phase === "ready" || phase === "generating")) return;
  closePopover();
  const n = +$("batch").value;
  const random = pressed("randomBtn");
  if ($("seed").value.trim() === "") $("seed").value = newSeed();
  const base = Math.min(2 ** 32 - 1, Math.max(0, Math.floor(Number($("seed").value) || 0)));
  const text = $("prompt").value.trim();
  const boost = pressed("boostBtn");
  const loraSnap = loras.filter((l) => l.on && !l.pending && loraReport.get(l.file)?.matched !== 0).map((l) => ({ id: l.id, name: l.name, file: l.file, strength: l.strength }));
  for (let k = 0; k < n; k++) {
    // random mode: a fresh seed per image; fixed seed + batch: consecutive seeds
    const seed = random ? newSeed() : (base + k) >>> 0;
    jobs.push({ text, boost, ui: { ...ui }, seed, loras: loraSnap });
    if (k === 0 || random) $("seed").value = seed;
  }
  store.set("seed", $("seed").value);
  store.set("prompt", $("prompt").value);
  renderStrip();
  if (!running) runQueue();
  else renderGo();
}

async function runQueue() {
  if (running || phase !== "ready" && phase !== "generating") return;
  phase = "generating";
  renderGo();
  while (jobs.length) {
    const job = jobs.shift();
    running = { ...job, abort: new AbortController(), frac: 0, thumb: document.createElement("canvas"), hasPreview: false };
    // follow the new image unless the user is looking at an older one
    if (!pinned) live = true;
    renderStrip();
    renderInfo();
    await runJob(running);
    running = null;
    renderStrip();
  }
  phase = "ready";
  live = false;
  pinned = false;
  renderInfo();
  $("progress").hidden = true;
  setStatus("Ready", "ok");
  renderGo();
  renderGpuChip();
}

async function runJob(job) {
  const { w: width, h: height, steps, sampler, scheduler, shift } = job.ui;
  $("errorCard").hidden = true;
  $("progress").hidden = !live;
  $("progressBar").style.width = "0%";
  const queued = () => (jobs.length ? ` · ${jobs.length} more queued` : "");
  $("progressText").textContent = `Reading your prompt…${queued()}`;
  setStatus(jobs.length ? `Creating image… (${jobs.length} queued)` : "Creating image…", "busy");
  let backgrounded = document.visibilityState === "hidden";
  const onVis = () => { if (document.visibilityState === "hidden") backgrounded = true; };
  document.addEventListener("visibilitychange", onVis);
  const t0 = performance.now();
  let sampleStart = 0;
  try {
    await syncLoras(job.loras.map((l) => ({ file: l.file, strength: l.strength })));
    const res = await pipe.generate({
      prompt: job.boost && !/^masterpiece/i.test(job.text) ? BOOST + job.text : job.text,
      width, height, steps, sampler, scheduler, shift, seed: job.seed,
      signal: job.abort.signal,
      onPreview: (img) => onPreview(job, img),
      onProgress: (p) => {
        if (p.phase === "sample") {
          if (!sampleStart) sampleStart = performance.now();
          const el = performance.now() - sampleStart;
          const left = p.frac > 0.08 ? ` · about ${fmtTime((el / p.frac) * (1 - p.frac))} left` : "";
          job.frac = p.frac * 0.92;
          $("progressBar").style.width = `${job.frac * 100}%`;
          $("progressText").textContent = `Step ${p.step + 1} of ${p.steps}${left}${queued()}`;
        } else if (p.phase === "compile") {
          $("progressText").textContent = `Building the WebNN graph for ${p.width} × ${p.height} (first image at this size only) ${Math.round(p.frac * 100)}%…${queued()}`;
        } else if (p.phase === "decode") {
          job.frac = 0.92 + p.frac * 0.08;
          $("progressBar").style.width = `${job.frac * 100}%`;
          $("progressText").textContent = `Finishing up…${queued()}`;
        }
        updateRunningTile();
      },
    });
    if (live) draw(res.image);
    if (!backgrounded && !job.loras.length && backend === "webgpu") {
      const runs = store.get("speed", []).filter((r) => r.px !== width * height);
      runs.push({ px: width * height, step: res.timings.perStep, decode: res.timings.decode + res.timings.encode });
      store.set("speed", runs.slice(-6));
    }
    const record = { text: job.text, boost: job.boost, ui: job.ui, seed: job.seed, loras: job.loras };
    await addToGallery(res.image, record, { ...res.timings, total: performance.now() - t0, backgrounded, width, height }, live);
  } catch (e) {
    if (e.name !== "AbortError") {
      console.error(e);
      showError(friendlyError(e));
      jobs.length = 0; // don't keep hammering a failing GPU
    }
    if (live) { if (current) showItem(current); else showPanel("empty"); }
  } finally {
    document.removeEventListener("visibilitychange", onVis);
  }
}

function stopAll() {
  jobs.length = 0;
  running?.abort.abort();
  renderStrip();
}

function renderGpuChip() {
  const g = pipe.gpu || {};
  $("gpuChip").title = `${g.name || "Graphics card"}${g.peak ? ` · up to ${fmtGB(g.peak)} of working memory used` : ""}`;
}

function toBlobURL(img) {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  c.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return new Promise((r) => c.toBlob((b) => r(URL.createObjectURL(b)), "image/png"));
}

// Finished images live as PNG blobs (not raw pixels), so a long session stays light on memory.
const GALLERY_MAX = 100;

async function addToGallery(img, record, timings, follow) {
  const url = await toBlobURL(img);
  const item = { url, width: img.width, height: img.height, record, timings };
  gallery.unshift(item);
  if (gallery.length > GALLERY_MAX) {
    const old = gallery.pop();
    URL.revokeObjectURL(old.url);
    if (current === old) current = null;
  }
  if (follow) current = item;
  renderInfo();
  renderStrip();
}

function select(item) {
  current = item;
  if (running) { live = false; pinned = true; $("progress").hidden = true; }
  showItem(item);
  openGallery(false);
  renderInfo();
  renderStrip();
}

function selectLive() {
  if (!running) return;
  live = true;
  pinned = false;
  $("progress").hidden = false;
  drawLive();
  openGallery(false);
  renderInfo();
  renderStrip();
}

function metaLine(parts, note) {
  const el = document.createElement("span");
  el.className = "meta";
  el.innerHTML = parts.map((p, i) => (i ? '<span class="sep">·</span>' : "") + p).join("") + (note ? `<span class="sep">·</span><span class="note">${esc(note)}</span>` : "");
  return el;
}

function settingsBits(r, w, h) {
  const bits = [`<b>${w} × ${h}</b>`, `${r.ui.steps} steps`, esc(SAMPLER_INFO[r.ui.sampler][0]), `seed ${r.seed}`];
  if (r.loras.length) bits.push(r.loras.length === 1 ? `LoRA: ${esc(r.loras[0].name)}` : `${r.loras.length} LoRAs`);
  return bits;
}

function renderInfo() {
  const el = $("info");
  el.innerHTML = "";
  if (live && running) {
    const m = metaLine(["<b>Creating</b>", ...settingsBits(running, running.ui.w, running.ui.h)]);
    m.title = running.text;
    el.append(m);
    return;
  }
  if (!current) return;
  const { record: r, timings: t } = current;
  const m = metaLine([...settingsBits(r, t.width, t.height), `made in ${fmtTime(t.total)}`], t.backgrounded ? "slower: the tab was in the background" : "");
  m.title = r.text;
  const actions = document.createElement("span");
  actions.className = "actions";
  const dl = document.createElement("a");
  dl.href = current.url;
  dl.download = `anima-${r.seed}.png`;
  dl.title = "Save this image";
  dl.innerHTML = `${ICONS.dl}<span>Save</span>`;
  const reuse = document.createElement("button");
  reuse.type = "button";
  reuse.innerHTML = `${ICONS.reuse}<span>Use these settings</span>`;
  reuse.title = "Restore this image's prompt, settings, seed and LoRA strengths";
  reuse.onclick = () => {
    $("prompt").value = r.text;
    store.set("prompt", r.text);
    setBoost(r.boost);
    Object.assign(ui, r.ui);
    saveUi();
    $("seed").value = r.seed;
    store.set("seed", String(r.seed));
    setRandom(false);
    for (const l of loras) {
      const m = r.loras.find((x) => x.id === l.id);
      l.on = !!m;
      if (m) l.strength = m.strength;
    }
    saveLoras();
    renderLoras();
    renderControls();
  };
  actions.append(dl, reuse);
  el.append(m, actions);
}

function ringSVG(frac) {
  const c = 2 * Math.PI * 14;
  return `<svg class="ring" viewBox="0 0 34 34"><circle class="bg" cx="17" cy="17" r="14"/><circle class="fg" cx="17" cy="17" r="14" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - frac)}"/></svg>`;
}

function updateRunningTile() {
  if (!running) return;
  for (const t of document.querySelectorAll(".tile.running")) {
    t.querySelector(".fg")?.setAttribute("stroke-dashoffset", String(2 * Math.PI * 14 * (1 - running.frac)));
    t.querySelector(".tile-bar i").style.width = `${running.frac * 100}%`;
  }
}

// Tiles in display order: waiting jobs (next one nearest), the one being made, then finished images (newest first).
function renderTiles(el) {
  el.innerHTML = "";
  jobs.forEach((job, i) => {
    const b = document.createElement("div");
    b.className = "tile queued";
    b.title = `Waiting: ${job.text}`;
    b.innerHTML = `<span>${i === 0 ? "Next" : `#${i + 1}`}</span>`;
    const x = document.createElement("button");
    x.type = "button";
    x.className = "x";
    x.title = "Remove from queue";
    x.setAttribute("aria-label", "Remove from queue");
    x.textContent = "×";
    x.onclick = () => { const k = jobs.indexOf(job); if (k >= 0) jobs.splice(k, 1); renderStrip(); };
    b.append(x);
    el.prepend(b); // last queued on the far left
  });
  if (running) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tile running" + (running.hasPreview ? " has-preview" : "");
    b.title = `Being made: ${running.text}`;
    b.setAttribute("aria-current", String(live));
    const c = document.createElement("canvas");
    if (running.hasPreview) {
      c.width = running.thumb.width;
      c.height = running.thumb.height;
      c.getContext("2d").drawImage(running.thumb, 0, 0);
    }
    b.append(c);
    b.insertAdjacentHTML("beforeend", `${ringSVG(running.frac)}<span class="tile-bar"><i style="width:${running.frac * 100}%"></i></span>`);
    b.onclick = selectLive;
    el.append(b);
  }
  for (const item of gallery) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tile";
    b.title = item.record.text;
    b.setAttribute("aria-current", String(!live && item === current));
    // reuse the strip's decoded <img> so re-rendering doesn't flash empty tiles
    let im = el.id === "strip" ? item.stripImg : null;
    if (!im) {
      im = document.createElement("img");
      im.src = item.url;
      im.alt = item.record.text;
      if (el.id === "strip") item.stripImg = im;
    }
    b.append(im);
    b.onclick = () => select(item);
    el.append(b);
  }
}

function renderStrip() {
  renderTiles($("strip"));
  if (galleryOpen) renderTiles($("galleryGrid"));
  const n = gallery.length;
  $("allCount").textContent = n ? `All ${n}` : "All";
  $("allBtn").disabled = !n && !running && !jobs.length;
  $("galleryTitle").textContent = n === 1 ? "1 image" : `${n} images`;
  renderGo();
}

function openGallery(on) {
  galleryOpen = on;
  $("gallery").hidden = !on;
  $("allBtn").setAttribute("aria-pressed", String(on));
  if (on) renderTiles($("galleryGrid"));
  else $("galleryGrid").innerHTML = "";
}

// ------------------------------------------------------------------ settings dialog

async function openSettings(focusKey = false) {
  const cached = await cachedMap();
  const models = manifest.models;
  $("versionField").hidden = models.length < 2;
  $("version").innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join("");
  $("version").value = sel.model;
  renderPresets($("presets"), cached, (p) => {
    $("settings").close();
    changeSelection({ dit: p.dit, te: p.te });
  });
  const fill = (el, entries, value) => {
    el.innerHTML = ["bf16", "int8", "w4a8"].filter((q) => entries[q]).map((q) =>
      `<option value="${q}">${PART_LABELS[q]} (${fmtGB(entries[q].size)})${cached.get(entries[q].path) === entries[q].size ? " ✓" : ""}</option>`).join("");
    el.value = value;
  };
  fill($("ditSel"), models.find((m) => m.id === sel.model).dit, sel.dit);
  fill($("teSel"), manifest.te, sel.te);
  $("hfToken").value = store.get("hfToken", "");
  const modelBytes = [...cached.values()].reduce((a, b) => a + b, 0);
  const loraBytes = await loraStorageBytes();
  const saved = [modelBytes && `Model files ${fmtGB(modelBytes)}`, loraBytes && `LoRAs ${fmtGB(loraBytes)}`].filter(Boolean);
  $("storageText").textContent = saved.length ? saved.join(" · ") : "Nothing downloaded yet";
  const busy = phase === "generating" || phase === "loading";
  $("clearBtn").disabled = !modelBytes || busy;
  $("clearLorasBtn").disabled = !loraBytes || busy;
  for (const el of [$("version"), $("ditSel"), $("teSel"), $("backendSel"), ...$("presets").querySelectorAll("button")]) el.disabled = busy;
  renderBackend();
  $("settings").showModal();
  if (focusKey) $("hfToken").focus();
}

// ------------------------------------------------------------------ optional WebNN engine

let webnnProbe = null;
async function renderBackend() {
  const sel_ = $("backendSel");
  sel_.value = backend;
  const help = $("backendHelp");
  const opt = sel_.querySelector('option[value="webnn"]');
  if (!hasWebNN()) {
    opt.disabled = true;
    help.innerHTML = `WebNN is the browser's built-in machine-learning API. It can run the image model on your
      graphics card's AI (tensor) cores, which is often faster. It's turned off in this browser. To try it in
      Chrome or Edge:<br>1. Open <code>${WEBNN_FLAG}</code> (Edge: <code>${WEBNN_FLAG.replace("chrome", "edge")}</code>) in a new tab.<br>
      2. Set <b>WebNN API</b> to <b>Enabled</b>.<br>3. Relaunch the browser and come back here.`;
    return;
  }
  help.textContent = "Checking WebNN…";
  const r = await (webnnProbe ||= probeWebNN());
  opt.disabled = !r.ok && backend !== "webnn";
  help.textContent = r.ok
    ? `Experimental. Runs the image model through WebNN${r.float16 ? " in half precision" : ""}, which can use your graphics card's AI cores. The text encoder and image decoder stay on WebGPU. The first image at each size takes longer while WebNN builds its graph. Images can differ slightly from the WebGPU engine. If it's slower on your device or something breaks, switch back.`
    : `WebNN is enabled but couldn't start (${r.reason}). Update your browser or graphics driver, or keep using WebGPU.`;
}

function setBackend(v) {
  backend = v === "webnn" ? "webnn" : "webgpu";
  store.set("backend", backend);
  chipText();
  ensureModel();
}

function changeSelection(patch) {
  if (phase === "generating" || phase === "loading") return;
  sel = { ...sel, ...patch };
  normalizeSelection();
  chipText();
  ensureModel();
}

// ------------------------------------------------------------------ init

async function init() {
  if (!navigator.gpu) {
    $("fatal").hidden = false;
    $("fatal").textContent = "This browser can't run Anima.\n\nIt needs WebGPU: use a recent version of Chrome or Edge on a computer with a graphics card.";
    return;
  }
  setStatus("Starting…", "busy");
  try {
    [manifest, pipe] = await Promise.all([fetchManifest(BASE), createEngine({ inPage: params.get("engine") === "page" })]);
  } catch (e) {
    $("fatal").hidden = false;
    $("fatal").textContent = "Couldn't find the model files.\n\n" + e.message;
    return;
  }
  normalizeSelection();
  if (backend !== "webgpu" && (backend !== "webnn" || !hasWebNN())) backend = "webgpu"; // flag turned off since

  // prompt
  $("prompt").value = store.get("prompt", EXAMPLES[0]);
  $("prompt").oninput = () => store.set("prompt", $("prompt").value);
  setBoost(store.get("boost", true));
  $("boostBtn").onclick = () => setBoost(!pressed("boostBtn"));
  $("exampleBtn").onclick = () => {
    exampleIdx = (exampleIdx + 1) % EXAMPLES.length;
    $("prompt").value = EXAMPLES[exampleIdx];
    store.set("prompt", $("prompt").value);
    $("prompt").focus();
  };

  // canvas sliders
  for (const [id, key] of [["wRange", "w"], ["hRange", "h"]]) {
    $(id).oninput = () => { ui[key] = +$(id).value; saveUi(); renderControls(); };
  }

  // seed
  setRandom(store.get("randomSeed", true));
  $("seed").value = store.get("seed", String(newSeed()));
  $("randomBtn").onclick = () => setRandom(!pressed("randomBtn"));
  $("seed").oninput = () => {
    $("seed").value = $("seed").value.replace(/\D/g, "").slice(0, 10);
    setRandom(false); // typing a seed means "use this one"
    store.set("seed", $("seed").value);
  };
  $("rollBtn").onclick = () => {
    $("seed").value = newSeed();
    setRandom(false); // a rolled seed should be the one that's used
    store.set("seed", $("seed").value);
  };

  // popover
  document.addEventListener("click", (e) => { if (!$("advPop").hidden && !$("advPop").contains(e.target)) closePopover(); });
  window.addEventListener("resize", closePopover);
  $("advPop").addEventListener("click", (e) => e.stopPropagation());

  // LoRA view
  $("loraEntry").onclick = () => showView("loras");
  $("loraBack").onclick = () => showView("main");
  $("loraLookup").onclick = doLookup;
  $("loraUrl").addEventListener("paste", () => setTimeout(doLookup, 0));
  $("loraUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); doLookup(); } });
  $("loraLocalInput").onchange = (e) => importFile(e.target.files[0]);
  $("loraAllOff").onclick = () => { for (const l of loras) l.on = false; saveLoras(); renderLoras(); renderGo(); };
  const lv = $("loraView");
  lv.addEventListener("dragover", (e) => e.preventDefault());
  lv.addEventListener("drop", (e) => {
    e.preventDefault();
    importFile(e.dataTransfer.files[0]);
  });
  renderLoras();

  // generate / queue
  $("batch").value = String(store.get("batch", 1));
  $("batch").onchange = () => { store.set("batch", +$("batch").value); renderGo(); };
  $("goBtn").onclick = enqueue;
  $("stopBtn").onclick = stopAll;
  $("allBtn").onclick = () => openGallery(!galleryOpen);
  $("galleryClose").onclick = () => openGallery(false);
  renderStrip();
  renderControls();
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); enqueue(); }
    if (e.key === "Escape") { closePopover(); if (galleryOpen) openGallery(false); }
  });

  // model & settings
  $("settingsBtn").onclick = () => openSettings();
  $("gpuChip").onclick = () => openSettings();
  $("welcomeGo").onclick = loadModel;
  $("loadingCancel").onclick = () => loadAbort?.abort();
  $("version").onchange = () => { $("settings").close(); changeSelection({ model: $("version").value }); };
  $("ditSel").onchange = () => { $("settings").close(); changeSelection({ dit: $("ditSel").value }); };
  $("teSel").onchange = () => { $("settings").close(); changeSelection({ te: $("teSel").value }); };
  $("backendSel").onchange = () => {
    if (phase === "generating" || phase === "loading") return;
    $("settings").close();
    setBackend($("backendSel").value);
  };
  $("hfToken").oninput = () => setToken($("hfToken").value);
  $("welcomeToken").oninput = () => setToken($("welcomeToken").value);
  $("keyToggle").onclick = () => {
    const show = $("hfToken").type === "password";
    $("hfToken").type = show ? "text" : "password";
    $("keyToggle").textContent = show ? "Hide" : "Show";
  };
  $("clearLorasBtn").onclick = async () => {
    if (!confirm("Remove all downloaded LoRAs from this browser?")) return;
    loras = [];
    saveLoras();
    renderLoras();
    await clearLoras();
    $("settings").close();
  };
  $("clearBtn").onclick = async () => {
    if (!confirm("Remove the downloaded model files from this browser? You'll need to download again to generate images.")) return;
    $("settings").close();
    await pipe.unload();
    await clearCache();
    ensureModel();
  };

  chipText();
  await ensureModel();
}

init();
