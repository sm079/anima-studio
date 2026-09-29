// Reads safetensors files (from a Blob/File, e.g. the OPFS cache) and uploads weights.
//
// Linear weight objects: { kind: "bf16" | "i8" | "w4", N, K, buf, scale?, srel?, codebook?, bias? }
// Quantized kinds follow the ComfyUI comfy_quant layout written by tools/build_assets.py.

const BYTES = { F32: 4, BF16: 2, F16: 2, I8: 1, U8: 1, F8_E4M3: 1 };

export function bf16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  return out;
}

function f16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class SafeTensors {
  static async open(blob, prefix = "") {
    const head = new DataView(await blob.slice(0, 8).arrayBuffer());
    const n = Number(head.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(await blob.slice(8, 8 + n).arrayBuffer()));
    delete header.__metadata__;
    return new SafeTensors(blob, header, 8 + n, prefix);
  }

  constructor(blob, header, dataStart, prefix) {
    this.blob = blob;
    this.header = header;
    this.dataStart = dataStart;
    this.prefix = prefix;
  }

  has(name) {
    return this.prefix + name in this.header;
  }

  info(name) {
    const t = this.header[this.prefix + name];
    if (!t) throw new Error(`missing tensor ${this.prefix + name}`);
    return t;
  }

  async bytes(name) {
    const t = this.info(name);
    const [a, b] = t.data_offsets;
    return new Uint8Array(await this.blob.slice(this.dataStart + a, this.dataStart + b).arrayBuffer());
  }

  async f32(name) {
    const t = this.info(name);
    const u8 = await this.bytes(name);
    if (t.dtype === "F32") return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
    if (t.dtype === "BF16") return bf16ToF32(u8);
    if (t.dtype === "F16") return f16ToF32(u8);
    // other types show up in small tensors, e.g. LoRA alphas saved as integers
    const buf = u8.slice().buffer; // copy: typed-array views need an aligned offset
    const other = { F64: Float64Array, I64: BigInt64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array }[t.dtype];
    if (other) return Float32Array.from(new other(buf), Number);
    throw new Error(`unsupported dtype ${t.dtype} for ${name}`);
  }

  // Gather rows of a 2D table (embeddings) without reading the whole tensor.
  async rows(name, ids) {
    const base = name.slice(0, -"weight".length);
    if (this.has(base + "comfy_quant")) return this.int8Rows(base, ids);
    const t = this.info(name);
    const [, dim] = t.shape;
    const rowBytes = dim * BYTES[t.dtype];
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * rowBytes;
      const u8 = new Uint8Array(await this.blob.slice(off, off + rowBytes).arrayBuffer());
      out.set(t.dtype === "BF16" ? bf16ToF32(u8) : t.dtype === "F16" ? f16ToF32(u8) : new Float32Array(u8.buffer), i * dim);
    }));
    return out;
  }

  // Per-row int8 table (tools/quant.py quantize_int8_rows): value = q * weight_scale[row].
  async int8Rows(base, ids) {
    const t = this.info(base + "weight");
    if (t.dtype !== "I8") throw new Error(`unsupported quantized embedding ${base}`);
    const dim = t.shape[1];
    const scales = this._scales?.[base] || (this._scales = { ...this._scales, [base]: await this.f32(base + "weight_scale") })[base];
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * dim;
      const q = new Int8Array(await this.blob.slice(off, off + dim).arrayBuffer());
      const s = scales[id];
      for (let j = 0; j < dim; j++) out[i * dim + j] = q[j] * s;
    }));
    return out;
  }

  async vector(gpu, name) {
    return gpu.upload(await this.f32(name));
  }

  // Loads "<base>weight" (+ comfy_quant companions) and optional "<base>bias". The result
  // carries its module name (base without the trailing dot) so LoRAs can find it.
  async linear(gpu, base) {
    const W = await this.linearWeights(gpu, base);
    W.name = base.replace(/\.$/, "");
    return W;
  }

  async linearWeights(gpu, base) {
    const w = this.info(base + "weight");
    const bias = this.has(base + "bias") ? await this.vector(gpu, base + "bias") : null;
    if (this.has(base + "comfy_quant")) {
      const meta = JSON.parse(new TextDecoder().decode(await this.bytes(base + "comfy_quant")));
      const N = w.shape[0];
      if (!meta.convrot) throw new Error(`${base}: only ConvRot-quantized linears are supported`);
      if (meta.format === "int8_tensorwise") {
        return { kind: "i8", N, K: w.shape[1], buf: gpu.upload(await this.bytes(base + "weight")), scale: await this.vector(gpu, base + "weight_scale"), bias };
      }
      if (meta.format === "asym_w4a8_int8") {
        const codebookData = await this.f32(base + "weight_codebook");
        return {
          kind: "w4", N, K: w.shape[1] * 2,
          buf: gpu.upload(await this.bytes(base + "weight")),
          srel: gpu.upload(await this.bytes(base + "weight_s_rel")),
          scale: await this.vector(gpu, base + "weight_s_channel"),
          codebook: gpu.upload(codebookData),
          codebookData,
          bias,
        };
      }
      throw new Error(`unsupported quant format ${meta.format}`);
    }
    if (w.dtype !== "BF16") throw new Error(`expected bf16 weight for ${base}, got ${w.dtype}`);
    const N = w.shape[0];
    const K = w.shape.slice(1).reduce((a, b) => a * b, 1);
    return { kind: "bf16", N, K, buf: gpu.upload(await this.bytes(base + "weight")), bias };
  }
}

// Stack linears that share an input into one [sum N, K] weight (e.g. q, k, v -> qkv), so one
// GEMM replaces several. Rows are independent in every format, so per-row scales concatenate;
// w4 additionally needs one shared codebook. Returns null when they can't be merged.
export function concatLinears(gpu, Ws) {
  const { kind, K } = Ws[0];
  if (!Ws.every((w) => w.kind === kind && w.K === K && !w.bias)) return null;
  if (kind === "w4") {
    const cb = Ws[0].codebookData;
    if (!Ws.every((w) => w.codebookData.every((v, i) => v === cb[i]))) return null;
  }
  const bytes = {
    buf: (n) => (kind === "bf16" ? n * K * 2 : kind === "i8" ? n * K : n * K / 2),
    scale: (n) => n * 4,
    srel: (n) => (n * K) / 16,
  };
  const fields = kind === "bf16" ? ["buf"] : kind === "i8" ? ["buf", "scale"] : ["buf", "scale", "srel"];
  const out = { kind, K, N: Ws.reduce((a, w) => a + w.N, 0), parts: [] };
  let colOff = 0;
  for (const w of Ws) { out.parts.push({ name: w.name, off: colOff, n: w.N }); colOff += w.N; }
  const enc = gpu.device.createCommandEncoder();
  for (const f of fields) {
    const total = Ws.reduce((a, w) => a + bytes[f](w.N), 0);
    const dst = gpu.device.createBuffer({ size: Math.ceil(total / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let off = 0;
    for (const w of Ws) {
      enc.copyBufferToBuffer(w[f], 0, dst, off, bytes[f](w.N));
      off += bytes[f](w.N);
    }
    out[f] = dst;
  }
  if (kind === "w4") {
    out.codebook = Ws[0].codebook;
    out.codebookData = Ws[0].codebookData;
  }
  gpu.flush();
  gpu.device.queue.submit([enc.finish()]);
  // the sources are no longer needed once the copies have run (destroy waits for the queue)
  for (const w of Ws) for (const f of fields) w[f].destroy();
  for (const w of Ws.slice(1)) if (kind === "w4") w.codebook.destroy();
  return out;
}
