// WebNN (Web Neural Network API) feature detection, usable from the page and from workers.
//
// Chrome ships WebNN behind a flag / origin trial: chrome://flags/#web-machine-learning-neural-network
// (or --enable-features=WebMachineLearningNeuralNetwork). See README "Optional: WebNN backend".

export const WEBNN_FLAG = "chrome://flags/#web-machine-learning-neural-network";

export function hasWebNN() {
  return typeof navigator !== "undefined" && !!navigator.ml && typeof MLGraphBuilder !== "undefined";
}

export async function createWebNNContext() {
  if (!hasWebNN()) throw new Error(`WebNN is not enabled in this browser (turn on ${WEBNN_FLAG})`);
  // deviceType is ignored by current Chrome builds (powerPreference picks the GPU) but older ones need it
  return navigator.ml.createContext({ powerPreference: "high-performance", deviceType: "gpu" });
}

const types = (limits, op, arg) => limits?.[op]?.[arg]?.dataTypes || [];

// What the WebNN backend will use on this device:
//   float16: fp16 matmuls/attention are supported (tensor cores on most GPUs)
//   ct:      the compute type for linears and attention ("float16" unless unsupported or not wanted)
//   int8:    int8 weights can be dequantized inside the graph (keeps quantized weights 8-bit in memory)
export function capabilities(context, prefer = "float16") {
  const L = context.opSupportLimits?.() || {};
  const float16 = types(L, "matmul", "a").includes("float16") && types(L, "softmax", "input").includes("float16");
  const ct = prefer === "float16" && float16 ? "float16" : "float32";
  const int8 = types(L, "dequantizeLinear", "input").includes("int8")
    && types(L, "dequantizeLinear", "scale").includes(ct)
    && (!L.constant?.dataTypes || L.constant.dataTypes.includes("int8"));
  return { float16, ct, int8 };
}

// For the settings UI: { ok, float16?, reason? }
export async function probeWebNN() {
  if (!hasWebNN()) return { ok: false, reason: "missing" };
  try {
    const ctx = await createWebNNContext();
    const caps = capabilities(ctx);
    ctx.destroy?.();
    return { ok: true, ...caps };
  } catch (e) {
    return { ok: false, reason: e.message || String(e) };
  }
}
