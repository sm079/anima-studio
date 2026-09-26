// Flow-matching samplers ported from ComfyUI's k-diffusion (comfy/k_diffusion/sampling.py).
// `denoise(x, sigma)` returns the x0 prediction. Latents are Float32Arrays.

import { TorchGenerator } from "./rng.js";

const timeShift = (shift, t) => (shift * t) / (1 + (shift - 1) * t);

// ModelSamplingDiscreteFlow sigma table (1000 steps, multiplier 1)
function sigmaTable(shift) {
  const table = [];
  for (let i = 1; i <= 1000; i++) table.push(Math.fround(timeShift(shift, i / 1000)));
  return table;
}

// ComfyUI "simple" scheduler
export function simpleSigmas(steps, shift) {
  const table = sigmaTable(shift);
  const ss = 1000 / steps;
  const out = [];
  for (let x = 0; x < steps; x++) out.push(table[table.length - 1 - Math.floor(x * ss)]);
  out.push(0);
  return out;
}

// ---- regularized incomplete beta and its inverse (for the beta scheduler)
function lnGamma(z) {
  // Lanczos approximation, g = 7
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < 9; i++) x += c[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function betacf(a, b, x) {
  // continued fraction (Numerical Recipes)
  const FPMIN = 1e-300;
  let c = 1, d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a - 1 + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + 1 + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

function betaCdf(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(lbt) * betacf(a, b, x)) / a;
  return 1 - (Math.exp(lbt) * betacf(b, a, 1 - x)) / b;
}

function betaPpf(p, a, b) {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0, hi = 1;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (betaCdf(mid, a, b) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ComfyUI "beta" scheduler (alpha = beta = 0.6): more steps at both ends of the schedule.
// Repeated timesteps are dropped, as in ComfyUI, so it can return fewer than `steps` steps.
function betaSigmas(steps, shift, alpha = 0.6, beta = 0.6) {
  const table = sigmaTable(shift);
  const out = [];
  let last = -1;
  for (let i = 0; i < steps; i++) {
    const t = Math.round(betaPpf(1 - i / steps, alpha, beta) * 999);
    if (t !== last) out.push(table[t]);
    last = t;
  }
  out.push(0);
  return out;
}

export const SCHEDULERS = { simple: simpleSigmas, beta: betaSigmas };

const axpby = (a, x, b, y) => {
  const o = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = a * x[i] + b * y[i];
  return o;
};

export async function sample(name, denoise, x, sigmas, { seed = 0, shift = 3, onStep } = {}) {
  const gen = new TorchGenerator(seed); // comfy default_noise_sampler: its own generator seeded with `seed`
  const noise = () => gen.randn(x.length);
  const n = sigmas.length - 1;

  if (name === "euler") {
    for (let i = 0; i < n; i++) {
      const s = sigmas[i], s1 = sigmas[i + 1];
      const den = await denoise(x, s, i);
      x = s1 === 0 ? den : axpby(s1 / s, x, 1 - s1 / s, den); // x + (x-den)/s*(s1-s)
      await onStep?.(i, x, den);
    }
    return x;
  }

  if (name === "euler_a") {
    // sample_euler_ancestral_RF, eta = 1
    for (let i = 0; i < n; i++) {
      const s = sigmas[i], s1 = sigmas[i + 1];
      const den = await denoise(x, s, i);
      if (s1 === 0) x = den;
      else {
        const sigmaDown = s1 * (s1 / s);
        const alphaIp1 = 1 - s1;
        const alphaDown = 1 - sigmaDown;
        const renoise = Math.sqrt(s1 * s1 - (sigmaDown * sigmaDown * alphaIp1 * alphaIp1) / (alphaDown * alphaDown));
        const r = sigmaDown / s;
        x = axpby(r, x, 1 - r, den);
        const eps = noise();
        const k = alphaIp1 / alphaDown;
        for (let j = 0; j < x.length; j++) x[j] = k * x[j] + eps[j] * renoise;
      }
      await onStep?.(i, x, den);
    }
    return x;
  }

  if (name === "dpmpp_2m") {
    const t = (sg) => -Math.log(sg);
    let old = null;
    for (let i = 0; i < n; i++) {
      const s = sigmas[i], s1 = sigmas[i + 1];
      const den = await denoise(x, s, i);
      if (s1 === 0) x = den;
      else {
        const h = t(s1) - t(s);
        const ratio = s1 / s;
        const c = -Math.expm1(-h);
        if (old === null) x = axpby(ratio, x, c, den);
        else {
          const hLast = t(s) - t(sigmas[i - 1]);
          const r = hLast / h;
          const dd = axpby(1 + 1 / (2 * r), den, -1 / (2 * r), old);
          x = axpby(ratio, x, c, dd);
        }
      }
      old = den;
      await onStep?.(i, x, den);
    }
    return x;
  }

  if (name === "er_sde") {
    // VP ER-SDE-Solver-3 for CONST (flow) models
    const sg = sigmas.slice();
    if (sg[0] >= 1) sg[0] = Math.fround(timeShift(shift, 1 - 1e-4));
    const lam = sg.map((v) => (v === 0 ? 0 : v / (1 - v))); // er_lambda = sigma / alpha
    const scaler = (v) => v * (Math.exp(Math.pow(v, 0.3)) + 10);
    const P = 200;
    let old = null, oldD = null;
    for (let i = 0; i < n; i++) {
      const den = await denoise(x, sg[i], i);
      const stage = Math.min(3, i + 1);
      if (sg[i + 1] === 0) x = den;
      else {
        const ls = lam[i], lt = lam[i + 1];
        const alphaS = sg[i] / ls, alphaT = sg[i + 1] / lt;
        const r = scaler(lt) / scaler(ls);
        x = axpby((alphaT / alphaS) * r, x, alphaT * (1 - r), den);
        if (stage >= 2) {
          const dt = lt - ls;
          const step = -dt / P;
          let s = 0, su = 0;
          for (let p = 0; p < P; p++) {
            const pos = lt + p * step;
            const sp = scaler(pos);
            s += 1 / sp;
            su += (pos - ls) / sp;
          }
          s *= step;
          su *= step;
          const dD = axpby(1 / (ls - lam[i - 1]), den, -1 / (ls - lam[i - 1]), old);
          const c2 = alphaT * (dt + s * scaler(lt));
          for (let j = 0; j < x.length; j++) x[j] += c2 * dD[j];
          if (stage >= 3) {
            const dU = axpby(2 / (ls - lam[i - 2]), dD, -2 / (ls - lam[i - 2]), oldD);
            const c3 = alphaT * ((dt * dt) / 2 + su * scaler(lt));
            for (let j = 0; j < x.length; j++) x[j] += c3 * dU[j];
          }
          oldD = dD;
        }
        const nz = Math.sqrt(Math.max(0, lt * lt - ls * ls * r * r));
        if (nz > 0) {
          const eps = noise();
          for (let j = 0; j < x.length; j++) x[j] += alphaT * eps[j] * nz;
        }
      }
      old = den;
      await onStep?.(i, x, den);
    }
    return x;
  }

  throw new Error(`unknown sampler ${name}`);
}
