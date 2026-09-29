// Shared helpers: rotary tables (cos block followed by sin block, each [L][D/2]).

function table(gpu, angles, L, half) {
  const cs = new Float32Array(2 * L * half);
  for (let i = 0; i < L * half; i++) {
    cs[i] = Math.cos(angles[i]);
    cs[L * half + i] = Math.sin(angles[i]);
  }
  const t = gpu.fromArray(cs, [cs.length]);
  t.sinOff = L * half;
  return t;
}

// Standard 1D rope: angle(l, i) = l / theta^(2i/D)
export function ropeTable(gpu, L, D, theta) {
  const half = D / 2;
  const a = new Float32Array(L * half);
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < half; i++) a[l * half + i] = l * Math.fround(1 / Math.pow(theta, (2 * i) / D));
  }
  return table(gpu, a, L, half);
}

// Cosmos VideoRopePosition3DEmb for a single frame: 64 slots = 22 temporal (position 0),
// 21 height, 21 width, with NTK-scaled theta (extrapolation ratio 4 for h/w).
export function ditRopeTable(gpu, H, W) {
  return table(gpu, ditRopeAngles(H, W), H * W, 64);
}

// Angles [H*W][64] of ditRopeTable (also used by the WebNN backend).
export function ditRopeAngles(H, W) {
  const headDim = 128;
  const dimH = Math.floor(headDim / 6) * 2; // 42
  const dimT = headDim - 2 * dimH; // 44
  const hTheta = 10000 * Math.pow(4, dimH / (dimH - 2));
  const hf = [];
  for (let j = 0; j < dimH / 2; j++) hf.push(Math.fround(1 / Math.pow(hTheta, Math.fround((2 * j) / dimH))));
  const half = 64;
  const L = H * W;
  const a = new Float32Array(L * half);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * half;
      // slots 0..21 are temporal at t=0 -> angle 0
      for (let j = 0; j < 21; j++) {
        a[o + dimT / 2 + j] = y * hf[j];
        a[o + dimT / 2 + 21 + j] = x * hf[j];
      }
    }
  }
  return a;
}
