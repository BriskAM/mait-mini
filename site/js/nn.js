// Dependency-free inference for the custom CNN, including Grad-CAM.
// Architecture (identical to the notebook's `build_custom_cnn`):
//   raw 0-255 pixels -> /255 -> 3 x [Conv3x3 same + ReLU -> BatchNorm -> MaxPool2x2] (32, 64, 128 filters)
//   -> global average pool -> Dense128 ReLU -> Dense43 softmax.   Augmentation and dropout are inactive at inference.
// Tensors are Float32Array in NHWC order (one image at a time).

export const NUM_CLASSES = 43;

function conv3x3(inp, H, W, Cin, w, b, Cout) {
  const out = new Float32Array(H * W * Cout);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * Cout;
      for (let co = 0; co < Cout; co++) out[o + co] = b[co];
      for (let ky = 0; ky < 3; ky++) {
        const yy = y + ky - 1;
        if (yy < 0 || yy >= H) continue;
        for (let kx = 0; kx < 3; kx++) {
          const xx = x + kx - 1;
          if (xx < 0 || xx >= W) continue;
          const ib = (yy * W + xx) * Cin, wb = (ky * 3 + kx) * Cin * Cout;
          for (let ci = 0; ci < Cin; ci++) {
            const v = inp[ib + ci];
            if (v === 0) continue;
            const wo = wb + ci * Cout;
            for (let co = 0; co < Cout; co++) out[o + co] += v * w[wo + co];
          }
        }
      }
      for (let co = 0; co < Cout; co++) if (out[o + co] < 0) out[o + co] = 0; // ReLU
    }
  }
  return out;
}

function batchNorm(x, C, scale, shift) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { const c = i % C; out[i] = x[i] * scale[c] + shift[c]; }
  return out;
}

// 2x2 max pool, stride 2. Returns the pooled tensor and, for each output element, the index of the winning input element.
function maxPool(x, H, W, C) {
  const h = H >> 1, w = W >> 1, out = new Float32Array(h * w * C), arg = new Int32Array(h * w * C);
  for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) for (let c = 0; c < C; c++) {
    let best = -Infinity, bi = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const idx = ((2 * y + dy) * W + (2 * xx + dx)) * C + c;
      if (x[idx] > best) { best = x[idx]; bi = idx; }
    }
    const o = (y * w + xx) * C + c; out[o] = best; arg[o] = bi;
  }
  return { out, arg };
}

function softmax(z) {
  let m = -Infinity; for (const v of z) if (v > m) m = v;
  const e = new Float32Array(z.length); let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - m); s += e[i]; }
  for (let i = 0; i < z.length; i++) e[i] /= s;
  return e;
}

export class SignNet {
  constructor(manifest, buffer) {
    const all = new Float32Array(buffer), p = {};
    for (const e of manifest.entries) p[e.name] = all.subarray(e.offset, e.offset + e.shape.reduce((a, b) => a * b, 1));
    this.p = p; this.size = manifest.input[0];
    const eps = manifest.bn_epsilon; this.bn = {};
    for (const i of [1, 2, 3]) {
      const g = p[`bn${i}.gamma`], be = p[`bn${i}.beta`], m = p[`bn${i}.mean`], v = p[`bn${i}.var`];
      const scale = new Float32Array(g.length), shift = new Float32Array(g.length);
      for (let c = 0; c < g.length; c++) { scale[c] = g[c] / Math.sqrt(v[c] + eps); shift[c] = be[c] - m[c] * scale[c]; }
      this.bn[i] = { scale, shift };
    }
  }

  static async load(base = "data/") {
    const [manifest, buf] = await Promise.all([fetch(base + "cnn_manifest.json").then(r => r.json()), fetch(base + "cnn_weights.bin").then(r => r.arrayBuffer())]);
    return new SignNet(manifest, buf);
  }

  // rgb: array of length 64*64*3 holding 0-255 pixel values (HWC). Returns probabilities, plus cached activations if keep is true.
  forward(rgb, keep = false) {
    const S = this.size, p = this.p;
    let x = new Float32Array(rgb.length); for (let i = 0; i < rgb.length; i++) x[i] = rgb[i] / 255;
    const stages = [];
    let H = S, Cin = 3;
    for (const [i, Cout] of [[1, 32], [2, 64], [3, 128]]) {
      const a = conv3x3(x, H, H, Cin, p[`conv${i}.w`], p[`conv${i}.b`], Cout);        // conv + ReLU
      const bn = batchNorm(a, Cout, this.bn[i].scale, this.bn[i].shift);
      const pooled = maxPool(bn, H, H, Cout);
      stages.push({ a, bn, pooled, H, C: Cout });
      x = pooled.out; H >>= 1; Cin = Cout;
    }
    const n = H * H, g = new Float32Array(Cin);                                         // global average pool
    for (let i = 0; i < n; i++) for (let c = 0; c < Cin; c++) g[c] += x[i * Cin + c];
    for (let c = 0; c < Cin; c++) g[c] /= n;
    const h1pre = new Float32Array(128);
    for (let o = 0; o < 128; o++) { let s = p["dense1.b"][o]; for (let i = 0; i < 128; i++) s += g[i] * p["dense1.w"][i * 128 + o]; h1pre[o] = s; }
    const h1 = h1pre.map(v => (v > 0 ? v : 0));
    const z = new Float32Array(NUM_CLASSES);
    for (let j = 0; j < NUM_CLASSES; j++) { let s = p["dense2.b"][j]; for (let i = 0; i < 128; i++) s += h1[i] * p["dense2.w"][i * NUM_CLASSES + j]; z[j] = s; }
    const probs = softmax(z);
    return keep ? { probs, stages, h1pre, n } : { probs };
  }

  // Grad-CAM: gradient of the class score w.r.t. the last conv layer's (post-ReLU) output,
  // averaged over positions to weight the channels, summed, ReLU'd and scaled to [0, 1]. Returns a 16x16 map.
  //   score = "logit" (default): the class score before softmax, as in the original Grad-CAM paper. Stays well-conditioned when the
  //                               model is ~100% sure.
  //   score = "prob":             the post-softmax probability, as the notebook does. Its gradient vanishes at saturation (p -> 1), so
  //                               the normalised map is then dominated by float rounding noise.
  gradcam(rgb, target = null, score = "logit") {
    const p = this.p, { probs, stages, h1pre, n } = this.forward(rgb, true);
    let cls = target; if (cls === null) { cls = 0; for (let j = 1; j < NUM_CLASSES; j++) if (probs[j] > probs[cls]) cls = j; }
    const dz = new Float32Array(NUM_CLASSES);                                            // gradient of the score w.r.t. the logits
    for (let j = 0; j < NUM_CLASSES; j++) dz[j] = score === "prob" ? probs[cls] * ((j === cls ? 1 : 0) - probs[j]) : (j === cls ? 1 : 0);
    const dh1pre = new Float32Array(128);
    for (let i = 0; i < 128; i++) { if (h1pre[i] <= 0) continue; let s = 0; for (let j = 0; j < NUM_CLASSES; j++) s += p["dense2.w"][i * NUM_CLASSES + j] * dz[j]; dh1pre[i] = s; }
    const dg = new Float32Array(128);
    for (let i = 0; i < 128; i++) { let s = 0; for (let o = 0; o < 128; o++) s += p["dense1.w"][i * 128 + o] * dh1pre[o]; dg[i] = s; }
    const last = stages[2], C = last.C, H = last.H;
    const dB = new Float32Array(H * H * C);                                              // back through pool and global average pool
    for (let o = 0; o < last.pooled.arg.length; o++) dB[last.pooled.arg[o]] += dg[o % C] / n;
    const alpha = new Float32Array(C);                                                   // d(BN out)/dA = scale, then mean over positions
    for (let i = 0; i < dB.length; i++) { const c = i % C; alpha[c] += dB[i] * this.bn[3].scale[c]; }
    for (let c = 0; c < C; c++) alpha[c] /= H * H;
    const cam = new Float32Array(H * H); let mx = 0;
    for (let i = 0; i < H * H; i++) { let s = 0; for (let c = 0; c < C; c++) s += alpha[c] * last.a[i * C + c]; cam[i] = s > 0 ? s : 0; if (cam[i] > mx) mx = cam[i]; }
    for (let i = 0; i < cam.length; i++) cam[i] /= mx + 1e-8;
    return { cam, side: H, cls, prob: probs[cls], probs };
  }
}

export function topK(probs, k = 5) {
  return Array.from(probs, (p, i) => ({ id: i, p })).sort((a, b) => b.p - a.p).slice(0, k);
}
