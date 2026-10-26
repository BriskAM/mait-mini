// Image helpers: canvas <-> 64x64 RGB arrays, controlled corruptions (matching the notebook's stress tests), Grad-CAM overlay.

export const SIZE = 64;

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Reads a 64x64 canvas into a Uint8ClampedArray of RGB triples (alpha dropped).
export function canvasToRGB(canvas) {
  const d = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, SIZE, SIZE).data, out = new Uint8ClampedArray(SIZE * SIZE * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { out[j] = d[i]; out[j + 1] = d[i + 1]; out[j + 2] = d[i + 2]; }
  return out;
}

export function rgbToImageData(rgb) {
  const img = new ImageData(SIZE, SIZE);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) { img.data[i] = rgb[j]; img.data[i + 1] = rgb[j + 1]; img.data[i + 2] = rgb[j + 2]; img.data[i + 3] = 255; }
  return img;
}

// Draws any image/video/canvas into a 64x64 canvas, centre-cropped to a square, and returns the RGB array.
export function fitToModelInput(source, target) {
  const w = source.videoWidth || source.naturalWidth || source.width, h = source.videoHeight || source.naturalHeight || source.height, s = Math.min(w, h);
  const ctx = target.getContext("2d", { willReadFrequently: true }); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, (w - s) / 2, (h - s) / 2, s, s, 0, 0, SIZE, SIZE);
  return canvasToRGB(target);
}

// ---- corruptions: same families and units as the notebook's robustness test --------------------------------------------
export const corrupt = {
  brightness(rgb, f) { const o = new Uint8ClampedArray(rgb.length); for (let i = 0; i < rgb.length; i++) o[i] = rgb[i] * f; return o; },
  noise(rgb, sigma, seed = 7) {
    const r = mulberry32(seed), o = new Uint8ClampedArray(rgb.length);
    for (let i = 0; i < rgb.length; i += 2) {
      const u = Math.max(r(), 1e-9), v = r(), m = Math.sqrt(-2 * Math.log(u)) * sigma;
      o[i] = rgb[i] + m * Math.cos(2 * Math.PI * v); if (i + 1 < rgb.length) o[i + 1] = rgb[i + 1] + m * Math.sin(2 * Math.PI * v);
    }
    return o;
  },
  // Gaussian blur with an odd kernel size k (sigma derived as OpenCV does); reflected borders.
  blur(rgb, k) {
    if (k <= 1) return rgb;
    const sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8, half = k >> 1, w = new Float32Array(k); let s = 0;
    for (let i = 0; i < k; i++) { w[i] = Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)); s += w[i]; }
    for (let i = 0; i < k; i++) w[i] /= s;
    const refl = (i) => (i < 0 ? -i : i >= SIZE ? 2 * SIZE - 2 - i : i), tmp = new Float32Array(rgb.length), out = new Uint8ClampedArray(rgb.length);
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) for (let c = 0; c < 3; c++) { let a = 0; for (let i = 0; i < k; i++) a += w[i] * rgb[(y * SIZE + refl(x + i - half)) * 3 + c]; tmp[(y * SIZE + x) * 3 + c] = a; }
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) for (let c = 0; c < 3; c++) { let a = 0; for (let i = 0; i < k; i++) a += w[i] * tmp[(refl(y + i - half) * SIZE + x) * 3 + c]; out[(y * SIZE + x) * 3 + c] = a; }
    return out;
  },
  // Rotation about the centre; edge pixels are replicated, as with cv2.BORDER_REPLICATE.
  rotate(rgb, deg) {
    if (!deg) return rgb;
    const a = (-deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a), c0 = (SIZE - 1) / 2, o = new Uint8ClampedArray(rgb.length);
    const at = (x, y, ch) => rgb[(Math.min(SIZE - 1, Math.max(0, y)) * SIZE + Math.min(SIZE - 1, Math.max(0, x))) * 3 + ch];
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const sx = cos * (x - c0) - sin * (y - c0) + c0, sy = sin * (x - c0) + cos * (y - c0) + c0, x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      for (let ch = 0; ch < 3; ch++) o[(y * SIZE + x) * 3 + ch] = (1 - fy) * ((1 - fx) * at(x0, y0, ch) + fx * at(x0 + 1, y0, ch)) + fy * ((1 - fx) * at(x0, y0 + 1, ch) + fx * at(x0 + 1, y0 + 1, ch));
    }
    return o;
  },
  // Black square covering `frac` of the image area, at a fixed pseudo-random position.
  occlude(rgb, frac, seed = 11) {
    if (frac <= 0) return rgb;
    const side = Math.round(SIZE * Math.sqrt(frac)), r = mulberry32(seed), x0 = Math.floor(r() * (SIZE - side + 1)), y0 = Math.floor(r() * (SIZE - side + 1)), o = Uint8ClampedArray.from(rgb);
    for (let y = y0; y < y0 + side; y++) for (let x = x0; x < x0 + side; x++) { const i = (y * SIZE + x) * 3; o[i] = o[i + 1] = o[i + 2] = 0; }
    return o;
  },
};

export const CORRUPTIONS = {
  rotate: { label: "Rotation", unit: "°", min: -45, max: 45, step: 1, neutral: 0, apply: (r, v) => corrupt.rotate(r, v) },
  brightness: { label: "Brightness", unit: "×", min: 0.2, max: 2.5, step: 0.05, neutral: 1, apply: (r, v) => corrupt.brightness(r, v) },
  blur: { label: "Blur (kernel)", unit: "", min: 1, max: 11, step: 2, neutral: 1, apply: (r, v) => corrupt.blur(r, v) },
  noise: { label: "Noise (σ)", unit: "", min: 0, max: 60, step: 1, neutral: 0, apply: (r, v) => (v > 0 ? corrupt.noise(r, v) : r) },
  occlude: { label: "Occlusion (area)", unit: "%", min: 0, max: 40, step: 1, neutral: 0, apply: (r, v) => corrupt.occlude(r, v / 100) },
};

// ---- Grad-CAM overlay ---------------------------------------------------------------------------------------------------
function jet(t) {
  const c = (v) => Math.max(0, Math.min(1, v));
  return [255 * c(1.5 - Math.abs(4 * t - 3)), 255 * c(1.5 - Math.abs(4 * t - 2)), 255 * c(1.5 - Math.abs(4 * t - 1))];
}

// Draws the 64x64 input enlarged onto `canvas`, with the coarse Grad-CAM map blended on top (bilinear upsampling, 45% opacity).
export function drawOverlay(canvas, rgb, cam, side, alpha = 0.45) {
  const n = canvas.width, base = document.createElement("canvas"); base.width = base.height = SIZE;
  base.getContext("2d").putImageData(rgbToImageData(rgb), 0, 0);
  const ctx = canvas.getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(base, 0, 0, n, n);
  if (!cam) return;
  const img = ctx.getImageData(0, 0, n, n), d = img.data, at = (x, y) => cam[Math.min(side - 1, Math.max(0, y)) * side + Math.min(side - 1, Math.max(0, x))];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const fx = ((x + 0.5) / n) * side - 0.5, fy = ((y + 0.5) / n) * side - 0.5, x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
    const v = (1 - ty) * ((1 - tx) * at(x0, y0) + tx * at(x0 + 1, y0)) + ty * ((1 - tx) * at(x0, y0 + 1) + tx * at(x0 + 1, y0 + 1)), [r, g, b] = jet(v), i = (y * n + x) * 4;
    d[i] = (1 - alpha) * d[i] + alpha * r; d[i + 1] = (1 - alpha) * d[i + 1] + alpha * g; d[i + 2] = (1 - alpha) * d[i + 2] + alpha * b;
  }
  ctx.putImageData(img, 0, 0);
}
