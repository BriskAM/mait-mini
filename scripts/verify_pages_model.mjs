#!/usr/bin/env node
// Checks that the in-browser CNN (site/js/nn.js) reproduces Keras: probabilities and Grad-CAM maps for the sample images.
//
//   node scripts/verify_pages_model.mjs --ref ref.json --tiles tiles.u8 [--site site]
//
// ref.json   {"probs": [[43 floats] ...], "cams": [[16x16 maps] ...], "cams_logit": [...]} produced by running the Keras model on the sample tiles
// tiles.u8   the same sample tiles as raw uint8 (N x 64 x 64 x 3), in the order of site/data/samples.json
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const site = arg("--site", "site"), refPath = arg("--ref"), tilesPath = arg("--tiles");
if (!refPath || !tilesPath) { console.error("usage: verify_pages_model.mjs --ref ref.json --tiles tiles.u8 [--site site]"); process.exit(2); }

const { SignNet } = await import(pathToFileURL(path.resolve(site, "js/nn.js")).href);
const manifest = JSON.parse(fs.readFileSync(path.join(site, "data/cnn_manifest.json"), "utf8"));
const weights = fs.readFileSync(path.join(site, "data/cnn_weights.bin"));
const net = new SignNet(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
const ref = JSON.parse(fs.readFileSync(refPath, "utf8")), tiles = fs.readFileSync(tilesPath), per = 64 * 64 * 3;
const n = ref.probs.length;
let maxP = 0, maxLogit = 0, maxProb = 0, agree = 0, skipped = 0, t0 = performance.now();
for (let k = 0; k < n; k++) {
  const rgb = new Uint8Array(tiles.buffer, tiles.byteOffset + k * per, per);
  const lg = net.gradcam(rgb, null, "logit"), pr = net.gradcam(rgb, null, "prob"), probs = lg.probs;
  for (let j = 0; j < 43; j++) maxP = Math.max(maxP, Math.abs(probs[j] - ref.probs[k][j]));
  const rl = ref.cams_logit[k].flat(), rp = ref.cams[k].flat();
  for (let i = 0; i < lg.cam.length; i++) maxLogit = Math.max(maxLogit, Math.abs(lg.cam[i] - rl[i]));
  if (Math.max(...probs) < 0.9999) for (let i = 0; i < pr.cam.length; i++) maxProb = Math.max(maxProb, Math.abs(pr.cam[i] - rp[i]));   // saturated cases are numerically meaningless
  else skipped++;
  const a = (v) => v.indexOf(Math.max(...v)); if (a(Array.from(probs)) === a(ref.probs[k])) agree++;
}
const ms = (performance.now() - t0) / n / 2;
console.log(`${n} images | prediction agreement ${agree}/${n} | max |dp| = ${maxP.toExponential(2)} | max |dCAM| logit = ${maxLogit.toExponential(2)}, prob = ${maxProb.toExponential(2)} (${skipped} saturated images skipped for prob) | ${ms.toFixed(1)} ms per Grad-CAM (Node)`);
const ok = agree === n && maxP < 1e-4 && maxLogit < 1e-3 && maxProb < 1e-3;
console.log(ok ? "PASS" : "FAIL"); process.exit(ok ? 0 : 1);
