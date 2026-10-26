// Live demo: classify a sample, an uploaded photo or a camera frame in the browser, explain it with Grad-CAM,
// and stress it with controlled corruptions. Everything runs locally; nothing is uploaded.
import { SignNet, topK } from "./nn.js";
import { SIZE, canvasToRGB, drawOverlay, rgbToImageData, CORRUPTIONS } from "./imaging.js";
import { $, $$, h, fmt, lineChart } from "./charts.js";
import { getThreshold, setThreshold, onThreshold } from "./state.js";

const loadImage = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
const short = (n) => n.replace("End of speed limit (80km/h)", "End of limit 80").replace("Speed limit (", "Limit ").replace("km/h)", " km/h");

export async function initDemo(names) {
  const status = $("#demo-status");
  const [net, sprite, meta] = await Promise.all([SignNet.load("data/"), loadImage("data/samples.png"), fetch("data/samples.json").then((r) => r.json())]);
  status.textContent = "Model ready. It runs entirely in your browser."; status.dataset.ready = "1";

  const st = { base: null, truth: null, hardKnown: false, vals: {}, cam: true, source: null, box: null, live: false, raf: 0, selectedTile: null };
  for (const [k, c] of Object.entries(CORRUPTIONS)) st.vals[k] = c.neutral;

  const tileCanvas = h("canvas", { width: SIZE, height: SIZE }), tctx = tileCanvas.getContext("2d", { willReadFrequently: true });
  const tileRGB = (i) => { const m = meta.items[i], c = meta.cols; tctx.clearRect(0, 0, SIZE, SIZE); tctx.drawImage(sprite, (m.tile % c) * SIZE, Math.floor(m.tile / c) * SIZE, SIZE, SIZE, 0, 0, SIZE, SIZE); return canvasToRGB(tileCanvas); };

  // ---- core: corrupt -> predict -> render -------------------------------------------------------------------------------
  const chain = (rgb, vals = st.vals) => { let x = rgb; for (const k of ["rotate", "brightness", "blur", "noise", "occlude"]) if (vals[k] !== CORRUPTIONS[k].neutral) x = CORRUPTIONS[k].apply(x, vals[k]); return x; };
  const inputCanvas = $("#view-input"), camCanvas = $("#view-cam");
  let pending = false;
  const schedule = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; render(); }); };

  function render() {
    if (!st.base) return;
    const rgb = chain(st.base), res = st.cam ? net.gradcam(rgb, null, "logit") : { ...net.forward(rgb), cls: null };
    const probs = res.probs, top = topK(probs, 5), t = getThreshold(), conf = top[0].p, answers = conf >= t;
    drawOverlay(inputCanvas, rgb, null, 0); camCanvas.hidden = !st.cam; if (st.cam) drawOverlay(camCanvas, rgb, res.cam, res.side);
    $("#pred-name").textContent = short(names[top[0].id]); $("#pred-conf").textContent = fmt.pct(conf, 1);
    const badge = $("#pred-badge"); badge.className = "badge " + (answers ? "ok" : "warn"); badge.replaceChildren(answers ? icon("check") : icon("pause"), answers ? "Would answer" : "Would defer to a human");
    $("#meter-fill").style.width = conf * 100 + "%"; $("#meter-thr").style.left = t * 100 + "%"; $("#meter-thr").title = `Threshold ${t.toFixed(3)}`;
    $("#pred-bars").replaceChildren(...top.map((r, i) => h("div", { class: "pbar" + (i === 0 ? " top" : "") }, h("span", { class: "pname" }, short(names[r.id])), h("span", { class: "ptrack" }, h("i", { style: { width: Math.max(0.5, r.p * 100) + "%" } })), h("b", {}, r.p < 0.001 ? "<0.1%" : fmt.pct(r.p, 1)))));
    const tr = $("#truth");
    if (st.truth == null) { tr.hidden = true; } else {
      tr.hidden = false; const ok = top[0].id === st.truth;
      tr.className = "truth " + (ok ? "ok" : "bad"); tr.replaceChildren(ok ? icon("check") : icon("cross"), h("span", {}, `Ground truth: ${short(names[st.truth])}. `, ok ? "Prediction matches." : "Prediction differs.", st.hardKnown ? " The Keras model from the notebook makes this same mistake." : ""));
    }
    $("#cam-note").textContent = st.cam ? "Warm colours mark regions that pushed the model toward this class." : "";
  }

  function icon(kind) {
    const d = { check: "M3 8.5l3.2 3.2L13 5", cross: "M4 4l8 8M12 4l-8 8", pause: "M5 3v10M11 3v10" }[kind];
    const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg"), p = document.createElementNS(NS, "path");
    svg.setAttribute("viewBox", "0 0 16 16"); svg.setAttribute("width", 14); svg.setAttribute("height", 14); svg.setAttribute("aria-hidden", "true");
    p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", "currentColor"); p.setAttribute("stroke-width", 2); p.setAttribute("stroke-linecap", "round"); p.setAttribute("stroke-linejoin", "round");
    svg.append(p); return svg;
  }

  const setBase = (rgb, truth = null, hard = false) => { st.base = rgb; st.truth = truth; st.hardKnown = hard; schedule(); };

  // ---- samples -----------------------------------------------------------------------------------------------------------
  const strip = $("#thumbs"), classSel = $("#class-select");
  meta.items.forEach((m, i) => {
    const c = h("canvas", { width: SIZE, height: SIZE, class: "thumb" + (m.hard ? " hard" : ""), tabindex: 0, role: "button", "aria-label": `Sample ${i + 1}: ${names[m.true]}${m.hard ? " (hard case)" : ""}` });
    c.getContext("2d").drawImage(sprite, (m.tile % meta.cols) * SIZE, Math.floor(m.tile / meta.cols) * SIZE, SIZE, SIZE, 0, 0, SIZE, SIZE);
    const pick = () => { $$(".thumb.sel", strip).forEach((x) => x.classList.remove("sel")); c.classList.add("sel"); st.selectedTile = i; useTab("samples"); setBase(tileRGB(i), m.true, m.hard); };
    c.addEventListener("click", pick); c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    c.pick = pick; strip.append(c);
  });
  classSel.append(h("option", { value: "" }, "Any class"), ...names.map((n, i) => h("option", { value: i }, `${i}. ${short(n)}`)));
  const randomPick = (filter) => { const pool = meta.items.map((m, i) => i).filter((i) => filter(meta.items[i])); strip.children[pool[Math.floor(Math.random() * pool.length)]].pick(); };
  $("#btn-random").addEventListener("click", () => randomPick((m) => classSel.value === "" || m.true === +classSel.value));
  $("#btn-hard").addEventListener("click", () => randomPick((m) => m.hard));
  classSel.addEventListener("change", () => { if (classSel.value !== "") randomPick((m) => m.true === +classSel.value); });
  $("#hard-count").textContent = meta.items.filter((m) => m.hard).length;

  // ---- tabs --------------------------------------------------------------------------------------------------------------
  function useTab(name) {
    for (const t of $$("[data-tab]")) { const on = t.dataset.tab === name; t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1; $("#pane-" + t.dataset.tab).hidden = !on; }
    if (name !== "camera") stopCamera();
    $("#cropper").hidden = name === "samples";
  }
  for (const t of $$("[data-tab]")) t.addEventListener("click", () => { useTab(t.dataset.tab); if (t.dataset.tab === "camera") startCamera(); });

  // ---- cropper (upload and camera) ----------------------------------------------------------------------------------------
  const cc = $("#crop-canvas"), cctx = cc.getContext("2d"), work = h("canvas", { width: SIZE, height: SIZE });
  const sw = (s) => s.videoWidth || s.naturalWidth || s.width, sh = (s) => s.videoHeight || s.naturalHeight || s.height;
  function drawCropper() {
    const s = st.source; if (!s) return;
    const w = sw(s), hh = sh(s), k = Math.min(1, 360 / Math.max(w, hh)); cc.width = Math.round(w * k); cc.height = Math.round(hh * k);
    cctx.drawImage(s, 0, 0, cc.width, cc.height);
    const b = st.box, x = b.cx * k - (b.size * k) / 2, y = b.cy * k - (b.size * k) / 2, z = b.size * k;
    cctx.fillStyle = "rgba(0,0,0,.45)"; cctx.beginPath(); cctx.rect(0, 0, cc.width, cc.height); cctx.rect(x, y, z, z); cctx.fill("evenodd");
    cctx.strokeStyle = "#fff"; cctx.lineWidth = 2; cctx.strokeRect(x, y, z, z);
    cc._k = k;
  }
  function commitCrop() {
    const s = st.source, b = st.box, ctx = work.getContext("2d", { willReadFrequently: true }); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(s, b.cx - b.size / 2, b.cy - b.size / 2, b.size, b.size, 0, 0, SIZE, SIZE); setBase(canvasToRGB(work), null, false);
  }
  function setSource(src) {
    st.source = src; const w = sw(src), hh = sh(src), size = Math.min(w, hh); st.box = { cx: w / 2, cy: hh / 2, size };
    const slider = $("#crop-size"); slider.min = 24; slider.max = Math.min(w, hh); slider.value = size; drawCropper(); commitCrop();
  }
  $("#crop-size").addEventListener("input", (e) => { const b = st.box, w = sw(st.source), hh = sh(st.source); b.size = +e.target.value; b.cx = Math.min(w - b.size / 2, Math.max(b.size / 2, b.cx)); b.cy = Math.min(hh - b.size / 2, Math.max(b.size / 2, b.cy)); drawCropper(); commitCrop(); });
  let drag = null;
  cc.addEventListener("pointerdown", (e) => { if (!st.source) return; cc.setPointerCapture(e.pointerId); drag = true; moveBox(e); });
  cc.addEventListener("pointermove", (e) => { if (drag) moveBox(e); });
  cc.addEventListener("pointerup", () => { drag = null; });
  function moveBox(e) {
    const r = cc.getBoundingClientRect(), k = cc._k, w = sw(st.source), hh = sh(st.source), b = st.box;
    b.cx = Math.min(w - b.size / 2, Math.max(b.size / 2, ((e.clientX - r.left) / r.width) * cc.width / k)); b.cy = Math.min(hh - b.size / 2, Math.max(b.size / 2, ((e.clientY - r.top) / r.height) * cc.height / k)); drawCropper(); if (!st.live) commitCrop();
  }
  async function fromFile(file) {
    if (!file || !file.type.startsWith("image/")) return; const img = await loadImage(URL.createObjectURL(file)); useTab("upload"); setSource(img);
  }
  $("#file-input").addEventListener("change", (e) => fromFile(e.target.files[0]));
  const drop = $("#dropzone");
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => fromFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) fromFile(f); });

  // ---- camera ----------------------------------------------------------------------------------------------------------
  const video = h("video", { playsinline: true, muted: true });
  let stream = null, liveTimer = 0;
  async function startCamera() {
    const msg = $("#camera-msg"); msg.textContent = "";
    if (!navigator.mediaDevices?.getUserMedia) { msg.textContent = "Camera access is not available in this browser or context."; return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 960 } } }); video.srcObject = stream; await video.play();
      st.live = true; setSource(video); $("#btn-live").textContent = "Freeze frame"; loop();
    } catch (err) { msg.textContent = "Could not start the camera (" + (err.name || "error") + "). You can still upload a photo."; }
  }
  function loop() { if (!st.live) return; drawCropper(); const now = performance.now(); if (now - liveTimer > 220) { liveTimer = now; commitCrop(); } st.raf = requestAnimationFrame(loop); }
  function stopCamera() { st.live = false; cancelAnimationFrame(st.raf); if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; } $("#btn-live").textContent = "Resume camera"; }
  $("#btn-live").addEventListener("click", () => { if (st.live) { st.live = false; commitCrop(); $("#btn-live").textContent = "Resume camera"; } else startCamera(); });

  // ---- controls: threshold, Grad-CAM toggle ----------------------------------------------------------------------------------
  const thr = $("#thr-slider"); thr.value = getThreshold();
  const thrLabel = () => { $("#thr-value").textContent = (+getThreshold()).toFixed(3); };
  thr.addEventListener("input", () => setThreshold(+thr.value)); onThreshold((v) => { thr.value = v; thrLabel(); schedule(); }); thrLabel();
  $("#cam-toggle").addEventListener("change", (e) => { st.cam = e.target.checked; schedule(); });

  // ---- stress lab ---------------------------------------------------------------------------------------------------------
  const rows = $("#stress-rows");
  const readout = {}, sliders = {};
  for (const [k, c] of Object.entries(CORRUPTIONS)) {
    const input = h("input", { type: "range", min: c.min, max: c.max, step: c.step, value: c.neutral, "aria-label": c.label }), out = h("output", {}, "");
    const show = () => { const v = +input.value; out.textContent = k === "brightness" ? v.toFixed(2) + "×" : k === "blur" ? (v <= 1 ? "off" : "k=" + v) : k === "rotate" ? v + "°" : k === "occlude" ? v + "%" : v === 0 ? "off" : "σ=" + v; };
    input.addEventListener("input", () => { st.vals[k] = +input.value; show(); schedule(); }); show(); sliders[k] = { input, show };
    rows.append(h("div", { class: "srow" }, h("label", {}, c.label), input, out, h("button", { type: "button", class: "btn tiny", onclick: () => sweep(k) }, "Sweep")));
  }
  const setVals = (vals) => { for (const [k, c] of Object.entries(CORRUPTIONS)) { st.vals[k] = vals[k] ?? c.neutral; sliders[k].input.value = st.vals[k]; sliders[k].show(); } schedule(); };
  const PRESETS = { Night: { brightness: 0.3 }, Glare: { brightness: 2.4 }, "Sensor noise": { noise: 25 }, Tilted: { rotate: 35 }, Mud: { occlude: 25 }, Blurry: { blur: 9 } };
  $("#presets").replaceChildren(...Object.entries(PRESETS).map(([n, v]) => h("button", { type: "button", class: "chip", onclick: () => setVals(v) }, n)), h("button", { type: "button", class: "chip reset", onclick: () => setVals({}) }, "Reset"));

  const sweepHost = $("#sweep-chart");
  function sweep(kind) {
    if (!st.base) return;
    const c = CORRUPTIONS[kind], n = 25, vals = Array.from({ length: n }, (_, i) => c.min + ((c.max - c.min) * i) / (n - 1)).map((v) => (kind === "blur" ? 1 + 2 * Math.round((v - 1) / 2) : v));
    const clean = net.forward(st.base).probs, base = topK(clean, 1)[0].id, pts = [], tpts = [];
    for (const v of vals) { const p = net.forward(c.apply(st.base, v)).probs; pts.push({ x: v, y: p[base] }); if (st.truth != null && st.truth !== base) tpts.push({ x: v, y: p[st.truth] }); }
    const series = [{ name: `Probability of "${short(names[base])}" (clean prediction)`, color: "var(--cnn)", points: pts }];
    if (tpts.length) series.push({ name: `Probability of true class "${short(names[st.truth])}"`, color: "var(--mn)", points: tpts });
    $("#sweep-title").textContent = `Sweep: ${c.label}`; $("#sweep-wrap").hidden = false;
    $("#sweep-legend").replaceChildren(...series.map((se) => h("li", {}, h("span", { class: "key-line", style: { background: se.color } }), se.name)));
    lineChart(sweepHost, { series, height: 220, xDomain: [c.min, c.max], yDomain: [0, 1], yFormat: fmt.pct0, xLabel: c.label + (c.unit ? ` (${c.unit})` : ""), hlines: [{ y: getThreshold(), label: "answer threshold" }], ariaLabel: `Probability versus ${c.label}`, xTickLabel: (v) => (kind === "brightness" ? (+v).toFixed(1) : Math.round(v)) });
  }

  // ---- parity check against the Keras numbers recorded in the notebook -----------------------------------------------------
  $("#btn-verify").addEventListener("click", async () => {
    const out = $("#verify-out"); out.textContent = "Checking..."; await new Promise((r) => setTimeout(r, 30));
    let same = 0, maxd = 0;
    meta.items.forEach((m, i) => { const p = net.forward(tileRGB(i)).probs, top = topK(p, 1)[0]; if (top.id === m.pred) same++; maxd = Math.max(maxd, Math.abs(top.p - m.conf)); });
    out.textContent = `${same} of ${meta.items.length} predictions identical to the notebook's Keras model; largest difference in confidence ${maxd.toExponential(1)}.`;
  });

  // first sample
  useTab("samples"); const firstGood = meta.items.findIndex((m) => !m.hard && m.true === 14); strip.children[firstGood >= 0 ? firstGood : 0].pick();
  return { net, setBase };
}
