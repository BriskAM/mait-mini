// Results dashboard: every chart is drawn from data/results.json (produced from the notebook's results bundle).
import { $, h, s, fmt, barChart, lineChart, chartCard, legend, tip, mount, rampColor, niceTicks } from "./charts.js";
import { getThreshold, setThreshold, onThreshold } from "./state.js";

const ENTITY = { custom_cnn: "var(--cnn)", "mobilenetv2_head_only": "var(--mn)", "mobilenetv2_fine-tuned": "var(--mnft)", ensemble: "var(--ens)" };
const short = (n) => n.replace("End of speed limit (80km/h)", "End of limit 80").replace("Speed limit (", "Limit ").replace("km/h)", " km/h");
const trunc = (n, k = 25) => (n.length > k ? n.slice(0, k - 1) + "\u2026" : n);

export function buildDashboard(R, sprite, meta) {
  const names = R.class_names, F = R.facts, M = R.models, order = ["custom_cnn", "mobilenetv2_head_only", "mobilenetv2_fine-tuned", "ensemble"];
  const state = { sel: null, hideDiag: true, listeners: [] };
  const select = (id) => { state.sel = id === state.sel ? null : id; state.listeners.forEach((f) => f()); };

  // ---- facts in prose ---------------------------------------------------------------------------------------------------
  document.querySelectorAll("[data-fact]").forEach((el) => {
    const v = el.dataset.fact.split(".").reduce((o, k) => o?.[k], F); if (v == null) return;
    el.textContent = el.dataset.fmt === "pct" ? fmt.pct0(v) : el.dataset.fmt === "dec" ? Number(v).toFixed(1) : typeof v === "number" ? fmt.int(v) : v;
  });

  // ---- KPI tiles --------------------------------------------------------------------------------------------------------
  const sel = M[R.selected], ladder = R.ladder;
  $("#hero-number").textContent = fmt.pct(sel.test_accuracy, 2);
  $("#kpis").replaceChildren(
    tile("Test macro-F1", fmt.pct(sel.test_macro_f1, 1), "every class counts equally"),
    tile("Parameters", fmt.int(F.cnn_params), `${F.tflite_mb} MB as TFLite`),
    tile("Top-3 accuracy", fmt.pct(sel.test_top3, 1), "right answer in the top three"),
    tile("Error reduction", Math.round((1 - ladder[0].val_accuracy) / (1 - ladder[5].val_accuracy)) + "×", "naive baseline to final CNN (validation)"));
  function tile(label, value, sub) { return h("div", { class: "tile" }, h("div", { class: "tile-label" }, label), h("div", { class: "tile-value" }, value), h("div", { class: "tile-sub" }, sub)); }

  // ---- model comparison ----------------------------------------------------------------------------------------------------
  const metrics = [
    { key: "test_accuracy", label: "Test accuracy", f: (v) => fmt.pct(v, 1), domain: [0, 1] }, { key: "test_macro_f1", label: "Test macro-F1", f: (v) => fmt.pct(v, 1), domain: [0, 1] },
    { key: "test_top3", label: "Top-3", f: (v) => fmt.pct(v, 1), domain: [0, 1] }, { key: "ece", label: "Calibration error (lower is better)", f: (v) => v.toFixed(3), domain: [0, 0.07], tick: (v) => v.toFixed(2) },
    { key: "latency_ms", label: "Latency (ms per image)", f: (v) => v.toFixed(0) + " ms", domain: [0, 200], tick: (v) => v }, { key: "parameters", label: "Parameters", f: (v) => (v / 1e6).toFixed(2) + " M", domain: [0, 2.8e6], tick: (v) => v / 1e6 + "M" }];
  let metric = metrics[0];
  $("#card-models").replaceChildren(chartCard({
    title: "Four models, one test set", subtitle: "Official GTSRB test split, 12,630 images, evaluated once. The CNN was chosen on validation accuracy.",
    legendItems: order.map((k) => ({ label: M[k].label, color: ENTITY[k] })),
    build(host) {
      const bar = h("div", { class: "chips", role: "group", "aria-label": "Metric" }), chart = h("div");
      const draw = () => { chart.replaceChildren(); barChart(chart, { orientation: "horizontal", labelWidth: 150, rowHeight: 40, data: order.map((k) => ({ key: k, label: M[k].label, value: M[k][metric.key], color: ENTITY[k], valueText: metric.f(M[k][metric.key]), title: M[k].label, tip: [{ name: metric.label, value: metric.f(M[k][metric.key]) }] })), domain: metric.domain, format: metric.f, tickFormat: metric.tick || ((v) => Math.round(v * 100) + "%"), ariaLabel: metric.label + " by model" }); };
      metrics.forEach((m, i) => bar.append(h("button", { type: "button", class: "chip" + (i === 0 ? " on" : ""), "aria-pressed": i === 0, onclick: (e) => { metric = m; [...bar.children].forEach((c) => { c.classList.remove("on"); c.setAttribute("aria-pressed", "false"); }); e.currentTarget.classList.add("on"); e.currentTarget.setAttribute("aria-pressed", "true"); draw(); } }, m.label)));
      host.append(bar, chart); draw();
    },
    table: { columns: [{ label: "Model", render: (r) => r.label }, { label: "Validation acc.", num: 1, render: (r) => fmt.pct(r.val_accuracy, 2) }, { label: "Test acc.", num: 1, render: (r) => fmt.pct(r.test_accuracy, 2) }, { label: "Macro-F1", num: 1, render: (r) => fmt.pct(r.test_macro_f1, 2) }, { label: "Top-3", num: 1, render: (r) => fmt.pct(r.test_top3, 2) }, { label: "ECE", num: 1, render: (r) => r.ece.toFixed(4) }, { label: "Params", num: 1, render: (r) => fmt.int(r.parameters) }, { label: "ms / image", num: 1, render: (r) => r.latency_ms.toFixed(1) }], rows: order.map((k) => M[k]) },
    note: "Latency is timed through single un-compiled calls on a Colab GPU runtime, so it includes Python overhead. Use it to compare models, not as a device benchmark. The MobileNetV2 variants share parameters and latency because they are the same network."
  }));

  // ---- ladder --------------------------------------------------------------------------------------------------------------
  const lc = ["var(--muted)", "var(--cnn)", "var(--cnn)", "var(--cnn)", "var(--cnn)", "var(--cnn)", "var(--mn)", "var(--mnft)", "var(--ens)"];
  const lab = ["Majority\nclass", "Linear\n16x16 px", "MLP\n32x32 px", "Basic CNN\n2 conv", "Deeper CNN\n+ BatchNorm", "+ Augment.\n+ dropout", "MobileNetV2\nfrozen", "MobileNetV2\nfine-tuned", "Ensemble\nCNN + MNV2"];
  let best = 0; const running = ladder.map((r) => (best = Math.max(best, r.val_accuracy)));
  $("#card-ladder").replaceChildren(chartCard({
    wide: true, title: "Progress ladder: change one thing at a time", subtitle: `Validation accuracy. Error fell from ${fmt.pct(1 - ladder[0].val_accuracy)} to ${fmt.pct(1 - ladder[5].val_accuracy)}, about ${Math.round((1 - ladder[0].val_accuracy) / (1 - ladder[5].val_accuracy))}× fewer mistakes, along the from-scratch track.`,
    legendItems: [{ label: "Naive baseline", color: "var(--muted)" }, { label: "Custom CNN, built up step by step", color: "var(--cnn)" }, { label: "MobileNetV2 frozen", color: "var(--mn)" }, { label: "MobileNetV2 fine-tuned", color: "var(--mnft)" }, { label: "Ensemble", color: "var(--ens)" }, { label: "Best so far", color: "var(--ink)", kind: "line" }],
    build(host) { barChart(host, { height: 330, bottom: 40, domain: [0, 1.0], format: (v) => fmt.pct(v, 1), tickFormat: fmt.pct0, stepLine: running, ariaLabel: "Validation accuracy at each step of the progress ladder", data: ladder.map((r, i) => ({ label: lab[i], value: r.val_accuracy, color: lc[i], valueText: (r.val_accuracy * 100).toFixed(1) + "%", title: `Step ${r.step}: ${r.model}`, tip: [{ name: "validation accuracy", value: fmt.pct(r.val_accuracy, 2) }, { name: "validation macro-F1", value: fmt.pct(r.val_macro_f1, 1) }, { name: "parameters", value: fmt.int(r.parameters) }] })) }); },
    table: { columns: [{ label: "Step", render: (r) => r.step }, { label: "Model", render: (r) => r.model }, { label: "Validation acc.", num: 1, render: (r) => fmt.pct(r.val_accuracy, 2) }, { label: "Validation macro-F1", num: 1, render: (r) => fmt.pct(r.val_macro_f1, 1) }, { label: "Parameters", num: 1, render: (r) => fmt.int(r.parameters) }], rows: ladder },
    note: "Bigger did not mean better: the basic CNN has 1.63 M parameters, the final CNN 116 K. After step 5 the best-so-far line is flat because neither transfer learning nor the ensemble beat the small CNN on this task."
  }));

  // ---- training curves -------------------------------------------------------------------------------------------------------
  const curve = (title, sub, hist, color, boundary) => chartCard({
    title, subtitle: sub, legendItems: [{ label: "Validation", color, kind: "line" }, { label: "Training", color: "var(--muted)", kind: "line" }],
    build(host) { const ep = (a) => a.map((y, i) => ({ x: i + 1, y })); lineChart(host, { height: 250, series: [{ name: "Training", color: "var(--muted)", points: ep(hist.accuracy), endLabel: false }, { name: "Validation", color, points: ep(hist.val_accuracy) }], yDomain: [Math.floor(Math.min(...hist.accuracy, ...hist.val_accuracy) * 10) / 10, 1], yFormat: fmt.pct0, xLabel: "epoch", vlines: boundary ? [{ x: boundary + 0.5, label: "fine-tuning starts" }] : [], endLabels: true, ariaLabel: title }); },
    table: { columns: [{ label: "Epoch", render: (r) => r.e }, { label: "Training acc.", num: 1, render: (r) => fmt.pct(r.a, 2) }, { label: "Validation acc.", num: 1, render: (r) => fmt.pct(r.v, 2) }], rows: hist.accuracy.map((a, i) => ({ e: i + 1, a, v: hist.val_accuracy[i] })) }
  });
  $("#card-curve-cnn").replaceChildren(curve("Custom CNN training", "Validation tracks training closely: little over-fitting.", R.history.custom_cnn, "var(--cnn)"));
  $("#card-curve-mn").replaceChildren(curve("MobileNetV2, two stages", "Fine-tuning lifts validation sharply, but training accuracy reaches ~100% while validation stalls near 90%.", R.history.mobilenet, "var(--mnft)", R.history.mobilenet_boundary));

  // ---- per class + confusion ---------------------------------------------------------------------------------------------------
  const pc = [...R.per_class].sort((a, b) => a.f1 - b.f1);
  const f1Card = chartCard({
    title: "F1 by class", subtitle: "Worst first. Click a class to inspect it in the confusion matrix.", legendItems: [{ label: "F1 below 0.90", color: "var(--cnn)" }, { label: "0.90 or above", color: "var(--muted)" }],
    build(host) { const draw = () => barChart(host, { orientation: "horizontal", labelWidth: 150, rowHeight: 16, valueLabels: true, domain: [0, 1], format: (v) => v.toFixed(2), tickFormat: (v) => v.toFixed(1), ariaLabel: "F1 score per class", selected: state.sel, onSelect: (d) => select(d.key),
      data: pc.map((c) => ({ key: c.id, label: trunc(short(c.name)), value: c.f1, color: c.f1 < 0.9 ? "var(--cnn)" : "var(--muted)", title: `${c.id}. ${c.name}`, tip: [{ name: "F1", value: c.f1.toFixed(3) }, { name: "precision", value: c.precision.toFixed(3) }, { name: "recall", value: c.recall.toFixed(3) }, { name: "test images", value: c.support }, { name: "training images", value: c.train_images }] })) });
      let m = draw(); state.listeners.push(() => { host.replaceChildren(); m = draw(); }); },
    table: { columns: [{ label: "ID", render: (r) => r.id }, { label: "Class", render: (r) => r.name }, { label: "Precision", num: 1, render: (r) => r.precision.toFixed(3) }, { label: "Recall", num: 1, render: (r) => r.recall.toFixed(3) }, { label: "F1", num: 1, render: (r) => r.f1.toFixed(3) }, { label: "Test imgs", num: 1, render: (r) => r.support }, { label: "Train imgs", num: 1, render: (r) => r.train_images }], rows: pc }
  });
  $("#card-f1").replaceChildren(f1Card);

  // confusion matrix (canvas)
  const cm = R.confusion, N = 43, rowSum = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const frac = (i, j) => cm[i][j] / Math.max(1, rowSum[i]);
  const maxOff = Math.max(...cm.flatMap((r, i) => r.map((v, j) => (i === j ? 0 : v / Math.max(1, rowSum[i])))));
  const cmHost = h("div", { class: "cm-host" }), canvas = h("canvas", { class: "cm", role: "img", "aria-label": "Confusion matrix of the custom CNN on the test set: rows true class, columns predicted class" });
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  let geo = null;
  function drawCM(W) {
    const lm = 30, cell = Math.max(5, Math.floor((W - lm) / N)), size = lm + cell * N, dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr; canvas.height = size * dpr; canvas.style.width = size + "px"; canvas.style.height = size + "px";
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, size, size);
    const surface = css("--surface"), grid = css("--grid"), muted = css("--muted"), ink = css("--ink");
    ctx.fillStyle = surface; ctx.fillRect(lm, lm, cell * N, cell * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = frac(i, j); if (!v || (state.hideDiag && i === j)) continue;
      const t = state.hideDiag ? Math.sqrt(v / maxOff) : Math.sqrt(v), [r, g, b] = rampColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(lm + j * cell + 0.5, lm + i * cell + 0.5, cell - 1, cell - 1);
    }
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.strokeRect(lm + 0.5, lm + 0.5, cell * N, cell * N);
    ctx.fillStyle = muted; ctx.font = "10px system-ui, sans-serif"; ctx.textAlign = "center";
    for (let k = 0; k < N; k += 5) { ctx.fillText(k, lm + k * cell + cell / 2, lm - 6); ctx.textAlign = "right"; ctx.fillText(k, lm - 5, lm + k * cell + cell / 2 + 3); ctx.textAlign = "center"; }
    if (state.sel != null) { ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.strokeRect(lm, lm + state.sel * cell, cell * N, cell); ctx.strokeRect(lm + state.sel * cell, lm, cell, cell * N); }
    geo = { lm, cell };
  }
  const cellAt = (e) => { const r = canvas.getBoundingClientRect(), j = Math.floor((e.clientX - r.left - geo.lm) / geo.cell), i = Math.floor((e.clientY - r.top - geo.lm) / geo.cell); return i >= 0 && j >= 0 && i < N && j < N ? [i, j] : null; };
  canvas.addEventListener("pointermove", (e) => { const c = cellAt(e); if (!c) return tip.hide(); const [i, j] = c; tip.show(e.clientX, e.clientY, `${i === j ? "Correct" : "Mistake"}`, [{ name: `true: ${short(names[i])}`, value: `${cm[i][j]} images` }, { name: `predicted: ${short(names[j])}`, value: fmt.pct(frac(i, j), 1) + " of the true class" }]); });
  canvas.addEventListener("pointerleave", tip.hide);
  canvas.addEventListener("click", (e) => { const c = cellAt(e); if (c) select(c[0]); });
  const cmCtl = h("label", { class: "check" }, h("input", { type: "checkbox", checked: true, onchange: (e) => { state.hideDiag = e.target.checked; cmRender.redraw(); } }), "Hide correct predictions (show only mistakes)");
  const detail = h("div", { class: "class-detail", "aria-live": "polite" });
  cmHost.append(canvas); let cmRender = mount(cmHost, (W) => { drawCM(W); return canvas; });
  state.listeners.push(() => { cmRender.redraw(); renderDetail(); });
  document.addEventListener("themechange", () => cmRender.redraw());
  const sampleFor = (id) => { const i = meta.items.findIndex((m) => m.true === id && !m.hard); return i < 0 ? null : i; };
  function renderDetail() {
    if (state.sel == null) { detail.replaceChildren(h("p", { class: "muted" }, "Select a class (click a bar or a matrix row) to see its precision, recall and what it is confused with.")); return; }
    const c = R.per_class[state.sel], asW = cm[state.sel].map((v, j) => ({ j, v })).filter((x) => x.j !== state.sel && x.v > 0).sort((a, b) => b.v - a.v).slice(0, 3);
    const asT = cm.map((r, i) => ({ i, v: r[state.sel] })).filter((x) => x.i !== state.sel && x.v > 0).sort((a, b) => b.v - a.v).slice(0, 3), idx = sampleFor(state.sel);
    const thumb = h("canvas", { width: 64, height: 64, class: "thumb big" }); if (idx != null) { const m = meta.items[idx]; thumb.getContext("2d").drawImage(sprite, (m.tile % meta.cols) * 64, Math.floor(m.tile / meta.cols) * 64, 64, 64, 0, 0, 64, 64); }
    detail.replaceChildren(h("div", { class: "cd-head" }, idx != null ? thumb : null, h("div", {}, h("h4", {}, `${c.id}. ${c.name}`), h("p", { class: "muted" }, `F1 ${c.f1.toFixed(2)} · precision ${c.precision.toFixed(2)} · recall ${c.recall.toFixed(2)} · ${c.support} test images · ${c.train_images} training images`))),
      h("div", { class: "cd-cols" }, h("div", {}, h("strong", {}, "Mistaken for"), h("ul", {}, asW.length ? asW.map((x) => h("li", {}, `${short(names[x.j])}: ${x.v}`)) : h("li", {}, "nothing"))), h("div", {}, h("strong", {}, "Wrongly predicted instead of"), h("ul", {}, asT.length ? asT.map((x) => h("li", {}, `${short(names[x.i])}: ${x.v}`)) : h("li", {}, "nothing")))), h("button", { class: "btn tiny", type: "button", onclick: () => select(state.sel) }, "Clear selection"));
  }
  renderDetail();
  $("#card-cm").replaceChildren(h("figure", { class: "card chart-card" },
    h("figcaption", { class: "card-head" }, h("div", {}, h("h3", {}, "Confusion matrix"), h("p", { class: "sub" }, "Rows are the true class, columns the predicted class, coloured by share of the true class's images."))), cmCtl, cmHost,
    h("div", { class: "ramp" }, h("span", {}, "0%"), h("i", { style: { background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => `rgb(${rampColor(t)})`).join(",")})` } }), h("span", {}, "scaled to the largest value shown")), detail));

  const tc = R.top_confusions;
  $("#card-top").replaceChildren(chartCard({ title: "Most frequent mistakes", subtitle: "Mostly look-alike signs that differ by a small detail.", build(host) { host.append(table()); }, table: null }));
  function table() { return h("ol", { class: "toplist" }, tc.slice(0, 8).map((t) => h("li", {}, h("button", { type: "button", class: "linkish", onclick: () => select(t.true) }, `${short(names[t.true])} → ${short(names[t.pred])}`), h("b", {}, t.count))));
  }

  // ---- reliability and abstention ---------------------------------------------------------------------------------------------
  const rel = R.reliability.filter((b) => b.n > 0);
  $("#card-rel").replaceChildren(chartCard({
    title: "Reliability", subtitle: `When the model says 90%, is it right 90% of the time? Expected calibration error ${sel.ece.toFixed(3)}.`, legendItems: [{ label: "Custom CNN", color: "var(--cnn)", kind: "line" }, { label: "Perfect calibration", color: "var(--muted)", kind: "line" }],
    build(host) { lineChart(host, { height: 250, markers: true, xDomain: [0, 1], yDomain: [0, 1], yFormat: fmt.pct0, xTickLabel: (v) => Math.round(v * 100) + "%", xTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "stated confidence", yLabel: "actual accuracy", xTipTitle: (x) => `confidence ${Math.round(x * 100)}%`,
      series: [{ name: "Perfect calibration", ref: true, color: "var(--muted)", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }, { name: "Custom CNN", color: "var(--cnn)", points: rel.map((b) => ({ x: b.confidence, y: b.accuracy })) }], ariaLabel: "Reliability diagram" }); },
    table: { columns: [{ label: "Confidence bin", render: (r) => `${Math.round(r.lo * 100)}-${Math.round(r.hi * 100)}%` }, { label: "Images", num: 1, render: (r) => fmt.int(r.n) }, { label: "Mean confidence", num: 1, render: (r) => fmt.pct(r.confidence, 1) }, { label: "Accuracy", num: 1, render: (r) => fmt.pct(r.accuracy, 1) }], rows: rel }
  }));

  const covHost = h("div"), readout = h("p", { class: "readout", "aria-live": "polite" }), thr = h("input", { type: "range", min: 0.3, max: 0.999, step: 0.001, value: getThreshold(), "aria-label": "Confidence threshold" }), thrOut = h("output", {});
  const nearest = (t) => R.coverage.reduce((b, c) => (Math.abs(c.t - t) < Math.abs(b.t - t) ? c : b));
  const drawCov = () => {
    const t = getThreshold(), c = nearest(t); thr.value = t; thrOut.textContent = t.toFixed(3);
    readout.replaceChildren("At threshold ", h("strong", {}, t.toFixed(3)), ", the model answers ", h("strong", {}, fmt.pct(c.coverage, 1)), " of images and is right on ", h("strong", {}, fmt.pct(c.accuracy, 1)), " of those. It defers the other ", fmt.pct(1 - c.coverage, 1), " to a human.");
    covHost.replaceChildren(); lineChart(covHost, { height: 250, xDomain: [0.3, 1], yDomain: [0.7, 1], yFormat: fmt.pct0, xTickLabel: (v) => (+v).toFixed(1), xLabel: "confidence threshold", vlines: [{ x: t, label: "your threshold" }], xTipTitle: (x) => `threshold ${(+x).toFixed(2)}`, ariaLabel: "Coverage and accuracy versus confidence threshold",
      series: [{ name: "Share of images answered", color: "var(--ink2)", points: R.coverage.map((c) => ({ x: c.t, y: c.coverage })) }, { name: "Accuracy on answered images", color: "var(--cnn)", points: R.coverage.filter((c) => c.accuracy != null).map((c) => ({ x: c.t, y: c.accuracy })) }] });
  };
  thr.addEventListener("input", () => setThreshold(+thr.value)); onThreshold(drawCov);
  $("#card-cov").replaceChildren(chartCard({
    title: "Answer only when sure", subtitle: "Raise the threshold: fewer images are answered, but the answers are more often right. The same threshold drives the live demo above.", legendItems: [{ label: "Accuracy on answered images", color: "var(--cnn)", kind: "line" }, { label: "Share of images answered", color: "var(--ink2)", kind: "line" }],
    build(host) { host.append(h("div", { class: "slider-row" }, h("label", {}, "Threshold"), thr, thrOut), readout, covHost); drawCov(); },
    table: { columns: [{ label: "Threshold", num: 1, render: (r) => r.t.toFixed(3) }, { label: "Answered", num: 1, render: (r) => fmt.pct(r.coverage, 1) }, { label: "Accuracy on answered", num: 1, render: (r) => (r.accuracy == null ? "-" : fmt.pct(r.accuracy, 2)) }], rows: R.coverage.filter((_, i) => i % 6 === 0) },
    note: "The threshold curve is computed on the test set, so it shows the trade-off. A deployed system should choose its threshold on separate validation data."
  }));

  // ---- robustness (small multiples) -------------------------------------------------------------------------------------------
  const rob = R.robustness, levels = { Darkening: ["×0.7", "×0.5", "×0.3"], Glare: ["×1.4", "×1.8", "×2.4"], "Gaussian noise": ["σ 10", "σ 25", "σ 50"], Blur: ["k 3", "k 5", "k 7"], Rotation: ["10°", "20°", "35°"], Occlusion: ["10%", "20%", "35%"] };
  const robKeys = Object.keys(rob), robName = (k) => Object.keys(levels).find((n) => k.startsWith(n));
  $("#card-rob").replaceChildren(chartCard({
    wide: true, title: "Robustness: what breaks it", subtitle: "Test accuracy of the custom CNN on 6,000 test images corrupted at three severities. Blur, darkening and glare are tolerated; noise, occlusion and large rotation are not.",
    build(host) { const grid = h("div", { class: "mini-grid" }); host.append(grid);
      robKeys.forEach((k) => { const n = robName(k), lv = levels[n], d = rob[k], cell = h("div", { class: "mini" }, h("h4", {}, n), h("div")); grid.append(cell);
        lineChart(cell.lastChild, { height: 170, markers: true, endLabels: true, xDomain: [0, 3], xTicks: [0, 1, 2, 3], yDomain: [0, 1], yFormat: fmt.pct0, xTickLabel: (v) => (v === 0 ? "clean" : lv[v - 1]), xTipTitle: (x) => (x === 0 ? "clean" : lv[x - 1]), ariaLabel: `${n} robustness`, series: [{ name: "Accuracy", color: "var(--cnn)", points: [d.clean, d.mild, d.medium, d.severe].map((y, x) => ({ x, y })) }] }); }); },
    table: { columns: [{ label: "Corruption", render: (r) => r.k }, { label: "Clean", num: 1, render: (r) => fmt.pct(r.d.clean, 1) }, { label: "Mild", num: 1, render: (r) => fmt.pct(r.d.mild, 1) }, { label: "Medium", num: 1, render: (r) => fmt.pct(r.d.medium, 1) }, { label: "Severe", num: 1, render: (r) => fmt.pct(r.d.severe, 1) }], rows: robKeys.map((k) => ({ k, d: rob[k] })) },
    note: "These are synthetic corruptions applied to the 64x64 images. You can reproduce the same families live in the stress lab above."
  }));

  // ---- data ---------------------------------------------------------------------------------------------------------------------
  const tcnt = R.eda.train_counts, ecnt = R.eda.test_counts;
  $("#card-classes").replaceChildren(chartCard({
    title: "Images per class", subtitle: `Training set: ${fmt.int(F.train_images)} images, ${F.imbalance}× between the largest and smallest class.`,
    build(host) { barChart(host, { height: 240, bottom: 26, labelEvery: 5, valueLabels: false, format: fmt.int, ariaLabel: "Training images per class", valueName: "training images", data: tcnt.map((v, i) => ({ label: String(i), value: v, color: "var(--cnn)", title: `${i}. ${names[i]}`, tip: [{ name: "training images", value: fmt.int(v) }, { name: "test images", value: fmt.int(ecnt[i]) }] })), annotations: [{ text: `largest: ${short(F.largest_class.name)} (${fmt.int(F.largest_class.images)})`, x: "end", y: 14 }, { text: `smallest: ${short(F.smallest_class.name)} (${F.smallest_class.images})`, x: "end", y: 28 }] }); },
    table: { columns: [{ label: "ID", render: (r) => r.i }, { label: "Class", render: (r) => names[r.i] }, { label: "Train", num: 1, render: (r) => fmt.int(r.t) }, { label: "Test", num: 1, render: (r) => fmt.int(r.e) }], rows: tcnt.map((t, i) => ({ i, t, e: ecnt[i] })) }
  }));
  const wh = R.eda.width_hist;
  $("#card-widths").replaceChildren(chartCard({
    title: "Image width", subtitle: `Signs are small: median ${F.width_median} px, ${Math.round(F.share_under_40px * 100)}% under 40 px. Everything is resized to 64×64.`,
    build(host) { barChart(host, { height: 240, bottom: 26, labelEvery: 2, valueLabels: false, format: fmt.int, ariaLabel: "Histogram of training image widths", valueName: "images", data: wh.counts.map((v, i) => ({ label: String(wh.edges[i]), value: v, color: "var(--cnn)", title: `${wh.edges[i]}-${wh.edges[i + 1]} px wide`, tip: [{ name: "images", value: fmt.int(v) }] })) }); },
    table: { columns: [{ label: "Width (px)", render: (r) => r.w }, { label: "Images", num: 1, render: (r) => fmt.int(r.n) }], rows: wh.counts.map((n, i) => ({ w: `${wh.edges[i]}-${wh.edges[i + 1]}`, n })) }
  }));
}
