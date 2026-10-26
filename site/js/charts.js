// Small dependency-free chart kit (SVG + canvas). Colours come from CSS variables, so light/dark needs no re-render for SVG.
// Conventions: thin marks with a rounded data end, 2px lines, hairline solid gridlines, a tooltip on hover AND keyboard focus,
// a legend whenever there are two or more series, and a table view for every chart.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v; else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "text") e.textContent = v; else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));   // text via text nodes, never innerHTML
  return e;
}

const NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

export const fmt = {
  pct: (v, d = 1) => (v * 100).toFixed(d) + "%",
  pct0: (v) => Math.round(v * 100) + "%",
  int: (v) => Math.round(v).toLocaleString("en-US"),
  num: (v, d = 3) => Number(v).toFixed(d),
};

export function niceTicks(min, max, count = 5) {
  const span = max - min || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// ---- tooltip: one shared element; values lead, labels follow ------------------------------------------------------------
let tipEl;
export const tip = {
  show(clientX, clientY, title, rows) {
    if (!tipEl) { tipEl = h("div", { class: "tip", role: "status" }); document.body.append(tipEl); }
    tipEl.replaceChildren(title ? h("div", { class: "tip-title" }, title) : null,
      ...rows.map((r) => h("div", { class: "tip-row" }, r.color ? h("span", { class: "tip-key", style: { background: r.color } }) : null, h("strong", {}, r.value), h("span", {}, r.name || ""))));
    tipEl.style.opacity = 1;
    const w = tipEl.offsetWidth, hgt = tipEl.offsetHeight;
    let x = clientX + 14, y = clientY + 14;
    if (x + w > innerWidth - 8) x = clientX - w - 14;
    if (y + hgt > innerHeight - 8) y = clientY - hgt - 14;
    tipEl.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  },
  hide() { if (tipEl) tipEl.style.opacity = 0; },
  at(el, title, rows) {                                  // anchor to an element (keyboard focus)
    const r = el.getBoundingClientRect(); tip.show(r.left + r.width / 2, r.top, title, rows);
  },
};

// Re-render `draw(width)` whenever the host is resized.
export function mount(host, draw) {
  let last = 0;
  const render = () => { const w = Math.floor(host.clientWidth); if (w < 40 || w === last) return; last = w; host.replaceChildren(draw(w)); };
  new ResizeObserver(render).observe(host); render();
  return { redraw: () => { last = 0; render(); } };
}

export function legend(items) {
  return h("ul", { class: "legend" }, items.map((it) => h("li", {},
    it.kind === "line" ? h("span", { class: "key-line", style: { background: it.color } }) : h("span", { class: "key-rect", style: { background: it.color } }), it.label)));
}

// ---- bar chart (vertical columns or horizontal bars) -------------------------------------------------------------------
// data: [{label, value, color, tip: [{name,value}], title, aria}]
export function barChart(host, o) {
  return mount(host, (W) => {
    const horizontal = o.orientation === "horizontal", n = o.data.length, fmtV = o.format || fmt.num;
    const max = o.domain ? o.domain[1] : Math.max(...o.data.map((d) => d.value)) * 1.08, min = o.domain ? o.domain[0] : 0;
    const m = horizontal ? { t: 6, r: o.valueLabels === false ? 12 : 56, b: o.axis === false ? 6 : 24, l: o.labelWidth || 120 } : { t: o.valueLabels === false ? 8 : 22, r: 8, b: o.bottom || 34, l: o.axis === false ? 8 : 44 };
    const band = horizontal ? (o.rowHeight || 22) : (W - m.l - m.r) / n, H = horizontal ? m.t + m.b + band * n : (o.height || 260);
    const pw = W - m.l - m.r, ph = H - m.t - m.b, thick = Math.min(24, (horizontal ? band : band) * (horizontal ? 0.58 : 0.62));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.ariaLabel || "bar chart", class: "chart" });
    const ticks = niceTicks(min, max, horizontal ? 4 : 5);
    for (const t of ticks) {
      const v = (t - min) / (max - min);
      if (horizontal) { const x = m.l + v * pw; svg.append(s("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, class: "grid" })); if (o.axis !== false) svg.append(s("text", { x, y: H - 6, class: "tick", "text-anchor": "middle" }, (o.tickFormat || fmtV)(t))); }
      else { const y = m.t + ph - v * ph; svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" })); if (o.axis !== false) svg.append(s("text", { x: m.l - 8, y: y + 4, class: "tick", "text-anchor": "end" }, (o.tickFormat || fmtV)(t))); }
    }
    svg.append(s("line", horizontal ? { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" } : { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, class: "axis" }));
    o.data.forEach((d, i) => {
      const v = Math.max(0, (Math.min(d.value, max) - min) / (max - min)), color = d.color || "var(--s1)", sel = o.selected != null && o.selected === d.key;
      const g = s("g", { class: "mark" + (sel ? " selected" : ""), tabindex: 0, role: "img", "aria-label": d.aria || `${d.label}: ${fmtV(d.value)}` });
      let x, y, w, hh, tx, ty, anchor;
      if (horizontal) { w = Math.max(2, v * pw); hh = thick; x = m.l; y = m.t + i * band + (band - thick) / 2; tx = x + w + 6; ty = y + hh / 2 + 4; anchor = "start"; }
      else { hh = Math.max(2, v * ph); w = thick; x = m.l + i * band + (band - thick) / 2; y = m.t + ph - hh; tx = x + w / 2; ty = y - 6; anchor = "middle"; }
      const r = Math.min(4, (horizontal ? w : hh) / 2), path = horizontal                                  // 4px rounded data end, square at the baseline
        ? `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + hh - r} Q${x + w},${y + hh} ${x + w - r},${y + hh} H${x} Z`
        : `M${x},${y + hh} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + hh} Z`;
      const hit = horizontal ? s("rect", { x: 0, y: m.t + i * band, width: W, height: band, class: "hit" }) : s("rect", { x: m.l + i * band, y: m.t, width: band, height: ph + m.b, class: "hit" });
      g.append(hit, s("path", { d: path, style: `fill:${color}` }));
      if (o.stepLine && !horizontal && o.stepLine[i] - d.value < 1e-9) ty -= 8;                    // keep the label clear of the running-best line
      if (o.valueLabels !== false && (o.valueLabelFilter ? o.valueLabelFilter(d, i) : true)) g.append(s("text", { x: tx, y: ty, class: "vlabel", "text-anchor": anchor }, d.valueText || fmtV(d.value)));
      if (horizontal) g.append(s("text", { x: m.l - 8, y: y + hh / 2 + 4, class: "ylabel", "text-anchor": "end" }, d.label));
      else if (o.showLabels !== false && (o.labelEvery ? i % o.labelEvery === 0 : true)) {
        const lines = String(d.label).split("\n"); lines.forEach((ln, k) => g.append(s("text", { x: x + w / 2, y: m.t + ph + 16 + k * 13, class: "xlabel", "text-anchor": "middle" }, ln)));
      }
      const rows = d.tip || [{ name: o.valueName || "", value: fmtV(d.value) }], title = d.title || d.label;
      const showTip = (ev) => (ev && ev.clientX != null && ev.type !== "focus" ? tip.show(ev.clientX, ev.clientY, title, rows) : tip.at(g, title, rows));
      g.addEventListener("pointermove", showTip); g.addEventListener("pointerleave", tip.hide); g.addEventListener("focus", showTip); g.addEventListener("blur", tip.hide);
      if (o.onSelect) { g.style.cursor = "pointer"; g.addEventListener("click", () => o.onSelect(d)); g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); o.onSelect(d); } }); }
      svg.append(g);
    });
    if (o.stepLine && !horizontal) {                       // running best, drawn as a step across the column bands
      let d = ""; o.stepLine.forEach((v, i) => { const y = m.t + ph - ((v - min) / (max - min)) * ph, x0 = m.l + i * band, x1 = x0 + band; d += (i ? `L${x0},${y}` : `M${x0},${y}`) + `L${x1},${y}`; });
      svg.append(s("path", { d, class: "stepline", "pointer-events": "none" }));
    }
    for (const a of o.annotations || []) svg.append(s("text", { x: a.x === "end" ? W - m.r : m.l + 4, y: a.y ?? m.t + 12, class: "annot", "text-anchor": a.x === "end" ? "end" : "start" }, a.text));
    return svg;
  });
}

// ---- line chart ----------------------------------------------------------------------------------------------------------
// series: [{name, color, points:[{x,y}], width, dashed(no: reserved for refs), ref:true (thin muted reference line), endLabel}]
let clipCount = 0;
export function lineChart(host, o) {
  return mount(host, (W) => {
    const H = o.height || 260, m = { t: 14, r: o.endLabels ? 70 : 16, b: o.xLabel ? 44 : 30, l: 46 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    const all = o.series.flatMap((se) => se.points), xs = all.map((p) => p.x), ys = all.map((p) => p.y).filter((v) => v != null);
    const [x0, x1] = o.xDomain || [Math.min(...xs), Math.max(...xs)], [y0, y1] = o.yDomain || [Math.min(...ys), Math.max(...ys)];
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * pw, Y = (v) => m.t + ph - ((v - y0) / (y1 - y0 || 1)) * ph;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.ariaLabel || "line chart", class: "chart" });
    const clipId = "clip" + ++clipCount; svg.append(s("defs", {}, s("clipPath", { id: clipId }, s("rect", { x: m.l - 6, y: m.t - 6, width: pw + 12, height: ph + 12 }))));
    for (const t of niceTicks(y0, y1, 4)) { svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }), s("text", { x: m.l - 8, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, (o.yFormat || fmt.num)(t))); }
    const xt = o.xTicks || niceTicks(x0, x1, 6);
    for (const t of xt) svg.append(s("text", { x: X(t), y: H - m.b + 16, class: "tick", "text-anchor": "middle" }, o.xTickLabel ? o.xTickLabel(t) : t));
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, class: "axis" }));
    if (o.xLabel) svg.append(s("text", { x: m.l + pw / 2, y: H - 6, class: "axis-title", "text-anchor": "middle" }, o.xLabel));
    if (o.yLabel) svg.append(s("text", { x: 12, y: m.t + ph / 2, class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ph / 2})` }, o.yLabel));
    for (const v of o.vlines || []) { svg.append(s("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: m.t + ph, class: "ref" }), s("text", { x: X(v.x) + 5, y: m.t + ph - 8, class: "annot" }, v.label)); }
    for (const hl of o.hlines || []) { svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(hl.y), y2: Y(hl.y), class: "ref" }), s("text", { x: W - m.r - 4, y: Y(hl.y) - 5, class: "annot", "text-anchor": "end" }, hl.label)); }
    for (const se of o.series) {
      const pts = se.points.filter((p) => p.y != null), d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
      if (se.ref) { svg.append(s("path", { d, class: "refline", "clip-path": `url(#${clipId})` })); continue; }
      svg.append(s("path", { d, class: "line", style: `stroke:${se.color}`, "stroke-width": se.width || 2, "clip-path": `url(#${clipId})` }));
      if (o.markers) for (const p of pts) svg.append(s("circle", { cx: X(p.x), cy: Y(p.y), r: 4, class: "dot", style: `fill:${se.color}` }));
      if (o.endLabels && se.endLabel !== false) { const last = pts[pts.length - 1]; if (last) svg.append(s("circle", { cx: X(last.x), cy: Y(last.y), r: 4, class: "dot", style: `fill:${se.color}` }), s("text", { x: X(last.x) + 9, y: Y(last.y) + 4, class: "vlabel" }, (o.yFormat || fmt.num)(last.y))); }
    }
    // crosshair + one tooltip listing every series at the nearest x
    const cross = s("line", { y1: m.t, y2: m.t + ph, class: "cross", opacity: 0 }), live = o.series.filter((se) => !se.ref);
    svg.append(cross);
    const overlay = s("rect", { x: m.l, y: m.t, width: pw, height: ph, class: "hit", tabindex: 0 });
    const xsAll = [...new Set(live.flatMap((se) => se.points.map((p) => p.x)))].sort((a, b) => a - b);
    let focusIdx = xsAll.length - 1;
    const at = (xv, cx, cy, byKey) => {
      let best = xsAll[0]; for (const v of xsAll) if (Math.abs(v - xv) < Math.abs(best - xv)) best = v;
      cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("opacity", 1);
      const rows = live.map((se) => { const p = se.points.find((q) => q.x === best); return p && p.y != null ? { color: se.color, name: se.name, value: (o.yFormat || fmt.num)(p.y) } : null; }).filter(Boolean);
      const title = o.xTipTitle ? o.xTipTitle(best) : `${o.xLabel || "x"}: ${o.xTickLabel ? o.xTickLabel(best) : best}`;
      if (byKey) { const r = overlay.getBoundingClientRect(); tip.show(r.left + (X(best) - m.l), r.top + 10, title, rows); } else tip.show(cx, cy, title, rows);
    };
    overlay.addEventListener("pointermove", (ev) => { const r = overlay.getBoundingClientRect(); at(x0 + ((ev.clientX - r.left) / r.width) * (x1 - x0), ev.clientX, ev.clientY); });
    overlay.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); tip.hide(); });
    overlay.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") focusIdx = Math.max(0, focusIdx - 1); else if (e.key === "ArrowRight") focusIdx = Math.min(xsAll.length - 1, focusIdx + 1); else return; e.preventDefault(); at(xsAll[focusIdx], 0, 0, true); });
    overlay.addEventListener("focus", () => at(xsAll[focusIdx], 0, 0, true)); overlay.addEventListener("blur", () => { cross.setAttribute("opacity", 0); tip.hide(); });
    svg.append(overlay);
    return svg;
  });
}

export function tableView(columns, rows) {
  const t = h("table", { class: "data-table" });
  t.append(h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col", class: c.num ? "num" : "" }, c.label)))));
  t.append(h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c, i) => h("td", { class: c.num ? "num" : "" }, c.render ? c.render(r) : r[c.key]))))));
  return h("div", { class: "table-wrap", tabindex: 0 }, t);
}

// A titled card with a Chart / Table toggle. build(host) draws the chart; table = {columns, rows}.
export function chartCard({ title, subtitle, legendItems, note, build, table, wide, id }) {
  const chartHost = h("div", { class: "chart-host" }), tableHost = h("div", { class: "table-host", hidden: true });
  const btnChart = h("button", { type: "button", class: "seg on", "aria-pressed": "true" }, "Chart"), btnTable = h("button", { type: "button", class: "seg", "aria-pressed": "false" }, "Table");
  const set = (showTable) => {
    chartHost.hidden = showTable; tableHost.hidden = !showTable; btnChart.classList.toggle("on", !showTable); btnTable.classList.toggle("on", showTable);
    btnChart.setAttribute("aria-pressed", String(!showTable)); btnTable.setAttribute("aria-pressed", String(showTable));
    if (showTable && !tableHost.firstChild && table) tableHost.append(tableView(table.columns, table.rows));
  };
  btnChart.addEventListener("click", () => set(false)); btnTable.addEventListener("click", () => set(true));
  const card = h("figure", { class: "card chart-card" + (wide ? " wide" : ""), id },
    h("figcaption", { class: "card-head" }, h("div", {}, h("h3", {}, title), subtitle ? h("p", { class: "sub" }, subtitle) : null),
      table ? h("div", { class: "seg-group", role: "group", "aria-label": "View" }, btnChart, btnTable) : null),
    legendItems ? legend(legendItems) : null, chartHost, tableHost, note ? h("p", { class: "note" }, note) : null);
  queueMicrotask(() => build(chartHost));
  return card;
}

// ---- heatmap on canvas (sequential single-hue ramp; zero = surface) ---------------------------------------------------
export const SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"];
export function rampColor(t) {                         // t in [0,1] -> interpolated step of the blue ramp
  const stops = SEQ_BLUE.map((hx) => [1, 3, 5].map((i) => parseInt(hx.slice(i, i + 2), 16))), p = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1), i = Math.floor(p), f = p - i;
  return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
}
