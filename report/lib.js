// Report builder library: formatting rules from Annexure A (A4; margins L1.5in R1.0in T1.0in B1.0in; Times New Roman;
// chapter heading bold 16, main heading bold 14, subheading italic 12, body 12 justified at 1.5 spacing; footer 10 pt
// with the department name left and the page number right; captions 10 pt centred; IEEE references).
const fs = require("fs");
const path = require("path");
const d = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, BorderStyle, ShadingType, Tab, TabStopType, PageBreak, HeadingLevel, LevelFormat, VerticalAlign, PositionalTab, PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader } = d;

const FONT = "Times New Roman";
const TEXT_W = 8306;                       // A4 width 11906 - left 2160 - right 1440 (DXA)
const LINE = 360;                          // 1.5 line spacing

// ---- registries (filled while content is built; resolved on the next pass) ------------------------------------------------
const R = { figs: [], tabs: [], heads: [], cites: {}, citeOrder: [], eqs: {}, listings: [] };
let prev = { figs: {}, tabs: {}, eqs: {}, cites: {}, pages: {} };       // numbers/pages from the previous pass
let state = { chapter: 0, fig: 0, tab: 0, eq: 0, lst: 0, appendix: null };
let REFS = {};
function setRefs(r) { REFS = r; }
function reset(previous) { numCounter = 0; prev = previous || prev; R.figs = []; R.tabs = []; R.heads = []; R.cites = {}; R.citeOrder = []; R.eqs = {}; R.listings = []; state = { chapter: 0, fig: 0, tab: 0, eq: 0, lst: 0, appendix: null }; }
function snapshot(pages) {
  const figs = {}, tabs = {}; R.figs.forEach((f) => (figs[f.key] = f.num)); R.tabs.forEach((t) => (tabs[t.key] = t.num));
  return { figs, tabs, eqs: { ...R.eqs }, cites: { ...R.cites }, pages: pages || prev.pages, headsList: R.heads.slice(), figsList: R.figs.slice(), tabsList: R.tabs.slice() };
}

// ---- inline markup -----------------------------------------------------------------------------------------------------------
// <b>..</b> <i>..</i> <sup>..</sup> <sub>..</sub> <c>..</c> (code)  {@a,b} citations  {fig:key} {tab:key} {eq:key}
function cite(keys) {
  const nums = keys.split(",").map((k) => {
    k = k.trim(); if (!REFS[k]) throw new Error("unknown reference: " + k);
    if (!R.cites[k]) { R.citeOrder.push(k); R.cites[k] = R.citeOrder.length; }
    return R.cites[k];
  });
  nums.sort((a, b) => a - b); const out = []; let i = 0;
  while (i < nums.length) { let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++; out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join("], [")); i = j + 1; }
  return "[" + out.join("], [") + "]";
}
function resolve(kind, key) {
  const m = kind === "fig" ? prev.figs : kind === "tab" ? prev.tabs : prev.eqs;
  return m[key] != null ? (kind === "fig" ? "Figure " : kind === "tab" ? "Table " : "Equation (") + m[key] + (kind === "eq" ? ")" : "") : (kind === "eq" ? "Equation (?)" : kind === "fig" ? "Figure ?" : "Table ?");
}
function runs(text, base = {}) {
  const out = []; const stack = { b: false, i: false, sup: false, sub: false, c: false };
  const re = /<(\/?)(b|i|sup|sub|c|ph)>|\{@([^}]+)\}|\{(fig|tab|eq):([^}]+)\}/g; let last = 0, m, ph = false;
  const push = (t) => { if (!t) return; out.push(new TextRun({ text: t, font: stack.c ? "Courier New" : FONT, size: stack.c ? (base.size || 24) - 2 : base.size || 24, bold: base.bold || stack.b, italics: base.italics || stack.i, superScript: stack.sup, subScript: stack.sub, color: base.color, highlight: ph ? "yellow" : undefined })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index)); last = re.lastIndex;
    if (m[2]) { if (m[2] === "ph") ph = m[1] !== "/"; else stack[m[2]] = m[1] !== "/"; }
    else if (m[3]) push(cite(m[3])); else push(resolve(m[4], m[5].trim()));
  }
  push(text.slice(last)); return out;
}

// ---- paragraphs --------------------------------------------------------------------------------------------------------------
const body = (text, o = {}) => new Paragraph({ children: runs(text, o), alignment: o.align || AlignmentType.JUSTIFIED, spacing: { line: o.line || LINE, after: o.after != null ? o.after : 120, before: o.before || 0 }, indent: o.indent, keepNext: o.keepNext, pageBreakBefore: o.pageBreakBefore });
const p = (text, o) => body(text, o);
function bullets(items, o = {}) { return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bul", level: 0 }, alignment: t.includes("<c>") ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: { line: o.line || LINE, after: 60 } })); }
let numCounter = 0;
function numbered(items) { const ref = "num" + (++numCounter); return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: ref, level: 0 }, alignment: t.includes("<c>") ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 60 } })); }

function chapter(title, o = {}) {                       // "Chapter 1: Introduction"
  state.chapter += 1; state.fig = 0; state.tab = 0; state.lst = 0; state.sec = 0; const label = `CHAPTER ${state.chapter}: ${title.toUpperCase()}`;
  R.heads.push({ level: 1, text: `Chapter ${state.chapter}: ${title}`, key: label });
  return new Paragraph({ children: [new TextRun({ text: label, font: FONT, size: 32, bold: true, color: "000000" })], heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, pageBreakBefore: true, spacing: { before: 0, after: 360, line: 276 }, keepNext: true });
}
function h2(text, o = {}) { R.heads.push({ level: 2, text, key: text }); return new Paragraph({ children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: "000000" })], heading: HeadingLevel.HEADING_2, alignment: AlignmentType.LEFT, spacing: { before: 240, after: 120, line: 276 }, keepNext: true }); }
function h3(text, o = {}) { if (o.toc !== false) R.heads.push({ level: 3, text, key: text }); return new Paragraph({ children: [new TextRun({ text, font: FONT, size: 24, italics: true, color: "000000" })], heading: HeadingLevel.HEADING_3, alignment: AlignmentType.LEFT, spacing: { before: 160, after: 80, line: 276 }, keepNext: true }); }
const h3u = (text) => h3(text, { toc: false });

function plainHeading(text, o = {}) {                   // unnumbered headings: Bibliography, Appendix titles, front-matter titles
  if (o.toc) R.heads.push({ level: 1, text, key: text.toUpperCase(), plain: true, needle: o.upper === false ? text : text.toUpperCase() });
  return new Paragraph({ children: [new TextRun({ text: o.upper === false ? text : text.toUpperCase(), font: FONT, size: o.size || 32, bold: true, color: "000000" })], heading: o.heading === false ? undefined : HeadingLevel.HEADING_1, alignment: o.align || AlignmentType.LEFT, pageBreakBefore: o.pageBreak !== false, spacing: { before: 0, after: o.after != null ? o.after : 360, line: 276 }, keepNext: true });
}

// ---- figures, tables, equations, listings -----------------------------------------------------------------------------------
function pngSize(file) { const b = fs.readFileSync(file); if (b[0] === 0x89) return [b.readUInt32BE(16), b.readUInt32BE(20)]; let i = 2; while (i < b.length) { if (b[i] !== 0xff) { i++; continue; } const t = b[i + 1]; if (t >= 0xc0 && t <= 0xc3) return [b.readUInt16BE(i + 7), b.readUInt16BE(i + 5)]; i += 2 + b.readUInt16BE(i + 2); } throw new Error("size " + file); }
function fig(key, file, title, o = {}) {
  state.fig += 1; const num = `${state.appendix || state.chapter}.${state.fig}`; R.figs.push({ key, num, title });
  const full = path.resolve(__dirname, "figures", file); const [w, h] = pngSize(full); const maxW = (o.width || 5.6) * 96, maxH = (o.maxHeight || 8.2) * 96; let W = maxW, H = (maxW * h) / w; if (H > maxH) { H = maxH; W = (maxH * w) / h; }
  const type = /\.jpe?g$/i.test(file) ? "jpg" : "png";
  return [new Paragraph({ children: [new ImageRun({ type, data: fs.readFileSync(full), transformation: { width: Math.round(W), height: Math.round(H) }, altText: { title, description: title, name: key } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60, line: 240 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: `Figure ${num}: ${title}`, font: FONT, size: 20 })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200, line: 240 } })];
}
function tbl(key, title, header, rows, widths, o = {}) {
  state.tab += 1; const num = `${state.appendix || state.chapter}.${state.tab}`; R.tabs.push({ key, num, title });
  const total = widths.reduce((a, b) => a + b, 0), cw = widths.map((w) => Math.round((w / total) * (o.tableWidth || TEXT_W)));
  const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" }, borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, head, rowAlign) => new TableCell({ children: String(text).split("\n").map((ln) => new Paragraph({ children: runs(ln, { size: o.size || 20, bold: head }), alignment: head ? AlignmentType.CENTER : (o.align && o.align[i]) || AlignmentType.LEFT, spacing: { line: 240, before: 30, after: 30 } })), width: { size: cw[i], type: WidthType.DXA }, borders, shading: head ? { type: ShadingType.CLEAR, fill: "D9D9D9", color: "auto" } : undefined, margins: { top: 30, bottom: 30, left: 70, right: 70 }, verticalAlign: VerticalAlign.CENTER });
  const t = new Table({ width: { size: cw.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: cw, alignment: AlignmentType.CENTER, rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))] });
  return [new Paragraph({ children: [new TextRun({ text: `Table ${num}: ${title}`, font: FONT, size: 20 })], alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80, line: 240 }, keepNext: true }), t, new Paragraph({ children: [], spacing: { after: 160, line: 240 } })];
}
function eq(key, text) {
  state.eq += 1; R.eqs[key] = state.eq;
  return new Paragraph({ children: [new TextRun({ children: [new Tab(), text, new Tab(), `(${state.eq})`], font: FONT, size: 24, italics: true })], tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }], spacing: { line: LINE, before: 60, after: 100 } });
}
function code(key, title, lines, kind = "Listing") {
  state.lst += 1; const num = `${state.appendix || state.chapter}.${state.lst}`; R.listings.push({ key, num, title });
  const out = [new Paragraph({ children: [new TextRun({ text: `${kind} ${num}: ${title}`, font: FONT, size: 20 })], alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60, line: 240 }, keepNext: true })];
  const wrapped = []; const MAX = 90;
  for (const ln of lines) { let s = ln; const ind = (s.match(/^ */) || [""])[0] + "    "; let first = true;
    while (s.length > MAX) { let k = s.lastIndexOf(" ", MAX); if (k < ind.length) k = MAX; wrapped.push(s.slice(0, k)); s = ind + s.slice(k).trimStart(); first = false; }
    wrapped.push(s); }
  wrapped.forEach((ln, i) => out.push(new Paragraph({ children: [new TextRun({ text: ln === "" ? " " : ln, font: "Courier New", size: 15 })], alignment: AlignmentType.LEFT, spacing: { line: 220, after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" }, indent: { left: 60, right: 60 }, keepLines: true, border: { left: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 } } })));
  out.push(new Paragraph({ children: [], spacing: { after: 160, line: 240 } })); return out;
}

// ---- CSV reader (handles quoted fields) ----------------------------------------------------------------------------------------
function readCsv(file) {
  const text = fs.readFileSync(path.resolve(__dirname, "data", file), "utf8").trim(); const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) { const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; }
    else if (c === '"') q = true; else if (c === ",") { row.push(cur); cur = ""; } else if (c === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; } else if (c !== "\r") cur += c; }
  row.push(cur); rows.push(row); const [h, ...rest] = rows; return rest.map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}

// ---- tables of contents (dot-leader lines, no drawn tables) ----------------------------------------------------------------
function tocLine(text, page, level, o = {}) {
  const indent = [0, 0, 360, 720][level] || 0; const size = 24;
  return new Paragraph({ children: [new TextRun({ text, font: FONT, size, bold: level === 1 }), new TextRun({ children: [new Tab(), String(page ?? "")], font: FONT, size, bold: level === 1 })],
    tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: d.LeaderType.DOT }], indent: { left: indent, right: 500, hanging: 0 }, spacing: { line: 300, before: level === 1 ? 100 : 0, after: 20 }, keepLines: true });
}
function pageOf(key, fallback = "") { return prev.pages[key] != null ? prev.pages[key] : fallback; }

// ---- footer and sections ---------------------------------------------------------------------------------------------------------
function footer(width = TEXT_W) {
  return new d.Footer({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: width }], children: [new TextRun({ text: "Department of Information Technology", font: FONT, size: 20 }), new TextRun({ children: [new Tab(), d.PageNumber.CURRENT], font: FONT, size: 20 })] })] });
}
const PAGE = { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 2160, header: 708, footer: 708 } };

module.exports = { readCsv, d, fs, path, FONT, TEXT_W, LINE, R, runs, body, p, bullets, numbered, chapter, h2, h3, h3u, plainHeading, fig, tbl, eq, code, tocLine, pageOf, footer, PAGE, reset, snapshot, setRefs, state: () => state, setAppendix: (a) => { state.appendix = a; state.fig = 0; state.tab = 0; state.lst = 0; }, getPrev: () => prev };
