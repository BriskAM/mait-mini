#!/usr/bin/env node
// Builds the minor-project report (report/out/Minor_Project_Report.docx). Usage: NODE_PATH=<docx> node build.js [--final]
// Three passes: (1) build, (2) render with LibreOffice and read real page numbers from the PDF, (3) rebuild with accurate contents.
const L = require("./lib"); const front = require("./front"); const REFS = require("./refs");
const { d, fs, path, FONT, TEXT_W, runs, FONT: F } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, NumberFormat, LevelFormat } = d;
const cp = require("child_process");
const OUT = path.resolve(__dirname, "out"); fs.mkdirSync(OUT, { recursive: true });
const NAME = "Minor_Project_Report";

const chapters = [];
for (const m of ["ch1_3", "ch4_6", "ch7_9", "ch10_11"]) { const f = path.resolve(__dirname, "content", m + ".js"); if (fs.existsSync(f)) chapters.push(require(f)); }

function bibliography() {
  L.setRefs(REFS); const out = [L.plainHeading("References", { toc: true, size: 32 })];
  L.R.citeOrder.forEach((k, i) => out.push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 22 }), new TextRun({ children: [new d.Tab()], size: 22 }), ...runs(REFS[k], { size: 22 })], tabStops: [{ type: d.TabStopType.LEFT, position: 600 }], indent: { left: 600, hanging: 600 }, alignment: AlignmentType.JUSTIFIED, spacing: { line: 276, after: 100 } })));
  return out;
}

function numberingConfig() {
  const cfg = [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }];
  for (let i = 1; i <= 40; i++) cfg.push({ reference: "num" + i, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] });
  return cfg;
}

function buildDoc(prev) {
  L.reset(prev); L.setRefs(REFS);
  const preA = [...front.declaration(), ...front.certificate(), ...front.acknowledgement(), ...front.vision(), ...front.outcomes(), ...front.sdg()];
  const rub = front.rubrics(), preB = [...front.abstract(), ...front.contents(prev), ...front.listOf("fig", prev), ...front.listOf("tab", prev)];
  // main content is built first so that numbering/citations are registered, then the bibliography
  const main = []; for (const c of chapters) main.push(...c()); main.push(...bibliography()); const backF = path.resolve(__dirname, "content", "back.js"); if (fs.existsSync(backF)) main.push(...require(backF)());
  const props = (start, fmt) => ({ page: { ...L.PAGE, pageNumbers: start ? { start, formatType: fmt } : undefined } });
  const doc = new Document({
    creator: "Akshit Mehta and Shashwat Kumar", title: front.TITLE, description: "ES-451 Minor Project report", features: { updateFields: false },
    styles: { default: { document: { run: { font: FONT, size: 24 } } }, paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, italics: true, color: "000000" }, paragraph: { outlineLevel: 2 } }] },
    numbering: { config: numberingConfig() },
    sections: [
      { properties: props(), children: front.cover() },
      { properties: props(2, NumberFormat.LOWER_ROMAN), footers: { default: L.footer() }, children: preA },
      { properties: { page: { size: { width: 11906, height: 16838, orientation: d.PageOrientation.LANDSCAPE }, margin: { top: 720, right: 720, bottom: 720, left: 720, header: 360, footer: 360 }, pageNumbers: { formatType: NumberFormat.LOWER_ROMAN } } }, footers: { default: L.footer(15398) }, children: rub },
      { properties: { page: { ...L.PAGE, pageNumbers: { formatType: NumberFormat.LOWER_ROMAN } } }, footers: { default: L.footer() }, children: preB },
      { properties: props(1, NumberFormat.DECIMAL), footers: { default: L.footer() }, children: main }] });
  return doc;
}

const roman = (n) => { const m = [[10, "x"], [9, "ix"], [5, "v"], [4, "iv"], [1, "i"]]; let s = ""; for (const [v, r] of m) while (n >= v) { s += r; n -= v; } return s; };
const norm = (s) => s.replace(/\s+/g, " ").trim();
function pagesFromPdf(pdf, snap) {
  const txt = cp.execFileSync("pdftotext", ["-layout", pdf, "-"], { maxBuffer: 1 << 28 }).toString().split("\f"); const pages = txt.map(norm);
  const mainStart = pages.findIndex((t) => t.includes("CHAPTER 1: INTRODUCTION")) + 1;      // 1-based physical page of chapter 1 (uppercase heading only in the body)
  if (mainStart < 2) throw new Error("chapter 1 not found in PDF");
  const out = {}; const arabic = (i) => i - mainStart + 1;
  for (const k of ["DECLARATION", "CERTIFICATE", "ACKNOWLEDGEMENT", "VISION OF THE INSTITUTE", "PROGRAM OUTCOMES (POs)", "SUSTAINABLE DEVELOPMENT GOALS (SDGs) ADDRESSED", "MINOR PROJECT RUBRICS", "ABSTRACT", "LIST OF FIGURES", "LIST OF TABLES"]) {
    const i = pages.findIndex((t, idx) => idx >= 1 && idx < mainStart - 1 && t.startsWith(k) || (idx >= 1 && idx < mainStart - 1 && t.slice(0, 160).includes(k + " ")) );
    if (i >= 0) out[k] = roman(i + 1);
  }
  let ptr = mainStart - 1;
  for (const h of snap.headsList) {
    const key = h.key, needle = norm(h.plain ? (h.needle || h.text) : h.level === 1 ? key : h.text);
    let i = pages.findIndex((t, idx) => idx >= ptr && t.includes(needle)); if (i < 0) { out[key] = ""; continue; } out[key] = arabic(i + 1); ptr = i;
  }
  for (const f of snap.figsList) { const i = pages.findIndex((t, idx) => idx >= mainStart - 1 && t.includes(`Figure ${f.num}:`)); out["F:" + f.num] = i >= 0 ? arabic(i + 1) : ""; }
  for (const f of snap.tabsList) { const i = pages.findIndex((t, idx) => idx >= mainStart - 1 && t.includes(`Table ${f.num}:`)); out["T:" + f.num] = i >= 0 ? arabic(i + 1) : ""; }
  return { out, total: pages.length - (pages[pages.length - 1] === "" ? 1 : 0), mainStart };
}

async function save(doc, name) { const buf = await Packer.toBuffer(doc); const f = path.join(OUT, name + ".docx"); fs.writeFileSync(f, buf); return f; }
function toPdf(docx) { cp.execFileSync("soffice", ["--headless", "--convert-to", "pdf", "--outdir", OUT, docx], { stdio: "ignore", timeout: 280000 }); return docx.replace(/\.docx$/, ".pdf"); }

(async () => {
  let prev = { figs: {}, tabs: {}, eqs: {}, cites: {}, pages: {}, headsList: [], figsList: [], tabsList: [] };
  let snap = null, stats = null;
  for (let pass = 1; pass <= 3; pass++) {
    const doc = buildDoc(prev); snap = L.snapshot(prev.pages); const f = await save(doc, NAME);
    if (pass === 3 || process.argv.includes("--once")) { console.log("pass", pass, "saved", f); if (process.argv.includes("--once")) break; }
    const pdf = toPdf(f); const r = pagesFromPdf(pdf, snap); stats = r;
    const changed = JSON.stringify(r.out) !== JSON.stringify(prev.pages); console.log(`pass ${pass}: ${r.total} pages total, chapter 1 on physical page ${r.mainStart}, page map ${changed ? "changed" : "stable"}`);
    prev = { ...snap, pages: r.out };
    if (!changed && pass >= 2) { const doc2 = buildDoc(prev); await save(doc2, NAME); toPdf(path.join(OUT, NAME + ".docx")); console.log("final rendered"); break; }
  }
})().catch((e) => { console.error(e); process.exit(1); });
