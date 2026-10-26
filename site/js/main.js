import { $, $$, h } from "./charts.js";
import { buildDashboard } from "./dashboard.js";
import { initDemo } from "./demo.js";

// ---- theme: auto -> light -> dark ---------------------------------------------------------------------------------------
const btn = $("#theme-btn"), order = ["auto", "light", "dark"];
let mode = document.documentElement.dataset.theme || "auto";
const applyTheme = () => {
  if (mode === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = mode;
  btn.textContent = "Theme: " + mode; try { mode === "auto" ? localStorage.removeItem("theme") : localStorage.setItem("theme", mode); } catch (e) { /* storage may be blocked */ }
  document.dispatchEvent(new Event("themechange"));
};
btn.addEventListener("click", () => { mode = order[(order.indexOf(mode) + 1) % order.length]; applyTheme(); });
btn.textContent = "Theme: " + mode;
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => document.dispatchEvent(new Event("themechange")));

// ---- nav highlight ------------------------------------------------------------------------------------------------------
const links = $$(".nav nav a"), io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }, { rootMargin: "-40% 0px -55% 0px" });
$$("main section[id]").forEach((s) => io.observe(s));

// ---- gallery ------------------------------------------------------------------------------------------------------------
const FIGS = [
  ["eda_track", "One physical sign, many frames", "Frames of a single Stop sign as the car approaches. This is why validation must split by sign."],
  ["eda_one_per_class", "All 43 classes", "One random training example per class."],
  ["eda_brightness", "Lighting varies widely", "Brightness distribution with the darkest and brightest samples."],
  ["eda_resolution", "Signs are small", "Width, aspect ratio and size of the training images."],
  ["augmentation_preview", "Augmentation", "One image and fifteen random augmentations, applied only during training."],
  ["gradcam_grid", "Grad-CAM, both models", "Five correct and five confident errors; heat shows influence on the prediction (notebook version, post-softmax score)."],
  ["confident_errors", "Most confident mistakes", "Mostly very dark or tiny images."],
  ["tsne_embeddings", "What the network learned", "t-SNE of the penultimate-layer features, coloured by sign family and by class."],
  ["robustness_examples", "The corruptions", "One image at each severity of each corruption used in the robustness test."],
];
const lb = $("#lightbox");
$("#gallery-grid").replaceChildren(...FIGS.map(([f, t, c]) => h("figure", {}, h("button", { type: "button", "aria-label": `Enlarge: ${t}`, onclick: () => { $("img", lb).src = `img/${f}.webp`; $("img", lb).alt = t; $("p", lb).textContent = t + ". " + c; lb.showModal(); } }, h("img", { src: `img/${f}.webp`, alt: t, loading: "lazy" })), h("figcaption", {}, h("b", {}, t), c))));
$(".close", lb).addEventListener("click", () => lb.close()); lb.addEventListener("click", (e) => { if (e.target === lb) lb.close(); });

// ---- data + components -----------------------------------------------------------------------------------------------------
const fail = (msg) => { const s = $("#demo-status"); s.textContent = msg; s.style.color = "var(--bad)"; };
try {
  const [R, meta] = await Promise.all([fetch("data/results.json").then((r) => r.json()), fetch("data/samples.json").then((r) => r.json())]);
  const sprite = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = "data/samples.png"; });
  buildDashboard(R, sprite, meta);
  await initDemo(R.class_names);
} catch (err) {
  console.error(err);
  fail(location.protocol === "file:" ? "This page needs to be served over http (for example: python -m http.server in the site folder); browsers block local file requests." : "Could not load the model or data: " + err.message);
}
