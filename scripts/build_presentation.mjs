import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "docs", "traffic_sign_recognition_ieee_presentation.pptx");
const SKILL_DIR = process.env.SKILL_DIR;
const TMP_DIR = process.env.TMP_DIR;
const { resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const family = resolvePresentationFont();
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const colors = { bg: "#08111F", ink: "#F4F7FB", muted: "#A8B8CC", cyan: "#24D6C8", purple: "#8F7CFF", orange: "#FFB454", line: "#29415E" };
const asset = name => path.join(ROOT, "artifacts", name);
const bytes = async name => new Uint8Array(await fs.readFile(asset(name)));

function box(slide, text, position, style = {}) {
  const shape = slide.shapes.add({ geometry: "textbox", position, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = text;
  shape.text.style = { typeface: family, fontSize: style.fontSize ?? 24, color: style.color ?? colors.ink, bold: style.bold ?? false, autoFit: "shrinkTextOnOverflow", ...(style.italic ? { italic: true } : {}) };
  return shape;
}
function base(title, kicker) {
  const slide = deck.slides.add(); slide.background.fill = colors.bg;
  box(slide, kicker.toUpperCase(), { left: 64, top: 35, width: 1150, height: 22 }, { fontSize: 13, color: colors.cyan, bold: true });
  box(slide, title, { left: 64, top: 70, width: 1150, height: 58 }, { fontSize: 34, bold: true });
  slide.shapes.add({ geometry: "line", position: { left: 64, top: 142, width: 1152, height: 0 }, fill: "none", line: { fill: colors.line, width: 1 } });
  return slide;
}
function notes(slide, text) { slide.speakerNotes.textFrame.setText(text); }
async function addImage(slide, file, position, alt) { slide.images.add({ blob: await bytes(file), contentType: "image/png", alt, fit: "contain", position }); }

let slide = deck.slides.add(); slide.background.fill = colors.bg;
box(slide, "GTSRB Traffic Sign Recognition", { left: 78, top: 130, width: 1120, height: 90 }, { fontSize: 48, bold: true });
box(slide, "An end to end TensorFlow pipeline", { left: 82, top: 240, width: 1000, height: 45 }, { fontSize: 25, color: colors.cyan });
box(slide, "IEEE project presentation", { left: 84, top: 315, width: 600, height: 35 }, { fontSize: 18, color: colors.muted });
box(slide, "Akshit Mehta", { left: 84, top: 570, width: 500, height: 30 }, { fontSize: 17, color: colors.muted });
notes(slide, "Opening: This project classifies a cropped traffic-sign image into one of 43 known GTSRB classes. The presentation follows the complete path from verified data to model prediction. Emphasize that this is image classification, not object detection or a safety-certified driving system.");

slide = base("Problem and objective", "01 / context");
box(slide, "Input", { left: 90, top: 220, width: 170, height: 35 }, { fontSize: 20, color: colors.orange, bold: true });
box(slide, "One cropped traffic-sign image", { left: 90, top: 270, width: 320, height: 75 }, { fontSize: 25, bold: true });
box(slide, "Model", { left: 540, top: 220, width: 170, height: 35 }, { fontSize: 20, color: colors.cyan, bold: true });
box(slide, "Neural network learns visual patterns", { left: 540, top: 270, width: 320, height: 75 }, { fontSize: 25, bold: true });
box(slide, "Output", { left: 990, top: 220, width: 170, height: 35 }, { fontSize: 20, color: colors.purple, bold: true });
box(slide, "Class ID and confidence", { left: 990, top: 270, width: 230, height: 75 }, { fontSize: 25, bold: true });
box(slide, "The goal is a reproducible classification pipeline with honest evaluation and interpretable evidence.", { left: 90, top: 450, width: 1050, height: 70 }, { fontSize: 25, color: colors.muted });
notes(slide, "Explain classification as choosing one label. The input is already cropped, so the system does not answer where a sign is in a road scene. The model predicts one of 43 labels and a probability-like confidence score.");

slide = base("Dataset and integrity", "02 / data");
box(slide, "39,209", { left: 90, top: 205, width: 260, height: 60 }, { fontSize: 38, color: colors.cyan, bold: true });
box(slide, "training images", { left: 92, top: 270, width: 250, height: 30 }, { fontSize: 17, color: colors.muted });
box(slide, "12,630", { left: 470, top: 205, width: 260, height: 60 }, { fontSize: 38, color: colors.purple, bold: true });
box(slide, "official test images", { left: 472, top: 270, width: 260, height: 30 }, { fontSize: 17, color: colors.muted });
box(slide, "43", { left: 850, top: 205, width: 160, height: 60 }, { fontSize: 38, color: colors.orange, bold: true });
box(slide, "classes, IDs 0–42", { left: 852, top: 270, width: 220, height: 30 }, { fontSize: 17, color: colors.muted });
box(slide, "The downloader verifies the published MD5 checksum before extraction and checks that every class is present.", { left: 90, top: 420, width: 1000, height: 70 }, { fontSize: 25 });
notes(slide, "The archive comes from Zenodo. Explain why checksums matter: a download can be incomplete or corrupted while still looking like a file. The MD5 fingerprint provides a reproducible integrity check. The test set stays separate until final evaluation.");

slide = base("Exploratory data analysis", "03 / understanding the data");
await addImage(slide, "eda/class_distribution.png", { left: 70, top: 175, width: 550, height: 360 }, "GTSRB class distribution");
await addImage(slide, "eda/sample_grid.png", { left: 680, top: 175, width: 500, height: 360 }, "Representative GTSRB examples");
box(slide, "The class counts are uneven, and the images vary in lighting, size, blur, and viewpoint.", { left: 90, top: 580, width: 1060, height: 50 }, { fontSize: 20, color: colors.muted });
notes(slide, "Read the left graph as examples per class. Some classes have a few hundred images and others have more than two thousand, which is class imbalance. The sample grid shows the visual variation. This is why we report macro F1 as well as accuracy.");

slide = base("Preparation and leakage control", "04 / data pipeline");
const prep = [["Resize", "64×64 or 96×96"], ["Normalize", "pixel values to 0–1"], ["Augment", "rotation, translation, zoom, brightness, contrast"], ["Split", "group related sequences before validation"]];
prep.forEach((row, i) => { const y = 190 + i * 90; box(slide, row[0], { left: 100, top: y, width: 230, height: 40 }, { fontSize: 24, color: colors.cyan, bold: true }); box(slide, row[1], { left: 410, top: y, width: 690, height: 40 }, { fontSize: 24 }); });
box(slide, "Validation examples guide training decisions. The official test set remains untouched until the end.", { left: 100, top: 585, width: 1050, height: 45 }, { fontSize: 21, color: colors.muted });
notes(slide, "Explain that pixels have different original sizes, so the model needs a fixed tensor shape. Normalization makes optimization numerically stable. Augmentation creates realistic variations while keeping the same label. Grouping related frames reduces near-duplicate leakage between training and validation.");

slide = base("Two model choices", "05 / architectures");
box(slide, "Custom CNN", { left: 95, top: 200, width: 430, height: 45 }, { fontSize: 30, color: colors.cyan, bold: true });
box(slide, "Learns edges, colors, shapes, and symbols from this dataset. Compact and easy to explain.", { left: 95, top: 270, width: 460, height: 130 }, { fontSize: 23 });
box(slide, "MobileNetV2", { left: 700, top: 200, width: 430, height: 45 }, { fontSize: 30, color: colors.purple, bold: true });
box(slide, "Lightweight transfer-learning network. Reuses general visual features and adapts the final layer to 43 classes.", { left: 700, top: 270, width: 460, height: 130 }, { fontSize: 23 });
box(slide, "Baseline versus reusable visual knowledge", { left: 95, top: 520, width: 1000, height: 45 }, { fontSize: 24, color: colors.orange, bold: true });
notes(slide, "A convolutional neural network applies small filters over an image. The custom CNN is the baseline because it is transparent. MobileNetV2 is efficient and can start with ImageNet features. A fair comparison needs matched training budgets.");

slide = base("Training and evaluation", "06 / experiment design");
box(slide, "Training", { left: 90, top: 190, width: 260, height: 40 }, { fontSize: 26, color: colors.cyan, bold: true });
box(slide, "Batches → predictions → loss → weight updates", { left: 90, top: 245, width: 480, height: 60 }, { fontSize: 23 });
box(slide, "Callbacks", { left: 90, top: 365, width: 260, height: 40 }, { fontSize: 26, color: colors.orange, bold: true });
box(slide, "Early stopping, learning-rate reduction, best checkpoint", { left: 90, top: 420, width: 500, height: 70 }, { fontSize: 23 });
box(slide, "Metrics", { left: 700, top: 190, width: 260, height: 40 }, { fontSize: 26, color: colors.purple, bold: true });
box(slide, "Accuracy: overall correctness\nMacro F1: equal weight per class\nConfusion matrix: which classes are mixed up", { left: 700, top: 245, width: 430, height: 170 }, { fontSize: 23 });
notes(slide, "One epoch means one pass through the training data. Loss is the error signal used for weight updates. Early stopping and checkpoints prevent keeping a worse later model. Accuracy is useful but can hide poor rare-class performance, so macro F1 and confusion matrices are important.");

slide = base("Measured results", "07 / official test set");
await addImage(slide, "models/model_comparison.png", { left: 80, top: 180, width: 600, height: 380 }, "Model accuracy and macro F1 comparison");
box(slide, "Custom CNN", { left: 760, top: 210, width: 300, height: 35 }, { fontSize: 23, color: colors.cyan, bold: true });
box(slide, "56.90% accuracy\n56.19% macro F1", { left: 760, top: 260, width: 350, height: 85 }, { fontSize: 28 });
box(slide, "MobileNetV2", { left: 760, top: 400, width: 300, height: 35 }, { fontSize: 23, color: colors.purple, bold: true });
box(slide, "62.86% accuracy\n55.74% macro F1", { left: 760, top: 450, width: 350, height: 85 }, { fontSize: 28 });
notes(slide, "These are full-dataset matched five-epoch runs on the Apple M1 Metal GPU. MobileNetV2 leads on overall accuracy, while the custom CNN is slightly ahead on macro F1. This is a reproducible baseline, not a maximum-performance benchmark.");

slide = base("Learning curves and error analysis", "08 / interpretation");
await addImage(slide, "models/custom_cnn/training_curves.png", { left: 75, top: 180, width: 500, height: 250 }, "Custom CNN learning curves");
await addImage(slide, "models/mobilenet_v2/training_curves.png", { left: 650, top: 180, width: 500, height: 250 }, "MobileNetV2 learning curves");
box(slide, "Learning curves show progress over epochs. Confusion matrices reveal specific class pairs that need improvement.", { left: 100, top: 500, width: 1050, height: 60 }, { fontSize: 24, color: colors.muted });
notes(slide, "A single MobileNetV2 epoch is now visible because the plot uses explicit epoch markers. If training accuracy rises while validation falls, that suggests overfitting. For deeper error analysis, the project now saves top confusion pairs and a top-confusions plot.");

slide = base("Explainable prediction and live demo", "09 / usability");
await addImage(slide, "explanations/mobilenet_v2_sample_gradcam.png", { left: 90, top: 190, width: 360, height: 300 }, "Grad-CAM heatmap over a traffic sign");
box(slide, "Grad-CAM", { left: 560, top: 200, width: 280, height: 40 }, { fontSize: 30, color: colors.orange, bold: true });
box(slide, "Highlights image regions that contributed to the predicted class. It helps us ask whether the model is looking at the sign instead of background noise.", { left: 560, top: 265, width: 560, height: 130 }, { fontSize: 24 });
box(slide, "Local demo: upload an image or use the webcam. The browser sends the frame to a local Keras server and receives class, top-k scores, and the heatmap.", { left: 90, top: 560, width: 1050, height: 55 }, { fontSize: 21, color: colors.muted });
notes(slide, "Grad-CAM is an explanation aid, not proof that the model reasons like a person. The heatmap is useful for debugging shortcuts. The demo server keeps model files local and intentionally does not place them in Git or the static hosted site.");

slide = base("Limitations and next steps", "10 / conclusion");
box(slide, "Current boundary", { left: 90, top: 190, width: 300, height: 40 }, { fontSize: 26, color: colors.orange, bold: true });
box(slide, "Cropped-image classification only\nNot safety-certified\nMatched five-epoch M1 Metal GPU baseline", { left: 90, top: 250, width: 470, height: 150 }, { fontSize: 23 });
box(slide, "Next experiment", { left: 690, top: 190, width: 300, height: 40 }, { fontSize: 26, color: colors.cyan, bold: true });
box(slide, "Run matched multi-epoch training with --require-gpu on a host that exposes the M1 Metal device\nAdd class balancing and compare learning rates\nExtend to YOLO-style detection for full scenes", { left: 690, top: 250, width: 480, height: 170 }, { fontSize: 23 });
box(slide, "Viva point: a trustworthy result includes the data boundary, split strategy, metrics, errors, and limitations.", { left: 90, top: 555, width: 1050, height: 50 }, { fontSize: 23, color: colors.purple, bold: true });
notes(slide, "Close by stating the boundary clearly. The full-data results were matched five-epoch runs on the Apple M1 Metal GPU. Mention class imbalance, cropped inputs, the optional YOLO detection scaffold, and the need for safety validation before any real-world use.");

await fs.mkdir(path.dirname(OUT), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(OUT);
await fs.mkdir(TMP_DIR, { recursive: true });
for (let i = 0; i < deck.slides.items.length; i++) {
  const preview = await deck.export({ slide: deck.slides.items[i], format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(OUT);
