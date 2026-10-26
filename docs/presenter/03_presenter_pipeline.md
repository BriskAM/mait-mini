# Presenter Pipeline — Run of Show, Parallel to the Notebook

This runs **alongside** `notebooks/traffic_sign_recognition_demo.ipynb`. Every block below is one notebook cell (numbered 0–55, in order from the top), so the Driver and Narrator can move through the notebook and this document together.

**Related:** concepts in `01_knowledge_base.md` (KB), questions in `02_viva_readiness.md`, and real numbers from the first Colab run in `04_results_and_findings.md` (the capture sheet at the end of this file is already filled from that run).

---

## 0. How to use this pipeline

### 0.1 Two ways to present
| Cut | Length | Cells to present |
|---|---|---|
| **Express cut** | about 20 minutes | Depth on Parts 3, 8, 9, 10 and 11; one sentence each on the rest (see the time plan) |
| **Core cut** | about 30 minutes | Cells marked **CORE** |
| **Full walkthrough** | about 40 minutes | Every cell; **OPT** cells are skippable if time is short |

Timings are guidelines for one narrator who rehearsed; measure your own in a mock run.

### 0.2 Roles
- **Narrator:** speaks the "Say" lines, in your own words.
- **Driver:** scrolls, runs cheap cells, zooms plots. Never re-runs training live.
- **Q&A lead:** answers questions; others wait to be invited.

### 0.3 Presenting approach
**Run all cells before the session** so outputs are already on screen. Present by scrolling. Re-run live only the cheap, impressive cells: cell 47 (`predict_sign`), cell 44 (selective prediction) and, if you like, cell 48 (upload a photo).

### 0.4 Phase before the presentation
| When | Task |
|---|---|
| Day before | Run all in `full` mode on a T4 GPU; save the executed notebook; download the results zip; fill the capture sheet (at the end of this file) |
| Morning | Open the executed notebook; collapse nothing important; set browser zoom to about 110%; test screen sharing |
| 15 minutes before | Reopen the notebook, check it is connected (do not restart the runtime, which would clear in-memory objects needed by cells 47–48) |
| 5 minutes before | Scroll to cell 0; open this file and `01_knowledge_base.md` on a second device |

### 0.5 Legend
- **Screen:** what the audience sees.
- **Do:** Driver action.
- **Say:** suggested narration (adapt it; `[ ]` means read the number from the screen or the capture sheet).
- **If asked:** where to look in the KB.
- **Check:** what good output looks like, and red flags.

---

## PART 1 — Opening and problem (about 3 min)

### Cell 0 · Title and agenda · CORE · 0.5 min
- **Screen:** title, table of sections.
- **Do:** none; stay on the title.
- **Say:** "This is a complete pipeline for recognising German traffic signs: from verified data to a model we can explain, stress-test and reuse. I will follow the sections in this table."
- **If asked:** KB Part 0.

### Cell 1 · Problem statement and workflow · CORE · 1 min
- **Screen:** task, why it matters, scope, evaluation protocol.
- **Do:** none.
- **Say:** "The input is a cropped photo of a sign; the output is one of 43 classes. Some mistakes matter more than others, for example confusing two speed limits, so we evaluate beyond accuracy. We fixed our protocol first: grouped validation, decisions on validation only, official test used once."
- **If asked:** KB A2, A8; viva Q1, Q2.
- **Check:** do not promise driving safety.

### Cell 2 · Workflow diagram (code) · CORE · 1 min
- **Screen:** eight-box pipeline figure.
- **Do:** none (already run).
- **Say:** "Eight stages. Everything I show follows this order; I will name each stage as we reach it."
- **Check:** figure appears with no text overlap.

---

## PART 2 — Setup and data (about 3 min)

### Cell 3 · Setup and reproducibility (markdown) · OPT · 0.5 min
- **Say:** "One switch controls the budget; all randomness is seeded; the notebook is self-contained."
- **If asked:** KB A29.

### Cell 4 · Setup code · CORE (brief) · 1 min
- **Screen:** printed mode, TensorFlow version, GPU name, configuration dictionary.
- **Do:** point at the GPU line and the config.
- **Say:** "We are running in `full` mode on a [GPU name]. These are the training budgets for this mode."
- **Check:** GPU line is not "none". If it says `smoke`, the run was on CPU: do not present those numbers.

### Cell 5 · Data acquisition (markdown) · OPT · 0.5 min
- **Say:** "We verify the 320 MB archive by checksum, use the official split, and assert the expected counts so a wrong dataset fails immediately."
- **If asked:** KB A30; viva Q10.

### Cell 6 · Download and verify (code) · OPT · 0.5 min
- **Screen:** MD5 line, "Dataset verified".
- **Say:** "The checksum matches the published one."

### Cell 7 · Label tables (code) · CORE · 1 min
- **Screen:** counts of train, test, classes, and the number of distinct tracks; a data frame head.
- **Say:** "39,209 training images, 12,630 test images, 43 classes. Notice the last number: about [tracks] distinct tracks. Hold that thought; it matters in section 5."
- **Check:** counts match the asserts; tracks about 1,300.

---

## PART 3 — Exploratory data analysis (about 4 min)

### Cell 8 · EDA introduction (markdown) · CORE · 0.5 min
- **Say:** "Every finding here changes a modelling decision. The table lists the four questions and what each decides."

### Cell 9 · Class distribution (code) · CORE · 1 min
- **Screen:** two bar charts (train, test), printed min, max, ratio.
- **Do:** point at the largest and smallest bars.
- **Say:** "The biggest class has [max] images, the smallest [min], about [ratio] times. The test set has the same shape, so imbalance is a property of the data. That is why we report macro-F1 as well as accuracy."
- **If asked:** KB A21; viva Q11.

### Cell 10 · Finding: imbalance (markdown) · OPT
- Read the bold finding only.

### Cell 11 · One example per class (code) · OPT · 0.5 min
- **Say:** "These are the 43 classes. Many look alike, for example the speed limits."

### Cell 12 · Image sizes (code) · CORE · 0.5 min
- **Screen:** width histogram, width×height map, aspect-ratio scatter, printed statistics.
- **Say:** "Images are small: median width [ ] pixels, [ ]% under 40 pixels. So we do not need high resolution; 64 by 64 is enough."
- **If asked:** viva Q36, Q37.

### Cell 13 · Finding: signs are small (markdown) · OPT

### Cell 14 · Lighting diversity (code) · OPT · 0.5 min
- **Say:** "Brightness varies widely: here are the darkest and brightest samples. This is why we add contrast augmentation and test lighting later."

### Cell 15 · Finding: illumination (markdown) · OPT

### Cell 16 · Tracks (code) · CORE · 1 min
- **Screen:** ten frames of one Stop sign, track statistics.
- **Do:** pause here; this is the key image of the talk.
- **Say:** "This is one physical Stop sign photographed as the car approaches. The frames are almost identical. The dataset has about [ ] such tracks with about [ ] frames each."
- **If asked:** KB A20.

### Cell 17 · Finding: images are not independent (markdown) · CORE · 0.5 min
- **Say:** "So the images are not independent, and that determines how we must split."

---

## PART 4 — Validation design (about 2.5 min)

### Cell 18 · Why a grouped split (markdown) · CORE · 1 min
- **Say:** "If we split images at random, copies of the same sign land in training and validation, the model has effectively seen the answer, and validation accuracy is inflated. We split by physical sign instead."
- **If asked:** viva Q31–Q35.

### Cell 19 · Random versus grouped split (code) · CORE · 1 min
- **Screen:** comparison table; sizes line.
- **Do:** point at "tracks in BOTH train and val".
- **Say:** "With a random split [leaked] tracks appear on both sides. With our grouped split, zero. We assert that in code. Validation now measures performance on signs the model has never seen."
- **Check:** grouped value must be 0; all 43 classes in validation must be True (the first run printed False, which is why the split was changed to a stratified one).

### Cell 20 · Result (markdown) · OPT · 0.5 min

---

## PART 5 — Preprocessing and augmentation (about 2 min)

### Cell 21 · Preprocessing and augmentation decisions (markdown) · CORE · 1 min
- **Screen:** two decision tables.
- **Say:** "64 by 64 cached once as small integers; normalisation and augmentation sit inside the model so the exported file accepts raw pixels. Augmentation is mild and realistic. Notice what is missing: horizontal flips."
- **If asked:** KB A17; viva Q15, Q36–Q40.

### Cell 22 · Caching images (code) · OPT · 0.5 min
- **Screen:** time taken, array shapes, memory.
- **Say:** "About half a gigabyte, so the whole dataset sits in memory and training is limited by the GPU, not the disk."

### Cell 23 · Augmentation preview and flip demo (code) · CORE · 1 min
- **Screen:** original and fifteen augmentations; flipped left/right sign.
- **Say:** "Augmented images remain recognisable. And here is why we never flip: a flipped left arrow looks like a right arrow but would keep its left label."

---

## PART 6 — Models (about 2.5 min)

### Cell 24 · Model design (markdown) · CORE · 1.5 min
- **Screen:** model A and B descriptions and tables.
- **Say:** "Two deliberately different models. A small CNN trained from scratch, about 116 thousand parameters. And MobileNetV2, a pretrained efficient network, trained in two stages: first only the new output layer, then fine-tuning the top layers at a lower learning rate. We expect a pretrained network to struggle with tiny sign crops at first; the ladder will test that."
- **If asked:** KB A10–A14, A18, A19; viva Q41–Q47.

### Cell 25 · Model definitions and summaries (code) · CORE · 1 min
- **Screen:** two model summaries with layer shapes and parameter counts.
- **Do:** point at the output shapes shrinking and at the parameter totals.
- **Say:** "The image shrinks from 64 to 32 to 16 to 8 as we pool; total parameters [ ]. MobileNetV2 has [ ] parameters, with only the new head trainable at this point."
- **Check:** MobileNetV2 trainable count is small before fine-tuning.

---

## PART 7 — Training (about 3 min)

### Cell 26 · Training recipe (markdown) · OPT · 0.5 min
- **Say:** "Adam optimiser, early stopping on validation accuracy keeping the best epoch, learning-rate reduction when progress stalls, and selection on validation only."

### Cell 27 · `fit`, plot helper, CNN training (code) · CORE · 1 min
- **Screen:** per-epoch log, then the CNN curves.
- **Do:** scroll to the figure.
- **Say:** "Blue is training, orange is validation. It stopped after [ ] epochs with best validation accuracy [ ]. If validation is above training early on, that is augmentation and dropout, which are active only in training."
- **If asked:** viva Q71.
- **Check:** curves rise and flatten; validation near training.

### Cell 28 · MobileNetV2 stage 1 (code) · OPT · 0.5 min
- **Say:** "Stage one trains only the new head."

### Cell 29 · MobileNetV2 stage 2 (code) · CORE · 1 min
- **Screen:** trainable parameter count, curves with the dashed boundary.
- **Say:** "Stage two unfreezes the last 50 layers at a lower learning rate, with batch-norm layers kept frozen. The dashed line is where fine-tuning starts; [describe the change in the curve]."
- **If asked:** viva Q44–Q47.

---

## PART 8 — Evaluation (about 4.5 min)

### Cell 30 · Evaluation introduction (markdown) · OPT · 0.5 min
- **Say:** "Metrics: accuracy, macro-F1, top-3, calibration error, latency and size. We also score the head-only model, a free ablation."

### Cell 31 · Metrics table and comparison chart (code) · CORE · 1.5 min
- **Screen:** table with highlighted best values; accuracy/F1 bars; accuracy versus latency bubbles.
- **Do:** read the Test accuracy and Macro-F1 columns, then the latency column.
- **Say:** "On the official test set the custom CNN scores [ ] and [ ]; MobileNetV2 head-only [ ]; fine-tuned [ ]. The gap between head-only and fine-tuned shows what fine-tuning bought. Parameters and latency show the cost."
- **If asked:** viva Q49–Q52.
- **Check:** do not claim one model is better if the numbers are within a fraction of a percent (single seed).

### Cell 32 · Model selection (code) · CORE · 0.5 min
- **Screen:** "Selected model (by validation accuracy)" line.
- **Say:** "We chose the model on validation accuracy. The test numbers printed next are reported once."

### Cell 33 · Per-class results (code) · CORE · 1 min
- **Screen:** F1 bars (red under 0.90), F1 versus class size, Spearman value, hardest classes table.
- **Say:** "Most classes are strong. The weakest are [names]. The correlation with class size is [ρ], so [imbalance does / does not] explain them."
- **If asked:** KB A21; viva Q53.

### Cell 34 · Confusion matrix and top confusions (code) · CORE · 1 min
- **Screen:** heat-map, top-ten table, bar chart.
- **Say:** "The diagonal is correct. The brightest off-diagonal cells are mistakes: [top pair] is the most common."
- **If asked:** viva Q54.

### Cell 35 · Errors inside families (code) · OPT · 0.5 min
- **Say:** "[ ]% of errors stay within the same sign family, against about [ ]% by chance. Mistakes are between similar signs."

### Cell 36 · Most confident errors (code) · OPT · 0.5 min
- **Say:** "These are the errors that matter in practice: confident and wrong. Many are dark, blurred or tiny."

---

## PART 9 — Progress ladder (about 2.5 min)

### Cell 37 · Ladder introduction (markdown) · CORE · 1 min
- **Screen:** step table.
- **Say:** "To show why the design works, we rebuild the story: start from a trivial baseline and change one thing per step, measured on validation. Steps 0 to 5 are from scratch, 6 and 7 transfer learning, 8 combines both."

### Cell 38 · Ladder rung training (code) · OPT · 0 min
- Do not narrate the training logs; go to the next cell.

### Cell 39 · Ensemble and ladder chart (code) · CORE · 1.5 min
- **Screen:** ladder table and the two-panel chart.
- **Do:** trace the black "best so far" line left to right.
- **Say:** "Guessing the most common class scores [ ]. A linear model on tiny pixel images reaches [ ]. Convolutions bring it to [ ], batch norm and a deeper network to [ ], augmentation and dropout to [ ]. Transfer learning starts at [ ] when frozen and rises to [ ] after fine-tuning. Averaging the two reaches [ ]. The right panel shows the same data as error rate on a log scale: error fell by a factor of about [ ]."
- **If asked:** viva Q57–Q60.
- **Check:** if any step is flat or lower, say so and give the caveat: short training budget, single seed.

---

## PART 10 — Explainability (about 2.5 min)

### Cell 40 · Explainability (markdown) · CORE · 0.5 min
- **Say:** "High accuracy does not prove the model looks at the right thing. Grad-CAM shows where it looked: red means strong influence."
- **If asked:** KB A25.

### Cell 41 · Grad-CAM grid (code) · CORE · 1.5 min
- **Screen:** top row inputs (five correct, five confident errors); rows of heat-maps for the custom CNN and MobileNetV2.
- **Do:** zoom on two correct examples, then on an error.
- **Say:** "On correct predictions the heat is on the symbol or rim of the sign [if that is what you see]. On this error it is [diffuse / on a small blurry symbol]. MobileNetV2's maps are blockier because its last feature map is only 4 by 4."
- **If asked:** viva Q61, Q62, Q106.
- **Check:** say only what the image shows.

### Cell 42 · t-SNE embeddings (code) · OPT · 0.5 min
- **Say:** "Each dot is a test image projected from the model's internal features. Clusters that touch are the classes it confuses, such as speed limits."
- **If asked:** KB A26; viva Q63.

---

## PART 11 — Reliability (about 3 min)

### Cell 43 · Reliability introduction (markdown) · CORE · 0.5 min
- **Say:** "A usable classifier must know when it is unsure and must degrade gracefully. Three checks: calibration, abstention and robustness."

### Cell 44 · Calibration and selective prediction (code) · CORE · 1.5 min
- **Screen:** confidence histograms, reliability diagram, coverage curve, printed thresholds.
- **Do:** read the "To reach ≥99% accuracy" line.
- **Say:** "Wrong predictions usually have lower confidence than right ones. So if we answer only above a threshold, to reach 99% accuracy the system answers [ ]% of images and defers on the rest. ECE is [ ]; bars below the diagonal mean over-confidence. One caveat: here the threshold is read off the test results to show the trade-off; a real system would set it on validation data."
- **If asked:** viva Q64–Q66, Q115.
- **Live option:** this cell is cheap to re-run.

### Cell 45 · Robustness (code) · CORE · 1 min
- **Screen:** six accuracy-versus-severity plots, example corrupted images, table.
- **Say:** "Accuracy under darkness, glare, noise, blur, rotation and occlusion at three severities. The model copes best with [ ] and worst with [ ]. These are synthetic corruptions, so they approximate but do not replace real field data."
- **If asked:** viva Q67.

---

## PART 12 — Live inference (about 1.5 min)

### Cell 46 · Live inference demo (markdown) · OPT · 0.2 min

### Cell 47 · `predict_sign` on four test images (code) · CORE · 1 min
- **Screen:** input, Grad-CAM, top-5 bars, for each image.
- **Do (live option):** re-run this cell; it is quick. Pick one example and read the top-5.
- **Say:** "Any image goes in; out come the top five classes with confidences and the heat-map. This one is [class] at [ ]%."
- **Check:** ground-truth is printed before each result.

### Cell 48 · Upload your own photo (code) · OPT · 0.5 min
- **Do:** only if rehearsed: tick `RUN_UPLOAD`, upload a clear sign photo, run.
- **Say:** "Here is a sign photographed by us." If it fails, say that a model trained on cropped GTSRB images can fail on uncropped phone photos; this is the detection limitation.

---

## PART 13 — Export and checks (about 1.5 min)

### Cell 49 · Export and production-readiness (markdown) · OPT · 0.5 min
- **Say:** "Training is not shipping. We save every model, reload them, and compare predictions; and we test TFLite conversion."

### Cell 50 · Save models and verify reload (code) · CORE · 0.5 min
- **Screen:** "Reload parity OK" lines with sizes.
- **Say:** "Saved and reloaded models give the same predictions within 1e-4."

### Cell 51 · TFLite conversion (code) · OPT · 0.3 min
- **Say:** "The converted mobile-sized model is [ ] MB and agrees with the original on [ ]% of 500 images."
- **Check:** if this printed "skipped", say conversion is optional and move on.

### Cell 52 · Bundle and download (code) · CORE · 0.3 min
- **Screen:** list of artifacts; download prompt.
- **Say:** "Everything is bundled so we can re-test without retraining."

---

## PART 14 — Wrap-up (about 2 min)

### Cell 53 · Wrap-up heading (markdown) · CORE
### Cell 54 · Summary printout (code) · CORE · 0.5 min
- **Screen:** selected model and headline metrics.
- **Say:** "In this run [model] reaches [ ]% accuracy and [ ]% macro-F1 on the official test set, ECE [ ]."

### Cell 55 · Decision log, limitations, future work, references (markdown) · CORE · 1.5 min
- **Do:** scroll the decision log slowly; stop on limitations.
- **Say:** "Thirteen decisions, each with the alternative we rejected. The limits: cropped signs only, one dataset, one seed, not safety-certified. Next: a detector, calibrated confidences, multiple seeds and non-German signs."
- **If asked:** KB Part D; viva Q101–Q115.
- **Closing line:** "We did not just chase accuracy; we made the evaluation honest, explained the model's behaviour, and checked it is reliable and reusable. Thank you; we are happy to take questions."

---

## Time plan

| Section | Core cut | Full |
|---|---:|---:|
| 1 Opening and problem | 2.5 | 3 |
| 2 Setup and data | 2 | 3.5 |
| 3 EDA | 3.5 | 4.5 |
| 4 Validation design | 2.5 | 3 |
| 5 Preprocessing and augmentation | 2 | 3 |
| 6 Models | 2.5 | 2.5 |
| 7 Training | 2.5 | 3 |
| 8 Evaluation | 4 | 5 |
| 9 Progress ladder | 2.5 | 2.5 |
| 10 Explainability | 2 | 2.5 |
| 11 Reliability | 3 | 3 |
| 12 Live inference | 1 | 1.7 |
| 13 Export | 1 | 1.7 |
| 14 Wrap-up | 2 | 2 |
| **Total** | **about 31** | **about 41** |

To reach the 20-minute express cut: show Parts 3, 8, 9, 10 and 11 in depth, give Parts 1, 2, 5, 6, 7 and 13 one sentence each, and skip all OPT cells. Decide the exact cut in your mock run and write the final order here:

`Final order for our presentation: ____________________________________________`

---

## Number capture sheet (filled from the final run; see `04_results_and_findings.md`)

| Item | Where in notebook | Value |
|---|---|---|
| GPU | cell 4 | Colab GPU:0 (model name not printed); TensorFlow 2.21.0 |
| Training tracks / frames per track | cells 7, 16 | 1,307 / 30 |
| Largest class / smallest class / ratio | cell 9 | 2,250 (Limit 50) / 210 (Limit 20) / 10.7 times |
| Median image width / share under 40 px | cell 12 | 43 px / 40% |
| Tracks on both sides, random split | cell 19 | 1,291 of 1,307 (100% of validation tracks seen in training) |
| Grouped split | cell 19 | 33,570 fit, 5,639 validation (188 tracks), 0 shared, all 43 classes |
| Custom CNN parameters | cell 25 | 116,203 |
| MobileNetV2 parameters / trainable at stage 1 / stage 2 | cells 25, 29 | 2,313,067 / 55,083 / 1,834,219 |
| CNN epochs run / best validation accuracy | cell 27 | 29 / 97.61% |
| MobileNetV2 epochs head / fine-tune | cells 28, 29 | 12 / 14 |
| Validation accuracy: CNN / head-only / fine-tuned | cell 31 | 97.61% / 72.58% / 90.55% |
| Test accuracy: CNN / head-only / fine-tuned | cell 31 | 96.96% / 75.16% / 91.26% |
| Test macro-F1: CNN / head-only / fine-tuned | cell 31 | 95.21% / 67.46% / 86.59% |
| Test top-3 and ECE of the selected CNN | cell 31 | 99.03% / 0.0105 |
| Latency CNN / MobileNetV2 | cell 31 | 18.0 / 153.0 ms |
| Selected model | cell 32 | Custom CNN |
| Spearman (class size versus F1) | cell 33 | +0.28 |
| Three weakest classes | cell 33 | Pedestrians 0.65, Dangerous curve to the right 0.82, Double curve 0.83 |
| Top confusion | cell 34 | Limit 60 to Limit 80 (36 images) |
| Errors within the same family / chance | cell 35 | 75% of 384 errors / about 23% |
| Ladder validation accuracy, steps 0 to 8 | cell 39 | 5.9, 70.2, 83.2, 93.4, 95.1, 97.6, 72.6, 90.5, 97.1 |
| Error reduction factor, baseline to best | cell 39 | about 39 times |
| Ensemble test accuracy / macro-F1 | cell 39 | 96.88% / 95.07% (best single model 96.96%) |
| Coverage for 99% accuracy | cell 44 | 95.5% at threshold 0.866 |
| Worst / best corruption at the severe level | cell 45 | noise 14.1% and occlusion 9.5% / blur 95.0% |
| TFLite size / agreement / speed | cell 51 | 0.24 MB / 100% / 1.24 ms |
| Results bundle size | cell 52 | about 86.5 MB |

---

## Hand-offs and timing cues

- **Handoff points (if two narrators):** Narrator A takes Parts 1–5 (problem, data, split, preprocessing); Narrator B takes Parts 6–14 (models, training, evaluation, ladder, explainability, reliability, export). The Driver stays constant.
- **Pace marks (core cut):** by minute 5 you should be at the track strip (cell 16); by minute 8 at the models (cell 25); by minute 12 at the metrics table (cell 31); by minute 15 at the ladder (cell 39); by minute 18 at reliability (cell 44); wrap up by minute 20.
- **If you are behind:** skip cells 11, 14, 22, 35, 36, 42, 48, 51. If you are very behind, skip all of Part 10's t-SNE and cell 45's example images.
- **If you are ahead:** run cell 47 live on extra images, or tick `RUN_UPLOAD`.

---

## Handling surprises during the demo

| Situation | Response |
|---|---|
| A cell shows an error | Do not debug live. Say "this cell needs a rerun", move on, and show the saved figure from the results bundle |
| A result is lower than the story predicts | Say so: "In this run, [observation]." The ladder and notes already explain single-seed noise and training budgets |
| A flat or negative ladder step | Report it as found and give the caveat; do not hide it |
| Runtime disconnected | Open the executed `.ipynb` copy or the saved figures; continue from there |
| An unexpected question | Use the four-part answer in `02_viva_readiness.md` section 1.1 |
