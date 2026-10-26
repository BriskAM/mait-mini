# Presenter Notes — GTSRB Demo Notebook

Companion to `notebooks/traffic_sign_recognition_demo.ipynb`. For the full presenter set, see `docs/presenter/` (`01_knowledge_base.md`, `02_viva_readiness.md`, `03_presenter_pipeline.md`); this file is the short quick-reference. Everything here is meant for the presenter only; the notebook itself contains just the pipeline, the rationale for each decision, and the results it computes.

> **Verify before you say it.** The interpretive statements below were written before the notebook had been run end to end. Treat each as a *hypothesis to confirm against the live output*, and reword anything the numbers do not support.

---

## 1. Before the demo

**Do a full dry run on Colab at least once**, ideally the day before. The notebook was generated and syntax-checked but not executed during development, so a first run may surface an environment-specific error.

| Item | Detail |
|---|---|
| Runtime | Runtime ▸ Change runtime type ▸ **T4 GPU** |
| Mode | `MODE` form field in section 2. `demo` is the default (my estimate: 10–20 min on a T4, not measured). `full` trains longer for final numbers. `smoke` is a ~2 min sanity run and is what the notebook falls back to without a GPU |
| Strategy | Run **all cells before the session** so outputs are already on screen; during the panel just scroll and explain. Re-run live only the cheap cells (section 12 inference, section 11 selective prediction) |
| Download | ~320 MB from Zenodo. If Zenodo is slow or down, set the `GTSRB_DIR` environment variable to an existing copy of the dataset |
| Output | The last cell downloads `gtsrb_demo_results.zip` (models, results, figures). Untick `DOWNLOAD` to skip |
| Optional live moment | Tick `RUN_UPLOAD` in section 12 and upload a photo of a sign |

**Suggested timing (about 20 minutes):** problem and workflow 2 · data and EDA 4 · validation split 2 · models and training 3 · evaluation and progress ladder 5 · explainability 2 · reliability 2 · inference and export 1 · wrap-up and questions.

---

## 2. Talk track per section

### Section 4 — EDA
- Lead with: *every EDA finding changes a decision.* Class imbalance → macro-F1. Small images → 64×64. Lighting range → contrast augmentation. Tracks → grouped split.
- The track strip is the single most persuasive slide: it makes the leakage problem obvious before you explain it.

### Section 5 — Grouped split
- Point at the table: with a random split, almost every validation track was also seen in training; the grouped split makes that zero.
- Message: *validation accuracy from a random split would be flattering and meaningless.*

### Section 8 — Training curves (moved from notebook)
Training accuracy can sit below validation accuracy early on because augmentation and dropout are active during training but not during validation. A gap that narrows later indicates healthy regularisation rather than over-fitting.

### Section 9 — Results (moved from notebook)
- **Head-only → fine-tuned** isolates what fine-tuning bought. A large jump supports the domain-gap explanation: a frozen ImageNet backbone has never seen 64-pixel sign crops, so a new head alone cannot close that gap.
- The **custom CNN** is expected to give the best accuracy per millisecond, since GTSRB signs are simple and low-resolution and a tiny purpose-built network can match or beat a general-purpose backbone at a fraction of the cost. *Confirm against the latency column.*
- Model choice was made on **validation** accuracy; test numbers are reported once. Say this explicitly.
- **Earlier full-length run** (from `docs/project_report.md`, for reference only; not reproduced by this notebook):

  | Model | Test accuracy | Macro-F1 |
  |---|---:|---:|
  | Custom CNN, 30 epochs | 97.12% | 95.14% |
  | MobileNetV2, frozen backbone, 25 epochs | 76.18% | 68.23% |

  That run used a frozen backbone at a 128 px resize of the original images; the notebook uses a 64 px cache upsampled inside the model plus a fine-tuning stage, so numbers will not match exactly.
- **Per-class F1 plot:** the dashed line is macro-F1; red bars are classes under 0.90. The printed Spearman correlation answers whether class size explains weak classes. If it is weak, the cause is visual similarity rather than data scarcity.

### Section 9.3 — Confusion and errors (moved from notebook)
- Mistakes tend to concentrate in visually near-identical families, for example speed limits that differ by a single digit at roughly 30 px. Check the "same sign family" percentage printed in the notebook.
- Many confident errors are low-quality images (very dark, blurred, tiny) where even a person would hesitate.
- The implied fixes are more detail for digit-bearing signs and rejecting low-confidence predictions, rather than simply more of the same data.

### Section 9A — Progress ladder
- This is the "how we got here" slide. Walk left to right: majority-class baseline, linear, MLP, CNN, deeper CNN with BatchNorm, augmentation and dropout, then the transfer-learning track, then the ensemble.
- The black step line is the best validation accuracy so far, so it only goes up. The right panel shows the same data as error rate on a log scale, which makes the late gains (for example 3% to 1% error) visible; accuracy bars alone hide them.
- Be clear that the ladder is on **validation** data, and that steps 5-7 reuse the models from section 8.
- The two tracks are intentional. The frozen-backbone MobileNetV2 may score *below* the from-scratch CNN (the earlier run did: 76% vs 97%); fine-tuning is what closes that gap. Present this as a finding, not a failure.
- **What the final full run showed (see `presenter/04_results_and_findings.md`):** the from-scratch track rose at every step, from 5.9% to 97.6% validation accuracy. The frozen MobileNetV2 started at 72.6% and reached 90.5% after fine-tuning, but never passed the custom CNN. The ensemble (97.1% validation, 96.9% test) did not beat the CNN (97.6% and 97.0%), so the one-model choice stands.
- Do not hide the flat "best so far" line after step 5. It is the finding: for this task a small purpose-built CNN beat a larger pretrained model.


### Section 10 — Explainability (moved from notebook)
- **Grad-CAM:** on correct predictions, expect heat on the symbol and rim (digits, arrow, triangle edge) rather than road or sky. On confident errors it is often diffuse or on a small blurry symbol. *Only claim this if the displayed images show it.*
- The custom CNN's heat-map is a 16×16 grid; MobileNetV2's is 4×4, so it is blurrier by design. Say this proactively so nobody reads it as a bug.
- **t-SNE:** tight, well-separated clusters mean a discriminative representation. Touching clusters (typically the speed-limit family) are the classes the model finds hard, and should line up with the confusion table.

### Section 11 — Reliability (moved from notebook)
- **Calibration and abstention:** wrong predictions usually carry lower confidence than correct ones, so a threshold trades a little coverage for a lot of accuracy. Quote the printed "to reach 99% accuracy you answer X% of images" line. This is the practical safety mechanism for a driver-assist setting.
- If the reliability diagram shows over-confidence, mention temperature scaling as the cheap standard fix.
- **Robustness (final run):** blur (95.0% at the severe level), darkening (82%) and glare (85%) are handled well. The steepest drops are **Gaussian noise** (34% at sigma 25), **heavy occlusion** (about 10% when 35% of the area is hidden) and **large rotation** (52% at 35 degrees). Noise is the notable weakness because the model was never trained with it; adding it to the augmentation is the first improvement to propose.
- These are **synthetic** corruptions; say so.

---

## 3. Anticipated panel questions (moved from notebook)

**Why not just use the biggest pretrained model?**
Signs are low-resolution and simple. Section 9 shows the size and latency cost. The small CNN is cheaper to deploy and, in the earlier recorded run, more accurate than the frozen backbone.

**Is 97% good enough?**
Competitive for a compact model on GTSRB, but accuracy hides asymmetric risk (confusing a 30 with an 80). That is why section 11 adds abstention and robustness analysis.

**Could the validation score be inflated?**
Not by track leakage: section 5 shows the grouped split removes it. The final number comes from the untouched official test set.

**How do you know it is not reading the background?**
Grad-CAM in section 10, subject to what the displayed heat-maps actually show.

**Can it run on a car's hardware?**
Section 13 converts it to TFLite and checks prediction agreement; the model is small. Quote the printed size and latency. This is not a benchmark on automotive hardware.

**Why no horizontal-flip augmentation?**
Flipping turns "turn left ahead" into something that looks like "turn right ahead" but keeps the old label. The notebook demonstrates it.

**Why is the validation set smaller than a typical 20%?**
One of seven folds of tracks (about 14%), to keep most tracks for training while still holding out whole signs.

**What would you do next?**
Add a detector for full scenes, calibrate confidences, and test on non-German signs.

---

## 4. Being upfront about limitations

These are also listed in the notebook's wrap-up; here is how to frame them.

- **Single seed.** One training run per model. A production claim would average several seeds with confidence intervals. Do not present small differences between models as significant.
- **Cropped signs only.** Real driving needs a detector first (the repository has an optional YOLO extension); end-to-end performance is bounded by it.
- **One country, one dataset.** Domain shift to other regions is untested.
- **Not safety-certified.** A research pipeline, not validated against automotive standards such as ISO 26262 or SOTIF.
- **Class weights and focal loss** were not explored for the rarest classes.
- **The notebook's numbers come from a single Colab run** and will vary slightly between runs and GPUs.

---

## 5. If something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| "No GPU detected" banner | Wrong runtime. Switch to T4 and rerun from the top. The notebook auto-switches to `smoke` so it still finishes, but the numbers are not meaningful |
| Checksum mismatch | Truncated download. Delete `data.zip` under `/content/gtsrb_demo` and rerun the download cell |
| ImageNet weights warning | No internet for the weights download. MobileNetV2 falls back to random initialisation, so its numbers will be poor; reconnect and rerun |
| TFLite conversion skipped | Non-fatal; everything else is unaffected |
| Out of memory | Switch `MODE` to `smoke`, or restart the runtime and run from the top |
| Wrong class shown in the left/right flip demo | Expected: it illustrates why flips are not used |

---

## 6. Ideas to improve further (not in the notebook)

Roughly in order of expected value for effort. None have been tried, so no gains are claimed.

| Idea | What it targets | Effort |
|---|---|---|
| Longer training in `full` mode, several seeds, report mean and spread | Credibility of every number; separates real gains from noise | Low |
| Higher input resolution for the digit-bearing speed-limit signs (64 vs 96 vs 128) | The most common confusion family | Low |
| Class weights, focal loss or class-aware sampling | Weak rare classes (check the F1 vs class size plot first) | Low |
| Test-time augmentation (average predictions over small shifts and zooms) | Cheap accuracy and calibration gain at inference | Low |
| Temperature scaling on the validation set | Over-confident probabilities; better abstention thresholds | Low |
| Label smoothing, cosine learning-rate schedule, AdamW | Generalisation | Low |
| Stronger augmentation (random erasing, brightness, blur, noise matched to section 11) | Robustness to the corruptions that hurt most | Medium |
| A larger fine-tuned backbone (EfficientNet-B0, ResNet-18) | Accuracy ceiling, at a latency cost | Medium |
| Knowledge distillation from the ensemble into the small CNN | Ensemble accuracy at single-model cost | Medium |
| Spatial-transformer or CLAHE contrast normalisation | Dark and low-contrast images | Medium |
| Detection front end (YOLO) for full scenes and video | Real-world usability; out of current scope | High |
| Out-of-distribution detection and non-German sign evaluation | Safety and domain shift | High |
