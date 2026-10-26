# Presenter Knowledge Base — GTSRB Traffic-Sign Recognition

**Audience:** presenters who know the basics of machine learning and need to be able to explain *everything* in the demo notebook, in plain language, under questioning.

**Companion documents**
- `02_viva_readiness.md` — question bank, answer technique, drills, checklists
- `03_presenter_pipeline.md` — cell-by-cell run of show for `notebooks/traffic_sign_recognition_demo.ipynb`
- `../demo_notebook_presenter_notes.md` — short talk-track and troubleshooting sheet

**How to read this file.** Part A explains every concept from the ground up, each with a plain-English definition, an analogy, where it appears in our notebook, and a one-sentence answer you can say out loud. Part B is the fact sheet (every number and setting). Part C walks through the code. Part D covers the design decisions and their honest weaknesses. Part E explains how to read each output. Part F is the glossary and formula crib.

> **Rule for numbers.** Anything this document calls a *setting* is read directly from the notebook source and is safe to quote. Anything that is a *result* (accuracy, F1, and so on) comes from your own Colab run; take it from the printed output, never from memory or from this file. Dataset facts taken from the GTSRB paper are marked "(paper)" so you can attribute them.

---

## Part 0 — The project in 30 seconds, 2 minutes, and 5 minutes

**30 seconds.** We built a pipeline that looks at a cropped photo of a German road sign and says which of 43 sign types it is. We verified the data, studied it, split it carefully so validation is honest, trained two kinds of neural network, compared them, checked what they look at, tested how they handle bad conditions, and exported them so they can be reused.

**2 minutes.** The dataset is GTSRB, 39,209 training and 12,630 test images across 43 classes. Our key data finding is that the training images are not independent: they come from about 1,300 physical signs, each photographed around 30 times as a car approaches. If you split images at random, near-duplicates end up in both training and validation and the validation score is inflated, so we split by physical sign. We train a small custom CNN from scratch and a pretrained MobileNetV2 in two stages (new head first, then fine-tuning). We pick the final model using validation data only, and report on the official test set once. Beyond accuracy we examine per-class results, confusions, Grad-CAM heat-maps, confidence calibration, behaviour under darkness, noise, blur, rotation and occlusion, and we export and re-load every model to prove it works.

**5 minutes.** Add: the progress ladder (baseline to final, one change per step), the ensemble, the abstention idea (answer only when confident), the export bundle, and the limitations (cropped signs only, one dataset, single seed, not safety-certified).

**The one sentence to remember.** *We did not just chase accuracy; we made the evaluation honest, explained the model's behaviour, and checked it is reliable and reusable.*

---

# Part A — Concepts from the ground up

Each entry: **What it is** / **Analogy** / **In our notebook** / **Say it in one sentence**.

## A1. How a computer sees an image
**What it is.** A colour image is a grid of pixels; each pixel has three numbers (red, green, blue) from 0 to 255. A 64×64 colour image is therefore an array of shape 64×64×3 = 12,288 numbers.
**Analogy.** A spreadsheet with three layers, one per colour.
**In our notebook.** We store every image as a 64×64×3 array of `uint8` (whole numbers 0–255). Models receive these raw numbers and rescale them internally.
**One sentence.** *To the model a sign is just a 64 by 64 by 3 block of numbers.*

## A2. Classification, classes, labels
**What it is.** Classification means choosing one label out of a fixed list. Here there are 43 classes (class 0 is "Speed limit 20", class 14 is "Stop", and so on). A label is the correct class number for an image.
**In our notebook.** `CLASS_NAMES` is the list of 43 names; `label`/`ClassId` are integers 0–42.
**One sentence.** *The task is to map each image to one of 43 sign types.*

## A3. Softmax and probabilities
**What it is.** The network's last layer produces 43 raw scores. Softmax converts them into 43 numbers that are positive and sum to 1, which we read as probabilities. The largest is the prediction, and its value is the model's *confidence*.
**Formula.** `softmax(z_i) = exp(z_i) / sum_j exp(z_j)`.
**In our notebook.** The final layer is `Dense(43, activation="softmax")`. "Confidence" in sections 11–12 is the maximum softmax probability.
**Caution.** A softmax probability is not automatically a trustworthy probability; that is what calibration (A22) checks.
**One sentence.** *Softmax turns the network's scores into probabilities that add up to one.*

## A4. Loss function and cross-entropy
**What it is.** The loss is a single number measuring how wrong the model is; training tries to make it small. For classification we use cross-entropy: it is `-log(probability the model gave to the correct class)`. Giving the right class probability 1.0 costs 0; giving it 0.01 costs a lot.
**In our notebook.** `loss="sparse_categorical_crossentropy"`. "Sparse" only means the labels are integers (0–42) instead of one-hot vectors.
**One sentence.** *Cross-entropy punishes the model heavily when it is confidently wrong.*

## A5. Gradient descent, optimiser, learning rate (Adam)
**What it is.** Training adjusts millions of numbers (weights) to reduce the loss. The gradient says which direction reduces the loss; the learning rate is the step size. Adam is a popular optimiser that adapts the step for each weight using running averages of past gradients.
**Analogy.** Walking down a foggy hill by feeling the slope under your feet; the learning rate is your stride length.
**In our notebook.** Adam with learning rate 1e-3 for the CNN and for MobileNetV2 stage 1; 1e-4 for stage 2 fine-tuning. A `ReduceLROnPlateau` callback shrinks the rate when validation loss stalls.
**One sentence.** *Adam nudges the weights downhill on the loss, with a step size we reduce when progress stalls.*

## A6. Epoch, batch, iteration
**What it is.** An epoch is one full pass over the training data. We process data in batches (128 images here) and update the weights after each batch.
**In our notebook.** `BATCH = 128`. Epoch counts are *upper limits*; early stopping usually ends sooner.
**One sentence.** *An epoch is one pass through the training images, done in batches of 128.*

## A7. Overfitting and underfitting
**What it is.** Overfitting: the model memorises training images and does worse on new ones (training accuracy high, validation accuracy lower and not improving). Underfitting: the model is too simple or under-trained and does poorly on both.
**Tell-tale on a curve.** Training and validation lines diverge (over-fitting) or both stay low (under-fitting).
**In our notebook.** Fought with augmentation, dropout, batch normalisation, early stopping and a small model. Section 8 plots the curves.
**One sentence.** *Overfitting is memorising the training set instead of learning the pattern.*

## A8. Training, validation and test sets
**What they are.** Training data teaches the model. Validation data is held out to make decisions (when to stop, which model to pick). Test data is touched once at the end to estimate real-world performance.
**Why separate.** If you tune on the test set, its score becomes optimistic because you have indirectly fitted to it.
**In our notebook.** Official train/test split from the benchmark. A validation set is carved out of training data using whole physical signs (A20). Model selection uses validation only.
**One sentence.** *Validation is for decisions and test is for the final, one-time report card.*

## A9. Neural networks in one paragraph
**What it is.** A network is layers of simple units. Each layer multiplies its input by weights, adds a bias and applies a non-linear function (such as ReLU). Stacking layers lets the network build complex features from simple ones: edges, then shapes, then whole symbols.
**One sentence.** *A neural network learns a stack of increasingly abstract features from raw pixels.*

## A10. Convolution, filters, feature maps
**What it is.** A convolutional layer slides small filters (here 3×3) across the image. Each filter detects one local pattern (an edge, a corner, a curve) wherever it occurs; the output is a feature map showing where that pattern was found. Filters are learned, not hand-designed.
**Why for images.** The same pattern matters anywhere in the image, so sharing filters across positions needs far fewer weights than connecting every pixel to every unit, and respects spatial structure.
**Parameter count.** A conv layer with `k×k` filters, `C_in` input channels and `C_out` filters has `k*k*C_in*C_out + C_out` parameters. Example: conv2 in our CNN is 3*3*32*64 + 64 = 18,496.
**In our notebook.** The custom CNN has three conv layers with 32, 64 and 128 filters. `padding="same"` keeps the map size unchanged.
**One sentence.** *Convolution scans small learned pattern detectors across the whole image.*

## A11. ReLU
**What it is.** `ReLU(x) = max(0, x)`. A cheap non-linearity that lets networks model complex functions and trains fast.
**In our notebook.** The activation after each conv and in the dense layer.

## A12. Pooling (max pooling, global average pooling)
**What it is.** Max pooling keeps the strongest response in each small window (2×2 here), halving width and height. It reduces computation and gives some tolerance to small shifts. Global average pooling averages each whole feature map into one number, turning a 3-D block into a vector.
**In our notebook.** Three max-pool steps take 64 to 32 to 16 to 8 pixels. Then global average pooling gives a 128-number vector.
**Why GAP instead of flatten + dense.** Far fewer parameters (less over-fitting) and it keeps feature maps intact, which Grad-CAM relies on.
**One sentence.** *Pooling shrinks the maps while keeping the strongest signals.*

## A13. Batch normalisation (BN)
**What it is.** Normalises each layer's activations to a stable range during training (using batch statistics), then rescales with learned parameters. This makes training faster and more stable. At inference it uses stored running averages instead of batch statistics.
**Why it matters for fine-tuning.** When we fine-tune MobileNetV2 we keep BN layers frozen so the pretrained running statistics are not disturbed by small, different data.
**One sentence.** *BatchNorm keeps the numbers flowing through the network well-scaled, which speeds and steadies training.*

## A14. Dropout
**What it is.** During training, randomly switch off a fraction of units (0.2 or 0.3 here) each step, so the network cannot rely on any single unit. Turned off at inference.
**Analogy.** Practising a team sport while random players sit out, so everyone learns to contribute.
**One sentence.** *Dropout randomly silences units during training to reduce memorisation.*

## A15. Dense (fully connected) layer
**What it is.** Every input is connected to every output. We use one of 128 units as the learned "embedding" and a final one of 43 units for the class scores.

## A16. Parameters
**What it is.** The learned numbers. Our custom CNN has on the order of 116 thousand; MobileNetV2 with our head has on the order of 2.3 million. The notebook prints exact counts; quote those. Trainable parameters are those being updated (frozen layers are not).
**One sentence.** *Parameters are the adjustable weights; fewer means smaller and faster.*

## A17. Data augmentation
**What it is.** Creating slightly altered training images on the fly (small rotation, shift, zoom, contrast change) so the model learns to be robust and sees more variety than the raw dataset has.
**Our settings.** Rotation ±14 degrees (factor 0.04 of a full turn), translation ±5%, zoom ±8%, contrast ±10%. Active only in training.
**Why no flip.** Flipping changes meaning: "Turn left ahead" becomes something that looks like "Turn right ahead" but keeps the old label.
**One sentence.** *Augmentation shows the model realistic variations of each image, but never ones that change the sign's meaning.*

## A18. Transfer learning, frozen vs fine-tuned
**What it is.** Start from a network already trained on a large dataset (ImageNet, over a million photos of everyday objects) because its early layers detect general patterns (edges, textures, shapes) useful for almost any image task. You attach a new output layer for your own classes.
- **Frozen backbone / "head only":** pretrained layers are not changed; only the new output layer learns.
- **Fine-tuning:** you then unfreeze some of the later pretrained layers and train them gently (a 10 times smaller learning rate) so they specialise to your data.
**Why two stages.** Training a random new head while the backbone is also updating can wreck the pretrained features with large, noisy gradients. Warm up the head first, then fine-tune.
**Domain gap.** ImageNet photos look different from tiny, low-resolution sign crops, so a frozen backbone is a poor fit and fine-tuning helps. (This is a hypothesis we test in the ladder.)
**One sentence.** *Transfer learning reuses knowledge from a big dataset, and fine-tuning adapts it to our specific signs.*

## A19. MobileNetV2
**What it is.** An efficient CNN designed for phones (Sandler et al., 2018). Its key ideas: depthwise separable convolutions (split a convolution into a cheap per-channel step plus a 1×1 mixing step) and inverted residual blocks with linear bottlenecks. It has roughly 2.3 million parameters in the standard model, versus tens of millions for large networks.
**In our notebook.** Input is the 64×64 image upsampled to 128×128 inside the model; the output is 1,280 features, fed to a dropout and a 43-way dense layer.

## A20. Data leakage and the grouped split (the most important idea)
**What it is.** Leakage happens when information from the evaluation data sneaks into training, making scores look better than reality.
**Our case.** The 39,209 training images come from about 1,307 physical signs (tracks), each photographed about 30 times. Consecutive frames look almost identical. A random image split puts frames of the same sign in both training and validation; the model "recognises" a sign it effectively memorised.
**Our fix.** a per-class track holdout: for each class, about one seventh of its tracks (at least one) go to validation, chosen with a seeded random generator; group = class + track id. This is about 14% of the data and guarantees every class is represented. All frames of a sign go wholly to training or wholly to validation. The notebook proves it: it counts tracks that appear in both sets for a random split versus the grouped split (the latter is zero) and asserts it.
**One sentence.** *We split by physical sign, not by image, so validation measures performance on signs the model has never seen.*

## A21. Class imbalance, accuracy vs macro-F1
**What it is.** Imbalance means some classes have far more images than others (the largest class has roughly ten times the smallest). Accuracy can then look good even if rare classes fail.
- **Precision** of a class: of the images predicted as that class, how many were right.
- **Recall** of a class: of the images truly in that class, how many were found.
- **F1** is their harmonic mean: `2PR/(P+R)`. It is high only when both are high.
- **Macro-F1** averages F1 over classes with equal weight, so rare classes count as much as common ones. **Weighted-F1** weights by class size.
**Top-3 accuracy.** Correct if the true class is among the three most likely predictions.
**One sentence.** *We report macro-F1 next to accuracy so rare sign types cannot hide behind common ones.*

## A22. Calibration and ECE
**What it is.** A model is calibrated if, among predictions with confidence about 80%, about 80% are correct. Many neural networks are over-confident. **Expected Calibration Error (ECE)** groups predictions into confidence bins and averages the gap between confidence and actual accuracy per bin (lower is better). Our ECE uses 15 bins; the reliability diagram draws 10.
**Fix if poor.** Temperature scaling (divide the scores by a single value learned on validation data).
**In our notebook.** Section 11.1.

## A23. Selective prediction (abstention)
**What it is.** Let the system refuse to answer when confidence is below a threshold, passing the decision to a human or another sensor. As the threshold rises, the system answers fewer images (coverage drops) but is more often right on those it answers.
**In our notebook.** The coverage-versus-accuracy curve and printed thresholds for 99%, 99.5% and 99.9% accuracy. Thresholds are chosen on the test results shown in the notebook for illustration; a real deployment would set them on validation data.
**Honest note.** Because we pick the threshold by looking at the test set here, treat it as a demonstration of the trade-off, not a tuned operating point.

## A24. Robustness testing
**What it is.** Deliberately corrupt test images (darker, brighter, noisy, blurred, rotated, partly covered) at three severities and measure accuracy. It shows which real-world conditions hurt the model. Corruptions are synthetic approximations.

## A25. Grad-CAM (explainability)
**What it is.** For one image and one predicted class, take the last convolutional feature maps, weight each map by how strongly the class score depends on it (the average gradient), sum, and keep only positive influence. The result is a coarse heat-map of where the network "looked" (Selvaraju et al., 2017).
**Resolution.** The map is the size of the last conv layer: 16×16 for our CNN, 4×4 for MobileNetV2 (so it looks blurrier). It is then enlarged to image size.
**Limits.** It shows influence, not proof of correct reasoning; a heat-map on the sign does not guarantee the right features, and a diffuse map does not prove failure.
**One sentence.** *Grad-CAM paints where in the image the model's decision came from.*

## A26. Embeddings and t-SNE
**What it is.** The "embedding" is the vector just before the final layer (the model's internal summary of an image). t-SNE squashes these high-dimensional vectors into 2-D for plotting while trying to keep similar images close.
**Caution.** t-SNE distances and cluster sizes are not literally meaningful; use it only to see grouping, and note it is a visualisation, not a measurement.

## A27. Ensembling
**What it is.** Combine several models; here we average their predicted probabilities. If the models make different mistakes, the average is more accurate than either. No extra training is needed.
**Cost.** Two models to run: roughly the sum of their latency and size.

## A28. Model export, TFLite, float16, parity
**What it is.** Saving the trained model to a file (`.keras`) so others can load it. TensorFlow Lite is a compact format for phones and embedded devices; float16 stores weights in half precision to roughly halve the size at small accuracy cost. A *parity check* confirms the reloaded or converted model gives the same predictions as the original.

## A29. Reproducibility
Fixed random seed (42), pinned dataset by MD5 checksum, one configuration dictionary, and a single `MODE` switch. Seeds make runs repeatable on the same hardware, not identical across different GPUs or library versions.

## A30. Checksums (MD5)
A fingerprint of a file. If even one byte of the download differs, the fingerprint changes. We compare against the published value `7faafd0d78d831c7eb4d555641a83a62`. Note: MD5 here protects against corruption, not malicious tampering.

---

# Part B — Fact sheet

## B1. Dataset (GTSRB)
| Fact | Value | Source |
|---|---|---|
| Name | German Traffic Sign Recognition Benchmark | IJCNN 2011 competition (paper) |
| Training images | 39,209 | asserted in notebook |
| Official test images | 12,630 | asserted in notebook |
| Classes | 43 | asserted in notebook |
| Training tracks (physical signs) | about 1,307, around 30 frames each | printed in notebook |
| Class size range | roughly 210 to 2,250 images (about 10 times) | printed in notebook (section 4.1) |
| Image sizes | 25 to 243 pixels wide in the training metadata, median 43 px, 40% under 40 px | printed in notebook |
| Archive | Zenodo record 13741936, about 320 MB | notebook |
| MD5 | `7faafd0d78d831c7eb4d555641a83a62` | notebook |
| Human performance | about 98.8% (paper) | quote only with attribution |
| Best published committee-of-CNNs | about 99.5% (paper) | quote only with attribution |

The test set is a separate set of images provided by the benchmark for evaluation; it is intended to be independent of the training signs. We rely on the benchmark's design for that and did not independently verify track separation.

## B2. Every setting in the notebook (Full mode)
| Setting | Value | Why |
|---|---|---|
| Seed | 42 | repeatability |
| Cache resolution `IMG` | 64×64 | signs are small; more pixels add no information |
| MobileNetV2 input `MB_SIZE` | 128×128 (upsampled inside the model) | pretrained network works better with larger inputs |
| Batch size | 128 | fits in memory, stable gradient |
| Validation | about 1/7 of each class's tracks (at least 1), seeded `numpy` generator (42); about 14% of images | leakage-free and every class represented |
| Augmentation | rotation 0.04 (about ±14°), translation 5%, zoom 8%, contrast 10% | mild, realistic |
| Custom CNN | 3 blocks (32, 64, 128 filters), 3×3, ReLU, BN, 2×2 max-pool, dropout 0.2; global average pool; dense 128 (ReLU); dropout 0.3; dense 43 softmax | small, regularised |
| CNN training | Adam 1e-3, up to 30 epochs, early stopping patience 5 on val accuracy, ReduceLROnPlateau factor 0.4 patience 2 min 1e-6 | |
| MobileNetV2 stage 1 | frozen backbone, dropout 0.25, Adam 1e-3, up to 12 epochs, patience 4 | train the head |
| MobileNetV2 stage 2 | last 50 backbone layers unfrozen, all BN layers frozen, Adam 1e-4, up to 15 epochs, patience 4 | adapt features gently |
| Ladder rungs 1–2 | up to 15 epochs (linear, MLP) | cheap baselines |
| Ladder rungs 3–4 | up to 30 epochs (same as final CNN) | fair comparison |
| Evaluation batch | 256 | |
| ECE bins | 15 (reliability plot: 10 bins) | |
| Latency | average over 200 single-image calls, batch size 1 | deployment-style |
| Robustness | 6,000 test images; 6 corruption types, 3 severities each | |
| t-SNE | 4,000 test images, perplexity 30, PCA initialisation | |
| EDA brightness sample | 3,000 training images | |
| TFLite | float16 weights, default optimisation; parity checked on 500 images | |
| Reload parity | 256 images; max probability difference must be below 1e-4 | |

Mode budgets: `smoke` (2,400 training and 1,200 test images, 1–2 epochs, sanity only), `demo` (CNN 12, ladder 10, head 6, fine-tune 8 epochs), `full` (the table above).

## B3. Corruption levels used in robustness testing
Darkening ×0.7, ×0.5, ×0.3. Glare ×1.4, ×1.8, ×2.4. Gaussian noise σ = 10, 25, 50. Blur kernel 3, 5, 7. Rotation 10°, 20°, 35°. Occlusion by a black square covering 10%, 20%, 35% of the area.

## B4. Sign families used in the confusion analysis
Speed limit (classes 0–5, 7, 8; 8 classes). Prohibitory (9, 10, 15, 16, 17; 5). Danger/warning (11, 18–31; 15). Mandatory (33–40; 8). Other/end/priority (6, 12, 13, 14, 32, 41, 42; 7). Total 43. This grouping is our own convenience grouping, not an official taxonomy.

## B5. Files the notebook produces (the export bundle)
`models/` (custom_cnn, mobilenetv2_head_only, mobilenetv2_fine-tuned, ensemble, plus `best_model.keras` and `best_model.tflite`), `results/<model>/` (metrics, classification report, confusion matrix, history, per-image predictions), `figures/` (every plot), `model_comparison.csv`, `progress_ladder.csv`, per-class metrics, top confusions, robustness table, `class_names.json`, `metrics.json`, `sample_test_images.npz` (1,000 test images), and a README with a loading snippet.

---

# Part C — Code walkthrough by notebook cell

Cell numbers match the notebook (0-based, as listed in `03_presenter_pipeline.md`). For each: purpose, the lines that matter, and the gotcha you should know.

**Cell 2 — workflow diagram.** Draws eight boxes with Matplotlib. Purely illustrative.

**Cell 4 — setup.** Reads `MODE`, imports libraries, seeds Python/NumPy/Keras, checks for a GPU (if none and the mode is not `smoke`, it switches to `smoke` and warns), defines `CFGS` (per-mode budgets) and `CFG`, constants, plotting style, working folders, and helpers `savefig` and `short` (shortens long sign names for plots). *Gotcha:* every figure is saved into the artifacts folder automatically via `savefig`.

**Cell 6 — download and verify.** Streams the zip from Zenodo with a progress hook, computes MD5 in 1 MB chunks, asserts it matches, extracts, and locates the dataset root. If the environment variable `GTSRB_DIR` is set it skips the download. *Gotcha:* the MD5 check runs on every execution; a corrupt download fails fast with a clear message.

**Cell 7 — label tables.** Builds `train_df` (path, label, group, frame, width, height) by walking class folders; filenames follow `class_track_frame.png`, so `group = class_track`. Joins width/height from `Train.csv`. Builds `test_df` from `Test.csv`. Asserts 39,209 / 12,630 / 43 and that all files exist. *Gotcha:* the group id is what makes the grouped split possible.

**Cells 9–17 — EDA.**
- 9: horizontal bars of images per class for train and test, sorted; prints min, max and imbalance ratio.
- 11: one random example per class in a 5×9 grid.
- 12: histograms of width, 2-D width×height histogram, aspect-ratio scatter, all from the CSV metadata (no image opening needed).
- 14: opens a random sample of images, computes mean grey level and spread, plots the brightness histogram and shows darkest and brightest examples.
- 16: shows ten frames of one physical "Stop" sign track to expose the near-duplicate structure and prints track statistics.

**Cell 19 — the split.** Compares a stratified random split (counts tracks present in both train and validation) against the per-class track holdout. Asserts no overlap in the grouped split and that all 43 classes appear in validation, prints a comparison table and the sizes, and checks all 43 classes appear in validation. In `smoke` mode it then subsamples.

**Cell 22 — caching.** `load_resized` opens an image, converts to RGB, resizes to 64×64 bilinear; `load_array` does this with 16 threads. Produces `X_fit`, `X_val`, `X_test` (uint8) and integer label arrays. Prints time and memory (about 0.5 GB).

**Cell 23 — augmentation.** `augmentation()` returns the Keras augmentation block. The cell previews one image and fifteen augmented versions, then demonstrates why flipping is wrong using class 34 (turn left) and 33 (turn right). *Gotcha:* if a class has no image in the (smoke) subset the demo falls back to a generic example.

**Cell 25 — models.**
- `build_custom_cnn()`: Input (raw 0–255) → augmentation → rescale to 0–1 → three [Conv, BN, MaxPool, Dropout] blocks named `conv1..3` → GAP → Dense(128) named `embedding` → Dropout → Dense(43) named `probs`. Augmentation is placed before rescaling so contrast jitter works on 0–255 values.
- `build_mobilenet()`: Input → augmentation → `Resizing(128)` → rescale to [-1, 1] (what MobileNetV2 expects) → pretrained `backbone` (average-pooled) frozen → Dropout → Dense(43). If ImageNet weights cannot be downloaded it falls back to random initialisation and prints a warning.
- Prints both model summaries.

**Cell 27 — `fit` and `plot_history`.** `fit` compiles with Adam and sparse cross-entropy, attaches EarlyStopping (monitor validation accuracy, restore best weights) and ReduceLROnPlateau, trains on the in-memory arrays, prints epochs and time. `plot_history` draws accuracy and loss curves, with an optional vertical line where fine-tuning begins. Then it trains the custom CNN.

**Cell 28 — MobileNetV2 stage 1.** Trains the head; saves a weights snapshot (used later for the head-only ablation and the head-only export); plots.

**Cell 29 — MobileNetV2 stage 2.** Sets the backbone trainable, re-freezes everything except the last 50 layers, re-freezes all BatchNorm layers, prints trainable vs total parameters, trains with learning rate 1e-4, merges the two histories and plots them with a boundary line. *Gotcha:* the model must be re-compiled after changing which layers are trainable; `fit` does this each time.

**Cell 31 — evaluation machinery.** `ece_score`, `score_from_probs` (accuracy, macro-F1, top-3, ECE from probabilities), `score` (predict then score), `latency_ms`. Evaluates three models: custom CNN, MobileNetV2 head-only (restored from the stage-1 snapshot, then the fine-tuned weights are put back), and MobileNetV2 fine-tuned, on both validation and test. Builds the comparison table with highlighting and the two-panel figure (accuracy/F1 bars, accuracy-vs-latency bubble chart). Saves `model_comparison.csv`.

**Cell 32 — model selection.** Chooses between the custom CNN and fine-tuned MobileNetV2 using **validation accuracy**, stores `BEST_NAME`, `BEST`, `RB` (its test results) and prints the test numbers. The ensemble is not a candidate because Grad-CAM, the embedding and the demo function expect a single model.

**Cell 33 — per-class.** `classification_report` for the selected model, a per-class table (precision, recall, F1, support, training images), a sorted F1 bar chart (red under 0.90), and a scatter of F1 against class size with the Spearman rank correlation printed. Shows the eight hardest classes.

**Cell 34 — confusion.** Confusion matrix (saved as `.npy`), row-normalised heat-map, top-ten off-diagonal pairs with names, bar chart of those pairs.

**Cell 35 — families.** Computes the share of errors where the true and predicted classes belong to the same sign family and compares it to the chance level implied by family sizes.

**Cell 36 — error gallery.** Twelve most confident wrong predictions, with true class, predicted class and confidence.

**Cell 38 — ladder training.** `ladder_model(kind)` builds the four early rungs: linear on 16×16 pixels, MLP on 32×32, basic two-layer CNN, deeper CNN with BN and global pooling (all without augmentation or dropout). Each is trained with `fit` and scored on validation.

**Cell 39 — ensemble and ladder chart.** Averages the CNN and fine-tuned MobileNetV2 probabilities for validation and test (adding `R["Ensemble"]`, `V["Ensemble"]`), builds the majority-class baseline from the training label counts, assembles `ladder_df` (steps 0–8, tracks, validation accuracy and macro-F1, parameters, best-so-far, gain), displays it, and draws the two-panel ladder chart (accuracy with best-so-far line; error rate on a log axis). Prints the error reduction factor and the ensemble's test result.

**Cell 41 — Grad-CAM.** `_cam_parts` builds, once per model, a helper model returning the last conv feature map and the output; the CNN uses `conv3`; MobileNetV2 uses `out_relu` inside the backbone (it is a nested model, so the forward pass is done layer by layer around it). `gradcam` computes the class score gradient with `tf.GradientTape`, averages over space for weights, forms the positive weighted sum, normalises to 0–1 and resizes. `overlay` draws it. The cell shows 5 correct and 5 confident-error examples for both models.

**Cell 42 — t-SNE.** Builds an `embedder` model ending at the input of the last dense layer, embeds up to 4,000 test images, runs t-SNE, plots coloured by family and by class.

**Cell 44 — calibration and abstention.** Confidence histograms (correct vs wrong, log scale), reliability diagram, coverage-vs-accuracy curve, and printed thresholds for 99%, 99.5%, 99.9% accuracy.

**Cell 45 — robustness.** Six corruption functions (brightness scaling, Gaussian noise, Gaussian blur, rotation with border replication, black-square occlusion), applied at three severities to the first 6,000 test images, plotted, with example images and a table.

**Cell 47 — `predict_sign`.** Accepts a file path or array, resizes to 64×64, predicts, and shows the input, Grad-CAM overlay and the top-5 bar chart; returns the top-5 list. Runs on four random test images.

**Cell 48 — optional upload.** Tick `RUN_UPLOAD` to upload your own image in Colab.

**Cell 50 — export.** Saves all models (including rebuilding the head-only model from its snapshot, and an ensemble model that averages the two), writes per-model results folders, tables, a 1,000-image sample archive and `metrics.json`, then re-loads the custom CNN and fine-tuned MobileNetV2 and asserts the predictions match.

**Cell 51 — TFLite.** Converts the selected model to TFLite with float16 weights, runs the interpreter on 500 test images and prints size, agreement with Keras, accuracy and per-image time. Wrapped in try/except so a conversion problem does not stop the notebook.

**Cell 52 — bundle.** Writes the README, zips everything, prints the contents, and downloads the zip in Colab when `DOWNLOAD` is ticked (default on).

**Cell 54–55 — wrap-up.** Prints the headline numbers of this run and the table, then the decision log, limitations, future work and references.

---

# Part D — Design decisions, alternatives and honest weaknesses

| # | Decision | Alternative | Why we chose it | Honest weakness / what a sharp examiner may say |
|---|---|---|---|---|
| 1 | Verify MD5 | Trust the download | Corrupt data would silently ruin results | MD5 detects corruption, not a malicious mirror |
| 2 | Official split, test used once | Own random split | Comparable with literature | Test used for the ladder's final ensemble line and for abstention thresholds, which is a mild reuse (see D-notes) |
| 3 | Grouped validation split | Random | Avoids leakage (A20) | Our first run used a plain grouped split that left classes 0, 19, 21, 23 and 29 out of validation; we fixed it with an explicit per-class track holdout and added an assertion (a first fix with `StratifiedGroupKFold` also failed on Colab because it has no coverage guarantee) |
| 4 | Macro-F1 plus accuracy | Accuracy only | Imbalance ≈ 10× | Macro-F1 treats a rare class and a common class equally, which may not match real-world risk |
| 5 | 64×64 cached arrays | Full-resolution tf.data streaming | Small images; speed | Information is lost for the larger images; digits on speed limits may need more pixels |
| 6 | No ROI crop | Crop to annotated box | Detectors output margins | Includes background pixels |
| 7 | Normalisation and augmentation inside the model | External preprocessing | No train/serve mismatch | Slightly larger model graph |
| 8 | No horizontal flip | Flip | Left/right meaning | Some symmetric classes could safely use flips; we chose simplicity |
| 9 | Custom CNN plus MobileNetV2 | One model | Shows trade-off | Only two architectures; no larger backbones |
| 10 | Two-stage transfer, BN frozen | Frozen only; full unfreeze | Standard stable recipe | 50 unfrozen layers is a chosen value, not tuned |
| 11 | Model chosen by validation accuracy | By test accuracy | Avoids optimism | If two models are within noise the choice is arbitrary (single seed) |
| 12 | Adam at 1e-3 and 1e-4 | SGD with momentum, schedules | Robust default | Learning rates not searched |
| 13 | Early stopping on validation accuracy | Fixed epochs | Avoids over-training | Validation accuracy can be noisy |
| 14 | No class weights | Class weights, focal loss | Check first whether imbalance hurts (Spearman) | Rare classes may be under-served |
| 15 | Grad-CAM, t-SNE | Metrics only | Verify what the model uses | Both are qualitative |
| 16 | Abstention by confidence | Always answer | Safer behaviour | Needs calibration; threshold shown is from test data |
| 17 | Synthetic robustness | Real hard images | Controlled, repeatable | Not the same as real night, rain, motion blur |
| 18 | Ensemble by averaging | Weighted/stacked | Simple, no training | Costs two models at inference |
| 19 | Single seed | Multiple seeds | Time | Small differences are not statistically meaningful |
| 20 | TFLite float16 | int8 quantisation | Easy, small accuracy loss | Not benchmarked on real embedded hardware |

**D-notes (things to be upfront about before you are asked).**
1. The ladder compares steps on validation accuracy; steps that differ by a fraction of a percent are within run-to-run noise with one seed.
2. The abstention thresholds printed in section 11.1 are read off test-set behaviour. That demonstrates the trade-off but a production threshold must be picked on validation data.
3. MobileNetV2 may score below the custom CNN. That is a result about *this task* (tiny low-resolution images), not a general statement about MobileNetV2.
4. The notebook does not claim safety certification.

---

# Part E — How to read each output (and what "good" and "bad" look like)

| Output | How to read it | Good | Red flag |
|---|---|---|---|
| Class distribution bars | Height = images per class | You can name the biggest and smallest class | Not mentioning imbalance |
| Image-size histograms | Most images are small | Link to the 64×64 decision | Saying "images are high-resolution" |
| Brightness histogram and extreme examples | Lighting varies | Link to contrast augmentation | — |
| Track strip | Same sign, many frames | Explain leakage | Calling frames independent |
| Split table | Tracks in both sets | Random split shows most validation tracks leaked; grouped shows 0 | — |
| Augmentation preview | Same sign, varied | Signs remain recognisable | Showing flips as valid |
| Model summaries | Layers and parameter counts | Know the CNN is much smaller | — |
| Training curves | Blue = train, orange = validation | Curves rise then flatten; validation near train | Validation falling while train rises (overfit); both flat and low (underfit) |
| Why train < validation early | Augmentation and dropout are on during training only | Explain it | Calling it a bug |
| Comparison table | Accuracy, macro-F1, top-3, ECE, parameters, latency | Quote validation vs test; model chosen by validation | Quoting test-based selection |
| Per-class F1 | Red = below 0.90 | Name two weak classes and a reason | — |
| F1 vs class size scatter | Is scarcity the cause? | Read the Spearman value aloud and interpret | Claiming imbalance is the cause without evidence |
| Confusion matrix | Diagonal = correct; off-diagonal = mistakes | Identify the bright off-diagonal pairs | — |
| Confusion pairs table | Most frequent mistakes | Explain visual similarity (digits) | — |
| Same-family share | Errors inside families | Contrast with chance level | — |
| Error gallery | Confident errors | Point to blur, darkness, small size | — |
| Ladder chart | Progress per step | Explain tracks and why transfer may start lower | Hiding a flat step |
| Grad-CAM | Hot = influential | Heat on symbol/rim | Over-claiming correctness from a single picture |
| t-SNE | Clusters of embeddings | Touching clusters = confusable | Reading distances literally |
| Confidence histogram | Wrong answers usually lower confidence | Use it to justify abstention | — |
| Reliability diagram | Bars below diagonal = over-confidence | Mention temperature scaling | — |
| Coverage curve | Accuracy up as coverage down | Quote "answers X% of images to reach 99%" | Not noting thresholds come from test data |
| Robustness lines | Accuracy vs severity | Name the worst corruption | Claiming real-world safety |
| Export checks | Parity within 1e-4; TFLite agreement | State sizes and agreement | — |

---

# Part F — Reference material

## F1. Formula crib
- Softmax: `p_i = e^{z_i} / Σ_j e^{z_j}`
- Cross-entropy for one image: `L = -log p_{true class}`
- Precision `P = TP/(TP+FP)`; Recall `R = TP/(TP+FN)`; F1 `= 2PR/(P+R)`
- Macro-F1 `= (1/43) Σ_c F1_c`
- Accuracy `= correct / total`
- Conv parameters `= k·k·C_in·C_out + C_out`; dense parameters `= n_in·n_out + n_out`
- ECE `= Σ_bins (n_bin/N) · |accuracy_bin − confidence_bin|`
- Grad-CAM weight for map k: `α_k = mean over positions of ∂score/∂A_k`; map `= ReLU(Σ_k α_k A_k)`
- Reduction factor of error from `e1` to `e2`: `e1/e2`

## F2. Glossary (A to Z)
- **Activation function:** non-linear function after a layer (ReLU, softmax).
- **Adam:** adaptive-learning-rate optimiser.
- **Augmentation:** training-time image alterations.
- **Backbone:** the pretrained feature-extracting part of a transfer model.
- **Batch / batch size:** group of images processed per weight update.
- **BatchNorm:** layer that stabilises activations.
- **Bias:** learned constant added in a layer.
- **Calibration:** whether confidence matches accuracy.
- **CNN:** convolutional neural network.
- **Confusion matrix:** table of true vs predicted classes.
- **Coverage:** share of inputs the system answers.
- **Cross-entropy:** loss for classification.
- **Dense layer:** fully connected layer.
- **Domain gap:** difference between pretraining data and target data.
- **Dropout:** random unit silencing in training.
- **ECE:** expected calibration error.
- **Embedding:** internal feature vector of an input.
- **Ensemble:** combining models.
- **Epoch:** one pass over the training set.
- **F1:** harmonic mean of precision and recall.
- **Feature map:** output of a conv filter.
- **Fine-tuning:** training some pretrained layers at a low learning rate.
- **Frozen:** not updated during training.
- **GAP:** global average pooling.
- **Grad-CAM:** gradient-based heat-map explanation.
- **GroupShuffleSplit:** scikit-learn splitter that keeps groups together but does not balance classes; we use an explicit per-class track holdout instead.
- **GTSRB:** German Traffic Sign Recognition Benchmark.
- **ImageNet:** large image dataset used for pretraining.
- **Inference:** using a trained model to predict.
- **Keras / TensorFlow:** the deep-learning library used.
- **Leakage:** evaluation information leaking into training.
- **Learning rate:** step size of weight updates.
- **Logit:** raw score before softmax.
- **Loss:** number training tries to minimise.
- **Macro / weighted average:** averaging over classes equally / by size.
- **MD5:** file fingerprint.
- **Overfitting / underfitting:** see A7.
- **Parameter:** learned weight.
- **Parity check:** confirm two models give the same predictions.
- **Pooling:** downsampling feature maps.
- **Precision / recall:** see A21.
- **ReLU:** `max(0, x)`.
- **ROI:** region of interest, the annotated sign box.
- **Selective prediction:** abstaining when unsure.
- **Softmax:** scores to probabilities.
- **Support:** number of true examples of a class.
- **t-SNE:** 2-D visualisation of high-dimensional data.
- **TFLite:** compact model format for devices.
- **Top-k accuracy:** correct if truth is in the top k predictions.
- **Track:** the sequence of frames of one physical sign.
- **Transfer learning:** reusing a pretrained model.
- **uint8:** whole numbers 0–255.
- **Validation set:** held-out data for decisions.

## F3. The five papers in one paragraph each
1. **Stallkamp et al., IJCNN 2011 (GTSRB).** Introduced the benchmark: real, varied photographs, 43 classes, a fixed official test set, and a competition comparing methods including humans. Justifies our scope, EDA and official-test protocol.
2. **Sandler et al., CVPR 2018 (MobileNetV2).** Efficient architecture using inverted residuals and linear bottlenecks for mobile devices. Justifies our lightweight transfer-learning model.
3. **Selvaraju et al., ICCV 2017 (Grad-CAM).** Gradient-weighted class activation maps for visual explanations of CNN decisions. Our explainability method.
4. **Kingma and Ba, ICLR 2015 (Adam).** Adaptive moment estimation optimiser. Our optimiser.
5. **Guo et al., ICML 2017 (calibration).** Showed modern networks are often over-confident, introduced ECE usage and temperature scaling. Basis of our calibration analysis and future-work item.

## F4. Where the rest of the project lives (repository map)
- `src/traffic_sign_recognition/` — reusable package: data loading, models, training, evaluation, Grad-CAM, inference.
- `scripts/` — dataset download, EDA, result building, error analysis, experiment grid, demo server, detection extension, document builders, and `build_demo_notebook.py` (generates the notebook).
- `notebooks/traffic_sign_recognition_demo.ipynb` — the panel demo (this knowledge base explains it). The older `traffic_sign_recognition_colab.ipynb` is the earlier lighter notebook.
- `docs/` — synopsis, report, literature survey, presenter guide, IEEE presentation, and these presenter documents.
- Optional extensions in the repository: YOLO-based detection scripts (need a separately supplied detector checkpoint) and a local webcam demo server. They are outside the notebook and outside GTSRB's classification scope.
- Note: the repository's source package uses slightly different settings from the notebook (for example 96-pixel MobileNetV2 input and stronger augmentation). If asked, say the notebook is a self-contained, standalone variant and its settings are the ones documented here.
