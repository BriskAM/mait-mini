const L = require("../lib");
const { chapter, h2, h3, h3u, p, bullets, numbered, fig, tbl, eq, code, fs, path, d } = L;
const { AlignmentType } = d;
const RIGHT = AlignmentType.RIGHT, CENTER = AlignmentType.CENTER, LEFT = AlignmentType.LEFT;
const csv = (f) => L.readCsv(f);
const pct = (v, dgt = 2) => (parseFloat(v) * 100).toFixed(dgt) + " %";
const hist = (n) => JSON.parse(fs.readFileSync(path.resolve(__dirname, "..", "data", `history_${n}.json`), "utf8"));

module.exports = function ch7_9() {
  const o = [];
  const metrics = JSON.parse(fs.readFileSync(path.resolve(__dirname, "..", "data", "metrics.json"), "utf8"));
  const ladder = csv("progress_ladder.csv"), per = csv("per_class_metrics_selected_model.csv"), rob = csv("robustness_selected_model.csv"), topc = csv("top_confusions_selected_model.csv");

  // =============================================== CHAPTER 7 ===============================================
  o.push(chapter("Implementation"));
  o.push(p("This chapter describes how the design of Chapter 6 was implemented: the data pipeline, the models and their training, the evaluation and explanation code, the export step, and the browser implementation. It follows the structure of the research format (data presentation, implementation, analysis) and then gives the screenshots and the source code of selected modules required of development projects."));
  o.push(h2("7.1 Data Presentation and Preparation"));
  o.push(h3("7.1.1 Loading and Verification"));
  o.push(p("The notebook builds two label tables. For training, it walks the class folders in sorted order and records, for each image, the path, the class, the track (class and track number taken from the file name) and the frame number; width and height are joined from <c>Train.csv</c>. For testing, it reads <c>Test.csv</c>. The code then asserts that there are 39,209 training images, 12,630 test images and 43 classes, that every file exists, and, for the archive, that the checksum matches. A failed assertion stops the run before any training, so that a wrong or truncated dataset cannot silently produce results."));
  o.push(h3("7.1.2 The Validation Split"));
  o.push(p("The split is the most consequential implementation detail of the project, and it required two corrections. The first version used the scikit-learn <c>GroupShuffleSplit</c> {@sklearnGroup}, which keeps all frames of a track together but does not balance classes. In the first full run the printed check <i>all 43 classes in validation</i> was <i>False</i>: five classes (0, 19, 21, 23 and 29) had no validation images, which biased validation accuracy, the ladder and the model selection (the test results were unaffected, because those classes were still trained on and the test set is separate). A first repair with <c>StratifiedGroupKFold</c> passed in a local check but failed the same assertion on Colab, because that splitter only attempts to balance classes and its behaviour depends on the library version. The final implementation therefore selects tracks explicitly (Algorithm 6.1): for every class, about one seventh of its tracks (at least one) are chosen with a seeded NumPy generator. It is deterministic, independent of library versions, and guarantees coverage; Listing 7.1 shows the code."));
  o.push(...tbl("leak", "Random split versus the track-grouped split used", ["Property", "Random image split", "Track-grouped split (used)"],
    [["Validation images", "5,602", "5,639"], ["Fitting images / tracks", "-", "33,570 / 1,119"], ["Validation tracks", "-", "188"], ["Tracks present on both sides", "1,291 of 1,307", "0"], ["Validation tracks already seen in training", "100 %", "0 %"], ["Classes present in validation", "-", "43 of 43 (asserted)"]],
    [3.2, 2.2, 2.4], { align: [LEFT, CENTER, CENTER] }));
  o.push(h3("7.1.3 Caching and Augmentation"));
  o.push(p("All images are resized to 64 x 64 pixels with bilinear resampling and stored as 8-bit arrays: 33,570 fitting, 5,639 validation and 12,630 test images, 0.64 GB in total, built in 25 seconds with 16 reader threads. The region of interest given in the CSV is deliberately <i>not</i> used to crop: a real detector returns boxes with some margin, so training on the full frame matches deployment better."));
  o.push(p("Augmentation is implemented as Keras layers placed at the start of each model, so it is active only while training. {tab:aug} lists the settings, and {fig:aug} shows one image and fifteen augmented versions. Horizontal flipping is deliberately absent: flipping a “turn left ahead” sign yields an image that looks like “turn right ahead” but would keep the original label."));
  o.push(...tbl("aug", "Augmentation settings (training only)", ["Transform", "Setting", "Reason"],
    [["Rotation", "factor 0.04 of a full turn (about 14 degrees either way)", "Camera tilt, slanted poles"], ["Translation", "5 % of width and height", "Imperfect crops"], ["Zoom", "8 %", "Different distances"], ["Contrast", "10 %", "Lighting variation"], ["Horizontal flip", "Not used", "Would change the meaning of directional signs"]],
    [1.5, 3.2, 3.0], { align: [LEFT, LEFT, LEFT] }));
  o.push(...fig("aug", "nb/augmentation_preview.png", "Original image and random augmentations (training only)", { width: 5.6 }));
  o.push(h3("7.1.4 Hyper-parameters"));
  o.push(...tbl("hyper", "Hyper-parameters and training settings of the reported run", ["Setting", "Custom CNN", "MobileNetV2 stage 1", "MobileNetV2 stage 2"],
    [["Optimiser", "Adam", "Adam", "Adam"], ["Learning rate", "1e-3", "1e-3", "1e-4"], ["Batch size", "128", "128", "128"], ["Epoch limit", "30", "12", "15"], ["Early stopping (patience, on validation accuracy)", "5", "4", "4"], ["Learning-rate reduction", "x0.4 after 2 stalled epochs", "x0.4 after 2", "x0.4 after 2"],
     ["Trainable parameters", "115,755", "55,083", "1,834,219"], ["Input to the network", "64 x 64", "64 x 64, upsampled to 128 x 128", "same"], ["Random seed", "42", "42", "42"]],
    [3.0, 1.6, 1.8, 1.6], { size: 19, align: [LEFT, CENTER, CENTER, CENTER] }));
  o.push(h2("7.2 Implementation of the Work"));
  o.push(h3("7.2.1 Models and Training"));
  o.push(p("Both models are built with the Keras functional API (Listing 7.2). A helper function compiles a model with Adam and sparse categorical cross-entropy, attaches early stopping (monitoring validation accuracy and restoring the best weights) and learning-rate reduction on plateau, trains on the in-memory arrays and returns the history and the elapsed time. For MobileNetV2 the second stage sets the backbone trainable, re-freezes all but its last 50 layers and every batch-normalisation layer, and recompiles with the lower learning rate. If the pretrained weights cannot be downloaded the code falls back to random initialisation with a visible warning, so that the notebook still runs offline."));
  o.push(h3("7.2.2 The Progress Ladder"));
  o.push(p("The ladder trains four additional models of increasing capability on the same fitting and validation sets and with the same optimiser, and reuses the final custom CNN and the two MobileNetV2 models. {tab:rungs} defines each step."));
  o.push(...tbl("rungs", "Definition of the progress-ladder steps", ["Step", "Model", "What changes"],
    [["0", "Majority-class baseline", "Always predict the most frequent training class"], ["1", "Linear classifier", "One dense softmax layer on 16 x 16 pixels"], ["2", "Multi-layer perceptron", "Two hidden layers (256, 128 units) on 32 x 32 pixels"], ["3", "Basic CNN", "Two convolution layers, flatten, dense layer; no normalisation, augmentation or dropout"],
     ["4", "Deeper CNN", "Three blocks, batch normalisation, global average pooling"], ["5", "Final custom CNN", "Adds augmentation and dropout"], ["6", "MobileNetV2, frozen", "Pretrained backbone, new head only"], ["7", "MobileNetV2, fine-tuned", "Top 50 backbone layers adapted at a lower rate"], ["8", "Ensemble", "Average of the probabilities of steps 5 and 7"]],
    [0.7, 2.4, 4.6], { size: 19, align: [CENTER, LEFT, LEFT] }));
  o.push(h3("7.2.3 Evaluation, Explanation and Reliability Code"));
  o.push(p("A scoring function converts a probability matrix into accuracy, macro-F1 (scikit-learn), top-3 accuracy and ECE with 15 bins, so the same code scores the validation and the test set and the ensemble. Latency is the mean time of 200 single-image calls. The per-class report, confusion matrix, most frequent confusions and confident errors are computed from the test predictions. For Grad-CAM the code builds a helper model that returns the last convolutional feature map and the network output; because the MobileNetV2 model contains its backbone as a nested model, the forward pass is carried out layer by layer around it. Embeddings for t-SNE are taken from the input of the last dense layer. The corruption functions apply brightness scaling, Gaussian noise, Gaussian blur, rotation with border replication and a black occluding square to 6,000 test images at three severities each."));
  o.push(h3("7.2.4 Export and Parity Checks"));
  o.push(p("The export step saves each model in the native Keras format together with its metrics, classification report, confusion matrix, training history and per-image predictions; the comparison, ladder, per-class, confusion and robustness tables; a 1,000-image sample of the test set; and a README that shows how to load a model. The two main models are then reloaded and their predictions on 256 images are compared with those of the in-memory models; a difference above 10<sup>-4</sup> stops the run. The selected model is also converted to TensorFlow Lite with 16-bit float weights and evaluated on 500 test images."));
  o.push(h3("7.2.5 The Browser Implementation"));
  o.push(p("To let anyone run the selected model without installation or a server, the trained network was re-implemented in plain JavaScript (<c>nn.js</c>, 150 lines, no libraries). A build script (<c>build_pages_data.py</c>) reads the weights directly from the saved Keras file and writes them as a single 465 KB float32 file with a JSON manifest that records each array's name and shape and the batch-normalisation epsilon. In the browser the weights are folded into per-channel scale and shift constants for the normalisation layers. The forward pass is a direct implementation of the architecture: a 3 x 3 convolution with ReLU, batch normalisation, 2 x 2 max pooling (recording the winning positions), three times, then global average pooling and two dense layers (Listing 7.3)."));
  o.push(p("Grad-CAM is implemented by hand-written back-propagation from the chosen class score to the output of the last convolutional layer, through the dense layers, the ReLU gate, global average pooling, the recorded max-pooling positions and the batch-normalisation scale (Listing 7.4). The channel weights are the spatial means of these gradients; the heat-map is the ReLU of the weighted sum of the activations, scaled to the range 0 to 1 and upsampled bilinearly. The same code supports two scores: the class logit (default) and the class probability (as in the notebook), which allowed the numerical comparison in Section 9.2.4."));
  o.push(p("The page itself is written as ES modules: <c>imaging.js</c> (crop, corruptions, overlay), <c>charts.js</c> (an SVG and canvas chart kit with tooltips on hover and keyboard focus and a table view for every chart), <c>demo.js</c> (the demonstration and stress lab) and <c>dashboard.js</c> (the result charts). A second build step turns the notebook's results bundle into <c>results.json</c> and a sprite of 116 sample test images (86 correctly classified and 30 misclassified by the Keras model, with the Keras confidence recorded for each). A GitHub Actions workflow publishes the <c>site</c> folder to GitHub Pages on every change."));
  o.push(h2("7.3 Analysis and Interpretation of Training"));
  const hc = hist("cnn"), hh = hist("head"), hf = hist("ft");
  o.push(p("{tab:train} records the training cost of each stage and {fig:curveCnn} and {fig:curveMn} show the learning curves."));
  o.push(...tbl("train", "Training cost of the reported run", ["Stage", "Epochs run (limit)", "Time", "Best validation accuracy"],
    [["Custom CNN", "29 (30)", "341 s", "97.61 %"], ["MobileNetV2 stage 1 (head)", "12 (12)", "244 s", "72.58 %"], ["MobileNetV2 stage 2 (fine-tune)", "14 (15)", "422 s", "90.55 %"], ["Ladder: linear classifier", "15 (15)", "17 s", "70.19 %"], ["Ladder: MLP", "15 (15)", "20 s", "83.24 %"], ["Ladder: basic CNN", "25 (30)", "107 s", "93.37 %"], ["Ladder: deeper CNN", "24 (30)", "150 s", "95.14 %"]],
    [3.2, 1.6, 1.2, 1.9], { align: [LEFT, CENTER, CENTER, CENTER] }));
  o.push(...fig("curveCnn", "nb/curves_custom_cnn.png", "Training and validation curves of the custom CNN", { width: 5.6 }));
  o.push(p(`The custom CNN's validation accuracy tracks its training accuracy closely: at the last epoch the training accuracy is ${pct(hc.accuracy[hc.accuracy.length - 1])} and the validation accuracy ${pct(hc.val_accuracy[hc.val_accuracy.length - 1])}, a gap of under two points. Training accuracy is lower than validation accuracy in the first epochs because augmentation and dropout are active only during training. Early stopping ended training after 29 of at most 30 epochs.`));
  o.push(...fig("curveMn", "nb/curves_mobilenetv2_two-stage.png", "Training and validation curves of MobileNetV2 across the two stages (the dashed line marks the start of fine-tuning)", { width: 5.6 }));
  o.push(p(`For MobileNetV2 the head-only stage plateaus: its validation accuracy stays near 72 % although the training accuracy reaches ${pct(hh.accuracy[hh.accuracy.length - 1], 1)}. Fine-tuning raises validation accuracy sharply to ${pct(hf.val_accuracy[hf.val_accuracy.length - 1], 1)}, but the training accuracy reaches ${pct(hf.accuracy[hf.accuracy.length - 1], 1)}: the fine-tuned model over-fits, memorising the training images far more than the small CNN does. This agrees with the expectation that the pretrained features are not matched to tiny sign crops and that a network with 2.3 million parameters has more capacity than 33,570 highly correlated images can constrain.`));
  o.push(h2("7.4 Screenshots"));
  o.push(p("The website is published at https://briskam.github.io/mait-mini/. {fig:s1} to {fig:s5} show the main views. In {fig:s1} the hero panel cycles real test images through the model; each frame shows the image, the Grad-CAM heat-map, the prediction, the confidence and whether it is correct. {fig:s2} shows a misclassified test image: the true class is Speed limit 120 km/h, the model answers Speed limit 100 km/h with 91.6 % confidence and the page states that the Keras model from the notebook makes the same mistake."));
  o.push(...fig("s1", "site/s_hero.jpg", "Website home view with the live prediction panel and headline results", { width: 5.4 }));
  o.push(...fig("s2", "site/s_demo.jpg", "Live demonstration of a misclassified test image with Grad-CAM", { width: 5.4 }));
  o.push(p("{fig:s3} shows the effect of added noise on another test image: the prediction changes to the wrong class with 80 % confidence, which is below the answer threshold of 0.87, so the page reports that the system would defer to a human. {fig:s4} shows the stress lab with a sweep of the noise level, which plots the model's probability for its clean prediction (and for the true class when they differ) across the range."));
  o.push(...fig("s3", "site/s_noise.jpg", "Noise applied to a test image; the model would defer", { width: 5.4 }));
  o.push(...fig("s4", "site/s_stress.jpg", "Stress lab with presets, sliders and a noise sweep", { width: 5.4 }));
  o.push(...fig("s5", "site/s_ladder.jpg", "Results dashboard: the progress ladder and training curves", { width: 5.4 }));
  o.push(h2("7.5 Source Code of Some Modules"));
  o.push(p("The listings below are the most important parts of the implementation. Further code is in the repository and in Appendix A."));
  o.push(...code("splitcode", "Per-class, track-grouped validation split (notebook)", [
    "rng = np.random.default_rng(SEED); val_groups = []",
    "for _, tracks in train_df.groupby(\"label\").group.unique().items():",
    "    # hold out ~1/7 of each class's tracks (at least one)",
    "    val_groups += list(rng.choice(tracks, max(1, round(len(tracks) / 7)), replace=False))",
    "is_val = train_df.group.isin(val_groups).values",
    "fit_df, val_df = train_df[~is_val].reset_index(drop=True), train_df[is_val].reset_index(drop=True)",
    "assert not set(fit_df.group) & set(val_df.group), \"Leakage: a track appears in both splits\"",
    "assert val_df.label.nunique() == NUM_CLASSES, \"Validation is missing some classes\""]));
  o.push(...code("cnncode", "Custom CNN definition (Keras functional API)", [
    "def build_custom_cnn():",
    "    inp = layers.Input((IMG, IMG, 3), name=\"image\")           # raw pixels 0-255",
    "    x = augmentation()(inp)                                   # active in training only",
    "    x = layers.Rescaling(1 / 255.0)(x)",
    "    for i, f in enumerate((32, 64, 128), 1):",
    "        x = layers.Conv2D(f, 3, padding=\"same\", activation=\"relu\", name=f\"conv{i}\")(x)",
    "        x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.2)(x)",
    "    x = layers.GlobalAveragePooling2D()(x)",
    "    x = layers.Dense(128, activation=\"relu\", name=\"embedding\")(x); x = layers.Dropout(0.3)(x)",
    "    return keras.Model(inp, layers.Dense(NUM_CLASSES, activation=\"softmax\", name=\"probs\")(x), name=\"custom_cnn\")"]));
  o.push(...code("jsfwd", "Forward pass of the network in JavaScript (nn.js, abridged)", [
    "forward(rgb, keep = false) {",
    "  let x = Float32Array.from(rgb, (v) => v / 255), H = 64, Cin = 3; const stages = [];",
    "  for (const [i, Cout] of [[1, 32], [2, 64], [3, 128]]) {",
    "    const a = conv3x3(x, H, H, Cin, p[`conv${i}.w`], p[`conv${i}.b`], Cout);   // conv + ReLU",
    "    const bn = batchNorm(a, Cout, this.bn[i].scale, this.bn[i].shift);",
    "    const pooled = maxPool(bn, H, H, Cout);                  // keeps arg-max positions",
    "    stages.push({ a, bn, pooled, H, C: Cout }); x = pooled.out; H >>= 1; Cin = Cout;",
    "  }",
    "  const g = globalAveragePool(x, H * H, Cin);                  // 128 numbers",
    "  const h1 = relu(dense(g, p['dense1.w'], p['dense1.b']));     // 128 units",
    "  const probs = softmax(dense(h1, p['dense2.w'], p['dense2.b']));   // 43 classes",
    "  return keep ? { probs, stages, h1pre, n } : { probs };",
    "}"]));
  o.push(...code("jscam", "Grad-CAM back-propagation in JavaScript (nn.js, abridged)", [
    "const dz = new Float32Array(43);                  // gradient of the score w.r.t. the logits",
    "for (let j = 0; j < 43; j++) dz[j] = score === \"prob\" ? probs[c] * ((j === c) - probs[j]) : (j === c ? 1 : 0);",
    "dh1pre[i] = h1pre[i] > 0 ? sum_j W2[i][j] * dz[j] : 0;      // dense 2 and ReLU gate",
    "dg[i]     = sum_o W1[i][o] * dh1pre[o];                    // dense 1",
    "for (o of pooledPositions) dB[arg[o]] += dg[o % C] / n;     // pooling and global average",
    "alpha[c] = (sum over positions of dB[.., c]) * bnScale[c] / (H * H);",
    "cam = relu( sum_c alpha[c] * A[.., c] );  cam /= max(cam) + 1e-8;"]));

  // =============================================== CHAPTER 8 ===============================================
  o.push(chapter("Testing"));
  o.push(p("The project has no separate unit-test suite for the Python package. Testing was instead built into the pipeline as assertions, into the export step as parity checks, and into dedicated verification scripts and manual test cases for the website. This chapter describes each level and lists test cases with their outcomes."));
  o.push(h2("8.1 Test Strategy"));
  o.push(...tbl("strategy", "Levels of testing used", ["Level", "What is tested", "How"],
    [["Data integrity", "Archive, counts, files, split properties", "Assertions that stop the notebook"], ["Component", "Models, scoring, export", "Reload parity, TFLite parity, shape and parameter checks"], ["Numerical", "Browser network versus Keras", "Verification script on 116 images; in-page check"],
     ["System (functional)", "Website behaviour end to end", "Manual test cases in a browser (Section 8.4)"], ["Non-functional", "Robustness, calibration, layout, accessibility", "Corruption study; narrow-viewport and theme checks"], ["Regression", "Behaviour after each change", "Re-running the notebook; re-running the verification script"]],
    [1.6, 2.7, 3.4], { size: 19 }));
  o.push(h2("8.2 Checks Embedded in the Pipeline"));
  o.push(...tbl("assert", "Automated checks in the notebook and their outcome in the reported run", ["Check", "Expected", "Outcome"],
    [["Archive MD5 equals 7faafd0d78d831c7eb4d555641a83a62", "Equal", "Passed"], ["Training images, test images, classes", "39,209; 12,630; 43", "Passed"], ["All image files exist", "True", "Passed"], ["No track in both fitting and validation sets", "Empty intersection", "Passed"],
     ["All 43 classes present in validation", "True", "Passed (failed in the first run, which led to the corrected split)"], ["Reloaded custom CNN reproduces predictions on 256 images", "Difference below 1e-4", "Passed (difference 0)"], ["Reloaded fine-tuned MobileNetV2", "Difference below 1e-4", "Passed (difference 0)"],
     ["Reloaded ensemble model", "Difference below 1e-4", "Passed (difference 0)"], ["TFLite model versus Keras on 500 test images", "Same predictions", "100 % agreement; 97.6 % accuracy on those images"]],
    [3.4, 1.9, 2.4], { size: 19 }));
  o.push(h2("8.3 Numerical Verification of the Browser Network"));
  o.push(p("The correctness of the JavaScript network is critical, because the website presents it as the trained model. It was verified in three ways. First, a Keras model with the identical architecture was built outside the notebook, the weights were loaded from the saved file, and its predictions on the 116 sample images were compared with the confidences that the notebook had recorded for the original model: the largest difference was 1.3 x 10<sup>-5</sup>, with identical predictions. This validates both the weight mapping and the batch-normalisation constants. Second, the JavaScript network was run in Node.js on the same images and compared with this Keras reference ({tab:parity}). Third, the web page contains a <i>Verify against Keras</i> button that repeats the comparison with the values recorded in the notebook, so the check can be repeated by any visitor."));
  o.push(...tbl("parity", "Agreement between the JavaScript network and Keras on 116 test images", ["Quantity", "Result"],
    [["Predictions identical", "116 of 116"], ["Largest difference in any class probability", "3.2 x 10^-6"], ["Largest difference in the Grad-CAM map (class-logit score)", "3.1 x 10^-6"], ["Largest difference in the Grad-CAM map (probability score), on the 66 images with confidence below 0.9999", "5.9 x 10^-4"],
     ["Images with confidence of at least 0.9999 (probability-score map not meaningful)", "50 of 116"], ["In-page check against notebook confidences", "116 of 116 identical; largest difference 1.3 x 10^-5"]],
    [5.0, 2.7], { align: [LEFT, LEFT] }));
  o.push(p("The check also revealed a property of the notebook's Grad-CAM implementation that is reported in Section 9.2.4: for the 50 near-certain images the probability-based maps differ between two correct implementations because their gradients are of the order of floating-point rounding."));
  o.push(h2("8.4 Functional Test Cases for the Website"));
  o.push(p("{tab:tc} lists the test cases executed on the deployed site (and on the local copy during development), using the browser's console to confirm that no errors were logged."));
  const tcs = [
    ["TC01", "Page load", "Open the site", "Model loads; status reads ready; no console errors", "Pass"], ["TC02", "Hero panel", "Wait on the home view", "Test images cycle with prediction and heat-map; counter advances", "Pass"],
    ["TC03", "Select a sample", "Click a thumbnail", "Prediction, confidence, ground truth and heat-map appear", "Pass"], ["TC04", "Hard case", "Press Hard case", "A misclassified image is shown; page notes Keras makes the same mistake", "Pass"],
    ["TC05", "Answer threshold", "Move the threshold slider", "Badge switches between Would answer and Would defer; coverage chart updates", "Pass"], ["TC06", "Preset", "Press Night or Sensor noise", "Sliders move; image darkens or noisy; prediction updates", "Pass"],
    ["TC07", "Sweep", "Press Sweep for noise or brightness", "A confidence-versus-severity line chart appears", "Pass"], ["TC08", "Verify against Keras", "Press the verify button", "Reports 116 of 116 identical; difference about 1e-5", "Pass"],
    ["TC09", "Upload", "Provide an image file", "Crop box appears; prediction updates; ground truth hidden", "Pass (synthetic image)"], ["TC10", "Camera", "Choose Camera and allow access", "Live frame with crop box; periodic prediction", "Not tested (no camera in the test environment)"],
    ["TC11", "Class inspection", "Click a bar in the F1 chart", "Class detail appears with confusions; matrix row highlighted", "Pass"], ["TC12", "Metric switch", "Select a different metric chip", "Bar chart redraws for the metric", "Pass"],
    ["TC13", "Table view", "Press Table on a chart card", "Data table replaces the chart", "Pass"], ["TC14", "Theme", "Cycle Theme auto, light, dark", "Colours switch; charts remain legible", "Pass"],
    ["TC15", "Narrow layout", "Set a 375 px viewport", "Single column; no horizontal overflow", "Pass"], ["TC16", "Keyboard", "Tab to a chart mark", "Tooltip appears on focus", "Pass (spot check)"],
    ["TC17", "Deployment", "Request the published URL and data files", "All files return HTTP 200; page works over HTTPS", "Pass"], ["TC18", "Hidden state", "Switch from a sample to an upload", "Ground-truth box disappears", "Pass (after fix, Section 8.5)"]];
  o.push(...tbl("tc", "Functional test cases for the website", ["ID", "Feature", "Steps", "Expected result", "Result"], tcs, [0.6, 1.2, 1.8, 3.0, 1.3], { size: 17 }));
  o.push(h2("8.5 Defects Found and Fixed"));
  o.push(...tbl("defects", "Defects found during development and testing", ["No.", "Defect", "Detection", "Resolution"],
    [["1", "Five classes had no validation images with <c>GroupShuffleSplit</c>; validation metrics and model selection were biased", "Printed check <i>all 43 classes in validation: False</i>; reproduced from file names", "Replaced by the explicit per-class track selection and added an assertion"],
     ["2", "A first repair with <c>StratifiedGroupKFold</c> passed locally but failed on Colab", "Assertion error in the notebook", "Deterministic NumPy-based selection independent of library version"], ["3", "Grad-CAM from the class probability was numerically unreliable for near-certain predictions", "Keras-versus-JavaScript map comparison", "Class-logit score used in the browser; difference documented"],
     ["4", "Ladder chart labels overlapped; Grad-CAM titles truncated speed limits", "Visual inspection of results", "Shorter labels and speed-limit abbreviations"], ["5", "A display rule overrode the HTML hidden attribute, leaving a ground-truth box visible after switching to an upload", "TC18 observation", "Global rule that makes the hidden attribute win"],
     ["6", "Figure counter also numbered the Chart/Table toggle; headline highlight clipped a letter", "Visual inspection", "Selector restricted; line height adjusted"]],
    [0.5, 3.3, 2.2, 2.4], { size: 18, align: [CENTER, LEFT, LEFT, LEFT] }));
  o.push(h2("8.6 Limitations of the Testing"));
  o.push(p("The Python package has no automated unit-test suite, so its modules rely on the assertions and the end-to-end run. The camera path of the website was not exercised. The website was not tested on Safari, on a range of physical phones or with assistive technology beyond keyboard focus. Timing figures were measured on the development laptop and on a Colab GPU and are not benchmarks of target hardware. Adding a pytest suite for the data and export code and automated browser tests for the website are listed as future work."));

  // =============================================== CHAPTER 9 ===============================================
  o.push(chapter("Results and Discussion"));
  o.push(p("All results in this chapter come from one complete run of the notebook in <i>full</i> mode on a Google Colab GPU runtime (TensorFlow 2.21.0, Keras 3.13.2, seed 42), unless stated otherwise. The test-set figures were computed once per model; validation figures are labelled as such."));
  o.push(h2("9.1 Results"));
  o.push(h3("9.1.1 Model Comparison"));
  o.push(p("{tab:cmp} compares the four models on the official test set. The custom CNN, which was chosen on validation accuracy (97.61 % against 90.55 % for fine-tuned MobileNetV2 and 97.07 % for the ensemble), achieves the highest test accuracy among the single models and the best calibration, with 116,203 parameters."));
  o.push(...tbl("cmp", "Performance of the four models on the official test set (12,630 images)", ["Model", "Val. acc.", "Test acc.", "Macro-F1", "Top-3", "ECE", "Parameters", "ms / image"],
    [["Custom CNN (selected)", "97.61 %", "<b>96.96 %</b>", "<b>95.21 %</b>", "99.03 %", "<b>0.0105</b>", "116,203", "18.0"], ["MobileNetV2, frozen", "72.58 %", "75.16 %", "67.46 %", "91.18 %", "0.0413", "2,313,067", "153.0"], ["MobileNetV2, fine-tuned", "90.55 %", "91.26 %", "86.59 %", "98.11 %", "0.0537", "2,313,067", "153.0"],
     ["Ensemble (CNN + MobileNetV2)", "97.07 %", "96.88 %", "95.07 %", "99.48 %", "0.0267", "2,429,270", "171.0"]],
    [2.3, 0.9, 0.9, 0.9, 0.8, 0.8, 1.1, 0.8], { size: 17, align: [LEFT, CENTER, CENTER, CENTER, CENTER, CENTER, RIGHT, CENTER] }));
  o.push(p("Fine-tuning improves MobileNetV2 by 16.1 points of test accuracy (75.16 % to 91.26 %) but it still ends 5.7 points below the custom CNN. The ensemble does not beat the best single model: it is slightly lower on both validation (97.07 % against 97.61 %) and test (96.88 % against 96.96 %), and its calibration is worse (ECE 0.0267 against 0.0105); its only advantage is a higher top-3 accuracy (99.48 % against 99.03 %). Latency was timed through single un-compiled calls on a GPU runtime and therefore includes Python overhead; it should be used to compare the models with each other, not as a device benchmark. On this basis MobileNetV2 is about eight to nine times slower than the CNN."));
  o.push(...fig("cmp", "nb/model_comparison.png", "Test accuracy and macro-F1 of the models, and accuracy versus latency", { width: 5.6 }));
  o.push(h3("9.1.2 Progress Ladder"));
  o.push(p("{tab:ladder} and {fig:ladder} show the ladder, measured on validation data. Along the from-scratch track every step improves the result: from 5.85 % for the majority-class baseline to 97.61 % for the final CNN, a reduction of the error rate from 94.1 % to 2.4 %, a factor of about 39."));
  o.push(...tbl("ladder", "Progress ladder: validation accuracy and macro-F1 at each step", ["Step", "Model", "Val. accuracy", "Val. macro-F1", "Parameters"],
    ladder.map((r) => [r.step, r.model.replace("Final custom CNN (+ augmentation, dropout)", "Final CNN (+ augmentation, dropout)"), pct(r.val_accuracy), pct(r.val_macro_f1, 1), Number(r.parameters).toLocaleString("en-US")]),
    [0.6, 3.6, 1.3, 1.3, 1.3], { size: 18, align: [CENTER, LEFT, CENTER, CENTER, RIGHT] }));
  o.push(...fig("ladder", "nb/progress_ladder.png", "Progress ladder: validation accuracy at each step (left) and the same data as error rate on a logarithmic scale (right)", { width: 5.7 }));
  o.push(p("The largest single gains after the linear model come from adding a hidden layer (+13.0 points), convolutions (+10.1 points), and batch normalisation with global pooling and regularisation (+1.8 and +2.5 points). The gains shrink as accuracy rises, which is why the error-rate view is informative. Bigger is not better here: the basic CNN has 1.63 million parameters, the final CNN 116 thousand, yet the final CNN is 4.2 points more accurate. The frozen MobileNetV2 (72.58 %) is below even the MLP; fine-tuning adds about 18 points but remains below the custom CNN. The ensemble (97.07 %) does not improve on its best member, so the “best so far” line is flat after step 5."));
  o.push(h3("9.1.3 Per-Class Performance"));
  const weak = per.slice().sort((a, b) => a.f1 - b.f1).slice(0, 8);
  o.push(p(`Of the 43 classes, 6 have an F1 score below 0.90 and 10 are at or above 0.99. {tab:weak} lists the eight weakest. The weakest class is Pedestrians, whose precision is high (0.94) but whose recall is only 0.50: half of the pedestrian signs are classified as something else, for example as Slippery road. Dangerous curve to the right shows the opposite pattern: recall is 1.00 but precision 0.69, because other classes (for example Wild animals crossing) are wrongly assigned to it. The Spearman rank correlation between class size and F1 is +0.28, a weak positive relationship, so class scarcity explains only part of the weakness; visual similarity among warning triangles matters more.`));
  o.push(...tbl("weak", "The eight classes with the lowest F1 on the test set", ["ID", "Class", "Precision", "Recall", "F1", "Test images", "Training images"],
    weak.map((r) => [r.class, r.name, parseFloat(r.precision).toFixed(3), parseFloat(r.recall).toFixed(3), parseFloat(r.f1).toFixed(3), r.support.replace(".0", ""), r.train_images]),
    [0.5, 3.0, 0.9, 0.8, 0.7, 0.9, 1.0], { size: 18, align: [CENTER, LEFT, CENTER, CENTER, CENTER, CENTER, CENTER] }));
  o.push(...fig("f1", "nb/per_class_f1.png", "F1 score of every class (left) and F1 against class size (right)", { width: 5.5, maxHeight: 5.2 }));
  o.push(h3("9.1.4 Confusion Analysis"));
  o.push(p("The custom CNN makes 384 errors on 12,630 test images. 75 % of them are confusions between signs of the same family (speed limits, prohibitory, warning, mandatory, other), against about 23 % expected by chance. {tab:conf} lists the ten most frequent confusions; the largest is Speed limit 60 km/h predicted as 80 km/h (36 images), a pair that differs in one digit."));
  const names = Object.fromEntries(per.map((r) => [r.class, r.name]));
  o.push(...tbl("conf", "The ten most frequent confusions on the test set", ["No.", "True class", "Predicted class", "Images"],
    topc.slice(0, 10).map((r, i) => [String(i + 1), names[r.true], names[r.pred], r.count]), [0.5, 3.2, 3.2, 0.9], { size: 18, align: [CENTER, LEFT, LEFT, CENTER] }));
  o.push(...fig("confm", "nb/confusion_matrix.png", "Row-normalised confusion matrix of the custom CNN on the test set", { width: 4.6, maxHeight: 5.6 }));
  o.push(...fig("conferr", "nb/confident_errors.png", "The twelve most confident errors on the test set", { width: 5.6 }));
  o.push(p("The most confident errors are dominated by very dark or very small images in which the digits of a speed limit are barely visible, so a wrong neighbouring speed limit is chosen with near-certainty."));
  o.push(h3("9.1.5 Explainability"));
  o.push(p("{fig:gcam} shows Grad-CAM maps of the custom CNN and of MobileNetV2 for five correct predictions and five confident errors (notebook version). For correct predictions the CNN's heat lies on the sign's symbol or rim: the arrow, the digits or the corners of a triangle. For the confident errors, which are dark speed-limit signs, the CNN's heat is scattered or sits on a fragment of one digit. MobileNetV2's final feature map is only 4 x 4, so its maps are coarse blocks that often cover the edge of the image or the background even for correct predictions, which is a reason to treat them with caution."));
  o.push(...fig("gcam", "nb/gradcam_grid.png", "Grad-CAM for the custom CNN (middle row) and MobileNetV2 (bottom row): five correct predictions (left) and five confident errors (right)", { width: 5.7 }));
  o.push(p("{fig:tsne} projects the 128-dimensional embedding of 4,000 test images with t-SNE. Images of the same class form tight clusters, and the clusters that touch are those of look-alike signs, in agreement with the confusion analysis."));
  o.push(...fig("tsne", "nb/tsne_embeddings.png", "t-SNE projection of the penultimate-layer embedding, coloured by sign family and by class", { width: 5.6 }));
  o.push(h3("9.1.6 Calibration and Selective Prediction"));
  o.push(p("The expected calibration error of the custom CNN is 0.0105, which is low; the fine-tuned MobileNetV2 is more over-confident (0.0537) and the ensemble is intermediate (0.0267). Wrong predictions are usually made with lower confidence than correct ones, so a threshold can trade coverage for accuracy ({tab:abst}, {fig:calib}). With a threshold of 0.866 the model answers 95.5 % of the test images and is right on at least 99.0 % of those; to reach 99.9 % it must defer on 17.4 % of the images. These thresholds are read off the test set and show the trade-off; a deployed system should choose its threshold on separate validation data."));
  o.push(...tbl("abst", "Selective prediction of the custom CNN on the test set", ["Target accuracy on answered images", "Confidence threshold", "Images answered", "Images deferred"],
    [["99.0 %", "0.866", "95.5 %", "4.5 %"], ["99.5 %", "0.972", "92.0 %", "8.0 %"], ["99.9 %", "0.999", "82.6 %", "17.4 %"]], [2.8, 1.8, 1.6, 1.5], { align: [CENTER, CENTER, CENTER, CENTER] }));
  o.push(...fig("calib", "nb/calibration_selective.png", "Confidence of correct and wrong predictions, reliability diagram and coverage-accuracy curve", { width: 5.7 }));
  o.push(h3("9.1.7 Robustness to Corruptions"));
  o.push(p("{tab:rob} and {fig:rob} show the accuracy of the custom CNN on 6,000 test images (clean accuracy 96.97 %) under six corruptions at three severities. Blur, darkening and glare are tolerated: accuracy stays above 94 % for blur even at the largest kernel and above 82 % for the strongest darkening or glare. The model is fragile to Gaussian noise (62.6 % at σ = 10 and 34.3 % at σ = 25, on a 0 to 255 scale), to occlusion (52.7 % when a fifth of the image is covered, 9.5 % when a third is) and to large rotation (51.6 % at 35 degrees, beyond the roughly 14 degrees used in training). Noise and occlusion were never seen in training, and the rotation result shows a cliff beyond the trained range."));
  o.push(...tbl("rob", "Accuracy of the custom CNN under corruptions (6,000 test images; clean accuracy 96.97 %)", ["Corruption", "Mild", "Medium", "Severe", "Levels (mild, medium, severe)"],
    rob.map((r) => { const k = Object.values(r)[0]; const lv = { "Darkening": "x0.7, x0.5, x0.3", "Glare": "x1.4, x1.8, x2.4", "Gaussian": "sigma 10, 25, 50", "Blur": "kernel 3, 5, 7", "Rotation": "10, 20, 35 degrees", "Occlusion": "10 %, 20 %, 35 % of area" }; const key = Object.keys(lv).find((x) => k.startsWith(x)); return [k.replace("×brightness", "(brightness)").replace(" (σ)", " (sigma)").replace(" (°)", "").replace(" (area share)", ""), pct(r.mild, 1), pct(r.medium, 1), pct(r.severe, 1), lv[key]]; }),
    [2.2, 1.0, 1.0, 1.0, 2.5], { size: 18, align: [LEFT, CENTER, CENTER, CENTER, LEFT] }));
  o.push(...fig("rob", "nb/robustness.png", "Accuracy of the custom CNN under increasing corruption severity", { width: 5.7 }));
  o.push(...fig("robex", "nb/robustness_examples.png", "Examples of the corruptions at the three severities", { width: 3.6, maxHeight: 6.6 }));
  o.push(h3("9.1.8 Export and Deployment"));
  o.push(p("The selected CNN converts to a TensorFlow Lite file of 0.24 MB (the Keras file is 1.5 MB), which agrees with the Keras model on 100 % of 500 test images, scores 97.6 % on them and runs at 1.24 ms per image on a single CPU thread. The browser implementation loads a 465 KB weight file, and a prediction with Grad-CAM takes about a tenth of a second in Node.js on the development laptop. The whole results bundle (four models, per-model predictions, tables, figures and a sample of test images) is 86.5 MB, of which the MobileNetV2 and ensemble models make up most."));
  o.push(h2("9.2 Discussion"));
  o.push(h3("9.2.1 Merits and Demerits of the Approach"));
  o.push(...tbl("merit", "Merits and demerits of the proposed approach and pipeline", ["Merits", "Demerits"],
    [["Leakage-free, class-complete validation; test set used once", "Handles cropped signs only; needs a detector for real scenes"], ["Compact model: 116,203 parameters, 0.24 MB as TFLite, 18 ms per image in a GPU runtime", "Fragile to noise, occlusion and large rotation"], ["Well calibrated (ECE 0.0105); supports abstention with measured trade-off", "Over-confident on inputs unlike its training data (for example drawings)"],
     ["Explained with Grad-CAM and t-SNE; explanation method checked numerically", "Grad-CAM maps are coarse and qualitative"], ["Each design step measured; transfer learning compared fairly", "Pretrained MobileNetV2 over-fits and does not beat the small CNN at this resolution"], ["Reproducible, with seeds, checksums and assertions", "Single seed; run-to-run spread about half a point"],
     ["Runs in the browser with no server; private; verified against Keras", "Only the custom CNN is deployed in the browser; weak classes (Pedestrians recall 0.50)"]],
    [3.9, 3.8], { size: 18, align: [LEFT, LEFT] }));
  o.push(h3("9.2.2 Comparison with Published Results"));
  o.push(p("The best published GTSRB results are close to or above human accuracy: about 98.8 % for human participants {@stallkamp2012} and about 99.5 % for a committee of CNNs {@ciresan2012}. The custom CNN of this study reaches 96.96 %, below both. This is expected from its design: a single small network on 64 x 64 inputs trained for a few minutes, without the input normalisation variants, multi-network committees or long training schedules used in the competition. The aim of the project was not to beat these numbers but to measure honestly, explain and stress a compact model. The result is nevertheless in the range expected for compact CNNs, and its error rate of about 3 % is dominated by identifiable factors (dark speed-limit signs, look-alike warning triangles, a few weak classes) that suggest concrete improvements (Chapter 11)."));
  o.push(h3("9.2.3 Why Did the Small CNN Beat MobileNetV2?"));
  o.push(p("Three observations are consistent with the domain-gap explanation proposed in Chapter 3. First, the frozen backbone performs barely better than a linear classifier on raw pixels (72.6 % against 70.2 %, and far below the 83.2 % of the MLP), meaning that ImageNet features at this scale add little. Second, fine-tuning helps substantially (+18 points), meaning that adapting the features matters. Third, the fine-tuned network over-fits (99.8 % training accuracy against 90.6 % validation), so its extra capacity is not constrained by the 33,570 highly correlated training images. None of these observations proves the explanation, and the result should not be generalised: it concerns tiny, low-resolution crops and a frozen or lightly tuned MobileNetV2, not pretrained networks in general. A different resolution, stronger augmentation or a larger unfrozen portion might change the outcome."));
  o.push(h3("9.2.4 Grad-CAM at Saturation"));
  o.push(p("The comparison of two correct implementations of Grad-CAM exposed a weakness of using the class probability as the score. When the network is almost certain, the probability is close to 1 and its derivative with respect to the logits, which is proportional to p(1 - p), is close to zero. The gradients that reach the last convolutional layer are then of the order of floating-point rounding, and after normalisation by the maximum the heat-map is dominated by numerical noise. This happened for 50 of the 116 sample images, and the maps of Keras and of the JavaScript code disagreed on exactly those images, while agreeing to 5.9 x 10<sup>-4</sup> on the other 66. Using the class logit instead, as in the original Grad-CAM paper {@selvaraju2017}, removes the problem: the two implementations then agree to 3.1 x 10<sup>-6</sup> on all 116 images. The website therefore uses the logit; the notebook figures in this report use the probability and are most reliable for the less confident cases."));
  o.push(h3("9.2.5 Run-to-Run Variation"));
  o.push(p("The first full run of the pipeline used a validation split with the defect described in Section 7.1.2; the second, reported here, used the corrected split. The two runs share the code, seed and budgets and differ only in the training data of the fitting set and the validation set, so they give a direct, if small, indication of run-to-run variation ({tab:var})."));
  o.push(...tbl("var", "Test accuracy in the two full runs", ["Model", "First run", "Reported run", "Difference"],
    [["Custom CNN", "96.52 %", "96.96 %", "+0.44"], ["MobileNetV2, fine-tuned", "92.35 %", "91.26 %", "-1.09"], ["Ensemble", "97.01 %", "96.88 %", "-0.13"]], [3.0, 1.5, 1.5, 1.4], { align: [LEFT, CENTER, CENTER, CENTER] }));
  o.push(p("Differences of about half a point to one point are therefore within the run-to-run spread, and the ordering of the ensemble and the CNN flipped on the test set between the runs. The large effects (baseline to CNN, frozen to fine-tuned, CNN against fine-tuned MobileNetV2) are many times larger than this spread and held in both runs. Only one seed was used in each run, so a proper confidence interval would require repetition with several seeds."));
  o.push(h2("9.3 Findings"));
  o.push(...numbered([
    "<b>Random validation leaks.</b> On GTSRB a random split puts 1,291 of 1,307 physical signs on both sides; every validation sign has been seen in training. Grouping by sign removes this, and the grouping must also guarantee class coverage.",
    "<b>A small, well-regularised CNN is enough.</b> A 116,203-parameter CNN reaches 96.96 % test accuracy and 95.21 % macro-F1 and is better calibrated and about eight times faster than the fine-tuned MobileNetV2.",
    "<b>Each design step matters, and size does not.</b> The ladder rises monotonically from 5.85 % to 97.61 %; a plain CNN with 14 times more parameters is 4 points worse than the final design.",
    "<b>Transfer learning needs fine-tuning, and here it still falls short.</b> A frozen backbone reaches 75.16 %, fine-tuning 91.26 %, and the fine-tuned model over-fits.",
    "<b>Errors are between look-alike signs.</b> 75 % of errors stay inside a sign family; the most frequent is Speed limit 60 predicted as 80. Class size explains only part of the weak classes (Spearman +0.28).",
    "<b>The model can say when it is unsure.</b> ECE is 0.0105; answering only above a confidence of 0.866 gives 99 % accuracy at 95.5 % coverage.",
    "<b>The weaknesses are noise, occlusion and large rotation.</b> Accuracy falls to 34 % at noise sigma 25, 53 % with a fifth of the image covered and 52 % at 35 degrees, while blur, darkening and glare are tolerated.",
    "<b>The most confident errors are very dark speed-limit signs.</b>",
    "<b>Probability-based Grad-CAM is unreliable at saturation.</b> The class logit should be used as the score.",
    "<b>Compact deployment is practical.</b> The selected network runs in the browser from a 465 KB file, matches Keras to about 3 x 10<sup>-6</sup>, and converts to a 0.24 MB TFLite file with 100 % agreement."]));
  return o.flat();
};
