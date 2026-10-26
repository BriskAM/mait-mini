const L = require("../lib");
const { h2, h3, h3u, p, bullets, numbered, tbl, code, plainHeading, fs, path } = L;
const { AlignmentType } = L.d;
const CENTER = AlignmentType.CENTER, LEFT = AlignmentType.LEFT, RIGHT = AlignmentType.RIGHT;
const csv = (f) => L.readCsv(f);

module.exports = function back() {
  const o = [];
  // ---------------------------------------------- Appendix A -----------------------------------------------------------------------------
  o.push(plainHeading("Appendix A: Source Code of Selected Modules (Annexure 1)", { toc: true, size: 28, upper: false }));
  L.setAppendix("A");
  o.push(p("The complete source code is available in the project repository (https://github.com/BriskAM/mait-mini). This appendix reproduces selected modules that complement the listings of Chapter 7."));
  o.push(...code("a1", "Dataset download with streaming MD5 verification (notebook)", [
    "def md5_of(path, chunk=1 << 20):",
    "    h = hashlib.md5()",
    "    with open(path, \"rb\") as f:",
    "        for block in iter(lambda: f.read(chunk), b\"\"): h.update(block)",
    "    return h.hexdigest()",
    "",
    "if not archive.exists() or md5_of(archive) != EXPECTED_MD5:",
    "    urllib.request.urlretrieve(URL, archive, hook)          # progress hook prints every ~25 MB",
    "got = md5_of(archive)",
    "assert got == EXPECTED_MD5, \"Checksum mismatch: delete data.zip and rerun.\"",
    "with zipfile.ZipFile(archive) as z: z.extractall(WORK / \"gtsrb\")"]));
  o.push(...code("a2", "Training helper with early stopping and learning-rate reduction (notebook)", [
    "def fit(model, epochs, lr, patience, tag):",
    "    model.compile(optimizer=keras.optimizers.Adam(lr), loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])",
    "    cbs = [keras.callbacks.EarlyStopping(monitor=\"val_accuracy\", mode=\"max\", patience=patience, restore_best_weights=True),",
    "           keras.callbacks.ReduceLROnPlateau(monitor=\"val_loss\", factor=0.4, patience=max(1, patience // 2), min_lr=1e-6)]",
    "    t0 = time.time()",
    "    h = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), batch_size=BATCH, epochs=epochs, callbacks=cbs, verbose=2)",
    "    print(f\"[{tag}] {len(h.history['loss'])} epochs in {time.time() - t0:.0f}s | best val acc {max(h.history['val_accuracy']):.4f}\")",
    "    return h.history, time.time() - t0"]));
  o.push(...code("a3", "Scoring: accuracy, macro-F1, top-3 accuracy and expected calibration error (notebook)", [
    "def ece_score(conf, correct, bins=15):",
    "    edges = np.linspace(0, 1, bins + 1); e = 0.0",
    "    for lo, hi in zip(edges[:-1], edges[1:]):",
    "        m = (conf > lo) & (conf <= hi)",
    "        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())",
    "    return e",
    "",
    "def score_from_probs(p, y):",
    "    pred = p.argmax(1); conf = p.max(1)",
    "    return dict(probs=p, pred=pred, conf=conf, accuracy=(pred == y).mean(),",
    "                macro_f1=f1_score(y, pred, average=\"macro\", labels=range(NUM_CLASSES), zero_division=0),",
    "                top3=np.mean([yy in np.argsort(pp)[-3:] for yy, pp in zip(y, p)]), ece=ece_score(conf, pred == y))"]));
  o.push(...code("a4", "Grad-CAM for a flat model (Python, notebook version)", [
    "def gradcam(model, x_uint8, target=None):",
    "    kind, cam_model, idx = _cam_parts(model); x = tf.convert_to_tensor(x_uint8[None].astype(\"float32\"))",
    "    with tf.GradientTape() as tape:",
    "        conv_out, probs = cam_model(x, training=False)       # last conv output and network output",
    "        cls = int(tf.argmax(probs[0])) if target is None else target",
    "        score = probs[:, cls]                                  # post-softmax score (see Section 9.2.4)",
    "    grads = tape.gradient(score, conv_out)",
    "    cam = tf.nn.relu(tf.reduce_sum(conv_out * tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :], axis=-1))[0].numpy()",
    "    cam = cam / (cam.max() + 1e-8)",
    "    return cv2.resize(cam, (64, 64), interpolation=cv2.INTER_CUBIC).clip(0, 1), cls"]));
  o.push(...code("a5", "Convolution and max pooling in JavaScript (nn.js)", [
    "function conv3x3(inp, H, W, Cin, w, b, Cout) {",
    "  const out = new Float32Array(H * W * Cout);",
    "  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {",
    "    const o = (y * W + x) * Cout;",
    "    for (let co = 0; co < Cout; co++) out[o + co] = b[co];",
    "    for (let ky = 0; ky < 3; ky++) { const yy = y + ky - 1; if (yy < 0 || yy >= H) continue;",
    "      for (let kx = 0; kx < 3; kx++) { const xx = x + kx - 1; if (xx < 0 || xx >= W) continue;",
    "        const ib = (yy * W + xx) * Cin, wb = (ky * 3 + kx) * Cin * Cout;",
    "        for (let ci = 0; ci < Cin; ci++) { const v = inp[ib + ci], wo = wb + ci * Cout;",
    "          for (let co = 0; co < Cout; co++) out[o + co] += v * w[wo + co]; } } }",
    "    for (let co = 0; co < Cout; co++) if (out[o + co] < 0) out[o + co] = 0;      // ReLU",
    "  }",
    "  return out;",
    "}"]));
  o.push(...code("a6", "Exporting the Keras weights for the browser (build_pages_data.py, abridged)", [
    "order = [(\"conv1\", \"conv2d\", 2), (\"bn1\", \"batch_normalization\", 4), (\"conv2\", \"conv2d_1\", 2), (\"bn2\", \"batch_normalization_1\", 4),",
    "         (\"conv3\", \"conv2d_2\", 2), (\"bn3\", \"batch_normalization_2\", 4), (\"dense1\", \"dense\", 2), (\"dense2\", \"dense_1\", 2)]",
    "for tag, group, n in order:",
    "    for i in range(n):",
    "        arr = np.asarray(h5[f\"layers/{group}/vars/{i}\"], dtype=\"<f4\")",
    "        entries.append({\"name\": f\"{tag}.{names[key][i]}\", \"shape\": list(arr.shape), \"offset\": offset})",
    "        chunks.append(arr.ravel()); offset += arr.size",
    "blob = np.concatenate(chunks).astype(\"<f4\")       # 116,203 float32 values = 465 KB",
    "(out / \"data\" / \"cnn_weights.bin\").write_bytes(blob.tobytes())"]));

  // ---------------------------------------------- Appendix B -----------------------------------------------------------------------------
  o.push(plainHeading("Appendix B: Per-Class Results of the Custom CNN", { toc: true, size: 28, upper: false }));
  L.setAppendix("B");
  const per = csv("per_class_metrics_selected_model.csv");
  o.push(p("{tab:perclass} gives the precision, recall and F1 score of every class for the selected custom CNN on the official test set, together with the number of test and training images."));
  o.push(...tbl("perclass", "Per-class test results of the custom CNN", ["ID", "Class", "Precision", "Recall", "F1", "Test", "Train"],
    per.map((r) => [r.class, r.name, parseFloat(r.precision).toFixed(3), parseFloat(r.recall).toFixed(3), parseFloat(r.f1).toFixed(3), r.support.replace(".0", ""), r.train_images]),
    [0.5, 3.6, 0.9, 0.8, 0.7, 0.6, 0.7], { size: 17, align: [CENTER, LEFT, CENTER, CENTER, CENTER, CENTER, CENTER] }));

  // ---------------------------------------------- Appendix C -----------------------------------------------------------------------------
  o.push(plainHeading("Appendix C: Reproducing the Work", { toc: true, size: 28, upper: false }));
  L.setAppendix("C");
  o.push(h2("C.1 Running the Experiments"));
  o.push(...numbered([
    "Open the notebook <c>notebooks/traffic_sign_recognition_demo.ipynb</c> in Google Colab (the badge in the repository README opens it directly).",
    "Choose <i>Runtime, Change runtime type, GPU</i>.",
    "In the first code cell set <c>MODE</c> to <c>smoke</c> (2-minute sanity run), <c>demo</c> or <c>full</c> (reported results).",
    "Run all cells. The notebook downloads and verifies the dataset, trains and evaluates the models, and finally downloads <c>gtsrb_demo_results.zip</c> with the models, results and figures."]));
  o.push(h2("C.2 Rebuilding the Website Data"));
  o.push(...numbered([
    "Unzip the results bundle to a folder.",
    "Run <c>python scripts/build_pages_data.py --results <folder> --data data/raw/gtsrb --out site</c> (needs numpy, pandas, Pillow and h5py).",
    "Serve the site locally with <c>python3 -m http.server 8000 --directory site</c> and open http://localhost:8000.",
    "Verify the browser network against Keras with <c>node scripts/verify_pages_model.mjs</c> (using reference outputs generated from the same weights).",
    "Push to the <c>main</c> branch; the workflow <c>.github/workflows/pages.yml</c> publishes the site to GitHub Pages."]));
  o.push(h2("C.3 Repository Layout"));
  o.push(...tbl("layout", "Layout of the project repository", ["Folder or file", "Contents"],
    [["<c>notebooks/</c>", "The demonstration notebook (generated by <c>scripts/build_demo_notebook.py</c>) and an earlier lighter notebook"], ["<c>src/traffic_sign_recognition/</c>", "Reusable Python package: data, models, training, evaluation, Grad-CAM, inference"],
     ["<c>scripts/</c>", "Dataset download, analysis and document builders; website data build; browser-network verification"], ["<c>site/</c>", "The static website: page, styles, JavaScript modules, generated data and figures"],
     ["<c>docs/</c>", "Earlier synopsis and report, literature survey, presenter documentation"], ["<c>report/</c>", "Sources that generate this report (diagrams, content, build script)"]],
    [2.6, 5.1], { size: 19, align: [LEFT, LEFT] }));

  // ---------------------------------------------- Appendix D -----------------------------------------------------------------------------
  o.push(plainHeading("Appendix D: Research and Publication Status (Annexure 2)", { toc: true, size: 28, upper: false }));
  L.setAppendix("D");
  o.push(p("As required by the departmental notice, the status of research output related to this project is as follows."));
  o.push(...tbl("pub", "Research and publication status", ["Item", "Status"],
    [["Book chapter", "None submitted, accepted or published"], ["Conference paper", "None submitted, accepted or published"], ["Research article", "None submitted, accepted or published"], ["Hard copy of research paper", "Not applicable"]],
    [3.0, 4.6], { align: [LEFT, LEFT] }));
  o.push(p("The experimental results of this project (the leakage analysis, the progress ladder and the reliability study) could form the basis of a short paper; this is listed as a possible follow-up."));
  return o.flat();
};
