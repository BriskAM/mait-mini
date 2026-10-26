# Sign Reader: the GitHub Pages site

A static, dependency-free site (plain HTML, CSS and ES modules; no build step, no CDN). It does three things:

1. **Runs the trained custom CNN in the browser** (`js/nn.js`): the weights exported from the notebook's Keras model are loaded as a 465 KB float32 file and the network is re-implemented in JavaScript, including a hand-written Grad-CAM backward pass. Nothing is uploaded anywhere.
2. **A stress lab** that applies the notebook's corruption families (rotation, brightness, blur, noise, occlusion) to the current image and sweeps each one.
3. **A results dashboard** drawn from the notebook's results bundle: model comparison, progress ladder, training curves, per-class F1, confusion matrix, calibration, abstention, robustness.

## Run locally

Browsers block `fetch` from `file://`, so serve the folder:

```bash
python3 -m http.server 8000 --directory site
```

Then open http://localhost:8000.

## Deploy

`.github/workflows/pages.yml` publishes `site/` on every push to `main` that touches it. One-time setup: **Settings > Pages > Build and deployment > Source: GitHub Actions**. The site is then served at `https://<user>.github.io/<repo>/`.

## Layout

```
index.html            page structure and copy
css/style.css         design tokens (light and dark) and components
js/nn.js              CNN inference + Grad-CAM (no dependencies)
js/imaging.js         canvas helpers, corruptions, Grad-CAM overlay
js/charts.js          small SVG/canvas chart kit (tooltips, keyboard focus, table views)
js/demo.js            live demo, upload/camera cropper, stress lab, parity check
js/dashboard.js       every results chart
js/state.js           the shared abstention threshold
data/                 generated: cnn_weights.bin, cnn_manifest.json, results.json, samples.png/.json
img/                  generated: notebook figures (WebP)
```

## Regenerate the data

`data/` and `img/` are generated from the zip the demo notebook downloads at the end of a run. Unzip it, then (needs numpy, pandas, Pillow, h5py):

```bash
python scripts/build_pages_data.py --results path/to/unzipped/results --data data/raw/gtsrb --out site
```

`--data` is only used for the class-count and image-width charts. Facts that appear in the notebook's printed output but not in the bundle (track counts, split sizes, parameter counts) are constants at the top of `scripts/build_pages_data.py`; update them if you re-run the notebook and they change.

## Check the JavaScript network against Keras

The page has a **Verify against Keras** button: it re-runs the 116 sample images and compares predictions and confidences with the values the notebook's Keras model recorded. For a stricter offline check of probabilities and Grad-CAM maps, generate Keras reference outputs for the same tiles and run:

```bash
node scripts/verify_pages_model.mjs --ref ref.json --tiles tiles.u8
```

When this was built it matched Keras to about 3e-6 in probability and 3e-6 in the Grad-CAM map on all 116 images.

## Notes and limits

- **Grad-CAM score.** The page backpropagates the class score before the softmax (as in the Grad-CAM paper). The notebook uses the post-softmax probability, whose gradient vanishes when the model is nearly certain, so its maps are numerically unreliable there (about 40% of the sample images). Both variants are implemented in `nn.js` (`score = "logit" | "prob"`).
- **Resampling.** Uploaded photos are shrunk with the browser's canvas resampler, which differs slightly from the PIL resize used in training. The built-in test images are the exact 64x64 arrays the model saw.
- **Only the custom CNN runs here.** MobileNetV2 and the ensemble appear in the charts but are larger and scored lower.
- **Data.** The sample images are cropped from the GTSRB test set (Stallkamp et al., IJCNN 2011). Check the dataset's terms before redistributing them publicly.
- Research and education only; not a driving system.
