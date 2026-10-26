# Results and Findings (final Colab run, full mode)

Source: `gtsrb_demo_results (1).zip` and the executed notebook `traffic_sign_recognition_demo (1).ipynb` from the re-run on 26 Oct 2026. TensorFlow 2.21.0, Keras 3.13.2, one Colab GPU, `MODE = "full"`, seed 42. Every number below is read from those two files. The notebook ran with no errors and printed `all 43 classes in validation: True`.

This is the run to present from. An earlier run (section 7) used a flawed validation split and is kept only as a record.

---

## 1. Headline results

### Official test set (12,630 images, evaluated once)
| Model | Accuracy | Macro-F1 | Top-3 | ECE | Parameters | Latency (ms/img, bs=1) |
|---|---:|---:|---:|---:|---:|---:|
| **Custom CNN (selected)** | **96.96%** | **95.21%** | 99.03% | 0.0105 | 116,203 | 18.0 |
| MobileNetV2, head only | 75.16% | 67.46% | 91.18% | 0.041 | 2,313,067 | 153.0 |
| MobileNetV2, fine-tuned | 91.26% | 86.59% | 98.11% | 0.054 | 2,313,067 | 153.0 |
| Ensemble (CNN + fine-tuned MobileNetV2) | 96.88% | 95.07% | 99.48% | 0.027 | 2,429,270 | 171.0 |

- **Selected model:** Custom CNN, chosen on validation accuracy (97.61%, against 90.55% for fine-tuned MobileNetV2 and 97.07% for the ensemble).
- **Fine-tuning gain:** +16.1 points of test accuracy (75.16% to 91.26%).
- **Ensemble:** no gain. It is slightly below the CNN on both validation (97.07% against 97.61%) and test (96.88% against 96.96%), and its calibration is worse (ECE 0.027 against 0.0105). Its only edge is top-3 accuracy (99.48% against 99.03%).
- **Latency caveat:** timed through single un-compiled calls on the Colab GPU runtime, so it includes Python overhead. Say "about eight to nine times slower than the CNN in the same setup", not an absolute figure.
- **Compared with the earlier recorded project run** (CNN 97.12% and 95.14%; frozen MobileNetV2 76.18% and 68.23%): similar, within about half a point for the CNN. The pipelines differ, so do not present this as an exact replication.

### Training cost
| Stage | Epochs run (of budget) | Time | Best validation accuracy |
|---|---|---:|---:|
| Custom CNN | 29 of 30 | 341 s | 97.61% |
| MobileNetV2 stage 1 (head) | 12 of 12 | 244 s | 72.58% |
| MobileNetV2 stage 2 (fine-tune) | 14 of 15 | 422 s | 90.55% |
| Ladder rungs 1 to 4 | 15, 15, 25, 24 | 17, 20, 107, 150 s | see ladder |

About 22 minutes of training in total, so the full mode is practical to re-run.

### Progress ladder (validation set)
| Step | Model | Val accuracy | Val macro-F1 | Parameters |
|---:|---|---:|---:|---:|
| 0 | Majority-class baseline | 5.85% | 0.003 | 0 |
| 1 | Linear classifier, 16x16 pixels | 70.19% | 0.611 | 33,067 |
| 2 | MLP, 32x32 pixels | 83.24% | 0.780 | 825,131 |
| 3 | Basic CNN (2 conv layers) | 93.37% | 0.918 | 1,630,699 |
| 4 | Deeper CNN + BatchNorm + global pooling | 95.14% | 0.938 | 116,203 |
| 5 | + augmentation and dropout (final CNN) | 97.61% | 0.969 | 116,203 |
| 6 | MobileNetV2, frozen | 72.58% | 0.653 | 2,313,067 |
| 7 | MobileNetV2, fine-tuned | 90.55% | 0.889 | 2,313,067 |
| 8 | Ensemble | 97.07% | 0.967 | 2,429,270 |

Error rate fell from 94.1% to 2.4% along the from-scratch track, a factor of about 39. Validation macro-F1 is now close to accuracy, as it should be with all 43 classes present.

### What the ladder says (your talking points)
1. **Every step on the from-scratch track helped**, from 70.2% to 97.6%. Gains shrink as accuracy rises (+13, +10, +1.8, +2.5 points), which is why the error-rate panel is useful.
2. **Bigger was not better.** The basic CNN has 1.63 million parameters; the final CNN has 116 thousand and is 4.2 points better. Global average pooling instead of a huge flatten-and-dense layer, batch norm, and regularisation matter more than size.
3. **Transfer learning needed fine-tuning.** The frozen backbone sits at 72.6%, just above the linear model. Fine-tuning adds about 18 points but still ends below the custom CNN.
4. **MobileNetV2 overfits after fine-tuning.** Final training accuracy is 99.84% against 90.55% validation; the CNN's gap is much smaller (99.45% against 97.59%).
5. **The ensemble does not beat the best single model** in this run, so the simple one-model choice stands.

---

## 2. Run-to-run variation (use this when asked about seeds and significance)

The earlier run and this run used the same pipeline and seed; they differed only in how the validation set was chosen (which changes the training data slightly). Results moved:

| Model | Earlier run (test accuracy) | This run (test accuracy) |
|---|---:|---:|
| Custom CNN | 96.52% | 96.96% |
| MobileNetV2, fine-tuned | 92.35% | 91.26% |
| Ensemble | 97.01% | 96.88% |

So **differences of about half a point to one point are within run-to-run noise**, and the ensemble-versus-CNN ordering flipped on the test set between the two runs. The large effects (baseline to CNN, frozen to fine-tuned, CNN versus fine-tuned MobileNetV2) are far larger than this spread and held in both runs. This is direct evidence for the "single seed" limitation, and for why we do not claim small differences.

---

## 3. Further findings

### Robustness (selected CNN, 6,000 test images, clean accuracy 97.0%)
| Corruption | Mild | Medium | Severe | Reading |
|---|---:|---:|---:|---|
| Blur (kernel 3, 5, 7) | 97.1% | 96.6% | 95.0% | almost unaffected |
| Darkening (x0.7, x0.5, x0.3) | 96.3% | 93.9% | 82.2% | graceful |
| Glare (x1.4, x1.8, x2.4) | 94.8% | 91.8% | 84.6% | graceful |
| Rotation (10, 20, 35 degrees) | 96.5% | 93.9% | 51.6% | cliff beyond the trained range (training uses about 14 degrees) |
| Occlusion (10%, 20%, 35% of area) | 82.6% | 52.7% | 9.5% | severe weakness |
| Gaussian noise (sigma 10, 25, 50) | 62.6% | 34.3% | 14.1% | **largest weakness**: even mild noise costs about 34 points |

The model never saw noise or occlusion in training, which explains the two worst rows; rotation collapses once it leaves the trained range. Blur, darkening and glare are handled well.

### Errors
- **384 errors in 12,630** test images; **75% are between signs of the same family**, against about 23% by chance.
- **Most frequent confusions:** Limit 60 predicted as Limit 80 (36 images), No entry as Priority road (20), Bumpy road as Road work (18), Limit 120 as Limit 80 (16), Wild animals crossing as Dangerous curve to the right (15), Priority road as Yield (14), Road work as Bumpy road (12), Pedestrians as Slippery road (12), Double curve as General caution (11), Limit 70 as Limit 30 (10).
- **Weakest classes** (6 of 43 have F1 below 0.90; 10 are at or above 0.99): Pedestrians 0.65 (precision 0.94 but recall only 0.50), Dangerous curve to the right 0.82 (recall 1.00, precision 0.69, so other signs get called this), Double curve 0.83, Bumpy road 0.85, Beware of ice/snow 0.89, Roundabout mandatory 0.89.
- **Size versus F1:** Spearman +0.28, a weak positive, so class scarcity explains only part of the weakness.
- **Confident errors** (Grad-CAM grid, the five most confident mistakes): all are dark, low-contrast speed-limit signs; two true 50 km/h signs read as Limit 30, and three true 60 km/h signs read as Limit 80, with the CNN at about 100% confidence. The CNN's heat sits on a fragment of one digit or is scattered. On correct predictions its heat is on the sign's symbol or rim.
- **MobileNetV2's maps** (4 by 4 resolution) are large blobs and often cover the edge of the image or the background rather than the sign, even on some correct predictions, so do not over-interpret them.

### Calibration and abstention (custom CNN, test set)
ECE 0.0105, which is good. To reach at least 99.0% accuracy the system answers 95.5% of images (threshold 0.866); for 99.5% it answers 92.0% (threshold 0.972); for 99.9% it answers 82.6% (threshold 0.999). These thresholds are read off test data, so treat them as a demonstration of the trade-off, not a tuned operating point.

### Export
Every reload parity check passed. The selected CNN converts to a 0.24 MB TFLite file (the Keras file is 1.5 MB) that agrees with the Keras model on 100% of 500 test images, scores 97.6% on them, and runs at 1.24 ms per image on a single CPU thread. The results bundle is about 86.5 MB.

---

## 4. Data facts (measured, unchanged between runs)

| Fact | Value |
|---|---|
| Training / test images, classes | 39,209 / 12,630 / 43 |
| Training tracks (physical signs) | 1,307; 30 frames per track (29 to 30) |
| Largest class | Speed limit 50 km/h, 2,250 images |
| Smallest class | Speed limit 20 km/h, 210 images |
| Imbalance | 10.7 times |
| Image width | 25 to 243 px, median 43 px, 40% under 40 px |
| Aspect ratio (width over height) | median 1.00, 5% to 95% range 0.91 to 1.13 |
| Brightness (grey level, 3,000 sample) | 82 plus or minus 48, from 7 (almost black) to 248 |
| Random split leakage | 1,291 of the 1,307 tracks appear on both sides; 100% of validation tracks had been seen in training |
| Grouped split used | 33,570 fit images (1,119 tracks); 5,639 validation images (188 tracks); zero track overlap; all 43 classes present |
| Cached arrays | 0.64 GB, built in 24 s |
| CNN parameters | 116,203 total, 115,755 trainable |
| MobileNetV2 parameters | 2,313,067 total; 55,083 trainable at stage 1; 1,834,219 trainable at stage 2 |

---

## 5. Script lines using the real results (adapt to your voice)

- *Results:* "The compact CNN scores 97.0% accuracy and 95.2% macro-F1 on the official test set with only 116 thousand parameters. The fine-tuned MobileNetV2 reaches 91.3%, and averaging the two does not help: 96.9%."
- *Ladder:* "Guessing the most common class gives 6%. A linear model on tiny pixel images already reaches 70%. A hidden layer takes it to 83%, convolutions to 93%, batch norm and global pooling to 95%, augmentation and dropout to 97.6%. Error fell about forty-fold."
- *Transfer learning:* "A frozen pretrained network scored 73%. Fine-tuning raised it to 91%, but it overfits: 99.8% training accuracy against 91% validation. For small low-resolution sign crops, the small purpose-built network won."
- *Reliability:* "The CNN is well calibrated, ECE 0.01. If we only answer when confident, we reach 99% accuracy while answering 95.5% of images."
- *Robustness (be candid):* "It handles blur, darkening and glare well. It is weak against noise and occlusion, which it never saw in training; that is the first thing we would fix."
- *Errors:* "Our most confident mistakes are very dark speed-limit signs read as a different speed limit."
- *Variation:* "Two runs of this pipeline gave 96.5% and 97.0% for the CNN, so we treat differences under a point as noise."

---

## 6. What to improve next, ranked by the measured weaknesses

| Priority | Change | Targets (evidence) |
|---|---|---|
| 1 | Add Gaussian noise and random-erasing (occlusion) augmentation, and widen the rotation range | Noise 34% at sigma 25, occlusion 53% at 20% area, rotation 52% at 35 degrees |
| 2 | Brightness augmentation and contrast normalisation (per-image standardisation or CLAHE) | Dark speed-limit signs are the confident errors; darkening drops accuracy to 82% |
| 3 | Higher input resolution for the CNN (96 pixels) | Limit 60 versus 80 (36), 120 versus 80 (16), 70 versus 30 (10) |
| 4 | Class weights or targeted augmentation for Pedestrians, Dangerous curve to the right, Double curve | Lowest-F1 classes; Pedestrians recall is 50% |
| 5 | Stronger regularisation for fine-tuned MobileNetV2 (more augmentation, fewer unfrozen layers, label smoothing) | Training 99.8% versus validation 90.6% |
| 6 | Several seeds for every model | Quantify the run-to-run spread seen in section 2 and settle the ensemble question |
| 7 | Temperature scaling and a validation-chosen abstention threshold | Ensemble and MobileNetV2 ECE are higher; the threshold is currently read from test data |

Priorities 1 and 2 are cheap, change only the augmentation block, and the robustness table already provides the before-and-after measure. If you add them to the notebook as new ladder rungs, the ladder can end on a real improvement.

---

## 7. Record of the earlier (flawed) run

The first full run used `GroupShuffleSplit`, which left classes 0, 19, 21, 23 and 29 without validation images. Test numbers were unaffected, but validation accuracy, macro-F1 and the ladder were biased (for example the final CNN's validation macro-F1 was 0.87). A first repair with `StratifiedGroupKFold` also failed on Colab because it gives no coverage guarantee. The notebook now holds out about one seventh of each class's tracks explicitly and asserts that all 43 classes are present. That earlier run's figures should not be used.
