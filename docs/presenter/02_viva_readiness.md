# Viva Readiness Guide — GTSRB Traffic-Sign Recognition

Read `01_knowledge_base.md` first. This guide turns that knowledge into answers you can deliver under pressure.

> **Numbers.** The numbers in the answers below come from the final full Colab run (`04_results_and_findings.md`). If you re-run the notebook, refresh them from the capture sheet in `03_presenter_pipeline.md`; never quote numbers from memory.

---

## 1. How to answer (technique)

### 1.1 The four-part answer: Claim, Reason, Evidence, Limit
1. **Claim:** answer the question in one sentence.
2. **Reason:** why (one or two sentences).
3. **Evidence:** point at the specific plot or printed number in the notebook.
4. **Limit:** what the answer does *not* show.

Example. *"Why a grouped split?"* Claim: "Because the training images are not independent." Reason: "Each physical sign appears in about thirty near-identical frames, so a random split puts copies of the same sign in training and validation." Evidence: "Section 5 shows the number of tracks present in both sets: large for a random split, zero for ours." Limit: "It protects validation; it does not make the official test set any different."

### 1.2 If you do not know
Do not bluff. Say: *"I am not certain; what I do know is [adjacent fact]. I would check [where] to confirm."* Examiners usually reward honest boundaries and punish confident errors. Then steer back to what you can show.

### 1.3 If the question is hostile or a trap
Stay calm, restate the question neutrally, answer the literal question, and concede valid criticism briefly. Section 5 of this guide lists the common traps.

### 1.4 If the question is about a number
Read it off the screen. If it is not on screen, say where it is produced. Never quote a result from memory unless you wrote it in your capture sheet.

### 1.5 Phrases to use and avoid
- Use: "on validation data", "on the official test set, evaluated once", "in this run", "with a single seed", "this suggests".
- Avoid: "proves", "guarantees", "state of the art", "production-ready for cars", "safe", "perfect", "always".

### 1.6 Suggested team roles (adapt to your team size)
- **Narrator:** speaks the story, section by section.
- **Driver:** operates Colab, scrolls, reruns cheap cells.
- **Q&A lead:** takes the first question; others add only when invited. Agree beforehand who owns which topics (data, models, evaluation, reliability) so nobody talks over each other.

---

## 2. The numbers and statements you must be able to say without notes

| Item | What to say |
|---|---|
| Task | Classify a cropped sign into 1 of 43 GTSRB classes |
| Data | 39,209 train, 12,630 official test, 43 classes |
| Key data insight | About 1,300 physical signs, around 30 frames each; images are not independent |
| Imbalance | Largest class about 10 times the smallest (read exact values from section 4.1) |
| Resolution | 64×64 cached; MobileNetV2 upsampled to 128×128 inside the model |
| Split | About 1/7 of each class's tracks held out (about 14%); zero tracks in both sets; all 43 classes in validation |
| Models | Custom CNN (about 116 thousand parameters) and MobileNetV2 (about 2.3 million) |
| Training | Adam; 1e-3, and 1e-4 for fine-tuning; early stopping; learning-rate reduction |
| Augmentation | Rotation ±14°, shift ±5%, zoom ±8%, contrast ±10%; no flips |
| Selection rule | Model chosen on validation accuracy; test reported once |
| Your results | Custom CNN 96.96% accuracy, 95.21% macro-F1 (test); MobileNetV2 head-only 75.16%; fine-tuned 91.26%; ensemble 96.88% |
| Reliability | ECE 0.0105; to reach 99% accuracy the system answers 95.5% of images |
| Robustness | Worst: noise (14%) and occlusion (10%) at the severe level; best handled: blur (95%) |
| Export | Reload parity passed; TFLite 0.24 MB, 100% agreement on 500 images |
| Limits | Cropped signs only; one country; single seed; not safety-certified |

---

## 3. Question bank

Answers are model answers; adapt the wording to your own voice. "KB" points to the knowledge-base section.

### Tier 1 — Basics (you must get every one of these right)

**1. What problem are you solving?**
Recognising which of 43 German traffic-sign types is shown in a cropped photo. Input is an image, output is a class and a confidence. (KB A2)

**2. Why is this problem important?**
Driver-assistance systems must read signs correctly under difficult conditions, and some mistakes (confusing speed limits) are more costly than others.

**3. What is a neural network?**
Layers of simple units that learn features from data by adjusting weights to reduce a loss. (KB A9)

**4. What is a CNN and why for images?**
A network using convolutions: small learned filters slid over the image to detect local patterns. Sharing filters across positions needs fewer parameters and respects spatial structure. (KB A10)

**5. What does softmax do?**
Turns the 43 raw output scores into probabilities that sum to one; the largest is the prediction. (A3)

**6. What loss did you use and why?**
Sparse categorical cross-entropy: the standard multi-class loss; it heavily penalises confident wrong answers. "Sparse" means integer labels. (A4)

**7. What is an epoch? A batch?**
An epoch is one pass over the training data; a batch is the group of 128 images used for each weight update. (A6)

**8. What is overfitting and how did you reduce it?**
Memorising the training data instead of generalising. We used augmentation, dropout, batch normalisation, a small model, early stopping and an honest validation split. (A7)

**9. Difference between training, validation and test data?**
Training learns; validation guides decisions; test gives one final unbiased estimate. (A8)

**10. Which dataset and where does it come from?**
GTSRB, a public benchmark from IJCNN 2011, downloaded from a Zenodo mirror and verified by MD5.

**11. What is accuracy and why was it not enough?**
The fraction correct. Because classes are imbalanced we also report macro-F1 so rare classes count equally. (A21)

**12. What is F1?**
The harmonic mean of precision and recall; high only if both are high.

**13. What is a confusion matrix?**
A table of true versus predicted classes; the diagonal is correct, off-diagonal cells are specific mistakes.

**14. What is data augmentation?**
Random realistic changes (small rotation, shift, zoom, contrast) applied during training to teach robustness. (A17)

**15. Why did you not flip the images?**
Flipping changes meaning for directional signs, for example turn-left becomes turn-right while keeping its old label.

**16. What is transfer learning?**
Reusing a model pretrained on a large dataset (ImageNet) and adapting it to our task. (A18)

**17. What is the difference between frozen and fine-tuned?**
Frozen: pretrained layers unchanged, only the new head learns. Fine-tuned: later pretrained layers are also updated at a low learning rate.

**18. What optimiser did you use?**
Adam, learning rate 1e-3, and 1e-4 for fine-tuning. (A5)

**19. What does dropout do?**
Randomly disables units during training to prevent reliance on individual units. It is off at test time.

**20. What does batch normalisation do?**
Keeps layer activations well-scaled so training is faster and more stable.

**21. What is early stopping?**
Stopping training when validation accuracy has not improved for several epochs, and restoring the best weights.

**22. What is a learning rate?**
The size of each weight update. Too large is unstable; too small is slow. We reduce it when progress stalls.

**23. What is Grad-CAM?**
A heat-map of which image regions most influenced a predicted class, computed from gradients at the last convolutional layer. (A25)

**24. What does top-3 accuracy mean?**
Correct if the true class is among the three highest-probability predictions.

**25. How do you know your results can be reproduced?**
Fixed seeds, checksum-verified data, a single config, and exported artifacts. Seeds give repeatability on the same hardware, not bit-identical results everywhere. (A29)

**26. What does "pixel" data look like to the model?**
A 64 by 64 by 3 array of values from 0 to 255, rescaled inside the model.

**27. What is a parameter?**
A learned weight. Our CNN has about 116 thousand; MobileNetV2 with our head about 2.3 million.

**28. What is inference?**
Using a trained model to predict on new images.

**29. What is the GPU for?**
Training is many matrix operations in parallel, which a GPU does far faster than a CPU.

**30. What does the notebook output at the end?**
A zip with all trained models, per-model results, tables, figures, a sample test set and a README on how to reload and test.

### Tier 2 — Project-specific (expect most of these)

**31. What is the most important finding in your data analysis?**
The images are grouped into tracks of about thirty near-identical frames from about 1,300 physical signs. This makes a random split leak, so we split by track. (A20)

**32. Prove the leakage is real.**
Section 5 counts the tracks present in both training and validation. For a random split it is almost every validation track; for the grouped split it is zero, and we assert that.

**33. Does grouping make the test score better or worse?**
It makes the *validation* score lower and honest. The test set is fixed by the benchmark and is not affected by how we split training data.

**34. How did you choose the validation split size?**
About one seventh of each class's tracks, about 14% of the data: a conventional size that leaves most tracks for training while giving enough validation images. We did not tune it.

**35. Is the grouped split stratified by class?**
It is now. Our first run used a plain `GroupShuffleSplit`, which does not balance classes, and it left five classes (0, 19, 21, 23, 29) with no validation images, so validation accuracy and macro-F1 were slightly biased. We noticed the printed check said False. A first fix with `StratifiedGroupKFold` also failed on Colab because it only tries to balance classes and gives no guarantee, so we now hold out about a seventh of each class's tracks explicitly and assert that all 43 classes are present. Test results were never affected because the test set is separate.

**36. Why 64×64?**
Median signs are only a few dozen pixels wide, so higher resolution mostly interpolates; 64 is cheap, fits in memory, and works for the CNN. For MobileNetV2 we upsample to 128 inside the model because the pretrained network works better with larger inputs.

**37. Isn't upsampling 64 to 128 pointless?**
It adds no new information, but it gives the pretrained filters a scale they were trained on and increases the final feature-map size. It also costs more compute. A different resolution study is listed as future work.

**38. Why did you not crop to the annotated region?**
Real detectors give boxes with some margin, so training on the full frame matches deployment better.

**39. Why store images as uint8 in memory instead of streaming?**
The whole dataset is about half a gigabyte, and in-memory arrays remove disk and decoding cost so training is limited by the GPU.

**40. Why put preprocessing inside the model?**
The exported model accepts raw 0–255 pixels, so there is no separate preprocessing code that could drift from training.

**41. Describe your custom CNN.**
Three blocks of convolution, batch norm, max pooling and dropout with 32, 64 and 128 filters, then global average pooling, a 128-unit dense layer, dropout and a 43-way softmax. (Part C, cell 25)

**42. Why global average pooling instead of flatten?**
Far fewer parameters and less over-fitting, and it keeps the feature maps Grad-CAM needs.

**43. Describe your MobileNetV2 model.**
Augmentation, resize to 128, rescale to [-1, 1], the pretrained MobileNetV2 backbone with average pooling, dropout, and a 43-way dense layer.

**44. Why two-stage training?**
A randomly initialised head produces large, noisy gradients that could damage pretrained features. Train the head first, then fine-tune gently.

**45. Why freeze batch-norm layers during fine-tuning?**
Their running statistics come from ImageNet; updating them with a different data distribution and small batches can destabilise the model. Keeping them frozen is the standard recommendation.

**46. Why unfreeze 50 layers?**
It covers roughly the last few blocks of the backbone. It is a chosen value, not tuned; tuning it is future work.

**47. Why is fine-tuning at 1e-4?**
Pretrained weights are already good, so we change them in small steps; a large rate would erase what was learned.

**48. How did you choose between the two models?**
By validation accuracy only. Then we report the chosen model's test numbers once.

**49. Why is MobileNetV2 (head only) worse than the custom CNN?**
Likely a domain gap: ImageNet features are not tuned to tiny sign crops. Our ladder tests this by showing the improvement after fine-tuning. This is our explanation and is consistent with the results; we did not run an experiment that isolates it further.

**50. Does that mean MobileNetV2 is a bad model?**
No. It means a frozen general-purpose backbone is a poor fit for this specific task at this resolution. After fine-tuning it reaches 91.26% test accuracy, up from 75.16%, though it still ends below the custom CNN (96.96%).

**51. What is your headline result?**
On the official test set the selected model, the custom CNN, reaches 96.96% accuracy and 95.21% macro-F1 (single run; a repeat of the pipeline gave 96.52%).

**52. How does that compare with the literature?**
The GTSRB paper reports about 98.8% for humans and about 99.5% for the best committee of CNNs. Our compact single model sits below human level (96.96% against about 98.8%); we do not claim state of the art.

**53. What do the per-class results show?**
Most classes are strong; the weakest are Pedestrians (F1 0.65, recall only 0.50), Dangerous curve to the right (0.82) and Double curve (0.83). The Spearman correlation between class size and F1 is +0.28, a weak positive, so imbalance explains only part of it; visual similarity matters more.

**54. What are the most common confusions?**
Limit 60 read as Limit 80 (36 images), No entry as Priority road (20), Bumpy road as Road work (18), Limit 120 as Limit 80 (16): typically visually similar signs such as different speed limits.

**55. Why do errors stay inside sign families?**
Signs in a family share shape and colour and differ in small details like a digit, which is hard at low resolution. The notebook prints the share of errors inside the same family against the chance level.

**56. What are the confident errors?**
The twelve wrong predictions with the highest confidence in section 9.3. Many are dark, blurred or tiny images.

**57. Explain your progress ladder.**
We started with a majority-class baseline and added one thing per step: linear model, MLP, CNN, deeper CNN with batch norm, augmentation and dropout, then the transfer-learning track and an ensemble. Each step is measured on validation, so we can see what each design decision contributes.

**58. What if a step in the ladder does not improve?**
Then it did not help under that training budget, and we say so. With a single seed, differences of a fraction of a percent are within noise.

**59. Why an ensemble and how does it work?**
We average the two models' probabilities. If they make different mistakes the average is better than either. It costs two models at inference.

**60. Why is the ensemble not the selected model?**
The Grad-CAM, embedding and demo code work on one model; the ensemble is reported and exported separately.

**61. How does Grad-CAM work?**
Weight the last convolutional maps by the average gradient of the class score, add them, keep positives, and enlarge to image size.

**62. Why are MobileNetV2's heat-maps blurrier?**
Its final map is only 4×4 versus 16×16 for the custom CNN.

**63. What does the t-SNE plot tell you?**
How the model groups images internally. Tight, separated clusters mean good features; touching clusters correspond to confusable classes. It is a visualisation, not a metric.

**64. What is calibration?**
Whether confidence matches actual accuracy. We measure it with ECE and a reliability diagram; the ECE of the custom CNN is 0.0105, which is good; the fine-tuned MobileNetV2 is more over-confident at 0.054.

**65. What is selective prediction and why include it?**
The system answers only when confident and abstains otherwise. It trades coverage for accuracy, which is a practical safety mechanism. In our run, to reach 99% accuracy it answers 95.5% of images (threshold 0.866).

**66. Where did you set the abstention threshold?**
For illustration we read it off the test results; a real system should set it on validation data. We say this openly.

**67. What did the robustness test show?**
Accuracy versus severity for six corruptions. The worst were Gaussian noise (14% accuracy at the severe level, 34% at sigma 25) and occlusion (10% when 35% of the area is hidden); the best handled was blur (95%), then darkening and glare (82% and 85%). These are synthetic approximations of real conditions.

**68. Why export so many files?**
So the team can re-test, re-plot and compare models later without retraining.

**69. What does the reload parity check do?**
Reloads the saved model and confirms predictions match the in-memory model within 1e-4, catching serialisation problems.

**70. Why TFLite and what did float16 do?**
To show deployability on small devices. Float16 weights roughly halve the size with minimal accuracy change; we verified agreement with the Keras model on 500 images.

### Tier 3 — Probing and technical depth

**71. Why does training accuracy sometimes start below validation accuracy?**
Augmentation and dropout are active during training but not validation, so training images are harder and noisier. It is expected, not a bug.

**72. Why sparse categorical cross-entropy rather than MSE?**
Cross-entropy matches probabilistic classification and gives strong gradients for wrong confident predictions; MSE on probabilities learns slowly.

**73. Why softmax rather than sigmoid outputs?**
Exactly one class is correct, so mutually exclusive probabilities are appropriate; sigmoids are for multi-label tasks.

**74. What is the vanishing gradient and does it matter here?**
Gradients shrinking in deep networks. ReLU, batch norm and residual connections mitigate it; our CNN is shallow and MobileNetV2 uses residual blocks.

**75. How many parameters in conv2 and why?**
3×3×32×64 + 64 = 18,496: each of 64 filters has 3×3×32 weights plus a bias.

**76. What does `padding="same"` do?**
Pads the borders so the output has the same width and height as the input.

**77. What does max pooling lose?**
Exact position within each 2×2 window; fine spatial detail.

**78. Why average pooling at the end?**
It summarises each feature map as one number, giving a compact vector and fewer parameters.

**79. How does batch norm behave at test time?**
It uses stored running mean and variance instead of the current batch's statistics.

**80. What is the difference between macro and weighted F1?**
Macro averages classes equally; weighted averages by class size. Macro exposes rare-class weakness.

**81. Why is the Spearman correlation used?**
It measures rank association without assuming a straight-line relationship; class size versus F1 need not be linear.

**82. What exactly is ECE?**
Group predictions by confidence; in each bin take |accuracy − average confidence|; average these gaps weighted by bin size.

**83. How would you fix poor calibration?**
Temperature scaling: divide logits by a single value fitted on validation data to minimise cross-entropy.

**84. Why is the model selected on validation accuracy and not loss?**
Accuracy is the quantity we care about; loss could pick an over-confident model. Both are available and would usually agree.

**85. Is your early stopping monitor a source of leakage?**
Early stopping uses validation, not test, so test remains clean. It does mean validation is slightly optimistic as an estimate, which is why we report on test.

**86. Test set reuse: did you use it for decisions anywhere?**
Not for training, model selection or hyper-parameters. We do display test-based figures for the ensemble, the abstention thresholds and robustness; those are descriptive, not used to choose the model. We state that openly.

**87. Is the official test set truly independent of the training signs?**
By the benchmark's design it is; we relied on that and did not verify it independently.

**88. Why are results single-run and what would you change?**
Time. We would repeat with several seeds, report mean and standard deviation, and test significance before claiming one model is better.

**89. Could the ensemble be overfitting the validation set?**
It has no trainable parameters (simple average), so there is little to overfit; but its apparent gain should still be confirmed across seeds.

**90. Is the split deterministic?**
Yes with a fixed `random_state` (42).

**91. How is the tf/Keras augmentation random across runs?**
Layers have seeds, and global seeds are set; GPU non-determinism can still cause small differences.

**92. Why does the model contain augmentation layers if they are off at inference?**
Keras makes them pass-through when `training=False`, so a single model handles both; reload parity confirms predictions match.

**93. How do you know the optimiser did not just get lucky?**
We do not claim that; this is one run per configuration. Multi-seed testing is listed as future work.

**94. What would change if you used class weights?**
Rare classes would contribute more to the loss, possibly raising macro-F1 at a small accuracy cost. We chose to check first whether imbalance was the cause (the Spearman test).

**95. What is a depthwise separable convolution?**
A convolution split into a per-channel spatial filter and a 1×1 convolution that mixes channels, giving similar accuracy with far fewer operations.

**96. Why MobileNetV2 and not ResNet or EfficientNet?**
It is small and fast, suitable for deployment; the aim was a lightweight comparison. Larger backbones are future work.

**97. What does `include_top=False` mean?**
Drop the original 1,000-class ImageNet output layer so we can attach our own 43-class head.

**98. What is `pooling="avg"` in MobileNetV2?**
The backbone averages its final feature map to a 1,280-number vector.

**99. How does the progress-ladder baseline "Always predict the most common class" score so low?**
It is right only for images of that class, about 6% of validation images (5.85% in our run).

**100. What does float16 quantisation change numerically?**
Weights are stored with about three decimal digits of precision instead of seven; the effect on predictions is tiny, which the parity check confirms.

### Tier 4 — Hostile and trap questions

**101. "97% is not good enough for a car."**
Agreed; this is not a deployable driving system. It is a classification study. Real use needs a detector, far more data under real conditions, calibration, redundancy and safety validation. Our abstention and robustness analyses are steps toward knowing when the model should not be trusted.

**102. "You tuned on the test set."**
No. Decisions used validation only; we report test once for the chosen model. The ensemble line, abstention thresholds and robustness figures are descriptive, and we point that out.

**103. "Your validation accuracy is higher than test; that is suspicious."**
Not necessarily: the validation tracks may be easier, and the test set has its own distribution. A large gap would deserve investigation; ours is small: 97.61% validation against 96.96% test.

**104. "Isn't it cheating to use ImageNet weights?"**
Pretraining is standard and legitimate; ImageNet is a general-purpose dataset, not GTSRB. We disclose it and compare against a model trained from scratch.

**105. "Your MobileNet result is bad; you implemented it wrong."**
A frozen backbone at low resolution is expected to underperform a purpose-built CNN here; fine-tuning improves it from 75.16% to 91.26% test accuracy. We built and tested two stages precisely to show this.

**106. "Why should we trust Grad-CAM?"**
Treat it as supporting evidence, not proof. It shows influence and agrees qualitatively with sensible behaviour; it can also look plausible for wrong reasons.

**107. "You only have one seed, so everything is anecdotal."**
Fair. Large effects such as baseline to CNN are far beyond noise; small differences are not claimed as significant. Multi-seed runs are the first improvement.

**108. "Why did you build a notebook instead of using your repository code?"**
The notebook is self-contained for Colab so it runs anywhere without copying files; the repository contains the reusable package and command-line scripts for the same pipeline.

**109. "What is original here?"**
The careful evaluation design (grouped split with a proof), the stepwise ladder, reliability and robustness analyses, and a reusable validated export. The models themselves are standard.

**110. "Does your model work on real photos from India or other countries?"**
Untested. GTSRB is German; other countries use different sign designs. Domain shift is a stated limitation.

**111. "What if the sign is not in the 43 classes?"**
The model must pick one of the 43 and may be confidently wrong. Out-of-distribution detection is future work; abstention helps but does not solve it.

**112. "Can an adversary fool it?"**
Almost certainly; we did not test adversarial attacks. Physical adversarial stickers on signs are a known risk in the literature.

**113. "What is the real-time speed?"**
We measured per-image latency at batch size 1 in the comparison table (18.0 ms for the CNN and 153.0 ms for MobileNetV2 on Colab's GPU, with the Python overhead included). That is not an embedded-hardware benchmark.

**114. "Why not use YOLO directly?"**
YOLO is a detector for full scenes and needs bounding-box training data; GTSRB provides cropped classification data. The repository has an optional detection extension that needs a separately supplied detector.

**115. "Is the abstention threshold legitimate?"**
As a demonstration of the trade-off, yes; for deployment it must be chosen on separate validation data. We say this in the notebook discussion.

---

## 4. Per-topic mini-scripts (30 seconds each, for team members who own a topic)

**Data owner.** "GTSRB has 39,209 training and 12,630 test images in 43 classes. We verified the download by checksum, found strong imbalance and mostly small, variously lit images, and discovered that images come in tracks of about thirty near-duplicates. That drives our split."

**Modelling owner.** "A small CNN from scratch, and MobileNetV2 trained in two stages: a new head on frozen pretrained features, then fine-tuning the top layers at a lower learning rate. Normalisation and augmentation live inside the model so it accepts raw pixels."

**Evaluation owner.** "We pick the model by validation accuracy, report the official test set once, and use macro-F1 as well as accuracy. Then per-class results, confusions, error gallery, and a ladder showing what each design step contributed."

**Reliability owner.** "Grad-CAM shows what the model looks at; calibration and abstention tell us when to trust it; robustness tests show which real-world conditions hurt. Finally every model is exported and reloaded to prove it works."

---

## 5. Common traps and the right reflex

| Trap | Right reflex |
|---|---|
| "Accuracy is X, so it is great" | Mention macro-F1, per-class results, and what the errors are |
| Saying the test set guided decisions | State clearly: validation for decisions, test once |
| Calling frames independent | They are near-duplicates; hence grouped split |
| Saying "state of the art" | Compare to the paper figures honestly and say you do not claim it |
| Over-reading Grad-CAM or t-SNE | Say "suggests", point to limits |
| Quoting a number not on screen | Say where it is produced and read it live |
| Claiming safety | Not certified; research pipeline |
| Hiding a flat ladder step | Show it and explain the training-budget and noise caveat |
| Confusing "epochs" setting with epochs actually run | Early stopping ends sooner; read the printed epoch count |
| Defending a mistake | Concede briefly, state the fix |

---

## 6. Self-assessment: can you do each of these in 30 seconds?

Tick when you can explain it aloud without notes. Everyone on the team should reach at least 80%; each person should own a topic at 100%.

**Concepts**
- [ ] What a pixel array is
- [ ] Softmax and cross-entropy
- [ ] Gradient descent, Adam, learning rate
- [ ] Epoch, batch, early stopping
- [ ] Overfitting versus underfitting, from a curve
- [ ] Train, validation, test roles
- [ ] Convolution, filter, feature map
- [ ] Pooling and global average pooling
- [ ] Batch normalisation and dropout
- [ ] Augmentation and why no flips
- [ ] Transfer learning, frozen versus fine-tuned
- [ ] Precision, recall, F1, macro versus weighted
- [ ] Confusion matrix reading
- [ ] Calibration and ECE
- [ ] Grad-CAM
- [ ] t-SNE caution
- [ ] Ensemble idea

**Project**
- [ ] The track structure and the leakage proof
- [ ] Why 64×64, uint8 in memory, preprocessing in the model
- [ ] CNN architecture from memory
- [ ] MobileNetV2 two-stage recipe and why BN is frozen
- [ ] How the model was selected
- [ ] The ladder, step by step
- [ ] The headline results with the exact numbers
- [ ] Three weakest classes and why
- [ ] Abstention and its honest caveat
- [ ] Robustness: worst and best corruption
- [ ] Export bundle and parity checks
- [ ] Five limitations

**Delivery**
- [ ] 30-second, 2-minute and 5-minute project summaries
- [ ] The Claim-Reason-Evidence-Limit pattern
- [ ] How to answer "I do not know"

---

## 7. Mock viva drills

### 7.1 Rapid fire (one minute each, partner asks, you answer in one sentence)
Ask in random order from: softmax, cross-entropy, Adam, overfitting, dropout, batch norm, pooling, transfer learning, macro-F1, ECE, Grad-CAM, grouped split, MD5, ensemble, TFLite, why no flip, why 64×64, why validation-based selection.

### 7.2 "Explain it to a first-year student" (3 minutes)
Pick a concept and explain it with an analogy and no jargon. Partner stops you at any undefined word.

### 7.3 "Defend your choice" (5 minutes)
Partner picks a row from the decision table in KB Part D and argues for the alternative. Defend with Reason, Evidence, Limit, and concede valid points.

### 7.4 "Trap round"
Partner asks Tier 4 questions in a hostile tone. Practise staying calm and conceding briefly.

### 7.5 "Read the plot"
Show a figure from the notebook with its title covered. The presenter must say what it shows, what is good, and what is a red flag (use KB Part E).

### 7.6 Full mock (20 minutes presentation + 15 minutes questions)
Run the whole notebook as in the presenter pipeline, then take questions from the bank at random. Record yourselves; check for jargon, overclaiming, and time.

---

## 8. Day-before and hour-before checklists

**Day before**
- [ ] Full Colab run completed with GPU; results zip downloaded
- [ ] Capture sheet in `03_presenter_pipeline.md` filled with real numbers
- [ ] Numbers in this guide still match the run you present from
- [ ] Notebook saved with outputs visible (File > Save a copy, or download the executed `.ipynb`)
- [ ] Backup: the executed `.ipynb`, the results zip and the figures folder on a USB stick and cloud drive
- [ ] Two mock vivas done
- [ ] Roles agreed

**Hour before**
- [ ] Colab session open and connected; do not rely on a fresh full run during the presentation
- [ ] Browser zoom set so plots are readable; notifications off
- [ ] Backup screenshots of every key figure in a slide file
- [ ] This guide and the knowledge base open on a second device
- [ ] Water, charger, internet check

**If Colab fails live.** Open the executed notebook copy or the exported figures and carry on from screenshots; say plainly what happened and continue the story. See `../demo_notebook_presenter_notes.md` section 5 for symptoms and fixes.

---

## 9. What to do after a hard question

- If you answered well, stop talking; silence is fine.
- If you were unsure, offer to follow up: "I will confirm that and send the answer." Then actually do it.
- Write the question in a log; add it to the bank above for the next rehearsal.
