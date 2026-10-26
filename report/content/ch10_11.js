const L = require("../lib");
const { chapter, h2, h3, h3u, p, bullets, numbered, tbl } = L;
const { AlignmentType } = L.d;
const CENTER = AlignmentType.CENTER, LEFT = AlignmentType.LEFT;

module.exports = function ch10_11() {
  const o = [];
  // =============================================== CHAPTER 10 ===============================================
  o.push(chapter("Summary and Conclusion"));
  o.push(h2("10.1 Summary of the Work"));
  o.push(p("This project set out to build a traffic-sign classifier for the 43 classes of the German Traffic Sign Recognition Benchmark and, more importantly, to evaluate it in a way that can be trusted. The work began with a verified acquisition of the dataset and an exploratory analysis whose central finding was structural: the 39,209 training images come from only 1,307 physical signs, each photographed about thirty times. A random validation split puts almost all of these signs on both sides of the split. The validation set was therefore built from whole signs, selected per class so that every class is represented, and the code asserts both properties."));
  o.push(p("Two model families were compared: a compact convolutional network of 116,203 parameters trained from scratch, and MobileNetV2 with ImageNet weights trained in two stages. A ladder of nine steps, from a majority-class baseline to the final network, measured the contribution of each design decision on validation data. The models were then evaluated once on the official test set. The selected custom CNN reached 96.96 % accuracy and 95.21 % macro-F1, ahead of fine-tuned MobileNetV2 (91.26 % and 86.59 %) and an ensemble of both (96.88 % and 95.07 %)."));
  o.push(p("Beyond accuracy, the study examined why and when the model fails. Per-class and confusion analyses showed that errors are mostly between look-alike signs of the same family, and that the weakest classes are Pedestrians, Dangerous curve to the right and Double curve. Grad-CAM and t-SNE showed what the network attends to and how it organises the classes. The network is well calibrated (expected calibration error 0.0105) and can trade coverage for accuracy: answering only above a confidence of 0.866 gives 99 % accuracy on 95.5 % of the images. Corruption tests showed tolerance to blur, darkening and glare and fragility to noise, occlusion and large rotation. A numerical comparison of two implementations showed that Grad-CAM based on the class probability is unreliable for nearly certain predictions."));
  o.push(p("Finally, the trained network was re-implemented in dependency-free JavaScript, verified against Keras, and published as a static website that classifies images locally, explains them and lets the user stress them. The same pipeline exports every trained model with its results for reuse."));
  o.push(h2("10.2 Conclusions"));
  o.push(...tbl("rqa", "Answers to the research questions", ["Question", "Answer"],
    [["RQ1. How much does a random split overstate performance, and how is the leakage removed?", "A random split shares 1,291 of 1,307 signs between training and validation, so every validation sign has been seen. Holding out whole signs, selected per class, removes the leakage and keeps all 43 classes in validation."],
     ["RQ2. Small CNN or MobileNetV2, and which decisions matter?", "The small CNN is better (96.96 % against 91.26 % test accuracy) and faster. Convolutions, batch normalisation with global pooling, augmentation and dropout each helped; size did not. Fine-tuning gained 16 points over a frozen backbone but over-fitted."],
     ["RQ3. Which classes are hardest and why?", "Pedestrians (F1 0.65), Dangerous curve to the right (0.82) and Double curve (0.83). Errors are between look-alike signs (75 % within a family); class size explains only part of it (Spearman +0.28)."],
     ["RQ4. Does the network attend to the sign, and is Grad-CAM reliable?", "For correct predictions the heat lies on the symbol or rim. Probability-based Grad-CAM is numerically unreliable for near-certain predictions; the class logit is reliable."],
     ["RQ5. Is confidence calibrated, and what does abstention buy?", "ECE is 0.0105. A threshold of 0.866 gives 99 % accuracy while answering 95.5 % of images."],
     ["RQ6. Which corruptions hurt most?", "Noise, occlusion and large rotation. Blur, darkening and glare are tolerated."]],
    [3.3, 4.4], { size: 19, align: [LEFT, LEFT] }));
  o.push(p("The overall conclusion is that, for a task like this, a carefully evaluated small network is a better engineering choice than a larger pretrained one, and that the quality of the evaluation, not only of the model, determines whether a reported accuracy can be believed. The project contributes a reproducible pipeline that makes this evaluation explicit and a verified, serverless demonstration of the resulting model."));
  o.push(h2("10.3 Learning Outcomes"));
  o.push(p("Through this project the team applied machine-learning, software-engineering and communication skills in a single piece of work: designing and running controlled experiments, discovering and correcting a flaw in its own evaluation, implementing a neural network and its explanation method from first principles, verifying numerical equivalence across two implementations, building and deploying an accessible web application, and documenting limitations honestly. These correspond to the program outcomes on engineering knowledge, problem analysis, modern tool usage, ethics, teamwork and communication."));

  // =============================================== CHAPTER 11 ===============================================
  o.push(chapter("Limitations of the Project and Future Work"));
  o.push(h2("11.1 Limitations"));
  o.push(...bullets([
    "<b>Cropped signs only.</b> The classifier assumes a tight crop of one sign. Locating signs in a scene needs a detector, which GTSRB cannot train; the repository contains optional detection scripts that require a separately supplied checkpoint and were not evaluated.",
    "<b>One country, one dataset.</b> GTSRB contains German signs. Signs from other countries, newer designs or other weather and camera conditions are untested, and performance measured on one dataset may not transfer to another {@torralba2011}.",
    "<b>Single seed.</b> Each configuration was trained once per run. Two runs of the pipeline differed by up to about one point for some models, so small differences between models are not statistically supported.",
    "<b>Resolution.</b> Images are reduced to 64 x 64 pixels. Details such as the digits of speed limits may be lost, and the most frequent confusion (60 versus 80 km/h) is of this kind.",
    "<b>Training did not include noise or occlusion.</b> The corruption study measures the cost of this omission but the model was not retrained to remove it.",
    "<b>Descriptive use of the test set.</b> The abstention thresholds and the robustness study use test-set behaviour to illustrate trade-offs; they were not used to choose a model, but a deployed threshold should be set on separate validation data.",
    "<b>Explanation is qualitative.</b> Grad-CAM shows influence, not causation, and MobileNetV2's maps are coarse.",
    "<b>Timing and hardware.</b> Latency was measured on a GPU runtime and a laptop with Python or browser overhead, not on automotive hardware.",
    "<b>Testing.</b> There is no automated unit-test suite for the Python package, and the camera input of the website was not exercised.",
    "<b>Not safety-certified.</b> The system is an educational research project. It is not designed, validated or certified for use in vehicles {@iso26262,iso21448}."]));
  o.push(h2("11.2 Future Work"));
  o.push(p("{tab:future} lists improvements ordered by how directly the results of this project motivate them."));
  o.push(...tbl("future", "Proposed future work and the evidence that motivates it", ["Priority", "Improvement", "Motivation from the results"],
    [["1", "Add Gaussian-noise and random-erasing augmentation and widen the rotation range", "Accuracy 34 % at noise sigma 25, 53 % with a fifth of the image covered, 52 % at 35 degrees"], ["2", "Brightness augmentation and contrast normalisation (for example CLAHE)", "Most confident errors are very dark speed-limit signs; strong darkening lowers accuracy to 82 %"],
     ["3", "Higher input resolution (for example 96 pixels) for the CNN", "Speed limit 60 versus 80 (36 images), 120 versus 80 (16), 70 versus 30 (10)"], ["4", "Class weights or targeted augmentation for the weakest classes", "Pedestrians has recall 0.50; Dangerous curve to the right has precision 0.69"],
     ["5", "Stronger regularisation for fine-tuned MobileNetV2 (more augmentation, fewer unfrozen layers, label smoothing)", "Training accuracy 99.8 % against 90.6 % validation"], ["6", "Repeat every experiment with several seeds and report intervals", "Run-to-run spread of up to about one point; ensemble ordering flipped"],
     ["7", "Temperature scaling and a validation-chosen abstention threshold", "The threshold was read off test data; ensemble and MobileNetV2 are less calibrated"], ["8", "Knowledge distillation from the ensemble into the small CNN {@hinton2015}", "Ensemble has the best top-3 accuracy but costs two models"],
     ["9", "A detection front end for full scenes and video {@redmon2016}", "Classifier handles cropped signs only"], ["10", "Out-of-distribution detection and evaluation on signs from other countries", "Over-confidence on non-sign inputs; GTSRB covers one country"],
     ["11", "Automated tests: a pytest suite for the package and browser tests for the website", "Testing limitations (Section 8.6)"]],
    [0.8, 3.6, 3.3], { size: 18, align: [CENTER, LEFT, LEFT] }));
  o.push(h2("11.3 Ethical and Safety Considerations"));
  o.push(p("Traffic-sign recognition is a safety-relevant technology, and over-stating the reliability of a classifier can cause harm. For this reason the website states that it handles cropped signs only, is not a driving system and can be over-confident on inputs unlike its training data; the report separates decisions made on validation data from descriptive analyses that use the test set; and the limitations above are stated explicitly. Images supplied to the website are processed locally and are never uploaded. The sample images shown come from a public research benchmark and are used for illustration with attribution; their redistribution should follow the dataset's terms."));
  return o.flat();
};
