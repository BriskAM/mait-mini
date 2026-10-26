const L = require("../lib");
const { chapter, h2, h3, h3u, p, bullets, numbered, fig, tbl, eq, code, d } = L;
const { AlignmentType } = d;
const RIGHT = AlignmentType.RIGHT, CENTER = AlignmentType.CENTER, LEFT = AlignmentType.LEFT;

module.exports = function ch4_6() {
  const o = [];
  // =============================================== CHAPTER 4 ===============================================
  o.push(chapter("Analysis"));
  o.push(p("This chapter analyses what the system must do and whether it can be built. It follows the structure required for development-based projects: software requirements, feasibility, tools and platform, and use case and data flow diagrams."));
  o.push(h2("4.1 Software Requirement Specifications"));
  o.push(p("<b>Purpose and scope.</b> The system consists of two cooperating parts. The first is an <i>experiment pipeline</i>, run as a notebook on a GPU runtime, that acquires and verifies GTSRB, trains and evaluates models, analyses their behaviour and exports models and results. The second is a <i>static website</i> that runs the selected model in a visitor's browser and presents the results. The intended users are the researchers (the project team) who run and extend the pipeline, and examiners or visitors who explore the demonstration."));
  o.push(p("<b>Assumptions and constraints.</b> The pipeline assumes an internet connection to download the dataset and pretrained weights and a GPU runtime for full-length training. The website assumes a modern browser with JavaScript and canvas support; camera input additionally requires a secure (HTTPS) context and user permission. The model accepts one cropped sign at a time. No server component is used."));
  o.push(h3("4.1.1 Functional Requirements of the Project"));
  o.push(...tbl("fr", "Functional requirements", ["ID", "Requirement", "Part"],
    [["FR1", "Download the GTSRB archive and verify it against the published MD5 checksum; stop with an error on mismatch.", "Pipeline"], ["FR2", "Assert the expected counts (39,209 training images, 12,630 test images, 43 classes) and that every referenced file exists.", "Pipeline"],
     ["FR3", "Profile the data: class distribution, image size, brightness, and the track structure of the training images.", "Pipeline"], ["FR4", "Create a validation split that shares no physical sign with the fitting split and contains every class.", "Pipeline"],
     ["FR5", "Cache all images at 64 x 64 resolution and apply training-time augmentation inside the model.", "Pipeline"], ["FR6", "Train a custom CNN and a two-stage MobileNetV2 with early stopping and learning-rate reduction.", "Pipeline"],
     ["FR7", "Run a ladder of single-factor experiments from a trivial baseline to the final model.", "Pipeline"], ["FR8", "Evaluate each model on the official test set and report accuracy, macro-F1, top-3 accuracy, calibration error, size and latency.", "Pipeline"],
     ["FR9", "Report per-class metrics, the confusion matrix, the most frequent confusions and the most confident errors.", "Pipeline"], ["FR10", "Generate Grad-CAM heat-maps and an embedding projection.", "Pipeline"],
     ["FR11", "Measure calibration, the accuracy-versus-coverage trade-off and robustness to six corruption types at three severities.", "Pipeline"], ["FR12", "Export every trained model, per-model predictions and metrics, figures and a sample of test images; verify that reloaded models reproduce predictions.", "Pipeline"],
     ["FR13", "Classify a selected test image, an uploaded image or a camera frame in the browser and show the top five classes with confidences.", "Website"], ["FR14", "Show a Grad-CAM heat-map for the current prediction.", "Website"],
     ["FR15", "Let the user apply rotation, brightness, blur, noise and occlusion to the current image and sweep each across its range.", "Website"], ["FR16", "Let the user set a confidence threshold below which the system defers, and show the resulting coverage and accuracy.", "Website"],
     ["FR17", "Present the experiment results as interactive charts with table alternatives.", "Website"], ["FR18", "Provide a check that compares the browser model with the values recorded from the Keras model.", "Website"]],
    [0.7, 5.5, 1.0], { align: [CENTER, LEFT, CENTER] }));
  o.push(h3("4.1.2 Non-functional Requirements of the Project"));
  o.push(...tbl("nfr", "Non-functional requirements", ["Category", "Requirement"],
    [["Reproducibility", "All random generators are seeded; the dataset is verified by checksum; one configuration object controls budgets; results are exported with the code that produced them."], ["Correctness of evaluation", "No decision may use the test set; validation may not share physical signs with training; assertions enforce this."],
     ["Performance", "A prediction on the website should complete in well under a second on an ordinary laptop; full training should complete in under one hour on a free GPU runtime."], ["Portability", "The pipeline runs unchanged on Google Colab and, with a local dataset path, on a workstation; the website runs on any modern browser."],
     ["Privacy", "Images supplied by a website visitor are processed locally and never transmitted."], ["Usability and accessibility", "The website works with keyboard navigation, provides table views for charts, supports light and dark themes and a narrow (phone) layout."],
     ["Maintainability", "The notebook and website data are generated by scripts; the browser network is covered by a numerical verification script."], ["Honesty and safety messaging", "The system must state that it handles cropped signs only, is not safety-certified and may be over-confident outside its training distribution."]],
    [2.0, 5.7], { align: [LEFT, LEFT] }));
  o.push(h2("4.2 Feasibility Study of the Project"));
  o.push(p("{tab:feas} summarises the feasibility analysis along the usual dimensions."));
  o.push(...tbl("feas", "Feasibility analysis", ["Dimension", "Assessment", "Conclusion"],
    [["Technical", "GTSRB is small enough (about 320 MB) to fit in memory at 64 x 64 resolution (0.64 GB). A 116,203-parameter CNN trains in about six minutes on a free GPU. A network of this size can be evaluated in plain JavaScript in tens of milliseconds.", "Feasible"],
     ["Economic", "All software is open source. Training used a free Colab runtime; hosting uses GitHub Pages. No licences or hardware purchases were needed.", "Feasible at near-zero cost"], ["Operational", "The pipeline runs by selecting a mode and running all notebook cells; the website needs no installation and no server administration.", "Feasible"],
     ["Schedule", "The work packages (Chapter 6) were small and largely sequential; a full training run takes under 25 minutes, which allowed several complete iterations, including a correction of the validation split.", "Feasible within the semester"],
     ["Legal and ethical", "GTSRB is a public research benchmark and is cited as required. The website processes images locally. Safety limitations are stated prominently. Redistribution of the cropped sample images should respect the dataset's terms.", "Feasible with the stated precautions"]],
    [1.3, 4.9, 1.5], { size: 19 }));
  o.push(h2("4.3 Tools, Technologies and Platform Used"));
  o.push(...tbl("tools", "Tools, technologies and platform", ["Tool / platform", "Version or detail", "Purpose"],
    [["Python", "3.x (Colab runtime)", "Pipeline language"], ["TensorFlow / Keras {@abadi2016,chollet2015}", "2.21.0 / 3.13.2 (Colab run); pinned 2.16.2 in the repository", "Model definition, training, export"], ["NumPy, pandas {@harris2020}", "NumPy 2.x; pandas 2.x", "Arrays, tables"],
     ["scikit-learn {@pedregosa2011}", "1.7.x", "Metrics, t-SNE, splitting utilities"], ["Matplotlib {@hunter2007}", "3.10.x", "Figures"], ["Pillow, OpenCV", "Pillow 11.x; OpenCV headless 4.12", "Image loading, corruptions"],
     ["Google Colaboratory {@colab}", "GPU runtime", "Training and evaluation"], ["TensorFlow Lite", "Bundled with TensorFlow", "Mobile-size model conversion"], ["HTML, CSS, JavaScript (ES modules)", "No third-party libraries", "Website, in-browser network, charts"],
     ["Node.js", "Recent LTS", "Numerical verification script"], ["Git and GitHub", "-", "Version control, collaboration"], ["GitHub Pages and Actions", "-", "Static hosting and deployment"], ["GTSRB archive {@zenodo}", "Zenodo record 13741936", "Dataset"]],
    [2.6, 2.9, 2.2], { size: 19 }));
  o.push(h2("4.4 Use Case Diagrams and Data Flow Diagrams"));
  o.push(p("{fig:usecase} shows the use cases of the system. The <i>researcher</i> runs the experiment pipeline; the <i>panel member or website visitor</i> uses the website. {tab:uc} describes the principal use cases."));
  o.push(...fig("usecase", "d_usecase.png", "Use case diagram of the traffic-sign recognition system", { width: 5.3 }));
  o.push(...tbl("uc", "Principal use cases", ["Use case", "Actor", "Main flow", "Outcome"],
    [["Train and compare models", "Researcher", "Select a mode; run the notebook; models are trained, early stopping selects the best epoch, results are scored on validation", "Trained models and training histories"], ["Evaluate on the test set", "Researcher", "Score each model once on the official test split; compute metrics and analyses", "Metrics, confusion matrix, figures"],
     ["Export results", "Researcher", "Save models, predictions, tables and figures; reload each model and compare predictions", "Verified results bundle"], ["Classify a sign image", "Visitor", "Pick a sample, upload a photo or use the camera; the model returns the top five classes", "Prediction, confidence, answer-or-defer decision"],
     ["Stress-test an image", "Visitor", "Move the corruption sliders or press Sweep; the prediction updates", "Confidence versus severity"], ["Verify against Keras", "Visitor", "Press the verify button; the page compares its predictions with recorded Keras values", "Agreement report"]],
    [1.6, 1.0, 3.4, 1.8], { size: 18 }));
  o.push(p("The context-level data flow diagram ({fig:dfd0}) shows the system as one process with four external entities: the GTSRB archive, the researcher, the website visitor and the hosting platform. {fig:dfd1} decomposes it into five processes (prepare data, train models, evaluate and analyse, export artefacts, browser inference) and five data stores."));
  o.push(...fig("dfd0", "d_dfd0.png", "Data flow diagram, level 0 (context diagram)", { width: 5.2 }));
  o.push(...fig("dfd1", "d_dfd1.png", "Data flow diagram, level 1", { width: 4.9, maxHeight: 7.6 }));

  // =============================================== CHAPTER 5 ===============================================
  o.push(chapter("Research Methodology"));
  o.push(h2("5.1 Methodology Used"));
  o.push(p("The study is an experimental one. {fig:workflow} gives the end-to-end workflow, which is executed in a single notebook so that every number in this report can be regenerated."));
  o.push(...fig("workflow", "d_workflow.png", "End-to-end workflow of the study", { width: 5.6 }));
  o.push(h3("5.1.1 Principles"));
  o.push(...bullets([
    "<b>Protocol before results.</b> The split, metrics and selection rule were fixed before models were compared (Section 2.4).",
    "<b>One change at a time.</b> The progress ladder changes a single factor per step, so the effect of each decision is attributable.",
    "<b>Validation for decisions, test once.</b> Early stopping and model selection use validation data. The test set is scored once per model.",
    "<b>Assertions over trust.</b> Data counts, file existence, split properties and reload parity are checked by code that stops the run if violated.",
    "<b>Seeds and checksums.</b> A fixed seed (42) and a checksum-verified archive make runs repeatable on the same hardware."]));
  o.push(h3("5.1.2 Evaluation Metrics"));
  o.push(p("Let <i>TP<sub>c</sub></i>, <i>FP<sub>c</sub></i> and <i>FN<sub>c</sub></i> be the true positives, false positives and false negatives of class <i>c</i> among <i>N</i> evaluated images and <i>C</i> = 43 classes. The metrics used are:"));
  o.push(eq("acc", "Accuracy = (1 / N) Σ_i [ ŷ_i = y_i ]"));
  o.push(eq("pr", "P_c = TP_c / (TP_c + FP_c),    R_c = TP_c / (TP_c + FN_c),    F1_c = 2 P_c R_c / (P_c + R_c)"));
  o.push(eq("macro", "Macro-F1 = (1 / C) Σ_c F1_c"));
  o.push(p("Macro-F1 gives every class the same weight, so rare classes are not hidden behind common ones. Top-3 accuracy counts a prediction as correct if the true class is among the three highest-probability classes. The loss minimised in training is the cross-entropy between the one-hot label and the predicted distribution:"));
  o.push(eq("ce", "L = − log p(y = y_true | x)"));
  o.push(p("<b>Calibration</b> is summarised by the expected calibration error {@guo2017}. Predictions are grouped into <i>M</i> = 15 equal-width confidence bins <i>B<sub>m</sub></i>; with <i>acc(B<sub>m</sub>)</i> the accuracy and <i>conf(B<sub>m</sub>)</i> the mean confidence in a bin:"));
  o.push(eq("ece", "ECE = Σ_m (|B_m| / N) · | acc(B_m) − conf(B_m) |"));
  o.push(p("<b>Selective prediction.</b> For a threshold <i>t</i> the system answers when the confidence is at least <i>t</i> and defers otherwise. Coverage is the fraction of images answered, and selective accuracy is the accuracy on the answered images. Sweeping <i>t</i> traces the trade-off between them {@geifman2017}."));
  o.push(p("<b>Grad-CAM.</b> For an image and a class <i>c</i> with score <i>y<sub>c</sub></i>, and the activations <i>A<sup>k</sup></i> of the last convolutional layer, the channel weights and the heat-map are {@selvaraju2017}:"));
  o.push(eq("gradcam", "α_k = (1 / Z) Σ_{i,j} ∂y_c / ∂A^k_{ij},    L_Grad-CAM = ReLU( Σ_k α_k A^k )"));
  o.push(h3("5.1.3 Experimental Design"));
  o.push(p("Three families of experiments are run. (i) A <b>progress ladder</b> of nine steps measured on validation data: a majority-class baseline, a linear classifier, a multi-layer perceptron, a basic CNN, a CNN with batch normalisation and global pooling, the final CNN with augmentation and dropout, MobileNetV2 with a frozen backbone, MobileNetV2 fine-tuned, and an ensemble of the two best models. (ii) A <b>final evaluation</b> of four models on the official test set. (iii) <b>Reliability studies</b> of the selected model: per-class analysis, Grad-CAM, calibration, selective prediction and corruption robustness. The training budgets of the three run modes of the notebook are shown in {tab:modes}; all reported results use the <i>full</i> mode."));
  o.push(...tbl("modes", "Training budgets of the notebook run modes (epochs are upper limits)", ["Mode", "Training images", "Custom CNN", "Ladder (linear, MLP)", "MobileNetV2 head", "MobileNetV2 fine-tune", "Purpose"],
    [["smoke", "2,400", "2", "1", "1", "1", "CPU sanity run"], ["demo", "all", "12", "10", "6", "8", "Quick demonstration"], ["full", "all", "30", "15", "12", "15", "Reported results"]],
    [0.9, 1.0, 1.0, 1.2, 1.1, 1.2, 1.6], { size: 18, align: [LEFT, CENTER, CENTER, CENTER, CENTER, CENTER, LEFT] }));
  o.push(h2("5.2 Data Collection and Analysis"));
  o.push(h3("5.2.1 Dataset and Acquisition"));
  o.push(p("The study uses the German Traffic Sign Recognition Benchmark {@stallkamp2011}, obtained as a single 320 MB archive from a public mirror {@zenodo}. The notebook downloads the archive, computes its MD5 checksum in streaming fashion, compares it with the published value (7faafd0d78d831c7eb4d555641a83a62) and extracts it only if they match. The archive contains a <c>Train</c> folder with one sub-folder per class, a <c>Test</c> folder, and CSV files (<c>Train.csv</c>, <c>Test.csv</c>) that give the original width, height, region of interest and class of every image. File names in the training folders have the form <c>class_track_frame.png</c>, which identifies the physical sign (the <i>track</i>) that each image belongs to."));
  o.push(...tbl("dataset", "Dataset facts measured by the notebook", ["Property", "Value"],
    [["Training images", "39,209"], ["Official test images", "12,630"], ["Classes", "43"], ["Physical signs (training tracks)", "1,307, with 29 to 30 frames each"], ["Largest class", "Speed limit 50 km/h: 2,250 images"], ["Smallest class", "Speed limit 20 km/h: 210 images"],
     ["Class imbalance", "10.7 times between largest and smallest"], ["Image width", "25 to 243 px; median 43 px; 40 % under 40 px"], ["Aspect ratio (width / height)", "Median 1.00; 5th to 95th percentile 0.91 to 1.13"], ["Brightness (grey level, 3,000-image sample)", "Mean 82, standard deviation 48; range 7 to 248"]],
    [3.2, 4.2], { align: [LEFT, LEFT] }));
  o.push(h3("5.2.2 Exploratory Data Analysis"));
  o.push(p("Each finding below changed a modelling decision."));
  o.push(p("<b>Class imbalance.</b> {fig:classes} shows the number of training images per class. Speed-limit and some warning classes have many images, while several classes have a few hundred. This is why macro-F1 is reported next to accuracy, and why the relationship between class size and per-class F1 is examined in Chapter 9."));
  o.push(...fig("classes", "nb/eda_class_distribution.png", "Number of images per class in the training and official test sets", { width: 5.4, maxHeight: 4.6 }));
  o.push(p("<b>Appearance of the classes.</b> {fig:onepc} shows one random training example for each of the 43 classes. Several groups of classes look alike, notably the speed limits, which differ only in the digits."));
  o.push(...fig("onepc", "nb/eda_one_per_class.png", "One random training example per class", { width: 5.4, maxHeight: 4.6 }));
  o.push(p("<b>Image size.</b> The images are small ({fig:res}). The median width is 43 pixels and 40 % are narrower than 40 pixels, so there is little detail to preserve. This supports caching images at 64 x 64 pixels: a higher resolution would only interpolate pixels that do not exist, at higher memory and training cost."));
  o.push(...fig("res", "nb/eda_resolution.png", "Distribution of image width, size and aspect ratio in the training images", { width: 5.6 }));
  o.push(p("<b>Lighting.</b> Brightness varies from almost black to washed out ({fig:bright}). The model must tolerate these changes, which motivates contrast augmentation during training and an explicit darkening and glare test later."));
  o.push(...fig("bright", "nb/eda_brightness.png", "Brightness distribution of the training images with the darkest and brightest examples", { width: 5.4, maxHeight: 4.2 }));
  o.push(p("<b>Track structure.</b> The most important finding is that the images are not independent. The 39,209 training images come from only 1,307 physical signs. {fig:track} shows ten frames of a single Stop sign photographed as a vehicle approaches it. Consecutive frames are almost identical in appearance, differing mainly in size. A validation split that separates the frames of one sign would therefore compare the model with images it has effectively already seen."));
  o.push(...fig("track", "nb/eda_track.png", "Frames of a single physical Stop sign (one training track)", { width: 5.6 }));
  o.push(h3("5.2.3 Leakage Analysis"));
  o.push(p("To measure the effect, the notebook compares a random, class-stratified image split with a track-grouped split, each holding out about one seventh of the data. In the random split, 1,291 of the 1,307 tracks have frames on both sides, and every validation track has already been seen in training (100 %). In the grouped split, no track is shared ({fig:split}). Chapter 7 describes how the grouped split is constructed so that every class also appears in validation."));
  o.push(...fig("split", "d_split.png", "Random image split versus the track-grouped split used in this study", { width: 5.4 }));
  o.push(h2("5.3 Tools and Technology Used"));
  o.push(p("The tools are listed in Section 4.3. Training and evaluation ran on a Google Colab GPU runtime with TensorFlow 2.21.0 and Keras 3.13.2. The dataset, once cached as 8-bit arrays at 64 x 64 resolution, occupies 0.64 GB, so training streams batches from memory and is limited by the GPU rather than by disk access. Preprocessing and augmentation are layers inside each model, so the exported model accepts raw pixel values and no separate preprocessing code can drift from training."));

  // =============================================== CHAPTER 6 ===============================================
  o.push(chapter("Design and Architecture"));
  o.push(h2("6.1 Structure Chart and Work Breakdown Structure"));
  o.push(p("{fig:wbs} decomposes the project into four work packages: data, modelling, evaluation and delivery. Each package contains the tasks listed under it, and {tab:wp} states the deliverable of each package."));
  o.push(...fig("wbs", "d_wbs.png", "Work breakdown structure of the project", { width: 5.6 }));
  o.push(...tbl("wp", "Work packages and deliverables", ["Work package", "Main tasks", "Deliverable"],
    [["1 Data", "Download and checksum; exploratory analysis; grouped split; cache and augmentation", "Verified, leakage-free training, validation and test arrays"], ["2 Modelling", "Custom CNN; MobileNetV2; two-stage tuning; ladder and ensemble", "Four trained models and training histories"],
     ["3 Evaluation", "Metrics; per-class analysis and errors; Grad-CAM and t-SNE; calibration; robustness", "Result tables and figures"], ["4 Delivery", "Export and parity; TFLite; browser engine; dashboard; report and viva material", "Results bundle, website, report"]],
    [1.4, 3.8, 2.5], { size: 19 }));
  o.push(h2("6.2 Explanation of Modules"));
  o.push(p("The implementation is divided into the following modules. The Python modules are in the package <c>traffic_sign_recognition</c> and are mirrored by self-contained cells of the notebook; the JavaScript modules form the website."));
  o.push(...tbl("modules", "Modules and their responsibilities", ["Module", "Responsibility"],
    [["Data acquisition and loading", "Download, checksum verification, extraction; build label tables from folder names and CSV files; assert counts and file existence (<c>data.py</c>; notebook cells 6 and 7)."], ["Splitting and caching", "Hold out whole tracks per class; check zero overlap and full class coverage; resize and cache arrays (<c>split_train_validation</c>; notebook cells 19 and 22)."],
     ["Augmentation", "Rotation, translation, zoom and contrast layers active only in training (<c>models.py</c>)."], ["Model definitions", "Custom CNN and MobileNetV2 builders (<c>build_custom_cnn</c>, <c>build_mobilenet_v2</c>)."],
     ["Training", "Compile with Adam; early stopping, learning-rate reduction; two-stage fine-tuning (<c>train.py</c>; notebook cells 27 to 29)."], ["Evaluation and reporting", "Accuracy, macro-F1, top-3, ECE, latency; classification report and confusion matrix (<c>reporting.py</c>; cells 31 to 36)."],
     ["Explainability", "Grad-CAM for flat and nested models; embedding extraction and t-SNE (<c>gradcam.py</c>; cells 41 and 42)."], ["Reliability", "Reliability diagram, coverage curve, corruption functions (cells 44 and 45)."],
     ["Export", "Save models, predictions, tables and figures; reload parity; TFLite conversion (cells 50 to 52)."], ["Browser engine (<c>nn.js</c>)", "Loads weights; forward pass; Grad-CAM backward pass."],
     ["Imaging and charts (<c>imaging.js</c>, <c>charts.js</c>)", "Canvas helpers, corruptions, overlay; SVG and canvas charts with tooltips and table views."], ["Application (<c>demo.js</c>, <c>dashboard.js</c>)", "Live demo, stress lab, verification; results dashboard."]],
    [2.4, 5.3], { size: 19, align: [LEFT, LEFT] }));
  o.push(h2("6.3 Flow Chart and Activity Diagram"));
  o.push(p("{fig:flow} shows the control flow of the experiment pipeline. The two decision points are the data check at the start, which stops the run if the checksum or counts are wrong, and the early-stopping decision in training, which keeps training while validation accuracy improves and restores the best weights otherwise."));
  o.push(...fig("flow", "d_flowchart.png", "Flow chart of the experiment pipeline", { width: 4.6, maxHeight: 8.3 }));
  o.push(p("The browser application has a second flow, shown in {fig:site}: files produced by the notebook are converted by a build script into static data files, which the page loads; each image then passes through cropping and optional corruption, the forward pass and Grad-CAM, and the user interface. All of it runs locally."));
  o.push(...fig("site", "d_site.png", "Data and control flow of the website", { width: 5.6 }));
  o.push(h2("6.4 ER Diagram and Class Diagram"));
  o.push(p("The data handled by the system can be described by the entity-relationship diagram in {fig:er}. A class contains many tracks, a track contains many images, a model is trained in many runs, each run produces many predictions, and each image appears in many predictions."));
  o.push(...fig("er", "d_er.png", "Entity-relationship diagram of the data and results", { width: 5.4 }));
  o.push(p("{fig:class} shows the structure of the code: the Python package (a) and the browser application (b). The Python package is organised as modules of functions rather than classes, so its modules are drawn as classes with their principal functions; <c>SignNet</c> is the only true class in the browser code."));
  o.push(...fig("class", "d_class.png", "Class and module diagram of the code", { width: 5.6, maxHeight: 7.0 }));
  o.push(h2("6.5 Model Architectures and Algorithms"));
  o.push(h3("6.5.1 Custom CNN"));
  o.push(p("The custom CNN ({fig:cnn}) is intentionally small. It contains three blocks, each a 3 x 3 convolution with ReLU, batch normalisation, 2 x 2 max pooling and dropout (0.2), with 32, 64 and 128 filters. Global average pooling reduces the final 8 x 8 x 128 map to a 128-vector, which passes through a 128-unit dense layer with dropout (0.3) to a 43-way softmax. Augmentation is applied to raw pixels before they are scaled to the range 0 to 1. {tab:cnn} lists the layers and parameters; the total is 116,203 parameters, of which 115,755 are trainable, and about 41 million multiply-accumulate operations per image."));
  o.push(...fig("cnn", "d_cnn.png", "Architecture of the custom CNN", { width: 5.7 }));
  o.push(...tbl("cnn", "Layers and parameters of the custom CNN", ["Layer", "Output shape", "Parameters"],
    [["Input (raw pixels, augmentation, rescale)", "64 x 64 x 3", "0"], ["Conv 3x3, 32 filters, ReLU", "64 x 64 x 32", "896"], ["BatchNorm, MaxPool 2x2, Dropout 0.2", "32 x 32 x 32", "128"], ["Conv 3x3, 64 filters, ReLU", "32 x 32 x 64", "18,496"],
     ["BatchNorm, MaxPool 2x2, Dropout 0.2", "16 x 16 x 64", "256"], ["Conv 3x3, 128 filters, ReLU", "16 x 16 x 128", "73,856"], ["BatchNorm, MaxPool 2x2, Dropout 0.2", "8 x 8 x 128", "512"], ["Global average pooling", "128", "0"],
     ["Dense 128, ReLU; Dropout 0.3", "128", "16,512"], ["Dense 43, softmax", "43", "5,547"], ["<b>Total</b>", "", "<b>116,203</b>"]],
    [3.8, 1.9, 1.3], { size: 19, align: [LEFT, CENTER, RIGHT] }));
  o.push(h3("6.5.2 MobileNetV2 Transfer Learning"));
  o.push(p("The transfer-learning model ({fig:mobile}) upsamples the 64 x 64 input to 128 x 128 inside the model, scales it to the range -1 to 1 as MobileNetV2 expects, passes it through the ImageNet-pretrained backbone with average pooling (a 1,280-dimensional vector), applies dropout (0.25) and a 43-way dense layer. It is trained in two stages. In stage 1 the backbone is frozen and only the new dense layer is trained (55,083 parameters). In stage 2 the last 50 backbone layers are unfrozen while all batch-normalisation layers stay frozen, and training continues at a ten times lower learning rate (1,834,219 of 2,313,067 parameters trainable)."));
  o.push(...fig("mobile", "d_mobilenet.png", "Two-stage training of the MobileNetV2 model", { width: 5.7 }));
  o.push(h3("6.5.3 Algorithms"));
  o.push(p("Algorithm 6.1 describes the construction of the leakage-free validation split. Algorithm 6.2 describes training with early stopping, and Algorithm 6.3 the Grad-CAM computation used in the browser. Algorithm 6.4 describes selective prediction."));
  o.push(...code("alg1", "Per-class, track-grouped validation split", [
    "Input: table of training images with (label, track); seed s; fraction f = 1/7",
    "Output: fitting set F and validation set V",
    "rng <- random generator seeded with s",
    "V_tracks <- empty list",
    "for each class c in 0..42:",
    "    T <- distinct tracks of class c",
    "    k <- max(1, round(f * |T|))",
    "    V_tracks <- V_tracks + k tracks drawn from T without replacement using rng",
    "V <- images whose track is in V_tracks;   F <- all other images",
    "assert no track occurs in both F and V",
    "assert every class occurs in V"], "Algorithm"));
  o.push(...code("alg2", "Training with early stopping and learning-rate reduction", [
    "Input: model, fitting set F, validation set V, learning rate a, patience p, epoch limit E",
    "best <- -infinity;  wait <- 0",
    "for epoch = 1..E:",
    "    update the weights on augmented mini-batches of F with Adam(a)",
    "    v <- accuracy of the model on V (augmentation off)",
    "    if validation loss did not improve for p/2 epochs: a <- 0.4 * a",
    "    if v > best: best <- v; save the weights; wait <- 0",
    "    else: wait <- wait + 1",
    "    if wait >= p: break",
    "restore the saved weights"], "Algorithm"));
  o.push(...code("alg3", "Grad-CAM for the last convolutional layer (class score before softmax)", [
    "Input: image x, network with last convolution output A (16 x 16 x 128), target class c",
    "forward pass: obtain A, the logits z and the probabilities p",
    "y_c <- z[c]                         # class score before the softmax",
    "back-propagate dy_c / dA through dense layers, global average pooling,",
    "    max pooling (to the arg-max positions) and batch normalisation",
    "alpha_k <- mean over positions of dy_c / dA[:, :, k]",
    "cam <- ReLU( sum_k alpha_k * A[:, :, k] )",
    "cam <- cam / (max(cam) + 1e-8)",
    "return cam, upsampled bilinearly to the image size"], "Algorithm"));
  o.push(...code("alg4", "Selective prediction with an answer threshold", [
    "Input: probability vector p for an image, threshold t",
    "conf <- max(p);  class <- argmax(p)",
    "if conf >= t: answer class with confidence conf",
    "else: defer to a human or another sensor",
    "Offline: for t in a grid, coverage(t) = mean(conf >= t); accuracy(t) = accuracy on {conf >= t}"], "Algorithm"));
  return o.flat();
};
