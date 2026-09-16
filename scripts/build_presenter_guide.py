#!/usr/bin/env python3
"""Build a beginner-friendly presenter guide for the GTSRB project."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"


def shade(cell, fill="173B57"):
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), fill)
    props.append(node)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for cell, value in zip(t.rows[0].cells, headers):
        shade(cell)
        cell.text = str(value)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in cell.paragraphs[0].runs:
            r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(9)
    doc.add_paragraph()


def figure(doc, path, caption, width=5.9):
    if not path.exists():
        return
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph(caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic = True; c.runs[0].font.size = Pt(9)


def setup(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5); section.page_height = Inches(11)
    section.top_margin = Inches(0.7); section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8); section.right_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"; normal.font.size = Pt(10.5); normal.font.color.rgb = RGBColor(31, 41, 55)
    normal.paragraph_format.space_after = Pt(5); normal.paragraph_format.line_spacing = 1.08
    for name, size, color in (("Title", 24, "173B57"), ("Heading 1", 16, "173B57"), ("Heading 2", 12, "0F766E"), ("Heading 3", 10.5, "173B57")):
        style = doc.styles[name]
        style.font.name = "Aptos Display"; style.font.size = Pt(size); style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(10); style.paragraph_format.space_after = Pt(4)
        ppr = style._element.get_or_add_pPr()
        border = ppr.find(qn("w:pBdr"))
        if border is not None:
            ppr.remove(border)
    header = section.header.paragraphs[0]
    header.text = "GTSRB Traffic Sign Recognition | Presenter Guide"
    header.runs[0].font.size = Pt(8); header.runs[0].font.color.rgb = RGBColor(100, 116, 139)
    footer = section.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Beginner guide • Classification-only project")
    footer.runs[0].font.size = Pt(8); footer.runs[0].font.color.rgb = RGBColor(100, 116, 139)
    doc.core_properties.title = "GTSRB Traffic Sign Recognition Presenter Guide"
    doc.core_properties.subject = "Beginner explanation and presentation notes"
    doc.core_properties.author = "MAIT Mini Project"


def p(doc, text, bold_prefix=None):
    para = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        para.add_run(bold_prefix).bold = True
        para.add_run(text[len(bold_prefix):])
    else:
        para.add_run(text)
    return para


def bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def build():
    doc = Document(); setup(doc)
    title = doc.add_paragraph(style="Title"); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Presenter Guide")
    sub = doc.add_paragraph("GTSRB Traffic Sign Recognition Project"); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(14); sub.runs[0].font.color.rgb = RGBColor(15, 118, 110)
    author = doc.add_paragraph("For a presenter with basic machine learning knowledge"); author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.runs[0].italic = True
    doc.add_paragraph()

    doc.add_heading("How to Use This Guide", level=1)
    p(doc, "This document is both a learning guide and a speaking script. Read the explanation before each section, then use the Presenter wording paragraph when explaining the project. You do not need to memorize every technical term. The main story is simple: we verified images, understood them, trained models, measured their mistakes, and packaged the result.")
    table(doc, ["Suggested duration", "Sections", "Main message"], [["5 minutes", "1–4", "What the problem and data are"], ["10 minutes", "5–8", "How the models learn and how results are measured"], ["5 minutes", "9–11", "Results, limitations, and live demo"]])

    doc.add_heading("1. One Minute Project Summary", level=1)
    p(doc, "The project recognizes a traffic sign from a cropped image. It uses the German Traffic Sign Recognition Benchmark, which has 43 sign categories. The pipeline downloads and verifies the dataset, explores its properties, resizes and normalizes images, creates a leakage-aware validation split, trains two neural networks, evaluates them on the official test set, and predicts a class for a new image.")
    p(doc, "Presenter wording: “Our objective is not to detect where a sign is in a full road scene. Our objective is classification: given one cropped sign image, predict which of 43 known classes it belongs to.”", "Presenter wording:")
    table(doc, ["Term", "Plain-English meaning"], [["Image classification", "Choose one label for an image"], ["Training", "Let a model learn patterns from labeled examples"], ["Validation", "Use held-out training examples to guide decisions"], ["Test set", "Final unseen examples used for reporting"], ["Inference", "Use a trained model to predict a new image"]])

    doc.add_heading("2. The Whole Pipeline", level=1)
    p(doc, "Think of the project as a factory line. Each stage produces something that the next stage needs. If the images are wrong, the analysis is wrong. If the split leaks near-duplicates, the score looks artificially good. If evaluation is incomplete, we do not know which classes fail.")
    table(doc, ["Stage", "Input", "Output", "Why it exists"], [["Acquire", "Zenodo archive", "Verified local files", "Trust the source"], ["Explore", "Images and labels", "EDA graphs", "Understand variation"], ["Prepare", "Raw images", "Normalized datasets", "Create consistent model input"], ["Train", "Training batches", "Saved models", "Learn visual patterns"], ["Evaluate", "Official test images", "Metrics and plots", "Measure generalization"], ["Infer", "One new image", "Class and confidence", "Use the model"]])
    p(doc, "Presenter wording: “The important point is that the model is only one part of the system. A machine-learning project is credible when the complete path from data to result is reproducible and measurable.”", "Presenter wording:")

    doc.add_heading("3. Dataset Acquisition and Integrity", level=1)
    p(doc, "The full GTSRB classification archive is downloaded from Zenodo. It contains 39,209 labeled training images, 12,630 official test images, and 43 class IDs from 0 through 42. The downloader calculates the MD5 hash of data.zip and compares it with the published value 7faafd0d78d831c7eb4d555641a83a62 before extracting anything.")
    p(doc, "Why use a checksum? A download can be interrupted or corrupted while still producing a file. A checksum is a compact fingerprint of the file. If even a small part changes, the computed fingerprint should change, so extraction is stopped instead of silently using bad data.")
    p(doc, "Run: python scripts/download_dataset.py", "Run:")
    bullets(doc, ["The script skips an already valid archive, so repeated runs are safe.", "It reports a clear error for a checksum mismatch or incomplete extraction.", "It checks that all expected classes 0–42 are present.", "Dataset and trained artifacts are excluded from Git because they are large generated files."])
    p(doc, "Presenter wording: “Before training, we prove that we have the intended archive. This makes the experiment repeatable on another machine.”", "Presenter wording:")

    doc.add_heading("4. Exploratory Data Analysis", level=1)
    p(doc, "EDA means exploratory data analysis. It is the stage where we ask what the data looks like before choosing or trusting a model. The project creates these graphs with python scripts/run_eda.py.")
    figure(doc, ARTIFACTS / "eda/class_distribution.png", "Figure 1. Training examples per class.")
    p(doc, "How to read it: the horizontal axis is the class ID and the vertical axis is the number of training images. Some classes have around 210 examples while others have more than 2,000. This is class imbalance. A model can achieve a reasonable overall accuracy while performing poorly on smaller classes, so we also report macro F1.", "How to read it:")
    figure(doc, ARTIFACTS / "eda/sample_grid.png", "Figure 2. Representative traffic-sign images.")
    p(doc, "How to read it: the grid shows the visual variety the model must handle. Signs can be dark, bright, blurred, tilted, partially obstructed, or photographed against different backgrounds. Similar-looking signs make the task harder.", "How to read it:")
    figure(doc, ARTIFACTS / "eda/image_statistics.png", "Figure 3. Sampled image dimensions and brightness.")
    p(doc, "How to read it: the left plot shows that source image width and height are not identical. The right plot shows a broad brightness distribution. Therefore, the model receives resized images and augmentation that teaches it not to depend on one exact size or lighting condition.", "How to read it:")
    figure(doc, ARTIFACTS / "eda/color_statistics.png", "Figure 4. Mean red, green, and blue intensity.")
    p(doc, "The RGB graph summarizes average channel intensity over a sample. It is not a class predictor by itself. It is a sanity check that color and illumination vary in the dataset and that the preprocessing pipeline should preserve useful color information.")

    doc.add_heading("5. Labels, Splits, and Leakage", level=1)
    p(doc, "A label is the correct answer attached to an image. Training images get their labels from their class directories. Test labels are read from Test.csv. The project keeps the official GTSRB test split untouched until final evaluation.")
    p(doc, "The training data is divided into a training subset and a validation subset. The model learns from the training subset; validation examples are used to monitor whether learning is improving on data it did not directly optimize against.")
    p(doc, "Leakage happens when information from the evaluation side accidentally appears in the training side. In image datasets, related frames can be nearly identical. This project groups related image sequences before splitting, which makes validation more honest.")
    p(doc, "Presenter wording: “Validation helps us choose when to stop and which checkpoint is best. The official test set is not used for those decisions; it is reserved for the final score.”", "Presenter wording:")

    doc.add_heading("6. Preprocessing and Augmentation", level=1)
    table(doc, ["Operation", "What happens", "Why"], [["Resize", "Convert images to 64×64 or 128×128", "Neural networks need consistent tensor shapes"], ["Normalize", "Scale pixel values from 0–255 to 0–1", "Stable numerical optimization"], ["Rotate", "Small random angle", "Handle tilted signs"], ["Translate", "Small horizontal or vertical shift", "Handle imperfect framing"], ["Zoom", "Small random scale change", "Handle distance variation"], ["Brightness and contrast", "Moderate lighting changes", "Improve robustness"]])
    p(doc, "A pixel is a number representing color intensity. A color image is stored as height × width × 3 because it has red, green, and blue channels. Normalization changes a pixel such as 128 into approximately 0.502. The visual meaning stays the same, but the numbers are easier for the optimizer to work with.")
    p(doc, "Augmentation creates a changed version of a training image without changing its label. A slightly rotated speed-limit sign is still a speed-limit sign. Augmentation is applied during training, not to the official test images.")

    doc.add_heading("7. Model One Custom CNN", level=1)
    p(doc, "CNN means convolutional neural network. A convolutional layer uses small learnable filters that slide across an image. Early filters often learn simple edges and colors. Deeper layers combine those patterns into shapes and symbols. Pooling reduces spatial size, dropout reduces over-reliance on individual units, and the final dense layer produces 43 class scores.")
    p(doc, "The custom CNN is the baseline. A baseline is a reference model: it tells us what performance a straightforward architecture can achieve. It is compact and easier to explain, inspect, and run on CPU.")
    p(doc, "Presenter wording: “The custom CNN learns its visual features from this dataset. Its advantage is transparency and low complexity; its limitation is that it starts without broad visual knowledge.”", "Presenter wording:")

    doc.add_heading("8. Model Two MobileNetV2", level=1)
    p(doc, "MobileNetV2 is a lightweight CNN designed to use fewer computations than many large networks. Transfer learning means starting from a model that has already learned general visual features, then adapting it to our 43 traffic-sign classes. The project also supports no-pretrained mode for offline smoke tests.")
    p(doc, "The comparison is useful because the two models learn in different ways: the custom CNN learns from scratch, while MobileNetV2 can reuse general edge, texture, and shape features. In a final benchmark, both models should receive matched training budgets.")

    doc.add_heading("9. Training and Callbacks", level=1)
    p(doc, "Training happens in batches. One batch is a small group of images processed together. An epoch means the model has seen the training data once. During each batch, the model predicts, compares its prediction with the true label, calculates a loss, and adjusts its weights using gradient descent.")
    table(doc, ["Callback", "Beginner explanation"], [["Early stopping", "Stop if validation performance stops improving"], ["Learning-rate reduction", "Make smaller updates when progress plateaus"], ["Best checkpoint", "Save the version with the best validation result"], ["Deterministic seed", "Make splits and random operations repeatable as far as hardware allows"]])
    p(doc, "Loss is an error value that training tries to minimize. Accuracy is the fraction of correct predictions. They are related but not identical: a model can improve its confidence and loss before its hard class decision changes.")

    doc.add_heading("10. Evaluation Metrics", level=1)
    table(doc, ["Metric", "Formula idea", "Question answered"], [["Accuracy", "Correct predictions ÷ all predictions", "How often is the prediction correct overall?"], ["Precision", "Correct predictions for a class ÷ all predicted as that class", "When the model says this class, how often is it right?"], ["Recall", "Correct predictions for a class ÷ all true examples of that class", "How many real examples of this class did it find?"], ["F1 score", "Balance of precision and recall", "How well does the model balance both errors?"], ["Macro F1", "Average F1 across classes", "Does performance remain fair across rare and common classes?"]])
    p(doc, "A confusion matrix is a table or heatmap where rows are the true classes and columns are predicted classes. A strong diagonal means many correct predictions. Bright off-diagonal cells show systematic confusion between two labels. This is more useful than accuracy alone because it tells us what the model confuses.")

    doc.add_heading("11. Literature Survey and Design Evidence", level=1)
    p(doc, "A literature survey answers two questions: what has already been established by researchers, and how did that evidence influence our implementation? The sources below were checked against the original papers or official documentation. They support design choices; they do not automatically prove that our particular model is optimal.")
    table(doc, ["Source", "Main idea", "Used here"], [
        ["Stallkamp et al. 2011 — GTSRB", "Defines a real-world multi-class traffic-sign benchmark with visual variation.", "Dataset choice, 43-class scope, official test evaluation."],
        ["Sandler et al. 2018 — MobileNetV2", "Efficient inverted residual and depthwise-convolution architecture.", "Lightweight transfer-learning comparison model."],
        ["Kingma and Ba 2015 — Adam", "Adaptive optimizer based on gradient moment estimates.", "Optimizer used in the training protocol."],
        ["Selvaraju et al. 2017 — Grad-CAM", "Gradient-based class-discriminative localization map.", "Qualitative explanation and error inspection."],
        ["scikit-learn — GroupShuffleSplit", "Splits samples by group rather than independently.", "Keeps related sequences together during validation."],
        ["TensorFlow and Keras documentation", "Streaming/prefetched input pipelines, augmentation, and callbacks.", "Efficient loading, training-only transformations, checkpointing, and stopping rules."],
    ])
    p(doc, "How to explain the connection: the GTSRB paper motivates the dataset and the need to handle variation; MobileNetV2 motivates a compact comparison architecture; Adam and Keras callbacks define a practical optimization protocol; GroupShuffleSplit protects validation from sequence-related leakage; tf.data keeps the full dataset manageable; and Grad-CAM gives a visual diagnostic. The optional YOLO code is not supported by the GTSRB classification labels, so it is documented as a future detector extension rather than reported as a trained detector.")
    p(doc, "Presenter wording: “Our literature survey is not a list of unrelated papers. Each reference has a traceable role: benchmark definition, architecture choice, optimizer, leakage control, data pipeline, training control, or interpretability. That traceability makes the implementation easier to justify and reproduce.”", "Presenter wording:")
    p(doc, "Verified reading list: GTSRB primary paper — https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf; MobileNetV2 — https://arxiv.org/abs/1801.04381; Adam — https://arxiv.org/abs/1412.6980; Grad-CAM — https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf; GroupShuffleSplit — https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupShuffleSplit.html; TensorFlow/Keras — https://www.tensorflow.org/guide/data_performance and https://keras.io/api/callbacks/.", "References:")

    doc.add_heading("12. Actual Results", level=1)
    metrics = json.loads((ROOT / "site/dist/data/results.json").read_text())["models"]
    table(doc, ["Model", "Input", "Run", "Accuracy", "Macro F1"], [[m["name"].replace("_", " ").title(), "64×64" if m["name"] == "custom_cnn" else "128×128", "Colab GPU · 30/25 epochs", f"{m['accuracy']:.2%}", f"{m['macro_f1']:.2%}"] for m in metrics])
    figure(doc, ARTIFACTS / "models/model_comparison.png", "Figure 5. Test accuracy and macro F1.")
    p(doc, "The Colab GPU run gives the custom CNN 97.12% test accuracy and 95.14% macro F1. MobileNetV2 gives 76.18% test accuracy and 68.23% macro F1. The custom CNN is ahead because it learns directly from the GTSRB images while the frozen MobileNetV2 transfer-learning setup was not fully adapted to this task.")
    p(doc, "Important qualification: these are reproducible 30/25-epoch baselines, not a maximum-performance claim. Results can vary with the Colab GPU, TensorFlow version, random seeds, and whether class weighting or fine-tuning is enabled. The MobileNetV2 configuration should be improved before treating it as the preferred model.")
    figure(doc, ARTIFACTS / "models/custom_cnn/training_curves.png", "Figure 6. Custom CNN learning curves.")
    figure(doc, ARTIFACTS / "models/mobilenet_v2/training_curves.png", "Figure 7. MobileNetV2 learning curves.")
    p(doc, "Learning curves show how training and validation loss or accuracy change over epochs. If training improves while validation worsens, that is overfitting: the model is memorizing training examples instead of learning patterns that generalize.")
    figure(doc, ARTIFACTS / "models/custom_cnn/evaluation/confusion_matrix.png", "Figure 8. Custom CNN confusion matrix.")
    figure(doc, ARTIFACTS / "models/mobilenet_v2/evaluation/confusion_matrix.png", "Figure 9. MobileNetV2 confusion matrix.")

    doc.add_heading("13. Live Demonstration Script", level=1)
    p(doc, "Use this order during a demo. Explain what each command proves before running it.")
    table(doc, ["Step", "Command", "What to point out"], [["1. Check data", "python scripts/download_dataset.py", "Checksum and class validation"], ["2. Generate EDA", "python scripts/run_eda.py", "Graphs come from the full dataset"], ["3. Train baseline", "python -m traffic_sign_recognition.train --model custom_cnn", "Model, checkpoint, and history are saved"], ["4. Evaluate", "python -m traffic_sign_recognition.evaluate --model-path artifacts/models/custom_cnn/model.keras", "Per-class metrics and confusion matrix"], ["5. Predict", "python -m traffic_sign_recognition.predict --model-path artifacts/models/custom_cnn/model.keras --image path/to/sign.png", "One image becomes a formatted prediction"], ["6. Open dashboard", "python -m http.server 8000 --directory site/dist", "EDA and results are presented visually"]])
    p(doc, "If time is limited, show the already generated graphs and run the prediction command. Full training is compute-heavy; the recorded benchmark in this report was produced on a Colab GPU, while the local M1 Metal setup is also supported for shorter experiments.")

    doc.add_heading("14. Questions You May Be Asked", level=1)
    table(doc, ["Question", "Answer"], [["Why 43 classes?", "That is the complete GTSRB classification label space."], ["Why not use the test set for validation?", "Using it for decisions would leak evaluation information and make the final score less trustworthy."], ["Why report macro F1?", "The class distribution is uneven, so macro F1 gives rare classes equal importance."], ["Why did the custom CNN score higher?", "The custom CNN learned directly from GTSRB. MobileNetV2 used a frozen transfer-learning backbone, so it needs fine-tuning for this task."], ["Is this object detection?", "The main pipeline classifies a cropped sign. An optional YOLO scaffold is included for future full-scene detection."], ["Can it be used for driving decisions?", "No. It is an academic classification demonstration and is not safety-certified."]])

    doc.add_heading("15. Glossary", level=1)
    table(doc, ["Word", "Meaning"], [["Batch", "Small group of examples processed together"], ["Class", "One possible category, such as a speed-limit sign"], ["Epoch", "One pass over the training data"], ["Feature", "A useful visual pattern such as an edge or shape"], ["Generalization", "Performing well on unseen examples"], ["Hyperparameter", "A setting chosen by the developer, such as learning rate"], ["Label", "The correct class attached to an image"], ["Model", "A learned function that maps pixels to class scores"], ["Overfitting", "Learning the training examples too specifically"], ["Tensor", "A numeric array used by the neural network"]])
    doc.add_heading("16. Final Closing Statement", level=1)
    p(doc, "Presenter wording: “This project shows the complete machine-learning lifecycle. We began with a verified dataset, used EDA to understand its challenges, prepared the data carefully, compared two models, evaluated performance with multiple metrics, inspected errors with confusion matrices, and packaged the workflow for reproducible use. The result is a classification system and an honest understanding of what it can and cannot do.”", "Presenter wording:")
    p(doc, "Remember the three points to emphasize: the data is verified, the test set is kept separate, and the results are interpreted with more than one number.")

    out = DOCS / "traffic_sign_recognition_presenter_guide.docx"
    out.parent.mkdir(exist_ok=True); doc.save(out)
    print(out)


if __name__ == "__main__":
    build()
