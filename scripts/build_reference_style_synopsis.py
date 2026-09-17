#!/usr/bin/env python3
"""Create a formal five-page synopsis matching the supplied reference style."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "traffic_sign_recognition_reference_style_synopsis.pdf"


def paragraph(text, style):
    return Paragraph(text, style)


def specs_table(rows, widths=(2.0 * inch, 2.1 * inch, 2.7 * inch)):
    data = [[Paragraph(f"<b>{a}</b>", styles["TableCell"]), Paragraph(f"<b>{b}</b>", styles["TableCell"]), Paragraph(f"<b>{c}</b>", styles["TableCell"])] for a, b, c in rows[:1]]
    data += [[Paragraph(str(a), styles["TableCell"]), Paragraph(str(b), styles["TableCell"]), Paragraph(str(c), styles["TableCell"])] for a, b, c in rows[1:]]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7E7E7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch, f"MAIT Mini Project | Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleRef", parent=styles["Title"], fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=20))
styles.add(ParagraphStyle(name="ProjectSynopsis", parent=styles["Heading1"], fontName="Times-Bold", fontSize=17, leading=21, alignment=TA_LEFT, spaceBefore=2, spaceAfter=16))
styles.add(ParagraphStyle(name="SectionRef", parent=styles["Heading2"], fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyRef", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8.7, leading=11))
styles.add(ParagraphStyle(name="RefEntry", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.5, leading=12, leftIndent=14, firstLineIndent=-14, spaceAfter=5))


def build():
    results = json.loads((ROOT / "site/dist/data/results.json").read_text())
    custom = next(x for x in results["models"] if x["name"] == "custom_cnn")
    mobile = next(x for x in results["models"] if x["name"] == "mobilenet_v2")

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.82 * inch, leftMargin=0.82 * inch, topMargin=0.72 * inch, bottomMargin=0.68 * inch, title="GTSRB Traffic Sign Recognition Project Synopsis")
    story = []
    story += [paragraph("GTSRB Traffic Sign Recognition Using Deep Learning", styles["TitleRef"]), paragraph("PROJECT SYNOPSIS", styles["ProjectSynopsis"])]
    story += [paragraph("1. Project Title", styles["SectionRef"]), paragraph("GTSRB Traffic Sign Recognition Using Deep Learning", styles["BodyRef"])]
    story += [paragraph("2. Problem Statement", styles["SectionRef"]), paragraph("Traffic signs communicate rules and warnings, but recognition is difficult when signs are small, tilted, blurred, unevenly illuminated, or visually similar to other classes. The German Traffic Sign Recognition Benchmark contains 43 categories and real-world variation, making it a useful computer-vision classification problem. The project addresses the classification task: given one cropped traffic-sign image, predict its class and confidence. Full-scene object detection and safety-critical driving decisions are outside the scope of this version.", styles["BodyRef"])]
    story += [paragraph("3. Project Abstract", styles["SectionRef"]), paragraph(f"This project builds a reproducible TensorFlow/Keras pipeline for the 43-class German Traffic Sign Recognition Benchmark. The complete archive is downloaded from Zenodo, verified with the published MD5 checksum, and checked for every class ID from 0 through 42. The pipeline performs exploratory data analysis, grouped validation splitting, resizing, normalization, moderate training-only augmentation, model training, official-test evaluation, error analysis, Grad-CAM visualization, and single-image inference. A compact custom CNN trained for 30 epochs on a Colab GPU achieved {custom['accuracy']:.2%} test accuracy and {custom['macro_f1']:.2%} macro F1. A frozen-backbone MobileNetV2 comparison trained for 25 epochs achieved {mobile['accuracy']:.2%} accuracy and {mobile['macro_f1']:.2%} macro F1. The result is an auditable classification system with documented limitations and reproducible artifacts.", styles["BodyRef"])]
    story.append(PageBreak())

    story += [paragraph("4. Literature Survey", styles["SectionRef"]), paragraph("The project design is based on verified primary papers and official documentation. Stallkamp et al. define GTSRB as a multi-category traffic-sign benchmark with changes in distance, illumination, weather, occlusion, and rotation; this motivates the dataset, EDA, and augmentation strategy. Sandler et al. describe MobileNetV2's inverted residuals, linear bottlenecks, and depthwise convolutions; this motivates a lightweight transfer-learning comparison. Kingma and Ba introduce Adam, used as the practical baseline optimizer. Selvaraju et al. introduce Grad-CAM, used to visualize regions contributing to a prediction. Official scikit-learn, TensorFlow, and Keras documentation support grouped validation, streaming input pipelines, augmentation, early stopping, learning-rate reduction, and checkpoint saving.", styles["BodyRef"])]
    story.append(specs_table([
        ("Source", "Verified contribution", "Project use"),
        ("Stallkamp et al. 2011", "GTSRB benchmark and visual variation", "Dataset boundary and evaluation"),
        ("Sandler et al. 2018", "Efficient MobileNetV2 architecture", "Transfer-learning comparison"),
        ("Kingma and Ba 2015", "Adaptive Adam optimizer", "Training optimizer"),
        ("Selvaraju et al. 2017", "Gradient-based visual explanations", "Grad-CAM diagnostic"),
        ("Official documentation", "Grouped split, tf.data, augmentation, callbacks", "Validation and pipeline design"),
    ]))
    story += [Spacer(1, 10), paragraph("5. System Specifications", styles["SectionRef"]), paragraph("The final benchmark was executed in Google Colab with a visible GPU. Local development and inference were also verified on an Apple M1 environment. Dataset and trained models remain outside Git because they are generated artifacts.", styles["BodyRef"])]
    story.append(specs_table([
        ("Parameter", "Specification", "Operational purpose"),
        ("Benchmark hardware", "Google Colab GPU", "30/25-epoch training and evaluation"),
        ("Local hardware", "Apple M1, 8 GB unified memory", "Development, smoke tests, inference"),
        ("Dataset", "39,209 train; 12,630 official test", "43-class recognition benchmark"),
        ("Storage policy", "Dataset and artifacts ignored by Git", "Keep repository reproducible and lightweight"),
    ]))
    story += [paragraph("6. Software and Environment Specifications", styles["SectionRef"])]
    story.append(specs_table([
        ("Software / library", "Version / platform", "Role"),
        ("Python", "3.11 project environment", "Reproducible execution"),
        ("TensorFlow / Keras", "GPU-capable Colab runtime", "Training and inference"),
        ("NumPy / pandas", "Pinned project dependencies", "Arrays and label tables"),
        ("scikit-learn", "GroupShuffleSplit and metrics", "Validation and evaluation"),
        ("Matplotlib / Pillow / OpenCV", "Pinned project dependencies", "EDA, images, plots, Grad-CAM"),
    ]))
    story.append(PageBreak())

    story += [paragraph("7. Methodology", styles["SectionRef"]), paragraph("The archive is downloaded and verified before extraction. Training labels are inferred from class directories, while official test labels are read from Test.csv. Related sequence-like identifiers are grouped before creating a validation split to reduce near-duplicate leakage. Images are resized to 64 × 64 for the custom CNN and 128 × 128 for MobileNetV2, converted to RGB, and normalized to [0, 1]. Rotation, translation, zoom, contrast, and brightness transformations are applied only during training.", styles["BodyRef"])]
    story.append(specs_table([
        ("Stage", "Implementation", "Reason"),
        ("Acquisition", "Zenodo download plus MD5 verification", "Trust the input archive"),
        ("EDA", "Class counts, sample grid, dimensions, brightness, color", "Understand data variation"),
        ("Validation", "Grouped 85/15 split from training data", "Reduce sequence leakage"),
        ("Baseline", "Compact CNN from scratch", "Transparent reference model"),
        ("Comparison", "ImageNet MobileNetV2 with frozen backbone", "Lightweight transfer-learning baseline"),
        ("Training control", "Adam, early stopping, LR reduction, checkpoint", "Stable and recoverable training"),
        ("Interpretability", "Grad-CAM over a final convolutional layer", "Inspect image regions used"),
    ]))
    story += [paragraph("8. Expected and Measured Outputs", styles["SectionRef"]), paragraph("Every run saves a Keras model, training history, learning curves, classification report, confusion matrix, and error-analysis plot. The dashboard summarizes EDA and model comparison. The standalone Colab notebook downloads the dataset directly, trains both models with live GPU output, and packages the resulting models and reports into a downloadable ZIP.", styles["BodyRef"])]
    story.append(specs_table([
        ("Model", "Input / run", "Measured official-test result"),
        ("Custom CNN", "64 × 64; 30 Colab GPU epochs", f"Accuracy {custom['accuracy']:.2%}; Macro F1 {custom['macro_f1']:.2%}"),
        ("MobileNetV2", "128 × 128; 25 Colab GPU epochs", f"Accuracy {mobile['accuracy']:.2%}; Macro F1 {mobile['macro_f1']:.2%}"),
    ]))
    story += [paragraph("9. Scope and Limitations", styles["SectionRef"]), paragraph("The result is a classification system for cropped sign images, not a complete autonomous-driving component. The custom CNN currently outperforms the frozen MobileNetV2 baseline in the Colab run; MobileNetV2 should be fine-tuned and retuned before drawing broader architectural conclusions. The optional YOLO code is a future detector scaffold because the classification archive does not include full-scene bounding-box annotations. Further work should include calibration, out-of-distribution detection, longer matched experiments, fine-tuning, and real-world safety validation.", styles["BodyRef"])]
    story.append(PageBreak())

    story += [paragraph("10. Project Deliverables", styles["SectionRef"]), paragraph("The repository contains the verified downloader, reusable TensorFlow modules, command-line training/evaluation/inference interfaces, EDA and reporting scripts, tests and CI configuration, a standalone Colab notebook, an interactive dashboard, optional Grad-CAM and YOLO extensions, an IEEE-style report, a presenter guide, and an IEEE presentation with speaker notes.", styles["BodyRef"])]
    story.append(specs_table([
        ("Deliverable", "Location", "Purpose"),
        ("Colab notebook", "notebooks/traffic_sign_recognition_colab.ipynb", "Standalone GPU execution"),
        ("Project report", "docs/traffic_sign_recognition_project_report.pdf", "Detailed technical record"),
        ("Synopsis", "docs/traffic_sign_recognition_synopsis.pdf", "Short project summary"),
        ("Presenter guide", "docs/traffic_sign_recognition_presenter_guide.pdf", "Beginner explanation and viva preparation"),
        ("Presentation", "docs/traffic_sign_recognition_ieee_presentation.pptx", "Presentation with speaker notes"),
        ("Literature survey", "docs/literature_survey.md", "Verified sources and design mapping"),
    ]))
    story += [paragraph("11. Conclusion", styles["SectionRef"]), paragraph("This project demonstrates the complete machine-learning lifecycle for traffic-sign recognition: verified data acquisition, exploratory analysis, leakage-aware validation, preprocessing, model training, GPU benchmarking, multi-metric evaluation, visual explanation, and communication. The Colab result shows that a well-trained compact CNN can reach high accuracy on this classification benchmark, while the weaker MobileNetV2 result identifies a concrete next experiment: fine-tune the transfer-learning backbone instead of leaving it frozen.", styles["BodyRef"])]
    story += [paragraph("12. References", styles["SectionRef"])]
    references = [
        "[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, The German Traffic Sign Recognition Benchmark: A multi-class classification competition, IJCNN, 2011. https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf",
        "[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR, 2018. https://arxiv.org/abs/1801.04381",
        "[3] D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization, ICLR, 2015. https://arxiv.org/abs/1412.6980",
        "[4] R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, ICCV, 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf",
        "[5] scikit-learn, GroupShuffleSplit documentation. https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupShuffleSplit.html",
        "[6] TensorFlow and Keras official documentation for tf.data, preprocessing layers, and callbacks. https://www.tensorflow.org/guide/data_performance and https://keras.io/api/callbacks/",
    ]
    story.extend(paragraph(ref, styles["RefEntry"]) for ref in references)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
