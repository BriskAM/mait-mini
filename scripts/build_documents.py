#!/usr/bin/env python3
"""Create the project synopsis and report from the reproducible run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"


def shade(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    props.append(shading)


def set_cell_text(cell, text: str, bold: bool = False, color: str = "1F2937") -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    table_width_twips = 4536
    column_width_twips = table_width_twips // len(headers)
    column_width = Inches(3.15 / len(headers))
    for row in table.rows:
        for cell in row.cells:
            cell.width = column_width
    for cell, header in zip(table.rows[0].cells, headers):
        shade(cell, "173B57")
        set_cell_text(cell, header, bold=True, color="FFFFFF")
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.width = column_width
            set_cell_text(cell, value)
    tbl_pr = table._tbl.tblPr
    tbl_width = tbl_pr.find(qn("w:tblW"))
    if tbl_width is None:
        tbl_width = OxmlElement("w:tblW")
        tbl_pr.append(tbl_width)
    tbl_width.set(qn("w:type"), "dxa")
    tbl_width.set(qn("w:w"), str(table_width_twips))
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = table._tbl.tblGrid
    for col, _ in zip(grid.gridCol_lst, headers):
        col.set(qn("w:w"), str(column_width_twips))
    doc.add_paragraph()
    return table


def add_figure(doc, path: Path, caption: str, width: float = 3.15):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.keep_together = True
    c.runs[0].italic = True
    c.runs[0].font.size = Pt(9)


def setup(doc: Document, title: str):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "540")
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.0
    for style_name, size, color in (("Title", 18, "000000"), ("Heading 1", 10, "000000"), ("Heading 2", 10, "000000")):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(6)
        style.paragraph_format.space_after = Pt(3)
        ppr = style._element.get_or_add_pPr()
        border = ppr.find(qn("w:pBdr"))
        if border is not None:
            ppr.remove(border)
    doc.styles["Heading 1"].font.all_caps = True
    header = section.header.paragraphs[0]
    header.text = "GTSRB Traffic Sign Recognition"
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.name = "Times New Roman"
    header.runs[0].font.color.rgb = RGBColor(0, 0, 0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("MAIT Mini Project")
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.name = "Times New Roman"
    footer.runs[0].font.color.rgb = RGBColor(0, 0, 0)
    doc.core_properties.title = title
    doc.core_properties.subject = "GTSRB traffic-sign image classification"
    doc.core_properties.author = "MAIT Mini Project"


def title_page(doc: Document, title: str, subtitle: str):
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(title)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(subtitle)
    r.font.size = Pt(11); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0, 0, 0)
    p = doc.add_paragraph("Akshit Mehta")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.name = "Times New Roman"
    p.runs[0].font.size = Pt(10)
    doc.add_paragraph()


def load_metrics():
    result = json.loads((ROOT / "site/dist/data/results.json").read_text())
    return result["models"]


def synopsis():
    doc = Document(); setup(doc, "GTSRB Traffic Sign Recognition Project Synopsis")
    title_page(doc, "Project Synopsis", "GTSRB Traffic-Sign Recognition")
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("This synopsis presents a reproducible 43-class traffic-sign image-classification pipeline. The work covers verified dataset acquisition, exploratory analysis, grouped validation, augmented training, model comparison, evaluation, inference, and project communication through a Colab notebook and dashboard website.")
    doc.add_paragraph("Index Terms—GTSRB, traffic-sign recognition, convolutional neural network, MobileNetV2, image classification.")
    doc.add_heading("1. Project Overview", level=1)
    doc.add_paragraph("This project develops an end-to-end image-classification system for recognizing German traffic signs. It covers data acquisition, integrity validation, exploratory data analysis, preprocessing, augmentation, model training, evaluation, command-line inference, a Colab workflow, and a presentation website.")
    doc.add_heading("2. Problem Statement", level=1)
    doc.add_paragraph("Traffic signs are visually diverse and may appear at different scales, viewpoints, lighting conditions, and backgrounds. The goal is to map a cropped traffic-sign image to one of 43 GTSRB class labels while preserving a reproducible evaluation boundary. This version addresses classification only; object detection and real-time camera streaming are outside scope.")
    doc.add_heading("3. Objectives", level=1)
    for item in ["Download and verify the complete public GTSRB classification archive.", "Profile class balance, image geometry, brightness, color, and representative examples.", "Train a compact CNN baseline and compare it with MobileNetV2 transfer learning.", "Use a grouped validation split, deterministic seeds, augmentation, checkpointing, and learning-rate control.", "Publish interpretable accuracy, macro F1, per-class metrics, confusion matrices, and learning curves.", "Provide local CLI commands, a Google Colab notebook, a static project website, and formal documentation."]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_heading("4. Dataset and EDA Snapshot", level=1)
    add_table(doc, ["Measure", "Observed value"], [["Training images", "39,209"], ["Official test images", "12,630"], ["Classes", "43 (IDs 0–42)"], ["Training class range", "210–2,250 images"], ["Archive MD5", "7faafd0d78d831c7eb4d555641a83a62"], ["Source", "Zenodo record 13741936"]])
    add_figure(doc, ARTIFACTS / "eda/class_distribution.png", "Figure 1. Training class distribution.")
    doc.add_heading("5. Methodology", level=1)
    doc.add_paragraph("Images are loaded from the official training directories and test CSV, resized to a consistent model-specific input size, normalized, and converted into TensorFlow datasets. Related image sequences are grouped before validation splitting to reduce near-duplicate leakage. Moderate random rotation, translation, zoom, brightness, and contrast transformations are applied during training.")
    add_table(doc, ["Model", "Role", "Input"], [["Custom CNN", "Transparent baseline", "64 × 64 × 3"], ["MobileNetV2", "Transfer-learning comparison", "96 × 96 × 3"]])
    doc.add_heading("6. Demonstration Results", level=1)
    rows = []
    for model in load_metrics():
        rows.append([model["name"].replace("_", " ").title(), f"{model['accuracy']:.2%}", f"{model['macro_f1']:.2%}"])
    add_table(doc, ["Model", "Test accuracy", "Macro F1"], rows)
    doc.add_paragraph("The recorded runs use the full training and official test sets. The custom CNN was trained for 30 epochs and MobileNetV2 for 25 epochs on a Colab GPU. These are reproducible project runs, not a claim of the best possible GTSRB score.")
    doc.add_heading("7. Deliverables", level=1)
    doc.add_paragraph("Source modules live in src/traffic_sign_recognition. Reproducible commands are documented in README.md. The Colab notebook is notebooks/traffic_sign_recognition_colab.ipynb, the static website is site/dist, and the detailed report is docs/traffic_sign_recognition_project_report.docx.")
    doc.add_heading("8. Expected Impact", level=1)
    doc.add_paragraph("The project demonstrates how a machine-learning experiment can be made inspectable from raw data through inference. Its artifacts help students and reviewers connect data quality, model design, evaluation choices, and limitations in one coherent pipeline.")
    out = DOCS / "traffic_sign_recognition_synopsis.docx"; out.parent.mkdir(exist_ok=True); doc.save(out)


def report():
    doc = Document(); setup(doc, "GTSRB Traffic Sign Recognition Project Report")
    title_page(doc, "Project Report", "GTSRB Traffic-Sign Recognition")
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("This report presents a reproducible TensorFlow/Keras pipeline for classifying the 43 categories in the German Traffic Sign Recognition Benchmark. The implementation downloads and verifies the full archive, analyzes its statistical structure, prepares grouped training and validation data, trains two neural architectures, evaluates the official test split, and exposes single-image prediction. A static dashboard and Colab notebook make the workflow easy to inspect and rerun.")
    doc.add_heading("1. Introduction", level=1)
    doc.add_paragraph("Traffic-sign recognition is a useful computer-vision benchmark because the classes are semantically meaningful yet visually similar. Signs can differ by shape, text, color, border, and symbol, while the captured images vary in illumination, viewpoint, scale, and background. A robust project therefore needs more than a model definition: it needs a trustworthy dataset boundary, leakage-aware validation, diagnostics, and clear reporting.")
    doc.add_heading("2. Requirements and Scope", level=1)
    doc.add_paragraph("The project targets classification of an already-cropped sign image. It does not locate signs in a full scene, issue driving instructions, or make safety-critical decisions. The implementation exposes model choice, image size, seed, epochs, batch size, learning rate, sample limits, and output directory through command-line options.")
    doc.add_heading("3. Dataset", level=1)
    doc.add_paragraph("The complete GTSRB classification archive is downloaded from the Zenodo record 13741936. The published MD5 checksum is verified before extraction. The extracted package contains the 39,209 labeled training images in class directories, the 12,630 official test images, and the test labels. All expected class IDs 0 through 42 are checked.")
    add_table(doc, ["Dataset property", "Value"], [["Training split", "39,209 images"], ["Official test split", "12,630 images"], ["Label space", "43 classes, IDs 0–42"], ["Training class minimum", "210 images"], ["Training class maximum", "2,250 images"], ["Archive integrity", "MD5 verified"]])
    add_figure(doc, ARTIFACTS / "eda/sample_grid.png", "Figure 1. Representative examples from the training set.")
    add_figure(doc, ARTIFACTS / "eda/image_statistics.png", "Figure 2. Sampled image dimensions and brightness distributions.")
    doc.add_heading("4. Exploratory Data Analysis", level=1)
    doc.add_paragraph("The class distribution is uneven, so raw accuracy alone is insufficient. Macro F1 is reported alongside accuracy to give every class equal weight. Sampled image dimensions show that the source images are not uniform, which motivates consistent resizing. Brightness and RGB statistics confirm meaningful illumination and color variation.")
    add_figure(doc, ARTIFACTS / "eda/class_distribution.png", "Figure 3. The 43-class training distribution.")
    add_figure(doc, ARTIFACTS / "eda/color_statistics.png", "Figure 4. Mean RGB channel intensity over a 5,000-image sample.")
    doc.add_heading("5. Data Preparation and Leakage Control", level=1)
    doc.add_paragraph("The loader supports both the Zenodo package layout and the official GTSRB directory layout. Training labels are inferred from class directories; test labels are read from Test.csv. Related images are assigned sequence-like group identifiers derived from their source paths and grouped before the validation split. This reduces the chance that near-duplicate frames from the same sequence appear on both sides of validation.")
    doc.add_paragraph("Images are resized and normalized to [0, 1]. The augmentation pipeline applies moderate rotation, translation, zoom, brightness, and contrast changes only during training. The official test set remains untouched until final evaluation.")
    doc.add_heading("6. Models", level=1)
    doc.add_heading("6.1 Custom CNN baseline", level=2)
    doc.add_paragraph("The baseline uses stacked convolutional blocks with batch normalization, max pooling, dropout, and a dense classification head. It is intentionally compact so its behavior is easy to explain and its CPU smoke test is fast.")
    doc.add_heading("6.2 MobileNetV2 comparison", level=2)
    doc.add_paragraph("MobileNetV2 provides a lightweight transfer-learning architecture. The project supports pretrained ImageNet weights when available and a no-pretrained option for offline or smoke-test execution. Its depthwise-separable convolutions provide a useful accuracy/compute comparison with the custom baseline.")
    doc.add_heading("7. Training Protocol", level=1)
    add_table(doc, ["Setting", "Custom CNN", "MobileNetV2"], [["Input size", "64 × 64", "128 × 128"], ["Seed", "42", "42"], ["Epochs in recorded run", "30", "25"], ["Batch size", "128", "128"], ["Validation", "Grouped split", "Grouped split"], ["Hardware", "Colab GPU", "Colab GPU"], ["Callbacks", "Early stopping, LR reduction, checkpoint", "Early stopping, LR reduction, checkpoint"]])
    doc.add_paragraph("Each run saves the best Keras model, a JSON history, training curves, a classification report, and a confusion matrix under artifacts/models. Repeated execution is deterministic at the seed and data-split level, subject to TensorFlow and hardware determinism.")
    doc.add_heading("8. Literature Survey and Design Rationale", level=1)
    doc.add_paragraph("The project design is grounded in verified primary research and official documentation. Stallkamp et al. established GTSRB as a realistic multi-class traffic-sign benchmark with variation in illumination, weather, distance, occlusion, and rotation; this motivates the 43-class scope, EDA, augmentation, and official-test evaluation. Sandler et al. introduced MobileNetV2's efficient inverted-residual architecture; this motivates its use as a lightweight transfer-learning comparison. Kingma and Ba introduced Adam, used here as a practical baseline optimizer. Selvaraju et al. introduced Grad-CAM, used here for qualitative inspection of whether predictions focus on plausible sign regions. Official scikit-learn documentation supports GroupShuffleSplit for keeping related groups together during validation, while TensorFlow and Keras documentation supports streaming and prefetched tf.data pipelines, training-time augmentation, early stopping, learning-rate reduction, and checkpoint saving.")
    add_table(doc, ["Reference", "Verified contribution", "Implementation role"], [["Stallkamp et al., IJCNN 2011", "GTSRB benchmark definition and real-world visual variation", "Dataset boundary and evaluation"], ["Sandler et al., CVPR 2018", "Efficient MobileNetV2 architecture", "Transfer-learning comparison"], ["Kingma and Ba, ICLR 2015", "Adaptive Adam optimizer", "Training optimizer"], ["Selvaraju et al., ICCV 2017", "Gradient-based visual explanations", "Grad-CAM diagnostic"], ["scikit-learn / TensorFlow / Keras documentation", "Grouped splitting, streaming input, augmentation, callbacks", "Validation and training pipeline"]])
    doc.add_paragraph("The detailed source-to-code mapping and verified links are provided in docs/literature_survey.md. These sources justify design choices; they do not imply that the present configuration is optimal. The optional YOLO code remains a detector scaffold because the GTSRB classification archive does not provide full-scene bounding-box labels.")
    add_figure(doc, ARTIFACTS / "models/custom_cnn/training_curves.png", "Figure 5. Custom CNN training and validation curves.")
    add_figure(doc, ARTIFACTS / "models/mobilenet_v2/training_curves.png", "Figure 6. MobileNetV2 training and validation curves.")
    doc.add_heading("9. Results", level=1)
    metrics = load_metrics()
    add_table(doc, ["Model", "Test accuracy", "Macro F1"], [[m["name"].replace("_", " ").title(), f"{m['accuracy']:.2%}", f"{m['macro_f1']:.2%}"] for m in metrics])
    doc.add_paragraph("In these full-data Colab runs, the custom CNN is ahead on both overall accuracy and macro F1. This result is substantially stronger than the short local baseline, while still remaining a reproducible project benchmark rather than a maximum-performance claim.")
    add_figure(doc, ARTIFACTS / "models/model_comparison.png", "Figure 7. Test accuracy and macro F1 comparison.")
    add_figure(doc, ARTIFACTS / "models/custom_cnn/evaluation/confusion_matrix.png", "Figure 8. Custom CNN confusion matrix.")
    add_figure(doc, ARTIFACTS / "models/mobilenet_v2/evaluation/confusion_matrix.png", "Figure 9. MobileNetV2 confusion matrix.")
    doc.add_heading("10. Error Analysis and Limitations", level=1)
    doc.add_paragraph("The confusion matrices and per-class reports expose which signs remain difficult to separate. Likely sources include visually similar speed-limit classes, class imbalance, small or blurred signs, and the short demonstration training budgets. The pipeline does not yet include class-balanced loss, systematic hyperparameter search, calibration, test-time augmentation, or human review of mislabeled examples.")
    doc.add_paragraph("The classifier assumes a cropped sign image. It should not be used as a standalone driving or safety system. A deployment-oriented extension would require object detection, temporal smoothing, uncertainty thresholds, out-of-distribution checks, and extensive testing under real-world conditions.")
    doc.add_heading("11. Reproducibility and Usage", level=1)
    for item in ["python scripts/download_dataset.py", "python scripts/run_eda.py", "python -m traffic_sign_recognition.train --model custom_cnn", "python -m traffic_sign_recognition.train --model mobilenet_v2", "python -m traffic_sign_recognition.evaluate --model-path artifacts/models/mobilenet_v2/model.keras", "python -m traffic_sign_recognition.predict --model-path artifacts/models/mobilenet_v2/model.keras --image path/to/sign.png"]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_heading("12. Conclusion and Future Work", level=1)
    doc.add_paragraph("The project demonstrates a complete and auditable traffic-sign recognition workflow: the source archive is verified, data assumptions are visualized, two models are trained, results are measured, and inference is packaged for reuse. Future work should use matched multi-epoch GPU training, class-aware sampling or loss functions, richer error analysis, and a detector-plus-classifier system for full road scenes.")
    doc.add_heading("References", level=1)
    doc.add_paragraph("Stallkamp, J., Schlipsing, M., Salmen, J., and Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. Proceedings of IJCNN.")
    doc.add_paragraph("GTSRB public dataset distribution and Zenodo mirror: https://zenodo.org/records/13741936")
    doc.add_paragraph("TensorFlow/Keras documentation: https://www.tensorflow.org/")
    out = DOCS / "traffic_sign_recognition_project_report.docx"; out.parent.mkdir(exist_ok=True); doc.save(out)


if __name__ == "__main__":
    synopsis(); report(); print("Created synopsis and project report in docs/")
