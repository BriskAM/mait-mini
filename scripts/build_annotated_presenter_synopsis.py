#!/usr/bin/env python3
"""Build an annotated presenter copy of the proposal synopsis."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "traffic_sign_recognition_annotated_presenter_synopsis.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontName="Times-Bold", fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="CoverSmall", parent=styles["BodyText"], fontName="Times-Roman", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle(name="PageTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=16))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Times-Bold", fontSize=14, leading=17, spaceBefore=8, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="Original", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=14, textColor=colors.black, spaceAfter=7))
styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], fontName="Times-Roman", fontSize=10, leading=13, textColor=colors.HexColor("#125A91"), spaceAfter=4))
styles.add(ParagraphStyle(name="Question", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.7, leading=12, textColor=colors.HexColor("#5A3B00"), spaceAfter=2))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=5))


def p(text, style="Original"):
    return Paragraph(text, styles[style])


def note(text):
    box = Table([[p(f"<b>Say this:</b> {text}", "Note")]], colWidths=[6.7 * inch])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF4FB")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#6FA8DC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return box


def question(text):
    return p(f"<b>Likely question:</b> {text}", "Question")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawCentredString(A4[0] / 2, 0.42 * inch, f"MAIT IT | Annotated Presenter Copy | Page {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.65 * inch, title="Annotated Presenter Copy - Traffic Sign Recognition Using Deep Learning")
    story = []

    story += [Spacer(1, 0.55 * inch), p("ANNOTATED PRESENTER COPY", "PageTitle"), p("Traffic Sign Recognition Using Deep Learning", "Cover"), p("Semester VII Minor Project Synopsis", "CoverSmall"), Spacer(1, 0.35 * inch)]
    story += [p("For the presenter", "Section"), note("This is a standalone speaking document. The blue boxes are complete explanations that can be read aloud in order. The formal black text is included for context, but the presenter does not need the synopsis, report, notebook, or presentation open at the same time."), p("Students: Akshit Mehta and Shashwat Kumar", "Original"), p("Guide: Mr. Ajay Kaushik, Assistant Professor, Department of Information Technology", "Original"), Spacer(1, 0.15 * inch)]
    story += [p("60-second opening script", "Section"), note("Our project proposes a deep-learning system that can recognize one cropped traffic-sign image and assign it to one of 43 GTSRB classes. We will first understand the dataset, then prepare the images, train a small CNN as a baseline, and compare it with MobileNetV2 transfer learning. We will evaluate both models using accuracy, macro F1, per-class metrics, and a confusion matrix. The aim is to present a complete and reproducible machine-learning workflow, not to claim a safety-critical autonomous-driving system."), question("What is the project in one sentence? Answer: It is a proposed 43-class traffic-sign image-classification system using CNNs and transfer learning."), p("Complete project story", "Section"), note("The problem is that traffic signs can be difficult to recognize when they are small, rotated, blurred, poorly illuminated, or visually similar. We will use the German Traffic Sign Recognition Benchmark, or GTSRB, which contains 43 traffic-sign classes and real-world variation. We will download and verify the archive, inspect class balance and image properties, and create a validation split that keeps related image sequences together to reduce leakage. We will resize and normalize the images, then apply moderate rotation, translation, zoom, brightness, and contrast augmentation only to the training images. We will train two proposed models: a transparent custom CNN built from convolution and pooling layers, and MobileNetV2, a lightweight pretrained network used for transfer learning. Training will use Adam, early stopping, learning-rate reduction, and best-checkpoint saving. We will evaluate with accuracy, macro F1, per-class precision and recall, a classification report, a confusion matrix, and error-analysis plots. Grad-CAM will be used to visualize which image regions influence a prediction. The project will remain a classification study for cropped images; it will not claim to detect signs in full road scenes or make driving decisions."), PageBreak()]

    story += [p("1. Project Title", "Section"), p("Traffic Sign Recognition Using Deep Learning", "Original"), note("The title tells the examiner exactly what the project does: recognize traffic signs, using deep-learning models. It is classification, because the input is assumed to be a cropped sign image and the output is one class label."), question("Is this object detection? Answer: No. Detection means finding signs inside a full road image. This proposal focuses on classifying an already-cropped sign."),
              p("2. Academic Context", "Section"), p("This synopsis is prepared for the Semester VII Minor Project of the B.Tech. Information Technology programme at Maharaja Agrasen Institute of Technology. The project is supervised by Mr. Ajay Kaushik.", "Original"), note("This section establishes that the document is a formal Semester VII IT mini-project proposal. If asked about the guide, state the name exactly as written on the cover and avoid adding qualifications that are not printed in the synopsis."), question("Why is there no project ID? Answer: No project ID has been assigned to the team."),
              p("3. Problem Statement", "Section"), p("Traffic signs communicate rules and warnings, but recognition is difficult when signs are small, tilted, blurred, unevenly illuminated, or visually similar to other classes. GTSRB contains 43 categories and real-world variation. The proposed task is to predict the class and confidence for one cropped traffic-sign image.", "Original"), note("Explain the problem using an everyday example: a speed-limit sign may be small, tilted, dark, or visually similar to another speed-limit sign. A model must learn the important shape, color, and symbol patterns while ignoring irrelevant variation."), question("Why not use full road images? Answer: Full-scene detection is a different and larger problem. Keeping classification separate makes this Semester VII proposal focused and measurable."), PageBreak()]

    story += [p("4. Project Abstract", "Section"), p("The project proposes a reproducible TensorFlow/Keras pipeline for the 43-class German Traffic Sign Recognition Benchmark. The archive will be verified, explored, preprocessed, augmented during training, and used to compare a custom CNN with MobileNetV2 transfer learning. The expected outcome is an auditable classification system with documented limitations and reproducible experiment instructions.", "Original"), note("Break this into five simple stages: data, preparation, models, evaluation, and explanation. Emphasize that the word 'proposes' is intentional: this synopsis describes planned work and does not present final results."), question("What does reproducible mean? Answer: Another person should be able to follow the same setup, use the same dataset and seed, run the same commands, and obtain a comparable experiment."),
              p("5. Literature Survey", "Section"), p("The literature survey connects the design to GTSRB, CNN-based traffic-sign recognition, MobileNet and MobileNetV2, residual learning, image augmentation, Adam optimization, Grad-CAM, and official machine-learning documentation.", "Original"), note("Do not list papers without explaining them. Say: the GTSRB paper justifies the dataset and its visual variation; MobileNetV2 justifies the lightweight comparison model; the augmentation survey supports rotations and brightness changes; Adam supports the optimizer choice; Grad-CAM supports visual explanations."), question("Why do we need references? Answer: They show that the dataset, model choice, augmentation strategy, optimizer, and explanation method are grounded in established work."), PageBreak()]

    story += [p("6. System Specifications", "Section"), p("Training is planned on Google Colab with GPU support, while local development may use an Apple M1 environment. Dataset and generated model artifacts will remain outside Git.", "Original"), note("The GPU is a practical choice because image-model training involves many repeated matrix operations. Colab makes the experiment easier to reproduce for the team, while the repository stores code and instructions rather than a large dataset."), question("Why not commit the dataset? Answer: It is large, externally published, and reproducibly downloadable, so committing it would make the repository unnecessarily heavy."),
              p("7. Software and Environment Specifications", "Section"), p("The planned stack includes Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, Pillow, and OpenCV.", "Original"), note("Give one role for each tool: TensorFlow/Keras trains models, NumPy handles arrays, pandas handles labels, scikit-learn provides grouped splitting and metrics, Matplotlib creates graphs, and Pillow/OpenCV handle image operations."),
              p("8. Methodology", "Section"), p("The archive will be verified before extraction. Labels will be read from class directories and the official test metadata. Related image sequences will be grouped before validation splitting. Images will be resized, converted to RGB, normalized, and augmented only during training. Early stopping, learning-rate reduction, and best-checkpoint saving will be used.", "Original"), note("This is the main pipeline. Say it as: acquire, inspect, split safely, preprocess, augment, train, evaluate, explain. The grouped split is important because consecutive frames from the same physical sign can look almost identical and cause overly optimistic validation results."), question("Why augment only training images? Answer: Validation and test data must represent the real evaluation distribution; changing them would make the measurement less trustworthy."), PageBreak()]

    story += [p("9. Expected Outputs", "Section"), p("The experiments are expected to generate trained models, learning curves, classification reports, confusion matrices, error-analysis plots, single-image predictions, and Grad-CAM explanations.", "Original"), note("These are expected outputs, not claimed results. Explain that accuracy alone is not enough: macro F1 and per-class metrics help reveal whether minority or visually similar classes are being ignored."), question("What is a confusion matrix? Answer: A table showing which true classes are predicted correctly and which pairs are confused with each other."),
              p("10. Scope and Limitations", "Section"), p("The proposal focuses on cropped-image classification and does not claim to be a complete autonomous-driving component. Full-scene detection, camera tracking, and safety-critical driving decisions are outside scope.", "Original"), note("This boundary is a strength. It prevents overclaiming. Say clearly that a high benchmark score would not mean the model is safe for real vehicles. Real-world deployment would require detection, calibration, robustness testing, and safety validation."),
              p("11. Project Deliverables", "Section"), p("The technical deliverables are the dataset pipeline, preprocessing and augmentation modules, CNN and MobileNetV2 models, evaluation package, inference and Grad-CAM tools, and the technical report.", "Original"), note("These are the outputs of the technical work. Supporting materials such as this presenter guide or the slide deck are not project deliverables; they are only communication aids for the team."), PageBreak()]

    story += [p("12. Conclusion", "Section"), p("The proposed project will study the complete machine-learning workflow for traffic-sign recognition, from data acquisition through evaluation and visual explanation. It will compare a transparent CNN with a lightweight transfer-learning model and document the findings and limitations.", "Original"), note("Close by returning to the main message: the project is not just a model. It is a complete, reproducible ML workflow with a clear problem boundary and an evaluation plan."),
              p("Viva questions and safe answers", "Section"), p("Why GTSRB?", "Question"), p("It is a recognized 43-class traffic-sign benchmark with variation in illumination, distance, weather, rotation, occlusion, and class frequency.", "Small"), p("Why use two models?", "Question"), p("The custom CNN is transparent and establishes a baseline; MobileNetV2 tests whether a lightweight pretrained feature extractor is useful.", "Small"), p("What is transfer learning?", "Question"), p("It starts from features learned on a large image dataset and adapts the final classifier to the traffic-sign classes.", "Small"), p("What is macro F1?", "Question"), p("It calculates F1 for each class and averages them equally, so common classes do not completely hide poor performance on rare classes.", "Small"), p("What will you do if the models perform differently?", "Question"), p("We will inspect learning curves, per-class metrics, and the confusion matrix rather than judging from accuracy alone. We will report the result honestly and discuss possible causes.", "Small"), p("What is the main limitation?", "Question"), p("The proposal classifies cropped signs. It does not detect signs in full road scenes or make safety-critical driving decisions.", "Small"),
              p("Final reminder for the presenter", "Section"), note("Speak slowly, explain the pipeline in order, and use the phrases 'proposed', 'planned', and 'expected'. Do not invent an accuracy number, dataset result, guide qualification, or project ID. If asked about a detail not in the synopsis, say that it will be finalized during implementation or confirmed with the guide."), p("Reference note: the numbered references in the formal synopsis are the sources behind the dataset, model, optimization, augmentation, interpretability, software, and MAIT-formatting choices.", "Small")]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
