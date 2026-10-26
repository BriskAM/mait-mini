// Front matter: cover, declaration, certificate, acknowledgement, institute pages, abstract, contents and lists.
const L = require("./lib");
const { d, fs, path, FONT, TEXT_W, runs, tocLine, pageOf } = L;
const { Paragraph, TextRun, ImageRun, AlignmentType, TabStopType, BorderStyle } = d;

const TITLE = "EXPLAINABLE AND RELIABLE TRAFFIC-SIGN RECOGNITION ON GTSRB";
const STUDENTS = [{ name: "Akshit Mehta", enr: "00314803123" }, { name: "Shashwat Kumar", enr: "00414803123" }];
const SUPERVISOR = { name: "Mr. Ajay Kumar Kaushik", desig: "Assistant Professor" };
const CENTER = AlignmentType.CENTER;
const logo = (file, w, h) => new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync(path.resolve(__dirname, "assets", file)), transformation: { width: w, height: h }, altText: { title: "MAIT logo", description: "Maharaja Agrasen Institute of Technology logo", name: file } })], alignment: CENTER, spacing: { before: 120, after: 120 } });
const line = (text, o = {}) => new Paragraph({ children: runs(text, { size: o.size || 24, bold: o.bold, italics: o.italics }), alignment: o.align || CENTER, spacing: { before: o.before || 0, after: o.after != null ? o.after : 120, line: o.line || 300 }, pageBreakBefore: o.pageBreak, keepNext: o.keepNext, indent: o.indent });
const title = (text, o = {}) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: o.size || 32, bold: true })], alignment: CENTER, pageBreakBefore: o.pageBreak !== false, spacing: { before: 0, after: o.after != null ? o.after : 480, line: 276 } });
const body = (text, o = {}) => L.body(text, o);
const sig = (s) => [line(`(${s.name})`, { align: AlignmentType.LEFT, before: 360, after: 0, bold: true }), line(`Enrolment No.: ${s.enr}`, { align: AlignmentType.LEFT, after: 0 })];

function cover() {
  const out = [];
  out.push(line("A Minor Project Report on", { italics: true, size: 24, after: 200, before: 240 }));
  out.push(new Paragraph({ children: [new TextRun({ text: TITLE, font: "Cambria", size: 48, bold: true })], alignment: CENTER, spacing: { before: 120, after: 200, line: 360 } }));
  out.push(line("(ES-451: Minor Project)", { after: 300 }));
  out.push(line("submitted in partial fulfilment of the requirements for the award of the degree of", { italics: true, after: 200 }));
  out.push(line("Bachelor of Technology", { size: 32, bold: true, after: 60 })); out.push(line("in", { italics: true, after: 60 })); out.push(line("Information Technology", { size: 32, bold: true, after: 360 }));
  out.push(line("Submitted by", { italics: true, after: 120 }));
  for (const s of STUDENTS) { out.push(line(s.name.toUpperCase(), { size: 26, bold: true, after: 0 })); out.push(line(`Enrolment No.: ${s.enr}`, { size: 22, after: 160 })); }
  out.push(line("Under the supervision of", { italics: true, before: 160, after: 120 }));
  out.push(line(SUPERVISOR.name, { bold: true, after: 0 })); out.push(line(SUPERVISOR.desig, { italics: true, after: 200 }));
  out.push(logo("image1.png", 150, 110));
  out.push(line("DEPARTMENT OF INFORMATION TECHNOLOGY", { bold: true, after: 0, before: 80 })); out.push(line("MAHARAJA AGRASEN INSTITUTE OF TECHNOLOGY", { bold: true, after: 0 }));
  out.push(line("PSP Area, Plot No. 1, Sector-22, Rohini, Delhi-110086, India", { after: 200 })); out.push(line("December 2026", { bold: true }));
  return out;
}

function declaration() {
  return [title("DECLARATION", { after: 480 }),
    body(`This is to certify that the material embodied in this Minor Project titled “${TITLE}”, being submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Information Technology, is based on our original work. It is further certified that this Minor Project work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma. Our indebtedness to other works has been duly acknowledged at the relevant places.`, { before: 120 }),
    line("", { after: 480 }), ...sig(STUDENTS[0]), ...sig(STUDENTS[1]),
    line("Date: ____________________", { align: AlignmentType.LEFT, before: 480, after: 0 }), line("Place: Delhi", { align: AlignmentType.LEFT, after: 0 }), line("Department of Information Technology", { align: AlignmentType.LEFT, before: 240, bold: true })];
}

function certificate() {
  const sg = (n, ds) => new Paragraph({ children: runs(`<b>${n}</b>`), spacing: { before: 0, after: 0, line: 276 } });
  return [title("CERTIFICATE", { after: 480 }),
    body(`This is to certify that the work embodied in this Minor Project titled “${TITLE}”, being submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Information Technology, is original and has been carried out by ${STUDENTS[0].name} (Enrolment No. ${STUDENTS[0].enr}) and ${STUDENTS[1].name} (Enrolment No. ${STUDENTS[1].enr}) under my supervision and guidance.`),
    body("It is further certified that this Minor Project work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma to the best of my knowledge and belief."),
    line("", { after: 720 }),
    new Paragraph({ children: runs(`<b>${SUPERVISOR.name}</b>`), spacing: { after: 0, line: 276 } }), new Paragraph({ children: runs(SUPERVISOR.desig), spacing: { after: 0, line: 276 } }), new Paragraph({ children: runs("Department of Information Technology"), spacing: { after: 0, line: 276 } }), new Paragraph({ children: runs("Maharaja Agrasen Institute of Technology"), spacing: { after: 360, line: 276 } }),
    new Paragraph({ children: runs("<b>Prof. (Dr.) Amita Goel</b>"), spacing: { after: 0, line: 276 } }), new Paragraph({ children: runs("Head, Department of Information Technology"), spacing: { after: 0, line: 276 } }), new Paragraph({ children: runs("Maharaja Agrasen Institute of Technology"), spacing: { after: 0, line: 276 } })];
}

function acknowledgement() {
  return [title("ACKNOWLEDGEMENT", { after: 360 }),
    body(`We express our sincere gratitude to our mentor and faculty supervisor, ${SUPERVISOR.name}, ${SUPERVISOR.desig}, Department of Information Technology, Maharaja Agrasen Institute of Technology, for the guidance, encouragement and feedback received throughout this project.`),
    body("We thank the Project Convenors, Dr. Shallu Bashambu and Mr. Vibhor Sharma, for organising the project reviews and defences and for the clear guidelines that helped us prepare this report. We are grateful to Prof. (Dr.) Amita Goel, Head of the Department of Information Technology, and to the Maharaja Agrasen Institute of Technology for providing an environment in which students are encouraged to carry out independent technical work."),
    body("We acknowledge the creators of the German Traffic Sign Recognition Benchmark, whose openly available dataset made this study possible, and the maintainers of the open-source tools on which we relied: TensorFlow and Keras, scikit-learn, NumPy, pandas, Matplotlib, Pillow and OpenCV. We also thank the Google Colaboratory service for the GPU runtime used to train the models."),
    body("Finally, we thank our families and friends for their constant encouragement, and our classmates for the many discussions that sharpened our ideas."),
    line("", { after: 360 }), ...sig(STUDENTS[0]), ...sig(STUDENTS[1])];
}

function instituteHeader(sub) { return [line("MAHARAJA AGRASEN INSTITUTE OF TECHNOLOGY", { bold: true, after: 60, size: 24, pageBreak: true }), logo("image2.png", 70, 69), ...(sub ? [line(sub, { bold: true, after: 200, size: 24 })] : [])]; }
const hang = (label, text, o = {}) => new Paragraph({ children: [new TextRun({ text: label, font: FONT, size: 22, bold: true }), new TextRun({ children: [new d.Tab(), text], font: FONT, size: 22 })], tabStops: [{ type: TabStopType.LEFT, position: 720 }], indent: { left: 720, hanging: 720 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 100, line: 276 } });
const sub = (text) => line(text, { bold: true, size: 24, before: 200, after: 140 });

function vision() {
  return [...instituteHeader(null), sub("VISION OF THE INSTITUTE"), body("To attain global excellence through education, innovation, research, and work ethics with the commitment to serve humanity.", { line: 276 }), sub("MISSION OF THE INSTITUTE"),
    hang("M1.", "To promote diversification by adopting advancement in science, technology, management, and allied discipline through continuous learning."), hang("M2.", "To foster moral values in students and equip them for developing sustainable solutions to serve both national and global needs in society and industry."),
    hang("M3.", "To digitize educational resources and process for enhanced teaching and effective learning."), hang("M4.", "To cultivate an environment supporting incubation, product development, technology transfer, capacity building and entrepreneurship."),
    hang("M5.", "To encourage faculty-student networking with alumni, industry, institutions, and other stakeholders for collective engagement."),
    ...instituteHeader("DEPARTMENT OF INFORMATION TECHNOLOGY").slice(0, 0),
    line("DEPARTMENT OF INFORMATION TECHNOLOGY", { bold: true, size: 24, before: 480, after: 100 }), sub("VISION OF THE DEPARTMENT"), body("To establish a center of excellence promoting Information Technology related education and research for preparing technocrats and entrepreneurs with ethical values.", { line: 276 }), sub("MISSION OF THE DEPARTMENT"),
    hang("M1.", "To excel in the field by imparting quality education and skills for software development and applications."), hang("M2.", "To establish a conducive environment that promotes intellectual growth and research."), hang("M3.", "To facilitate students to acquire entrepreneurial skills for innovation and product development."), hang("M4.", "To encourage students to participate in competitive events and industry interaction with a focus on continuous learning.")];
}

function outcomes() {
  const po = [["PO1", "Engineering knowledge: Apply knowledge of mathematics, natural science, computing, engineering fundamentals and an engineering specialization as specified in WK1 to WK4 respectively to develop to the solution of complex engineering problems."],
    ["PO2", "Problem analysis: Identify, formulate, review research literature and analyze complex engineering problems reaching substantiated conclusions with consideration for sustainable development. (WK1 to WK4)"],
    ["PO3", "Design/Development of Solutions: Design creative solutions for complex engineering problems and design/develop systems/components/processes to meet identified needs with consideration for the public health and safety, whole-life cost, net zero carbon, culture, society and environment as required. (WK5)"],
    ["PO4", "Conduct Investigations of Complex Problems: Conduct investigations of complex engineering problems using research-based knowledge including design of experiments, modelling, analysis & interpretation of data to provide valid conclusions. (WK8)."],
    ["PO5", "Engineering Tool Usage: Create, select and apply appropriate techniques, resources and modern engineering & IT tools, including prediction and modelling recognizing their limitations to solve complex engineering problems. (WK2 and WK6)"],
    ["PO6", "The Engineer and The World: Analyze and evaluate societal and environmental aspects while solving complex engineering problems for its impact on sustainability with reference to economy, health, safety, legal framework, culture and environment. (WK1, WK5, and WK7)."],
    ["PO7", "Ethics: Apply ethical principles and commit to professional ethics, human values, diversity and inclusion; adhere to national & international laws. (WK9)"],
    ["PO8", "Individual and Collaborative Team work: Function effectively as an individual, and as a member or leader in diverse/multi-disciplinary teams."],
    ["PO9", "Communication: Communicate effectively and inclusively within the engineering community and society at large, such as being able to comprehend and write effective reports and design documentation, make effective presentations considering cultural, language, and learning differences."],
    ["PO10", "Project Management and Finance: Apply knowledge and understanding of engineering management principles and economic decision-making and apply these to one's own work, as a member and leader in a team, and to manage projects and in multidisciplinary environments."],
    ["PO11", "Life-Long Learning: Recognize the need for, and have the preparation and ability for i) independent and life-long learning ii) adaptability to new and emerging technologies and iii) critical thinking in the broadest context of technological change. (WK8)"]];
  const hang2 = (a, b) => new Paragraph({ children: [new TextRun({ text: a, font: FONT, size: 22, bold: true }), new TextRun({ children: [new d.Tab(), b], font: FONT, size: 22 })], tabStops: [{ type: TabStopType.LEFT, position: 900 }], indent: { left: 900, hanging: 900 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 90, line: 264 } });
  return [...instituteHeader(null).slice(0, 2), line("PROGRAM OUTCOMES (POs)", { bold: true, size: 24, after: 160 }), ...po.map(([a, b]) => hang2(a, b)),
    ...instituteHeader("DEPARTMENT OF INFORMATION TECHNOLOGY"), line("PROGRAM SPECIFIC OUTCOMES (PSOs)", { bold: true, size: 24, after: 160 }),
    hang("PSO1", "Students are equipped with programming technologies applied in the industry."), hang("PSO2", "Students are able to provide efficient solutions for emerging challenges in the computation domain such as Artificial Intelligence, Data Science, Web Technologies."), hang("PSO3", "Students are able to apply domain knowledge and expertise in the field of Information Technology for enhancing research capability to transform innovative ideas into reality."),
    line("PROGRAM EDUCATIONAL OBJECTIVES (PEOs)", { bold: true, size: 24, before: 360, after: 160 }),
    hang("PEO1", "The graduates will utilize their professional skills in Information Technology to address real-world and societal challenges."), hang("PEO2", "The graduates would become successful entrepreneurs and motivators."), hang("PEO3", "The graduates would pursue research globally and engage in producing novel products to serve humanity."), hang("PEO4", "The graduates would exhibit collaborative abilities with adherence to professional ethics.")];
}

function sdg() {
  return [title("SUSTAINABLE DEVELOPMENT GOALS (SDGs) ADDRESSED", { size: 28, after: 360 }),
    body("<b>SDG addressed:</b> SDG 9 (Industry, Innovation and Infrastructure); SDG 11 (Sustainable Cities and Communities); SDG 3 (Good Health and Well-being)."),
    body("<b>How the project contributes towards the above SDGs:</b>", { before: 160 }),
    body("<b>SDG 11 (target 11.2) and SDG 3 (target 3.6).</b> Road traffic injuries are a major cause of death worldwide {@who2023}. Driver-assistance systems that recognise traffic signs reliably can reduce errors caused by missed or misread signs. This project studies, on a public benchmark, how to build such a classifier so that its measured accuracy is honest, its decisions can be inspected, and it can say when it is unsure, which are the properties a safety-relevant component needs before any further validation."),
    body("<b>SDG 9 (targets 9.5 and 9.c).</b> The work promotes innovation and access to technology: it delivers a small model (116,203 parameters) that runs entirely in a web browser without a server, so the technique can be examined and reused on low-cost devices and in education without specialised infrastructure. All code, results and the interactive demonstration are released openly."),
    body("The project does not claim to be a certified driving system; its limitations are discussed in Chapter 11.")];
}

function rubrics() {                                   // the department's four-page rubrics sheet, one landscape image per page
  const png = (f) => { const b = fs.readFileSync(path.resolve(__dirname, "assets", f)); return [b.readUInt32BE(16), b.readUInt32BE(20), b]; };
  const out = [title("MINOR PROJECT RUBRICS", { size: 28, after: 120, pageBreak: false })];
  for (let i = 1; i <= 4; i++) {
    const [w, h, data] = png(`rubrics-${i}.png`); let W = 10.3 * 96, H = (W * h) / w; if (H > 6.3 * 96) { H = 6.3 * 96; W = (H * w) / h; }
    out.push(new Paragraph({ children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(W), height: Math.round(H) }, altText: { title: `Project rubrics, page ${i}`, description: "Department of Information Technology project rubrics", name: `rubrics-${i}` } })], alignment: CENTER, spacing: { before: 0, after: 0 }, pageBreakBefore: i > 1 }));
  }
  return out;
}

function abstract() {
  return [title("ABSTRACT", { after: 360 }),
    body(`Reliable recognition of traffic signs is a building block of driver-assistance systems, but published accuracy figures on the German Traffic Sign Recognition Benchmark (GTSRB) say little about how a model behaves on unseen signs, why it errs, or when it should not be trusted. This project builds and evaluates a complete, reproducible traffic-sign classification pipeline that treats these questions as first-class goals.`, { line: 340 }),
    body(`The 39,209 training images of GTSRB were shown to come from only 1,307 physical signs photographed about 30 times each. A random validation split places 1,291 of those signs on both sides and inflates validation scores, so a per-class, track-grouped split with zero shared signs was used instead. Two models were compared: a compact 116,203-parameter convolutional network trained from scratch and MobileNetV2 transfer learning in two stages. Each design decision was measured on a progress ladder from a trivial baseline (5.9 % validation accuracy) to the final network (97.6 %), a 39-fold reduction in error.`, { line: 340 }),
    body(`On the official test set, scored once, the custom CNN reached 96.96 % accuracy and 95.21 % macro-F1, ahead of fine-tuned MobileNetV2 (91.26 %) and an ensemble of both (96.88 %). Grad-CAM, calibration (expected calibration error 0.0105), selective prediction (99 % accuracy while answering 95.5 % of images) and corruption tests showed that the network tolerates blur, darkening and glare but is fragile to noise, occlusion and large rotation, and that its most confident errors are very dark speed-limit signs. Two runs of the pipeline differed by about half a point, which bounds what small differences can be trusted.`, { line: 340 }),
    body(`The trained network was re-implemented in dependency-free JavaScript and verified against Keras (largest probability difference 3.2 x 10<sup>-6</sup> over 116 images), and is published as a static website that classifies test images, uploads and camera frames locally, explains each prediction and stress-tests it. The report documents the methodology, implementation, testing, results and limitations, including that the system handles only cropped signs from one country and is not safety-certified.`, { line: 340 }),
    new Paragraph({ children: runs("<b>Keywords:</b> traffic-sign recognition, GTSRB, convolutional neural network, transfer learning, Grad-CAM, calibration, robustness, data leakage."), spacing: { before: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED })];
}

// ---- contents ------------------------------------------------------------------------------------------------------------------------
function contents(prev) {
  const out = [title("TABLE OF CONTENTS", { after: 360 })];
  const pre = [["Declaration", "DECLARATION"], ["Certificate", "CERTIFICATE"], ["Acknowledgement", "ACKNOWLEDGEMENT"], ["Vision and Mission of the Institute and the Department", "VISION OF THE INSTITUTE"], ["Program Outcomes, Program Specific Outcomes and Program Educational Objectives", "PROGRAM OUTCOMES (POs)"],
    ["Sustainable Development Goals (SDGs) Addressed", "SUSTAINABLE DEVELOPMENT GOALS (SDGs) ADDRESSED"], ["Minor Project Rubrics", "MINOR PROJECT RUBRICS"], ["Abstract", "ABSTRACT"], ["List of Figures", "LIST OF FIGURES"], ["List of Tables", "LIST OF TABLES"]];
  for (const [t, k] of pre) out.push(tocLine(t, pageOf(k), 2));
  for (const h of (prev.headsList || [])) out.push(tocLine(h.text, pageOf(h.key), h.level === 1 ? 1 : h.level));
  return out;
}
function listOf(kind, prev) {
  const isFig = kind === "fig", arr = (isFig ? prev.figsList : prev.tabsList) || [];
  return [title(isFig ? "LIST OF FIGURES" : "LIST OF TABLES", { after: 300 }), new Paragraph({ children: [new TextRun({ text: isFig ? "Figure No. and Title" : "Table No. and Title", font: FONT, size: 22, bold: true }), new TextRun({ children: [new d.Tab(), "Page No."], font: FONT, size: 22, bold: true })], tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }], spacing: { after: 120 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 2 } } }),
    ...arr.map((f) => new Paragraph({ children: [new TextRun({ text: `${isFig ? "Figure" : "Table"} ${f.num}: ${f.title}`, font: FONT, size: 22 }), new TextRun({ children: [new d.Tab(), String(pageOf((isFig ? "F:" : "T:") + f.num, ""))], font: FONT, size: 22 })], tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: d.LeaderType.DOT }], spacing: { line: 276, after: 40 }, indent: { left: 0, right: 500 }, keepLines: true }))];
}

module.exports = { cover, declaration, certificate, acknowledgement, vision, outcomes, sdg, rubrics, abstract, contents, listOf, TITLE, STUDENTS };
