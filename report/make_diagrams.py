#!/usr/bin/env python3
"""Draw the report's diagrams (use case, DFD, WBS, flow chart, ER, class, architecture) into report/figures/.

Nodes measure their own text so nothing overflows. Needs matplotlib only. Colours are light fills with black outlines so the
figures print well in greyscale.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8})
FILL = {"blue": "#dce8f7", "grey": "#ececec", "green": "#dff0df", "amber": "#fbeccb", "red": "#f6dcd9", "white": "#ffffff"}


class G:
    """A drawing canvas with text-measured nodes and connectors."""

    def __init__(self, w, h, xlim, ylim):
        self.fig, self.ax = plt.subplots(figsize=(w, h)); self.ax.set_xlim(*xlim); self.ax.set_ylim(*ylim); self.ax.axis("off")
        self.fig.canvas.draw(); self.r = self.fig.canvas.get_renderer()

    def measure(self, text, fs, bold=False, style="normal"):
        t = self.ax.text(0, 0, text, fontsize=fs, fontweight="bold" if bold else "normal", style=style, linespacing=1.3)
        bb = t.get_window_extent(self.r); (x0, y0), (x1, y1) = self.ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]]); t.remove(); return x1 - x0, y1 - y0

    def node(self, cx, cy, text, fc="white", fs=7.5, shape="rect", padx=0.3, pady=0.22, minw=0, minh=0, bold=False, lw=1.0, ls="-"):
        tw, th = self.measure(text, fs, bold)
        if shape == "ellipse": w, h = max(minw, tw * 1.42 + padx), max(minh, th * 1.55 + pady)
        elif shape == "diamond": w, h = max(minw, tw * 1.95 + padx), max(minh, th * 2.0 + pady)
        else: w, h = max(minw, tw + 2 * padx), max(minh, th + 2 * pady)
        if shape == "rect": self.ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=FILL.get(fc, fc), ec="black", lw=lw, ls=ls))
        elif shape == "term": self.ax.add_patch(Ellipse((cx, cy), w * 1.1, h * 1.2, fc=FILL.get(fc, fc), ec="black", lw=lw))
        elif shape == "ellipse": self.ax.add_patch(Ellipse((cx, cy), w, h, fc=FILL.get(fc, fc), ec="black", lw=lw))
        elif shape == "diamond": self.ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], fc=FILL.get(fc, fc), ec="black", lw=lw))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)
        return Node(cx, cy, w, h, shape)

    def label(self, x, y, text, fs=7, ha="center", style="italic", bold=False):
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=fs, style=style, fontweight="bold" if bold else "normal", bbox=dict(fc="white", ec="none", pad=0.8))

    def arrow(self, pts, label=None, both=False, dashed=False, lw=1.0, lpos=0.5, lseg=None, fs=7, loff=(0, 0)):
        for a, b in zip(pts[:-2], pts[1:-1]): self.ax.plot([a[0], b[0]], [a[1], b[1]], "k", lw=lw, ls="--" if dashed else "-", solid_capstyle="butt")
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="<|-|>" if both else "-|>", mutation_scale=9, lw=lw, color="black", linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))
        if both and len(pts) == 2: pass
        if label:
            segs = list(zip(pts[:-1], pts[1:])); i = lseg if lseg is not None else max(range(len(segs)), key=lambda k: abs(segs[k][0][0] - segs[k][1][0]) + abs(segs[k][0][1] - segs[k][1][1]))
            (x1, y1), (x2, y2) = segs[i]; self.label(x1 + (x2 - x1) * lpos + loff[0], y1 + (y2 - y1) * lpos + loff[1], label, fs=fs)

    def save(self, name): self.fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05); plt.close(self.fig)


class Node:
    def __init__(self, cx, cy, w, h, shape): self.cx, self.cy, self.w, self.h, self.shape = cx, cy, w, h, shape
    def ep(self, deg):
        import math
        return (self.cx + self.w / 2 * math.cos(math.radians(deg)), self.cy + self.h / 2 * math.sin(math.radians(deg)))

    def p(self, side, off=0.0):
        cx, cy, w, h = self.cx, self.cy, self.w, self.h
        return {"l": (cx - w / 2, cy + off), "r": (cx + w / 2, cy + off), "t": (cx + off, cy + h / 2), "b": (cx + off, cy - h / 2)}[side]


# 1 -- workflow ---------------------------------------------------------------------------------------------------------------------
def workflow():
    g = G(6.3, 2.9, (0, 12.8), (0, 5.8)); steps = ["1  Download and\nverify data\n(MD5, counts)", "2  Exploratory\ndata analysis", "3  Leakage-free\ntrack-grouped\nsplit", "4  Cache 64x64\nand augmentation",
                                                   "5  Train custom CNN\nand MobileNetV2\n(two stages)", "6  Evaluate once on\nofficial test set", "7  Explain, calibrate\nand stress-test", "8  Export models,\nresults and website"]
    cols = ["blue", "blue", "amber", "blue", "green", "green", "amber", "grey"]; xs = [1.5, 4.7, 7.9, 11.1]; nodes = []
    for i, (t, c) in enumerate(zip(steps, cols)): nodes.append(g.node(xs[i] if i < 4 else xs[7 - i], 4.4 if i < 4 else 1.4, t, c, fs=6.6, minw=2.5, minh=1.7, padx=0.1))
    for i in range(3): g.arrow([nodes[i].p("r"), nodes[i + 1].p("l")])
    g.arrow([nodes[3].p("b"), nodes[4].p("t")])
    for i in range(4, 7): g.arrow([nodes[i].p("l"), nodes[i + 1].p("r")])
    g.save("d_workflow")


# 2 -- use case ----------------------------------------------------------------------------------------------------------------------
def actor(g, x, y, name):
    a = g.ax; a.add_patch(Ellipse((x, y + 0.75), 0.4, 0.4, fc="white", ec="black", lw=1)); a.plot([x, x], [y + 0.55, y - 0.15], "k", lw=1); a.plot([x - 0.35, x + 0.35], [y + 0.4, y + 0.4], "k", lw=1)
    a.plot([x, x - 0.3], [y - 0.15, y - 0.65], "k", lw=1); a.plot([x, x + 0.3], [y - 0.15, y - 0.65], "k", lw=1); a.text(x, y - 0.95, name, ha="center", va="top", fontsize=7.5, fontweight="bold", linespacing=1.2)


def usecase():
    g = G(6.3, 5.6, (0, 12.6), (-1.3, 9.6)); g.ax.add_patch(Rectangle((2.6, -0.6), 7.6, 9.8, fc="white", ec="black", lw=1.3)); g.ax.text(6.4, 8.8, "Traffic-Sign Recognition System", ha="center", fontsize=8.5, fontweight="bold")
    left = ["Download and\nverify dataset", "Explore and\nprofile data", "Train and\ncompare models", "Evaluate on\nthe test set", "Generate\nGrad-CAM", "Robustness and\ncalibration tests", "Export models\nand results"]
    right = ["Classify a\nsign image", "Upload a photo\nor use camera", "Stress-test\nan image", "Set the answer\nthreshold", "View the results\ndashboard", "Verify against\nKeras"]
    ls = [7.7, 6.5, 5.3, 4.1, 2.9, 1.7, 0.5]; rs = [7.3, 6.0, 4.7, 3.4, 2.1, 0.8]
    for t, y in zip(left, ls): g.node(4.55, y, t, "blue", fs=6.2, shape="ellipse", minw=3.0, minh=0.9)
    for t, y in zip(right, rs): g.node(8.35, y, t, "green", fs=6.2, shape="ellipse", minw=3.0, minh=0.9)
    actor(g, 0.9, 4.0, "Researcher /\nStudent"); actor(g, 11.8, 4.0, "Panel member /\nWebsite visitor")
    for y in ls: g.ax.plot([1.3, 3.05], [4.3, y], "k", lw=0.5)
    for y in rs: g.ax.plot([11.4, 9.85], [4.3, y], "k", lw=0.5)
    g.save("d_usecase")


# 3 -- DFD level 0 ----------------------------------------------------------------------------------------------------------------------
def dfd0():
    g = G(6.3, 3.4, (0, 12.6), (0, 6.8)); c = g.node(6.3, 3.4, "0\nTraffic-Sign\nRecognition System", "blue", fs=8, shape="ellipse", bold=True, minw=4.2, minh=2.5)
    a = g.node(1.5, 5.6, "GTSRB archive\n(Zenodo)", "grey", fs=7, minw=2.5, minh=0.95); b = g.node(1.5, 1.2, "Researcher", "grey", fs=7, minw=2.5, minh=0.95)
    v = g.node(11.1, 5.6, "Website visitor", "grey", fs=7, minw=2.5, minh=0.95); h = g.node(11.1, 1.2, "GitHub Pages /\nColab runtime", "grey", fs=7, minw=2.5, minh=0.95)
    g.arrow([a.p("r"), c.ep(145)], "images, labels", lpos=0.45, loff=(0, 0.3))
    g.arrow([b.p("r", 0.25), c.ep(215)], "configuration", lpos=0.4, loff=(0, 0.32)); g.arrow([c.ep(235), b.p("r", -0.25)], "metrics, figures, models", lpos=0.45, loff=(0, -0.38))
    g.arrow([v.p("l", 0.25), c.ep(35)], "image", lpos=0.5, loff=(0, 0.32)); g.arrow([c.ep(15), v.p("l", -0.25)], "prediction, Grad-CAM", lpos=0.5, loff=(0.2, -0.4))
    g.arrow([c.ep(-30), h.p("l")], "site files", lpos=0.5, loff=(0, -0.35))
    g.save("d_dfd0")


# 4 -- DFD level 1 -------------------------------------------------------------------------------------------------------------------------
def dfd1():
    g = G(6.3, 6.8, (0, 12.6), (-0.3, 12.6))
    P = lambda y, t: g.node(4.6, y, t, "blue", fs=7, shape="ellipse", minw=3.1, minh=1.35)
    def store(x, y, t):
        w, h = g.measure(t, 7); w += 0.5; g.ax.plot([x, x + w], [y + 0.32, y + 0.32], "k", lw=1.1); g.ax.plot([x, x + w], [y - 0.32, y - 0.32], "k", lw=1.1); g.ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=7); return Node(x + w / 2, y, w, 0.64, "rect")
    p1, p2, p3, p4, p5 = P(11.3, "1.0  Prepare data\n(verify, split, cache)"), P(8.7, "2.0  Train models"), P(6.1, "3.0  Evaluate and\nanalyse"), P(3.5, "4.0  Export artifacts"), P(0.9, "5.0  Browser\ninference")
    d1, d2, d3, d4, d5 = store(8.6, 11.3, "D1  Raw dataset"), store(8.6, 10.0, "D2  Cached arrays"), store(8.6, 7.4, "D3  Trained models"), store(8.6, 4.8, "D4  Metrics, figures"), store(8.6, 2.2, "D5  Site data bundle")
    e1 = g.node(0.95, 11.3, "GTSRB\narchive", "grey", fs=6.8, minw=1.7, minh=0.9); e2 = g.node(0.95, 7.4, "Researcher", "grey", fs=6.8, minw=1.7, minh=0.9); e3 = g.node(0.95, 0.9, "Website\nvisitor", "grey", fs=6.8, minw=1.7, minh=0.9)
    g.arrow([e1.p("r"), p1.p("l")], "images, labels", loff=(0, 0.3)); g.arrow([p1.p("r"), d1.p("l")], "verified files", loff=(0, 0.3))
    g.arrow([p1.ep(-35), d2.p("l", 0.1)], "cached arrays", lpos=0.55, loff=(0.1, 0.3)); g.arrow([d2.p("l", -0.15), p2.ep(35)], "training data", lpos=0.55, loff=(0.1, -0.3))
    g.arrow([e2.p("t"), (0.95, 8.7), p2.p("l")], "settings", lseg=1, loff=(0, 0.3))
    g.arrow([p2.ep(-35), d3.p("l", 0.1)], "models", lpos=0.55, loff=(0.2, 0.3)); g.arrow([d3.p("l", -0.15), p3.ep(35)], "models", lpos=0.55, loff=(0.2, -0.3))
    g.arrow([p3.ep(-35), d4.p("l", 0.1)], "metrics, figures", lpos=0.5, loff=(0.45, 0.3)); g.arrow([p3.p("l"), (0.95, 6.1), e2.p("b")], "reports", lseg=0, loff=(0, 0.3))
    g.arrow([d4.p("l", -0.15), p4.ep(35)], "results", lpos=0.55, loff=(0.2, -0.3)); g.arrow([p4.ep(-35), d5.p("l", 0.1)], "weights, JSON", lpos=0.55, loff=(0.2, 0.3)); g.arrow([d5.p("l", -0.15), p5.ep(35)], "site files", lpos=0.55, loff=(0.2, -0.3))
    g.arrow([e3.p("r", 0.2), p5.p("l", 0.2)], "image", loff=(0, 0.28)); g.arrow([p5.p("l", -0.2), e3.p("r", -0.2)], "prediction", loff=(0, -0.28))
    g.save("d_dfd1")


# 5 -- work breakdown structure ---------------------------------------------------------------------------------------------------------
def wbs():
    g = G(6.3, 4.6, (0, 12.6), (0, 9.2)); root = g.node(6.3, 8.5, "Traffic-Sign Recognition Project", "amber", fs=8, bold=True, minh=0.85)
    groups = [("1  Data", ["1.1  Download, MD5", "1.2  Exploration", "1.3  Grouped split", "1.4  Cache, augment"]), ("2  Modelling", ["2.1  Custom CNN", "2.2  MobileNetV2", "2.3  Two-stage tuning", "2.4  Ladder, ensemble"]),
              ("3  Evaluation", ["3.1  Metrics", "3.2  Per-class, errors", "3.3  Grad-CAM, t-SNE", "3.4  Calibration", "3.5  Robustness"]), ("4  Delivery", ["4.1  Export, parity", "4.2  TFLite", "4.3  Web engine", "4.4  Dashboard", "4.5  Report, viva"])]
    xs = [1.55, 4.7, 7.85, 11.0]; top = 7.2; g.ax.plot([xs[0], xs[-1]], [top + 0.5, top + 0.5], "k", lw=1); g.ax.plot([6.3, 6.3], [root.cy - root.h / 2, top + 0.5], "k", lw=1)
    for (name, items), x in zip(groups, xs):
        gn = g.node(x, top, name, "blue", fs=7.5, bold=True, minw=2.9, minh=0.8); g.ax.plot([x, x], [top + 0.5, gn.cy + gn.h / 2], "k", lw=1)
        spine_x = x - 1.35; ys = [top - 1.15 - i * 1.0 for i in range(len(items))]; g.ax.plot([spine_x, spine_x], [gn.cy - gn.h / 2, ys[-1]], "k", lw=0.9)
        for it, y in zip(items, ys):
            g.ax.plot([spine_x, spine_x + 0.2], [y, y], "k", lw=0.9); g.ax.add_patch(Rectangle((spine_x + 0.2, y - 0.34), 2.45, 0.68, fc="white", ec="black", lw=0.9)); g.ax.text(spine_x + 0.32, y, it, ha="left", va="center", fontsize=5.6)
    g.save("d_wbs")


# 6 -- flow chart ----------------------------------------------------------------------------------------------------------------------------
def flowchart():
    g = G(6.3, 9.0, (0, 12.6), (-4.0, 18.2)); cx = 4.8; W = 6.0
    items = [("term", "Start", "grey"), ("rect", "Download archive, verify MD5, extract", "blue"), ("rect", "Build label tables; assert 39,209 / 12,630 / 43", "blue"), ("dia", "Checksum and\ncounts valid?", "amber"),
             ("rect", "Hold out about 1/7 of each class's tracks\nfor validation; assert zero shared tracks", "blue"), ("rect", "Resize to 64x64; cache as uint8 arrays", "blue"), ("rect", "Build custom CNN and MobileNetV2", "blue"),
             ("rect", "Train with Adam, early stopping and\nLR reduction on plateau", "green"), ("dia", "Validation accuracy\nimproved recently?", "amber"), ("rect", "Restore best weights;\nfine-tune MobileNetV2 (stage 2)", "green"),
             ("rect", "Select model on validation;\nscore the official test set once", "blue"), ("rect", "Explain, calibrate, stress-test, export", "amber"), ("term", "End", "grey")]
    y = 17.6; nodes = []
    for kind, t, c in items:
        tw, th = g.measure(t, 7); h = (0.8 if kind == "term" else max(1.6, th * 2.0 + 0.22)) if kind != "rect" else max(0.8, th + 0.44); y_c = y - h / 2
        n = g.node(cx, y_c, t, c, fs=7, shape={"term": "term", "rect": "rect", "dia": "diamond"}[kind], minw=W if kind == "rect" else 0, minh=h); nodes.append(n); y = y_c - h / 2 - (0.95 if kind == "dia" else 0.5)
    for a, b in zip(nodes[:-1], nodes[1:]): g.arrow([a.p("b"), b.p("t")])
    d1, train, d2 = nodes[3], nodes[7], nodes[8]
    stop = g.node(10.3, d1.cy, "Stop:\nfix the data", "red", fs=7, minw=2.3, minh=0.9); g.arrow([d1.p("r"), stop.p("l")], "No", lpos=0.45, loff=(0, 0.25)); g.label(cx + 0.45, d1.cy - d1.h / 2 - 0.45, "Yes")
    g.arrow([d2.p("r"), (9.6, d2.cy), (9.6, train.cy), train.p("r")], "Yes: keep training", lseg=0, lpos=0.55, loff=(0.1, 0.28)); g.label(cx + 1.6, d2.cy - d2.h / 2 - 0.45, "No: patience reached")
    ys = [n.cy for n in nodes]; g.ax.set_ylim(ys[-1] - 1.0, 18.2)
    g.save("d_flowchart")


# 7 -- ER diagram -----------------------------------------------------------------------------------------------------------------------------------
def er():
    g = G(6.3, 4.2, (0, 12.8), (0, 8.2)); ents = {"CLASS": (1.7, 6.2, ["class_id (PK)", "sign_name", "family"]), "TRACK": (6.4, 6.2, ["track_id (PK)", "class_id (FK)", "frames"]), "IMAGE": (11.1, 6.2, ["image_id (PK)", "track_id (FK)", "width, height", "pixels 64x64x3"]),
                                                   "MODEL": (1.7, 1.9, ["model_id (PK)", "architecture", "parameters"]), "RUN": (6.4, 1.9, ["run_id (PK)", "model_id (FK)", "epochs, lr, seed", "val / test metrics"]), "PREDICTION": (11.1, 1.9, ["image_id (FK)", "run_id (FK)", "predicted class", "confidence"])}
    boxes = {}
    for n, (cx, cy, attrs) in ents.items():
        txt = "\n".join(attrs); tw, th = g.measure(txt, 6.5); w = max(2.7, tw + 0.4, g.measure(n, 7, True)[0] + 0.4); h = th + 0.3; hh = 0.6
        g.ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec="black", lw=1)); g.ax.add_patch(Rectangle((cx - w / 2, cy + h / 2), w, hh, fc=FILL["blue"], ec="black", lw=1))
        g.ax.text(cx, cy + h / 2 + hh / 2, n, ha="center", va="center", fontweight="bold", fontsize=7); g.ax.text(cx - w / 2 + 0.15, cy, txt, ha="left", va="center", fontsize=6.5, linespacing=1.3)
        boxes[n] = Node(cx, cy + hh / 2, w, h + hh, "rect")
    def rel(a, b, name, ca, cb, vertical=False):
        A, B = boxes[a], boxes[b]
        if vertical: g.arrow([A.p("b"), B.p("t")], None); g.label(A.cx + 0.2, (A.cy + B.cy) / 2, name, ha="left", fs=6.5); g.ax.text(A.cx + 0.12, A.cy - A.h / 2 - 0.3, ca, fontsize=6.5); g.ax.text(B.cx + 0.12, B.cy + B.h / 2 + 0.3, cb, fontsize=6.5)
        else:
            (x1, y1), (x2, y2) = A.p("r"), B.p("l"); g.ax.plot([x1, x2], [y1, y2], "k", lw=1); g.label((x1 + x2) / 2, y1 + 0.35, name, fs=6.5); g.ax.text(x1 + 0.08, y1 - 0.3, ca, fontsize=6.5); g.ax.text(x2 - 0.25, y2 - 0.3, cb, fontsize=6.5)
    rel("CLASS", "TRACK", "has", "1", "N"); rel("TRACK", "IMAGE", "contains", "1", "N"); rel("MODEL", "RUN", "trained as", "1", "N"); rel("RUN", "PREDICTION", "produces", "1", "N"); rel("IMAGE", "PREDICTION", "is predicted in", "1", "N", vertical=True)
    g.save("d_er")


# 8 -- class / module diagram ----------------------------------------------------------------------------------------------------------------------
def classdiag():
    g = G(6.3, 6.6, (0, 14.8), (0, 13.6))
    def cls(cx, cy, name, attrs, meths, fc="white"):
        body = "\n".join(attrs + (["-" * 3] if attrs and meths else []) + meths); tw, th = g.measure(body, 6.0); w = max(2.8, tw + 0.3, g.measure(name, 6.6, True)[0] + 0.4); h = th + 0.3; hh = 0.58
        g.ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec="black", lw=1)); g.ax.add_patch(Rectangle((cx - w / 2, cy + h / 2), w, hh, fc=FILL[fc], ec="black", lw=1)); g.ax.text(cx, cy + h / 2 + hh / 2, name, ha="center", va="center", fontweight="bold", fontsize=6.6)
        if attrs and meths:
            a_txt, m_txt = "\n".join(attrs), "\n".join(meths); ta = g.measure(a_txt, 6.0)[1]; g.ax.text(cx - w / 2 + 0.12, cy + h / 2 - 0.15, a_txt, ha="left", va="top", fontsize=6.0, linespacing=1.3)
            ly = cy + h / 2 - 0.15 - ta - 0.08; g.ax.plot([cx - w / 2, cx + w / 2], [ly, ly], "k", lw=0.6); g.ax.text(cx - w / 2 + 0.12, ly - 0.08, m_txt, ha="left", va="top", fontsize=6.0, linespacing=1.3)
        else: g.ax.text(cx - w / 2 + 0.12, cy, body, ha="left", va="center", fontsize=6.0, linespacing=1.3)
        return Node(cx, cy + hh / 2, w, h + hh, "rect")
    g.label(0.2, 13.1, "(a) Training and evaluation package (Python)", ha="left", style="normal", bold=True, fs=7.2)
    data = cls(1.9, 10.4, "data.py", [], ["load_labeled_data(root)", "split_train_validation()", "make_dataset(frame)"], "blue"); models = cls(5.6, 10.4, "models.py", [], ["augmentation()", "build_custom_cnn()", "build_mobilenet_v2()"], "blue")
    train = cls(9.2, 10.4, "train.py", [], ["set_seed(seed)", "main()  # command line"], "green"); rep = cls(12.4, 10.4, "reporting.py", [], ["save_history()", "save_evaluation()"], "amber")
    inf = cls(1.9, 6.7, "inference.py", [], ["prepare_image()", "predict_array()", "predict_image()"], "blue"); gc = cls(5.6, 6.7, "gradcam.py", [], ["explain(model, img)"], "green"); ev = cls(9.2, 6.7, "evaluate.py", [], ["main()  # command line"], "green")
    g.arrow([train.p("l"), models.p("r")], dashed=True, lw=0.8); g.arrow([train.p("t"), (train.cx, 12.35), (data.cx, 12.35), data.p("t")], dashed=True, lw=0.8); g.arrow([train.p("r"), rep.p("l")], dashed=True, lw=0.8)
    g.arrow([ev.p("t", 0.8), (ev.cx + 0.8, 9.0), (rep.cx, 9.0), rep.p("b")], dashed=True, lw=0.8); g.arrow([gc.p("l"), inf.p("r")], dashed=True, lw=0.8)
    g.ax.plot([0.1, 14.7], [4.6, 4.6], "k", lw=0.8, ls=":"); g.label(0.2, 4.15, "(b) Browser application (JavaScript)", ha="left", style="normal", bold=True, fs=7.2)
    demo = cls(1.9, 1.7, "demo.js", ["state: base, vals, cam"], ["initDemo(names)", "render()", "sweep(kind)"], "amber"); dash = cls(5.6, 1.7, "dashboard.js", ["state: sel, hideDiag"], ["buildDashboard(R)", "drawCM(width)", "select(classId)"], "amber")
    sn = cls(9.2, 1.7, "SignNet (nn.js)", ["p: weights", "bn: scale, shift"], ["load(base)", "forward(rgb)", "gradcam(rgb, cls)"], "blue"); im = cls(12.4, 1.7, "imaging.js, charts.js", [], ["canvasToRGB()", "corrupt.*", "barChart()", "lineChart()"], "green")
    g.arrow([demo.p("t"), (demo.cx, 3.85), (sn.cx - 0.5, 3.85), sn.p("t", -0.5)], dashed=True, lw=0.8); g.arrow([dash.p("t"), (dash.cx, 3.55), (im.cx, 3.55), im.p("t")], dashed=True, lw=0.8)
    g.ax.text(7.0, -0.3, "Dashed arrows denote 'uses' dependencies.", ha="center", fontsize=6.6, style="italic")
    g.save("d_class")


# 9 -- CNN architecture ------------------------------------------------------------------------------------------------------------------------------
def cnn():
    g = G(6.3, 3.1, (0, 13.4), (0, 6.2)); blocks = [("Input\n64x64x3", "grey", 3.6), ("Conv 32\n3x3, ReLU\nBatchNorm\nMaxPool\n32x32x32", "blue", 3.1), ("Conv 64\n3x3, ReLU\nBatchNorm\nMaxPool\n16x16x64", "blue", 2.7), ("Conv 128\n3x3, ReLU\nBatchNorm\nMaxPool\n8x8x128", "blue", 2.3),
                                                       ("Global\naverage\npool\n128", "amber", 1.9), ("Dense 128\nReLU\ndropout\n0.3", "amber", 1.9), ("Dense 43\nsoftmax", "green", 1.5)]
    x = 0.15; prev = None
    for t, c, h in blocks:
        tw, _ = g.measure(t, 5.9); w = tw + 0.3; n = g.node(x + w / 2, 3.3, t, c, fs=5.9, padx=0.15, minw=w, minh=h, pady=0.1)
        if prev: g.arrow([prev.p("r"), n.p("l")])
        prev = n; x += w + 0.32
    g.ax.text(6.7, 0.6, "Augmentation is active in training only (rotation, translation, zoom, contrast, before rescaling to [0, 1]).", ha="center", fontsize=6.6, style="italic")
    g.save("d_cnn")


# 10 -- MobileNetV2 two-stage -----------------------------------------------------------------------------------------------------------------------
def mobilenet():
    g = G(6.3, 3.5, (0, 12.8), (0, 7.2)); g.ax.text(0.1, 6.8, "Stage 1: train the head, backbone frozen (Adam, learning rate 1e-3)", fontsize=7.5, fontweight="bold")
    def row(y, seq):
        x = 0.1; prev = None
        for t, c in seq:
            tw, _ = g.measure(t, 6.5); w = tw + 0.35; n = g.node(x + w / 2, y, t, c, fs=6.5, padx=0.15, minw=w, minh=1.3)
            if prev: g.arrow([prev.p("r"), n.p("l")])
            prev = n; x += w + 0.3
    row(5.4, [("Input\n64x64", "grey"), ("Resize\n128x128,\nscale to [-1, 1]", "grey"), ("MobileNetV2 backbone\nImageNet weights\nFROZEN", "red"), ("Dropout\n0.25", "white"), ("Dense 43\nsoftmax\nTRAINED", "green")])
    g.ax.text(0.1, 3.6, "Stage 2: fine-tune the top 50 backbone layers, BatchNorm frozen (learning rate 1e-4)", fontsize=7.5, fontweight="bold")
    row(2.2, [("Input\n64x64", "grey"), ("Resize\n128x128,\nscale to [-1, 1]", "grey"), ("Lower\nlayers\nFROZEN", "red"), ("Top 50\nlayers\nTRAINED", "green"), ("Dropout\n0.25", "white"), ("Dense 43\nsoftmax\nTRAINED", "green")])
    g.ax.text(6.4, 0.3, "Trainable parameters: 55,083 in stage 1; 1,834,219 of 2,313,067 in stage 2.", ha="center", fontsize=6.8, style="italic")
    g.save("d_mobilenet")


# 11 -- leakage diagram ---------------------------------------------------------------------------------------------------------------------------------
def split():
    import random
    g = G(6.3, 3.3, (0, 12.6), (0, 7)); rnd = random.Random(3)
    g.ax.text(0.2, 6.7, "(a) Random image split: frames of the same sign land on both sides", fontsize=7.8, fontweight="bold"); g.ax.text(0.2, 3.35, "(b) Track-grouped split (used): each sign is wholly in training or validation", fontsize=7.8, fontweight="bold")
    cols = ["#6aa7e6", "#f0a070", "#6cc7a0", "#e8c660", "#c79ae0", "#e58a8a"]
    for row, y in [("a", 4.8), ("b", 1.5)]:
        for t in range(6):
            for f in range(8):
                x = 0.3 + t * 1.45 + f * 0.17; val = (rnd.random() < 0.18) if row == "a" else (t == 3)
                g.ax.add_patch(Rectangle((x, y + (0.9 if val else 0)), 0.15, 0.5, fc=cols[t], ec="black", lw=0.4))
        g.ax.text(10.1, y + 0.25, "training", fontsize=7.2, va="center"); g.ax.text(10.1, y + 1.15, "validation", fontsize=7.2, va="center"); g.ax.plot([0.2, 9.9], [y + 0.78, y + 0.78], "k", lw=0.4, ls=":")
    g.ax.text(0.3, 4.3, "Each colour is one physical sign (a track of about 30 frames).", fontsize=6.8, style="italic"); g.ax.text(0.3, 0.9, "Result: zero shared tracks and all 43 classes present in validation.", fontsize=6.8, style="italic")
    g.save("d_split")


# 12 -- Grad-CAM flow ----------------------------------------------------------------------------------------------------------------------------------------
def gradcam():
    g = G(6.3, 2.6, (0, 12.8), (0, 5.4)); xs = [1.15, 3.55, 6.35, 9.1]
    seq = [("Input image\n64x64", "grey"), ("Convolution\nblocks", "blue"), ("Last conv output A\n16x16x128", "amber"), ("Class score\ny_c (logit)", "blue")]
    nodes = [g.node(x, 4.4, t, c, fs=6.6, minw=2.2, minh=1.2) for x, (t, c) in zip(xs, seq)]
    for a, b in zip(nodes[:-1], nodes[1:]): g.arrow([a.p("r"), b.p("l")])
    alpha = g.node(2.9, 1.2, "Channel weights alpha_k =\nmean over positions of dy_c / dA_k", "white", fs=6.6, minw=3.8, minh=1.2); heat = g.node(9.9, 1.2, "Heat-map = ReLU( sum_k alpha_k A_k )\nscaled to [0, 1], overlaid on the image", "green", fs=6.6, minw=4.4, minh=1.2)
    g.arrow([nodes[3].p("b"), (9.1, 2.95), (2.9, 2.95), alpha.p("t")], "gradients by back-propagation", lseg=1, lpos=0.62, loff=(0, 0.25))
    g.arrow([alpha.p("r"), heat.p("l")]); g.ax.plot([6.35, 6.35], [nodes[2].cy - 0.6, 1.2], "k", lw=1); g.label(6.35, 1.75, "A", fs=6.6)
    g.save("d_gradcam")


# 13 -- website architecture ---------------------------------------------------------------------------------------------------------------------------------------
def siteflow():
    g = G(6.3, 3.6, (0, 12.8), (0, 7.4)); top = [("Colab notebook\n(training, results)", "blue"), ("results.zip\n.keras, CSV, JSON", "grey"), ("build_pages_data.py\n(weights to float32)", "amber"), ("site/data and img\n(static files)", "green")]
    xs = [1.6, 4.65, 7.9, 11.2]; ns = [g.node(x, 6.3, t, c, fs=6.6, minw=2.6, minh=1.2) for x, (t, c) in zip(xs, top)]
    for a, b in zip(ns[:-1], ns[1:]): g.arrow([a.p("r"), b.p("l")])
    g.ax.add_patch(Rectangle((0.1, 0.2), 12.6, 4.3, fc="#f7f7f7", ec="black", lw=1, ls="--")); g.ax.text(0.3, 4.15, "Visitor's browser: everything runs locally, nothing is uploaded", fontsize=7.2, fontweight="bold", va="center")
    bt = [("Image\nsample, upload,\ncamera", "grey"), ("imaging.js\ncrop, corrupt,\nresize to 64x64", "green"), ("nn.js (SignNet)\nforward pass\nand Grad-CAM", "blue"), ("User interface\ntop-5, heat-map,\nanswer or defer", "amber")]
    bx = [1.7, 4.75, 7.95, 11.1]; bn = [g.node(x, 2.2, t, c, fs=6.4, minw=2.6, minh=1.7) for x, (t, c) in zip(bx, bt)]
    for a, b in zip(bn[:-1], bn[1:]): g.arrow([a.p("r"), b.p("l")])
    g.arrow([ns[3].p("b"), (11.2, 5.0), (7.95, 5.0), bn[2].p("t")], "cnn_weights.bin (465 KB)", lseg=1, loff=(0, 0.25), dashed=True)
    g.save("d_site")


if __name__ == "__main__":
    for fn in (workflow, usecase, dfd0, dfd1, wbs, flowchart, er, classdiag, cnn, mobilenet, split, gradcam, siteflow): fn()
    print("wrote", sorted(p.name for p in OUT.glob("d_*.png")))
