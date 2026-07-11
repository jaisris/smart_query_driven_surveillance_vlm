# -*- coding: utf-8 -*-
"""Clean, orthogonally-routed system architecture diagram (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12.5, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C = {
    "input": ("#ECEFF1", "#546E7A"),
    "data":  ("#E3F2FD", "#1565C0"),
    "model": ("#E8F0FE", "#1A73E8"),
    "retr":  ("#E6F4EA", "#188038"),
    "anom":  ("#FEF3E0", "#E37400"),
    "ui":    ("#F3E8FD", "#8430CE"),
    "cache": ("#F1F3F4", "#80868B"),
}
ARROW = "#455A64"


def box(cx, cy, w, h, title, sub="", kind="data", fs=12):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=2.2",
        linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=3))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#1A1A1A", zorder=4)
        ax.text(cx, cy - h * 0.25, sub, ha="center", va="center",
                fontsize=fs - 3, color="#444", zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#1A1A1A", zorder=4)
    return dict(cx=cx, cy=cy, w=w, h=h,
                L=cx - w / 2, R=cx + w / 2, T=cy + h / 2, B=cy - h / 2)


def straight(p1, p2, color=ARROW, lw=2.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=17, lw=lw, color=color,
        zorder=2, shrinkA=1, shrinkB=2, linestyle=("--" if dashed else "-")))


def orth(pts, color=ARROW, lw=2.0, dashed=False):
    """Polyline through pts with an arrowhead on the final segment."""
    ls = "--" if dashed else "-"
    for i in range(len(pts) - 2):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=17, lw=lw,
        color=color, zorder=2, shrinkA=0, shrinkB=2, linestyle=ls))


# ---------------- main vertical pipeline (x = 30) ----------------
xm = 30
b_in = box(xm, 93, 30, 6.5, "Input Surveillance Video", kind="input")
b_load = box(xm, 81, 30, 8, "VideoLoader", "adaptive frame sampling (OpenCV, streaming)", "data")
b_fp = box(xm, 67.5, 32, 9, "FrameProcessor", "YOLOv8 detection + DeepSORT tracking", "model")
b_emb = box(xm, 54, 32, 8.5, "EmbeddingBuilder", "streaming CLIP encode · static-frame skip", "model")
b_faiss = box(xm, 41, 32, 8, "FAISS  IndexFlatIP", "cosine-similarity search", "retr")
b_loc = box(xm, 28, 32, 8.5, "TemporalLocalizer", "merge top-K frames into segments", "retr")
b_ui = box(42, 10, 54, 8, "Streamlit User Interface", "segments, timestamps, scores, anomaly alerts", "ui")

for a, b in [(b_in, b_load), (b_load, b_fp), (b_fp, b_emb), (b_emb, b_faiss), (b_faiss, b_loc)]:
    straight((xm, a["B"]), (xm, b["T"]))
straight((xm, b_loc["B"]), (xm, b_ui["T"]))   # localizer -> UI

# ---------------- query lane (x = 72) ----------------
b_q = box(72, 54, 26, 7.5, "Text Query", "natural language", "input")
b_ce = box(72, 41, 26, 8, "CLIPEncoder", "query embedding (512-d)", "model")
straight((72, b_q["B"]), (72, b_ce["T"]))
straight((b_ce["L"], 41), (b_faiss["R"], 41), color="#188038")   # -> FAISS (level)

# ---------------- anomaly lane ----------------
b_an = box(72, 25, 30, 8.5, "AnomalyEngine", "loitering + intrusion  (opt. VadCLIP)", "anom")
# track histories: FrameProcessor -> right rail -> down -> AnomalyEngine
orth([(b_fp["R"], 67.5), (90, 67.5), (90, 25), (b_an["R"], 25)], color="#E37400")
ax.text(67, 70.5, "track histories", ha="center", va="center",
        fontsize=9.5, style="italic", color="#E37400", zorder=5)
# anomaly -> UI
orth([(72, b_an["B"]), (72, 10), (b_ui["R"], 10)], color="#E37400")

# ---------------- content cache (x = 7) ----------------
box(7, 67.5, 11, 27, "", "", "cache")
ax.text(7, 76.5, "Content\nCache", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#3C4043", zorder=4)
ax.text(7, 62, ".npy embeddings\n+ track state\n(SHA-256 key)",
        ha="center", va="center", fontsize=8.5, color="#5F6368", zorder=4)
orth([(12.5, 81), (b_load["L"], 81)], color="#80868B", dashed=True, lw=1.5)
orth([(12.5, 54), (b_emb["L"], 54)], color="#80868B", dashed=True, lw=1.5)

# ---------------- title + legend ----------------
ax.text(50, 99, "System Architecture and Data Flow", ha="center", va="top",
        fontsize=15, fontweight="bold", color="#143A66")
legend = [("Input / Data", "data"), ("Model", "model"), ("Retrieval", "retr"),
          ("Anomaly", "anom"), ("Interface", "ui")]
lx = 20
for name, kind in legend:
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((lx, 1.2), 2.6, 2.4, boxstyle="round,pad=0.1",
                 facecolor=fill, edgecolor=edge, lw=1.4, zorder=3))
    ax.text(lx + 3.2, 2.4, name, ha="left", va="center", fontsize=9, color="#3C4043")
    lx += 12.5

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = r"C:\Users\MSUSERSL123\Jaisri\BITS\smart_query_driven_surveillance_vlm\Docs\arch_diagram.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("SAVED ->", out)
