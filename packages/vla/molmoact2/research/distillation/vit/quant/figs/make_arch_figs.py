"""Publication-style architecture figures for the MolmoAct2 vision-tower
distillation + W4A6 quantization work.

Fig 1  teacher (SigLIP2-SO400M ViT) vs the two students (CNN, TinyViT), with the
       shared frozen downstream head and the distillation seam objective.
Fig 2  the W4A6 fake-quant of a student block (which ops are quantized vs stay FP)
       and the 4-stage distill -> PTQ -> QAT -> co-finetune pipeline.

Every hyperparameter/param count is sourced from research/vit_distill/distill/
{config.py,student.py,teacher.py} and quant/quant_student.py. Renders to
artifacts/arch/ (PNG + PDF). Pure matplotlib, no network.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# --------------------------------------------------------------------------- #
# Verified specs (single source of truth for the figures)
# --------------------------------------------------------------------------- #
TEACHER = dict(
    name="Teacher: MolmoAct2 SigLIP2-SO400M/14 ViT (frozen)",
    params="\u2248 0.40 B", dtype="bf16",
    rows=[
        ("Input crop", "378\u00d7378\u00d73  RGB (agentview / wrist)"),
        ("Patchify", "patch 14 \u2192 27\u00d727 = 729 tokens \u00b7 588 px/patch \u00b7 no CLS"),
        ("Patch embed", "Linear 588\u21921152  +  learned pos(729)"),
        ("25\u00d7 Encoder block", "MHSA 16 heads\u00d772  \u00b7  MLP 1152\u21924304\u21921152  \u00b7  2\u00d7LN (pre-norm)"),
        ("Seam tap", "concat ViT layers \u22123 & \u22129  \u2192  729\u00d72304"),
    ],
    color="#dbe7f3", edge="#3f6fa3",
)

CNN = dict(
    name="Student A \u00b7 CNN  (FPGA-first, local RF)",
    params="3.83 M", dtype="fp32 / bf16",
    rows=[
        ("Input patches", "729\u00d7588   (identical contract to teacher)"),
        ("Stem", "Linear 588\u2192384  +  pos(729)"),
        ("16\u00d7 ConvBlock", "LN \u2192 DWConv 3\u00d73 (groups=384) \u2192 PW Linear 384\u2192384 \u2192 GELU  (residual)"),
        ("Head", "LN \u2192 Linear 384\u21922304"),
        ("Seam out", "729\u00d72304   (matches teacher)"),
    ],
    color="#dcefe0", edge="#3f8f5c",
)

TINY = dict(
    name="Student B \u00b7 TinyViT  (global attn, teacher-like)",
    params="2.73 M", dtype="fp32 / bf16",
    rows=[
        ("Input patches", "729\u00d7588   (identical contract to teacher)"),
        ("Stem", "Linear 588\u2192240  +  pos(729)"),
        ("4\u00d7 AttnBlock", "LN \u2192 MHSA 8 heads\u00d730 \u2192 proj  \u00b7  LN \u2192 MLP 240\u2192480\u2192240 (ratio 2.0)"),
        ("Head", "LN \u2192 Linear 240\u21922304"),
        ("Seam out", "729\u00d72304   (matches teacher)"),
    ],
    color="#faecd7", edge="#c8892b",
)

SHARED = ("Shared frozen downstream (unchanged for every encoder):   "
          "2\u00d72 attention pool  \u2192  vision\u2192LLM projector  \u2192  196 tokens \u00d7 2560  "
          "\u2192  LLM (36 decoder layers)  +  flow-matching action expert")

FROZEN_C = "#ececec"; FROZEN_E = "#8a8a8a"
ACCENT = "#b3202a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.0,
})


def stack_box(ax, cx, top, w, spec, row_h=8.4, gap=1.6, title_h=7.2):
    """Draw a titled vertical stack of sub-rows; return (left, right, bottom)."""
    n = len(spec["rows"])
    total_h = title_h + n * row_h + (n - 1) * gap + 4
    left, bottom = cx - w / 2, top - total_h
    # outer card
    ax.add_patch(FancyBboxPatch(
        (left, bottom), w, total_h, boxstyle="round,pad=0.6,rounding_size=2.2",
        linewidth=1.6, edgecolor=spec["edge"], facecolor="white", zorder=1))
    # title band
    ax.add_patch(FancyBboxPatch(
        (left + 1.2, top - title_h - 1.2), w - 2.4, title_h,
        boxstyle="round,pad=0.2,rounding_size=1.5",
        linewidth=0, facecolor=spec["edge"], zorder=2))
    ax.text(cx, top - title_h / 2 - 1.2, spec["name"], ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(left + w - 2.4, top - title_h - 3.9, f"params {spec['params']}  \u00b7  {spec['dtype']}",
            ha="right", va="center", fontsize=8.2, style="italic",
            color=spec["edge"], zorder=3)
    y = top - title_h - 5.4
    for i, (k, v) in enumerate(spec["rows"]):
        ax.add_patch(FancyBboxPatch(
            (left + 2.2, y - row_h), w - 4.4, row_h,
            boxstyle="round,pad=0.15,rounding_size=1.2",
            linewidth=1.0, edgecolor=spec["edge"], facecolor=spec["color"], zorder=2))
        ax.text(left + 4.2, y - row_h / 2 + 1.3, k, ha="left", va="center",
                fontsize=8.6, fontweight="bold", zorder=3)
        ax.text(left + 4.2, y - row_h / 2 - 1.9, v, ha="left", va="center",
                fontsize=7.7, zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(cx, y - row_h - gap + 0.2), xytext=(cx, y - row_h - 0.2),
                        arrowprops=dict(arrowstyle="-|>", color=spec["edge"], lw=1.1), zorder=2)
        y -= row_h + gap
    return left, left + w, bottom


def fig1(outdir):
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 190); ax.set_ylim(0, 112); ax.axis("off")

    ax.text(95, 108.5, "Vision-tower distillation: one frozen teacher \u2192 two lightweight students",
            ha="center", va="center", fontsize=15.5, fontweight="bold")
    ax.text(95, 104.3, "identical I/O contract (729\u00d7588 patches \u2192 729\u00d72304 seam)  \u00b7  "
            "students are \u2248100\u2013150\u00d7 smaller and feed the same frozen head",
            ha="center", va="center", fontsize=9.8, style="italic", color="#444")

    top = 99
    tcx, ccx, ycx = 32, 102, 160
    tl, tr, tb = stack_box(ax, tcx, top, 56, TEACHER)
    cl, cr, cb = stack_box(ax, ccx, top, 56, CNN)
    yl, yr, yb = stack_box(ax, ycx, top, 52, TINY)

    # distillation objective: dashed links teacher seam <-> student seams via a loss tag
    lbx0, lbx1, lby0, lby1 = 51, 81, 23.5, 31
    ax.add_patch(FancyBboxPatch((lbx0, lby0), lbx1 - lbx0, lby1 - lby0,
                                boxstyle="round,pad=0.3,rounding_size=1.5",
                                linewidth=1.3, edgecolor=ACCENT, facecolor="#fbeaec", zorder=5))
    ax.text((lbx0 + lbx1) / 2, lby1 - 2.6, "Distillation seam loss",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color=ACCENT, zorder=6)
    ax.text((lbx0 + lbx1) / 2, lby0 + 3.3, "cosine + normalized-MSE\n(student vs frozen teacher, 729\u00d72304)",
            ha="center", va="center", fontsize=7.4, style="italic", color=ACCENT, zorder=6)
    ax.annotate("", xy=(lbx0, lby1 - 1.5), xytext=(tcx, tb - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                linestyle=(0, (5, 3)), connectionstyle="arc3,rad=-0.15"), zorder=4)
    for sx, rad in ((ccx, 0.12), (ycx, 0.10)):
        ax.annotate("", xy=(sx, tb - 0.5), xytext=(lbx1, lby1 - 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                    linestyle=(0, (5, 3)), connectionstyle=f"arc3,rad={rad}"), zorder=4)

    # shared frozen downstream band
    band_y, band_h = 6.5, 12.5
    ax.add_patch(FancyBboxPatch((8, band_y), 174, band_h, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.4, edgecolor=FROZEN_E, facecolor=FROZEN_C, zorder=1))
    ax.text(95, band_y + band_h - 3.1, "FROZEN \u00b7 SHARED",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color=FROZEN_E, zorder=2)
    ax.text(95, band_y + 4.6, SHARED, ha="center", va="center", fontsize=8.6, zorder=2)
    for sx in (tcx, ccx, ycx):
        ax.annotate("", xy=(sx, band_y + band_h), xytext=(sx, tb - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=FROZEN_E, lw=1.2), zorder=1)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_distillation.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 : quantization + pipeline
# --------------------------------------------------------------------------- #
QUANT_C = "#e7dcf3"; QUANT_E = "#6a3fa3"
FP_C = "#eef1f4"; FP_E = "#7a8794"


def op_box(ax, x, y, w, h, title, sub, fc, ec, tcolor="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.4",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=tcolor, zorder=3)
    ax.text(x + w / 2, y + h / 2 - 1.6, sub, ha="center", va="center",
            fontsize=7.4, color=tcolor, zorder=3)


def fig2(outdir):
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 180); ax.set_ylim(0, 112); ax.axis("off")

    ax.text(90, 108.5, "W4A6 fake-quantization (Brevitas) + QAT / LoRA / action-expert co-finetune",
            ha="center", va="center", fontsize=15.5, fontweight="bold")
    ax.text(90, 104.3, "4-bit per-channel weights, 6-bit per-tensor activations  \u00b7  "
            "LayerNorm & GELU stay floating-point (FINN streamlining)  \u00b7  fake-quant math in fp32 (STE)",
            ha="center", va="center", fontsize=9.6, style="italic", color="#444")

    # ---- Panel A : which ops get quantized -------------------------------- #
    ax.add_patch(FancyBboxPatch((6, 58), 168, 42, boxstyle="round,pad=0.5,rounding_size=2",
                                linewidth=1.3, edgecolor="#bbbbbb", facecolor="white", zorder=1))
    ax.text(10, 96.5, "A.  Student block under W4A6 fake-quant", ha="left", va="center",
            fontsize=11, fontweight="bold")

    # quantized ops
    op_box(ax, 12, 74, 44, 15, "QuantLinear   (stem / qkv / proj / MLP / head)",
           "W: int4  per-channel, float scale\nA(in): int6  per-tensor, signed", QUANT_C, QUANT_E)
    op_box(ax, 62, 74, 44, 15, "QuantConv2d   (depthwise 3\u00d73, CNN)",
           "W: int4  per-channel\nA(in): int6  per-tensor", QUANT_C, QUANT_E)
    op_box(ax, 112, 74, 56, 15, "QuantScaledDotProductAttention  (TinyViT)",
           "int6 signed on  q\u00b7k\u00b7v, softmax-in,\nattn-weights, sdpa-out  (QONNX-exportable)", QUANT_C, QUANT_E)

    # FP-kept ops
    op_box(ax, 12, 60, 44, 11.5, "LayerNorm  \u00b7  GELU", "kept floating-point", FP_C, FP_E, tcolor="#3a444d")
    op_box(ax, 62, 60, 44, 11.5, "pos embed  \u00b7  residual add", "kept floating-point", FP_C, FP_E, tcolor="#3a444d")
    op_box(ax, 112, 60, 56, 11.5, "Deploy target",
           "FINN dataflow \u2192 FPGA (streamlined int)", "#fff4d6", "#c8a02b", tcolor="#6a5410")

    # legend chips
    ax.add_patch(Rectangle((12, 91.4), 3, 2.4, facecolor=QUANT_C, edgecolor=QUANT_E, lw=1))
    ax.text(16, 92.6, "quantized (fake-quant, trainable via STE)", va="center", fontsize=8)
    ax.add_patch(Rectangle((92, 91.4), 3, 2.4, facecolor=FP_C, edgecolor=FP_E, lw=1))
    ax.text(96, 92.6, "kept floating-point", va="center", fontsize=8)

    # ---- Panel B : pipeline ---------------------------------------------- #
    ax.text(10, 52, "B.  Training pipeline  (what is trainable at each stage)",
            ha="left", va="center", fontsize=11, fontweight="bold")

    stages = [
        ("1 \u00b7 Distill (fp32)", "#dcefe0", "#3f8f5c",
         ["trainable: student ViT", "loss: seam cosine + nMSE", "teacher frozen"]),
        ("2 \u00b7 PTQ calibrate", "#eef1f4", "#7a8794",
         ["no training", "Brevitas sets W/A scales", "\u2192 W4A6 fake-quant"]),
        ("3 \u00b7 QAT recovery", "#e7dcf3", "#6a3fa3",
         ["trainable: fake-quant ViT (STE)", "loss: seam cosine + nMSE", "(+ downstream-consistency)", "data: LIBERO"]),
        ("4 \u00b7 Co-finetune", "#f7dede", "#b3202a",
         ["trainable: fake-quant ViT", "+ LoRA(VLM) + action expert", "frozen: base VLM", "loss: flow-matching (closed-loop)", "12k steps \u00b7 bf16"]),
    ]
    x0, w, h, y = 8, 36, 30, 14
    gap = (168 - len(stages) * w) / (len(stages) - 1)
    centers = []
    for i, (title, fc, ec, lines) in enumerate(stages):
        x = x0 + i * (w + gap)
        centers.append(x + w / 2)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.8",
                                    linewidth=1.6, edgecolor=ec, facecolor="white", zorder=2))
        ax.add_patch(FancyBboxPatch((x + 1, y + h - 6.4), w - 2, 5.6,
                                    boxstyle="round,pad=0.15,rounding_size=1.2",
                                    linewidth=0, facecolor=ec, zorder=3))
        ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="center",
                color="white", fontsize=9.2, fontweight="bold", zorder=4)
        for j, ln in enumerate(lines):
            ax.text(x + 2.4, y + h - 8.6 - j * 3.4, "\u2022 " + ln, ha="left", va="center",
                    fontsize=7.3, zorder=4)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.5, y + h / 2), xytext=(x + w + 0.5, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6), zorder=2)
    ax.annotate("", xy=(centers[-1] + w / 2 + 4.5, y + h / 2), xytext=(centers[-1] + w / 2 + 0.5, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#c8a02b", lw=1.8), zorder=2)
    ax.text(centers[-1] + w / 2 + 2.5, y + h / 2 + 3, "deploy", ha="center", fontsize=7.5,
            color="#6a5410", rotation=0)

    # closed-loop results chip
    ax.add_patch(FancyBboxPatch((8, 1.5), 164, 6.6, boxstyle="round,pad=0.3,rounding_size=1.5",
                                linewidth=1.2, edgecolor="#3f6fa3", facecolor="#eef4fa", zorder=1))
    ax.text(90, 5.7, "Closed-loop success  (LIBERO, 4 suites \u00d7 20 ep)",
            ha="center", fontsize=8.6, fontweight="bold", color="#2c527a")
    ax.text(90, 2.9, "W8A8 PTQ 97.5     \u00b7     W4A6 QAT:  CNN 95.0  /  TinyViT 98.8     \u00b7     "
            "+ co-finetune (stage 4) \u2192 target \u2265 99  (running)",
            ha="center", fontsize=8.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig2_quantization.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.abspath(os.path.join(here, "..", "..", "..", "..", "artifacts", "arch"))
    os.makedirs(outdir, exist_ok=True)
    fig1(outdir)
    fig2(outdir)
    print("wrote figures to", outdir)
    for f in sorted(os.listdir(outdir)):
        print("  ", f)


if __name__ == "__main__":
    main()
