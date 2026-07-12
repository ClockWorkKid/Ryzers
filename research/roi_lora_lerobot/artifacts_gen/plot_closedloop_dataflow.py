"""Closed-loop inference dataflow, drawn exactly as the deployed eval overlay executes
(research/roi_lora_lerobot/docker/roi_overlay/{molmoact2_hf_model/,}modeling_molmoact2.py).

Two figures:
  1) standard  (select=gate_distill / forward-pass gate): single frame, single pass.
  2) predictive(select=gate_predict): gate scored on frame t prunes frame t+H (causal carry).

Both prove: ONE forward pass, NO action-attention teacher at eval, NO LLM attention
rollout for FastV (keep-set reuses the ViT gate scores). Token counts: keep 0.25 example
(ViT seam=6, patches 729->364 @50%; LLM seq 479->185 @25% img keep, FastV L0).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join("artifacts", "predict_eval")
os.makedirs(OUT_DIR, exist_ok=True)

C_FULL = "#d1495b"   # runs on FULL token set (expensive)
C_RED = "#3f9b52"    # runs on REDUCED token set (cheap)
C_GATE = "#2e5eaa"   # the learned gate (cheap MLP)
C_DATA = "#6c757d"   # observation / passive tensors
C_ACT = "#8e44ad"    # action output
C_BG_VIT = "#eef4fb"
C_BG_LLM = "#fdf2ec"


def box(ax, x, y, w, h, text, fc, ec="black", tc="white", fs=8.2, lw=1.0, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, zorder=4, wrap=True)


def arrow(ax, p0, p1, color="black", lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# ============================ FIGURE 1: STANDARD ============================
def standard():
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    # lane backgrounds
    ax.add_patch(FancyBboxPatch((1, 24), 98, 17, boxstyle="round,pad=0.2", fc=C_BG_VIT, ec="#c7d8ee", zorder=0))
    ax.add_patch(FancyBboxPatch((1, 3.5), 98, 16.5, boxstyle="round,pad=0.2", fc=C_BG_LLM, ec="#eecab6", zorder=0))
    ax.text(2.4, 39.3, "STAGE 1 — Vision (ViT), seam=6", fontsize=9.5, color="#2e5eaa", fontweight="bold")
    ax.text(2.4, 18.3, "STAGE 2 — LLM decoder (36 layers) + flow-matching action expert",
            fontsize=9.5, color="#b5581f", fontweight="bold")

    y1 = 29.5; h = 7.5
    box(ax, 2.5, y1, 9, h, "obs frame $t$\n(image crops)\n729 patches/crop", C_DATA)
    box(ax, 14, y1, 10.5, h, "patch-embed\n+ pos-emb\n(729 patches)", C_DATA)
    box(ax, 27, y1, 12, h, "ViT blocks 0..5\non ALL 729\n(seam, FULL)", C_FULL)
    box(ax, 41.5, y1, 13, h, "GATE MLP\n(task-conditioned)\nscores 729 patches", C_GATE)
    box(ax, 57, y1, 11.5, h, "top-k keep\n364 / 729\n(50%)", C_GATE)
    box(ax, 71, y1, 12.5, h, "ViT blocks 6..N\non 364\n(REDUCED)", C_RED)
    box(ax, 86, y1, 11.5, h, "scatter-back\n-> pool -> 392\nLLM img tokens", C_DATA)

    for x0, x1 in [(11.5, 14), (24.5, 27), (39, 41.5), (54.5, 57), (68.5, 71), (83.5, 86)]:
        arrow(ax, (x0, y1 + h / 2), (x1, y1 + h / 2))

    # drop from stage1 to stage2
    arrow(ax, (91.7, y1), (91.7, 20.2), color="#444", lw=2.0)
    ax.text(93.2, 24.7, "seq=479\n(87 txt+392 img)", fontsize=7.5, color="#444", va="center")

    y2 = 8.0
    box(ax, 70.5, y2, 27, h, "LLM layers 0..L0-1\non FULL seq 479\n(FULL)", C_FULL)
    box(ax, 40, y2, 26, h, "FastV cut @ L0\nreuse GATE scores -> drop 294 img\nseq 479 -> 185", C_GATE)
    box(ax, 12.5, y2, 24, h, "LLM layers L0..35\n+ action cross-attn\non 185 (REDUCED)", C_RED)
    box(ax, 2.5, y2, 8, h, "flow\ndenoise", C_ACT)

    arrow(ax, (70.5, y2 + h / 2), (66, y2 + h / 2))
    arrow(ax, (40, y2 + h / 2), (36.5, y2 + h / 2))
    arrow(ax, (12.5, y2 + h / 2), (10.5, y2 + h / 2))
    arrow(ax, (6.5, y2), (6.5, 3.4), color=C_ACT, lw=2.0)
    box(ax, 2.5, 0.0, 20, 3.2, "ACTION CHUNK  ->  executed in env  ->  next obs frame $t{+}H$", C_ACT, fs=8.5)

    # the key reuse arrow: gate scores drive BOTH prunes, computed ONCE
    arrow(ax, (48, y1), (53, y2 + h), color=C_GATE, lw=2.0, rad=-0.25, style="-|>")
    ax.text(64, 22.0, "same gate scores reused for the FastV keep-set\n"
                      "(no re-scoring, no LLM attention rollout)",
            fontsize=8.2, color=C_GATE, ha="center", fontweight="bold")

    ax.text(50, 43.0, "Closed-loop inference — STANDARD (forward-pass gate, select=gate_distill)",
            fontsize=12.5, ha="center", fontweight="bold")
    ax.text(50, 41.4, "ONE forward pass per replan · NO action-attention teacher at eval · "
                      "FastV keep-set = ViT gate scores (output_attentions=False)",
            fontsize=9, ha="center", color="#222")

    # legend (bottom row, right of the action-chunk box)
    for i, (c, t) in enumerate([(C_FULL, "runs on FULL tokens"), (C_RED, "runs on REDUCED tokens"),
                                (C_GATE, "learned gate (cheap MLP)"), (C_DATA, "tensors / obs"),
                                (C_ACT, "action output")]):
        box(ax, 26 + i * 14.5, 0.7, 1.8, 1.6, "", c, tc="white")
        ax.text(28.2 + i * 14.5, 1.5, t, fontsize=7.4, va="center")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "closed_loop_standard.png")
    fig.savefig(p, dpi=135); plt.close(fig); print("wrote", p)


# ============================ FIGURE 2: PREDICTIVE ============================
def predictive():
    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    ax.text(50, 44.6, "Closed-loop inference — PREDICTIVE / CAUSAL (select=gate_predict)",
            fontsize=12.5, ha="center", fontweight="bold")
    ax.text(50, 43.0, "The gate NEVER sees the frame it prunes: frame $t$ is pruned by the keep-set "
                      "the gate predicted at replan $t{-}H$ (its past).",
            fontsize=9, ha="center", color="#222")

    # three replans as columns
    cols = [(6, "replan $t{-}H$"), (39, "replan $t$"), (72, "replan $t{+}H$")]
    cw = 22; y = 20; h = 15
    for cx, lab in cols:
        ax.add_patch(FancyBboxPatch((cx, y), cw, h, boxstyle="round,pad=0.2", fc="#f4f6f8", ec="#c9d2db", zorder=0))
        ax.text(cx + cw / 2, y + h - 1.2, lab, fontsize=10, ha="center", fontweight="bold", color="#333")
        # obs
        box(ax, cx + 1.5, y + h - 5.2, cw - 3, 3.0, "obs frame", C_DATA, fs=8)
        # pipeline block (single pass)
        box(ax, cx + 1.5, y + 5.2, cw - 3, 5.2,
            "gate+FastV pipeline\n(ONE forward pass)\nViT 729->364 · LLM 479->185", C_RED, fs=7.8)
        # gate score output
        box(ax, cx + 1.5, y + 1.0, cw - 3, 3.0, "gate scores THIS frame\n-> top-k keep-set", C_GATE, fs=7.6)
        arrow(ax, (cx + cw / 2, y + h - 5.2), (cx + cw / 2, y + 10.4), lw=1.5)
        arrow(ax, (cx + cw / 2, y + 5.2), (cx + cw / 2, y + 4.0), color=C_GATE, lw=1.5)
        # action out
        box(ax, cx + 1.5, y - 4.6, cw - 3, 3.0, "action chunk -> env", C_ACT, fs=7.8)
        arrow(ax, (cx + cw / 2, y + 5.2), (cx + cw / 2, y - 1.6), color=C_ACT, lw=1.5, rad=0.0)

    # CARRY arrows: gate scores at t-H -> prune ViT at t ; gate scores at t -> prune at t+H
    arrow(ax, (6 + cw - 2, y + 2.5), (39 + 2, y + 7.8), color=C_GATE, lw=2.4, rad=-0.28)
    arrow(ax, (39 + cw - 2, y + 2.5), (72 + 2, y + 7.8), color=C_GATE, lw=2.4, rad=-0.28)
    ax.text(30, y + 12.4, "carry keep-set (t-H) -> prunes frame $t$", fontsize=8.2, color=C_GATE,
            ha="center", fontweight="bold")
    ax.text(63, y + 12.4, "carry keep-set ($t$) -> prunes frame $t{+}H$", fontsize=8.2, color=C_GATE,
            ha="center", fontweight="bold")

    # implementation note box
    ax.add_patch(FancyBboxPatch((3, 2.5), 94, 11.5, boxstyle="round,pad=0.3", fc="#f7f7ef", ec="#d9d9b0", zorder=0))
    ax.text(5, 12.8, "How the carry is implemented (predict_action_chunk, @torch.no_grad):", fontsize=9,
            fontweight="bold", color="#555")
    notes = [
        "1.  vb.roi_external_keep_idx = self._roi_pred_prev_keep_idx   # keep-set predicted at t-H prunes the CURRENT ViT (gather_keep, single pass)",
        "2.  encode_image scores the gate on frame t's seam features too, but ONLY to stash roi_last_gate_scores (it does NOT re-prune frame t)",
        "3.  FastV cut @L0 reuses those same stashed gate scores for the LLM keep-set  (no teacher, output_attentions=False)",
        "4.  after the pass:  self._roi_pred_prev_keep_idx = top-k(roi_last_gate_scores)   # becomes the keep-set for replan t+H",
        "5.  reset() sets _roi_pred_prev_keep_idx=None between episodes -> episode's FIRST replan falls back to scoring its own frame",
    ]
    for i, n in enumerate(notes):
        ax.text(5, 11.2 - i * 1.75, n, fontsize=8.0, color="#222", family="monospace")

    # legend
    for i, (c, t) in enumerate([(C_RED, "single-pass pipeline"), (C_GATE, "gate / carried keep-set"),
                                (C_DATA, "obs"), (C_ACT, "action")]):
        box(ax, 5 + i * 20, 40.0, 2.0, 1.5, "", c)
        ax.text(7.3 + i * 20, 40.75, t, fontsize=8, va="center")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "closed_loop_predictive.png")
    fig.savefig(p, dpi=135); plt.close(fig); print("wrote", p)


standard()
predictive()
