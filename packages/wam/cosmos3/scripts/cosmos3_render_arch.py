# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Render the Cosmos3-Nano-Policy-DROID module-flow + latency diagram from the measured profile.

Reads cosmos3_latency_profile.json (per-component inclusive GPU time + params, measured on
Strix Halo gfx1151 / ROCm 7.2.2 with CUDA-event forward hooks) and draws one annotated figure:
inputs -> Wan2.2 VAE encode -> fused MoT token sequence -> Qwen3-VL MoT backbone (denoise loop
x num_steps, CFG cond/uncond) -> action head + vision head -> Wan2.2 VAE decode (video path).

Action-path latencies are measured. The VAE-decode / video-path numbers are annotated from
artifacts/cosmos3_decode_bottleneck (the gfx1151 conv3d hang analysis) and open-loop eval from
openloop_metrics.json -- both are constants below with provenance.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cosmos3_latency_profile.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "cosmos3_arch_latency.png")

d = json.load(open(JSON))
comp = {c["name"]: c for c in d["components"]}
hot = {h["name"]: h for h in d["hotspots"]}
cfg = d["cfg"]
num_steps = d["num_steps"]

net = comp.get("net", {"incl_ms": 0, "calls": 1, "params": 0})
net_s = net["incl_ms"] / 1000.0
net_calls = max(net["calls"], 1)                       # CFG cond/uncond x steps (=8 for 4 steps)
per_call_s = net_s / net_calls
per_step_s = net_s / num_steps
lm_s = hot.get("net.language_model.model", {}).get("self_s", 0.0)
te_s = hot.get("net.time_embedder.mlp.2", {}).get("self_s", 0.0)
action_total_s = d["clean_action_total_s"]
vae_enc_s = max(action_total_s - net_s, 0.0)           # VAE encode of the conditioning frame + I/O
vram = d["infer_peak_vram_gb"]
total_B = d["total_params"] / 1e9

# ---- params of the small net sub-modules (from module_tree) ----
tree = {c["name"]: c for c in d["module_tree"][0]["children"]} if d.get("module_tree") else {}
def p(name, default=0):
    return tree.get(name, {}).get("params", default)
lm_B = p("language_model") / 1e9

# ---- annotated constants (provenance in docstring) ----
DECODE_EAGER_S = 253.7          # full Wan2.2 3D decode, bf16 eager, gfx1151 (decode analysis)
DECODE_CONV_S = 32.7            # single worst conv (up_blocks.3.resnets.0.conv1), 92% of conv3d
TINY_DECODE_S = 0.5            # taew2_2 tiny Conv2D bypass (target)
EVAL_RMSE = 0.202               # open-loop overall RMSE (raw), openloop_metrics.json
EVAL_LAT_P50 = 26.2             # open-loop p50 latency (s)
EVAL_COLD = 75.2                # cold first-call latency (s)
EVAL_QUERIES = 60

C = {
    "in": "#5b6472", "vae": "#2e8b57", "proj": "#7a5aa6", "mot": "#5a4fcf",
    "time": "#d08326", "head": "#2b8a9e", "dec": "#2e8b57", "loop": "#c0392b",
    "panel": "#f4f5f7", "edge": "#2b2f36", "fuse": "#3b4252",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, ax = plt.subplots(figsize=(21, 13.5))
ax.set_xlim(0, 120); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, title, body=None, tc="white", fs=11, bfs=9.3, alpha=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=C["edge"], lw=1.4, alpha=alpha, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", color=tc,
            fontsize=fs, fontweight="bold", zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", color=tc,
                fontsize=bfs, zorder=3, linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None, color=C["edge"], lw=2.0, ls="-", lc=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.3, label, ha="center", va="bottom",
                fontsize=8.4, color=lc or color, fontweight="bold", zorder=4)


def secs(s):
    return f"{s:.2f} s" if s >= 1 else f"{s*1000:.0f} ms"


# ---- title ----
ax.text(60, 98.4, "Cosmos3-Nano-Policy-DROID  \u00b7  15.2B Mixture-of-Transformers World-Action Model  \u2014  module flow & measured latency",
        ha="center", va="top", fontsize=16, fontweight="bold", color="#1a1d22")
ax.text(60, 94.8,
        f"concat view {cfg['image_height']}\u00d7{cfg['image_width']} (wrist / L / R) + {cfg['action_dim']}-d proprio + text  \u00b7  "
        f"UniPC {num_steps} steps, CFG guidance {cfg['guidance']:.0f} \u2192 {net_calls} net calls  \u00b7  action chunk {cfg['action_chunk_size']}\u00d7{cfg['action_dim']}  \u00b7  "
        f"AMD Strix Halo gfx1151 / ROCm 7.2.2 \u00b7 bf16",
        ha="center", va="top", fontsize=10.5, color="#454b54")

# ============================ FLOW ============================
# Inputs
box(1, 78, 15, 8.5, C["in"], "3 camera frames", f"wrist + L + R\nconcat {cfg['image_height']}\u00d7{cfg['image_width']}", fs=10.5)
box(1, 68, 15, 7.5, C["in"], "Proprioception", "7 joint + 1 gripper", fs=10.5)
box(1, 57.5, 15, 8, C["in"], "Text prompt", '"pour the cup\ninto the bowl"', fs=10.0, bfs=8.2)

# Wan2.2 VAE encoder
box(19, 74.5, 18, 12, C["vae"], "Wan2.2 VAE encode",
    f"AutoencoderKLWan\nz=48, 16\u00d7 spatial / 4\u00d7 temporal\nconditioning frame \u2192 latent\n{secs(vae_enc_s)}  (incl. I/O)")
# projections in
box(19, 58, 18, 12, C["proj"], "Token adapters",
    f"vae2llm [{p('vae2llm')/1e6:.1f}M]  vision\u2192tok\n"
    f"action2llm [{p('action2llm')/1e6:.1f}M]  proprio\u2192tok\n"
    f"(DomainAwareLinear)")

arrow(16, 82, 19, 81, color=C["in"])
arrow(16, 71.5, 19, 66, color=C["in"], rad=-0.1)
arrow(16, 61.5, 27, 58, color=C["in"], rad=-0.1)

# Fused token sequence
box(40.5, 60, 12.5, 22, C["fuse"], "Fused token\nsequence",
    "video latent tok\n+ action tok\n+ proprio tok\n+ text tok\n\n(one MoT stream)", fs=11, bfs=9.0)
arrow(37, 80, 42, 78, color=C["vae"], lc=C["vae"], rad=-0.1)
arrow(37, 64, 41, 68, color=C["proj"], rad=-0.15)

# MoT backbone
box(56, 45, 30, 40, C["mot"], "Qwen3-VL MoT backbone",
    f"language_model  [{lm_B:.1f}B]  \u00b7  36 layers\nMoE experts: und + gen (mlp_moe / mlp_moe_gen)",
    fs=12.5, bfs=9.3)
for i, (t, b) in enumerate([
        ("Self-Attn + RoPE", "packed und+gen"),
        ("Cross to text", "prompt K/V"),
        ("MoE FFN (gen)", "mlp_moe_gen experts"),
        ("AdaLN modulation", "timestep e")]):
    bx = 58 + (i % 2) * 14.5
    by = 70 - (i // 2) * 9
    box(bx, by, 13.2, 7.6, "#8079e6", t, b, fs=9.6, bfs=8.0)
ax.text(71, 52.6, f"per step {secs(per_step_s)} (CFG \u00d72)  \u00b7  per net call {secs(per_call_s)}",
        ha="center", va="center", color="white", fontsize=10.0, fontweight="bold")
ax.text(71, 48.6, f"language_model = {100*lm_s/net_s:.0f}% of net compute",
        ha="center", va="center", color="#e6e3ff", fontsize=9.4, fontweight="bold")

# time embedder
box(40.5, 45, 12.5, 11, C["time"], "Timestep embed",
    f"[{p('time_embedder')/1e6:.1f}M]\nsin \u2192 MLP\n{secs(te_s)}", fs=10.2, bfs=8.8)
arrow(53, 51, 56, 55, color=C["time"], rad=0.1)

# fused -> MoT
arrow(53, 71, 56, 69, color=C["fuse"], lw=2.6)

# denoise loop
ax.add_patch(FancyArrowPatch((84, 85), (58, 85), arrowstyle="-|>", mutation_scale=16,
             lw=2.4, color=C["loop"], connectionstyle="arc3,rad=-0.45", zorder=3))
ax.text(71, 90.5, f"\u00d7 {num_steps} UniPC denoise steps  (\u00d72 CFG cond/uncond = {net_calls} net calls)  \u2014  {secs(net_s)} total",
        ha="center", va="center", color=C["loop"], fontsize=10.6, fontweight="bold")

# Heads
hx = 90
box(hx, 72, 17, 13, C["head"], "Action head",
    f"llm2action [{p('llm2action')/1e6:.1f}M]\ntok \u2192 action chunk\n{cfg['action_chunk_size']}\u00d7{cfg['action_dim']} joint-pos\n(deployed output)", fs=10.6, bfs=8.8)
arrow(86, 70, 93, 72, color=C["mot"])
box(hx, 60, 17, 9, C["proj"], "Vision head",
    f"llm2vae [{p('llm2vae')/1e6:.1f}M]\ntok \u2192 video latent", fs=10.2, bfs=8.6)
arrow(86, 62, 93, 64, color=C["mot"], rad=0.05)

# VAE decode (video path, dashed)
box(hx, 45.5, 17, 12.5, C["dec"], "Wan2.2 VAE decode",
    f"3D causal conv \u00b7 video path\n{secs(DECODE_EAGER_S)} eager \u00b7 bf16 HANG*\n\u2192 tiny Conv2D bypass {secs(TINY_DECODE_S)}",
    ls=(0, (5, 3)), bfs=8.4)
arrow(98.5, 60, 98.5, 58, color=C["dec"], ls=(0, (5, 3)))
box(hx, 35.5, 17, 8, C["fuse"], "Imagined video", "GT | pred montage\n(offline)", fs=10.2, bfs=8.6, alpha=0.9)
arrow(98.5, 45.5, 98.5, 43.5, color=C["dec"], ls=(0, (5, 3)))

# ============================ BOTTOM PANEL ============================
ax.add_patch(FancyBboxPatch((1, 2), 118, 29, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc=C["panel"], ec=C["edge"], lw=1.3, zorder=0))
ax.text(2.5, 29.4, "Measured per-component runtime  (CUDA-event forward hooks, inclusive GPU time, Strix Halo gfx1151 / ROCm 7.2.2, bf16)",
        ha="left", va="top", fontsize=12, fontweight="bold", color="#1a1d22")

rows = [
    ("Component", "class", "params", "calls", "GPU time", "% action path"),
    ("Wan2.2 VAE encode + I/O", "AutoencoderKLWan", "~", "\u00d71", f"{secs(vae_enc_s)}", f"{100*vae_enc_s/action_total_s:.1f}%"),
    ("MoT net (denoise, CFG)", "Cosmos3VFMNetwork", f"{total_B:.2f}B", f"\u00d7{net_calls}", f"{secs(net_s)}", f"{100*net_s/action_total_s:.1f}%"),
    ("  \u2514 language_model (MoT)", "Qwen3VLTextForCausalLM", f"{lm_B:.2f}B", f"\u00d7{net_calls}", f"{secs(lm_s)}", f"{100*lm_s/action_total_s:.1f}%"),
    ("  \u2514 time_embedder", "TimestepEmbedder", f"{p('time_embedder')/1e6:.1f}M", f"\u00d7{2*num_steps*2}", f"{secs(te_s)}", f"{100*te_s/action_total_s:.1f}%"),
    ("  \u2514 adapters + action head", "(Domain)Linear \u00d74", f"{(p('vae2llm')+p('llm2vae')+p('action2llm')+p('llm2action'))/1e6:.1f}M", "\u2014", "<0.1 s", "<0.5%"),
    ("Wan2.2 VAE decode (video)", "AutoencoderKLWan (3D)", f"{total_B:.2f}B*", "\u00d71", f"{secs(DECODE_EAGER_S)}*", "video-path only"),
]
colx = [3, 34, 62, 74, 86, 101]
y0 = 25.2
for r, row in enumerate(rows):
    yy = y0 - r * 3.05
    if r == 0:
        ax.add_patch(plt.Rectangle((2, yy - 0.8), 116, 2.7, fc="#dfe3e8", ec="none", zorder=0))
    for cxi, cx in enumerate(colx):
        ax.text(cx, yy, str(row[cxi]), ha="left", va="center",
                fontsize=9.6, fontweight="bold" if r == 0 else "normal", color="#1a1d22")

ax.text(2.5, 5.7,
        f"Deployed action path:  {action_total_s:.1f} s  ({num_steps} UniPC steps, CFG) \u00b7 per step {per_step_s:.2f} s \u00b7 no VAE decode  |  "
        f"cold first call {EVAL_COLD:.0f} s  |  peak VRAM {vram:.1f} GB  |  total params {total_B:.2f}B ({lm_B:.1f}B in the MoT)",
        ha="left", va="center", fontsize=9.6, color="#1a1d22", fontweight="bold")
ax.text(2.5, 3.5,
        f"Open-loop DROID eval: overall RMSE {EVAL_RMSE:.3f} (raw joint-pos) over {EVAL_QUERIES} queries, p50 {EVAL_LAT_P50:.1f} s/call.   "
        f"*VAE decode (video path only): Wan2.2 3D conv HANGS in bf16/fp16 on gfx1151 (MIOpen); worst conv {DECODE_CONV_S:.0f} s = 92%. "
        f"Fix = tiny Conv2D decoder (taew2_2), off the action path.",
        ha="left", va="center", fontsize=8.6, color="#454b54", style="italic")

fig.savefig(OUT, dpi=145, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
