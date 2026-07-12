"""Why the two-stage keep-0.25 run jumped 84% -> 96%: it is the FastV cut depth (L0 3->9).
Three panels: (A) per-layer LLM token load schematic, (B) per-suite accuracy shift,
(C) compute/latency vs accuracy trade. Data from FEEDBACK_PRUNING_LORA_REPORT + pruning_stats."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join("artifacts", "predict_eval", "l0_3_vs_9_explained.png")

N_LAYERS = 36
FULL = 479      # LLM seq before cut (87 text + 392 image)
RED = 185       # after cut, keep 0.25 (87 text + 98 image)
C_FULL = "#d1495b"   # expensive: full-token layer
C_RED = "#3f9b52"    # cheap: reduced-token layer

def tl(l0):  # token-layer compute proxy
    return l0 * FULL + (N_LAYERS - l0) * RED

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 1.0], wspace=0.34)

# ---- Panel A: per-layer LLM token load ----
axA = fig.add_subplot(gs[0, 0])
for col, (l0, tag, acc) in enumerate([(3, "L0=3\n(prev, 84.0%)", 84.0),
                                      (9, "L0=9\n(now, 96.0%)", 96.0)]):
    x0 = col * 1.3
    for layer in range(N_LAYERS):
        full = layer < l0
        w = (FULL if full else RED) / FULL * 0.9
        axA.barh(layer, w, left=x0, height=0.85,
                 color=C_FULL if full else C_RED, edgecolor="white", linewidth=0.2)
    axA.text(x0 + 0.45, N_LAYERS + 1.5, tag, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.axhline(l0 - 0.5, xmin=(x0)/ (1.3*2+1.0), xmax=1, color="k", lw=0)  # placeholder
    # cut annotation
    axA.annotate(f"cut @L0={l0}", (x0 + 0.95, l0 - 0.5), fontsize=8, color="k",
                 xytext=(x0 + 1.02, l0 - 0.5), va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.8))
axA.set_ylim(-1, N_LAYERS + 4)
axA.set_xlim(-0.1, 3.1)
axA.set_ylabel("LLM decoder layer  (0 = first)")
axA.set_xticks([]);
axA.set_title("A. Per-layer LLM token load\n(bar width = tokens processed)", fontsize=10)
axA.legend(handles=[Patch(color=C_FULL, label=f"full seq ({FULL} tok)"),
                    Patch(color=C_RED, label=f"pruned seq ({RED} tok, keep 0.25)")],
           loc="center right", fontsize=8, ncol=1, framealpha=0.9)

# ---- Panel B: per-suite accuracy shift ----
axB = fig.add_subplot(gs[0, 1])
suites = ["spatial", "object", "goal", "long-10", "AVG"]
l3 = [77, 92, 83, 84, 84.0]
l9 = [94, 100, 95, 95, 96.0]
x = range(len(suites)); w = 0.38
b1 = axB.bar([i - w/2 for i in x], l3, w, label="L0=3 (84.0%)", color="#8c9bb0")
b2 = axB.bar([i + w/2 for i in x], l9, w, label="L0=9 (96.0%)", color=C_RED)
for bars in (b1, b2):
    for b in bars:
        axB.annotate(f"{b.get_height():.0f}", (b.get_x()+b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=8)
for i in x:
    axB.annotate(f"+{l9[i]-l3[i]:.0f}", (i, max(l9[i], l3[i]) + 3.5), ha="center",
                 fontsize=8, color=C_FULL, fontweight="bold")
axB.set_xticks(list(x)); axB.set_xticklabels(suites)
axB.set_ylim(60, 108); axB.set_ylabel("closed-loop success %")
axB.set_title("B. Where the +12 pts come from\n(gains concentrate in wide-context suites)", fontsize=10)
axB.legend(loc="lower right", fontsize=8); axB.grid(axis="y", alpha=0.25)

# ---- Panel C: compute vs accuracy trade ----
axC = fig.add_subplot(gs[0, 2])
labels = ["dense", "L0=3\n(84%)", "L0=9\n(96%)"]
llm_saved = [0, 100*(1 - tl(3)/tl(0)), 100*(1 - tl(9)/tl(0))]  # % LLM token-layer saved vs dense (l0=0 -> all reduced? use full dense)
# dense = all 36 layers full
dense_tl = N_LAYERS * FULL
llm_saved = [0, 100*(1 - tl(3)/dense_tl), 100*(1 - tl(9)/dense_tl)]
strix = [343.6, 156.7, 170.0]   # prunable-stack ms (ViT+LLM); L0=9 estimated
acc = [None, 84.0, 96.0]
xc = range(len(labels))
bars = axC.bar(xc, strix, color=["#b0b0b0", "#8c9bb0", C_RED], width=0.6)
for i, b in enumerate(bars):
    txt = f"{strix[i]:.0f} ms"
    if i > 0:
        txt += f"\n(-{100*(1-strix[i]/strix[0]):.0f}% vs dense)\nLLM saved {llm_saved[i]:.0f}%"
    axC.annotate(txt, (b.get_x()+b.get_width()/2, b.get_height()+4), ha="center", va="bottom", fontsize=8)
axC.set_xticks(list(xc)); axC.set_xticklabels(labels)
axC.set_ylim(0, 400); axC.set_ylabel("Strix-Halo prunable stack (ViT+LLM) latency, ms")
axC.set_title("C. Cost of the deeper cut\n~9% slower stack  ->  +12 pts accuracy", fontsize=10)
axC.grid(axis="y", alpha=0.25)
axC.annotate("", xy=(2, 300), xytext=(1, 300),
             arrowprops=dict(arrowstyle="->", color=C_FULL, lw=1.6))
axC.text(1.5, 308, "+12 pts\n+~9% latency", ha="center", color=C_FULL, fontsize=8, fontweight="bold")

fig.suptitle("Two-stage keep-0.25: why moving the FastV cut from L0=3 to L0=9 lifts 84% -> 96% "
             "(ViT stage identical; only the LLM cut depth changes)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=130)
print("wrote", OUT, "| token-layer L0=3:", tl(3), "L0=9:", tl(9), "dense:", dense_tl)
