# Run D — Curriculum LLM/VLM Distillation (MolmoAct2-LIBERO)

Fourth distillation arm (tracked alongside the three ViT variants: siglip_nano, hybrid_droid,
cnn_fpga). Goal: shrink the **language-model backbone** of MolmoAct2 by ~100× compute and,
combined with the already-distilled ViT, make the **full VLA pipeline much smaller** — verified
in closed-loop LIBERO.

## Teacher architecture (allenai/MolmoAct2-LIBERO)

| Part | Shape | Notes |
|---|---|---|
| ViT (SigLIP2) | 27 layers, 1152-d | seam `vit_layers=(-3,-9)` → `[*,729,2304]` (already distilled ~100×) |
| Connector | pool 1152 → projector 1152→2560 | fuses vision additively into `inputs_embeds` at `<im_patch>` |
| **LLM** | **36 layers × 2560-d**, 32 heads / 8 KV heads × 128, interm 9728 | OLMo/Molmo decoder: pre-norm RMSNorm, GQA, RoPE (θ=5e6), Qwen3 QK-norm, SwiGLU. **~5 B params** |
| Action expert | 768-d × **36 blocks** | flow-matching; **one block per LLM layer**, cross-attends to **per-layer LLM KV** (1024-d) |

**Hard constraint:** the action expert enforces `action.num_layers == llm.num_layers` and reads
**per-layer KV**, not just the final hidden. So the student LLM must (a) keep 36 layers and
(b) preserve per-layer KV content — otherwise the frozen action expert breaks.

## Student — "thin twin" (`student.py`)

Keep 36 layers; shrink width. Consumes the teacher's fused `inputs_embeds` [B,N,2560] via a
`2560→hidden` down-projection (embedding lookup + vision fusion are cheap and shared; the
36-layer stack is where the FLOPs are), and up-projects the final hidden `hidden→2560`.

| dim | value |
|---|---|
| hidden | **208** |
| layers | 36 |
| heads / kv_heads | 8 / 8 |
| head_dim | 26 (even → RoPE pairs; q_dim=kv_dim=208) |
| intermediate | 832 |
| **compression** | **112× decoder FLOPs** (`flops.py`, seq=512) |
| stack params | ~26 M (+ tiny projs) |

Distill heads (`DistillHeads`) project student features → teacher dims for regression; they are
**Stage-1-only** and dropped at deploy (deploy uses the final `up_proj` + retrained action-expert
KV projections).

## Two-stage curriculum

### Stage 1 — LLM-only (`train_stage1.py`, `losses.py`)
Freeze teacher. On real LIBERO batches, capture per-layer hidden + per-layer raw KV + final
hidden (on the fly — too large to dump), and regress:
- `hidden` : (1−cos)+MSE per layer (projected 208→2560)
- `kv`     : MSE on projected K and V per layer — **weighted 2×** (this is the action-expert conditioning path)
- `final`  : (1−cos)+MSE on up-projected final hidden

Gate to Stage 2 on per-layer hidden/KV cosine fidelity.

### Stage 2 — joint (assemble + LoRA + task)
Assemble **distilled ViT** (from the ViT runs) + **distilled LLM** (Stage 1) + small **KV
adapters** (student KV 208 → action-expert context 1024) into the **frozen** action expert;
attach **LoRA**. Jointly train with multi-level distillation vs the frozen teacher
(ViT↔LLM interface, per-layer LLM, final) **plus the flow-matching task loss**. This reuses
`hidden_kd`/`kv_kd` from `losses.py` and the finetune data path from the ViT runs.

### Eval
Closed-loop LIBERO (100 ep/suite), added as the 4th arm to the N-way aggregator.

## Files
- `flops.py` — torch-free FLOP profiler / dim tuner (teacher vs student).
- `student.py` — `LLMStudent` thin twin + `DistillHeads` + `build_student`.
- `losses.py` — multi-level distillation losses (hidden / KV / final).
- `train_stage1.py` — Stage-1 trainer; `--synthetic` numerical smoke, real teacher capture
  via `build_input_embeddings` + `transformer(output_hidden_states, collect_layer_kv_states)`.

## Autonomous decisions (chosen, not asked)
1. **Keep 36 layers, shrink width** (vs fewer layers) — required to keep the frozen action
   expert's per-layer-KV conditioning usable without rebuilding it.
2. **Reuse teacher `inputs_embeds`** via down-projection — the 100× target is the transformer
   stack; embedding/vision-fusion compute is negligible and shared.
3. **KV distillation weighted 2×** — it is the exact signal the action expert consumes.
4. **On-the-fly teacher targets** (no feature dump) — 37 hidden + 36 KV per sample is 100s of MB.
5. **hidden=208 / 112×** (headroom over 100×) rather than 224/98.8×.
