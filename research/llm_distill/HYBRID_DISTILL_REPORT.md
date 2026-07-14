# Hybrid LLM-Backbone Distillation for MolmoAct2 — KV-Anchoring + Token-Reduction

**Date:** 2026-07-14  ·  **Benchmark:** LIBERO closed-loop (`libero_spatial`, task 0, fixed seeds 1000+)
**Compute:** AMD Instinct (single-GPU per run, Apptainer SIF). Teacher = `MolmoAct2-LIBERO` (36-layer Qwen3 backbone + flow-matching action expert).

## TL;DR

Two concurrent distillation workstreams on the width-reduced (`qwen06w`, Qwen3-0.6B width, ~6× decoder-FLOP)
student, both training the **stock flow-matching action loss** with the frozen action expert (LoRA on
`context_{k,v}_proj`):

- **Workstream A — all-layer teacher-KV anchoring: SUCCESS.** Adding a per-layer KV-anchoring term on top of
  the flow loss fixes the *orthogonal-KV* failure of functional-only distillation. First-chunk teacher↔student
  KV cosine goes **0.001 → 0.822 (K) / 0.007 → 0.500 (V)** and closed-loop success goes **0% → 30%** (teacher 100%).
- **Workstream B — top-down token-reduction (6× student on the 25% stream): controlled negative.** The 6×
  student trains cleanly on the reduced stream (held-out flow **0.046**, per-layer KV cos ~**0.9** to the
  reduced-teacher) but scores **0%** closed-loop. A control proves this is a capacity limit, not a harness bug:
  the **full-width teacher through the identical prune-before-student harness scores 100%**.

## Background: the failure this fixes

The MolmoAct2 action expert cross-attends to the LLM's **per-layer K/V** (one DiT block per LLM layer, 1:1),
so distillation reduces **width**, not depth (student depth is fixed at 36). Two prior attempts both failed:

1. **Pure representational** (match per-layer KV only): reached ~0.99 KV cosine but 0% closed-loop — feature
   cosine on a fixed target is deceptive and starves the task objective.
2. **Functional-only** (flow loss only, KV free to co-adapt): the closed-loop diagnostic showed the student's
   KV becomes **orthogonal** to the teacher's at *identical* inputs (cos ≈ 0), and it never grasps — teacher
   100% vs student 0%.

The hybrid below keeps the functional objective but *anchors* the student's KV field to the teacher's proven,
robust one, so the student cannot collapse into an orthogonal shortcut.

---

## Workstream A — teacher-KV anchoring at every layer

**Method.** During training only, run a `no_grad` teacher-transformer forward on the same fused
`inputs_embeds` to source per-layer teacher KV at the exact collection point as the student's, then add an
anchoring term (masked to valid tokens) across all 36 layers:

- **directional** — `mean_layers[(1 − cos(sk,tk)) + (1 − cos(sv,tv))]` (fixes the cos≈0 orthogonality),
- **magnitude** — `β · normalized_MSE(sk,tk)+…` (fixes drifting KV norms),
- **total** — `flow_loss + λ · anchor` (`λ=1.0`, `mode=both`, `β=0.1`).

The teacher transformer is kept resident (frozen, bf16, ~+6 GB) only when anchoring is on. The anchor term is
gated to `self.training`, so the flow-only reload/verify checkpoint gate and held-out eval are unaffected.
Warm-started from Qwen3-0.6B; 6000 steps.

**Result (10 episodes, `libero_spatial` task 0, seeds 1000–1009).**

| Metric | Functional-only (baseline) | + all-layer KV anchoring |
|---|---|---|
| First-chunk **K** cosine (teacher vs student) | **0.001** (orthogonal) | **0.822** |
| First-chunk **V** cosine | **0.007** (orthogonal) | **0.500** |
| Closed-loop success | **0 % (0/10)** | **30 % (3/10)** |
| Teacher reference | 100 % | 100 % |
| Held-out flow loss (student / teacher) | — | **0.234 / 0.582** |

The orthogonality is removed (cos ≈ 0 → high) and closed-loop success rises from 0 to non-zero, meeting the
success gate. Checkpoint reload verification PASS on every save.

---

## Workstream B — top-down token-reduction distillation

**Method.** Wire the trained causal-predictive ROI front-end (`gate_predict`, keep-50 % @ ViT seam-6 →
FastV group-drop to 25 % image tokens; the ~95 % deployable `keep050_fv025_l09` checkpoint) into the
distillation data path (`capture_teacher_reduced`). We **prune before the student** (cut at the input, not
FastV's mid-forward L0 cut) so the entire student LLM + action expert run on the reduced sequence — the direct
test of the "condensed space lets a smaller LLM succeed" hypothesis. Full-token teacher supplies the top-down
flow target; optional KV anchoring is applied on the kept tokens.

Reduction measured on real LIBERO batches: **LLM sequence 479 → 185**, image tokens **392 → 98 (25 %)**.

**Arm (i): 6× `qwen06w` on the 25 % stream.** Warm-started, 6000 steps, `anchor_w=1.0 mode=both β=0.1`,
`fastv_keep=0.25`.

| Quantity | Value |
|---|---|
| Held-out flow loss (student) | **0.046** |
| Reduced-teacher (full-width LLM on 25 % stream) flow | 0.346 |
| Anchor loss plateau (→ per-layer KV cos ~0.9) | 0.235 |
| Checkpoint reload verify | PASS |
| **Closed-loop success (task 0)** | **0 % (0/5, 0/10)** |

**Control — is it the harness or the student?** We ran the **full-width teacher LLM through the identical
prune-before-student harness** (same reduced sequence build + the overlay's native flow-matching denoise loop):

| Configuration | Closed-loop success |
|---|---|
| Full-width teacher, prune-before-student harness (**control**) | **100 % (3/3)** |
| 6× student, same harness | 0 % |

This isolates the result: the eval harness is correct and the 25 % token stream is sufficient for the task
(input-cut is viable); arm (i)'s 0 % is a genuine **capacity limit of the 6× width** on the condensed stream.
For context, that same 6× width reaches 30 % on *full* tokens (Workstream A), so width-6× + token-4× compounds
past its capacity.

**Arm (ii): 2× `w1792` capacity control — not run.** `w1792` (head_dim 64 / width 1792) has no clean
warm-start source (`qwen06w` maps 1:1 from Qwen3-0.6B; no Qwen release matches `w1792`'s geometry), so it needs
a from-scratch init + a fresh full training run. Deferred as a scope decision.

---

## Key takeaways

1. **Representational anchoring is the enabling lever.** A per-layer teacher-KV anchor on top of the flow
   loss repairs functional-only distillation at 6× width (KV cos ~0 → 0.82; closed-loop 0 → 30 %). Neither
   pure-representational nor functional-only distillation works alone.
2. **Open-loop flow loss is not predictive of closed-loop success.** Arm (i) has a *lower* held-out flow loss
   (0.046) than the anchored A student (0.234) yet fails closed-loop — closed-loop rollout, not teacher-forced
   velocity MSE, is the real metric.
3. **The 25 % token stream is not the bottleneck.** The full-width teacher succeeds at 100 % through the exact
   reduced harness; aggressive width × token compression together exceed the small student's capacity.

## Reproduction

All runs use the eval overlay = the validated ROI training overlay (`roi_prune.py` + `modeling_molmoact2.py`),
Apptainer SIF, single AMD Instinct GPU. Large weights/videos live only on the remote compute store.

```bash
# 0) warm-start the student from Qwen3-0.6B (once)
python warmstart.py --preset qwen06w --out <out>/warmstart/qwen06w_init.pt

# A) all-layer KV-anchored joint training (6000 steps)
python train_joint.py --preset qwen06w --init <out>/warmstart/qwen06w_init.pt \
  --anchor-weight 1.0 --anchor-mode both --anchor-beta 0.1 --keep-teacher --steps 6000
# A) diagnostics: closed-loop success + first-chunk teacher-vs-student KV cosine
python diag_closed_loop.py --student-ckpt <run>/latest.pt --n-episodes 10 --suite libero_spatial --task-ids 0

# B) front-end validation + top-down reduced-token training
python validate_reduced.py --roi-ckpt <roi_gate_fastv_ckpt> --fastv-keep 0.25
python train_reduced.py  --roi-ckpt <roi_gate_fastv_ckpt> --preset qwen06w \
  --init <out>/warmstart/qwen06w_init.pt --fastv-keep 0.25 \
  --anchor-weight 1.0 --anchor-mode both --anchor-beta 0.1 --steps 6000
# B) closed-loop eval (student) and the harness control (full-width teacher)
python eval_reduced.py --roi-ckpt <roi_gate_fastv_ckpt> --student-ckpt <run>/step_6000.pt --fastv-keep 0.25
python eval_reduced.py --roi-ckpt <roi_gate_fastv_ckpt> --teacher-through-reduced --fastv-keep 0.25
```

## Artifacts

Small curves/configs are mirrored under `artifacts/llm_distill/`; per-episode rollout videos, KV dumps
(`kv_raw.npz`), and checkpoints stay on the remote compute store (large binaries out of version control).
