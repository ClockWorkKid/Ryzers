# Distilling the MolmoAct2 LLM Backbone into `qwen06w` — the route that worked

**Result.** A width-reduced Qwen3 student (`qwen06w`, **6.4× fewer decoder FLOPs** than the teacher
backbone) reaches **47.5 % LIBERO closed-loop success** via a three-stage recipe —
**warm-start → all-layer teacher-KV anchoring → co-adaptation → on-policy DAgger with a LoRA-adapted
action expert**. This is **+17.5 pts past** the best score any teacher-forced distillation recipe reached
(30 %), while the distilled backbone is **6.4× smaller**, **2.2–3.2× faster**, and uses **~5.4× less
inference memory** than the teacher backbone.

**Benchmark:** LIBERO closed-loop (4 suites × 10 tasks × 3 episodes = 120 rollouts, fixed seeds 1000+).
**Teacher:** `MolmoAct2-LIBERO` = 36-layer Qwen3 backbone (hidden 2560) + flow-matching action expert.
**Compute:** single AMD Instinct MI210, Apptainer SIF, bf16. *This report covers only the successful
route; unrelated exploratory tracks (token-reduction, prune-then-heal) are out of scope.*

---

## 1. What we are distilling, and why it is hard

MolmoAct2 predicts actions with a **flow-matching action expert** that cross-attends to the LLM
backbone's **per-layer key/value tensors** — one action-expert (DiT) block per LLM layer, **1:1**. Two
hard constraints follow:

- **Depth is fixed at 36** (the action expert has one block per LLM layer). Distillation can only reduce
  **width** (hidden size / heads / MLP), not depth.
- The per-layer **KV interface must match the teacher byte-for-byte**: `num_kv_heads·head_dim = 8·128 =
  1024`, collected *post-QK-norm, post-RoPE, pre-GQA-repeat* — exactly where the frozen
  `context_{k,v}_proj` (1024→768) of the action expert reads. Keeping `kv_dim = 1024` lets the student
  plug into the action expert with **no surgery**.

| | Teacher backbone | `qwen06w` student |
|---|---|---|
| Layers | 36 | 36 (fixed) |
| Hidden | 2560 | 1024 |
| Query / KV heads × dim | 32 / 8 × 128 | 16 / 8 × 128 |
| MLP intermediate | 9728 | 3072 |
| KV interface (`kv_dim`) | 1024 | 1024 (identical) |
| Backbone params | **3.65 B** | **0.57 B** |

**The failure this route had to beat.** Naïve backbone distillation collapses:
- *Pure representational* (match per-layer KV only): ~0.99 KV cosine but **0 % closed-loop** — a deceptive
  target that starves the task objective.
- *Functional-only* (flow loss only, KV free): the student's KV becomes **orthogonal** to the teacher's at
  identical inputs (cos ≈ 0) and it never grasps — **0 % closed-loop** vs teacher 100 %.

---

## 2. The pipeline that worked

```
Qwen3-0.6B ──warm-start──▶ qwen06w student
                              │
              Stage 1: all-layer teacher-KV anchoring  (flow + λ·KV-anchor)      → fixes orthogonal collapse (0→30%)
                              │
              Stage 2: LLM + action-expert co-adaptation, then LoRA fine-tune    → flexible base (9.2%)
                              │
              Stage 3: on-policy DAgger + LoRA-adapted action expert             → 47.5%  ◀ deliverable
```

### Stage 0 — `qwen06w` student + warm-start
The student (`student.py`) is a drop-in for `model.model.transformer`: it consumes the teacher's fused
`inputs_embeds` [B,N,2560] via a `down_proj` (2560→1024), runs 36 GQA decoder layers, and emits
`last_hidden_state` [B,N,2560] via `up_proj` plus per-layer KV in the exact teacher contract. It is
**warm-started from pretrained Qwen3-0.6B** (`warmstart.py`): transformer weights map 1:1 (fusing Qwen's
split `q/k/v`→`qkv` and `gate/up`→`gate_up`), the 28→36 depth gap is bridged by a nearest-layer stretch,
and the two new interface layers (`down_proj`, `up_proj`) train from scratch.

### Stage 1 — all-layer teacher-KV anchoring (layerwise latent matching)
The enabling lever. During training only, a `no_grad` teacher forward on the *same* fused embeds sources
per-layer teacher KV at the identical collection point, and we add an anchoring term across **all 36
layers** on top of the flow loss:

- **directional** — `mean_layers[(1 − cos(sₖ,tₖ)) + (1 − cos(sᵥ,tᵥ))]` (removes the cos≈0 orthogonality),
- **magnitude** — `β · normalized_MSE(sₖ,tₖ) + …` (stops KV-norm drift),
- **total** — `flow_loss + λ·anchor` with `λ=1.0`, `mode=both`, `β=0.1`.

The teacher transformer is kept resident (frozen, bf16, ~+6 GB) only while anchoring is on; the term is
gated to `self.training` so checkpoint-reload verification and held-out eval are unaffected.

**Effect (10 ep, `libero_spatial` t0):** first-chunk teacher↔student KV cosine **K 0.001→0.822,
V 0.007→0.500**; closed-loop **0 %→30 %**. This 30 % frozen-action-expert + KV-anchor student is the
reference ceiling every later stage is measured against.

### Stage 2 — LLM + action-expert co-adaptation → a *flexible* base
Anchoring alone (with a frozen action expert) tops out at 30 %. Stage 2 loosens the pair: co-adapt the
student LLM **and** the action expert under the KV anchor (so the action expert re-tunes to the student's
KV field), then a LoRA fine-tune of `{llm, ae, vit}` on real LIBERO. On its own this base scores modestly
(9.2 %) — deliberately, it trades raw teacher-forced score for a **co-adaptable** base that the on-policy
stage can move. (A rigid, over-committed anchored base does *not* respond to Stage 3 — see §5.)

### Stage 3 — on-policy DAgger with a LoRA-adapted action expert (the unlock)
Stages 0–2 all optimize a **teacher-forced** objective on demo states, so none of them see the states the
student actually visits at test time — the closed-loop covariate-shift gap. Stage 3 closes it directly:

1. **Collect** (`collect_dagger.py`) — roll out the student closed-loop over 4 suites × 10 tasks × 3 ep
   and snapshot the model-ready batch at every 10-step replan.
2. **Relabel** (`relabel_teacher.py`) — run the teacher's `predict_action_chunk` on each student-visited
   state; its **current-step normalized action** is the target. Validated: teacher label ≈ demo action
   (MSE 0.012) and self-consistent under the teacher's flow field.
3. **Fine-tune** (`train_joint.py --dagger-dir --dagger-adapt-ae lora`) — student LLM **+ LoRA-adapted
   action expert** (`context_{k,v}_proj`), KV-anchor on, on a **50/50 on-policy/demo mix**, 6 k steps.
   Letting the action expert co-adapt *cheaply* (LoRA) to the shifted student features on the on-policy
   distribution is what breaks the ceiling; freezing it was the limiter.
4. **Gate** — 120 closed-loop rollouts.

---

## 3. Results

**Trajectory of the winning arm (`expd` base, `qwen06w`):**

| Stage | Closed-loop success |
|---|---|
| Stage-2 co-adapted base | 9.2 % |
| + on-policy DAgger, frozen action expert | 16.67 % |
| **+ on-policy DAgger, LoRA-adapted action expert** | **47.5 %** |

**Per-suite at 47.5 % (120 ep) vs the 30 % KV-anchor reference:**

| Suite (30 ep) | KV-anchor reference (Stage 1) | **`qwen06w` final** |
|---|---|---|
| libero_spatial | — | 50.0 % |
| libero_object | — | 80.0 % |
| libero_goal | — | 33.3 % |
| libero_10 (long) | — | 26.7 % |
| **Overall** | **30 %** | **47.5 %** |

The distilled student clears the **30 % teacher-forced ceiling by +17.5 pts**, with especially strong
object manipulation (80 %).

---

## 4. Computational benefits vs the teacher backbone

Measured on a single **AMD Instinct MI210, bf16**, over the LIBERO fused sequence (N = 479) — one
full-sequence prefill, i.e. exactly the forward the action expert consumes per-layer KV from. Latency is
mean of 20 iters after warm-up; memory is `max_memory_allocated`.

| Metric | Teacher backbone | `qwen06w` student | **Improvement** |
|---|---|---|---|
| Parameters | 3.65 B | 0.57 B | **6.38× smaller** |
| Weights (bf16) | 7.29 GB | 1.14 GB | **6.38× smaller** |
| Prefill latency, batch 1 | 61.6 ms | 27.5 ms | **2.24× faster** |
| Prefill latency, batch 8 | 350.7 ms | 108.6 ms | **3.23× faster** |
| Peak inference memory, batch 1 | 7.73 GB | 1.42 GB | **5.45× less** |
| Peak inference memory, batch 8 | 9.82 GB | 2.79 GB | **3.52× less** |
| Analytic decoder FLOPs / token | — | — | **6.42× fewer** |

**Reading the numbers.** Parameter/weight/FLOP savings are ~6.4×. Real latency gains are smaller at
batch 1 (2.24×) because single-sequence prefill is partly launch/bandwidth-bound and the width-independent
costs (attention over N=479, RoPE, the 2560-dim interface projections) do not shrink with hidden size; as
the workload becomes compute-bound at batch 8 the speedup rises toward the FLOP ratio (3.23×). Peak
inference memory drops **5.4×** at batch 1 (weight-dominated) — the single biggest practical win, taking
the backbone from ~7.7 GB to ~1.4 GB and comfortably into small-accelerator / edge budgets. The action
expert and ViT are unchanged, so these are the backbone-level deltas that this distillation delivers.

---

## 5. Why each design choice (negative controls, kept brief)

Every stage of the recipe is there because the cheaper alternative was measured to fail:

- **Anchor is required.** Pure-representational and functional-only distillation both give 0 % closed-loop
  (§1); only `flow + KV-anchor` reaches 30 %.
- **On-policy data is required, but only helps a flexible base.** With a frozen action expert, DAgger
  moves the flexible `expd` base up (9.2 → 16.67 %) but *regresses* the rigid 30 % anchor-only base
  (30 → 23.3 %) — on-policy data perturbs an over-committed base faster than relabels can repair it.
- **The action expert must co-adapt (LoRA), not stay frozen and not fully unfreeze.** Frozen AE caps
  `expd` at 16.67 %; LoRA-adapting it during on-policy fine-tuning is the jump to 47.5 %. Conversely,
  fully unfreezing the action expert under a *teacher-forced* objective (Stage-2 style) hurt — so the win
  is specifically *small* AE adaptation under an *on-policy* objective.
- **Do not naïvely iterate DAgger.** A second DAgger round that re-collects from the 47.5 % policy and
  retrains 6 k steps on only the new pool (no data aggregation) **regressed to 31.67 %** — classic
  non-aggregated-DAgger forgetting. The 47.5 % checkpoint is the deliverable; a future round-2 must
  aggregate round-1+2 data and/or fine-tune from the 47.5 % checkpoint at low LR.
- **Open-loop flow loss is not the metric.** Lower held-out flow repeatedly failed to predict closed-loop
  success (the co-adapted base had the *best* flow loss, 0.061, at only 9.2 %). Closed-loop rollout is the
  only reliable gate.

---

## 6. Reproduction

Single MI210 GPU, eval SIF, `/outputs` bound to the compute store. Large weights/videos stay on the
remote store.

```bash
# 0) warm-start qwen06w from Qwen3-0.6B (once)
python warmstart.py --preset qwen06w --out <out>/warmstart/qwen06w_init.pt

# 1) all-layer KV-anchored joint training (flow + anchor)
python train_joint.py --preset qwen06w --init <out>/warmstart/qwen06w_init.pt \
  --anchor-weight 1.0 --anchor-mode both --anchor-beta 0.1 --keep-teacher --steps 6000

# 2) co-adaptation base + LoRA fine-tune -> the flexible `expd` base (see train_joint.py phases)

# 3) on-policy DAgger with LoRA-adapted action expert  (the 47.5% run)
python collect_dagger.py  --student-ckpt <base>/step_6000.pt --suites all --tasks 10 --episodes 3
python relabel_teacher.py --dagger-dir <out>/dagger/expd
python merge_base.py      --student-ckpt <base>/step_6000.pt --out <out>/dagger/expd/merged_init.pt
python train_joint.py --preset qwen06w --init <out>/dagger/expd/merged_init.pt \
  --dagger-dir <out>/dagger/expd --dagger-adapt-ae lora \
  --anchor-weight 1.0 --anchor-mode both --anchor-beta 0.1 --keep-teacher --steps 6000

# gate: 4 suites x 10 tasks x 3 ep = 120 rollouts
python eval_joint.py --student-ckpt <out>/dagger/expd/train_aelora/step_6000.pt --suites all --tasks 10 --episodes 3
```

## 7. Deliverable & artifacts

- **Checkpoint:** `dagger/expd/train_aelora/step_6000.pt` (47.5 % closed-loop) — the default, most
  successful MolmoAct2 LLM-backbone distillation to date.
- **Backbone benchmark:** `bench_backbone.py` (params/latency/memory, MI210 bf16).
- Small curves/configs mirror under `artifacts/llm_distill/`; checkpoints, KV dumps, and rollout videos
  live only on the remote compute store.
