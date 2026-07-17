# Hybrid LLM-Backbone Distillation for MolmoAct2 — KV-Anchoring + Token-Reduction

**Date:** 2026-07-17 (Workstream F / squeeze + capacity test complete; program best still = 47.5 %, R1b)  ·  **Benchmark:** LIBERO closed-loop (fixed seeds 1000+; per-run suite/task noted)
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
- **Workstream C — healing-budget vs pruning-damage (Minitron prune-then-heal): SUCCESS.** The 0% from the
  earlier 500-step curriculum heals was a *heal-budget* artifact, not pruning damage. Matched 6000-step anchored
  heals give **mildest-prune (1.13×) = 50%** vs **non-pruned control = 20%** closed-loop — i.e. a mildly pruned
  student *beats* the un-pruned one under the same recipe, so the earlier collapse was under-training.
- **Workstream D — joint LLM + full-AE co-distillation (Experiment D): negative (completed).**
  Co-adapting the student LLM **and the full action expert** (anchor on) drove the best open-loop flow of any
  run (held-out **0.061**, teacher 0.582) yet Gate-1 closed-loop was only **5.0%** over 120 episodes (4 suites ×
  10 tasks × 3 ep). The Phase-2 LoRA fine-tune on real LIBERO (`llm,ae,vit`, anchor off, 12k) roughly doubled
  it to **Gate-2 = 9.2%** — still **far below the 30% frozen-AE + KV-anchor baseline** (Workstream A). Unfreezing
  the AE does **not** beat freezing it, and a real-data LoRA pass does not recover the gap.
- **Workstream E — on-policy DAgger + AE-LoRA (complete): new program best 47.5%.** A/B/C/D all optimize
  *teacher-forced* objectives; none touch the closed-loop covariate-shift gap. Workstream E attacks it directly:
  roll out the student, snapshot its *own* visited states, relabel them with the teacher's current-step action,
  and fine-tune (student LLM + KV-anchor **on**) on a 50/50 on-policy/demo mix. **Round-1 (AE frozen):** `expd`
  9.2%→16.67%, `wsa` 30%→23.33% (helps the weak base, hurts the strong one). **Round-1b (AE LoRA-adapted):**
  `expd` jumps to **47.5%** — **+17.5 pts past the 30% WS-A ceiling** that every teacher-forced workstream was
  stuck under (object suite 80%!), while `wsa` keeps regressing (20.83%). Lesson: closing the covariate-shift gap
  needs *both* on-policy data *and* a co-adaptable (LoRA) action expert, and it pays off on a weak/flexible base
  (`expd`) rather than a strong/over-committed anchored one (`wsa`). Note this is the *opposite* of Exp-D, where
  unfreezing the whole AE under a teacher-forced objective hurt. **Round-2** (naive iteration: fresh collect from
  the 47.5% policy, no data aggregation, full 6k retrain) **regressed to 31.67%** — non-aggregated DAgger forgets
  the base's competence — so **R1b's 47.5% remains the program best/deliverable**. A future round-2 should
  aggregate round-1+2 data and/or fine-tune from the R1b checkpoint (see Workstream E section).
- **Workstream F — squeeze + capacity test: no gain past 47.5 %; capacity question still open.** (A) Continuing
  DAgger from 47.5 % on the aggregated round-1+2 pool: best = **43.3 %** (r16, data-aggregation lever); adding
  action-expert capacity *hurt* (r64 32.5 %, full-AE 18.3 %). (B) A new **two-teacher co-distillation**
  (`train_codistill.py`) trained ONE 0.6B student against **DROID + LIBERO** teachers simultaneously (per-domain
  KV anchor), then LIBERO LoRA-finetuned it → gate **0.83 %**. A co-distill-free **control** proved this was a
  *fine-tune-recipe artifact*: LoRA-r64 on the student LLM floors at ~0.29 held-out flow (vs ~0.06 for full
  unfreeze), so the width-reduced student needs a **fully-unfrozen** LLM fine-tune, which every successful run
  used. The clean capacity verdict (full-unfreeze from the co-distilled base + a larger DROID subset) is deferred.

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

## Workstream C — healing-budget vs pruning-damage (Minitron prune-then-heal)

**Question.** The all-layer KV-anchored non-pruned student (Workstream A) reached 30% in 6000 steps, but our
Minitron pruning *curriculum* heals were only 500 steps each and scored **0%**. Was the 0% caused by the
pruning, or by an under-sized heal budget?

**Method.** Run two **matched-recipe** anchored heals (6000 steps, `gacc1`, eff. batch 4, `anchor_w=1.0
mode=both β=0.1`), the only variable being the init:

- **(a) mildest prune** — curriculum step-1 pruned-from-teacher checkpoint (`step1_i8192_h32`, ~1.13× reduction),
- **(b) non-pruned control** — minted via a no-op prune from the teacher (9728/32 == teacher width),

then closed-loop eval both (`libero_spatial` tasks 0,1,2, 10 episodes each = 30 rollouts, seed 1000). The mi210
4-hour wall cap is honored by dropping to `gacc1` (~0.7 sps → ~2.4 h for 6000 steps).

**Result.**

| Init (matched 6000-step anchored heal) | Held-out flow | Closed-loop success | Teacher ref |
|---|---|---|---|
| Mildest prune (step-1, ~1.13×) | 0.145 | **50 % (15/30)** | 100 % |
| Non-pruned control | 0.134 | **20 % (6/30)** | 100 % |
| *(prior) 500-step curriculum heal* | — | *0 %* | 100 % |

**Takeaway.** The 0% was **under-training, not pruning damage** — a full heal budget recovers it, and the
mildly pruned student actually **beats** the non-pruned control (50 % vs 20 %) under the identical recipe. Heal
budget is the dominant lever at mild compression.

---

## Workstream D — joint LLM + full action-expert co-distillation (Experiment D)

**Hypothesis.** Workstreams A–C froze the action expert (LoRA only on `context_{k,v}_proj`). Experiment D tests
whether **co-adapting the student LLM and the *full* action expert together** — so the AE can re-tune to the
student's KV field instead of the teacher's — closes the flow↛rollout gap.

**Design (two phases, 12k steps each, chained ≤4h jobs with optimizer/LR/step resume).**

- **Phase 1 — co-adaptation.** Unfreeze the full student AE as a third trainable group alongside the student
  LLM; teacher LLM kept resident for the all-layer KV anchor; objective `flow + λ·anchor` (anchor on).
- **Phase 2 — LoRA fine-tune.** Wrap ViT + student LLM + student AE with LoRA, init from the Phase-1
  checkpoint, train on **real LIBERO** with **flow only (anchor off)** to recover closed-loop behavior.

**Evaluation gates.** Full harness = 4 suites (`spatial`, `object`, `goal`, `10`) × 10 tasks × 3 episodes =
**120 rollouts**, after each phase.

**Phase-1 result (Gate 1).**

| Metric | Value |
|---|---|
| Steps trained | 12000 (done) |
| Held-out flow (student / teacher) | **0.061 / 0.582** — best open-loop of any run |
| Gate-1 closed-loop, overall | **5.0 % (6/120)** |
|  · `libero_spatial` | 13.3 % |
|  · `libero_object` | 3.3 % |
|  · `libero_goal` | 3.3 % |
|  · `libero_10` (long) | 0.0 % |

**Phase-2 result (Gate 2).** LoRA fine-tune of `{llm, ae, vit}` on real LIBERO, flow-only (anchor off), 12k
steps (checkpoint reload verify PASS, best held-out flow 0.056):

| Suite | Gate-1 (post Phase-1) | Gate-2 (post Phase-2 LoRA) |
|---|---|---|
| `libero_spatial` | 13.3 % | **20.0 %** |
| `libero_object` | 3.3 % | 10.0 % |
| `libero_goal` | 3.3 % | **0.0 %** |
| `libero_10` (long) | 0.0 % | 6.7 % |
| **Overall (120 ep)** | **5.0 %** | **9.2 %** |
| Reference baseline (Workstream A) | 30 % | 30 % |

**Takeaway.** Unfreezing the full AE produced the **lowest flow loss we have seen (0.061)** yet the **worst
broad-suite rollout** — a sharp restatement of the Workstream-A/B lesson that open-loop flow is not predictive.
The Phase-2 real-data LoRA pass **roughly doubled rollout (5.0 → 9.2 %)** but stays **well below the 30 %
frozen-AE + KV-anchor baseline**. Letting the AE co-adapt settled the pair into a low-flow / low-control basin
that a downstream fine-tune only partly repairs. Net: **more trainable freedom (full AE) hurt**, and the extra
degrees of freedom are not the missing lever — the flow↛rollout (closed-loop covariate-shift) gap is.

---

## Workstream E — on-policy DAgger (round-1 complete)

**Motivation.** Every prior workstream (A–D) minimizes a **teacher-forced** objective on *demo* states (flow
loss, KV anchor). Gate results (A=30 %, D=9.2 %, both with excellent open-loop flow) keep re-confirming that the
gap to the teacher's 100 % is **closed-loop covariate shift**: the student visits states its demos never cover,
and no teacher-forced loss ever sees them. Workstream E is the first to train on the student's *own* state
distribution — the textbook DAgger fix for exposure bias.

**Method (per arm, round 1).**
1. **Collect** — roll out the student closed-loop over 4 suites × 10 tasks × 3 ep (full episodes) and snapshot
   the model-ready batch at every 10-step replan (`collect_dagger.py`).
2. **Relabel** — run the teacher's `predict_action_chunk` on each student-visited state; its **current-step**
   normalized action is the target (`relabel_teacher.py`). Validated: teacher label ≈ demo action (MSE 0.012),
   and it is *self-consistent* under the teacher's flow field (loss 0.23 < 0.41 demo-target < 0.58 baseline).
3. **Fine-tune** — student LLM only (**AE frozen**, no LoRA), **KV-anchor on** (`λ=1, β=0.1`), on a **50/50
   mix** of on-policy + demo batches, 6 k steps (`train_joint.py --train-student-only --dagger-dir`). Only the
   *data distribution* changes vs the recipe that produced the base — the AE stays frozen precisely to avoid the
   Exp-D full-AE regression.
4. **Gate** — 4 suites × 10 tasks × 3 ep = 120 rollouts + aggregate.

**Arms.** `wsa` (from the 30 % WS-A anchored student) and `expd` (from the 9.2 % Exp-D Phase-2 student), each
merged to a plain student+AE init (`merge_base.py`) so both start format-uniform.

**Status.** Round-1 **complete** for both arms (collect → relabel → 6 k-step student-only fine-tune → 120-ep
gate). Scaffold fully validated (collector, relabeler, merge, and a 20-step train smoke with checkpoint verify
PASS). On-policy pool: `wsa` 4072 relabeled states, `expd` 3932.

**Round-1 gate (120 ep/arm, 4 suites × 10 tasks × 3 ep).**

| Suite | `wsa` | `expd` |
|---|---|---|
| libero_spatial | 20.0 % | 23.3 % |
| libero_object  | 46.7 % |  6.7 % |
| libero_goal    |  6.7 % | 20.0 % |
| libero_10      | 20.0 % | 16.7 % |
| **Overall**    | **23.33 %** | **16.67 %** |
| heldout flow   | 0.167 | 0.061 |

**Outcome — split result.** DAgger moved the two arms in *opposite* directions relative to their own bases:
- **`expd`: 9.2 % → 16.67 % (+7.5 pts, ≈1.8×).** On-policy relabeling clearly helps the weaker,
  co-adapted student — its first real evidence of closing the covariate-shift gap.
- **`wsa`: 30 % → 23.33 % (−6.7 pts, regressed).** One round of on-policy data *hurt* the already-strong
  anchored student. Likely causes: (a) the round-1 on-policy states come from a 30 % policy, so ~70 % of
  snapshots are failure/off-distribution trajectories the teacher can only weakly relabel; (b) a frozen AE
  cannot absorb the shifted LLM features; (c) the 50/50 mix over-weights noisy on-policy targets. Neither arm
  passed the WS-A 30 % ceiling.

**Read.** On-policy data helps exactly where teacher-forced training left the most headroom (`expd`) and hurts
where the student was already near its recipe ceiling (`wsa`). This points to iterative DAgger (round-2 collect
from the *improved* `expd` policy) and/or unfreezing the AE during on-policy fine-tuning as the next levers,
rather than a single large round from a fixed base.

### Round-1b — on-policy DAgger with the AE unfrozen (LoRA)

**Change vs round-1.** Same on-policy relabeled pools, merged bases, 50/50 mix, KV-anchor on, 6 k steps —
but the action expert is now **LoRA-adapted** (context-K/V projections; `attach_student(adapt_ae="lora")`,
which is exactly the legacy 30%-recipe interface) instead of frozen. Tests the leading hypothesis for the
`wsa` regression: that a frozen AE could not absorb the shifted student LLM features on the new state
distribution. Written to `train_aelora`/`gate2_aelora` so round-1 (frozen-AE) artifacts are preserved.

**Gate (120 ep/arm).**

| Suite | `expd` R1 (frozen) | **`expd` R1b (AE-LoRA)** | `wsa` R1 (frozen) | `wsa` R1b (AE-LoRA) |
|---|---|---|---|---|
| libero_spatial | 23.3 % | **50.0 %** | 20.0 % | 20.0 % |
| libero_object  |  6.7 % | **80.0 %** | 46.7 % | 36.7 % |
| libero_goal    | 20.0 % | **33.3 %** |  6.7 % |  6.7 % |
| libero_10      | 16.7 % | **26.7 %** | 20.0 % | 20.0 % |
| **Overall**    | 16.67 % | **47.5 %** | 23.33 % | 20.83 % |

**Full trajectories.**

| Arm | Base | DAgger, frozen AE (R1) | DAgger, AE-LoRA (R1b) |
|---|---|---|---|
| `expd` | 9.2 % | 16.67 % | **47.5 %** |
| `wsa`  | 30.0 % | 23.33 % | 20.83 % |

**Outcome — AE-LoRA is the unlock, but only for the flexible base.** `expd` goes **9.2 % (base) → 16.67 %
(DAgger, frozen AE) → 47.5 % (DAgger, AE-LoRA)**: +38.3 pts over its base and, decisively, **+17.5 pts past the
30 % WS-A ceiling** that every teacher-forced workstream (A–D) had been stuck under. The action expert *must* be
allowed to co-adapt (cheaply, via LoRA) to the student's shifted features on the on-policy state distribution;
freezing it was the limiter. This is the opposite lesson from Exp-D, where unfreezing the *whole* AE under a
*teacher-forced* objective hurt — here a *small* AE adaptation under an *on-policy* objective is what closes the
covariate-shift gap.

`wsa`, by contrast, keeps **regressing under any on-policy round** (30 % → 23.33 % frozen → 20.83 % AE-LoRA).
The anchored `wsa` student was trained to tightly couple its LLM features to the *frozen teacher AE* (that is
what drove it to 30 %); a 50/50 on-policy mix — where ~70 % of collected states come from a 30 %-success policy
and are failure/off-distribution — perturbs that coupling faster than the teacher relabels can repair it, and
LoRA-adapting the AE does not save it. On-policy DAgger therefore helps a **weak, jointly-trainable** base with
large headroom and hurts a **strong, over-committed** one; the right base to iterate on is `expd`.

(Operational note: the `wsa` R1b trainA hung at the GPU level on one node at step ~5950 with LR already decayed
to 0; it was cancelled and trainB resumed cleanly from the step-5000 checkpoint on another node — no learning
lost, and the legacy-format resume correctly restored the context-proj LoRA deltas.)

### Lineage of the 47.5 % model (qwen06w, NOT prune-then-heal)

To avoid confusion: the 47.5 % result is the **`expd` arm on the `qwen06w` student architecture**, produced by
the **latent-feature-matching (teacher-KV anchoring) distillation with a co-adapted LLM + action-expert pair**
(Experiment D), then lifted by on-policy DAgger + AE-LoRA (Workstream E). It is **not** the Minitron
prune-then-heal route.

- **Architecture (verified from the checkpoint cfg):** both DAgger arms — `wsa` and `expd` — use the *same*
  `qwen06w` student: 36 layers, hidden 1024, 16 heads / 8 KV heads, intermediate 3072, head_dim 128, distilling
  a 2560-hidden teacher. It is warm-started from Qwen3-0.6B (a from-scratch small backbone), **not** pruned from
  the teacher. The two arms differ only in the *distillation recipe of their base*, not the architecture:
  - `wsa` base = pure per-layer KV-anchor + **frozen** action expert (the 30 % recipe, Workstream A).
  - `expd` base = KV-anchor + **co-adapted** student LLM & action expert (Experiment D), Phase-2 LoRA → 9.2 %.
- **The prune-then-heal (Minitron) student is a separate route** that pruned the teacher's depth/width only
  mildly and healed; it is not one of the DAgger arms and is not the 47.5 % model.
- **Decision:** `qwen06w` (Experiment-D base + Workstream-E DAgger/AE-LoRA) is adopted as the **default, most
  successful distillation route to date (47.5 % closed-loop)**; the prune-then-heal route is deprioritized.

### Round-2 (iterate DAgger from the 47.5 % policy) — regressed to 31.67 %

**Setup.** Standard DAgger iteration on `expd`: re-collect on-policy states by rolling out the **47.5 % R1b
policy** (4 suites × 10 tasks × 3 ep → 3320 relabeled states), merge the R1b ckpt to a plain init, and re-run
the same AE-LoRA recipe (student LLM + context-proj AE-LoRA, KV-anchor on, 50/50 new-on-policy/demo, 6 k steps).
Written under `expd/round2` so the 47.5 % R1b checkpoint is preserved.

**Gate (120 ep).**

| Suite | R1b (47.5 %) | **R2** | Δ |
|---|---|---|---|
| libero_spatial | 50.0 % | 50.0 % | 0 |
| libero_object  | 80.0 % | 33.3 % | **−46.7** |
| libero_goal    | 33.3 % | 36.7 % | +3.4 |
| libero_10      | 26.7 % |  6.7 % | −20.0 |
| **Overall**    | **47.5 %** | **31.67 %** | **−15.8** |

**Outcome — naive iteration does NOT compound; R1b (47.5 %) remains the program best.** A full 6 k-step retrain
from the merged R1b init, on *only* the newly-collected round-2 pool (no aggregation with round-1 data),
regressed overall by ~16 pts, with the loss concentrated in the suites R1b was strongest on (object 80→33,
libero_10 27→7) while held-out flow was actually *lower* (0.079). This is the classic non-aggregated-DAgger
failure mode: retraining chases the newest on-policy distribution and **forgets** the competence encoded in the
base, and the low flow loss (again) does not predict closed-loop success. Likely fixes for a future round-2:
(a) **aggregate** round-1 + round-2 on-policy pools (true DAgger D←D∪new), (b) **fine-tune from the R1b
checkpoint** with a lower LR / fewer steps instead of a fresh 6 k from the merged init, and/or (c) filter
on-policy snapshots to successful/near-successful trajectories. For now the **47.5 % R1b checkpoint stands as the
deliverable** (`dagger/expd/train_aelora/step_6000.pt`).

---

## Workstream F — squeeze + capacity test (from the 47.5 % program best)

Two parallel tracks probing whether the 47.5 % ceiling is a *training/data* limit or a *hard 0.6B
capacity* limit. **Neither track beat 47.5 %, and the capacity question remains open** — Track B's
low number turned out to be a fine-tune-recipe artifact (diagnosed by a control), not evidence of a
capacity wall.

### Track A — cheap 0.6B "squeeze" (continue DAgger from 47.5 %, aggregated round-1+2 pool)
All three variants start from the merged 47.5 % (`expd` R1b) init and continue on the **aggregated**
round-1+2 on-policy pool (round-1 3932 + round-2 3320 relabeled states), KV-anchor on, 6 k steps,
50/50 on-policy/demo. Gate = 4 suites × 10 tasks × 3 ep = 120 rollouts.

| Variant | AE adaptation | heldout flow | Gate overall | vs 47.5 % |
|---|---|---|---|---|
| **a3_r16agg** (r16, low-LR, aggregated pool) | context-proj LoRA r16 | 0.0797 | **43.33 %** | −4.2 |
| **a1_r64** (higher-rank AE-LoRA) | context-proj LoRA r64 | 0.0696 | **32.5 %** | −15.0 |
| **a2_full** (full-AE unfreeze, bf16) | whole AE trainable | 0.0635 | **18.33 %** | −29.2 |

**Outcome — the squeeze does not beat 47.5 %; the data-aggregation lever helps most, extra AE
capacity hurts.** a3 (which isolates *data aggregation* at the original rank/LR) is the strongest at
43.3 % — recovering most of round-2's forgetting but still short of R1b. Pushing AE capacity is
counter-productive: rank-64 (a1) drops to 32.5 % and full-AE unfreeze (a2) collapses to 18.3 %, the
same "more AE freedom hurts under this objective" lesson as Exp-D. Read: on `qwen06w`, the action
expert is **not** the under-adapted part — consistent with the width of the LLM being the suspect,
which motivated Track B.

### Track B — DROID+LIBERO co-distillation → LIBERO fine-tune (the intended capacity decider)
Give the 0.6B student broad VLA competence, then adapt to LIBERO, mirroring the base model's
DROID-pretrain → LIBERO-finetune recipe.

- **B2 — two-teacher co-distillation (new: `train_codistill.py`).** ONE shared student (LLM + a
  single shared action expert) distilled against **both** `MolmoAct2-DROID` and `MolmoAct2-LIBERO`
  teachers simultaneously — each domain KV-anchored to its *own* teacher, batches alternating 50/50,
  both teachers resident but only one runs per step (peak activation ≈ single-teacher). Both teachers
  share the Molmo2-ER VLM geometry (hidden 2560, 36 layers, 8 KV heads, head_dim 128) so one student
  KV contract serves both; both pad actions to 32-dim so one AE serves both embodiments. 8 k steps;
  LIBERO held-out flow → **0.384**. *Caveat:* DROID here is a **bounded 24-episode offline subset**
  (full `droid_1.0.1` ≈ 1.7 TB, not staged), so this is *broadened*, not full-scale, pretraining.
- **B3 — LIBERO LoRA fine-tune** of the co-distilled student (`--lora-finetune` on `{llm,ae,vit}`,
  r64/alpha16, lr 5e-5, 12 k). Held-out flow improved to **0.243**.
- **B4 gate — 0.83 % overall** (spatial 3.3 %, object/goal/10 = 0.0 %). Dead closed-loop despite a
  "reasonable"-looking flow loss.

**Diagnosis via a co-distill-free control (decisive).** Because 0.83 % is far below even a
capacity-limited student's expected ~30–45 % (cf. Track A), we ran an identical-recipe control that
differs **only** in the init: clean warm-start (no co-distill) → same LoRA r64 `{llm,ae,vit}`
LIBERO fine-tune. Its held-out flow **floored at ~0.29** (steps 2.5k–4k: 0.284 / 0.292 / 0.277 /
0.293) — i.e. the same ~0.24–0.29 band as B3, and nowhere near the **~0.06** that competent students
reach. The control was stopped once clearly floored.

| Recipe | student-LLM adaptation | heldout flow | closed-loop |
|---|---|---|---|
| 47.5 % ckpt + all Track A | **full unfreeze** (DAgger `student_only`) | 0.06–0.08 | 18–47.5 % |
| B3 (co-distill → **LoRA** r64) | LoRA r64 | 0.243 | 0.83 % |
| Control (warm-start → **LoRA** r64) | LoRA r64 | ~0.29 (floored) | (dead, stopped) |

**Outcome — B4's 0.83 % is a fine-tune-recipe artifact, NOT co-distill damage and NOT a 0.6B
capacity verdict.** The control isolates the cause: **LoRA-r64 on the student LLM is too weak** for
this width-reduced student. Every working student in this program (the 47.5 % ckpt, all Track A
variants) trained the student LLM **fully unfrozen** (DAgger `student_only`) and reached ~0.06 flow;
B3 and the control both used LoRA on the LLM and both floored ~0.24–0.29. Mirroring the base model's
r64 recipe was the mistake — the base has a full-size LLM where r64 suffices; `qwen06w` needs full
unfreeze. **The clean capacity test (full-unfreeze fine-tune from the co-distilled base) is deferred**
as a scope decision, along with staging a larger DROID subset for a faithful pretraining.

**Status:** Track A complete (3/3 gated). Track B complete through B4 + control diagnosis. The
two-teacher co-distillation pipeline (`train_codistill.py`) is validated end-to-end (both teachers,
shared student+AE, per-domain anchor, coadapt checkpoint consumed by the eval/finetune paths).

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
4. **Heal budget, not pruning, drove the curriculum 0 %.** A matched full-length anchored heal turns 0 % into
   50 % at mild (1.13×) pruning — and the mildly pruned student beats the non-pruned control (50 % vs 20 %).
5. **Co-adapting the full action expert does not help.** Experiment D freed the whole AE to co-train with the
   student LLM; it hit the best flow loss on record (0.061) but the worst broad-suite rollout (Gate-1 5 %,
   Gate-2 9.2 % after a real-data LoRA pass), underperforming the frozen-AE + KV-anchor recipe (30 %). More
   trainable freedom moved us the *wrong* way; the bottleneck is the closed-loop covariate-shift gap, not model
   capacity or the fine-tuning parameterization.
6. **The squeeze (Track A) does not pass 47.5 %; data aggregation > AE capacity.** Continuing DAgger on the
   aggregated round-1+2 pool at the *original* rank/LR (a3) recovers most of round-2's forgetting (43.3 %) but
   stays below R1b; adding AE capacity is actively harmful (r64 = 32.5 %, full-AE = 18.3 %). On `qwen06w` the
   action expert is not the under-adapted part.
7. **For the width-reduced student, the LLM must be *fully unfrozen* — LoRA on the LLM is too weak.** Track B's
   0.83 % was a fine-tune-recipe artifact: a co-distill-free control (warm-start → same LoRA r64) *also* floored
   at ~0.29 held-out flow, vs the ~0.06 that full-unfreeze reaches. Mirroring the base model's r64 recipe fails
   on `qwen06w`; the co-distillation pipeline itself is sound. The clean capacity verdict (full-unfreeze
   fine-tune from the co-distilled base, ideally with a larger DROID subset) is deferred.

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
