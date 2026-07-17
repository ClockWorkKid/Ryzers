# MolmoAct2 LLM-backbone distillation (hybrid: functional + KV-anchoring)

Distill the MolmoAct2-LIBERO Qwen3 LLM backbone (36 layers) into a width-reduced student
that the **frozen flow-matching action expert** can still use. We train the **action task loss**
(functional distillation) **plus an all-layer teacher-KV anchoring term** — a hybrid of functional
and representational distillation. See **`HYBRID_DISTILL_REPORT.md`** for full results.

## Why hybrid (functional + anchoring), not either alone
The action expert cross-attends to **per-layer LLM K/V** (one DiT block per LLM layer, 1:1).
Two single-objective attempts both fail:
- **Pure representational** (match KV only): ~0.99 KV cosine but **0% closed-loop** — a deceptive,
  over-tight target that starves the task objective.
- **Functional-only** (flow loss only): the student's KV becomes **orthogonal** to the teacher's at
  identical inputs (cos ≈ 0) and it never grasps — **0% closed-loop**.

The fix is to keep the functional flow loss but **anchor** the student's per-layer KV field to the
teacher's (directional cosine + normalized-magnitude), so the student cannot collapse into an
orthogonal shortcut while the action expert co-adapts (LoRA on `context_{k,v}_proj`). This lifts the
6× student from **0% → 30%** closed-loop with first-chunk KV cosine **0.001 → 0.822**.

## Workstream B — top-down token-reduction (exploratory)
A second track runs a compressed student on the **25% task-relevant token stream** from the trained
ROI gate + FastV front-end (prune-before-student), supervised by the full-token teacher. Result: the
6× student trains cleanly (flow 0.046) but fails closed-loop (0%), while the **full-width teacher
through the identical harness scores 100%** — a controlled negative showing the 6× width, not the
token stream, is the limit. Details in `HYBRID_DISTILL_REPORT.md`.

## Hard constraints (from the shipped checkpoint config)
- Teacher LLM: 36 layers, hidden 2560, 32 Q / 8 KV heads x 128 (kv_dim 1024), interm 9728,
  rope_theta 5e6, qk_norm qwen3, vocab 154624 (+128).
- Action expert: hidden 768, **36 layers (one per LLM layer -> student depth is FIXED at 36)**,
  consumes per-layer KV of dim `8*128=1024` via frozen `context_{k,v}_proj` (1024->768).
- Reduction therefore comes from **width**, not depth.

## Student (experiment 1: Qwen3-0.6B width, ~6x decoder FLOP reduction)
- 36 layers, hidden 1024, 16 Q / 8 KV heads x 128 (kv_dim 1024 == teacher -> KV interface fits
  the frozen action-expert projections with no surgery), interm 3072, rope_theta 5e6, qk_norm.
- Consumes the teacher's fused `inputs_embeds` [B,N,2560] via `down_proj` 2560->1024; emits
  `last_hidden_state` [B,N,2560] via `up_proj` (for the discrete head) and per-layer KV
  [B,8,N,128] in the exact teacher contract (post-QK-norm, post-RoPE, pre-GQA-repeat).
- Warm-started from pretrained Qwen3-0.6B (28 layers) via nearest-layer remap to 36 layers.

## Modules (validated independently against real data before assembly; see .cursorrules rule 2)
1. `student.py`         - StudentTextModel: drop-in for `model.model.transformer`, teacher KV contract.
2. `warmstart.py`       - Qwen3-0.6B -> student weight remap (width-reduced, 28->36 layers).
3. `data.py`            - real LIBERO batches; `capture_teacher_reduced` = gate+FastV prune-before-student front-end (B).
4. `joint_patch.py`     - attach student + action-expert LoRA; hybrid `student_joint_loss` (flow + all-layer KV anchor).
5. `train_joint.py`     - Workstream A trainer (`--anchor-weight/-mode/-beta/-keep-teacher`); flow-only reload gate.
6. `eval_joint.py`      - swap student for closed-loop LIBERO eval.
7. `diag_closed_loop.py`- teacher-vs-student closed-loop diagnostic: success, action MSE, first-chunk KV cosine.
8. `train_reduced.py`   - Workstream B trainer: student on the 25% reduced stream (+ optional anchoring on kept tokens).
9. `validate_reduced.py`- module validation of the reduced-token path (student fwd/bwd on real reduced batches).
10. `eval_reduced.py`   - Workstream B closed-loop eval (prune-before-student + native denoise); `--teacher-through-reduced` control.

Weights/large artifacts live only on the remote compute store; only small curves/configs are mirrored
to `artifacts/llm_distill/`.
