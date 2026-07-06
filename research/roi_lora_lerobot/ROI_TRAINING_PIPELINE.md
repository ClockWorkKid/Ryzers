# ROI-guided pre-ViT pruning — trainable pipeline (MolmoAct2 LoRA-VLM, MI300X)

End-to-end pipeline that bakes stage-D **pre-ViT vision-token pruning** into the
MolmoAct2 LeRobot LoRA-VLM training image and trains it multi-GPU on AMD MI300X.
Pruning is a runtime no-op unless enabled, so the same image runs baseline too.

## 1. What "ROI pre-ViT pruning" does
At the patch-embedding -> ViT seam (`molmoact2_hf_model/modeling_molmoact2.py::encode_image`):
```
x = patch_embedding(images); x = add_pos_emb(x)         # [B, N=729, 1152]
scores  = score_patches(x, select)                      # ROI saliency
keep    = topk(scores, K=round(keep_frac*N))            # [B, K]
x_kept  = gather(x, keep)                               # ViT resblocks run on K << N
feats   = resblocks(x_kept)
feats   = scatter_back(feats, keep, N, placeholder)     # learned mask token fills pruned slots
```
Scatter-back preserves the full 729-patch grid so the downstream pooling /
`pooled_patches_idx` contract (and LLM token count) is unchanged — the saving is
in the **ViT resblocks**.

Selectors (`policy.roi_prune_select`): `random` | `energy` (‖post-pos-emb‖, param-free) |
`gate` (learnable 2-layer MLP + straight-through top-k). `placeholder=mask` uses a
learned mask token (registered on the backbone, added to a dedicated optim group).

## 2. Config knobs (`configuration_molmoact2.py`)
```
--policy.roi_prune_enable=true
--policy.roi_prune_select=energy|gate|random
--policy.roi_prune_keep_frac=0.25|0.5|0.75    # fraction of patches kept
--policy.roi_prune_placeholder=mask|zeros
--policy.roi_prune_gate_hidden=256            # gate MLP width
--policy.roi_prune_gate_lr=<lr>               # dedicated LR for gate/mask params
```
Defaults keep behaviour identical to upstream (enable=false, keep_frac=1.0).

## 3. Bake into the image
The ROI code is layered onto the editable lerobot install via a build-context
overlay (`docker/roi_overlay/` = `roi_prune.py` + 3 patched molmoact2 files):
```dockerfile
COPY roi_overlay/ /opt/lerobot/src/lerobot/policies/molmoact2/
RUN chmod -R a+rX /opt/lerobot/src/lerobot/policies/molmoact2   # non-root runtime user must traverse
```
Build: `docker build -t molmoact2-lerobot-train:rocm942 -f Dockerfile.instinct-gfx942 .`
(base ROCm 7.2 / torch 2.10; gfx942). The sanity stage asserts `roi_prune` imports
and the ROI config fields exist.

> Gotcha (fixed): `COPY` inherits build-context perms/owner (root); without the
> `chmod -R a+rX` the copied `molmoact2_hf_model/` lands as `drwx------ root` and
> the non-root training user gets `ModuleNotFoundError` at import.

## 4. Launch (multi-GPU, chained 2h + resume)
Per config, run `accelerate launch --num_processes=<G> --multi_gpu --mixed_precision=bf16`
with the ROI flags plus the standard LoRA-VLM args:
```
-m lerobot.scripts.lerobot_train \
  --dataset.repo_id=allenai/MolmoAct2-LIBERO-Dataset --dataset.video_backend=pyav \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.enable_lora_vlm=true --policy.action_mode=both \
  --policy.model_dtype=bfloat16 --policy.num_flow_timesteps=8 --policy.gradient_checkpointing=true \
  --policy.roi_prune_enable=true --policy.roi_prune_select=energy --policy.roi_prune_keep_frac=0.5 \
  --policy.roi_prune_placeholder=mask \
  --batch_size=8 --steps=6000 --save_freq=500 --output_dir=<out>
```
A SLURM batch script chains 2h jobs (`--dependency=afterany`) that resume from
`checkpoints/last` so runs survive the wall-clock limit. For co-scheduling several
keep-levels on one 8-GPU node, request `--gres=gpu:<G>` per job and forward SLURM's
`ROCR_VISIBLE_DEVICES` into the container as `HIP_VISIBLE_DEVICES` (the
`--device=/dev/dri` mount otherwise exposes all GPUs).

## 5. Results — this run (4 configs, effective batch 16, 2 GPU each)
Convergence (flow-matching loss) — all levels stable and monotonic; see
`artifacts/roi_training/loss_curves.png`. More pruning -> higher loss but still converges.

Vision-encoder compute saved (exact, N=729 patches, 25 resblocks, d=1152) —
`artifacts/roi_training/compute_saved.png`:

| prune | keep_frac | kept patches | ViT resblock FLOP reduction | speedup |
| ----- | --------- | ------------ | --------------------------- | ------- |
| 25%   | 0.75      | 547/729      | 26.8%                       | 1.37x   |
| 50%   | 0.50      | 364/729      | 52.6%                       | 2.11x   |
| 75%   | 0.25      | 182/729      | 76.9%                       | 4.33x   |

## 6. Finding — the `energy` selector is positional-dominated
`‖patch_embed + pos_emb‖` is dominated by the positional embedding (fixed corner/
grid hotspots), so `energy` keeps a near-fixed spatial pattern across frames
(`mask_viz.png`, col "energy"). The content-only norm `‖patch_embed‖` (no pos) is
mildly image-adaptive and tracks the gripper/objects. Implication: for true content
ROI, score on the pre-pos embedding or use the learnable `gate` selector. The
current design still trains stably and delivers the FLOP savings above.

## 7. Files
- `roi_prune/pre_vit_prune.py` (+ `test_pre_vit_prune.py`): standalone primitives + unit tests.
- `vendored/molmoact2/`, `docker/roi_overlay/`: the ROI source layered into the image.
- `docker/Dockerfile.instinct-gfx942`, `docker/Dockerfile.strix-gfx1151`: images (ROI baked).
- `artifacts_gen/`: `compute_saved.py`, `plot_loss_curves.py`, `plot_masks.py`.
- `artifacts/roi_training/`: generated figures + data.
