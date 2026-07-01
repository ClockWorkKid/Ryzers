# Experiment Guide

This guide indexes the scripts used across the research branch. It is focused on reproducibility on the Strix test box, not packaging for upstream Ryzers.

## Core research themes

- Post-ViT random token pruning
- Pre-ViT random patch-group pruning
- Async hold/blend realtime planning
- Action-attention feedback pruning

## Main runners

- `./run_ablation_detached.sh`
- `./run_groupdrop_detached.sh`
- `./run_preenc_detached.sh`
- `./run_attnfb_smoke_detached.sh`
- `./run_attnfb_validate.sh`
- `./run_attnfb_bench.sh`
- `./rt_smoothness_run.sh`
- `./interactive_rt_groupdrop_run.sh`
- `./rt_latency_timeline_run.sh`
- `./rt_latency_timeline_groupdrop_run.sh`

## Key patch and analysis scripts

- `./patch_vis_reduce.py`
- `./patch_vis_groupdrop.py`
- `./patch_vis_preenc.py`
- `./patch_vis_attnfeedback.py`
- `./patch_lerobot_attnfb.py`
- `./bench_opt_study.py`
- `./bench_groupdrop_latency.py`
- `./bench_preenc_latency.py`
- `./bench_attnfb_latency.py`
- `./validate_groupdrop.py`
- `./validate_preenc.py`
- `./validate_attnfeedback.py`
- `./plot_ablation.py`
- `./plot_preenc.py`
- `./plot_rt_smoothness.py`
- `./plot_attnfb_overlay.py`
- `./rt_smoothness_ablation.py`
- `./rt_blend_video.py`
- `./rt_latency_timeline.py`

## Data and outputs

- Presentation snapshots: `data/presentation/`
- Existing research data: `data/efficiency/`
- Existing research figures: `assets/efficiency/`

## Suggested execution order

1. Patch target model file (`patch_vis_*` or `patch_lerobot_attnfb.py`).
2. Run smoke check (`smoke_*` or realtime interactive runner).
3. Run detached sweep/bench script for the variant.
4. Validate + plot (`validate_*`, `plot_*`).
5. Capture summary CSV/PNG into `data/` and `assets/`.