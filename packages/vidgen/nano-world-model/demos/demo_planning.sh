#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Application: CEM/MPC planning over the diffusion world model (DINO-WM protocol).
# Upstream defaults from src/scripts/eval/planning_{pusht,point_maze}.sh:
#   horizon=5, replan_every=5, goal_source=dset, full_sequence scheduling, 50 episodes.
# Writes planning_results/episode_*.mp4 + planning_results.json (success_rate, state_dist).
#
# ENV=pusht (default, light sim deps) or point_maze (needs MuJoCo 2.10 + d4rl — best effort).
#
# COMPUTE PROFILE (PLANNING_PROFILE): the upstream CEM budget is a multi-DAY run on a single
# gfx1151 (Strix-Halo) APU — one DFoT rollout of the CEM population costs minutes, and the loop
# is (opt_steps x replans x episodes) such rollouts. We therefore default to a reduced "fast"
# profile that still exercises the full CEM/MPC loop and yields watchable episode videos + a
# success rate overnight. Set PLANNING_PROFILE=paper (300/30/30, 50 evals) or =default
# (100/10/30, 20 evals) to reproduce the upstream budgets when time/hardware allow.
set -euo pipefail

ENV_NAME="${ENV_NAME:-pusht}"
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
REPO=/repos/nanowm
OUT_DIR="${OUT_DIR:-/outputs}"

declare -A CKPTS=(
  [pusht]="knightnemo/nanowm-b2-dino-wm-pusht-100k"
  [point_maze]="knightnemo/nanowm-b2-dino-wm-point-maze-30k" )

# --- compute profiles: CEM_SAMPLES CEM_TOPK CEM_OPTSTEPS DDIM_STEPS N_EVALS MAXSTEPS_PUSHT ---
PLANNING_PROFILE="${PLANNING_PROFILE:-fast}"
case "$PLANNING_PROFILE" in
  paper)   PS=300; PT=30; PO=30; PD=20; PE=50; PM=20 ;;   # upstream planning=dino_wm_pusht
  default) PS=100; PT=10; PO=30; PD=20; PE=20; PM=20 ;;   # upstream planning/base.yaml
  fast|*)  PS=48;  PT=8;  PO=3;  PD=8;  PE=4;  PM=10 ;;   # gfx1151 overnight demo budget
esac
# Per-knob overrides win over the profile.
CEM_SAMPLES="${CEM_SAMPLES:-$PS}"; CEM_TOPK="${CEM_TOPK:-$PT}"; CEM_OPTSTEPS="${CEM_OPTSTEPS:-$PO}"
DDIM_STEPS="${DDIM_STEPS:-$PD}";   N_EVALS="${N_EVALS:-$PE}";   N_PLOT="${N_PLOT:-4}"
declare -A MAXSTEPS=( [pusht]="${MAXSTEPS_PUSHT:-$PM}" [point_maze]="${MAXSTEPS_MAZE:-10}" )

CKPT="${CKPT:-${CKPTS[$ENV_NAME]}}"
RUN_DIR="$OUT_DIR/planning_$ENV_NAME"
mkdir -p "$RUN_DIR"

# ensure the matching DINO-WM dataset (planning replays GT actions from a val trajectory)
DOMAIN="$ENV_NAME" bash /ryzers/scripts/download_datasets.sh || echo "[plan] dataset fetch warn"

cd "$REPO"
CKPT_ESCAPED="${CKPT//=/\\=}"
echo "[plan] env=$ENV_NAME ckpt=$CKPT profile=$PLANNING_PROFILE n_evals=$N_EVALS"
echo "[plan] CEM samples=$CEM_SAMPLES topk=$CEM_TOPK opt_steps=$CEM_OPTSTEPS ddim=$DDIM_STEPS max_steps=${MAXSTEPS[$ENV_NAME]}"
START=$(date +%s.%N)
python src/main.py \
  experiment=planning \
  model=nanowm_b2 \
  dataset=dino_wm/"$ENV_NAME" \
  "ckpt_path=${CKPT_ESCAPED}" \
  planning.env_name="$ENV_NAME" \
  planning.goal_source=dset \
  planning.goal_H=5 \
  planning.horizon=5 \
  planning.replan_every=5 \
  planning.max_episode_steps="${MAXSTEPS[$ENV_NAME]}" \
  planning.n_evals="$N_EVALS" \
  planning.n_plot_samples="$N_PLOT" \
  planning.num_sampling_steps="$DDIM_STEPS" \
  planning.cem.num_samples="$CEM_SAMPLES" \
  planning.cem.topk="$CEM_TOPK" \
  planning.cem.opt_steps="$CEM_OPTSTEPS" \
  model.scheduling_mode=full_sequence \
  wandb.enabled=false \
  hydra.run.dir="$RUN_DIR"
END=$(date +%s.%N)
echo "[plan] done in $(python -c "print(f'{${END}-${START}:.1f}')")s -> $RUN_DIR"
ls -la "$RUN_DIR/planning_results" 2>/dev/null || true
