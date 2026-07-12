#!/usr/bin/env bash
# Standalone closed-loop LIBERO eval launcher for ROI-pruned checkpoints. Runs the
# in-container sweep (roi_eval_sweep.sh) with the eval overlay bind-mounted, so the
# trained gate (+ optional FastV cut) drives single-pass pruning in closed loop --
# no teacher, no dual pass. One machine, N GPUs; SLURM orchestration is optional and
# not required (chain segments externally with --dependency if you want).
#
# Regime:
#   gate-only  : leave ROI_FASTV_INFER unset  (pre-ViT scatter-back only)
#   FastV-wired: ROI_FASTV_INFER=1            (gate also drives the in-LLM cut at L0)
#
# Usage (edit RUNS to point at your checkpoints under $OUT):
#   RUNS="fv050:roi_fastv_keep050_fv050:012000" NGPU=8 WPG=2 bash roi_eval.sh
#   RUNS="pred:roi_predict_keep050_fv050_l09:012000" ROI_FASTV_INFER=1 \
#     RESDIR=/outputs/roi_eval_predict bash roi_eval.sh
set -u
WORK="${WORK:-$HOME/molmoact2}"
CACHE="${CACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/outputs}"
OV="${OV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/roi_overlay" && pwd)}"
SWEEP="${SWEEP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/roi_eval_sweep.sh}"
MP=/opt/lerobot/src/lerobot/policies/molmoact2
IMG="${IMG:-molmoact2-lerobot-eval:rocm942}"

# Stage the sweep script where the container reads it (normalize CRLF from Windows edits).
cp -f "$SWEEP" "$OUT/roi_eval_sweep.sh"
sed -i 's/\r$//' "$OUT/roi_eval_sweep.sh" 2>/dev/null || true

# gate_predict checkpoints also need the temporal-split processor overlay if present.
PROC_MOUNT=""
[ -f "$OV/processor_molmoact2.py" ] && PROC_MOUNT="-v $OV/processor_molmoact2.py:$MP/processor_molmoact2.py:ro"

docker run --rm ${EXTRA_DOCKER:-} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache \
  -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
  -e N_EP="${N_EP:-10}" -e SEED="${SEED:-1000}" -e NGPU="${NGPU:-8}" -e WPG="${WPG:-2}" \
  -e STD_TASKS="${STD_TASKS:-0 1 2 3 4 5 6 7 8 9}" -e L90_TASKS="${L90_TASKS:-0 9 18 27 36 45 54 63 72 81}" \
  -e RESDIR="${RESDIR:-}" \
  -e RUNS="${RUNS:-}" \
  -e SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}" \
  -e MUJOCO_GL="${MUJOCO_GL:-osmesa}" -e PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" \
  -e ROI_PRUNE_DEBUG="${ROI_PRUNE_DEBUG:-}" -e TASK_IDS_ALL="${TASK_IDS_ALL:-}" \
  -e ROI_FASTV_INFER="${ROI_FASTV_INFER:-}" \
  -e MASK_DUMP="${MASK_DUMP:-}" \
  -v "$CACHE":/cache -v "$OUT":/outputs \
  -v "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  -v "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  -v "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  -v "$OV/molmoact2_hf_model/modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  $PROC_MOUNT \
  "$IMG" \
  bash -lc "bash /outputs/roi_eval_sweep.sh"
