#!/usr/bin/env bash
#SBATCH --job-name=lp_policy
#SBATCH --output=%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
# Turn each merged curriculum-stage HF checkpoint into a lerobot policy dir that lerobot-eval can
# load. Per stage (env STAGES, default "s1 s2 s3 s4 s5"):
#   [1/3] filter_config.py  -> config.json (rank-agnostic, checkpoint_path=/ckpt, LoRA off)
#   [2/3] rewrap_policy.py   -> lerobot model.safetensors (merged HF weights re-wrapped)
#   [3/3] gen_processors.py  -> norm_tag=libero pre/post processors
# Runs inside the eval image. Requires a training lerobot config saved as
# $POLICYROOT/_lerobot_cfg_src.json (copy any trained stage's config.json there once).
#
# Env: WORK, IMG, MERGEDROOT, POLICYROOT, SCR, STAGES.
set -u
SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"
WORK="${WORK:-$HOME/molmoact2}"
IMG="${IMG:-molmoact2-libero-plus-eval:rocm950}"
MERGEDROOT="${MERGEDROOT:-$WORK/outputs/lp_curric_merged}"
POLICYROOT="${POLICYROOT:-$WORK/eval_ckpts}"
STAGES="${STAGES:-s1 s2 s3 s4 s5}"
mkdir -p "$POLICYROOT"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "NO_EVAL_IMAGE $IMG on $(hostname)"; exit 1; }
[ -f "$POLICYROOT/_lerobot_cfg_src.json" ] || { echo "MISSING $POLICYROOT/_lerobot_cfg_src.json (copy a trained stage config.json there)"; exit 2; }
DRUN(){ docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined "$@"; }

for st in $STAGES; do
  MERGED="$MERGEDROOT/$st"
  POLICY="$POLICYROOT/lp_${st}_policy"
  echo "=============== $(date -Is) build policy for $st ==============="
  [ -d "$MERGED" ] || { echo "SKIP $st: merged missing $MERGED"; continue; }
  mkdir -p "$POLICY"
  echo "--- [1/3] filter config ---"
  DRUN -v "$POLICY":/policy -v "$POLICYROOT":/src -v "$SCR":/work \
    -e SRC_CFG=/src/_lerobot_cfg_src.json -e OUT_CFG=/policy/config.json \
    "$IMG" python3 /work/filter_config.py || { echo "FILTER_FAIL $st"; continue; }
  echo "--- [2/3] rewrap HF->lerobot weights (GPU) ---"
  DRUN -v "$MERGED":/ckpt:ro -v "$POLICY":/policy -v "$SCR":/work \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HIP_VISIBLE_DEVICES=0 \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    "$IMG" python3 /work/rewrap_policy.py || { echo "REWRAP_FAIL $st"; continue; }
  echo "--- [3/3] gen norm processors ---"
  DRUN -v "$POLICY":/policy -v "$SCR":/work \
    -e HIP_VISIBLE_DEVICES=0 -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    "$IMG" python3 /work/gen_processors.py || { echo "PROC_FAIL $st"; continue; }
  echo "--- policy dir $st ---"; ls -la "$POLICY"
  [ -s "$POLICY/model.safetensors" ] && echo "OK $st model.safetensors present" || echo "MISSING model.safetensors $st"
done
echo "=== ALL DONE $(date -Is) ==="
