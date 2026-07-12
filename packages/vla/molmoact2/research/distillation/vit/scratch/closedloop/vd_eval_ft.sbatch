#!/usr/bin/env bash
#SBATCH --job-name=vd_eval_ft
#SBATCH -p mi210
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_eval_%x_%j.log
# Closed-loop eval of the FINETUNED distilled-full model (student_vit + LoRA +
# action expert). Full run (100 ep/suite):
#   sbatch --export=ALL,CKPT=/outputs/student_ft_lora/checkpoints/012000/pretrained_model vd_eval_ft.sbatch
# Quick load smoke (1 suite, 1 task, 1 ep):
#   sbatch -t 00:40:00 --export=ALL,CKPT=<ckpt>,SUITES=libero_spatial,TASK_IDS=0,N_EP=1,RESDIR=/outputs/vd_eval_ft_smoke vd_eval_ft.sbatch
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
MODE=ft
# stage launchers + sweep + student module into /outputs (bind target)
for f in run_eval_ft.py run_eval_student.py vit_student.py vd_eval_sweep.sh; do
  [ -f "$HOME/$f" ] && { cp -f "$HOME/$f" "$OUT/$f"; sed -i 's/\r$//' "$OUT/$f"; }
done
# variant -> student build kwargs + distilled placeholder ckpt (arch must match the
# finetuned checkpoint that --policy.path loads). NAME unset -> original ft-hybrid.
NAME="${NAME:-}"
case "$NAME" in
  siglip_nano)  STKW='{"variant":"tinyvit","dim":240,"num_attn_blocks":4,"attn_heads":8}'; PLACE=/outputs/siglip_nano_full.pt ;;
  hybrid_droid) STKW='{"variant":"hybrid","dim":256,"num_conv_blocks":4,"num_attn_blocks":3,"attn_heads":8}'; PLACE=/outputs/hybrid_droid_full.pt ;;
  cnn_fpga)     STKW='{"variant":"cnn","dim":384,"num_conv_blocks":16,"conv_kernel":3}'; PLACE=/outputs/cnn_fpga_full.pt ;;
  *)            STKW='{}'; PLACE=/outputs/hybrid_full.pt ;;
esac
if [ -n "$NAME" ]; then
  CKPT="${CKPT:-/outputs/ft_${NAME}/checkpoints/012000/pretrained_model}"
  RESDIR="${RESDIR:-/outputs/vd_eval_ft_${NAME}}"
else
  CKPT="${CKPT:-/outputs/student_ft_lora/checkpoints/012000/pretrained_model}"
  RESDIR="${RESDIR:-/outputs/vd_eval_ft}"
fi
NG=$(echo "${ROCR_VISIBLE_DEVICES:-0,1,2,3}" | awk -F, '{print NF}')
echo "[vd_eval_ft] host=$(hostname) job=${SLURM_JOB_ID:-none} name=${NAME:-hybrid} ckpt=$CKPT kw=$STKW ngpu=$NG t=$(date -Is)"
[ -f "$SIF" ] || { echo "[vd_eval_ft] SIF missing: $SIF"; exit 2; }
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa \
  --env MODE="$MODE" --env N_EP="${N_EP:-5}" --env SEED="${SEED:-1000}" \
  --env NGPU="$NG" --env STAGGER="${STAGGER:-120}" \
  --env RESDIR="$RESDIR" \
  --env CKPT="$CKPT" \
  --env SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}" \
  --env TASK_IDS="${TASK_IDS:-0,1,2,3,4}" \
  --env VIT_STUDENT_MODULE=/outputs/vit_student.py \
  --env VIT_STUDENT_CKPT="$PLACE" \
  --env VIT_STUDENT_KWARGS="$STKW" \
  --bind "$OUT":/outputs \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "bash /outputs/vd_eval_sweep.sh"
echo "[vd_eval_ft] DONE rc=$? t=$(date -Is)"
