#!/usr/bin/env bash
#SBATCH --job-name=llmd_eval
#SBATCH -p mi210
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/llmd_eval_%x_%j.log
# Run D closed-loop LIBERO eval: jointly distilled student ViT (hybrid_droid) +
# thin-twin student LLM + KV adapters (+ LoRA adapters when EVAL_LORA=1), loading
# trained weights from --policy.path (CKPT). Actions are generated from STUDENT KV
# (run_eval_joint.py patches generate_actions_from_inputs), so success rate
# reflects the compressed model. One process per suite, staggered loads.
#   Full:  sbatch ~/llmd_eval.sbatch          (CKPT=llmd_lora last, EVAL_LORA=1)
#   Smoke: SMOKE=1 sbatch ~/llmd_eval.sbatch  (1 suite/1 task/1 ep vs joint ckpt)
set -u
U=$USER
CACHE=/shared_nobackup/$U/molmoact2/hf_cache
OUT=/shared_nobackup/$U/molmoact2/outputs
SIF=/shared_nobackup/$U/molmoact2/images/molmoact2-lerobot-eval.sif
OV=$HOME/roi_lerobot_overlay
MP=/opt/lerobot/src/lerobot/policies/molmoact2
MODE=joint

# stage launcher + student + lora + vit student + sweep into the bind target
mkdir -p "$OUT/llm_distill"
for f in run_eval_joint.py student.py lora.py; do
  [ -f "$HOME/llm_distill/$f" ] && { cp -f "$HOME/llm_distill/$f" "$OUT/llm_distill/$f"; sed -i 's/\r$//' "$OUT/llm_distill/$f"; }
done
for f in vit_student.py vd_eval_sweep.sh; do
  [ -f "$HOME/$f" ] && { cp -f "$HOME/$f" "$OUT/$f"; sed -i 's/\r$//' "$OUT/$f"; }
done

VIT_KW="${VIT_KW:-{\"variant\":\"hybrid\",\"dim\":256,\"num_conv_blocks\":4,\"num_attn_blocks\":3,\"attn_heads\":8}}"
STCKPT="${STCKPT:-/outputs/llm_distill/stage1/student_final.pt}"
VIT_PLACE="${VIT_PLACE:-/outputs/hybrid_droid_full.pt}"
EVAL_LORA="${EVAL_LORA:-1}"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  # validate the inference-path swap against the joint (non-LoRA) checkpoint
  CKPT="${CKPT:-/outputs/llmd_joint/checkpoints/last/pretrained_model}"
  EVAL_LORA="${EVAL_LORA_SMOKE:-0}"
  RESDIR="${RESDIR:-/outputs/vd_eval_joint_smoke}"
  SUITES="${SUITES:-libero_spatial}"; TASK_IDS="${TASK_IDS:-0}"; N_EP="${N_EP:-1}"
else
  CKPT="${CKPT:-/outputs/llmd_lora/checkpoints/last/pretrained_model}"
  RESDIR="${RESDIR:-/outputs/vd_eval_joint}"
  SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
  TASK_IDS="${TASK_IDS:-0,1,2,3,4}"; N_EP="${N_EP:-5}"
fi

NG=$(echo "${ROCR_VISIBLE_DEVICES:-0,1,2,3}" | awk -F, '{print NF}')
echo "[llmd_eval] host=$(hostname) job=${SLURM_JOB_ID:-none} smoke=$SMOKE ckpt=$CKPT eval_lora=$EVAL_LORA kw=$VIT_KW suites=[$SUITES] ngpu=$NG t=$(date -Is)"
[ -f "$SIF" ] || { echo "[llmd_eval] SIF missing: $SIF"; exit 2; }
[ -d "$OUT/${CKPT#/outputs/}" ] || { echo "[llmd_eval] ERROR ckpt dir missing: $CKPT"; exit 4; }
[ -f "$OUT/${STCKPT#/outputs/}" ] || { echo "[llmd_eval] ERROR stage1 ckpt missing: $STCKPT"; exit 4; }
[ -f "$OUT/llm_distill/run_eval_joint.py" ] || { echo "[llmd_eval] missing run_eval_joint.py"; exit 3; }
[ -f "$OUT/llm_distill/lora.py" ] || { echo "[llmd_eval] missing lora.py"; exit 3; }
[ -f "$OUT/vit_student.py" ] || { echo "[llmd_eval] missing vit_student.py"; exit 3; }

apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env USER="$U" --env LOGNAME="$U" \
  --env HF_HOME=/cache --env XDG_CACHE_HOME=/cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_DOWNLOAD_TIMEOUT=120 --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa \
  --env MODE="$MODE" --env N_EP="$N_EP" --env SEED="${SEED:-1000}" \
  --env NGPU="$NG" --env STAGGER="${STAGGER:-120}" \
  --env RESDIR="$RESDIR" --env CKPT="$CKPT" \
  --env SUITES="$SUITES" --env TASK_IDS="$TASK_IDS" \
  --env VIT_STUDENT_MODULE=/outputs/vit_student.py \
  --env VIT_STUDENT_CKPT="$VIT_PLACE" --env VIT_STUDENT_KWARGS="$VIT_KW" \
  --env LLMD_STUDENT_CKPT="$STCKPT" --env LLMD_STUDENT_MODULE=/outputs/llm_distill/student.py \
  --env LORA_MODULE=/outputs/llm_distill/lora.py --env EVAL_LORA="$EVAL_LORA" \
  --env LORA_R="${LORA_R:-16}" --env LORA_ALPHA="${LORA_ALPHA:-32}" --env LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
  --bind "$OUT":/outputs \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" bash -lc "bash /outputs/vd_eval_sweep.sh"
echo "[llmd_eval] DONE rc=$? t=$(date -Is)"
