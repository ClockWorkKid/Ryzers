#!/usr/bin/env bash
# Closed-loop LIBERO eval sweep for the ViT-distillation study. Runs INSIDE the
# eval SIF. One shared pristine checkpoint (base_teacher) for both modes;
# MODE=student swaps encode_image for the distilled student via
# run_eval_student.py. Identical seeds -> bit-identical episodes across modes.
#
# One process PER SUITE (all tasks via --env.task_ids) so the ~11GB checkpoint is
# loaded once per suite (4 loads, not 20). Launches are STAGGERed so their
# CPU-heavy load phases do not overlap (avoids the OOM that kills co-loading
# workers on MI210).
#
#   MODE=teacher : lerobot-eval on the pristine MolmoAct2-LIBERO policy
#   MODE=student : same policy, ViT seam replaced by hybrid_full.pt student
#
# Grid: SUITES x TASK_IDS x N_EP. Default 4 suites x 5 tasks x 5 ep = 100 eps.
set -uo pipefail
OUT=/outputs
MODE="${MODE:-teacher}"
CKPT="${CKPT:-/outputs/base_teacher/checkpoints/000000/pretrained_model}"
RESDIR="${RESDIR:-$OUT/vd_eval_$MODE}"
N_EP="${N_EP:-5}"; SEED="${SEED:-1000}"
NGPU="${NGPU:-4}"; STAGGER="${STAGGER:-120}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4}"
CAM='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
mkdir -p "$RESDIR"
RESULTS=$RESDIR/results.csv
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_BLAS_PREFER_HIPBLASLT=0

if [ "$MODE" = "student" ]; then
  export VIT_STUDENT_CKPT="${VIT_STUDENT_CKPT:-/outputs/hybrid_full.pt}"
  export VIT_STUDENT_MODULE="${VIT_STUDENT_MODULE:-/outputs/vit_student.py}"
  EVAL=(python /outputs/run_eval_student.py)
elif [ "$MODE" = "ft" ]; then
  # Finetuned distilled-full model: attach student_vit arch, load finetuned
  # student + LoRA + action-expert from --policy.path (CKPT). CKPT must point at
  # the finetuned checkpoint's pretrained_model dir.
  export VIT_STUDENT_MODULE="${VIT_STUDENT_MODULE:-/outputs/vit_student.py}"
  export VIT_STUDENT_CKPT="${VIT_STUDENT_CKPT:-/outputs/hybrid_full.pt}"
  EVAL=(python /outputs/run_eval_ft.py)
elif [ "$MODE" = "joint" ]; then
  # Run D: jointly distilled student ViT + thin-twin student LLM (+ optional
  # LoRA). Attaches all student modules + KV adapters, loads trained weights from
  # --policy.path (CKPT), and generates actions from student KV. LLMD_STUDENT_CKPT
  # / EVAL_LORA / VIT_STUDENT_KWARGS / LORA_* pass through from the sbatch env.
  export VIT_STUDENT_MODULE="${VIT_STUDENT_MODULE:-/outputs/vit_student.py}"
  export LLMD_STUDENT_MODULE="${LLMD_STUDENT_MODULE:-/outputs/llm_distill/student.py}"
  export LORA_MODULE="${LORA_MODULE:-/outputs/llm_distill/lora.py}"
  EVAL=(python /outputs/llm_distill/run_eval_joint.py)
else
  unset VIT_STUDENT_CKPT
  LE=lerobot-eval; [ -x /opt/venv/bin/lerobot-eval ] && LE=/opt/venv/bin/lerobot-eval
  EVAL=("$LE")
fi

[ -f "$RESULTS" ] || echo "mode,suite,pc_success,n_episodes,eval_s,status,timestamp" > "$RESULTS"

run_suite () {
  local suite="$1" gpu="$2"
  local name="eval_${MODE}_${suite}"
  local odir=$RESDIR/$name log=$RESDIR/${name}.log
  if [ -f "$odir/eval_info.json" ]; then echo "[skip] $name (done)"; return; fi
  mkdir -p "$odir"
  echo "===== $name gpu=$gpu mode=$MODE tasks=[$TASK_IDS] n_ep=$N_EP $(date '+%F %T') ====="
  HIP_VISIBLE_DEVICES=$gpu "${EVAL[@]}" \
    --policy.path="$CKPT" \
    --policy.inference_action_mode=continuous \
    --policy.chunk_size=10 --policy.n_action_steps=10 \
    --policy.model_dtype=bfloat16 --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=false \
    --policy.device=cuda \
    --policy.per_episode_seed=true --policy.eval_seed="$SEED" \
    --env.type=libero --env.task="$suite" --env.task_ids="[$TASK_IDS]" \
    --env.camera_name_mapping="$CAM" \
    --eval.batch_size=1 --eval.n_episodes="$N_EP" --seed="$SEED" \
    --output_dir="$odir" > "$log" 2>&1
  local rc=$?
  python3 - "$log" "$MODE" "$suite" "$RESULTS" "$rc" <<'PY' || true
import re, sys, time
log, mode, suite, res, rc = sys.argv[1:6]
t = open(log, errors="ignore").read()
def last(p):
    m = re.findall(p, t); return m[-1] if m else ""
pc = last(r"'pc_success':\s*([\d.]+)"); ne = last(r"'n_episodes':\s*(\d+)"); es = last(r"'eval_s':\s*([\d.]+)")
st = "done" if (pc != "" and rc == "0") else "failed"
open(res, "a").write(f"{mode},{suite},{pc},{ne},{es},{st},{time.strftime('%F %T')}\n")
print(f"[result] {mode} {suite} pc_success={pc} n={ne} status={st}")
PY
}

echo "########## VD EVAL START mode=$MODE suites=[$SUITES] tasks=[$TASK_IDS] n_ep=$N_EP seed=$SEED ngpu=$NGPU stagger=${STAGGER}s resdir=$RESDIR ckpt=$CKPT $(date '+%F %T') ##########"
i=0; pids=()
for s in $SUITES; do
  g=$((i % NGPU))
  ( sleep $((i * STAGGER)); run_suite "$s" "$g" ) &
  pids+=($!)
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p"; done

REMAIN=0
for s in $SUITES; do [ -f "$RESDIR/eval_${MODE}_${s}/eval_info.json" ] || REMAIN=$((REMAIN + 1)); done
echo "########## VD EVAL DONE mode=$MODE remaining_suites=$REMAIN $(date '+%F %T') ##########"
[ "$REMAIN" -eq 0 ] && touch "$RESDIR/_VD_EVAL_DONE"
