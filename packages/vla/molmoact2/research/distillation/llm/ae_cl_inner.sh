#!/usr/bin/env bash
# Closed-loop LIBERO eval INSIDE the eval SIF for a quantized action expert.
# One process per suite, staggered across GPUs. The AE fake-quant swap is done by
# eval_closedloop_ae.py (reads QUANT_STATE). Mirrors quant_cl_eval_inner.sh.
set -uo pipefail
OUT=/outputs
CKPT="${CKPT:-/outputs/base_teacher/checkpoints/000000/pretrained_model}"
ARM="${ARM:?set ARM}"
RESDIR="${RESDIR:-$OUT/ae_cl/$ARM}"
N_EP="${N_EP:-10}"; SEED="${SEED:-1000}"; NGPU="${NGPU:-4}"; STAGGER="${STAGGER:-120}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4}"
CAM='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
# QUANT_STATE empty => stock FP action expert (apples-to-apples baseline).
export QUANT_STATE="${QUANT_STATE:-}"
mkdir -p "$RESDIR"; RESULTS=$RESDIR/results.csv
[ -f "$RESULTS" ] || echo "arm,suite,pc_success,n_episodes,eval_s,status,timestamp" > "$RESULTS"

run_suite () {
  local suite="$1" gpu="$2"
  local name="eval_${ARM}_${suite}" odir log
  odir=$RESDIR/$name; log=$RESDIR/${name}.log
  if [ -f "$odir/eval_info.json" ]; then echo "[skip] $name (done)"; return; fi
  mkdir -p "$odir"
  echo "===== $name gpu=$gpu tasks=[$TASK_IDS] n_ep=$N_EP $(date '+%F %T') ====="
  HIP_VISIBLE_DEVICES=$gpu python /work/eval_closedloop_ae.py \
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
  python3 - "$log" "$ARM" "$suite" "$RESULTS" "$rc" <<'PY' || true
import re, sys, time
log, arm, suite, res, rc = sys.argv[1:6]
t = open(log, errors="ignore").read()
def last(p):
    m = re.findall(p, t); return m[-1] if m else ""
pc = last(r"'pc_success':\s*([\d.]+)"); ne = last(r"'n_episodes':\s*(\d+)"); es = last(r"'eval_s':\s*([\d.]+)")
st = "done" if (pc != "" and rc == "0") else "failed"
open(res, "a").write(f"{arm},{suite},{pc},{ne},{es},{st},{time.strftime('%F %T')}\n")
print(f"[result] {arm} {suite} pc_success={pc} n={ne} status={st}")
PY
}

echo "########## AE CL EVAL arm=$ARM suites=[$SUITES] n_ep=$N_EP seed=$SEED ngpu=$NGPU resdir=$RESDIR $(date '+%F %T') ##########"
i=0; pids=()
for s in $SUITES; do
  g=$((i % NGPU))
  ( sleep $((i * STAGGER)); run_suite "$s" "$g" ) &
  pids+=($!); i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p"; done
REMAIN=0
for s in $SUITES; do [ -f "$RESDIR/eval_${ARM}_${s}/eval_info.json" ] || REMAIN=$((REMAIN + 1)); done
echo "########## AE CL EVAL DONE arm=$ARM remaining=$REMAIN $(date '+%F %T') ##########"
[ "$REMAIN" -eq 0 ] && touch "$RESDIR/_DONE_$ARM"
