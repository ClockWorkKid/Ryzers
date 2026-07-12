#!/usr/bin/env bash
# Closed-loop LIBERO accuracy sweep for the ROI-pruned checkpoints (seam gate,
# gate+FastV two-stage, or causal gate_predict). Runs INSIDE the eval container.
# Sweeps an explicit RUNS list of "label:rundir:ckptstep" triples.
#
# Each unit = (run x suite x TASK), N_EP episodes; resumable at task level (a unit
# with eval_info.json is skipped) and chainable across segments.
#
# REGIME (which path fires at inference is chosen by the mounted overlay + flag):
#   * gate-only  : ROI_FASTV_INFER unset -> pre-ViT scatter-back only (LLM sees full
#                  pooled grid). Safe/known; use for a first accuracy read.
#   * FastV-wired: ROI_FASTV_INFER=1 -> the trained gate additionally drives the
#                  in-LLM cut at L0 (matches two-stage / gate_predict training).
#                  Set RESDIR distinctly so regimes never mix.
set -uo pipefail
OUT=/outputs
RESDIR="${RESDIR:-$OUT/roi_eval_gateonly}"
mkdir -p "$RESDIR"
RESULTS=$RESDIR/results.csv

N_EP="${N_EP:-10}"
SEED="${SEED:-1000}"
NGPU="${NGPU:-8}"
WPG="${WPG:-2}"
NW=$((NGPU*WPG))
# "label:rundir:ckptstep" space-separated. rundir is relative to /outputs; ckptstep
# is the zero-padded checkpoint step (e.g. 012000). Override for your checkpoints.
RUNS="${RUNS:-fv025:roi_fastv_keep050_fv025:012000 fv050:roi_fastv_keep050_fv050:012000}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"
L90_TASKS="${L90_TASKS:-0 9 18 27 36 45 54 63 72 81}"
STD_TASKS="${STD_TASKS:-0 1 2 3 4 5 6 7 8 9}"
CAM='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

tasks_for () { if [ "$1" = "libero_90" ]; then echo "$L90_TASKS"; else echo "$STD_TASKS"; fi; }

# osmesa (CPU/llvmpipe) is mandatory on compute-only datacenter GPUs (EGL fails);
# override to egl on a graphics-capable GPU.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_BLAS_PREFER_HIPBLASLT=0
# FastV-wired regime: exported into the eval subprocess so the overlay's inference
# path applies the in-LLM cut. Unset -> gate-only.
export ROI_FASTV_INFER="${ROI_FASTV_INFER:-}"

LEROBOT_EVAL=lerobot-eval
[ -x /opt/venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/venv/bin/lerobot-eval

if [ ! -f "$HOME/.libero/config.yaml" ]; then
  echo "[libero] bootstrapping $HOME/.libero/config.yaml"
  printf 'N\n' | python3 -c "import libero.libero" >/dev/null 2>&1 || true
fi

[ -f "$RESULTS" ] || echo "label,rundir,ckpt,suite,task,pc_success,n_episodes,eval_s,eval_ep_s,status,timestamp" > "$RESULTS"

run_one () {
  local label="$1" rundir="$2" step="$3" suite="$4" task="$5" gpu="$6"
  local ckpt=$OUT/$rundir/checkpoints/$step/pretrained_model
  local name="eval_${label}_${suite}_t$(printf '%02d' "$task")"
  local odir=$RESDIR/$name log=$RESDIR/${name}.log
  if [ -f "$odir/eval_info.json" ]; then echo "[skip] $name (done)"; return; fi
  if [ ! -f "$ckpt/config.json" ]; then echo "[MISS] $name: no ckpt at $ckpt"; return; fi
  mkdir -p "$odir"
  echo "===== $name gpu=$gpu n_ep=$N_EP fastv_infer=${ROI_FASTV_INFER:-0} $(date '+%F %T') ====="
  local RD=""
  [ -n "${MASK_DUMP:-}" ] && RD="ROI_MASK_DUMP=$odir/maskdump"
  env $RD HIP_VISIBLE_DEVICES=$gpu ROI_FASTV_INFER="${ROI_FASTV_INFER:-}" ROI_PRUNE_DEBUG="${ROI_PRUNE_DEBUG:-}" "$LEROBOT_EVAL" \
    --policy.path="$ckpt" \
    --policy.inference_action_mode=continuous \
    --policy.model_dtype=bfloat16 --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=false \
    --policy.device=cuda \
    --policy.per_episode_seed=true --policy.eval_seed="$SEED" \
    --env.type=libero --env.task="$suite" --env.task_ids="[$task]" \
    --env.camera_name_mapping="$CAM" \
    --eval.batch_size=1 --eval.n_episodes="$N_EP" --seed="$SEED" \
    --output_dir="$odir" > "$log" 2>&1
  local rc=$?
  python3 - "$log" "$label" "$rundir" "$step" "$suite" "$task" "$RESULTS" "$rc" <<'PY' || true
import re, sys, time
log, label, rundir, step, suite, task, res, rc = sys.argv[1:9]
t = open(log, errors="ignore").read()
def last(p):
    m = re.findall(p, t); return m[-1] if m else ""
pc = last(r"'pc_success':\s*([\d.]+)"); ne = last(r"'n_episodes':\s*(\d+)")
es = last(r"'eval_s':\s*([\d.]+)"); ep = last(r"'eval_ep_s':\s*([\d.]+)")
st = "done" if (pc != "" and rc == "0") else "failed"
open(res, "a").write(f"{label},{rundir},{step},{suite},{task},{pc},{ne},{es},{ep},{st},{time.strftime('%F %T')}\n")
print(f"[result] {label} {suite} t{task} pc_success={pc} n={ne} status={st}")
PY
}

JOBS=()
for r in $RUNS; do
  label="${r%%:*}"; rest="${r#*:}"; rundir="${rest%%:*}"; step="${rest#*:}"
  for s in $SUITES; do
    if [ -n "${TASK_IDS_ALL:-}" ]; then TL="$TASK_IDS_ALL"; else TL="$(tasks_for "$s")"; fi
    for t in $TL; do JOBS+=("$label $rundir $step $s $t"); done
  done
done
echo "########## ROI EVAL START $(date '+%F %T') units=${#JOBS[@]} ngpu=$NGPU wpg=$WPG workers=$NW n_ep=$N_EP resdir=$RESDIR fastv_infer=${ROI_FASTV_INFER:-0} ##########"
echo "runs=[$RUNS]"; echo "suites=[$SUITES]"
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for w in $(seq 0 $((NW-1))); do
    [ $i -lt ${#JOBS[@]} ] || break
    g=$((w % NGPU))
    set -- ${JOBS[$i]}
    run_one "$1" "$2" "$3" "$4" "$5" "$g" &
    pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
REMAIN=0
for j in "${JOBS[@]}"; do
  set -- $j; nm="eval_${1}_${4}_t$(printf '%02d' "$5")"
  [ -f "$RESDIR/$nm/eval_info.json" ] || REMAIN=$((REMAIN+1))
done
echo "########## ROI EVAL SEGMENT DONE $(date '+%F %T') remaining_units=$REMAIN ##########"
[ "$REMAIN" -eq 0 ] && touch "$RESDIR/_ROI_EVAL_DONE"
