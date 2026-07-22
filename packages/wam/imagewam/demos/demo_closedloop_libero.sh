#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop LIBERO rollouts in MuJoCo (headless) driven by FLUX.2 ImageWAM. Runs the
# *upstream* single-task evaluator (experiments/libero/eval_libero_single.py) per task inside
# the combined image, then aggregates a success_summary.json. Requires the sim chain:
#   ryzers build libero imagewam            # -> image with /opt/LIBERO
#   ryzers run /ryzers/demos/demo_closedloop_libero.sh
#   SUITE=libero_object NUM_TASKS=2 NUM_TRIALS=5 ryzers run /ryzers/demos/demo_closedloop_libero.sh
#
# The "dream vs actual sim frame" two-column visualization is produced separately by
# demo_closedloop_dream.sh (klein's dream is infer_video_flux2, not the video infer_joint
# path that EVALUATION.visualize_future_video assumes).
set -uo pipefail
if [ ! -d /opt/LIBERO ]; then
  echo "ERROR: simulation/libero base not found (no /opt/LIBERO)." >&2
  echo "       Build the chain:  ryzers build libero imagewam" >&2
  exit 1
fi

REPO="${IMAGEWAM_REPO:-/repos/imagewam}"
FLUX2_SRC="${FLUX2_SRC:-/repos/flux2}"
SUITE="${SUITE:-libero_object}"
NUM_TASKS="${NUM_TASKS:-3}"
NUM_TRIALS="${NUM_TRIALS:-5}"
VARIANT="${FLUX2_VARIANT:-4b}"
TAG="${TAG:-cl_${SUITE}}"
OUT="${OUT_DIR:-/outputs}/closedloop/$TAG"
mkdir -p "$OUT"

CKPT="${CKPT_PATH:-/models/imagewam_release/libero/flux2_klein_${VARIANT}/model.pt}"
STATS="${DATASET_STATS_PATH:-/models/imagewam_release/libero/flux2_klein_${VARIANT}/dataset_stats.json}"
AE="${FLUX2_AE_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/ae.safetensors}"
DIT="${FLUX2_MODEL_PATH:-/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
QWEN3="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
for f in "$CKPT" "$STATS" "$AE" "$DIT"; do
  [ -f "$f" ] || { echo "missing weight: $f  (run scripts/download_checkpoints.sh + convert_klein_vae.py)" >&2; exit 1; }
done

# LIBERO MuJoCo needs GL; the libero base is set up for egl (GPU). Keep upstream's headless env.
export PYTHONPATH="${REPO}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${REPO}/experiments/libero:/opt/LIBERO:${PYTHONPATH:-}"

cd /tmp
for t in $(seq 0 $((NUM_TASKS-1))); do
  echo "########## $SUITE task $t (trials=$NUM_TRIALS) ##########"
  python "$REPO/experiments/libero/eval_libero_single.py" \
    --config-name sim_libero_omnigen2 \
    task="libero_flux2_klein_${VARIANT}_base_imagewam" \
    ckpt="$CKPT" gpu_id=0 mixed_precision=bf16 \
    EVALUATION.task_suite_name="$SUITE" EVALUATION.task_id="$t" EVALUATION.num_trials="$NUM_TRIALS" \
    EVALUATION.action_horizon="${ACTION_HORIZON:-16}" EVALUATION.replan_steps="${REPLAN_STEPS:-12}" \
    EVALUATION.visualize_future_video=false \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.output_dir="$OUT" \
    model.flux2_src_path="$FLUX2_SRC" model.flux2_model_path="$DIT" \
    model.ae_model_path="$AE" model.variant="klein-base-${VARIANT}" \
    model.qwen3_model_spec="$QWEN3" model.load_text_encoder=true \
    model.pack_proprio_after_text=true model.proprio_dim=8 \
    2>&1 | grep -aE "Task [0-9]+|success|Success|completed|Saved|Traceback|Error|error executing|env_num"
done

# aggregate (eval inserts a <ckpt_tag>/ dir under output_dir, so search recursively)
python3 - "${OUT_DIR:-/outputs}/closedloop" "$SUITE" "$OUT" <<'PY'
import sys, json, glob, os
root, suite, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(root, "**", f"*{suite}*", "*task*results.json"), recursive=True)
               + glob.glob(os.path.join(root, "**", "gpu*_task*_results.json"), recursive=True))
seen=set(); tot_s=tot_n=0; rows=[]
for f in files:
    if f in seen or suite not in f: continue
    seen.add(f)
    try: r=json.load(open(f))
    except Exception: continue
    s=int(r.get("successes",0)); n=int(r.get("total_episodes", r.get("num_trials",0)))
    if n==0: continue
    tot_s+=s; tot_n+=n; rows.append((r.get("task_id"), s, n, r.get("task_description") or r.get("task")))
for tid,s,n,desc in sorted(rows, key=lambda x:(x[0] if x[0] is not None else 0)):
    print(f"  task {tid}: {s}/{n} ({100.0*s/n:.1f}%)  {str(desc)[:50]}")
overall=(100.0*tot_s/tot_n) if tot_n else 0.0
print(f"OVERALL: {tot_s}/{tot_n} ({overall:.1f}%)")
os.makedirs(out, exist_ok=True)
summ = os.path.join(out, "success_summary.json")
json.dump({"suite": suite, "overall_successes": tot_s, "overall_episodes": tot_n,
           "overall_success_rate_pct": round(overall,2),
           "per_task": [{"task_id":tid,"successes":s,"episodes":n,
                         "success_rate_pct": round(100.0*s/n,2), "task":desc} for tid,s,n,desc in rows]},
          open(summ,"w"), indent=2)
print("wrote", summ)
PY
