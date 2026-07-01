#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Vision-token reduction ablation for MolmoAct2-LIBERO. Sweeps VIS_KEEP_FRAC
# across LIBERO suites, recording closed-loop success + a fast per-fraction
# latency decomposition. Runs INSIDE the molmoact2 container (see
# run_ablation_detached.sh). NOT -e: we keep going past any single failed config.
#
# Resumable: configs already marked done in results.csv are skipped.
# Knobs (env): N_EP, SUITES, FRACS, NUM_STEPS, SEED.
set -uo pipefail

OUT=/ryzers/outputs/ablation_visdrop
mkdir -p "$OUT"
RESULTS="$OUT/results.csv"
LAT="$OUT/latency.csv"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
NUM_STEPS="${NUM_STEPS:-4}"
N_EP="${N_EP:-4}"
SEED="${SEED:-1000}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"
FRACS="${FRACS:-0.05 0.10 0.25 0.50 0.75 1.00}"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_BLAS_PREFER_HIPBLASLT=0

LEROBOT_EVAL=lerobot-eval
[ -x /opt/libero-venv/bin/lerobot-eval ] && LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval

[ -f "$RESULTS" ] || echo "suite,keep_frac,pc_success,n_episodes,eval_s,eval_ep_s,status,timestamp" > "$RESULTS"
[ -f "$LAT" ]     || echo "keep_frac,vision_ms,llm_prefill_ms,flow_ms_per_step,total_ms,infer_per_s,timestamp" > "$LAT"

echo "########## ABLATION START $(date '+%F %T') ##########"
echo "suites=[$SUITES] fracs=[$FRACS] n_ep=$N_EP num_steps=$NUM_STEPS"

# ---------------------------------------------------------------------------
# Track 1: fast latency decomposition per fraction (DROID continuous bench).
# ---------------------------------------------------------------------------
for f in $FRACS; do
  if grep -q "^$f," "$LAT" 2>/dev/null; then echo "[lat] skip frac=$f (done)"; continue; fi
  echo "===== latency frac=$f $(date '+%T') ====="
  VIS_KEEP_FRAC="$f" OUT_JSON="$OUT/lat_$f.json" MAX_CROPS_SWEEP=8 NUM_STEPS="$NUM_STEPS" \
    STEPS_DECOMP=2,10 TIMED_N=5 python /scripts/bench_opt_study.py > "$OUT/lat_$f.log" 2>&1 || true
  python - "$OUT/lat_$f.log" "$f" "$LAT" <<'PY' || true
import re, sys, time
log, frac, latcsv = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log, errors="ignore").read()
vis = llm = flow = total = ips = ""
m = re.search(r"vision=([\d.]+)\s*ms.*?llm_prefill\D*?([\d.]+)\s*ms.*?flow\D*?([\d.]+)\s*ms/step", txt, re.S)
if m: vis, llm, flow = m.group(1), m.group(2), m.group(3)
m2 = re.search(r"\[crops=\d+\][^\n]*?total=([\d.]+)\s*ms\s*\(([\d.]+)\s*infer/s\)", txt)
if m2: total, ips = m2.group(1), m2.group(2)
with open(latcsv, "a") as fh:
    fh.write(f"{frac},{vis},{llm},{flow},{total},{ips},{time.strftime('%F %T')}\n")
print(f"[lat] frac={frac} vision={vis} llm={llm} flow={flow} total={total} ips={ips}")
PY
done

# ---------------------------------------------------------------------------
# Track 2: closed-loop accuracy per (suite, fraction).
# ---------------------------------------------------------------------------
for suite in $SUITES; do
  for f in $FRACS; do
    if grep -q ",done," "$RESULTS" 2>/dev/null && grep -q "^$suite,$f," "$RESULTS" 2>/dev/null; then
      echo "[eval] skip $suite frac=$f (already in results.csv)"; continue
    fi
    name="run_${suite}_${f}"
    odir="$OUT/$name"
    log="$OUT/${name}.log"
    mkdir -p "$odir"
    echo "===== eval suite=$suite frac=$f n_ep=$N_EP $(date '+%F %T') ====="
    VIS_KEEP_FRAC="$f" "$LEROBOT_EVAL" \
      --policy.type=molmoact2 \
      --policy.checkpoint_path="$CKPT" \
      --policy.inference_action_mode=continuous \
      --policy.enable_depth_reasoning=False \
      --policy.enable_adaptive_depth=False \
      --policy.enable_cuda_graph=False \
      --policy.num_steps="$NUM_STEPS" \
      --policy.norm_tag=libero \
      --policy.device=cuda \
      --env.type=libero \
      --env.task="$suite" \
      --eval.batch_size=1 \
      --eval.n_episodes="$N_EP" \
      --seed="$SEED" \
      --output_dir="$odir/run" > "$log" 2>&1
    rc=$?
    python - "$log" "$suite" "$f" "$RESULTS" "$rc" <<'PY' || true
import re, sys, time
log, suite, frac, res, rc = sys.argv[1:6]
txt = open(log, errors="ignore").read()
# take the LAST aggregated dict (overall)
pcs = re.findall(r"'pc_success':\s*([\d.]+)", txt)
nep = re.findall(r"'n_episodes':\s*(\d+)", txt)
es  = re.findall(r"'eval_s':\s*([\d.]+)", txt)
eps = re.findall(r"'eval_ep_s':\s*([\d.]+)", txt)
pc  = pcs[-1] if pcs else ""
ne  = nep[-1] if nep else ""
e_s = es[-1]  if es  else ""
e_p = eps[-1] if eps else ""
status = "done" if (pc != "" and rc == "0") else "failed"
with open(res, "a") as fh:
    fh.write(f"{suite},{frac},{pc},{ne},{e_s},{e_p},{status},{time.strftime('%F %T')}\n")
print(f"[eval] {suite} frac={frac} pc_success={pc} n={ne} status={status}")
PY
  done
done

echo "########## ABLATION DONE $(date '+%F %T') ##########"
touch "$OUT/_ABLATION_DONE"
