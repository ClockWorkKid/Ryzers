#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Overnight orchestrator (runs INSIDE the nanowm container). Executes the full demo suite +
# model analysis + applications sequentially, each guarded so one failure never stops the rest.
# Per-step logs land in /outputs/logs, a machine-readable status in /outputs/status.json, and a
# human summary in /outputs/SUMMARY.md. Small filmstrip PNGs are produced for quick inspection.
#
# Scope is env-driven (host sets these):
#   ROLLOUT_DOMAINS  space-separated: dino_wm_point_maze dino_wm_wall ...
#   DO_ANALYSIS/DO_PLANNING/DO_V3D  1/0
#   DO_DEFORMABLE (rope+granular, ~15GB) / DO_CSGO / DO_RT1  1/0 (best-effort)
set -u
OUT_DIR="${OUT_DIR:-/outputs}"
LOGS="$OUT_DIR/logs"; mkdir -p "$LOGS"
STATUS="$OUT_DIR/status.json"
SUMMARY="$OUT_DIR/SUMMARY.md"
: > "$STATUS.tmp"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_step() {  # name, then command
  local name="$1"; shift
  local log_file="$LOGS/$name.log"
  log "START $name"
  local t0=$(date +%s)
  ( "$@" ) >"$log_file" 2>&1
  local rc=$?
  local t1=$(date +%s)
  log "END   $name rc=$rc ($((t1-t0))s)"
  echo "  {\"step\":\"$name\",\"rc\":$rc,\"seconds\":$((t1-t0))}," >> "$STATUS.tmp"
}

# ---- 1. Model analysis (guaranteed; needs only pusht ckpt) ----
if [ "${DO_ANALYSIS:-1}" = "1" ]; then
  run_step model_analysis python /ryzers/demos/model_analysis.py \
    --domain dino_wm_pusht --out "$OUT_DIR/analysis"
fi

# ---- 2. Per-domain rollout demos ----
for d in ${ROLLOUT_DOMAINS:-dino_wm_pusht dino_wm_point_maze dino_wm_wall}; do
  run_step "rollout_$d" env DOMAIN="$d" bash /ryzers/demos/demo_rollout.sh
done

# ---- 2b. deformable (rope+granular) — best effort, heavy dataset ----
if [ "${DO_DEFORMABLE:-0}" = "1" ]; then
  for d in dino_wm_rope dino_wm_granular; do
    run_step "rollout_$d" env DOMAIN="$d" bash /ryzers/demos/demo_rollout.sh
  done
fi

# ---- 2c. CSGO long rollout (flagship) — best effort, subset val data ----
if [ "${DO_CSGO:-0}" = "1" ]; then
  run_step "rollout_csgo" env DOMAIN=csgo ROLLOUT_LENGTH="${CSGO_ROLLOUT_LENGTH:-50}" \
    bash /ryzers/demos/demo_rollout.sh
fi

# ---- 2d. RT-1 — best effort ----
if [ "${DO_RT1:-0}" = "1" ]; then
  run_step "rollout_rt1" env DOMAIN=rt1 bash /ryzers/demos/demo_rollout.sh
fi

# ---- 3. Planning application ----
if [ "${DO_PLANNING:-1}" = "1" ]; then
  run_step "planning_pusht" env ENV_NAME=pusht bash /ryzers/demos/demo_planning.sh
fi

# ---- 4. video -> 3D application (best effort) ----
if [ "${DO_V3D:-1}" = "1" ]; then
  run_step "video_to_3d" bash /ryzers/demos/demo_video_to_3d.sh
fi

# ---- 5. Filmstrips for quick inspection ----
for cmp in "$OUT_DIR"/rollout_*/sample_0000_compare.mp4; do
  [ -f "$cmp" ] || continue
  dname=$(basename "$(dirname "$cmp")")
  python /ryzers/demos/make_filmstrip.py --video "$cmp" \
    --out "$OUT_DIR/$dname/filmstrip.png" >>"$LOGS/filmstrip.log" 2>&1 || true
done

# ---- 6. Assemble status.json + SUMMARY.md ----
{ echo "["; sed '$ s/,$//' "$STATUS.tmp"; echo "]"; } > "$STATUS"
rm -f "$STATUS.tmp"

python - "$OUT_DIR" > "$SUMMARY" <<'PY'
import sys, os, json, glob
out = sys.argv[1]
print("# NanoWM overnight demo run — summary\n")
st = json.load(open(os.path.join(out, "status.json"))) if os.path.exists(os.path.join(out,"status.json")) else []
print("## Steps\n")
print("| step | result | seconds |")
print("|:--|:--|--:|")
for s in st:
    print(f"| {s['step']} | {'OK' if s['rc']==0 else 'FAIL('+str(s['rc'])+')'} | {s['seconds']} |")
print("\n## Rollout timing\n")
print("| domain | history | rollout_len | steps | total_s | s/sample |")
print("|:--|--:|--:|--:|--:|--:|")
for tj in sorted(glob.glob(os.path.join(out, "rollout_*/timing.json"))):
    d = json.load(open(tj))
    print(f"| {d['domain']} | {d['history_length']} | {d['rollout_length']} | "
          f"{d['num_sampling_steps']} | {d['total_seconds']} | {d['seconds_per_sample']} |")
ap = os.path.join(out, "analysis", "analysis.json")
if os.path.exists(ap):
    a = json.load(open(ap))
    print("\n## Model breakdown (NanoWM-B/2, pusht)\n")
    print(f"- Params: {a['total_params']/1e6:.1f} M · arch {a['arch']} · depth {a['depth']} · d={a['hidden_size']}")
    print(f"- DiT: {a['dit_gflops_per_forward']:.1f} GFLOP/forward · {a.get('dit_forward_ms',0):.1f} ms/forward")
    print(f"- Tokens: {a['tokens_per_frame_spatial']}/frame spatial × {a['temporal_len']} temporal "
          f"= {a['total_spatiotemporal_tokens']} spatiotemporal")
    v = a.get('vae', {})
    if v:
        print(f"- SD-VAE: {v.get('params',0)/1e6:.1f} M · enc {v.get('encode_ms',0):.1f} ms · dec {v.get('decode_ms',0):.1f} ms")
    print(f"- See system_diagram.png for the full annotated pipeline.")
PY

log "ALL DONE. See $SUMMARY"
cat "$SUMMARY"
