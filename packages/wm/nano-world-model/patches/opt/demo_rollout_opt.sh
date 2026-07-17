#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NanoWM per-domain video-generation demo (standalone). Wraps upstream src/sample/rollout.py
# with the authors' released checkpoint + matching config for a given DOMAIN, autoregressively
# rolling out future frames and decoding to gen / gt / comparison mp4 + a timing.json.
#
# DOMAIN in: dino_wm_point_maze | dino_wm_pusht | dino_wm_wall | dino_wm_rope |
#            dino_wm_granular | rt1 | csgo
#
# Sampling config follows the upstream default axes (sequential scheduling, the checkpoint's own
# pred/schedule, history_stabilization_level=0.02). num_sampling_steps defaults to 50 (the
# upstream "quality/speed knee" from docs/applications/long_rollout.md; raise to 250 to match the
# metric-repro eval). history_length is auto-derived from the checkpoint config's n_context_frames
# (rollout.py requires it < model.num_frames). Everything is overridable by env var.
set -euo pipefail

DOMAIN="${DOMAIN:-dino_wm_pusht}"
REPO=/repos/nanowm
RESULTS_DIR="${RESULTS_DIR:-/models/results}"
OUT_DIR="${OUT_DIR:-/outputs}"
CKPT_DIR="$RESULTS_DIR/$DOMAIN"
CONFIG="${CONFIG:-$CKPT_DIR/config.yaml}"
CKPT="${CKPT:-$CKPT_DIR/model.pt}"
SAVE_PATH="${SAVE_PATH:-$OUT_DIR/rollout_$DOMAIN}"

# Map DOMAIN -> dataset key understood by scripts/download_datasets.sh.
declare -A DSKEY=(
  [dino_wm_point_maze]=point_maze [dino_wm_pusht]=pusht [dino_wm_wall]=wall
  [dino_wm_rope]=rope [dino_wm_granular]=granular [rt1]=rt1 [csgo]=csgo )

# 1) Ensure checkpoint (config.yaml + model.pt) is present.
if [ ! -f "$CKPT" ]; then
  echo "[demo] fetching checkpoint for $DOMAIN"
  DOMAIN="$DOMAIN" RESULTS_DIR="$RESULTS_DIR" bash /ryzers/scripts/download_checkpoints.sh
fi

# 2) Ensure dataset (context frames) is present.
if [ -n "${FETCH_DATA:-1}" ]; then
  echo "[demo] ensuring dataset for $DOMAIN"
  DOMAIN="${DSKEY[$DOMAIN]}" bash /ryzers/scripts/download_datasets.sh || {
    echo "[demo] dataset fetch failed for $DOMAIN"; exit 3; }
fi

# 2.5) Domain-specific config reconciliation so a released checkpoint runs on a *bounded* local
# subset (rollout.py has no CLI override for dataset.loader). Idempotent; only touches CONFIG.
case "$DOMAIN" in
  rt1)
    # The shipped rt1 config points at the original (gated) lerobot repo and loads all 87k episodes.
    # Repoint to the public IPEC mirror and cap episodes so lerobot fetches a bounded subset on-demand.
    RT1_REPO="${RT1_REPO:-IPEC-COMMUNITY/fractal20220817_data_lerobot}"
    RT1_N_ROLLOUT="${RT1_N_ROLLOUT:-64}"
    python - "$CONFIG" "$RT1_REPO" "$RT1_N_ROLLOUT" <<'PY'
import sys, yaml
cfg, repo, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(cfg)); ld = c["dataset"]["loader"]
ld["data_path"] = repo; ld["n_rollout"] = n
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print(f"[demo] rt1 config: data_path={repo} n_rollout={n}")
PY
    ;;
  csgo)
    # The exhaustive csgo val loader needs fixed_start_indices to match #trajectories. On a bounded
    # shard subset, build a reduced val split (only present episodes) + aligned start indices.
    python - "$CONFIG" "${CSGO_DATA_DIR:-/models/datasets/csgo}" <<'PY'
import sys, os, glob, yaml, numpy as np
cfg, data_dir = sys.argv[1], sys.argv[2]
repo = "/repos/nanowm"
full_txt = os.path.join(repo, "src/wm_datasets/data_source/game/csgo_splits/test_split.txt")
full_npy = os.path.join(repo, "src/wm_datasets/data_source/game/csgo_splits/csgo_validation_start_indices.npy")
full = [l.strip() for l in open(full_txt) if l.strip()]
starts = np.load(full_npy)
present = set()
for f in glob.glob(os.path.join(data_dir, "*", "hdf5_dm_july2021_*.hdf5")):
    present.add(int(os.path.basename(f).split("_")[-1].replace(".hdf5", "")))
rf, rs = [], []
for fn, st in zip(full, starts):
    if int(fn.split("_")[-1].replace(".hdf5", "")) in present:
        rf.append(fn); rs.append(st)
if not rf:
    print("[demo] csgo: no present episodes found; leaving config untouched"); sys.exit(0)
out = os.path.join(data_dir, "csgo_splits_reduced"); os.makedirs(out, exist_ok=True)
rtxt = os.path.join(out, "test_split.txt"); rnpy = os.path.join(out, "val_start_indices.npy")
open(rtxt, "w").write("\n".join(rf) + "\n"); np.save(rnpy, np.array(rs))
c = yaml.safe_load(open(cfg)); ld = c["dataset"]["loader"]
ld["val_file_list"] = rtxt; ld["val_start_indices"] = rnpy
# Resolve the '${csgo_data_dir}' interpolation to the concrete mounted dir (else 0 trajectories).
ld["data_path"] = data_dir
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print(f"[demo] csgo config: reduced val to {len(rf)}/{len(full)} present episodes; data_path={data_dir}")
PY
    ;;
esac

# 3) Derive history_length from the checkpoint config (n_context_frames), clamped < num_frames.
HISTORY_LENGTH="${HISTORY_LENGTH:-$(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
m = c.get("model", {})
nc = int(m.get("n_context_frames", 1) or 1)
nf = int(m.get("num_frames", 4) or 4)
print(max(1, min(nc, nf - 1)))
PY
)}"

# Domain-tuned rollout defaults (overridable). rollout consumes ~(history+rollout)*frame_interval
# raw frames, so short-trajectory envs need a smaller rollout_length. wall trajectories are only
# ~50 frames @ interval 5 -> cap rollout_length so slices have headroom.
case "$DOMAIN" in
  csgo)              NUM_SAMPLES="${NUM_SAMPLES:-2}"; BATCH_SIZE="${BATCH_SIZE:-2}";
                     ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-50}" ;;
  dino_wm_wall)      NUM_SAMPLES="${NUM_SAMPLES:-4}"; BATCH_SIZE="${BATCH_SIZE:-4}";
                     ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-8}" ;;
  *)                 NUM_SAMPLES="${NUM_SAMPLES:-4}"; BATCH_SIZE="${BATCH_SIZE:-4}";
                     ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-16}" ;;
esac
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"
SCHEDULING_MODE="${SCHEDULING_MODE:-sequential}"
FPS="${FPS:-8}"

mkdir -p "$SAVE_PATH"
cd "$REPO"
echo "[demo] $DOMAIN | history=$HISTORY_LENGTH rollout=$ROLLOUT_LENGTH steps=$NUM_SAMPLING_STEPS samples=$NUM_SAMPLES"

# FastWAM optimization knobs forwarded to rollout.py (defaults preserve upstream fp32 full-window).
DIT_AUTOCAST="${DIT_AUTOCAST:-off}"
SEED="${SEED:-0}"
OPT_ARGS=(--dit_autocast "$DIT_AUTOCAST" --seed "$SEED")
if [ "${FULL_WINDOW:-0}" = "1" ]; then OPT_ARGS+=(--full_window); fi
if [ "${A5:-1}" = "0" ]; then OPT_ARGS+=(--no_a5); fi
if [ -n "${VAE_DECODE_PRECISION:-}" ]; then OPT_ARGS+=(--vae_decode_precision "$VAE_DECODE_PRECISION"); fi
if [ "${GT_VIDEO:-1}" = "0" ]; then OPT_ARGS+=(--no_gt_video); fi
echo "[demo] opt: dit_autocast=$DIT_AUTOCAST seed=$SEED full_window=${FULL_WINDOW:-0} a5=${A5:-1} vae_decode=${VAE_DECODE_PRECISION:-config} gt_video=${GT_VIDEO:-1}"

START=$(date +%s.%N)
python src/sample/rollout.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --save_path "$SAVE_PATH" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --rollout_length "$ROLLOUT_LENGTH" \
  --history_length "$HISTORY_LENGTH" \
  --num_sampling_steps "$NUM_SAMPLING_STEPS" \
  --scheduling_mode "$SCHEDULING_MODE" \
  --history_stabilization_level 0.02 \
  --fps "$FPS" \
  "${OPT_ARGS[@]}"
END=$(date +%s.%N)

ELAPSED=$(python -c "print(f'{${END}-${START}:.1f}')")
PERSAMPLE=$(python -c "print(f'{(${END}-${START})/max(1,${NUM_SAMPLES}):.1f}')")
cat > "$SAVE_PATH/timing.json" <<JSON
{
  "domain": "$DOMAIN",
  "history_length": $HISTORY_LENGTH,
  "rollout_length": $ROLLOUT_LENGTH,
  "num_sampling_steps": $NUM_SAMPLING_STEPS,
  "scheduling_mode": "$SCHEDULING_MODE",
  "num_samples": $NUM_SAMPLES,
  "batch_size": $BATCH_SIZE,
  "total_seconds": $ELAPSED,
  "seconds_per_sample": $PERSAMPLE,
  "device": "gfx1151"
}
JSON
echo "[demo] done in ${ELAPSED}s -> $SAVE_PATH"
ls -la "$SAVE_PATH"
