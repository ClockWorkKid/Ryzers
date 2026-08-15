#!/usr/bin/env bash
# Run the LIBERO-plus eval sweep for ONE stage on ONE node: stage the policy + merged ckpt + HF
# cache node-local, then run ONE worker per GPU (GPUS concurrent osmesa renders; 4 is the validated
# CPU-render contention sweet spot). Each GPU runs its assigned id-list shards sequentially; each
# shard writes eval_info.json, giving resume-by-shard durability across segment resubmits.
#   args: PLAN_TSV  NODE_NAME  STAGE
# Env: WORK, IMG, POLICYROOT, MERGEDROOT, HFCACHE, OUT, KIND(full|subset), IMG_TAR(optional),
#      GPUS, SCR(dir holding eval_shard_ids.sh).
set -u
PLAN="$1"; NODE="$2"; STAGE="$3"
SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"
WORK="${WORK:-$HOME/molmoact2}"
IMG="${IMG:-molmoact2-libero-plus-eval:rocm950}"
KIND="${KIND:-full}"
POLICYROOT="${POLICYROOT:-$WORK/eval_ckpts}"
MERGEDROOT="${MERGEDROOT:-$WORK/outputs/lp_curric_merged}"
HFCACHE="${HFCACHE:-$WORK/hf_cache}"
OUT="${OUT:-$WORK/libero_plus_eval/${STAGE}/${KIND}}"
LOCAL="${LOCAL:-/tmp/${USER}_lpeval_${STAGE}}"
IMG_TAR="${IMG_TAR:-}"     # optional: path to a saved image tar to auto-load if IMG missing
mkdir -p "$LOCAL" "$OUT"

# Ensure the eval docker image exists on THIS node. When nodes are picked dynamically some may not
# have the image; without this, docker run fails -> 0 shards. Load from a shared tar if provided.
if [ -z "$(docker images -q "$IMG" 2>/dev/null)" ]; then
  if [ -n "$IMG_TAR" ] && [ -f "$IMG_TAR" ]; then
    echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] eval image missing -> docker load from $IMG_TAR"
    docker load -i "$IMG_TAR" || { echo "[$NODE/$STAGE] FATAL: docker load failed"; exit 4; }
  else
    echo "[$NODE/$STAGE] FATAL: image $IMG missing and no IMG_TAR provided"; exit 4
  fi
fi

echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] staging node-local ..."
rsync -rlpt --no-owner --no-group --delete "$POLICYROOT/lp_${STAGE}_policy/" "$LOCAL/policy/"
rsync -rlpt --no-owner --no-group --delete "$MERGEDROOT/${STAGE}/" "$LOCAL/ckpt/"
mkdir -p "$LOCAL/hf_cache"; [ -d "$HFCACHE" ] && rsync -rlpt --no-owner --no-group "$HFCACHE/" "$LOCAL/hf_cache/" || true
[ -s "$LOCAL/policy/model.safetensors" ] || { echo "[$NODE/$STAGE] FATAL: model.safetensors missing after staging"; exit 2; }
echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] staging done."

# Kill ORPHANED eval containers left over from a previous segment killed at walltime (docker run
# --rm kills the client but the container keeps running its shard). Left alive they add a 2nd
# render per GPU -> render contention and can re-run the same shard. Safe: resume-by-shard means any
# killed in-progress shard (no eval_info.json) is simply redone.
orphans=$(docker ps --filter ancestor="$IMG" -q)
if [ -n "$orphans" ]; then
  echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] killing $(echo "$orphans" | grep -c .) orphan eval container(s) before launch"
  echo "$orphans" | xargs -r docker kill >/dev/null 2>&1 || true
  sleep 3
fi

run_gpu(){
  local gpu="$1"
  awk -F'\t' -v g="$gpu" '$3==g' "$PLAN" | sort -t"$(printf '\t')" -k4,4n | \
  while IFS=$'\t' read -r idx pnode g2 proc suite idlist; do
    local first="${idlist%%,*}"; local last="${idlist##*,}"
    local name="shard_${suite}_${first}_${last}"
    if [ -f "$OUT/$name/eval_info.json" ]; then
      echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] gpu$gpu SKIP $name"; continue
    fi
    echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] gpu$gpu START $name (n=$(echo "$idlist" | tr ',' '\n' | wc -l))"
    docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -e HIP_VISIBLE_DEVICES="$gpu" -e HF_HOME=/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -v "$LOCAL/policy":/policy:ro -v "$LOCAL/ckpt":/ckpt:ro -v "$LOCAL/hf_cache":/hf_cache \
      -v "$OUT":/out -v "$SCR":/work \
      "$IMG" bash /work/eval_shard_ids.sh "$suite" "$idlist" "/out/$name" \
      > "$OUT/$name.log" 2>&1
    echo "EXIT $? $name" >> "$OUT/$name.log"
    echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] gpu$gpu DONE $name"
  done
}

gpus=$(awk -F'\t' '{print $3}' "$PLAN" | sort -un)
pids=""
for g in $gpus; do run_gpu "$g" & pids="$pids $!"; done
echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] workers for gpus [$gpus]; pids:$pids"
( while kill -0 $pids 2>/dev/null; do
    dc=$(ls -d "$OUT"/shard_*/ 2>/dev/null | while read -r d; do [ -f "$d/eval_info.json" ] && echo x; done | wc -l)
    echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] shards_with_json=$dc"; sleep 180
  done ) & SAMP=$!
wait $pids; kill "$SAMP" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] [$NODE/$STAGE] NODE_DONE"
