#!/usr/bin/env bash
#SBATCH --job-name=lp_eval
#SBATCH --output=%x_%N_%j.log
#SBATCH --nodes=1
#SBATCH -c 48
#SBATCH --gres=gpu:4
#SBATCH --time=03:55:00
#SBATCH --signal=B:USR1@300
# One 4-GPU LIBERO-plus eval SEGMENT for ONE stage (4 concurrent osmesa renders). Self-chaining
# fixed-length segments + resume-by-shard + a no-progress loop guard. On rank completion the chain
# ENDS and the node is freed (no cross-rank auto-advance, so cascaded stages never hog nodes).
#
# Submit with:  sbatch -p "$EVAL_PARTITION" --export=ALL,STAGE=s5,KIND=full,WORK=...,EVAL_PARTITION=... sweep.sbatch
# Env: STAGE (s1..s5), KIND (full|subset), WORK, IMG, EVAL_PARTITION, IMG_TAR(optional).
set -u
SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)}"
STAGE="${STAGE:?set STAGE=s1..s5}"
KIND="${KIND:-full}"
WORK="${WORK:-$HOME/molmoact2}"
EVAL_PARTITION="${EVAL_PARTITION:-}"
NODE=$SLURMD_NODENAME
PLANDIR="${PLANDIR:-$WORK/plans}"
PLAN="$PLANDIR/plan_${KIND}_${STAGE}.tsv"
OUT="${OUT:-$WORK/libero_plus_eval/${STAGE}/${KIND}}"
export WORK KIND OUT SCR IMG="${IMG:-molmoact2-libero-plus-eval:rocm950}" IMG_TAR="${IMG_TAR:-}"
export POLICYROOT="${POLICYROOT:-$WORK/eval_ckpts}" MERGEDROOT="${MERGEDROOT:-$WORK/outputs/lp_curric_merged}" HFCACHE="${HFCACHE:-$WORK/hf_cache}"

incomplete_count(){
  awk -F'\t' '{f=$6; sub(/,.*/,"",f); l=$6; sub(/.*,/,"",l); printf "shard_%s_%s_%s\n",$5,f,l}' "$PLAN" | \
  while read -r name; do [ -f "$OUT/$name/eval_info.json" ] || echo x; done | wc -l
}
resubmit(){
  local left; left=$(incomplete_count)
  if [ "$left" -gt 0 ]; then
    # Self-healing placement: prefer a DIFFERENT node with >=4 free GPUs (pick_node excludes this
    # being-vacated node AND any node already running one of my eval jobs, so two stages never
    # co-locate -> avoids render contention). If none free, fall back to reclaiming this node.
    local pa=""; [ -n "$EVAL_PARTITION" ] && pa="-p $EVAL_PARTITION"
    local nn; nn=$(EVAL_PARTITION="$EVAL_PARTITION" bash "$SCR/pick_node.sh" "$NODE"); local wnode="${nn:-$NODE}"
    echo "[chain] $STAGE $left incomplete -> resubmit on [$wnode] $(sbatch $pa -w "$wnode" --export=ALL,STAGE=$STAGE,KIND=$KIND,WORK=$WORK,EVAL_PARTITION=$EVAL_PARTITION,IMG_TAR=$IMG_TAR "$SCR/sweep.sbatch" | grep -oE '[0-9]+')"
  else
    echo "[chain] $STAGE $KIND complete -> chain ends; node freed"
  fi
}
trap 'echo "[chain] USR1 near walltime $STAGE"' USR1
[ -f "$PLAN" ] || { echo "MISSING plan $PLAN"; exit 3; }
INC0=$(incomplete_count); T0=$(date +%s)
echo "[sbatch] host=$NODE job=${SLURM_JOB_ID:-none} stage=$STAGE kind=$KIND start=$(date -Is) incomplete_at_start=$INC0"
bash "$SCR/sweep_node.sh" "$PLAN" "$NODE" "$STAGE" &
wait $! || true
INC1=$(incomplete_count); T1=$(date +%s); DT=$((T1-T0))
if [ "$INC1" -ge "$INC0" ] && [ "$DT" -lt 600 ]; then
  echo "[chain] NO PROGRESS ($INC0->$INC1) in ${DT}s -> ABORT chain (setup failure)"; exit 1
fi
resubmit
echo "[sbatch] stage=$STAGE kind=$KIND segment done=$(date -Is)"
