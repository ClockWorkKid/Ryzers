#!/usr/bin/env bash
# One-shot DAgger pipeline status: Slurm queue, per-arm train step / collect / relabel
# counts, and gate summaries when ready. Safe to run repeatedly (e.g. from a poll loop).
# Set WORK to your workspace root (holds outputs/llm_distill/dagger).
set -u
WORK="${WORK:-$HOME/molmoact2}"
O="$WORK/outputs/llm_distill/dagger"
ARMS="${ARMS:-expd}"
echo "TS=$(date -u +%H:%M:%S) UTC"
echo "--- queue ---"
squeue --me -o '%.10i %.20j %.2t %.8M %R' 2>/dev/null
for a in $ARMS; do
  L="$O/$a/train_aelora/train_log.jsonl"; [ -f "$L" ] || L="$O/$a/train/train_log.jsonl"
  if [ -f "$L" ]; then
    st=$(grep -aoE '"step": [0-9]+' "$L" | tail -1)
    ho=$(grep -aoE '"heldout_flow": [0-9.]+' "$L" | tail -1)
    echo "--- $a train: ${st:-<none>} | ${ho:-} ---"
  fi
  C="$O/$a/collect/manifest.json"; [ -f "$C" ] && echo "  $a collect states=$(grep -aoE '"total_states": [0-9]+' "$C" | tail -1)"
  R="$O/$a/relabel/manifest.json"; [ -f "$R" ] && echo "  $a relabel states=$(grep -aoE '"total_states": [0-9]+' "$R" | tail -1)"
  for g in gate2_aelora gate2 gate; do
    S="$O/$a/$g/summary.json"
    if [ -f "$S" ]; then echo "--- $a GATE ($g) ---"; cat "$S"; echo; break; fi
  done
done
