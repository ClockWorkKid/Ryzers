#!/usr/bin/env bash
# Pick a GPU node for a 4-GPU eval segment, preferring the EMPTIEST node (most free GPUs). Free-GPU
# count is a good proxy for CPU-render contention: a fully idle node runs the osmesa renders much
# faster than one shared with other users' CPU-heavy jobs. Excludes the caller-supplied node
# (arg1, just vacated), nodes already running one of my eval jobs (no co-location), and
# DOWN/DRAIN nodes. Prints the best node, or nothing if none has >=4 free (caller then queues).
# Env: EVAL_PARTITION (SLURM partition to search), JOB_NAME (my eval job name, default lp_eval).
set -u
self_exclude="${1:-}"
EVAL_PARTITION="${EVAL_PARTITION:-}"
JOB_NAME="${JOB_NAME:-lp_eval}"
mine=$(squeue -u "$USER" -h -t RUNNING -o '%j %N' | awk -v j="$JOB_NAME" '$1==j{print $2}' | sort -u)
if [ -n "$EVAL_PARTITION" ]; then
  nodes=$(sinfo -h -p "$EVAL_PARTITION" -N -o '%n' | sort -u)
else
  nodes=$(sinfo -h -N -o '%n' | sort -u)
fi
best=""; bestfree=3   # require >=4 free
for n in $nodes; do
  [ "$n" = "$self_exclude" ] && continue
  echo "$mine" | grep -qx "$n" && continue
  line=$(scontrol show node "$n" 2>/dev/null)
  echo "$line" | grep -qE 'State=.*(DOWN|DRAIN|NOT_RESPONDING)' && continue
  cfg=$(echo "$line" | grep -oE 'CfgTRES=[^ ]*' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+$'); cfg=${cfg:-8}
  alloc=$(echo "$line" | grep -oE 'AllocTRES=[^ ]*' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+$'); alloc=${alloc:-0}
  free=$((cfg - alloc))
  if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$n; fi
done
[ -n "$best" ] && echo "$best"
exit 0
