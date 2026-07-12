#!/usr/bin/env bash
# Submit a chain of resume-aware 2h finetune segments on gpu-node.
# Each segment runs toward STEPS (resuming from checkpoints/last); once the
# STEPS checkpoint exists the seg script no-ops. afterany chaining tolerates the
# 2h wall (rc=124 = graceful cap).
set -u
NODE="${NODE:-gpu-node}"
N="${N:-4}"
RUN="${RUN:-student_ft_lora}"
STEPS="${STEPS:-12000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EXP="ALL,RUN=$RUN,STEPS=$STEPS,SAVE_FREQ=$SAVE_FREQ"
dep=""
for i in $(seq 1 "$N"); do
  if [ -z "$dep" ]; then
    j=$(sbatch --parsable -w "$NODE" -t 02:00:00 --export="$EXP" "$HOME/student_ft.sbatch")
  else
    j=$(sbatch --parsable -w "$NODE" -t 02:00:00 --dependency=afterany:"$dep" --export="$EXP" "$HOME/student_ft.sbatch")
  fi
  echo "segment $i -> job $j (dep=${dep:-none})"
  dep="$j"
done
echo "chain submitted: run=$RUN steps=$STEPS node=$NODE segments=$N"
