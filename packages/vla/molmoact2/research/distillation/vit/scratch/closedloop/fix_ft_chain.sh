#!/usr/bin/env bash
# Cancel the stalled ft/eval/agg jobs (which never checkpointed because save_freq
# was unreachable within a 4h segment) and resubmit the pipeline with the fixed
# save_freq (now baked into student_ft_mi210.sbatch). Distillations are already done,
# so no distill dependency is passed -> ft segments start immediately on idle mi210.
set -u
H=$HOME
echo "== cancelling stalled ft_/ev_/vd_agg jobs =="
squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^(ft_|ev_|vd_agg)/ {print $1}' | while read -r j; do
  scancel "$j" && echo "  scancel $j"
done
sleep 4
echo "== remaining jobs =="
squeue -u "$USER" -o '%.7i %.9P %.13j %.2t %.7M %R'
echo "== resubmit pipeline (FTSEG=5, no distill dep) =="
sed -i 's/\r$//' "$H/student_ft_mi210.sbatch" "$H/vd_eval_ft.sbatch" "$H/vd_aggregate.sbatch" "$H/submit_pipeline.sh" 2>/dev/null || true
FTSEG=5 bash "$H/submit_pipeline.sh"
echo "== queue after resubmit =="
squeue -u "$USER" -o '%.7i %.9P %.13j %.2t %.7M %R'
