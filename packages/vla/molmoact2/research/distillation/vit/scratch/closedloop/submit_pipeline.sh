#!/usr/bin/env bash
# Wire the full post-distillation pipeline as a SLURM dependency DAG:
#   per variant: [ft seg1 .. segN] (afterany-chained, first dep on the distill job)
#                -> eval (afterany last ft segment)
#   then: aggregate (afterany all evals)
# Finetune segments resume from `last`, so the chain finishes the 12k-step run and
# extra segments no-op. Pass distill job ids via env (already-running distillations).
set -u
H=$HOME
FTSEG="${FTSEG:-4}"                 # 4h segments per variant (~3.0s/step -> ~4600 steps/seg)
DJ_siglip_nano="${DJ_siglip_nano:-}"
DJ_hybrid_droid="${DJ_hybrid_droid:-}"
DJ_cnn_fpga="${DJ_cnn_fpga:-}"

submit_variant() {
  local NAME="$1" DJ="$2"
  local dep last i j
  if [ -n "$DJ" ]; then dep="afterany:$DJ"; else dep=""; fi
  last=""
  for i in $(seq 1 "$FTSEG"); do
    if [ -n "$dep" ]; then
      j=$(sbatch --parsable --job-name="ft_$NAME" --dependency="$dep" --export=ALL,NAME="$NAME" "$H/student_ft_mi210.sbatch")
    else
      j=$(sbatch --parsable --job-name="ft_$NAME" --export=ALL,NAME="$NAME" "$H/student_ft_mi210.sbatch")
    fi
    echo "ft $NAME seg$i -> $j (dep=${dep:-none})" >&2
    dep="afterany:$j"; last="$j"
  done
  # eval after last ft segment
  local e
  e=$(sbatch --parsable --job-name="ev_$NAME" --dependency="afterany:$last" --export=ALL,NAME="$NAME" "$H/vd_eval_ft.sbatch")
  echo "eval $NAME -> $e" >&2
  echo "$e"
}

EV=""
for spec in "siglip_nano:$DJ_siglip_nano" "hybrid_droid:$DJ_hybrid_droid" "cnn_fpga:$DJ_cnn_fpga"; do
  NAME="${spec%%:*}"; DJ="${spec##*:}"
  e=$(submit_variant "$NAME" "$DJ" | tail -1)
  EV="${EV:+$EV:}$e"
done

a=$(sbatch --parsable --job-name=vd_agg --dependency="afterany:$EV" "$H/vd_aggregate.sbatch")
echo "aggregate -> $a (dep=afterany:$EV)"
echo "PIPELINE SUBMITTED (ft segments/variant=$FTSEG)"
