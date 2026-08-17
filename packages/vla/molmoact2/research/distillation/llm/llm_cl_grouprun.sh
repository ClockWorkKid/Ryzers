#!/usr/bin/env bash
# Run a GROUP of closed-loop LLM-backbone evals sequentially on one 4-GPU node.
# Each arm (a tag:blob spec; empty blob = FP baseline) runs llm_cl_inner.sh, which
# fans the 4 LIBERO suites across the 4 GPUs. Env: ARMS, NEP, CKPT.
set -u
NEP="${NEP:-10}"
CKPT="${CKPT:-/outputs/base_teacher/checkpoints/000000/pretrained_model}"
echo "[llm_cl_group] arms='$ARMS' nep=$NEP t=$(date -Is)"
for spec in $ARMS; do
  tag="${spec%%:*}"; blob="${spec#*:}"
  [ "$blob" = "$tag" ] && blob=""   # no ':' -> FP baseline
  echo "===== ARM $tag  blob='${blob:-<FP>}'  $(date -Is) ====="
  ARM="$tag" QUANT_STATE="$blob" CKPT="$CKPT" \
    RESDIR="/outputs/llm_cl/$tag" N_EP="$NEP" NGPU=4 STAGGER="${STAGGER:-90}" \
    bash /work/llm_cl_inner.sh
done
echo "[llm_cl_group] ALL DONE arms='$ARMS' t=$(date -Is)"
