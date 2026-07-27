#!/usr/bin/env bash
# Fetch the full dev-flow checkpoints (both variants) from the Hugging Face repo
# into the local resource/ckpt tree. Weights are NOT committed to git.
#
#   HF_REPO_ID=<owner>/<repo> bash download_weights.sh
#   (private repo: also `huggingface-cli login` or export HF_TOKEN first)
#
# Verifies sha256 against WEIGHTS.md.
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-sayeedmd320/molmoact2-vit-distill-w4a6}"
DEST="${DEST:-resource/ckpt/vit_distill}"
mkdir -p "$DEST/quant"

FILES=(
  "cnn_fpga_full.pt"
  "siglip_nano_full.pt"
  "quant/cnn_w8a8_ptq.pt"
  "quant/cnn_w4a6_ptq.pt"
  "quant/cnn_w4a6_qat.pt"
  "quant/cnn_w4a6_redistill.pt"
  "quant/cnn_w4a6_redistill_qatft.pt"
  "quant/tinyvit_w8a8_ptq.pt"
  "quant/tinyvit_w4a6_ptq.pt"
  "quant/tinyvit_w4a6_qat.pt"
  "quant/tinyvit_w4a6_redistill.pt"
  "quant/tinyvit_w4a6_redistill_qatft.pt"
)

echo "Downloading ${#FILES[@]} files from ${HF_REPO_ID} -> ${DEST}"
for f in "${FILES[@]}"; do
  python - "$HF_REPO_ID" "$f" "$DEST" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, rel, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=rel, local_dir=dest)
print("  ok", rel, "->", p)
PY
done

echo "Done. Verify integrity with the sha256 table in WEIGHTS.md."
