#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Apply the patch stack and run the attention-feedback latency bench. Runs INSIDE
# the molmoact2 container.
set -uo pipefail
HF=/root/.cache/huggingface
P=/scripts
export MOLMOACT2_DTYPE=bfloat16 DTYPE=bfloat16
export VIS_GROUPDROP_KEEP_FRAC="${VIS_GROUPDROP_KEEP_FRAC:-0.5}"
RESET="${RESET:-1}"

if [ "$RESET" = "1" ]; then
  echo "===== reset modeling to pristine ====="
  python - <<'PY'
from huggingface_hub import hf_hub_download
for repo in ["allenai/MolmoAct2-DROID", "allenai/MolmoAct2-Think-LIBERO"]:
    try:
        print("redownloaded", hf_hub_download(repo, "modeling_molmoact2.py", force_download=True))
    except Exception as e:
        print("skip", repo, repr(e))
PY
  find "$HF/modules/transformers_modules" -name modeling_molmoact2.py -delete 2>/dev/null || true
fi
echo "===== apply patch stack ====="
python "$P/patch_vis_reduce.py" "$HF"
python "$P/patch_vis_groupdrop.py" "$HF"
python "$P/patch_vis_attnfeedback.py" "$HF"
echo "===== bench ====="
python "$P/bench_attnfb_latency.py"
