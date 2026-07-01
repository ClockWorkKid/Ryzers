#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Reset the HF modeling snapshot to pristine (RESET=1), apply the pruning patch
# stack (reduce -> groupdrop -> attnfeedback), then run the attention-feedback
# isolation check. Runs INSIDE the molmoact2 container.
set -uo pipefail
HF=/root/.cache/huggingface
P=/scripts
export MOLMOACT2_DTYPE=bfloat16 DTYPE=bfloat16
export VIS_GROUPDROP_KEEP_FRAC="${VIS_GROUPDROP_KEEP_FRAC:-0.5}"
export VIS_ATTNFB=1 VIS_ATTNFB_SELFCARRY=1 VIS_ATTNFB_DEBUG=1
RESET="${RESET:-1}"

if [ "$RESET" = "1" ]; then
  echo "===== reset modeling to pristine (both repos) ====="
  python - <<'PY'
from huggingface_hub import hf_hub_download
for repo in ["allenai/MolmoAct2-DROID", "allenai/MolmoAct2-Think-LIBERO"]:
    try:
        p = hf_hub_download(repo, "modeling_molmoact2.py", force_download=True)
        print("redownloaded", p)
    except Exception as e:
        print("skip", repo, repr(e))
PY
  # drop the trust_remote_code cached copies so they regenerate from the (re-patched) snapshot
  find "$HF/modules/transformers_modules" -name modeling_molmoact2.py -delete 2>/dev/null || true
fi

echo "===== apply patch stack ====="
python "$P/patch_vis_reduce.py" "$HF"
python "$P/patch_vis_groupdrop.py" "$HF"
python "$P/patch_vis_attnfeedback.py" "$HF"

echo "===== validate ====="
python "$P/validate_attnfeedback.py"
rc=$?
echo "exit=$rc"
exit $rc
