# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Sourced by the shared-base MolmoAct2 LIBERO demos: launch the lerobot MolmoAct2 policy
# server in the isolated /opt/libero-venv (numpy 2 / transformers 5.3) as a localhost HTTP
# service, so the shared simulation/libero harness (numpy 1.26 base venv) can drive it via
# adapters/molmoact2_libero_policy.py. Expects CKPT/THINK/SUITE/NUM_STEPS/MM2_SERVER_PORT/
# OUT_DIR to be exported by the caller. Sets an EXIT trap to stop the server.
LV=/opt/libero-venv/bin/python
if [ ! -x "$LV" ]; then
  echo "ERROR: /opt/libero-venv not found (MolmoAct2 policy stack)." >&2
  echo "       Build the chain:  ryzers build simulation/libero molmoact2" >&2
  exit 1
fi
mkdir -p "${OUT_DIR:-/outputs}"
SERVER_LOG="${OUT_DIR:-/outputs}/mm2_policy_server.log"
echo "[demo] starting MolmoAct2 policy server (/opt/libero-venv) on 127.0.0.1:${MM2_SERVER_PORT:-8790} (log: $SERVER_LOG)"
"$LV" /ryzers/molmoact2_policy_server.py > "$SERVER_LOG" 2>&1 &
MM2_SERVER_PID=$!
trap 'kill $MM2_SERVER_PID 2>/dev/null || true' EXIT
