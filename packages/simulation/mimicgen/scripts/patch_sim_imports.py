# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Sim-side (client) port patches for the MimicGen harness on gfx1151.

Only the simulator/client half of VERA's port fixes lives here — the harness installs the
websocket *client* (`vera.server.protocol.websocket_policy_client`) but not the model server or
the planner/IDM/tracker stack (those are patched by the model layer that builds FROM this base).

WebSocket keepalive (gfx1151 chunk stalls). At the shipped sample_steps a single closed-loop
chunk can stall well past the 600s ping timeout on gfx1151 (hipblaslt->unfused-cublas fallbacks +
memory pressure), tripping the client keepalive mid-denoise ("sent 1011 keepalive ping timeout")
and killing the episode. The server does finish the chunk, so we disable the client keepalive; the
finite PING_TIMEOUT_SECS is kept only to bound the initial handshake (open_timeout). Idempotent.
"""
import sys
from pathlib import Path

SENTINEL = "[ryzers-sim-patch]"

WS_CLIENT_OLD = (
    "PING_INTERVAL_SECS = 60\n"
    "PING_TIMEOUT_SECS = 600\n"
)
WS_CLIENT_NEW = (
    "# " + SENTINEL + " disable client keepalive (gfx1151 chunk stalls > 600s trip it); keep\n"
    "# PING_TIMEOUT_SECS finite for the initial handshake open_timeout only.\n"
    "PING_INTERVAL_SECS = None\n"
    "PING_TIMEOUT_SECS = 600\n"
)


def _patch(path: Path, old: str, new: str) -> str:
    if not path.exists():
        return f"SKIP (missing): {path}"
    text = path.read_text()
    if SENTINEL in text:
        return f"SKIP (already patched): {path}"
    if old not in text:
        return f"WARN (block not found, upstream drift?): {path}"
    path.write_text(text.replace(old, new, 1))
    return f"PATCHED: {path}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/repos/vera")
    ws_client = root / "vera" / "server" / "protocol" / "websocket_policy_client.py"
    print(_patch(ws_client, WS_CLIENT_OLD, WS_CLIENT_NEW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
