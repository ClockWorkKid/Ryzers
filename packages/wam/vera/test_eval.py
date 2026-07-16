# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Closed-loop eval smoke for the VERA sim image on Strix Halo (gfx1151).

Complements `test.py` (core planner+IDM env). Proves the `.[eval]` sim extra imports
and — the top gfx1151 risk — that MuJoCo can render **offscreen** in-container with no
X11. Tries EGL first (hardware, on the iGPU) then OSMesa (software) as a fallback, each
in a fresh subprocess since MuJoCo binds its GL backend once per process. Passes if the
sim deps import and at least one GL backend renders a frame; prints which backend works
so the closed-loop wrappers can pin `MUJOCO_GL` accordingly. No weights, no network.
"""
import os
import subprocess
import sys

# Minimal MuJoCo model — one lit body so a rendered frame has non-trivial content.
_RENDER_PROBE = r"""
import os, sys, numpy as np
backend = os.environ.get("MUJOCO_GL", "?")
import mujoco
xml = '''
<mujoco>
  <visual><global offwidth="128" offheight="128"/></visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <body pos="0 0 0.3"><geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.2 1"/></body>
  </worldbody>
</mujoco>
'''
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=128, width=128)
r.update_scene(d)
img = r.render()
assert img.shape == (128, 128, 3), img.shape
print(f"RENDER_OK backend={backend} mujoco={mujoco.__version__} "
      f"frame={img.shape} mean={float(np.asarray(img).mean()):.1f}")
"""


def _try_render(backend: str) -> bool:
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    print(f"--- offscreen render probe: MUJOCO_GL={backend} ---", flush=True)
    p = subprocess.run([sys.executable, "-c", _RENDER_PROBE], env=env,
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    print(out, flush=True)
    return "RENDER_OK" in p.stdout


def main() -> int:
    ok = True

    # 1) Sim stack imports (report versions; a missing one fails the smoke).
    print("== sim deps ==", flush=True)
    for name in ("gymnasium", "gym_pusht", "mujoco", "robosuite", "robomimic", "mimicgen"):
        try:
            mod = __import__(name)
            print(f"  {name:12s}: {getattr(mod, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:12s}: FAIL {type(e).__name__}: {e}", file=sys.stderr)
            ok = False

    # 1b) MimicGen must be NVlabs/mimicgen (the runner needs create_env). The PyPI "mimicgen"
    # stub imports at top level but lacks `mimicgen.utils` — guard against that regression.
    try:
        from mimicgen.utils.robomimic_utils import create_env  # noqa: F401
        print("  mimicgen.utils.robomimic_utils.create_env: ok (NVlabs/mimicgen)")
    except Exception as e:  # noqa: BLE001
        print(f"  mimicgen.create_env: FAIL {type(e).__name__}: {e} "
              "(need NVlabs/mimicgen, not the PyPI stub)", file=sys.stderr)
        ok = False

    # 2) VERA closed-loop entry points (server + controller import path).
    print("== vera closed-loop imports ==", flush=True)
    try:
        import vera.server      # noqa: F401  websocket policy server + live viewer
        import vera.controller  # noqa: F401  PushT / MimicGen eval controllers
        print("  vera.server, vera.controller: ok")
    except Exception as e:  # noqa: BLE001
        print(f"  vera.{{server,controller}}: FAIL {type(e).__name__}: {e}", file=sys.stderr)
        ok = False

    # 3) Offscreen GL — EGL (hw) first, OSMesa (sw) fallback.
    print("== mujoco offscreen render ==", flush=True)
    egl = _try_render("egl")
    osmesa = egl or _try_render("osmesa")
    if egl:
        print("RENDER backend chosen: egl (hardware)")
    elif osmesa:
        print("RENDER backend chosen: osmesa (software fallback — EGL unavailable on gfx1151)")
    else:
        print("FAIL: neither EGL nor OSMesa rendered offscreen.", file=sys.stderr)
        ok = False

    print("PASS: VERA eval/sim env OK" if ok else "FAIL: VERA eval smoke incomplete",
          file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
