# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Offline unit checks for the RTC-style chunk stitching in interactive_server_rt.py.

Pure-numpy, no model / sim / GPU. Validates:
  1. ActivePlan cursor / motion-step invariant.
  2. splice() latency alignment: dropping the already-executed prefix aligns the new
     chunk onto the old one, and the tail past the blend window is the pure new chunk.
  3. The ramp blend strictly reduces the chunk-boundary acceleration/jerk vs a hard
     switch when the two chunks disagree (the smoothness property we care about).
  4. smoothness_metrics() finite-difference sanity (constant velocity => ~0 accel/jerk).

Run inside the molmoact2 image venv (has numpy):
  /opt/libero-venv/bin/python /work/scripts/strix/test_rt_blend.py /work/docker/packages/vla/molmoact2/interactive_server_rt.py
"""
import importlib.util
import sys

import numpy as np


def _load(server_path):
    spec = importlib.util.spec_from_file_location("rt_server", server_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module top is import-safe; main() is __main__-guarded
    return mod


def _const_chunk(n, delta, grip=0.0):
    """n env-space actions (1 env) with constant motion delta + fixed gripper."""
    a = np.zeros((n, 1, 7), dtype=np.float32)
    a[:, 0, :6] = np.asarray(delta, dtype=np.float32)
    a[:, 0, 6] = grip
    return [a[i] for i in range(n)]


def _boundary_jerk(seq):
    """Max |jerk| over a sequence of executed 1-env actions (motion dims only)."""
    arr = np.asarray([s[0, :6] for s in seq], dtype=np.float64)
    accel = np.diff(arr, axis=0)
    jerk = np.diff(accel, axis=0)
    return float(np.max(np.linalg.norm(jerk, axis=1)))


def main():
    server_path = sys.argv[1] if len(sys.argv) > 1 else "interactive_server_rt.py"
    m = _load(server_path)
    ActivePlan, splice, smoothness = m.ActivePlan, m.splice, m.smoothness_metrics

    # ---- 1. ActivePlan invariant: base + cursor tracks the executed motion step ----
    old = ActivePlan(_const_chunk(15, [0.1, 0, 0, 0, 0, 0]), base=0)
    executed = 0
    for _ in range(5):
        assert old.base + old.cursor == executed, "base+cursor must equal motion step"
        a = old.next_action()
        assert a is not None
        executed += 1
    assert old.remaining() == 10
    assert old.base + old.cursor == 5
    print("[ok] ActivePlan cursor/motion-step invariant")

    # ---- 2. splice alignment ----
    # New chunk planned from obs at motion step 3; lands now (motion step 5) => drop 2.
    W = 4
    new = _const_chunk(15, [-0.05, 0, 0, 0, 0, 0], grip=1.0)
    plan_base_motion, cur_motion = 3, 5
    drop = cur_motion - plan_base_motion
    merged = splice(old, new, drop, W, gripper_hyst=0.4)
    assert merged.base == old.base + old.cursor == 5, "merged plan must start at current motion step"
    assert len(merged.actions) == len(new) - drop, "merged length = new chunk minus dropped prefix"
    # Tail past the blend window must be the pure new chunk motion.
    for k in range(W, len(merged.actions)):
        np.testing.assert_allclose(merged.actions[k][0, :6], new[drop + k][0, :6], rtol=0, atol=1e-6)
    # Inside the window the result is a convex blend (strictly between old and new on x).
    for k in range(W):
        x = merged.actions[k][0, 0]
        lo, hi = sorted((old.actions[old.cursor + k][0, 0], new[drop + k][0, 0]))
        assert lo - 1e-6 <= x <= hi + 1e-6, f"blend step {k} not within [old,new]"
    print("[ok] splice latency alignment + tail equals new chunk")

    # ---- 3. blend reduces boundary jerk vs a hard switch ----
    executed_prefix = [old.actions[i] for i in range(old.cursor)]  # 5 already-executed
    hard = executed_prefix + new[drop:]                 # naive: snap to new chunk
    soft = executed_prefix + list(merged.actions)       # ramp-blended
    jerk_hard, jerk_soft = _boundary_jerk(hard), _boundary_jerk(soft)
    assert jerk_soft < jerk_hard, f"blend should lower boundary jerk: soft={jerk_soft} hard={jerk_hard}"
    print(f"[ok] boundary jerk reduced: hard={jerk_hard:.4f} -> blend={jerk_soft:.4f}")

    # ---- 3b. gripper hysteresis: tiny change is ignored, big change flips ----
    old_grip = ActivePlan(_const_chunk(6, [0.0] * 6, grip=0.0), base=0)
    _ = old_grip.next_action()  # cursor=1 so old.remaining()>0 and last_g=0.0
    tiny = splice(old_grip, _const_chunk(6, [0.0] * 6, grip=0.2), drop=0, blend_steps=0, gripper_hyst=0.4)
    assert abs(tiny.actions[0][0, 6] - 0.0) < 1e-6, "small gripper change must be suppressed"
    big = splice(old_grip, _const_chunk(6, [0.0] * 6, grip=1.0), drop=0, blend_steps=0, gripper_hyst=0.4)
    assert abs(big.actions[0][0, 6] - 1.0) < 1e-6, "large gripper change must flip"
    print("[ok] gripper hysteresis")

    # ---- 4. smoothness_metrics sanity ----
    const = [c[0] for c in [np.array([[0.1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)] * 20]
    ma, mj = smoothness(const)
    assert ma < 1e-6 and mj < 1e-6, "constant velocity => ~0 accel/jerk"
    jit = [np.array([[0.1 * ((-1) ** i), 0, 0, 0, 0, 0, 0]], dtype=np.float32)[0] for i in range(20)]
    ma2, mj2 = smoothness(jit)
    assert ma2 > ma and mj2 > mj, "alternating motion => larger accel/jerk"
    print(f"[ok] smoothness_metrics: const=({ma:.3g},{mj:.3g}) jitter=({ma2:.3g},{mj2:.3g})")

    print("\nALL RT-BLEND UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
