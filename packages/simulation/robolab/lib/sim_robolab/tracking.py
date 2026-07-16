# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gate G2: JOINT_POSITION controller tracking validation (no model).

Runs the deterministic ScriptedJointPolicy through the real task scene (real YCB assets, real
Franka + Robotiq embodiment) via the standard predict -> scene.step -> JOINT_POSITION path,
recording commanded vs achieved joint angles every control step. Produces:
  - tracking_<task>.png : per-joint commanded (dashed) vs achieved (solid) overlay + error,
                          plus gripper command vs achieved openness (rule 2.a).
  - tracking_<task>.mp4 : the composed 3-view motion (sanity of the real scene).
And prints per-joint RMSE / max abs error and an overall PASS/FAIL (steady-state, post-warmup).

Env: TASK, SEED, STEPS, WARMUP, AMPLITUDE, PERIOD, OUT_DIR.
"""
import os
from datetime import datetime

import numpy as np

from sim_robolab.envutil import env_int, env_str
from sim_robolab.render import banner_frame, compose_view, save_mp4
from sim_robolab.scene import build_scene
from sim_robolab.scripted_policy import ScriptedJointPolicy

# Gate thresholds (rad) on the 7 arm joints, measured after WARMUP steps.
RMS_TOL = 0.05
MAXABS_TOL = 0.15


def run_tracking(task, seed, steps, warmup, amplitude, period, out_dir, render_every=1):
    scene = build_scene(task, seed=seed)
    policy = ScriptedJointPolicy(amplitude=amplitude, period_steps=period,
                                 gripper_period_steps=period)
    obs = scene.reset()
    policy.reset(scene.description)

    cmd, ach = [], []           # [T,7] commanded / achieved arm joint angles
    gcmd, gach = [], []         # gripper command / achieved openness
    frames = []
    pending = []
    for t in range(steps):
        if not pending:
            pending = list(policy.predict_action_chunk(obs, scene.description))
        row = np.asarray(pending.pop(0), dtype=np.float64)
        scene.step(row)
        cmd.append(row[:7].copy())
        ach.append(scene.arm_qpos().copy())
        gcmd.append(float(row[7]))
        gach.append(scene.gripper_openness())
        obs = scene.observe()
        if t % render_every == 0:
            frames.append(banner_frame(compose_view(obs["view"]),
                                       f"scripted joint tracking: {task}", 720))

    scene.close()
    return np.array(cmd), np.array(ach), np.array(gcmd), np.array(gach), frames


def _metrics(cmd, ach, warmup):
    err = ach[warmup:] - cmd[warmup:]
    rms = np.sqrt(np.mean(err ** 2, axis=0))          # [7]
    maxabs = np.max(np.abs(err), axis=0)              # [7]
    return rms, maxabs


def plot_tracking(cmd, ach, gcmd, gach, rms, maxabs, warmup, path, control_hz=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = cmd.shape[0]
    tsec = np.arange(T) / float(control_hz)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()
    for j in range(7):
        ax = axes[j]
        ax.plot(tsec, cmd[:, j], "--", color="tab:orange", lw=1.6, label="commanded")
        ax.plot(tsec, ach[:, j], "-", color="tab:blue", lw=1.2, label="achieved")
        ax.axvline(warmup / control_hz, color="gray", ls=":", lw=0.8)
        ax.set_title(f"joint {j}:  RMSE={rms[j]:.4f}  max|e|={maxabs[j]:.4f} rad", fontsize=9)
        ax.set_ylabel("rad", fontsize=8)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    # gripper panel
    axg = axes[7]
    axg.plot(tsec, gcmd, "--", color="tab:orange", lw=1.4, label="cmd action")
    axg.plot(tsec, gach, "-", color="tab:green", lw=1.2, label="achieved openness")
    axg.set_title("gripper", fontsize=9)
    axg.grid(alpha=0.3)
    axg.legend(fontsize=8, loc="upper right")
    for ax in axes[6:]:
        ax.set_xlabel("time (s)", fontsize=8)
    fig.suptitle("G2: JOINT_POSITION controller tracking (commanded vs achieved)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    task = env_str("TASK", "BananaInBowl")
    seed = env_int("SEED", 0)
    steps = env_int("STEPS", 200)
    warmup = env_int("WARMUP", 20)
    amplitude = float(env_str("AMPLITUDE", "0.2"))
    period = env_int("PERIOD", 100)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[g2] tracking {task}: steps={steps} amp={amplitude} period={period}", flush=True)
    cmd, ach, gcmd, gach, frames = run_tracking(
        task, seed, steps, warmup, amplitude, period, out_dir
    )
    rms, maxabs = _metrics(cmd, ach, warmup)

    ts = datetime.now().strftime("%H%M%S")
    plot_path = os.path.join(out_dir, f"tracking_{task}_{ts}.png")
    plot_tracking(cmd, ach, gcmd, gach, rms, maxabs, warmup, plot_path)
    if frames:
        save_mp4(frames, os.path.join(out_dir, f"tracking_{task}_{ts}.mp4"), fps=20)

    print("[g2] per-joint tracking (post-warmup):", flush=True)
    for j in range(7):
        print(f"    joint {j}: RMSE={rms[j]:.4f} rad  max|e|={maxabs[j]:.4f} rad", flush=True)
    ok = bool(np.mean(rms) < RMS_TOL and np.max(maxabs) < MAXABS_TOL)
    print(f"[g2] mean RMSE={np.mean(rms):.4f} (tol {RMS_TOL})  "
          f"max|e|={np.max(maxabs):.4f} (tol {MAXABS_TOL})", flush=True)
    print(f"[g2] {'PASS' if ok else 'FAIL'}: saved {plot_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
