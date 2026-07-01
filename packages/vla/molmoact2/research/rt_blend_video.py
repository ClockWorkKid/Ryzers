# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Record a side-by-side async-HOLD vs async-BLEND comparison video for one LIBERO task.

Both panels run the SAME task/seed at the SAME wall-clock control rate with the same real
inference latency. Left = async-hold (the robot pauses while it thinks); right =
async-blend (RTC-style: the planner runs ahead and chunks are latency-aligned + blended so
motion stays continuous). Each panel's banner shows the mode and a THINKING / BLENDING tag.

This is a fully generative comparison (no ground-truth reference), so per workspace rule 2.b
we place the two modes side by side rather than GT-left/pred-right.

Reuses build_engine() from rt_smoothness_ablation.py and the banner helper + ActivePlan/
splice from interactive_server_rt.py. Run inside molmoact2:latest libero venv.

Env: SUITE (libero_object), TASK_ID (3), SEED (1000), THINK (0), NUM_STEPS (4), RT_HZ (20),
RT_MAX_STEPS (200), WALL_CAP (90), RT_REPLAN_AT (-1), RT_BLEND_STEPS (4), PANEL_RES (440),
OUT_DIR (/outputs/rt_smoothness), RT_SERVER (/work/interactive_server_rt.py),
ABL (/work/rt_smoothness_ablation.py).
"""
import importlib.util
import os
import threading
import time
from datetime import datetime

import numpy as np

SUITE = os.environ.get("SUITE", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID", "3"))
SEED = int(os.environ.get("SEED", "1000"))
RT_HZ = float(os.environ.get("RT_HZ", "20"))
RT_MAX_STEPS = int(os.environ.get("RT_MAX_STEPS", "200"))
WALL_CAP = float(os.environ.get("WALL_CAP", "90"))
RT_REPLAN_AT = int(os.environ.get("RT_REPLAN_AT", "-1"))
RT_BLEND_STEPS = int(os.environ.get("RT_BLEND_STEPS", "4"))
PANEL_RES = int(os.environ.get("PANEL_RES", "440"))
OUT_DIR = os.environ.get("OUT_DIR", "/outputs/rt_smoothness")
RT_SERVER = os.environ.get("RT_SERVER", "/work/interactive_server_rt.py")
ABL = os.environ.get("ABL", "/work/rt_smoothness_ablation.py")
DT = 1.0 / RT_HZ


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srv = _load(RT_SERVER, "rt_server")
abl = _load(ABL, "rt_ablation")
ActivePlan, splice = srv.ActivePlan, srv.splice
banner = srv._banner_frame


def _render(env, size):
    le = env.envs[0]
    img = le._env.sim.render(width=size, height=size, camera_name="agentview")
    return np.asarray(img)[::-1, ::-1]


def record(env, instruction, policy, plan_chunk, make_proc, n_envs, blend):
    """Threaded RT loop (mirrors rt_smoothness_ablation.run_threaded) but renders + banners
    a frame each control tick. Returns (frames, hold_pct, success)."""
    import lerobot.scripts.lerobot_eval as ev
    policy.reset()
    env.reset(seed=SEED)
    generator = ev._make_rollout_action_generator(policy, [SEED])
    lock = threading.Lock()
    sh = {"obs": None, "done": False, "stop": False, "last_grip": 0.0,
          "motion_steps": 0, "hold_steps": 0, "plan": None, "chunk_len": 0, "blend_until": -1}
    sh["obs"], _ = env.reset(seed=SEED)

    def replan_at():
        if not blend:
            return 0
        if RT_REPLAN_AT >= 0:
            return RT_REPLAN_AT
        return max(1, sh["chunk_len"] // 2) if sh["chunk_len"] else 6

    def planner():
        while not sh["stop"] and not sh["done"]:
            with lock:
                plan = sh["plan"]
                rem = plan.remaining() if plan is not None else 0
            if rem > replan_at():
                time.sleep(0.005)
                continue
            with lock:
                cur = sh["obs"]
            base_motion = sh["motion_steps"]
            try:
                proc = make_proc(cur, instruction, n_envs)
                chunk, _ = plan_chunk(proc, generator)
            except Exception as e:  # noqa: BLE001
                print("plan error:", e, flush=True)
                time.sleep(0.02)
                continue
            with lock:
                sh["chunk_len"] = len(chunk)
                cur_motion = sh["motion_steps"]
                old = sh["plan"]
                if blend and old is not None and old.remaining() > 0:
                    sh["plan"] = splice(old, chunk, cur_motion - base_motion, RT_BLEND_STEPS, 0.4)
                    sh["blend_until"] = sh["motion_steps"] + sh["hold_steps"] + RT_BLEND_STEPS
                else:
                    drop = (cur_motion - base_motion) if blend else 0
                    d = int(max(0, min(drop, len(chunk) - 1)))
                    sh["plan"] = ActivePlan(chunk[d:], cur_motion)

    pth = threading.Thread(target=planner, daemon=True)
    pth.start()

    frames, success = [], False
    label = "async-blend (RTC)" if blend else "async-hold"
    t0 = time.perf_counter()
    next_t = time.perf_counter()
    while (not sh["done"] and sh["motion_steps"] < RT_MAX_STEPS
           and (time.perf_counter() - t0) < WALL_CAP):
        with lock:
            plan = sh["plan"]
            action = plan.next_action() if plan is not None else None
            if action is not None:
                sh["motion_steps"] += 1
        holding = action is None
        if holding:
            action = np.zeros((n_envs, 7), dtype=np.float32)
            action[:, 6] = sh["last_grip"]
            sh["hold_steps"] += 1
        else:
            sh["last_grip"] = float(action[0, 6])
        obs, done, success = abl._step_success(env, action, success, sh["obs"])
        with lock:
            sh["obs"] = obs
        ticks = sh["motion_steps"] + sh["hold_steps"]
        blending = (not holding) and blend and ticks <= sh["blend_until"]
        tag = "THINKING" if holding else ("BLENDING" if blending else "")
        hp = 100.0 * sh["hold_steps"] / max(1, ticks)
        frames.append(banner(_render(env, PANEL_RES), f"{label}  hold {hp:.0f}%", PANEL_RES, tag))
        if done:
            sh["done"] = True
        next_t += DT
        s = next_t - time.perf_counter()
        if s > 0:
            time.sleep(s)
        else:
            next_t = time.perf_counter()
    sh["done"] = True
    sh["stop"] = True
    pth.join(timeout=5.0)
    hp = 100.0 * sh["hold_steps"] / max(1, sh["motion_steps"] + sh["hold_steps"])
    print(f"[record {label}] frames={len(frames)} hold%={hp:.0f} success={success}", flush=True)
    return frames, hp, success


def compose(left, right):
    """Stack two equal-height frame lists side by side; pad the shorter by holding its last
    frame so both panels stay in lockstep wall-clock time."""
    n = max(len(left), len(right))
    if left:
        left = left + [left[-1]] * (n - len(left))
    if right:
        right = right + [right[-1]] * (n - len(right))
    div = np.full((left[0].shape[0], 6, 3), 30, dtype=np.uint8)
    return [np.concatenate([left[i], div, right[i]], axis=1) for i in range(n)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policy, build_env, plan_chunk, make_proc, torch = abl.build_engine()
    env = build_env(TASK_ID)
    instruction = getattr(env.envs[0], "task_description", "") or "complete the task"

    # warm up kernels once
    import lerobot.scripts.lerobot_eval as ev
    wobs, _ = env.reset(seed=SEED)
    wgen = ev._make_rollout_action_generator(policy, [SEED])
    with torch.inference_mode():
        policy.select_action(make_proc(wobs, instruction, 1), generator=wgen)
    policy.reset()
    print("[warmup] done", flush=True)

    hold_frames, _, _ = record(env, instruction, policy, plan_chunk, make_proc, 1, blend=False)
    blend_frames, _, _ = record(env, instruction, policy, plan_chunk, make_proc, 1, blend=True)
    frames = compose(hold_frames, blend_frames)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"blend_vs_hold_{SUITE}_t{TASK_ID}_{ts}.mp4")
    import imageio
    with imageio.get_writer(path, fps=int(RT_HZ), codec="libx264", quality=7,
                            macro_block_size=1, output_params=["-pix_fmt", "yuv420p"]) as w:
        for fr in frames:
            w.append_data(fr)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
