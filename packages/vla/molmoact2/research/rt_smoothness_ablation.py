# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Headless ablation: how chunk-stitching affects real-time motion smoothness.

Runs the MolmoAct2 x LIBERO closed loop under three regimes, all at the same fixed
wall-clock control rate (RT_HZ) and the SAME real inference latency, so the planner
genuinely has to keep up with a 20 Hz robot:

  sync         : single-threaded stop-and-decide. Execute the whole chunk, then BLOCK
                 to plan the next one (robot frozen / commanded to hold during the plan).
  async-hold   : the current RT demo. A planner thread refills only when the buffer is
                 empty; the robot HOLDs its pose (zero delta) while it thinks.
  async-blend  : RTC-style. The planner runs ahead (RT_REPLAN_AT) and each new chunk is
                 latency-aligned + ramp-blended onto the one in flight (splice()).

For every episode we record the per-tick commanded action stream (including holds, which
are the "ugly boundary pauses"), then report success, hold/stall %, throughput
(motion steps / wall second), and smoothness (mean |accel|, max |jerk| of the commanded
motion deltas). Per-episode action streams are dumped as .npy for the plot script.

This is research (token-drop / efficiency track), not part of the shipped ryzers demo.

Env knobs:
  SUITE (libero_object), TASKS (3,2 | comma list), SEED (1000), THINK (0), NUM_STEPS (4),
  CKPT, RT_HZ (20), RT_MAX_STEPS (400), RT_REPLAN_AT (-1=auto), RT_BLEND_STEPS (4),
  RT_GRIPPER_HYST (0.4), MODES (sync,hold,blend), OUT_DIR (/outputs/rt_smoothness),
  RT_SERVER (/ryzers/interactive_server_rt.py).
Usage: /opt/libero-venv/bin/python rt_smoothness_ablation.py
"""
import importlib.util
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

SUITE = os.environ.get("SUITE", "libero_object")
TASKS = [int(x) for x in os.environ.get("TASKS", "3,2").split(",") if x.strip() != ""]
SEED = int(os.environ.get("SEED", "1000"))
THINK = os.environ.get("THINK", "0") == "1"
CKPT = os.environ.get("CKPT", "allenai/MolmoAct2-Think-LIBERO")
RT_HZ = float(os.environ.get("RT_HZ", "20"))
RT_MAX_STEPS = int(os.environ.get("RT_MAX_STEPS", "250"))  # cap on MOTION steps (task progress)
WALL_CAP = float(os.environ.get("WALL_CAP", "150"))        # per-episode wall-clock safety cap (s)
RT_REPLAN_AT = int(os.environ.get("RT_REPLAN_AT", "-1"))
RT_BLEND_STEPS = int(os.environ.get("RT_BLEND_STEPS", "4"))
RT_GRIPPER_HYST = float(os.environ.get("RT_GRIPPER_HYST", "0.4"))
MODES = [m.strip() for m in os.environ.get("MODES", "sync,hold,blend").split(",") if m.strip()]
OUT_DIR = os.environ.get("OUT_DIR", "/outputs/rt_smoothness")
RT_SERVER = os.environ.get("RT_SERVER", "/ryzers/interactive_server_rt.py")
DT = 1.0 / RT_HZ


def _load_helpers(path):
    spec = importlib.util.spec_from_file_location("rt_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ActivePlan, mod.splice, mod.smoothness_metrics


ActivePlan, splice, smoothness_metrics = _load_helpers(RT_SERVER)


def build_engine():
    import draccus
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.scripts.lerobot_eval as ev

    depth = "True" if THINK else "False"
    base_args = [
        "--policy.type=molmoact2", f"--policy.checkpoint_path={CKPT}",
        "--policy.inference_action_mode=continuous",
        f"--policy.enable_depth_reasoning={depth}", f"--policy.enable_adaptive_depth={depth}",
        "--policy.enable_cuda_graph=False", "--policy.norm_tag=libero",
        "--policy.device=cuda", "--env.type=libero",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        f"--seed={SEED}", "--output_dir=/tmp/rt_ablation_eval",
    ]
    if os.environ.get("NUM_STEPS"):
        base_args.append(f"--policy.num_steps={os.environ['NUM_STEPS']}")

    base_cfg = draccus.parse(EvalPipelineConfig, args=base_args + [f"--env.task={SUITE}", f"--env.task_ids=[{TASKS[0]}]"])
    policy = make_policy(cfg=base_cfg.policy, env_cfg=base_cfg.env, rename_map=base_cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=base_cfg.policy, pretrained_path=base_cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": base_cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=base_cfg.env, policy_cfg=base_cfg.policy)
    ACTION = ev.ACTION
    preprocess_observation = ev.preprocess_observation

    def build_env(task_id):
        cfg = draccus.parse(EvalPipelineConfig, args=base_args + [f"--env.task={SUITE}", f"--env.task_ids=[{task_id}]"])
        envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
        env = envs[SUITE][task_id]
        # Holds also step the env in async modes, so the per-tick env-step count far
        # exceeds motion steps; keep the gymnasium TimeLimit huge and let our motion /
        # wall caps govern episode length instead.
        for e in env.envs:
            e._max_episode_steps = 100000
            try:
                e._env.env.horizon = 100000  # robosuite internal horizon (best-effort)
            except Exception:
                pass
        env.reset(seed=SEED)
        return env

    import torch

    def plan_chunk(proc_obs, generator):
        with torch.inference_mode():
            a0 = policy.select_action(proc_obs, generator=generator)
        raw = [a0]
        q = policy._action_queues[0]
        while q:
            a = q.popleft()
            if a.ndim == 1:
                a = a.unsqueeze(0)
            raw.append(a.to(device=a0.device, dtype=torch.float32))
        out = []
        for a in raw:
            pa = postprocessor(a)
            pa = env_post({ACTION: pa})[ACTION]
            out.append(pa.to("cpu").numpy())
        infer_s = float(getattr(policy, "_last_model_inference_s", 0.0))
        return out, infer_s

    def make_proc(obs, instruction, n_envs):
        proc = preprocess_observation(obs)
        proc["task"] = [instruction for _ in range(n_envs)]
        proc = env_pre(proc)
        proc = preprocessor(proc)
        return proc

    return policy, build_env, plan_chunk, make_proc, torch


def _step_success(env, action, success, last_obs):
    try:
        obs, _, terminated, truncated, info = env.step(action)
    except ValueError:
        # robosuite raises "executing action in terminated episode" if its internal
        # horizon fired without gymnasium surfacing it; treat as a clean episode end.
        return last_obs, True, success
    if "final_info" in info:
        try:
            success = bool(np.asarray(info["final_info"]["is_success"]).any())
        except Exception:
            pass
    done = bool(np.any(terminated) or np.any(truncated))
    return obs, done, success


def run_sync(env, instruction, policy, plan_chunk, make_proc, n_envs):
    """Stop-and-decide: execute the whole chunk, then block to plan. The planning gap is
    recorded as hold ticks (robot commanded to its last pose) in the action stream."""
    policy.reset()
    obs, _ = env.reset(seed=SEED)
    import lerobot.scripts.lerobot_eval as ev
    generator = ev._make_rollout_action_generator(policy, [SEED])
    applied, success, last_grip = [], False, 0.0
    motion_steps = hold_steps = 0
    infers = []
    t0 = time.perf_counter()
    done = False
    while not done and motion_steps < RT_MAX_STEPS and (time.perf_counter() - t0) < WALL_CAP:
        proc = make_proc(obs, instruction, n_envs)
        chunk, infer_s = plan_chunk(proc, generator)
        infers.append(infer_s)
        # planning gap: robot frozen, commanded hold -> record as hold ticks
        for _ in range(max(0, int(round(infer_s / DT)))):
            hold = np.zeros((n_envs, 7), dtype=np.float32)
            hold[:, 6] = last_grip
            applied.append(hold[0].copy())
            hold_steps += 1
        for a in chunk:
            if done or motion_steps >= RT_MAX_STEPS:
                break
            last_grip = float(a[0, 6])
            applied.append(np.asarray(a[0], dtype=np.float32).copy())
            motion_steps += 1
            obs, done, success = _step_success(env, a, success, obs)
            time.sleep(DT)
    wall = time.perf_counter() - t0
    return dict(applied=applied, success=success, motion_steps=motion_steps,
                hold_steps=hold_steps, wall=wall,
                infer_ms=1000.0 * (sum(infers) / len(infers)) if infers else 0.0)


def run_threaded(env, instruction, policy, plan_chunk, make_proc, n_envs, blend):
    """async-hold (blend=False) and async-blend (blend=True): SIM thread at RT_HZ, planner
    thread refilling. Mirrors interactive_server_rt.run_command, headless (no rendering)."""
    policy.reset()
    obs0, _ = env.reset(seed=SEED)
    import lerobot.scripts.lerobot_eval as ev
    generator = ev._make_rollout_action_generator(policy, [SEED])

    lock = threading.Lock()
    sh = {"obs": obs0, "done": False, "stop": False, "last_grip": 0.0,
          "motion_steps": 0, "hold_steps": 0, "plan": None, "chunk_len": 0,
          "infers": []}

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
                chunk, infer_s = plan_chunk(proc, generator)
            except Exception as e:  # noqa: BLE001
                print("plan error:", e, flush=True)
                time.sleep(0.02)
                continue
            with lock:
                sh["infers"].append(infer_s)
                sh["chunk_len"] = len(chunk)
                cur_motion = sh["motion_steps"]
                old = sh["plan"]
                if blend and old is not None and old.remaining() > 0:
                    sh["plan"] = splice(old, chunk, cur_motion - base_motion, RT_BLEND_STEPS, RT_GRIPPER_HYST)
                else:
                    drop = (cur_motion - base_motion) if blend else 0
                    d = int(max(0, min(drop, len(chunk) - 1)))
                    sh["plan"] = ActivePlan(chunk[d:], cur_motion)

    pth = threading.Thread(target=planner, daemon=True)
    pth.start()

    applied, success = [], False
    t0 = time.perf_counter()
    next_t = time.perf_counter()
    steps = 0
    while (not sh["done"] and sh["motion_steps"] < RT_MAX_STEPS
           and (time.perf_counter() - t0) < WALL_CAP):
        with lock:
            plan = sh["plan"]
            action = plan.next_action() if plan is not None else None
            if action is not None:
                sh["motion_steps"] += 1
        if action is None:
            action = np.zeros((n_envs, 7), dtype=np.float32)
            action[:, 6] = sh["last_grip"]
            sh["hold_steps"] += 1
        else:
            sh["last_grip"] = float(action[0, 6])
        applied.append(np.asarray(action[0], dtype=np.float32).copy())
        obs, done, success = _step_success(env, action, success, sh["obs"])
        with lock:
            sh["obs"] = obs
        steps += 1
        if done:
            sh["done"] = True
        next_t += DT
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()
    sh["done"] = True
    sh["stop"] = True
    pth.join(timeout=5.0)
    wall = time.perf_counter() - t0
    inf = sh["infers"]
    return dict(applied=applied, success=success, motion_steps=sh["motion_steps"],
                hold_steps=sh["hold_steps"], wall=wall,
                infer_ms=1000.0 * (sum(inf) / len(inf)) if inf else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policy, build_env, plan_chunk, make_proc, torch = build_engine()
    n_envs = 1

    # Warm up JIT/flash kernels once so the first timed plan isn't a 20s outlier.
    try:
        env0 = build_env(TASKS[0])
        wobs, _ = env0.reset(seed=SEED)
        import lerobot.scripts.lerobot_eval as ev
        wgen = ev._make_rollout_action_generator(policy, [SEED])
        proc = make_proc(wobs, "pick up the object", n_envs)
        with torch.inference_mode():
            policy.select_action(proc, generator=wgen)
        policy.reset()
        print("[warmup] done", flush=True)
    except Exception as e:  # noqa: BLE001
        print("warmup failed:", e, flush=True)

    rows = []
    runners = {"sync": lambda env, instr: run_sync(env, instr, policy, plan_chunk, make_proc, n_envs),
               "hold": lambda env, instr: run_threaded(env, instr, policy, plan_chunk, make_proc, n_envs, blend=False),
               "blend": lambda env, instr: run_threaded(env, instr, policy, plan_chunk, make_proc, n_envs, blend=True)}

    for task_id in TASKS:
        env = build_env(task_id)
        instruction = getattr(env.envs[0], "task_description", "") or "complete the task"
        for mode in MODES:
            if mode not in runners:
                print("unknown mode, skip:", mode, flush=True)
                continue
            t = time.perf_counter()
            res = runners[mode](env, instruction)
            ma, mj = smoothness_metrics(res["applied"])
            total = res["motion_steps"] + res["hold_steps"]
            hold_pct = 100.0 * res["hold_steps"] / max(1, total)
            thru = res["motion_steps"] / max(1e-6, res["wall"])
            tag = f"{SUITE}_t{task_id}_{mode}"
            np.save(os.path.join(OUT_DIR, f"{tag}_actions.npy"),
                    np.asarray(res["applied"], dtype=np.float32))
            row = dict(suite=SUITE, task_id=task_id, mode=mode,
                       success=int(bool(res["success"])), motion_steps=res["motion_steps"],
                       hold_steps=res["hold_steps"], hold_pct=round(hold_pct, 1),
                       wall_s=round(res["wall"], 2), throughput_sps=round(thru, 2),
                       infer_ms=round(res.get("infer_ms", 0.0), 0),
                       mean_accel=round(ma, 5), max_jerk=round(mj, 5))
            rows.append(row)
            print(f"[{tag}] success={row['success']} hold%={row['hold_pct']} "
                  f"thru={row['throughput_sps']}sps infer={row['infer_ms']}ms "
                  f"accel={row['mean_accel']} jerk={row['max_jerk']} "
                  f"({time.perf_counter()-t:.1f}s)", flush=True)
        try:
            env.close()
        except Exception:
            pass

    # CSV + JSON summary (small artifacts the laptop can pull)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = ["suite", "task_id", "mode", "success", "motion_steps", "hold_steps", "hold_pct",
            "wall_s", "throughput_sps", "infer_ms", "mean_accel", "max_jerk"]
    csv_path = os.path.join(OUT_DIR, f"summary_{ts}.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(os.path.join(OUT_DIR, f"summary_{ts}.json"), "w") as f:
        json.dump({"config": {"suite": SUITE, "tasks": TASKS, "rt_hz": RT_HZ,
                              "max_steps": RT_MAX_STEPS, "blend_steps": RT_BLEND_STEPS,
                              "replan_at": RT_REPLAN_AT, "think": THINK,
                              "num_steps": os.environ.get("NUM_STEPS", "")},
                   "rows": rows}, f, indent=2)

    # per-mode aggregate
    print("\n=== aggregate (mean over tasks) ===", flush=True)
    for mode in MODES:
        mr = [r for r in rows if r["mode"] == mode]
        if not mr:
            continue
        def avg(k):
            return sum(r[k] for r in mr) / len(mr)
        print(f"{mode:6s}  succ={avg('success'):.2f}  hold%={avg('hold_pct'):.1f}  "
              f"thru={avg('throughput_sps'):.2f}sps  infer={avg('infer_ms'):.0f}ms  "
              f"accel={avg('mean_accel'):.4f}  jerk={avg('max_jerk'):.4f}", flush=True)
    print("\nwrote", csv_path, flush=True)


if __name__ == "__main__":
    main()
