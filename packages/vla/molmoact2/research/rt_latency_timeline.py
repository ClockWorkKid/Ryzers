# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Latency-overlap breakdown of the RT demo: async-HOLD (no intermediate inference) vs
async-BLEND (intermediate inference = plan the next chunk WHILE the current one executes).

Runs the real MolmoAct2 x LIBERO closed loop for both modes at the same wall-clock rate
with the same real inference latency, logging precise wall-clock events:
  * plan windows : [t_start, t_end] of every policy forward (the planner "thinking"),
                   plus the motion step it was triggered at and attached at.
  * control ticks: per 1/RT_HZ tick, the motion step and whether the robot was EXECUTING
                   a committed action or HOLDING (stalled, waiting on a plan).

From those it computes how much planning is HIDDEN under execution (the overlap that buys
back latency) and renders a multi-panel timeline + a metrics table. bf16 loader assumed
(apply_dtype_patch) so plan latency matches deployment.

Env: SUITE (libero_object), TASK_ID (3), SEED (1000), THINK (0), NUM_STEPS (4), RT_HZ (20),
MAX_STEPS (80), WALL_CAP (60), RT_REPLAN_AT (-1=auto ~chunk/2), RT_BLEND_STEPS (4),
RT_GRIPPER_HYST (0.4), OUT_DIR (/outputs/rt_smoothness), ABL (/work/rt_smoothness_ablation.py).
"""
import importlib.util
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

SUITE = os.environ.get("SUITE", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID", "3"))
SEED = int(os.environ.get("SEED", "1000"))
RT_HZ = float(os.environ.get("RT_HZ", "20"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "80"))
WALL_CAP = float(os.environ.get("WALL_CAP", "60"))
RT_REPLAN_AT = int(os.environ.get("RT_REPLAN_AT", "-1"))
RT_BLEND_STEPS = int(os.environ.get("RT_BLEND_STEPS", "4"))
RT_GRIPPER_HYST = float(os.environ.get("RT_GRIPPER_HYST", "0.4"))
OUT_DIR = os.environ.get("OUT_DIR", "/outputs/rt_smoothness")
ABL = os.environ.get("ABL", "/work/rt_smoothness_ablation.py")
DT = 1.0 / RT_HZ


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abl = _load(ABL, "rt_ablation")
ActivePlan, splice = abl.ActivePlan, abl.splice


def run(env, instruction, policy, plan_chunk, make_proc, blend):
    """Threaded RT loop instrumented with plan-window + per-tick event logs."""
    policy.reset()
    obs0, _ = env.reset(seed=SEED)
    import lerobot.scripts.lerobot_eval as ev
    generator = ev._make_rollout_action_generator(policy, [SEED])
    lock = threading.Lock()
    t0 = time.perf_counter()
    sh = {"obs": obs0, "done": False, "stop": False, "last_grip": 0.0, "t0": t0,
          "motion_steps": 0, "hold_steps": 0, "plan": None, "chunk_len": 0,
          "plans": [], "ticks": []}

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
                trig = sh["motion_steps"]
            base_motion = trig
            ts = time.perf_counter() - sh["t0"]
            try:
                proc = make_proc(cur, instruction, 1)
                chunk, infer_s = plan_chunk(proc, generator)
            except Exception as e:  # noqa: BLE001
                print("plan error:", e, flush=True)
                time.sleep(0.02)
                continue
            te = time.perf_counter() - sh["t0"]
            with lock:
                sh["chunk_len"] = len(chunk)
                cur_motion = sh["motion_steps"]
                old = sh["plan"]
                if blend and old is not None and old.remaining() > 0:
                    sh["plan"] = splice(old, chunk, cur_motion - base_motion, RT_BLEND_STEPS, RT_GRIPPER_HYST)
                else:
                    drop = (cur_motion - base_motion) if blend else 0
                    d = int(max(0, min(drop, len(chunk) - 1)))
                    sh["plan"] = ActivePlan(chunk[d:], cur_motion)
                sh["plans"].append(dict(t_start=ts, t_end=te, infer_ms=infer_s * 1000.0,
                                        trig_step=trig, attach_step=cur_motion,
                                        chunk_len=len(chunk)))

    pth = threading.Thread(target=planner, daemon=True)
    pth.start()

    success = False
    next_t = time.perf_counter()
    while (not sh["done"] and sh["motion_steps"] < MAX_STEPS
           and (time.perf_counter() - t0) < WALL_CAP):
        with lock:
            plan = sh["plan"]
            action = plan.next_action() if plan is not None else None
            if action is not None:
                sh["motion_steps"] += 1
        holding = action is None
        if holding:
            action = np.zeros((1, 7), dtype=np.float32)
            action[:, 6] = sh["last_grip"]
            sh["hold_steps"] += 1
        else:
            sh["last_grip"] = float(action[0, 6])
        with lock:
            sh["ticks"].append((time.perf_counter() - t0, sh["motion_steps"], int(holding)))
        obs, done, success = abl._step_success(env, action, success, sh["obs"])
        with lock:
            sh["obs"] = obs
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
    wall = time.perf_counter() - t0
    return dict(blend=blend, wall=wall, motion_steps=sh["motion_steps"],
                hold_steps=sh["hold_steps"], success=int(bool(success)),
                plans=sh["plans"], ticks=sh["ticks"])


def segments(ticks, holding_val):
    """Collapse the per-tick stream into (start, width) spans of a given hold/exec state."""
    out = []
    i, n = 0, len(ticks)
    while i < n:
        if ticks[i][2] != holding_val:
            i += 1
            continue
        j = i
        while j + 1 < n and ticks[j + 1][2] == holding_val:
            j += 1
        start = ticks[i][0]
        end = ticks[j][0] + DT
        out.append((start, end - start))
        i = j + 1
    return out


def overlap_time(windows, spans):
    """Total time the plan windows overlap the given spans (exec or hold)."""
    tot = 0.0
    for a, wlen in [(w["t_start"], w["t_end"] - w["t_start"]) for w in windows]:
        b = a + wlen
        for c, slen in spans:
            d = c + slen
            tot += max(0.0, min(b, d) - max(a, c))
    return tot


def metrics(res):
    ticks = res["ticks"]
    plans = res["plans"]
    exec_spans = segments(ticks, 0)
    hold_spans = segments(ticks, 1)
    plan_total = sum(p["t_end"] - p["t_start"] for p in plans)
    hidden = overlap_time(plans, exec_spans)         # planning concurrent with motion
    stalled = overlap_time(plans, hold_spans)        # planning concurrent with a stall
    total_ticks = res["motion_steps"] + res["hold_steps"]
    infers = [p["infer_ms"] for p in plans]
    return dict(
        n_plans=len(plans),
        mean_infer_ms=round(float(np.mean(infers)) if infers else 0.0, 1),
        mean_plan_wall_ms=round(1000.0 * plan_total / max(1, len(plans)), 1),
        plan_total_s=round(plan_total, 2),
        hidden_s=round(hidden, 2),
        stalled_s=round(stalled, 2),
        overlap_pct=round(100.0 * hidden / max(1e-6, plan_total), 1),
        hold_pct=round(100.0 * res["hold_steps"] / max(1, total_ticks), 1),
        wall_s=round(res["wall"], 2),
        motion_steps=res["motion_steps"],
        throughput_sps=round(res["motion_steps"] / max(1e-6, res["wall"]), 2),
        success=res["success"],
        exec_spans=exec_spans, hold_spans=hold_spans,
    )


def plot(rh, rb, mh, mb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    C_PLAN, C_EXEC, C_HOLD, C_HID = "#d9534f", "#5cb85c", "#bbbbbb", "#f0ad4e"
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.1, 0.9], hspace=0.55)
    ax_h, ax_b, ax_z, ax_t = (fig.add_subplot(gs[i]) for i in range(4))

    def draw_timeline(ax, res, m, title, note):
        ax.broken_barh([(p["t_start"], p["t_end"] - p["t_start"]) for p in res["plans"]],
                       (2.6, 0.8), facecolors=C_PLAN, edgecolor="white", lw=0.5)
        ax.broken_barh(m["exec_spans"], (1.4, 0.8), facecolors=C_EXEC, edgecolor="none")
        ax.broken_barh(m["hold_spans"], (1.4, 0.8), facecolors=C_HOLD, edgecolor="none")
        ax.set_yticks([3.0, 1.8])
        ax.set_yticklabels(["planner\n(thinking)", "robot\nmotion"])
        ax.set_ylim(0.9, 3.8)
        xmax = max(res["wall"], res["plans"][-1]["t_end"] if res["plans"] else res["wall"])
        ax.set_xlim(0, xmax)
        ax.set_xlabel("wall-clock time (s)")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.text(0.995, 1.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#333")

    draw_timeline(ax_h, rh, mh,
                  "async-HOLD  (no intermediate inference — stop-and-decide)",
                  f"hold {mh['hold_pct']:.0f}%   thru {mh['throughput_sps']:.1f} steps/s   "
                  f"plan hidden under motion: {mh['overlap_pct']:.0f}%")
    draw_timeline(ax_b, rb, mb,
                  "async-BLEND  (intermediate inference — plan the next chunk while executing)",
                  f"hold {mb['hold_pct']:.0f}%   thru {mb['throughput_sps']:.1f} steps/s   "
                  f"plan hidden under motion: {mb['overlap_pct']:.0f}%")
    # shade the hidden-planning overlap on the blend panel
    for p in rb["plans"]:
        for c, slen in mb["exec_spans"]:
            a, b = max(p["t_start"], c), min(p["t_end"], c + slen)
            if b > a:
                ax_b.axvspan(a, b, ymin=0.02, ymax=0.98, color=C_HID, alpha=0.18, lw=0)

    # ---- zoom on the first ~3 blend replan cycles, annotated ----
    plans = rb["plans"]
    zmax = plans[min(3, len(plans) - 1)]["t_end"] + 0.4 if len(plans) > 1 else rb["wall"]
    ax_z.broken_barh([(p["t_start"], p["t_end"] - p["t_start"]) for p in plans],
                     (2.6, 0.8), facecolors=C_PLAN, edgecolor="white", lw=0.5)
    ax_z.broken_barh(mb["exec_spans"], (1.4, 0.8), facecolors=C_EXEC, edgecolor="none")
    ax_z.broken_barh(mb["hold_spans"], (1.4, 0.8), facecolors=C_HOLD, edgecolor="none")
    for k, p in enumerate(plans):
        if p["t_start"] > zmax:
            break
        mid = 0.5 * (p["t_start"] + p["t_end"])
        ax_z.annotate(f"plan {k}\n{p['infer_ms']:.0f} ms", (mid, 3.45), ha="center",
                      va="bottom", fontsize=8, color=C_PLAN)
        ax_z.annotate("", xy=(p["t_start"], 2.55), xytext=(p["t_start"], 1.4),
                      arrowprops=dict(arrowstyle="->", color="#555", lw=1))
        if k > 0:
            ax_z.text(p["t_start"], 1.2, f"next plan starts\n@motion step {p['trig_step']}",
                      ha="center", va="top", fontsize=7.5, color="#555")
    ax_z.set_yticks([3.0, 1.8])
    ax_z.set_yticklabels(["planner\n(thinking)", "robot\nmotion"])
    ax_z.set_ylim(0.6, 3.9)
    ax_z.set_xlim(0, zmax)
    ax_z.set_xlabel("wall-clock time (s)")
    ax_z.set_title("BLEND zoom: each new plan is launched mid-chunk, so its latency overlaps motion",
                   fontsize=11, fontweight="bold", loc="left")
    for c, slen in mb["exec_spans"]:
        for p in plans:
            a, b = max(p["t_start"], c), min(p["t_end"], c + slen)
            if b > a and a < zmax:
                ax_z.axvspan(a, b, color=C_HID, alpha=0.22, lw=0)

    fig.legend(handles=[Patch(color=C_PLAN, label="planner thinking (inference)"),
                        Patch(color=C_EXEC, label="robot executing committed action"),
                        Patch(color=C_HOLD, label="robot holding / stalled"),
                        Patch(color=C_HID, label="planning hidden under motion (latency saved)")],
               loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.995))

    # ---- table ----
    ax_t.axis("off")
    rows = [
        ("plans generated", mh["n_plans"], mb["n_plans"]),
        ("mean model inference (ms)", mh["mean_infer_ms"], mb["mean_infer_ms"]),
        ("mean plan wall window (ms)", mh["mean_plan_wall_ms"], mb["mean_plan_wall_ms"]),
        ("total planning time (s)", mh["plan_total_s"], mb["plan_total_s"]),
        ("planning hidden under motion (s)", mh["hidden_s"], mb["hidden_s"]),
        ("planning overlap with motion (%)", f"{mh['overlap_pct']:.0f}", f"{mb['overlap_pct']:.0f}"),
        ("planning spent stalled / holding (s)", mh["stalled_s"], mb["stalled_s"]),
        ("robot hold / stall (%)", f"{mh['hold_pct']:.0f}", f"{mb['hold_pct']:.0f}"),
        ("throughput (motion steps/s)", mh["throughput_sps"], mb["throughput_sps"]),
        ("motion steps in window", mh["motion_steps"], mb["motion_steps"]),
        ("wall time (s)", mh["wall_s"], mb["wall_s"]),
        ("task success", mh["success"], mb["success"]),
    ]
    tbl = ax_t.table(cellText=[[r[0], r[1], r[2]] for r in rows],
                     colLabels=["metric", "async-HOLD", "async-BLEND"],
                     colWidths=[0.5, 0.25, 0.25], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
        if r > 0 and c == 2:
            cell.set_facecolor("#eafaf1")

    fig.suptitle(f"RT chunk-stitching latency overlap — {SUITE} task {TASK_ID}, "
                 f"{RT_HZ:.0f} Hz, bf16", fontsize=13, fontweight="bold", y=0.965)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", path, flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policy, build_env, plan_chunk, make_proc, torch = abl.build_engine()
    env = build_env(TASK_ID)
    instruction = getattr(env.envs[0], "task_description", "") or "complete the task"
    import lerobot.scripts.lerobot_eval as ev
    wobs, _ = env.reset(seed=SEED)
    wgen = ev._make_rollout_action_generator(policy, [SEED])
    with torch.inference_mode():
        policy.select_action(make_proc(wobs, instruction, 1), generator=wgen)
    policy.reset()
    print("[warmup] done", flush=True)

    rh = run(env, instruction, policy, plan_chunk, make_proc, blend=False)
    rb = run(env, instruction, policy, plan_chunk, make_proc, blend=True)
    mh, mb = metrics(rh), metrics(rb)
    print("HOLD :", {k: mh[k] for k in mh if k not in ("exec_spans", "hold_spans")}, flush=True)
    print("BLEND:", {k: mb[k] for k in mb if k not in ("exec_spans", "hold_spans")}, flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(OUT_DIR, f"latency_timeline_{SUITE}_t{TASK_ID}_{ts}.png")
    plot(rh, rb, mh, mb, png)
    drop = ("exec_spans", "hold_spans")
    with open(os.path.join(OUT_DIR, f"latency_timeline_{SUITE}_t{TASK_ID}_{ts}.json"), "w") as f:
        json.dump({"config": dict(suite=SUITE, task_id=TASK_ID, rt_hz=RT_HZ,
                                  replan_at=RT_REPLAN_AT, blend_steps=RT_BLEND_STEPS),
                   "hold": {k: mh[k] for k in mh if k not in drop},
                   "blend": {k: mb[k] for k in mb if k not in drop},
                   "hold_plans": rh["plans"], "blend_plans": rb["plans"]}, f, indent=2)


if __name__ == "__main__":
    main()
