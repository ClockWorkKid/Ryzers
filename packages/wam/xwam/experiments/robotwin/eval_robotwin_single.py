# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Single-task X-WAM RoboTwin closed-loop launcher (Ryzer).

Mirrors FastWAM's eval_robotwin_single.py symlink pattern without Hydra: symlink
`<ROBOTWIN_ROOT>/policy/xwam_policy -> experiments/robotwin/xwam_policy`, then run RoboTwin's
model-agnostic `script/eval_policy.py` (which runs the SAPIEN render smoke, drives get_model/
eval/reset_model, and writes per-episode videos + a `_result_*.txt`). All knobs come from env.

Env:
  TASK (required)          RoboTwin task name, e.g. beat_block_hammer
  TASK_CONFIG=demo_clean   demo_clean | demo_randomized
  NUM_EPISODES=1           episodes to evaluate
  SEED=0                   base seed
  GPU_ID=0                 CUDA_VISIBLE_DEVICES
  ROBOTWIN_ROOT=/opt/RoboTwin
  EVAL_OUTPUT_DIR=/outputs/robotwin/<task>
  CKPT_ROOT, EXP, WAN_CKPT_DIR                (X-WAM weights; forwarded to get_model)
  DENOISE_STEPS=50, ACTION_DENOISE_STEPS=10, ACTION_LENGTH=32, REPLAN_STEPS=<action_length>, CFG=0.0
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_NAME = "xwam_policy"
POLICY_SRC = HERE / POLICY_NAME


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _ensure_symlink(robotwin_root: Path):
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy dir not found: {policy_root}")
    target = policy_root / POLICY_NAME
    src = POLICY_SRC.resolve()
    if target.is_symlink():
        if target.resolve() != src:
            target.unlink()
            target.symlink_to(src, target_is_directory=True)
    elif target.exists():
        raise RuntimeError(f"{target} exists and is not a symlink; remove it manually.")
    else:
        target.symlink_to(src, target_is_directory=True)
    return target


def main() -> int:
    task = _env("TASK") or _env("TASKS")
    if not task:
        print("FAIL: set TASK=<robotwin_task_name>.", file=sys.stderr)
        return 2
    task = task.split()[0]  # single-task launcher

    robotwin_root = Path(_env("ROBOTWIN_ROOT", "/opt/RoboTwin"))
    if not robotwin_root.exists():
        print(f"FAIL: RoboTwin base not found at {robotwin_root} (build: ryzers build robotwin xwam).",
              file=sys.stderr)
        return 2

    task_config = _env("TASK_CONFIG", "demo_clean")
    num_episodes = int(_env("NUM_EPISODES", "1"))
    seed = int(_env("SEED", "0"))
    gpu_id = _env("GPU_ID", "0")

    ckpt_root = _env("CKPT_ROOT", "/models/xwam/checkpoints")
    exp = _env("EXP", "robotwin_sft")
    exp_path = _env("EXP_PATH", str(Path(ckpt_root) / exp))
    wan_ckpt = _env("WAN_CKPT_DIR", "/models/xwam/wan22_5b")
    action_length = int(_env("ACTION_LENGTH", "32"))
    replan_steps = int(_env("REPLAN_STEPS", str(action_length)))

    out_dir = _env("EVAL_OUTPUT_DIR", f"/outputs/robotwin/{task}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    _ensure_symlink(robotwin_root)

    def ov(k, v):
        return [f"--{k}", str(v)]

    overrides = []
    overrides += ov("task_name", task)
    overrides += ov("task_config", task_config)
    overrides += ov("policy_name", POLICY_NAME)
    overrides += ov("seed", seed)
    overrides += ov("instruction_type", _env("INSTRUCTION_TYPE", "unseen"))
    overrides += ov("eval_num_episodes", num_episodes)
    overrides += ov("eval_output_dir", out_dir)
    # X-WAM knobs -> usr_args -> get_model (repr() so eval_policy keeps paths as strings)
    overrides += ov("exp_path", repr(str(exp_path)))
    overrides += ov("wan_checkpoint_dir", repr(str(wan_ckpt)))
    overrides += ov("steps", repr(_env("STEPS", "last")))
    overrides += ov("denoise_steps", int(_env("DENOISE_STEPS", "50")))
    overrides += ov("action_denoise_steps", int(_env("ACTION_DENOISE_STEPS", "10")))
    overrides += ov("action_length", action_length)
    overrides += ov("replan_steps", replan_steps)
    overrides += ov("cfg", float(_env("CFG", "0.0")))

    cmd = [sys.executable, "-u", "script/eval_policy.py",
           "--config", f"policy/{POLICY_NAME}/deploy_policy.yml", "--overrides", *overrides]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    xwam_repo = _env("XWAM_REPO", "/repos/xwam")
    env["PYTHONPATH"] = os.pathsep.join(
        [xwam_repo, str(robotwin_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    print(f"[xwam-robotwin] task={task} config={task_config} episodes={num_episodes} "
          f"exp_path={exp_path} action_length={action_length} replan={replan_steps}")
    proc = subprocess.run(cmd, cwd=str(robotwin_root), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
