# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""GR-1 closed-loop CALVIN evaluation on Strix Halo (gfx1151), headless (PyBullet + EGL).

Reuses upstream evaluation/calvin_evaluation.py (GR1CalvinEvaluation) + evaluation/
calvin_env_wrapper_raw.py (CalvinEnvWrapperRaw) and CALVIN's task oracle. For each language-
annotated task in the CALVIN *debug* validation split, we reset the simulator to the real episode
start state, then let GR-1 drive the arm closed-loop (its own predicted actions fed back to the
sim) while the task oracle checks completion. Renders a rollout GIF per task and reports the
success rate. This is the full-pipeline validation (rule 2) on ROCm before optimization (phase 4).
"""
import argparse, json, os, glob
import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf
import hydra

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
from gr1_optim import apply_optimizations  # phase 4: env-toggled bf16/SDPA/compile (default off)

OBS_SPACE = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [],
             "state_obs": ["robot_obs"], "actions": ["rel_actions"], "language": ["language"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    ap.add_argument("--mae-ckpt", default=os.environ.get("MAE_CKPT", "/models/mae_pretrain_vit_base.pth"))
    ap.add_argument("--policy-ckpt", default=os.environ.get("POLICY_CKPT", "/models/snapshot_ABCD.pt"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS", "/repos/gr1/logs/configs.json"))
    ap.add_argument("--calvin-root", default=os.environ.get("CALVIN_ROOT", "/repos/calvin"))
    ap.add_argument("--data-dir", default=os.environ.get("DATASET_DIR", "/data/calvin_debug_dataset"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("EP_LEN") or "180"))
    ap.add_argument("--num-tasks", type=int, default=int(os.environ.get("NUM_TASKS") or "0"), help="0 = all annotations")
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/outputs"), "calvin"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED") or "0"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda", 0)

    with open(args.configs) as f:
        variant = json.load(f)

    val_dir = os.path.join(args.data_dir, args.split)
    model = GR1CalvinEvaluation(args.mae_ckpt, args.policy_ckpt, variant, device)
    apply_optimizations(model)  # no-op unless GR1_AMP/GR1_SDPA/GR1_COMPILE are set
    env = CalvinEnvWrapperRaw(val_dir, OBS_SPACE, device)

    conf_dir = os.path.join(args.calvin_root, "calvin_models", "conf")
    task_cfg = OmegaConf.load(os.path.join(conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)

    ann = np.load(os.path.join(val_dir, "lang_annotations", "auto_lang_ann.npy"),
                  allow_pickle=True).item()
    anns, tasks, idx = ann["language"]["ann"], ann["language"]["task"], ann["info"]["indx"]
    n = len(anns) if args.num_tasks in (0, None) else min(args.num_tasks, len(anns))

    results = []
    for i in range(n):
        task_id, lang = str(tasks[i]), str(anns[i])
        start = int(idx[i][0])
        ep0 = np.load(os.path.join(val_dir, f"episode_{start:07d}.npz"))
        env.reset(robot_obs=ep0["robot_obs"], scene_obs=ep0["scene_obs"])
        start_info = env.get_info()
        model.reset()
        obs = env.get_obs()

        frames, success, at = [], False, -1
        for step in range(args.max_steps):
            action = model.step(obs, lang)
            obs, _, _, info = env.step(action)
            frames.append(np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8))
            done_tasks = task_oracle.get_task_info_for_set(start_info, info, {task_id})
            if task_id in done_tasks:
                success, at = True, step + 1
                break

        tag = "succ" if success else "fail"
        gif = os.path.join(args.out, f"{i:02d}_{task_id}_{tag}.gif")
        imageio.mimsave(gif, frames, duration=1000.0 / args.fps, loop=0)
        print(f"[{i}] {task_id:28s} '{lang}' -> {tag}"
              + (f" @step {at}" if success else f" ({len(frames)} steps)") + f"  gif={os.path.basename(gif)}")
        results.append({"idx": i, "task": task_id, "lang": lang, "success": success,
                        "success_step": at, "steps": len(frames)})

    n_succ = sum(r["success"] for r in results)
    sr = n_succ / max(len(results), 1)
    print(f"\nCALVIN debug closed-loop: {n_succ}/{len(results)} succeeded ({sr*100:.1f}%)")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"success_rate": sr, "n": len(results), "results": results}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
