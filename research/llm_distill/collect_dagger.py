"""DAgger step 1 -- on-policy state collection.

Roll out the STUDENT closed-loop in LIBERO and snapshot the *model-ready batch* at every
action-chunk replan (i.e. the student-visited states where the policy must decide). The teacher
relabels these offline (``relabel_teacher.py``) to produce (student-state -> teacher-action)
training pairs that attack the closed-loop covariate-shift gap.

Reuses ``diag_closed_loop``'s policy/env builders and the unified student swap so the collected
states are byte-identical to what the eval harness produces. The snapshot is taken *before*
``select_action`` on the steps where the action queue is empty (a new chunk is generated), which
is exactly the input ``predict_action_chunk`` consumes -- model-agnostic, so the teacher can be
run on it verbatim during relabeling.
"""

from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

import diag_closed_loop as CL


def snapshot_batch(b):
    """Deep-copy the model-ready batch to CPU (keep dtypes; task stays a list of strings)."""
    out = {}
    for k, v in b.items():
        out[k] = v.detach().to("cpu").clone() if torch.is_tensor(v) else v
    return out


def make_task_env(args, cfg, ds_meta, suite, task_id):
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    env_cfg = LiberoEnv(task=suite, task_ids=[int(task_id)],
                        camera_name_mapping={"agentview_image": "image",
                                             "robot0_eye_in_hand_image": "wrist_image"})
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    vec = next(iter(next(iter(envs.values())).values()))
    return vec, env_pre, env_post, pre, post


def collect_episode(policy, env, env_pre, env_post, pre, post, preprocess_observation,
                    seed, max_steps_cap, shard):
    from lerobot.policies.molmoact2.modeling_molmoact2 import ACTION
    policy.reset()
    obs, info = env.reset(seed=[seed])
    img_key = None
    done = np.array([False])
    max_steps = env.call("_max_episode_steps")[0]
    if max_steps_cap:
        max_steps = min(max_steps, max_steps_cap)
    step, success, nsnap = 0, False, 0
    while not bool(done.all()) and step < max_steps:
        pobs = preprocess_observation(obs)
        if img_key is None:
            img_key = CL._find_image_key(pobs)
        try:
            pobs["task"] = list(env.call("task_description"))
        except Exception:
            pobs["task"] = [""]
        pobs = env_pre(pobs)
        pobs = pre(pobs)
        # snapshot only on replan steps (queue empty -> predict_action_chunk will run)
        if len(policy._action_queue) == 0:
            shard.append(snapshot_batch(pobs))
            nsnap += 1
        with torch.inference_mode():
            action = policy.select_action(pobs)
        action = post(action)
        tr = env_post({ACTION: action})
        anp = tr[ACTION].to("cpu").numpy()
        obs, reward, terminated, truncated, info = env.step(anp)
        if "final_info" in info and isinstance(info["final_info"], dict):
            success = success or bool(info["final_info"]["is_success"][0])
        elif "is_success" in info:
            iv = info["is_success"]
            success = success or bool(iv[0] if hasattr(iv, "__len__") else iv)
        done = terminated | truncated | done
        step += 1
    return {"steps": step, "success": bool(success), "n_snap": nsnap}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--suites", default="libero_spatial",
                    help="comma list of LIBERO suites to roll out")
    ap.add_argument("--task-ids", default="0", help="comma list of task ids per suite")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/rollout (0=env default)")
    ap.add_argument("--out", default="/outputs/llm_distill/dagger/collect")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = get_args()
    from lerobot.envs.utils import preprocess_observation
    os.makedirs(args.out, exist_ok=True)
    cap = args.max_steps or None
    suites = [s for s in args.suites.split(",") if s]
    task_ids = [int(x) for x in str(args.task_ids).split(",") if x != ""]

    print(f"[collect] building policy + swapping student {os.path.basename(args.student_ckpt)}", flush=True)
    policy, cfg, ds_meta = CL.build_policy(args)
    step = CL.swap_in_student(policy, args)
    policy.eval()
    print(f"[collect] student step={step}; suites={suites} tasks={task_ids} n_ep={args.n_episodes}", flush=True)

    manifest = {"student_ckpt": os.path.basename(args.student_ckpt), "student_step": step,
                "suites": suites, "task_ids": task_ids, "n_episodes": args.n_episodes,
                "seed": args.seed, "shards": [], "total_states": 0}
    t_all = time.time()
    for suite in suites:
        for tid in task_ids:
            env, env_pre, env_post, pre, post = make_task_env(args, cfg, ds_meta, suite, tid)
            for ep in range(args.n_episodes):
                seed = args.seed + ep
                shard = []
                t0 = time.time()
                r = collect_episode(policy, env, env_pre, env_post, pre, post,
                                    preprocess_observation, seed, cap, shard)
                name = f"{suite}_t{tid}_ep{ep}_s{seed}.pt"
                path = os.path.join(args.out, name)
                torch.save({"batches": shard, "meta": {"suite": suite, "task_id": tid, "ep": ep,
                            "seed": seed, **r}}, path)
                manifest["shards"].append({"file": name, "suite": suite, "task_id": tid,
                                           "ep": ep, "seed": seed, **r})
                manifest["total_states"] += r["n_snap"]
                print(f"[collect] {name}: steps={r['steps']} success={r['success']} "
                      f"states={r['n_snap']} ({time.time()-t0:.0f}s)", flush=True)
    manifest["elapsed_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[collect] DONE total_states={manifest['total_states']} "
          f"shards={len(manifest['shards'])} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
