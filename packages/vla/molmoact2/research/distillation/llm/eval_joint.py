"""Closed-loop LIBERO evaluation of a distilled student (functional distillation).

Self-contained driver that reuses lerobot's tested eval machinery (make_env, processors,
eval_policy_all, metrics) but assembles the policy ourselves: load the teacher policy, swap
in the trained student (transformer.forward path -> _extract_kv_states -> action expert,
validated in Module 1), and merge the action-expert LoRA. This is the FINAL gate -- closed-
loop success, never KV cosine.
"""

from __future__ import annotations
import argparse, json, os, time
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="0", help="comma-separated task ids")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--output-dir", default="/outputs/llm_distill/eval/tmp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--teacher-baseline", action="store_true",
                    help="skip student swap -> eval the teacher (sanity control)")
    return ap.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.configs.types import FeatureType
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy_all
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = 1; device = args.device; num_workers = 0
    ds, ds_meta, pre_ds, dcfg = D.build_dataset_and_preprocessor(A)

    cfg = MolmoAct2Config(
        checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
        action_mode="continuous", model_dtype="bfloat16", device=args.device,
    )
    if hasattr(cfg, "inference_action_mode"):
        cfg.inference_action_mode = "continuous"
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}

    print(f"[eval] building policy (teacher={args.teacher})", flush=True)
    policy = MolmoAct2Policy(cfg).to(device)

    if not args.teacher_baseline:
        ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        JP.assemble_student_for_eval(policy, ck)
        print(f"[eval] swapped student from {args.student_ckpt} "
              f"(step={ck.get('step')}, phase={ck.get('phase', 'legacy')})", flush=True)
    policy.eval()

    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)

    task_ids = [int(x) for x in str(args.task_ids).split(",") if x != ""]
    env_cfg = LiberoEnv(
        task=args.suite, task_ids=task_ids,
        camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
    )
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)

    print(f"[eval] rollout suite={args.suite} tasks={task_ids} n_ep={args.n_episodes}", flush=True)
    t0 = time.time()
    res = eval_policy_all(
        envs, policy, env_pre, env_post, pre, post,
        n_episodes=args.n_episodes, start_seed=args.seed,
    )
    dt = time.time() - t0

    agg = res.get("aggregated", res)
    out = {"suite": args.suite, "task_ids": task_ids, "n_episodes": args.n_episodes,
           "student_ckpt": os.path.basename(args.student_ckpt) if not args.teacher_baseline else "TEACHER",
           "aggregated": agg, "eval_s": round(dt, 1)}
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    pc = "?"
    if isinstance(agg, dict):
        if isinstance(agg.get("overall"), dict):
            pc = agg["overall"].get("pc_success", "?")
        elif "pc_success" in agg:
            pc = agg["pc_success"]
    print(f"[eval] DONE pc_success={pc} eval_s={dt:.1f} -> {args.output_dir}/eval_result.json", flush=True)
    print("[eval] AGG " + json.dumps(agg, default=str)[:500], flush=True)


if __name__ == "__main__":
    main()
