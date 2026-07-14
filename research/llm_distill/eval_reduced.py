"""Closed-loop LIBERO eval of the Workstream-B reduced-token student.

Loads the trained ROI gate+FastV policy (the frozen deployable front-end + teacher action
expert), swaps in the width-reduced student, and runs closed-loop rollouts where each action
chunk is generated on the 25% task-relevant token stream (prune-before-student): gate keeps 50%
of ViT patches @ seam6, FastV group-drop keeps 25% of image tokens, the student LLM runs on the
reduced fused sequence, and its per-layer KV drives the frozen action expert's flow-matching ODE.

We bypass the overlay's mid-forward FastV cut (incompatible with a monolithic student forward)
by reducing at the INPUT; the student forward reuses the trainer's exact
``_prepare_joint_training_backbone_inputs`` path (KV byte-identical to training), then the action
chunk is denoised with the overlay's proven native flow-matching loop (from _generate_actions_fastv).
"""

from __future__ import annotations
import argparse, json, os, time, types
import numpy as np
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-ckpt", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--fastv-keep", type=float, default=0.25)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="0")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--output-dir", default="/outputs/llm_distill/eval/reduced")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roi-teacher-baseline", action="store_true",
                    help="skip student swap -> eval the full-width ROI teacher via its own FastV path (ceiling)")
    ap.add_argument("--teacher-through-reduced", action="store_true",
                    help="CONTROL: skip student swap but run the full-width teacher LLM through the "
                         "prune-before-student reduced harness (isolates harness vs. student-width)")
    return ap.parse_args()


def make_reduced_predict(fastv_keep):
    """Return a predict_action_chunk that reduces the sequence via gate+FastV before the student.

    Faithfulness contract: the student KV is produced by the SAME call the trainer uses
    (``_prepare_joint_training_backbone_inputs`` on a reduced ``model_inputs`` -> student), so
    the eval-time KV is byte-identical to training for the same observation. The action chunk is
    then denoised with the overlay's proven native flow-matching loop (a verbatim copy of
    ``_generate_actions_fastv``'s denoise stage), NOT the stock ``generate_actions_from_inputs``.
    """
    def reduced_predict(self, batch, **kwargs):
        from lerobot.policies.molmoact2.modeling_molmoact2 import _mask_action_dim_tensor
        backbone = self._backbone()
        action_expert = backbone._require_action_expert()
        vb = getattr(backbone, "vision_backbone", None)
        model_inputs = self._model_inputs(batch)
        setter = getattr(self, "_set_gate_task_tokens", None)
        if callable(setter):
            setter(model_inputs)
        # Match TRAINING: capture_teacher_reduced never set a causal carry, so the ViT gate
        # scored the CURRENT frame (roi_external_keep_idx=None). Use the same current-frame
        # keep-set at eval so the student sees the token distribution it was trained on.
        if vb is not None:
            vb.roi_external_keep_idx = None
            vb.roi_last_gate_scores = None

        model_dtype = torch.bfloat16
        dev = next(self.parameters()).device
        with torch.autocast(device_type=dev.type, dtype=model_dtype):
            hidden, _causal, position_ids, _cache = self._prepare_joint_training_backbone_inputs(model_inputs)
            B, S, D = hidden.shape
            if not self._fastv_group_ctx_from_gate(model_inputs):
                raise RuntimeError("eval_reduced: gate group-ctx unavailable")
            _new_ppi, keep_over = self._teacher_group_drop(float(fastv_keep))
            col_idx = self._fastv_col_idx_from_keep(model_inputs, keep_over)
            if col_idx is None:
                raise RuntimeError("eval_reduced: ragged keep-set")
            s_new = int(col_idx.shape[1])
            red_embeds = hidden.gather(1, col_idx.unsqueeze(-1).expand(B, s_new, D))
            pos2d = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0)
            if pos2d.shape[0] == 1 and B > 1:
                pos2d = pos2d.expand(B, -1)
            red_pos = pos2d.gather(1, col_idx)
            am = model_inputs.get("attention_mask")
            if am is not None and torch.is_tensor(am) and am.dim() == 2:
                red_mask = am.gather(1, col_idx).to(hidden.dtype)
            else:
                red_mask = torch.ones(B, s_new, device=hidden.device, dtype=hidden.dtype)

            # ---- student forward EXACTLY as the trainer does it (reduced model_inputs) ----
            red_mi = {"inputs_embeds": red_embeds, "attention_mask": red_mask, "position_ids": red_pos}
            r_hidden, r_causal, r_pos, r_cache = self._prepare_joint_training_backbone_inputs(red_mi)
            student = backbone.transformer  # swapped student masquerades as transformer
            sout = student(inputs_embeds=r_hidden, attention_mask=r_causal, position_ids=r_pos,
                           cache_position=r_cache, collect_layer_kv_states=True, use_cache=False)
            kv = [(backbone._cache_to_sequence(k), backbone._cache_to_sequence(v))
                  for k, v in sout.past_key_values]

            # ---- native flow-matching denoise (verbatim from _generate_actions_fastv) ----
            horizon = self._generation_action_horizon()
            max_action_dim = int(backbone.config.max_action_dim)
            traj_dtype = action_expert.action_embed.weight.dtype
            trajectory = torch.randn((B, horizon, max_action_dim), device=dev, dtype=traj_dtype,
                                     generator=kwargs.get("generator"))
            action_dim_is_pad = batch.get("action_dim_is_pad")
            mask_enabled = bool(self.config.mask_action_dim_padding)
            if mask_enabled:
                trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
            context = action_expert.prepare_context(
                encoder_kv_states=kv, encoder_attention_mask=None, state_embeddings=None,
                batch_size=B, seq_len=trajectory.shape[1], device=dev, dtype=trajectory.dtype)
            steps = int(kwargs.get("num_steps") or backbone.config.flow_matching_num_steps)
            if steps <= 0:
                raise ValueError(f"num_steps must be >= 1, got {steps}.")
            flow_timesteps = [torch.full((B,), idx / steps, device=dev, dtype=torch.float32)
                              for idx in range(steps)]
            modulation_cache = action_expert.get_or_prepare_modulation_cache(
                flow_timesteps, cache_key=(steps, B, dev, trajectory.dtype))
            dt = 1.0 / steps
            for idx in range(steps):
                sm = modulation_cache[idx]
                velocity = action_expert.forward_with_context(
                    trajectory, sm.conditioning, context=context, modulation=sm)
                if mask_enabled:
                    velocity = _mask_action_dim_tensor(velocity, action_dim_is_pad)
                trajectory = trajectory + dt * velocity
                if mask_enabled:
                    trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
            traj = trajectory

        action_dim = self._output_action_dim(batch)
        return traj[:, : self.config.n_action_steps, :action_dim].to(dtype=torch.float32)
    return reduced_predict


def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy_all

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = 1; device = args.device; num_workers = 0
    _, ds_meta, _, _ = D.build_dataset_and_preprocessor(A)

    print(f"[evalB] loading ROI policy {args.roi_ckpt}", flush=True)
    policy = MolmoAct2Policy.from_pretrained(args.roi_ckpt).to(device)
    cfg = policy.config

    if args.teacher_through_reduced:
        # CONTROL: full-width teacher LLM (no swap) through the prune-before-student harness.
        policy.predict_action_chunk = types.MethodType(make_reduced_predict(args.fastv_keep), policy)
        print(f"[evalB] CONTROL: full-width teacher through reduced harness fastv_keep={args.fastv_keep}", flush=True)
    elif not args.roi_teacher_baseline:
        ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        keys = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")
        scfg = {k: v for k, v in ck["cfg"].items() if k in keys}
        stu, _ = S.build_student(scfg)
        stu.load_state_dict(ck["student"], strict=True)
        JP.load_student_and_merge_lora(policy, stu, ck)
        policy.predict_action_chunk = types.MethodType(make_reduced_predict(args.fastv_keep), policy)
        print(f"[evalB] swapped reduced student (step={ck.get('step')}) fastv_keep={args.fastv_keep}", flush=True)
    else:
        os.environ["ROI_FASTV_INFER"] = "1"
        print("[evalB] ROI teacher baseline (full-width LLM + native FastV path)", flush=True)
    policy.eval()

    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    task_ids = [int(x) for x in str(args.task_ids).split(",") if x != ""]
    env_cfg = LiberoEnv(task=args.suite, task_ids=task_ids,
                        camera_name_mapping={"agentview_image": "image",
                                             "robot0_eye_in_hand_image": "wrist_image"})
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)

    print(f"[evalB] rollout suite={args.suite} tasks={task_ids} n_ep={args.n_episodes}", flush=True)
    t0 = time.time()
    res = eval_policy_all(envs, policy, env_pre, env_post, pre, post,
                          n_episodes=args.n_episodes, start_seed=args.seed)
    dt = time.time() - t0
    agg = res.get("aggregated", res)
    out = {"suite": args.suite, "task_ids": task_ids, "n_episodes": args.n_episodes,
           "mode": "roi_teacher" if args.roi_teacher_baseline else "reduced_student",
           "student_ckpt": os.path.basename(args.student_ckpt), "fastv_keep": args.fastv_keep,
           "aggregated": agg, "eval_s": round(dt, 1)}
    with open(os.path.join(args.output_dir, "eval_result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    pc = "?"
    if isinstance(agg, dict):
        pc = agg.get("overall", {}).get("pc_success", agg.get("pc_success", "?")) if agg else "?"
    print(f"[evalB] DONE pc_success={pc} eval_s={dt:.1f} -> {args.output_dir}/eval_result.json", flush=True)
    print("[evalB] AGG " + json.dumps(agg, default=str)[:600], flush=True)


if __name__ == "__main__":
    main()
