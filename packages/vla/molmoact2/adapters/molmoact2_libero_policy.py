# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""MolmoAct2 LIBERO policy adapter for the shared simulation/libero harness.

Implements the model-agnostic `sim_libero.Policy` seam by wrapping MolmoAct2's *lerobot*
policy (the same `make_policy` / `make_pre_post_processors` path the validated
`lerobot-eval --policy.type=molmoact2 --env.type=libero` run and the bundled interactive
server use), but driving it from the shared harness's env instead of lerobot's own env.

Selected at runtime by the sim harness via
  POLICY_FACTORY=molmoact2_libero_policy:build_policy

Env knobs: CKPT (default allenai/MolmoAct2-Think-LIBERO), THINK (1), NUM_STEPS
(flow-matching denoise steps; blank=model default), SUITE, REPLAN_STEPS, NUM_STEPS_WAIT.

STATUS: skeleton on the spin-off branch `benchmark-molmoact2-libero`. The plumbing that
loads the lerobot policy is the proven path; the observation mapping + action extraction
(marked `# VALIDATE-ON-BOX`) must be confirmed against the actual lerobot MolmoAct2 policy
on strix-halo before this is merged into `benchmark`. See docs/LIBERO_SIMBASE_ADAPTATION.md.
"""
import os

import numpy as np

from sim_libero.policy import Policy

DEFAULT_CKPT = "allenai/MolmoAct2-Think-LIBERO"


class MolmoAct2LiberoPolicy(Policy):
    name = "molmoact2"

    def __init__(self, policy, preprocessor, postprocessor, action_dim=7):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_dim = action_dim
        # MolmoAct2 runs a receding-horizon action queue; replan cadence mirrors fastwam.
        self.replan_steps = int(os.environ.get("REPLAN_STEPS") or 5)
        self.num_steps_wait = int(os.environ.get("NUM_STEPS_WAIT") or 5)

    def reset(self, instruction):
        # MolmoAct2-Think caches its depth/spatial plan once per episode; clear it so a new
        # instruction takes effect (mirror the interactive server's per-command reset).
        try:
            self.policy.reset()
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def _obs_to_batch(self, obs, instruction):
        """Map the raw robosuite obs dict -> the lerobot observation batch the MolmoAct2
        policy preprocessor expects.

        VALIDATE-ON-BOX: confirm the exact keys/shapes/orientation the MolmoAct2 lerobot
        policy consumes. Best current understanding (LIBERO lerobot convention):
          observation.images.image        <- agentview_image
          observation.images.wrist_image  <- robot0_eye_in_hand_image
          observation.state               <- [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
          task                            <- instruction (string)
        `sim_libero.get_libero_image` rotates images 180 deg to match FastWAM training;
        MolmoAct2's lerobot LIBERO env may use a different orientation -- verify and match
        it here (use the raw obs images with the correct flip).
        """
        import torch

        def _img(key):
            arr = np.ascontiguousarray(obs[key])  # HWC uint8; orientation VALIDATE-ON-BOX
            return torch.from_numpy(arr)

        eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
        # axis-angle preferred; fall back to quat if the axis-angle key is absent.
        eef_rot = np.asarray(
            obs.get("robot0_eef_axis_angle", obs.get("robot0_eef_quat", np.zeros(3)))[:3],
            dtype=np.float32,
        )
        grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
        state = torch.from_numpy(np.concatenate([eef_pos, eef_rot, grip]).astype(np.float32))

        return {
            "observation.images.image": _img("agentview_image"),
            "observation.images.wrist_image": _img("robot0_eye_in_hand_image"),
            "observation.state": state,
            "task": [instruction],
        }

    def predict_action_chunk(self, obs, instruction):
        import torch

        batch = self._obs_to_batch(obs, instruction)
        with torch.no_grad():
            batch = self.preprocessor(batch)
            # VALIDATE-ON-BOX: lerobot policies expose either predict_action_chunk(batch)
            # -> [B, T, action_dim] or select_action(batch) -> [B, action_dim] (drained
            # from an internal queue). Prefer the chunk API; fall back to single-step.
            if hasattr(self.policy, "predict_action_chunk"):
                chunk = self.policy.predict_action_chunk(batch)
            else:
                chunk = self.policy.select_action(batch)
            out = self.postprocessor({"action": chunk})
            action = out["action"] if isinstance(out, dict) else out

        action = np.asarray(action.detach().to("cpu").float().numpy())
        action = np.atleast_2d(action.reshape(-1, self.action_dim))  # -> [T, 7]
        return action.astype(np.float32)


def build_policy():
    import draccus
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    ckpt = os.environ.get("CKPT") or DEFAULT_CKPT
    think = (os.environ.get("THINK", "1") == "1")
    depth = "True" if think else "False"
    suite = os.environ.get("SUITE") or "libero_object"

    # Parse the eval config to obtain policy + env metadata WITHOUT building lerobot's env
    # (the shared sim_libero harness owns the env). --env.type=libero is only used for the
    # action/state feature metadata make_policy needs.
    args = [
        "--policy.type=molmoact2", f"--policy.checkpoint_path={ckpt}",
        "--policy.inference_action_mode=continuous",
        f"--policy.enable_depth_reasoning={depth}", f"--policy.enable_adaptive_depth={depth}",
        "--policy.enable_cuda_graph=False", "--policy.norm_tag=libero",
        "--policy.device=cuda", "--env.type=libero",
        f"--env.task={suite}", "--env.task_ids=[0]",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        "--output_dir=/tmp/molmoact2_simbase",
    ]
    if os.environ.get("NUM_STEPS"):
        args.append(f"--policy.num_steps={os.environ['NUM_STEPS']}")

    cfg = draccus.parse(EvalPipelineConfig, args=args)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    print(f"[molmoact2_libero_policy] policy ready (ckpt={ckpt}, think={think}, "
          f"suite={suite})", flush=True)
    return MolmoAct2LiberoPolicy(policy, preprocessor, postprocessor)
