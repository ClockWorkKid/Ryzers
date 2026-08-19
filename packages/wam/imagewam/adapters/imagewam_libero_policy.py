# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ImageWAM LIBERO policy adapter for the simulation/libero interactive harness.

Implements the model-agnostic `sim_libero.Policy` seam by wrapping the FLUX.2 ImageWAM
world-action model. Reuses the *validated* eval machinery from
experiments/libero/eval_libero_single.py verbatim (config compose, model instantiate +
checkpoint load, processor/normalizer, and `_predict_action_chunk`) so interactive rollouts
match the closed-loop numbers.

Selected at runtime by the sim harness via
  POLICY_FACTORY=imagewam_libero_policy:build_policy

Env: CKPT, DATASET_STATS, MIXED_PRECISION (bf16), REPLAN_STEPS, NUM_STEPS_WAIT,
NUM_INFERENCE_STEPS, ACTION_HORIZON, FLUX2_VARIANT (4b), FLUX2_SRC, FLUX2_MODEL_PATH,
FLUX2_AE_MODEL_PATH, FLUX2_QWEN3_MODEL_SPEC, IMAGEWAM_REPO (/repos/imagewam).
Requires /repos/imagewam(+/src), /repos/imagewam/experiments/libero, the flux2 src, /opt/sim
and /opt/LIBERO on PYTHONPATH (the demo sets this).
"""
import os

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

import experiments.libero.eval_libero_single as E
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from sim_libero.policy import Policy

IMAGEWAM_REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
CONFIG_DIR = os.path.join(IMAGEWAM_REPO, "configs")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
DEFAULT_CKPT = f"/models/imagewam_release/libero/flux2_klein_{VARIANT}/model.pt"
DEFAULT_STATS = f"/models/imagewam_release/libero/flux2_klein_{VARIANT}/dataset_stats.json"


class ImageWAMLiberoPolicy(Policy):
    name = "imagewam"

    def __init__(self, model, processor, cfg, action_horizon, input_w, input_h, device):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.action_horizon = action_horizon
        self.input_w = input_w
        self.input_h = input_h
        self.device = device
        self.replan_steps = int(cfg.EVALUATION.get("replan_steps", 12))
        self.num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))

    def predict_action_chunk(self, obs, instruction):
        action, _imgs, _pred = E._predict_action_chunk(
            obs=obs,
            task_description=instruction,
            model=self.model,
            processor=self.processor,
            cfg=self.cfg,
            action_horizon=self.action_horizon,
            input_w=self.input_w,
            input_h=self.input_h,
            model_device=self.device,
        )
        return np.asarray(action, dtype=np.float32)


def _env(name, default=None):
    val = os.environ.get(name)
    return val if val else default


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    ckpt = _env("CKPT", DEFAULT_CKPT)
    stats = _env("DATASET_STATS", DEFAULT_STATS)
    mixed = _env("MIXED_PRECISION", "bf16")
    suite = _env("SUITE", "libero_object")
    flux2_src = _env("FLUX2_SRC", "/repos/flux2")
    dit = _env("FLUX2_MODEL_PATH", "/models/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors")
    ae = _env("FLUX2_AE_MODEL_PATH", "/models/flux2/FLUX.2-klein-base-4B/ae.safetensors")
    qwen3 = _env("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")

    overrides = [
        f"task=libero_flux2_klein_{VARIANT}_base_imagewam",
        f"ckpt={ckpt}",
        "gpu_id=0",
        f"mixed_precision={mixed}",
        f"EVALUATION.task_suite_name={suite}",
        "EVALUATION.task_id=0",
        "EVALUATION.num_trials=1",
        f"EVALUATION.dataset_stats_path={stats}",
        "EVALUATION.output_dir=/tmp/imagewam_interactive",
        f"EVALUATION.action_horizon={_env('ACTION_HORIZON', '16')}",
        f"EVALUATION.replan_steps={_env('REPLAN_STEPS', '12')}",
        f"EVALUATION.num_inference_steps={_env('NUM_INFERENCE_STEPS', '20')}",
        f"model.flux2_src_path={flux2_src}",
        f"model.flux2_model_path={dit}",
        f"model.ae_model_path={ae}",
        f"model.variant=klein-base-{VARIANT}",
        f"model.qwen3_model_spec={qwen3}",
        "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true",
        "model.proprio_dim=8",
    ]
    if _env("NUM_STEPS_WAIT"):
        overrides.append(f"EVALUATION.num_steps_wait={os.environ['NUM_STEPS_WAIT']}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="sim_libero_omnigen2", overrides=overrides)

    device = E._resolve_eval_device(cfg)
    dtype = E._mixed_precision_to_model_dtype(mixed)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    E._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()

    stats_path = E._resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    print(f"[imagewam_libero_policy] model ready (ckpt={ckpt}, horizon={action_horizon}, "
          f"input={input_w}x{input_h}, replan={cfg.EVALUATION.get('replan_steps', 12)})", flush=True)
    return ImageWAMLiberoPolicy(model, processor, cfg, action_horizon, input_w, input_h, device)
