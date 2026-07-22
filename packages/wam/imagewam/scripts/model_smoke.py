# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full-model smoke test for FLUX.2 ImageWAM on AMD Strix Halo (gfx1151).

Direct PyTorch port: builds the *real* FLUX.2 [klein] ImageWAM (Qwen3 text encoder +
FLUX.2 autoencoder + image-editing DiT + ActionDiT) via the upstream hydra config, loads
the released LIBERO checkpoint, and runs ONE `infer_action` on a synthetic observation to
prove the whole flow-matching action path executes end-to-end on ROCm. With
VISUALIZE_DREAM=1 it also runs `infer_joint` to produce the "dreamed" edited future frame
(ImageWAM's single-image dream, not a video clip).

Exercises the full model as a single unit (submodule-level checks live in
scripts/module_smoke.py per workspace rule 2). No simulator/dataset required: the
observation is synthetic; we only assert the predicted action chunk has the right shape and
is finite. Env knobs:
  IMAGEWAM_REPO   = /repos/imagewam
  CONFIG_NAME     = sim_libero_omnigen2      (shared FLUX.2/OmniGen2 eval config)
  FLUX2_VARIANT   = 4b
  FLUX2_SRC / FLUX2_MODEL_PATH / FLUX2_AE_MODEL_PATH / FLUX2_QWEN3_MODEL_SPEC
  CKPT_PATH       = /models/imagewam_release/libero/flux2_klein_4b/model.pt
  DATASET_STATS_PATH = .../dataset_stats.json
  NUM_STEPS       = 20                        (flow-matching denoise steps)
Reports first-call (incl. one-time ROCm warmup) and steady-state latency. Exits non-zero on
any failure so `ryzers run` / CI catches a broken image.
"""
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero_omnigen2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
CKPT = os.environ.get("CKPT_PATH", "/models/imagewam_release/libero/flux2_klein_4b/model.pt")
STATS = os.environ.get("DATASET_STATS_PATH", "/models/imagewam_release/libero/flux2_klein_4b/dataset_stats.json")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
FLUX2_MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "")
FLUX2_AE_MODEL_PATH = os.environ.get("FLUX2_AE_MODEL_PATH", "")
FLUX2_QWEN3_MODEL_SPEC = os.environ.get("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")
VISUALIZE_DREAM = os.environ.get("VISUALIZE_DREAM", "0").lower() in ("1", "true", "yes")


def _compose_cfg():
    """Compose the upstream eval config outside hydra.main, mirroring the resolvers that
    eval_libero_single.py installs at import time."""
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    for name, fn in (
        ("eval", eval),
        ("max", lambda x: max(x)),
        ("split", lambda s, idx: s.split("/")[int(idx)]),
    ):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass

    overrides = [
        f"task=libero_flux2_klein_{VARIANT}_base_imagewam",
        f"ckpt={CKPT}",
        f"EVALUATION.dataset_stats_path={STATS}",
        f"model.flux2_src_path={FLUX2_SRC}",
        f"model.flux2_model_path={FLUX2_MODEL_PATH}",
        f"model.ae_model_path={FLUX2_AE_MODEL_PATH}",
        f"model.variant=klein-base-{VARIANT}",
        f"model.qwen3_model_spec={FLUX2_QWEN3_MODEL_SPEC}",
        "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true",
        "model.proprio_dim=8",
    ]
    cfg_dir = os.path.join(REPO, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    return cfg


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    for p in (REPO, os.path.join(FLUX2_SRC, "src"), FLUX2_SRC):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    for path, what in ((CKPT, "checkpoint"), (STATS, "dataset stats")):
        if not os.path.exists(path):
            print(f"FAIL: {what} not found: {path}\n"
                  f"      run /ryzers/scripts/download_checkpoints.sh first.", file=sys.stderr)
            return 1

    from hydra.utils import instantiate

    cfg = _compose_cfg()

    # Dimensions come straight from the upstream config so the smoke tracks the real LIBERO
    # 2-cam setup (video_size, proprio 8, action 7).
    video_size = list(cfg.data.train.video_size)
    height, width = int(video_size[0]), int(video_size[1])
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_dim = int(cfg.data.train.processor.action_output_dim)
    # Mirror eval_libero_single.py: null EVALUATION.action_horizon => num_frames - 1.
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (int(action_horizon_cfg) if action_horizon_cfg is not None
                      else int(cfg.data.train.num_frames) - 1)
    print(f"config           : {CONFIG_NAME}  HxW={height}x{width}  "
          f"action_horizon={action_horizon}  proprio_dim={proprio_dim}  action_dim={action_dim}")

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded     : {time.time() - t0:.1f}s  params={n_params / 1e9:.2f}B")

    image = (torch.rand(1, 3, height, width) * 2.0 - 1.0)
    proprio = torch.zeros(1, proprio_dim)

    def _kwargs():
        return dict(prompt=PROMPT, input_image=image, action_horizon=action_horizon,
                    proprio=proprio, num_inference_steps=NUM_STEPS, seed=0)

    def _run():
        with torch.no_grad():
            return model.infer_action(**_kwargs())

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync(); t0 = time.time(); _run(); _sync()
    cold_ms = (time.time() - t0) * 1000.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); t1 = time.time(); out = _run(); _sync()
    warm_ms = (time.time() - t1) * 1000.0

    action = out["action"]
    action = action.detach().to(dtype=torch.float32, device="cpu").numpy()
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"infer_action     : {action.shape}  (steps={NUM_STEPS})")
    print(f"  first-call     : {cold_ms:.0f} ms  (incl. one-time ROCm warmup)")
    print(f"  steady-state   : {warm_ms:.0f} ms  peak={peak_gb:.1f} GB")

    if action.ndim != 2 or action.shape[-1] != action_dim:
        print(f"FAIL: expected (T, {action_dim}) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(action).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1

    if VISUALIZE_DREAM:
        # ImageWAM's "dream" for the FLUX.2 klein image stack is a single edited future frame
        # via infer_video_flux2 (2D flux AE decode) -- NOT the wan/omnigen video infer_joint
        # path (which needs a temporal VAE). Dispatch on model.stack for correctness.
        stack = str(getattr(model, "stack", ""))
        with torch.no_grad():
            if stack == "flux2":
                dream_out = model.infer_video_flux2(
                    prompt=PROMPT, input_image=image, proprio=proprio,
                    num_inference_steps=NUM_STEPS, seed=0)
                dream = dream_out.get("image")
            else:
                num_frames = int(cfg.data.train.num_frames)
                freq = int(cfg.data.train.action_video_freq_ratio)
                nvf = (num_frames - 1) // freq + 1
                dream_out = model.infer_joint(num_video_frames=nvf, **_kwargs())
                dream = dream_out.get("video")
        shp = tuple(dream.shape) if hasattr(dream, "shape") else type(dream)
        finite = bool(torch.isfinite(dream).all()) if hasattr(dream, "shape") else "n/a"
        print(f"dreamed image    : {shp}  finite={finite}  (stack={stack})")

        # Save the dreamed frame so P5/P6 two-column visualizations can reuse it.
        out_dir = os.environ.get("OUT_DIR", "/outputs")
        try:
            import numpy as _np
            from PIL import Image as _Image
            arr = dream.detach().to(torch.float32).cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = _np.transpose(arr, (1, 2, 0))
            arr = ((arr.clip(-1, 1) + 1.0) * 127.5).astype("uint8") if arr.min() < 0 \
                else (arr.clip(0, 1) * 255).astype("uint8")
            os.makedirs(out_dir, exist_ok=True)
            _Image.fromarray(arr).save(os.path.join(out_dir, "dream_smoke.png"))
            print(f"dreamed image    : saved -> {out_dir}/dream_smoke.png")
        except Exception as e:  # visualization is best-effort in the smoke
            print(f"dreamed image    : save skipped ({type(e).__name__}: {e})")

    print("PASS: ImageWAM (FLUX.2) full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
