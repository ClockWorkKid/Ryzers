# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 1: full-model smoke test for VLA-JEPA on AMD Strix Halo (gfx1151).

Downloads the real VLA-JEPA checkpoint (+ the Qwen3-VL base VLM and V-JEPA2 encoder)
into the mounted HF cache on first run, repoints the checkpoint's ``config.yaml`` to
those local base models, loads the model via the upstream
``baseframework.from_pretrained`` loader, and runs a single ``predict_action`` on a
dummy multi-view observation to prove the whole VLA path executes end-to-end on ROCm.

This is a FIRST-CUT smoke: the dummy observation dims (views/state) are best-effort and
may be refined once we validate against the real checkpoint config (milestone M3).

Env (see config.yaml): MODEL_REPO, CKPT_REL, BASE_VLM, BASE_ENCODER, DTYPE,
                       STATE_DIM (override), N_VIEWS (override), INSTRUCTION.
Exits non-zero on any failure so `ryzers run` / CI catches a broken image.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL_REPO = os.environ.get("MODEL_REPO", "ginwind/VLA-JEPA")
CKPT_REL = os.environ.get("CKPT_REL", "LIBERO/checkpoints/VLA-JEPA-LIBERO.pt")
BASE_VLM = os.environ.get("BASE_VLM", "Qwen/Qwen3-VL-2B-Instruct")
BASE_ENCODER = os.environ.get("BASE_ENCODER", "facebook/vjepa2-vitl-fpc64-256")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
N_VIEWS = int(os.environ.get("N_VIEWS") or "2")
STATE_DIM = int(os.environ.get("STATE_DIM") or "8")
INSTRUCTION = os.environ.get("INSTRUCTION", "pick up the object")


def resolve_checkpoint() -> str:
    """Download only the needed suite of the VLA-JEPA repo and return the local .pt path.

    The repo ships several suites (Pretrain/LIBERO/Real-world/SimplerEnv), each with a
    multi-GB ``.pt``. We only need the suite that CKPT_REL points at, so restrict the
    snapshot to that top-level dir instead of pulling the entire repo.
    """
    from huggingface_hub import snapshot_download

    suite = CKPT_REL.split("/", 1)[0]
    print(f"downloading    : {MODEL_REPO} (suite '{suite}/' only)")
    local = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[f"{suite}/*"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    ckpt = os.path.join(local, CKPT_REL)
    if not os.path.isfile(ckpt):
        print(f"FAIL: checkpoint not found at {ckpt}", file=sys.stderr)
        sys.exit(1)
    return ckpt


def repoint_config(ckpt_path: str) -> None:
    """Repoint the checkpoint's config.yaml base model paths to the local HF ids.

    read_mode_config() expects <run_dir>/config.yaml where run_dir = ckpt.parents[1].
    Upstream stores the authors' local training paths in base_vlm/base_encoder; we
    override them so transformers resolves the models from the HF cache.
    """
    from pathlib import Path
    from omegaconf import OmegaConf

    run_dir = Path(ckpt_path).parents[1]
    cfg_yaml = run_dir / "config.yaml"
    if not cfg_yaml.exists():
        print(f"FAIL: missing {cfg_yaml} (from_pretrained needs it)", file=sys.stderr)
        sys.exit(1)
    cfg = OmegaConf.load(str(cfg_yaml))
    try:
        old_vlm = OmegaConf.select(cfg, "framework.qwenvl.base_vlm")
        old_enc = OmegaConf.select(cfg, "framework.vj2_model.base_encoder")
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", BASE_VLM, force_add=True)
        OmegaConf.update(cfg, "framework.vj2_model.base_encoder", BASE_ENCODER, force_add=True)
        OmegaConf.save(cfg, str(cfg_yaml))
        print(f"repointed cfg  : base_vlm {old_vlm} -> {BASE_VLM}")
        print(f"                 base_encoder {old_enc} -> {BASE_ENCODER}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not repoint config.yaml ({e}); using as-is", file=sys.stderr)


def main() -> int:
    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")

    ckpt = resolve_checkpoint()
    repoint_config(ckpt)

    # Also warm the base models into the cache (predict path constructs them on load).
    from huggingface_hub import snapshot_download
    for repo in (BASE_VLM, BASE_ENCODER):
        snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)

    from starVLA.model.framework.base_framework import baseframework

    t0 = time.time()
    print(f"loading model  : {ckpt}")
    vla = baseframework.from_pretrained(ckpt)
    vla = vla.to("cuda:0").to(DTYPE).eval()
    n_params = sum(p.numel() for p in vla.parameters())
    print(f"model loaded   : {time.time() - t0:.1f}s params={n_params / 1e9:.2f}B")

    # One dummy multi-view observation.
    imgs = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            for _ in range(N_VIEWS)]
    state = np.zeros((1, STATE_DIM), dtype=np.float32)

    t1 = time.time()
    out = vla.predict_action(batch_images=[imgs], instructions=[INSTRUCTION], state=[state])
    dt = (time.time() - t1) * 1000.0
    actions = np.asarray(out["normalized_actions"], dtype=np.float32)
    print(f"predict_action : {actions.shape} in {dt:.0f} ms")
    if not np.isfinite(actions).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1
    print("PASS: VLA-JEPA full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
