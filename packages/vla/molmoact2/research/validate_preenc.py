# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Isolation validation for the stage-D pre-encoder pruner (patch_vis_preenc.py).

One model load (DROID policy, bf16). Times the WHOLE vision backbone forward (ViT
included) via a hook, then compares:

  * keep=1.0  -> no-op path (original encode_image), reference action + vision_ms
  * keep=K    -> pruned path, action + vision_ms  (sweeps VIS_PREENC_KEEP_FRAC)

Success = pruned actions finite, action-L1 vs full small (cf. stage-B ~0.056), and
vision_ms drops roughly with keep_frac (the whole point: the ViT now scales).
Set KEEPS env to sweep, e.g. KEEPS=1.0,0.5,0.25,0.1
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

SERVER_DIR = os.environ.get("DROID_SERVER_DIR", "/repos/molmoact2/examples/droid")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")
]
NUM_STEPS = int(os.environ.get("NUM_STEPS", "4"))
WARMUP = int(os.environ.get("WARMUP", "2"))
TIMED_N = int(os.environ.get("TIMED_N", "5"))
KEEPS = [float(x) for x in os.environ.get("KEEPS", "1.0,0.5,0.25,0.1").split(",")]
SEED = int(os.environ.get("SEED", "0"))

TASK = "Put the black objects into the drawer and close the drawer."
ROBOT_STATE = np.array(
    [-0.12726949, -0.30641943, 0.09134164, -2.4143615,
     -0.26460838, 2.068765, 0.123698, 0.0], dtype=np.float32,
)


def main() -> int:
    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID  # noqa: E402
    from huggingface_hub import hf_hub_download  # noqa: E402

    print(f"torch={torch.__version__} hip={torch.version.hip} dtype={DTYPE}", flush=True)
    policy = Policy(repo_id=REPO_ID, device="cuda:0", dtype=DTYPE)

    vb = policy.model.model.vision_backbone
    vis = {"ms": 0.0}
    orig_fwd = vb.forward

    def fwd_hook(images, pooled_patches_idx):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_fwd(images, pooled_patches_idx)
        torch.cuda.synchronize()
        vis["ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    vb.forward = fwd_hook

    ext = Image.open(hf_hub_download(REPO_ID, "assets/sample_exterior_1_left_rgb.png")).convert("RGB")
    wrist = Image.open(hf_hub_download(REPO_ID, "assets/sample_wrist_left_rgb.png")).convert("RGB")
    images = [ext, ext, wrist]

    @torch.inference_mode()
    def predict():
        out = policy.model.predict_action(
            processor=policy.processor, images=images, task=TASK, state=ROBOT_STATE,
            norm_tag=NORM_TAG, inference_action_mode="continuous",
            enable_depth_reasoning=False, num_steps=NUM_STEPS,
            normalize_language=True, enable_cuda_graph=False,
        )
        raw = out.actions if hasattr(out, "actions") else out
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        a = np.asarray(raw, dtype=np.float32)
        return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

    ref = None
    print("=" * 70, flush=True)
    print(f"{'keep':>6} {'vision_ms':>10} {'speedup':>8} {'L1_vs_full':>11} {'finite':>7}", flush=True)
    full_ms = None
    for kf in KEEPS:
        os.environ["VIS_PREENC_KEEP_FRAC"] = str(kf)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        for _ in range(WARMUP):
            predict()
        vlist = []
        act = None
        for _ in range(TIMED_N):
            act = predict()
            vlist.append(vis["ms"])
        v = float(np.median(vlist))
        if kf >= 1.0 and ref is None:
            ref = act
            full_ms = v
        l1 = float("nan") if ref is None else float(np.mean(np.abs(act - ref)))
        sp = float("nan") if full_ms is None else full_ms / v
        print(f"{kf:>6.2f} {v:>10.1f} {sp:>8.2f} {l1:>11.4f} {str(bool(np.isfinite(act).all())):>7}", flush=True)
    print("VALIDATE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
