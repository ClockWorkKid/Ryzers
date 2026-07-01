# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Latency decomposition for Route-A group-drop, per keep_frac.

Mirrors bench_opt_study's stage split (vision via hook; flow/step via a 2-vs-10
num_steps slope; prefill = remainder) but sweeps VIS_GROUPDROP_KEEP_FRAC. Here BOTH
vision AND prefill scale down with keep_frac (whole pooling groups are dropped, so
fewer patches enter the ViT and fewer pooled tokens enter the LLM). Writes
latency.csv with the curated schema.
"""
import os
import sys
import statistics
import time

import numpy as np
import torch
from PIL import Image

SERVER_DIR = os.environ.get("DROID_SERVER_DIR", "/repos/molmoact2/examples/droid")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")
]
WARMUP = int(os.environ.get("WARMUP", "2"))
TIMED_N = int(os.environ.get("TIMED_N", "5"))
EVAL_STEPS = int(os.environ.get("NUM_STEPS", "4"))   # the fast-path config used in eval
KEEPS = [float(x) for x in os.environ.get("KEEPS", "1.00,0.75,0.50,0.25").split(",")]
OUT = os.environ.get("OUT", "/ryzers/outputs/ablation_groupdrop/latency.csv")

TASK = "Put the black objects into the drawer and close the drawer."
ROBOT_STATE = np.array(
    [-0.12726949, -0.30641943, 0.09134164, -2.4143615,
     -0.26460838, 2.068765, 0.123698, 0.0], dtype=np.float32,
)


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID  # noqa: E402
    from huggingface_hub import hf_hub_download  # noqa: E402

    print(f"torch={torch.__version__} dtype={DTYPE} keeps={KEEPS}", flush=True)
    policy = Policy(repo_id=REPO_ID, device="cuda:0", dtype=DTYPE)
    vb = policy.model.model.vision_backbone
    vis = {"ms": 0.0}
    orig_fwd = vb.forward

    def fwd_hook(images, ppi):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_fwd(images, ppi)
        torch.cuda.synchronize()
        vis["ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    vb.forward = fwd_hook

    ext = Image.open(hf_hub_download(REPO_ID, "assets/sample_exterior_1_left_rgb.png")).convert("RGB")
    wrist = Image.open(hf_hub_download(REPO_ID, "assets/sample_wrist_left_rgb.png")).convert("RGB")
    images = [ext, ext, wrist]

    @torch.inference_mode()
    def predict(num_steps):
        policy.model.predict_action(
            processor=policy.processor, images=images, task=TASK, state=ROBOT_STATE,
            norm_tag=NORM_TAG, inference_action_mode="continuous",
            enable_depth_reasoning=False, num_steps=num_steps,
            normalize_language=True, enable_cuda_graph=False,
        )

    def timed(num_steps):
        lat, vms = [], []
        for _ in range(TIMED_N):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            predict(num_steps)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
            vms.append(vis["ms"])
        return statistics.median(lat), statistics.median(vms)

    with open(OUT, "w") as fh:
        fh.write("keep_frac,vision_ms,llm_prefill_ms,flow_ms_per_step,total_ms,infer_per_s,timestamp\n")
    print(f"{'keep':>6} {'vision':>8} {'prefill':>8} {'flow/st':>8} {'total@'+str(EVAL_STEPS):>9} {'infer/s':>8}", flush=True)
    for kf in KEEPS:
        os.environ["VIS_GROUPDROP_KEEP_FRAC"] = str(kf)
        for _ in range(WARMUP):
            predict(EVAL_STEPS)
        t2, _ = timed(2)
        t10, v10 = timed(10)
        flow = (t10 - t2) / 8.0
        prefill = t10 - v10 - flow * 10
        total = v10 + prefill + flow * EVAL_STEPS
        ips = 1000.0 / total
        with open(OUT, "a") as fh:
            fh.write(f"{kf:.2f},{v10:.0f},{prefill:.0f},{flow:.1f},{total:.0f},{ips:.2f},{time.strftime('%F %T')}\n")
        print(f"{kf:>6.2f} {v10:>8.0f} {prefill:>8.0f} {flow:>8.1f} {total:>9.0f} {ips:>8.2f}", flush=True)
    print("wrote", OUT, flush=True)
    print("BENCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
