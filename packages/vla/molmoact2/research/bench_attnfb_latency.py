# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Latency of the attention-feedback frame types on the DROID sample, so the
cadence (VIS_ATTNFB_SURVEY_EVERY=N) accuracy/latency tradeoff can be quantified:

  full      : keep=1.0, no attnfb            (clean baseline)
  prune      : keep=0.5, attnfb steady frame  (SDPA, deterministic group-drop)
  survey     : keep=0.5, attnfb survey frame  (FULL encode + EAGER attn + harvest)

Effective per-frame latency at cadence N = (survey + (N-1)*prune) / N. Writes
latency_attnfb.csv. Runs INSIDE the molmoact2 container after the patch stack.
"""
import os
import statistics
import sys
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
KEEP = float(os.environ.get("VIS_GROUPDROP_KEEP_FRAC", "0.5"))
CADENCES = [int(x) for x in os.environ.get("CADENCES", "2,4,8").split(",")]
OUT = os.environ.get("OUT", "/ryzers/outputs/attnfb/latency_attnfb.csv")
TASK = "Put the black objects into the drawer and close the drawer."
ROBOT_STATE = np.array(
    [-0.12726949, -0.30641943, 0.09134164, -2.4143615,
     -0.26460838, 2.068765, 0.123698, 0.0], dtype=np.float32,
)


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID
    from huggingface_hub import hf_hub_download

    policy = Policy(repo_id=REPO_ID, device="cuda:0", dtype=DTYPE)
    vb = policy.model.model.vision_backbone
    vis = {"ms": 0.0}
    orig = vb.forward

    def hook(images, ppi):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(images, ppi)
        torch.cuda.synchronize()
        vis["ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    vb.forward = hook
    ext = Image.open(hf_hub_download(REPO_ID, "assets/sample_exterior_1_left_rgb.png")).convert("RGB")
    wrist = Image.open(hf_hub_download(REPO_ID, "assets/sample_wrist_left_rgb.png")).convert("RGB")
    images = [ext, ext, wrist]

    @torch.inference_mode()
    def predict():
        policy.model.predict_action(
            processor=policy.processor, images=images, task=TASK, state=ROBOT_STATE,
            norm_tag=NORM_TAG, inference_action_mode="continuous",
            enable_depth_reasoning=False, num_steps=4,
            normalize_language=True, enable_cuda_graph=False,
        )

    def timed(force_survey=False):
        lat, vms = [], []
        for _ in range(TIMED_N):
            if force_survey:
                vb._attn_feedback_saliency_out = None
                vb._attn_feedback_saliency_in = None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            predict()
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0); vms.append(vis["ms"])
        return statistics.median(lat), statistics.median(vms)

    rows = {}

    # full baseline: no attnfb, keep=1.0
    os.environ.pop("VIS_ATTNFB", None)
    os.environ["VIS_GROUPDROP_KEEP_FRAC"] = "1.0"
    for _ in range(WARMUP): predict()
    rows["full"] = timed()

    # attnfb: survey (forced) and steady prune
    os.environ["VIS_ATTNFB"] = "1"
    os.environ["VIS_ATTNFB_SELFCARRY"] = "1"
    os.environ["VIS_GROUPDROP_KEEP_FRAC"] = str(KEEP)
    vb._attn_feedback_saliency_out = None; vb._attn_feedback_saliency_in = None
    for _ in range(WARMUP): predict()           # 1st warmup surveys, rest prune -> steady
    rows["prune"] = timed(force_survey=False)
    rows["survey"] = timed(force_survey=True)

    print(f"{'frame':>8} {'total_ms':>9} {'vision_ms':>10}", flush=True)
    for k in ("full", "prune", "survey"):
        print(f"{k:>8} {rows[k][0]:>9.0f} {rows[k][1]:>10.0f}", flush=True)
    with open(OUT, "w") as fh:
        fh.write("frame,keep,total_ms,vision_ms\n")
        fh.write(f"full,1.0,{rows['full'][0]:.0f},{rows['full'][1]:.0f}\n")
        fh.write(f"prune,{KEEP},{rows['prune'][0]:.0f},{rows['prune'][1]:.0f}\n")
        fh.write(f"survey,{KEEP},{rows['survey'][0]:.0f},{rows['survey'][1]:.0f}\n")
        fh.write("# cadence_N,eff_total_ms,eff_vision_ms,speedup_vs_full\n")
        s_t, s_v = rows["survey"]; p_t, p_v = rows["prune"]; f_t = rows["full"][0]
        for N in CADENCES:
            eff_t = (s_t + (N - 1) * p_t) / N
            eff_v = (s_v + (N - 1) * p_v) / N
            fh.write(f"# {N},{eff_t:.0f},{eff_v:.0f},{f_t/eff_t:.2f}\n")
            print(f"cadence N={N}: eff_total={eff_t:.0f}ms vision={eff_v:.0f}ms speedup={f_t/eff_t:.2f}x", flush=True)
    print("wrote", OUT, flush=True)
    print("BENCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
