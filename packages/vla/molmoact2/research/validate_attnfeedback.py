# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Isolation check for attention-feedback deterministic group-drop (VIS_ATTNFB).

Run inside the molmoact2 container AFTER applying patch_vis_groupdrop.py and
patch_vis_attnfeedback.py. Confirms, on the DROID sample (whose trusted
gradient/occlusion peak groups live in artifacts/.../diag_summary.json):

  1. HARVEST  -- a survey frame (no prior saliency) populates a finite, non-uniform
     per-group saliency from LLM self-attention (output_attentions path works).
  2. PRUNE    -- the next frame uses that saliency deterministically: _groupdrop_keep
     is set, vision_ms drops vs the survey frame, actions stay finite, and the
     SAME top groups are kept across repeats (deterministic, unlike random).
  3. SIGNAL   -- reports the percentile rank of the known gradient/occlusion peak
     groups inside the harvested saliency (>0.5 == better than random) and the
     sink-suppression (top1 mass) vs the raw (un-debiased) attention. Dumps
     saliency.npy + report.json for the overlay/plot step.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("VIS_ATTNFB", "1")
os.environ.setdefault("VIS_ATTNFB_SELFCARRY", "1")
os.environ.setdefault("VIS_ATTNFB_DEBUG", "1")
KEEP = float(os.environ.get("VIS_GROUPDROP_KEEP_FRAC", "0.5"))
os.environ["VIS_GROUPDROP_KEEP_FRAC"] = str(KEEP)

SERVER_DIR = os.environ.get("DROID_SERVER_DIR", "/repos/molmoact2/examples/droid")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")
]
OUT = os.environ.get("OUT", "/ryzers/outputs/attnfb/validate")
# Trusted peak groups for the DROID sample (artifacts/strix/interp/diag/diag_summary.json
# and interp_summary.json -> droid_sample): grad_l2 group 433, occlusion group 432.
PEAK_GROUPS = [int(x) for x in os.environ.get("PEAK_GROUPS", "433,432").split(",")]
TASK = "Put the black objects into the drawer and close the drawer."
ROBOT_STATE = np.array(
    [-0.12726949, -0.30641943, 0.09134164, -2.4143615,
     -0.26460838, 2.068765, 0.123698, 0.0], dtype=np.float32,
)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID
    from huggingface_hub import hf_hub_download

    policy = Policy(repo_id=REPO_ID, device="cuda:0", dtype=DTYPE)
    inner = policy.model.model            # MolmoAct2 inner model (has vision_backbone)
    vb = inner.vision_backbone
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
    def run():
        out = policy.model.predict_action(
            processor=policy.processor, images=images, task=TASK, state=ROBOT_STATE,
            norm_tag=NORM_TAG, inference_action_mode="continuous",
            enable_depth_reasoning=False, num_steps=4,
            normalize_language=True, enable_cuda_graph=False,
        )
        a = out.actions if hasattr(out, "actions") else out
        return np.asarray(a.detach().float().cpu() if hasattr(a, "detach") else a)

    def saliency():
        s = getattr(vb, "_attn_feedback_saliency_out", None)
        return None if s is None else s.float().cpu().numpy().reshape(-1)

    ok = True
    # reset any carry
    vb._attn_feedback_saliency_out = None
    vb._attn_feedback_saliency_in = None

    # ---- Frame 1: SURVEY ----
    act1 = run()
    vis_survey = vis["ms"]
    keep_survey = getattr(vb, "_groupdrop_keep", None)
    sal = saliency()
    survey_full = keep_survey is None
    sal_finite = sal is not None and bool(np.all(np.isfinite(sal)))
    sal_nonuniform = sal is not None and float(np.std(sal)) > 0
    print(f"[survey] vision_ms={vis_survey:.1f} groupdrop_keep={'None' if survey_full else 'set'} "
          f"sal_len={0 if sal is None else sal.size} finite={sal_finite} std={0 if sal is None else float(np.std(sal)):.4g} "
          f"act_finite={bool(np.all(np.isfinite(act1)))}", flush=True)
    if not (survey_full and sal_finite and sal_nonuniform):
        ok = False

    # ---- Frame 2 + 3: PRUNE (deterministic) ----
    act2 = run(); vis_prune = vis["ms"]
    keep2 = getattr(vb, "_groupdrop_keep", None)
    keep2 = None if keep2 is None else keep2.detach().cpu().clone()
    sal_in2 = getattr(vb, "_attn_feedback_saliency_in", None)
    sal_in2 = None if sal_in2 is None else sal_in2.detach().cpu().clone()
    act3 = run()
    keep3 = getattr(vb, "_groupdrop_keep", None)
    keep3 = None if keep3 is None else keep3.detach().cpu().clone()
    sal_in3 = getattr(vb, "_attn_feedback_saliency_in", None)
    sal_in3 = None if sal_in3 is None else sal_in3.detach().cpu().clone()
    pruned = keep2 is not None
    same_sal = (sal_in2 is not None and sal_in3 is not None and torch.equal(sal_in2, sal_in3))
    if keep2 is not None and keep3 is not None and keep2.shape == keep3.shape:
        n_diff = int((keep2 != keep3).sum())
    else:
        n_diff = -1
    print(f"[det   ] keep2={None if keep2 is None else tuple(keep2.shape)}/{None if keep2 is None else int(keep2.sum())} "
          f"keep3={None if keep3 is None else tuple(keep3.shape)}/{None if keep3 is None else int(keep3.sum())} "
          f"n_diff={n_diff} same_saliency_in={same_sal}", flush=True)
    # determinism: identical kept-group masks across repeats
    det = pruned and keep3 is not None and bool(torch.equal(keep2, keep3))
    vis_dropped = vis_prune < vis_survey * 0.95
    act_finite = bool(np.all(np.isfinite(act2))) and bool(np.all(np.isfinite(act3)))
    print(f"[prune ] vision_ms={vis_prune:.1f} (survey {vis_survey:.1f}) pruned={pruned} "
          f"deterministic={det} act_finite={act_finite}", flush=True)
    if not (pruned and det and vis_dropped and act_finite):
        ok = False

    # ---- Signal quality on the harvested survey saliency ----
    report = {"keep_frac": KEEP, "sal_len": 0 if sal is None else int(sal.size),
              "vision_ms_survey": vis_survey, "vision_ms_prune": vis_prune}
    if sal is not None and sal.size > 0:
        order = np.argsort(-sal)                       # high saliency first
        ranks = {int(g): (int(np.where(order == g)[0][0]) if g < sal.size else -1)
                 for g in PEAK_GROUPS}
        pct = {g: (1.0 - r / max(1, sal.size - 1)) if r >= 0 else None for g, r in ranks.items()}
        k = max(1, int(round(KEEP * sal.size)))
        kept_top = set(order[:k].tolist())
        in_topk = {g: (g in kept_top) for g in PEAK_GROUPS}
        s = sal / (sal.sum() + 1e-9)
        top1_mass = float(s.max())
        report.update({"peak_groups": PEAK_GROUPS, "peak_percentile": pct,
                       "peak_in_top%dpct" % int(KEEP * 100): in_topk,
                       "saliency_top1_mass": top1_mass,
                       "saliency_entropy": float(-(s * np.log(s + 1e-12)).sum())})
        np.save(os.path.join(OUT, "saliency.npy"), sal)
        ext.save(os.path.join(OUT, "frame_ext.png"))
        print(f"[signal] peak_percentile={ {g: (None if v is None else round(v,3)) for g,v in pct.items()} } "
              f"in_top{int(KEEP*100)}pct={in_topk} top1_mass={top1_mass:.4f}", flush=True)
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("VALIDATE_OK" if ok else "VALIDATE_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
