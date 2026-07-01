# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""RT forward-pass optimization study on gfx1151 (Strix Halo).

Covers the plan's `decompose` + `vision-tokens` todos in one model-load session:

1. Stage decomposition: times the vision backbone via a forward hook; isolates the
   flow loop via a num_steps slope; the remainder is the LLM prefill. Also reports
   the vision-token count produced by the image processor.
2. Vision-token sweep: lowers `max_crops` (Molmo multi-crop tokenizer) and measures
   prefill latency + action-L1 vs the max_crops=8 bf16 reference.

Reuses the upstream DROID `Policy` (real load + bf16 patch) and the canonical README
sample shipped in the checkpoint assets, so no gated data is needed.

Run inside molmoact2:latest via ryzer_shell.sh, e.g.:
    bash ~/molmoact2-xarm6/scripts/ryzer_shell.sh \
        "MAX_CROPS_SWEEP=8,4,2,1 NUM_STEPS=4 python /scripts/bench_opt_study.py"

Env: DTYPE (bfloat16), WARMUP (3), TIMED_N (6), NUM_STEPS (4),
STEPS_DECOMP ("2,10" for the flow slope), MAX_CROPS_SWEEP ("8,4,2,1").
"""
import json
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
WARMUP = int(os.environ.get("WARMUP", "3"))
TIMED_N = int(os.environ.get("TIMED_N", "6"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "4"))
STEPS_DECOMP = [int(x) for x in os.environ.get("STEPS_DECOMP", "2,10").split(",")]
MAX_CROPS_SWEEP = [int(x) for x in os.environ.get("MAX_CROPS_SWEEP", "8,4,2,1").split(",")]
OUT_JSON = os.environ.get("OUT_JSON", "/ryzers/outputs/opt_study_vision.json")
REF_NPY = os.environ.get("REF_NPY", "/ryzers/outputs/opt_study_ref_actions.npy")

TASK = "Put the black objects into the drawer and close the drawer."
ROBOT_STATE = np.array(
    [-0.12726949, -0.30641943, 0.09134164, -2.4143615,
     -0.26460838, 2.068765, 0.123698, 0.0], dtype=np.float32,
)


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def find_vision_backbone(model):
    """Locate the MolmoAct2 vision backbone module across wrapper levels."""
    for path in ("model.vision_backbone", "vision_backbone",
                 "model.model.vision_backbone"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and obj is not None and isinstance(obj, torch.nn.Module):
            return obj, path
    return None, None


def main() -> int:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    fh = open(OUT_JSON.replace(".json", ".txt"), "w")
    log("=" * 72, fh)
    log(f"RT opt study: decompose + vision tokens  ({time.strftime('%Y-%m-%d %H:%M:%S')})", fh)
    log("=" * 72, fh)
    log(f"torch={torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}", fh)
    log(f"dtype={DTYPE} warmup={WARMUP} timed_n={TIMED_N} num_steps={NUM_STEPS} "
        f"steps_decomp={STEPS_DECOMP} max_crops_sweep={MAX_CROPS_SWEEP}", fh)

    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID  # noqa: E402
    from huggingface_hub import hf_hub_download  # noqa: E402

    policy = Policy(repo_id=REPO_ID, device="cuda:0", dtype=DTYPE)
    n_params = sum(p.numel() for p in policy.model.parameters()) / 1e9
    log(f"model={type(policy.model).__name__} params={n_params:.2f}B", fh)

    vb, vb_path = find_vision_backbone(policy.model)
    log(f"vision_backbone @ {vb_path}: {type(vb).__name__ if vb else None}", fh)

    vstate = {"ms": 0.0, "tok": None}

    def _pre(_m, _i):
        torch.cuda.synchronize()
        vstate["t0"] = time.perf_counter()

    def _post(_m, _i, out):
        torch.cuda.synchronize()
        vstate["ms"] = (time.perf_counter() - vstate["t0"]) * 1000.0
        o = out[0] if isinstance(out, (tuple, list)) and out else out
        if torch.is_tensor(o):
            vstate["tok"] = int(np.prod(o.shape[:-1]))

    if vb is not None:
        vb.register_forward_pre_hook(_pre)
        vb.register_forward_hook(_post)

    ext = Image.open(hf_hub_download(REPO_ID, "assets/sample_exterior_1_left_rgb.png")).convert("RGB")
    wrist = Image.open(hf_hub_download(REPO_ID, "assets/sample_wrist_left_rgb.png")).convert("RGB")
    images = [ext, ext, wrist]
    log(f"sample: ext={ext.size} wrist={wrist.size}", fh)

    def set_max_crops(c):
        ip = getattr(policy.processor, "image_processor", policy.processor)
        for attr in ("max_crops",):
            if hasattr(ip, attr):
                setattr(ip, attr, c)
        if hasattr(policy.processor, "max_crops"):
            policy.processor.max_crops = c

    @torch.inference_mode()
    def predict(num_steps):
        out = policy.model.predict_action(
            processor=policy.processor, images=images, task=TASK, state=ROBOT_STATE,
            norm_tag=NORM_TAG, inference_action_mode="continuous",
            enable_depth_reasoning=False, num_steps=num_steps,
            normalize_language=True, enable_cuda_graph=False,
        )
        raw = out.actions if hasattr(out, "actions") else out
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        a = np.asarray(raw, dtype=np.float32)
        return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

    def timed(num_steps, n):
        lats, vis = [], []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict(num_steps)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
            vis.append(vstate["ms"])
        return lats, vis

    results = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "hip": torch.version.hip, "dtype": str(DTYPE), "params_B": round(n_params, 2),
        "decompose": {}, "vision_sweep": [],
    }

    # ---- Stage decomposition (flow slope @ default max_crops=8) ----
    set_max_crops(8)
    for _ in range(WARMUP):
        ref_actions = predict(NUM_STEPS)
    np.save(REF_NPY, ref_actions)
    decomp = {}
    for steps in STEPS_DECOMP:
        lats, vis = timed(steps, TIMED_N)
        decomp[steps] = {"total_ms": round(statistics.mean(lats), 1),
                         "vision_ms": round(statistics.mean(vis), 1)}
        log(f"[decomp steps={steps:2d}] total={decomp[steps]['total_ms']:.0f} ms  "
            f"vision={decomp[steps]['vision_ms']:.0f} ms  tokens={vstate['tok']}", fh)
    s_lo, s_hi = min(STEPS_DECOMP), max(STEPS_DECOMP)
    flow_per_step = (decomp[s_hi]["total_ms"] - decomp[s_lo]["total_ms"]) / max(1, (s_hi - s_lo))
    vision_ms = decomp[s_hi]["vision_ms"]
    prefill_ms = decomp[s_hi]["total_ms"] - vision_ms - flow_per_step * s_hi
    results["decompose"] = {
        "vision_tokens": vstate["tok"],
        "vision_ms": round(vision_ms, 1),
        "llm_prefill_ms": round(prefill_ms, 1),
        "flow_per_step_ms": round(flow_per_step, 1),
        "total_at_steps": decomp,
    }
    log(f"  => vision={vision_ms:.0f} ms | llm_prefill≈{prefill_ms:.0f} ms | "
        f"flow≈{flow_per_step:.1f} ms/step | vis_tokens={vstate['tok']}", fh)

    # ---- Vision-token sweep (max_crops) ----
    log("-" * 72, fh)
    for c in MAX_CROPS_SWEEP:
        set_max_crops(c)
        try:
            for _ in range(WARMUP):
                a = predict(NUM_STEPS)
            lats, vis = timed(NUM_STEPS, TIMED_N)
        except Exception as e:  # noqa: BLE001
            log(f"[crops={c}] FAILED: {type(e).__name__}: {str(e)[:160]}", fh)
            results["vision_sweep"].append({"max_crops": c, "error": str(e)[:200]})
            continue
        l1 = float(np.mean(np.abs(a - ref_actions)))
        row = {
            "max_crops": c, "vision_tokens": vstate["tok"],
            "total_ms": round(statistics.mean(lats), 1),
            "vision_ms": round(statistics.mean(vis), 1),
            "infer_per_s": round(1000.0 / statistics.mean(lats), 2),
            "action_l1_vs_c8": round(l1, 4),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        }
        results["vision_sweep"].append(row)
        log(f"[crops={c}] tokens={row['vision_tokens']} total={row['total_ms']:.0f} ms "
            f"({row['infer_per_s']} infer/s) vision={row['vision_ms']:.0f} ms "
            f"L1_vs_c8={row['action_l1_vs_c8']}", fh)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote {OUT_JSON}", fh)
    log("PASS", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
