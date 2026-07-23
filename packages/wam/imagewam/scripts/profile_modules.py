# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""P8 per-module latency / computation analysis of FLUX.2 ImageWAM (stack='flux2') on AMD
Strix Halo (gfx1151), bf16.

Instruments the two real inference paths and reports where time goes, separating the FIXED
per-call cost (Qwen3 text encode, VAE image encode, video prefill) from the PER-STEP diffusion
cost (the "known" knob). Also measures ImageWAM-specific wins:
  * text-embedding caching across replans (instruction is fixed per episode -> encode Qwen3 once,
    reuse via the `context`/`context_mask` args that infer_action_flux2 already accepts);
Reports param counts per submodule, a JSON, and a bar chart.

Paths:
  ACTION  (infer_action_flux2)   : text -> proprio -> img_encode -> video_prefill -> [action loop] 
  DREAM   (infer_video_flux2)    : text -> proprio -> img_encode -> [video-DiT loop] -> vae_decode
Inputs are synthetic tensors of the correct shapes (latency is shape- not content-dependent).
"""
import os
import sys
import json
import time

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
CKPT = os.environ.get("CKPT_PATH", "/models/imagewam_release/libero/flux2_klein_4b/model.pt")
STATS = os.environ.get("DATASET_STATS_PATH", "/models/imagewam_release/libero/flux2_klein_4b/dataset_stats.json")
FLUX2_MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "")
FLUX2_AE_MODEL_PATH = os.environ.get("FLUX2_AE_MODEL_PATH", "")
QWEN3 = os.environ.get("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
STEPS = int(os.environ.get("NUM_STEPS") or "20")
WARMUP = int(os.environ.get("WARMUP") or "3")
ITERS = int(os.environ.get("ITERS") or "10")
ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON") or "16")
H = int(os.environ.get("IMG_H") or "224")
W = int(os.environ.get("IMG_W") or "448")
PROMPT = os.environ.get("PROMPT", "A video recorded from a robot's point of view executing the following "
                         "instruction: pick up the alphabet soup and place it in the basket")

for p in (REPO, os.path.join(FLUX2_SRC, "src"), FLUX2_SRC):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# ------------------------- timing infrastructure -------------------------
_ACC = {}
_ORDER = []
_ON = {"v": False}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(label, fn):
    def wrap(*a, **k):
        if not _ON["v"]:
            return fn(*a, **k)
        _sync(); t0 = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        if label not in _ACC:
            _ACC[label] = 0.0
            _ORDER.append(label)
        _ACC[label] += time.perf_counter() - t0
        return out
    return wrap


def _patch(obj, name, label):
    orig = getattr(obj, name)
    setattr(obj, name, _timed(label, orig))


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    overrides = [
        f"task=libero_flux2_klein_{VARIANT}_base_imagewam", f"ckpt={CKPT}",
        f"EVALUATION.dataset_stats_path={STATS}", f"model.flux2_src_path={FLUX2_SRC}",
        f"model.flux2_model_path={FLUX2_MODEL_PATH}", f"model.ae_model_path={FLUX2_AE_MODEL_PATH}",
        f"model.variant=klein-base-{VARIANT}", f"model.qwen3_model_spec={QWEN3}",
        "model.load_text_encoder=true", "model.pack_proprio_after_text=true", "model.proprio_dim=8",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        return compose(config_name="sim_libero_omnigen2", overrides=overrides)


def _stats(xs):
    xs = sorted(xs)
    return {"mean_ms": round(1000 * float(np.mean(xs)), 2), "std_ms": round(1000 * float(np.std(xs)), 2),
            "min_ms": round(1000 * xs[0], 2), "max_ms": round(1000 * xs[-1], 2)}


def _param_breakdown(model):
    groups = {}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = sum(groups.values())
    out = {k: round(v / 1e9, 3) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])}
    out["_total_B"] = round(total / 1e9, 3)
    return out


def main() -> int:
    from hydra.utils import instantiate
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need ROCm device", file=sys.stderr); return 1
    print(f"device : {torch.cuda.get_device_name(0)}  torch={torch.__version__} hip={torch.version.hip}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = _compose_cfg()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    params = _param_breakdown(model)
    print("params(B):", json.dumps(params))

    img = (torch.rand(1, 3, H, W, device="cuda", dtype=torch.bfloat16) * 2 - 1)
    proprio = torch.randn(1, 8, device="cuda", dtype=torch.bfloat16)

    # ---- patch sub-stages (bound methods) ----
    _patch(model, "_prepare_flux2_infer_text", "text_encode(Qwen3)")
    _patch(model, "_append_proprio_to_context_if_enabled", "proprio")
    _patch(model, "_encode_flux2_image_tokens", "img_encode(VAE)")
    _patch(model.video_expert, "pre_dit", "video_pre_dit")
    _patch(model.video_expert, "post_dit", "video_post_dit")
    _patch(model.mot, "prefill_flux2_video_cache", "video_prefill(KVcache)")
    _patch(model.mot, "forward_action_with_video_cache", "action_MoT_loop")
    _patch(model.action_expert, "pre_dit", "action_pre/post")
    _patch(model.action_expert, "post_dit", "action_pre/post")
    _patch(model.infer_action_scheduler, "step", "sched_step")
    _patch(model, "_forward_flux2_video_only", "video_DiT_loop")
    _patch(model.infer_video_scheduler, "step", "sched_step")
    if hasattr(model, "_decode_flux2_image_tokens"):
        _patch(model, "_decode_flux2_image_tokens", "vae_decode")

    def run_action(context=None, context_mask=None, prompt=PROMPT):
        with torch.no_grad():
            return model.infer_action_flux2(prompt=prompt, input_image=img, action_horizon=ACTION_HORIZON,
                                            proprio=proprio, num_inference_steps=STEPS, seed=0,
                                            context=context, context_mask=context_mask)

    def run_video():
        with torch.no_grad():
            return model.infer_video_flux2(prompt=PROMPT, input_image=img, proprio=proprio,
                                           num_inference_steps=STEPS, seed=0)

    def bench(run_fn, tag):
        _ON["v"] = False
        for _ in range(WARMUP):
            run_fn()
        # clean end-to-end (no per-stage sync overhead)
        totals = []
        for _ in range(ITERS):
            _sync(); t0 = time.perf_counter(); run_fn(); _sync(); totals.append(time.perf_counter() - t0)
        # per-module breakdown (with sync)
        _ON["v"] = True
        per_call = {}
        for _ in range(ITERS):
            _ACC.clear()
            run_fn()
            for k, v in _ACC.items():
                per_call.setdefault(k, []).append(v)
        _ON["v"] = False
        breakdown = {k: _stats(v) for k, v in per_call.items()}
        print(f"\n===== {tag} =====  end-to-end {_stats(totals)['mean_ms']} ms/call ({STEPS} steps)")
        for k in _ORDER:
            if k in breakdown:
                print(f"  {k:26s} {breakdown[k]['mean_ms']:8.2f} ms")
        return {"end_to_end": _stats(totals), "modules": breakdown, "steps": STEPS}

    res = {"device": torch.cuda.get_device_name(0), "dtype": "bf16", "params_B": params,
           "action_horizon": ACTION_HORIZON, "img_hw": [H, W], "iters": ITERS, "steps": STEPS}
    res["action_path"] = bench(run_action, "ACTION path (infer_action_flux2)")
    res["dream_path"] = bench(run_video, "DREAM path (infer_video_flux2)")

    # ---- ImageWAM-specific win: cache text across replans ----
    text_cache = {}
    try:
        _ON["v"] = False
        with torch.no_grad():
            ctx, ctx_mask = model._prepare_flux2_infer_text(PROMPT, None, None)
        for _ in range(WARMUP):
            run_action(context=ctx.clone(), context_mask=ctx_mask.clone(), prompt=None)
        totals = []
        for _ in range(ITERS):
            _sync(); t0 = time.perf_counter()
            run_action(context=ctx.clone(), context_mask=ctx_mask.clone(), prompt=None)
            _sync(); totals.append(time.perf_counter() - t0)
        naive = res["action_path"]["end_to_end"]["mean_ms"]
        cached = _stats(totals)["mean_ms"]
        text_cache = {"action_naive_ms": naive, "action_textcached_ms": cached,
                      "saved_ms": round(naive - cached, 2),
                      "saved_pct": round(100 * (naive - cached) / naive, 1),
                      "note": "instruction fixed per episode -> encode Qwen3 once, pass context/context_mask"}
        print(f"\n===== TEXT-CACHE WIN =====  naive {naive} ms -> cached {cached} ms "
              f"(-{text_cache['saved_ms']} ms, -{text_cache['saved_pct']}%)")
    except Exception as e:  # noqa: BLE001
        text_cache = {"error": f"{type(e).__name__}: {e}"}
        print("text-cache A/B failed:", text_cache["error"])
    res["text_cache_win"] = text_cache

    json.dump(res, open(os.path.join(OUT_DIR, "p8_latency.json"), "w"), indent=2)
    _make_chart(res, os.path.join(OUT_DIR, "p8_latency.png"))
    print("\nPASS: per-module latency analysis complete -> p8_latency.json / p8_latency.png")
    return 0


def _make_chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def bars(ax, path_res, title, palette):
        mods = path_res["modules"]
        items = sorted(mods.items(), key=lambda kv: -kv[1]["mean_ms"])
        labels = [k for k, _ in items]
        vals = [v["mean_ms"] for _, v in items]
        y = range(len(labels))
        ax.barh(list(y), vals, color=palette)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=7)
        ax.set_xlabel("ms (mean)")
        ax.set_title(f"{title}\nend-to-end {path_res['end_to_end']['mean_ms']:.0f} ms "
                     f"({path_res['steps']} steps, bf16)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    bars(ax[0], res["action_path"], "ACTION path (deployment / closed-loop)", "tab:blue")
    bars(ax[1], res["dream_path"], "DREAM path (image world-model)", "tab:purple")
    tc = res.get("text_cache_win", {})
    sub = ""
    if "saved_ms" in tc:
        sub = (f"ImageWAM win: caching Qwen3 text across replans -> "
               f"action {tc['action_naive_ms']:.0f} ms -> {tc['action_textcached_ms']:.0f} ms "
               f"(-{tc['saved_pct']:.0f}%)")
    fig.suptitle(f"ImageWAM FLUX.2-{VARIANT} per-module latency on {res['device']}   |   {sub}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
