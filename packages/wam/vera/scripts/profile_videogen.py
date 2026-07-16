# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-component latency + FLOPs profile of the VERA DROID WAN 14B video planner on ROCm.

Runs ONE language-conditioned generation (scene-2, first prompt by default) and attributes:
  * inclusive GPU time per top-level pipeline component (UMT5 text encoder, CLIP image
    encoder, WanVAE encode/decode, WanModel DiT x num_steps) via CUDA-event forward hooks
    on every nn.Module, plus the top self-time leaf hotspots (rule: identify bottlenecks);
  * theoretical FLOPs per component using torch FlopCounterMode -- each top-level module's
    forward is wrapped to measure ONE real call (real inputs, deterministic modules) and the
    total is per-call-flops x observed-call-count, so we never have to know generate()'s
    internals or the denoise-step knob.

Design mirrors agent_scripts/cosmos3_latency_profile.py but discovers components dynamically
from the pipeline object (WanPipeline is a plain wrapper, not an nn.Module), so it is robust
to attribute renames across the vera codebase.

Env: same checkpoint vars as videogen_droid.py, plus
  PROF_MODE {scene2|scene1}   which demo clip to condition on (default scene2)
  PROF_SEED (default 11)
  SKIP_FLOPS=1                 latency pass only (skip the FlopCounterMode pass)
  OUT_DIR                      where the JSON report is written (default /outputs)
"""
import collections
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vera.video_model.link.wan_pipeline import (
    WanPipeline,
    VideoCondition,
    GenerationConfig,
)

CKPT_DIR = Path(os.environ.get("VERA_DROID_CKPT_DIR", "/models/vera-ckpts/wan-droid-14b"))
CLIPS_DIR = Path(os.environ.get("VERA_DROID_CLIPS_DIR", "/models/vera-ckpts/droid-demo-clips"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_H, MODEL_W_TOTAL, N_VIEWS = 128, 576, 3
VIEW_W = MODEL_W_TOTAL // N_VIEWS
TARGET_FPS = 15
SEED = int(os.environ.get("PROF_SEED", "11"))
PROF_MODE = os.environ.get("PROF_MODE", "scene2").lower()
SKIP_FLOPS = os.environ.get("SKIP_FLOPS", "").strip() in ("1", "true", "yes")


# --- context prep (subset of videogen_droid.py) ------------------------------
def load_video_frames(path, max_frames=None):
    import av
    container = av.open(str(path))
    fps = float(container.streams.video[0].average_rate)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames), fps


def trim_and_resample(frames, src_fps, target_fps):
    ratio = src_fps / target_fps
    idx = [int(round(i * ratio)) for i in range(int(len(frames) / ratio))]
    return frames[[i for i in idx if i < len(frames)]]


def stitch(paths, order):
    aligned = {}
    for name, p in paths.items():
        frames, fps = load_video_frames(p)
        aligned[name] = trim_and_resample(frames, fps, TARGET_FPS)
    n = min(v.shape[0] for v in aligned.values())
    resized = [np.stack([np.array(Image.fromarray(f).resize((VIEW_W, MODEL_H), Image.LANCZOS))
                         for f in aligned[name][:n]]) for name in order]
    return np.concatenate(resized, axis=2)


def to_ctx(frames_uint8):
    t = torch.from_numpy(frames_uint8).float() / 127.5 - 1.0
    return t.permute(0, 3, 1, 2).unsqueeze(0)


def _resolve(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])


def _nparams(mod):
    return sum(p.numel() for p in mod.parameters(recurse=True))


def discover_components(pipeline):
    """Return [(attr_name, nn.Module)] for the meaningful sub-modules of the pipeline.

    VERA wraps text-encoder / VAE / CLIP / DiT inside a single WanImageToVideo module, so if the
    pipeline holds exactly one top-level nn.Module we descend one level (its named_children) to
    surface those parts; otherwise we use the top-level modules directly.
    """
    tops, seen = [], set()
    for name, val in sorted(vars(pipeline).items()):
        if isinstance(val, torch.nn.Module) and id(val) not in seen:
            seen.add(id(val))
            tops.append((name, val))
    if len(tops) == 1:
        parent_name, parent = tops[0]
        children = [(f"{parent_name}.{cn}", cv) for cn, cv in parent.named_children()
                    if isinstance(cv, torch.nn.Module)]
        if children:
            return children
    return tops


def _build_context():
    if PROF_MODE == "scene1":
        canvas = stitch({
            "external": _resolve(CLIPS_DIR / "external_cam.mp4"),
            "wrist": _resolve(CLIPS_DIR / "wrist_cam.mp4"),
            "real": _resolve(CLIPS_DIR / "real.mov"),
        }, ["external", "real", "wrist"])
        prompt = ("a white robot arm reaches down to pick up a line of crackers and then "
                  "places it into a white tray.")
    else:
        ss = CLIPS_DIR / "second_set"
        canvas = stitch({
            "varied_1": _resolve(ss / "varied_camera_1_35317039.mp4"),
            "varied_2": _resolve(ss / "varied_camera_2_39509833.mp4"),
            "hand": _resolve(ss / "hand_camera_16779706.mp4"),
        }, ["varied_1", "varied_2", "hand"])
        prompt = ("a white robot arm approaches the tennis ball. Then, its gripper closes on "
                  "the tennis ball.")
    return canvas, prompt


def main() -> int:
    if not os.environ.get("VERA_WAN14B_CKPT_ROOT"):
        print("FAIL: VERA_WAN14B_CKPT_ROOT not set.", file=sys.stderr)
        return 2
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}",
          flush=True)
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        return 1

    t0 = time.time()
    pipeline = WanPipeline.from_config(
        config_path=str(CKPT_DIR / "algo_config.yaml"),
        ckpt_path=str(CKPT_DIR / "video_model.ckpt"),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    load_s = time.time() - t0
    CTX = pipeline.required_pixel_frames
    CHUNK = pipeline.future_pixel_frames
    print(f"[load] {load_s:.1f}s  context={CTX} frames -> generates {CHUNK} frames", flush=True)

    components = discover_components(pipeline)
    print("[components] top-level nn.Modules on the pipeline:", flush=True)
    for nm, mod in components:
        print(f"    {nm:<20}{type(mod).__name__:<28}{_nparams(mod)/1e6:>9.1f}M params", flush=True)
    print(f"[cfg] GenerationConfig fields: "
          f"{[f for f in getattr(GenerationConfig, '__dataclass_fields__', {})]}", flush=True)

    canvas, prompt = _build_context()
    ctx = to_ctx(canvas[:CTX])
    print(f"[input] canvas {canvas.shape} -> context tensor {tuple(ctx.shape)}\n[input] prompt: {prompt}",
          flush=True)

    def run_gen():
        return pipeline.generate(
            VideoCondition(context_frames=ctx, text=prompt),
            GenerationConfig(decode_outputs=["rgb"]),
        )

    # ---- warmup (kernel autotune) --------------------------------------------
    torch.manual_seed(SEED)
    print("[warmup] running one generation to autotune kernels ...", flush=True)
    tw = time.time()
    _ = run_gen()
    torch.cuda.synchronize()
    print(f"[warmup] done in {time.time()-tw:.1f}s", flush=True)

    # ---- clean latency pass: CUDA-event forward hooks on every submodule ------
    name_of, params_of, class_of = {}, {}, {}
    top_of = {}  # id(submodule) -> top-level component attr name
    for attr, top in components:
        for sub_name, m in top.named_modules():
            name_of[id(m)] = f"{attr}.{sub_name}" if sub_name else attr
            params_of[id(m)] = _nparams(m)
            class_of[id(m)] = type(m).__name__
            top_of[id(m)] = attr

    pending = collections.defaultdict(list)
    pairs = []
    handles = []

    def pre_hook(mod, _inp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        pending[id(mod)].append(ev)

    def post_hook(mod, _inp, _out):
        st = pending[id(mod)]
        if not st:
            return
        start = st.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        pairs.append((id(mod), start, end))

    for _, top in components:
        for m in top.modules():
            handles.append(m.register_forward_pre_hook(pre_hook))
            handles.append(m.register_forward_hook(post_hook))

    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _ = run_gen()
    torch.cuda.synchronize()
    hooked_total_s = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    for h in handles:
        h.remove()

    incl_ms = collections.defaultdict(float)
    calls = collections.defaultdict(int)
    for mid, start, end in pairs:
        incl_ms[mid] += start.elapsed_time(end)
        calls[mid] += 1

    comp_rows = []
    for attr, top in components:
        cid = id(top)
        comp_rows.append({
            "name": attr,
            "class": class_of[cid],
            "params": params_of[cid],
            "calls": calls.get(cid, 0),
            "incl_ms": incl_ms.get(cid, 0.0),
        })
    comp_rows.sort(key=lambda c: c["incl_ms"], reverse=True)

    # self-time hotspots (inclusive - sum(direct children inclusive))
    children_ids = collections.defaultdict(list)
    for mid, nm in name_of.items():
        parent = nm.rsplit(".", 1)[0] if "." in nm else "<root>"
        children_ids[parent].append(mid)
    self_ms = {}
    for mid, nm in name_of.items():
        kids = children_ids.get(nm, [])
        self_ms[mid] = incl_ms.get(mid, 0.0) - sum(incl_ms.get(k, 0.0) for k in kids)
    hot = sorted(
        ((name_of[mid], class_of[mid], self_ms[mid], calls.get(mid, 0))
         for mid in self_ms if self_ms[mid] > 1.0),
        key=lambda r: r[2], reverse=True,
    )[:25]

    # ---- FLOPs pass: wrap each component callable, measure its FIRST real call ----
    # We wrap forward (+ encode/decode where present) so the real call runs *inside* a
    # FlopCounterMode on its first invocation (no recompute, negligible overhead), and count
    # every invocation ourselves -> total FLOPs = gflop_per_call x calls.
    flop_stats = {}  # key -> {"gflops_per_call": float, "calls": int}
    if not SKIP_FLOPS:
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except Exception as e:
            print(f"[flops] FlopCounterMode unavailable ({e}); skipping.", flush=True)
            FlopCounterMode = None
        if FlopCounterMode is not None:
            restore = []  # (obj, method_name, original_callable)

            def wrap_callable(key, obj, method_name):
                orig = getattr(obj, method_name)

                def wrapped(*a, **k):
                    st = flop_stats.setdefault(key, {"gflops_per_call": 0.0, "calls": 0})
                    st["calls"] += 1
                    if st["gflops_per_call"] == 0.0:
                        try:
                            with FlopCounterMode(display=False) as fc:
                                out = orig(*a, **k)
                            st["gflops_per_call"] = fc.get_total_flops() / 1e9
                            print(f"[flops] {key}: {st['gflops_per_call']:.2f} GFLOP/call",
                                  flush=True)
                            return out
                        except Exception as ex:
                            print(f"[flops] {key}: measurement failed ({ex})", flush=True)
                            st["gflops_per_call"] = -1.0
                    return orig(*a, **k)

                setattr(obj, method_name, wrapped)
                restore.append((obj, method_name, orig))

            for attr, top in components:
                wrap_callable(attr, top, "forward")
                for meth in ("encode", "decode"):
                    if callable(getattr(top, meth, None)):
                        wrap_callable(f"{attr}.{meth}", top, meth)
            print("[flops] running one generation with per-component FLOP measurement ...",
                  flush=True)
            torch.manual_seed(SEED)
            tf = time.time()
            try:
                _ = run_gen()
            except Exception as ex:
                print(f"[flops] flops-gen raised {ex}", flush=True)
            torch.cuda.synchronize()
            print(f"[flops] done in {time.time()-tf:.1f}s", flush=True)
            for obj, method_name, orig in restore:
                setattr(obj, method_name, orig)

    # ---- report ---------------------------------------------------------------
    print("\n=== TOP-LEVEL COMPONENT LATENCY (inclusive GPU time, one generation) ===", flush=True)
    print(f"{'component':<24}{'class':<26}{'params':>10}{'calls':>7}{'incl_s':>9}{'%wall':>8}"
          f"{'GFLOP/call':>12}{'totalTFLOP':>12}", flush=True)
    total_tflop = 0.0
    for c in comp_rows:
        pct = 100.0 * c["incl_ms"] / 1000.0 / max(hooked_total_s, 1e-6)
        st = flop_stats.get(c["name"], {})
        gpc = max(st.get("gflops_per_call", 0.0), 0.0)
        fcalls = st.get("calls", c["calls"])
        tot_tflop = gpc * fcalls / 1e3
        total_tflop += tot_tflop
        c["gflops_per_call"] = gpc
        c["flop_calls"] = fcalls
        c["total_tflop"] = tot_tflop
        print(f"{c['name']:<24}{c['class']:<26}{c['params']/1e6:>9.1f}M{c['calls']:>7}"
              f"{c['incl_ms']/1000.0:>9.2f}{pct:>7.1f}%{gpc:>12.2f}{tot_tflop:>12.2f}",
              flush=True)

    extra = {k: v for k, v in flop_stats.items() if k not in {c["name"] for c in comp_rows}}
    if extra:
        print("\n=== EXTRA CALLABLES FLOPs (encode/decode etc.) ===", flush=True)
        for k, v in sorted(extra.items()):
            gpc = max(v["gflops_per_call"], 0.0)
            print(f"  {k:<28}{gpc:>10.2f} GFLOP/call x{v['calls']:<4} = {gpc*v['calls']/1e3:.2f} TFLOP",
                  flush=True)
            total_tflop += gpc * v["calls"] / 1e3

    print("\n=== TOP SELF-TIME HOTSPOTS (leaf/near-leaf GPU cost) ===", flush=True)
    for nm, cl, sm, cc in hot:
        print(f"  {sm/1000.0:>7.2f}s  x{cc:<4} {cl:<24} {nm}", flush=True)

    print(f"\nwall (hooked) generation : {hooked_total_s:.1f}s   peak VRAM {peak_vram_gb:.1f} GB",
          flush=True)
    print(f"total measured FLOPs     : {total_tflop:.2f} TFLOP", flush=True)

    out = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "checkpoint": str(CKPT_DIR),
        "prof_mode": PROF_MODE,
        "context_frames": int(CTX),
        "future_frames": int(CHUNK),
        "load_s": load_s,
        "hooked_total_s": hooked_total_s,
        "peak_vram_gb": peak_vram_gb,
        "total_tflop": total_tflop,
        "components": comp_rows,
        "flop_stats": flop_stats,
        "hotspots": [
            {"name": nm, "class": cl, "self_s": sm / 1000.0, "calls": cc}
            for nm, cl, sm, cc in hot
        ],
    }
    mpath = OUT_DIR / "vera_videogen_profile.json"
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {mpath}", flush=True)
    print("PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
