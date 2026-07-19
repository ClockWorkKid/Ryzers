# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phase-4 optimization profile + A/B harness for the Micro-World Ryzer port (Strix Halo gfx1151).

Loads the cheap t2v (Wan2.1-T2V-1.3B) pipeline ONCE (fits full in the ~47 GB GPU-addressable pool) and sweeps a matrix
of levers in-process, so model-load cost is amortized and every A/B is apples-to-apples on the same
resident weights (rule 0.5: keep GPU usage bounded -- one small model, one process):

  * component profile  -- CUDA-event inclusive timers via wrapped forwards attribute time to
                          DiT (transformer) vs conv3d VAE decode vs T5 text-encode.
  * conv override A/B   -- torch.backends.cudnn.enabled False(override)/True(stock MIOpen); the WAN
                          conv3d VAE is the VERA-measured 8.8x hotspot on gfx1151.
  * steps sweep         -- num_inference_steps in {30,20,10}; latency + fidelity vs the 30-step ref.
  * TeaCache A/B        -- built-in TeaCache off / thr=0.10 / thr=0.20; latency + fidelity.
  * cross-attn K/V A/B  -- bit-exact cache of the (constant-context) cross-attn K/V across denoise
                          steps; latency + max|delta| vs uncached.

Fidelity is measured on the decoded pixel tensor vs the reference config (max|delta|, PSNR). Writes
mw_opt_ab.json to OUT_DIR. Env: NUM_FRAMES, SAMPLE_SIZE, SEED, GUIDANCE, PROMPT, OUT_DIR, MODELS_DIR,
CONFIG, RUN_SET (comma list of blocks to run: conv,steps,teacache,kv ; default all).
Run from /repos/micro-world so the config's relative paths resolve.
"""
import json
import math
import os
import time

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from microworld.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from microworld.models.cache_utils import get_teacache_coefficients
from microworld.pipeline import WanFunPipeline
from microworld.utils.utils import filter_kwargs
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
CFG = os.environ.get("CONFIG", "/repos/micro-world/config/wan2.1/wan_civitai.yaml")
BASE = os.path.join(MODELS_DIR, "Diffusion_Transformer/Wan2.1-T2V-1.3B")

def env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


NUM_FRAMES = int(env("NUM_FRAMES", "25"))
SS = env("SAMPLE_SIZE", "256,448")
SAMPLE_SIZE = [int(x) for x in SS.replace("x", ",").split(",")]
SEED = int(env("SEED", "43"))
GUIDANCE = float(env("GUIDANCE", "3.0"))
SHIFT = float(env("SHIFT", "3"))
PROMPT = os.environ.get("PROMPT") or (
    "A first-person view walking through a blocky Minecraft world: a sandy desert with cacti and "
    "scattered stone platforms under a clear sky, part of a sword visible in the foreground.")
NEG = "bad detailed, static, blur, messy, error, distorted, low quality, watermark, text"
RUN_SET = set((os.environ.get("RUN_SET") or "conv,steps,teacache,kv").split(","))

# ---- component CUDA-event timers (wrap instance forwards) ----
_events = {"dit": [], "vae": [], "t5": []}


def _reset_events():
    for k in _events:
        _events[k].clear()


def _wrap(obj, attr, bucket):
    orig = getattr(obj, attr)

    def timed(*a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig(*a, **k)
        e.record()
        _events[bucket].append((s, e))
        return out
    setattr(obj, attr, timed)


def _sum_ms(bucket):
    tot = 0.0
    for s, e in _events[bucket]:
        tot += s.elapsed_time(e)
    return tot / 1000.0


def load_pipeline():
    config = OmegaConf.load(CFG)
    tkw = OmegaConf.to_container(config["transformer_additional_kwargs"])
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(BASE, tkw.get("transformer_subpath", "./")),
        transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DTYPE)
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(BASE, config["vae_kwargs"].get("vae_subpath", "Wan2.1_VAE.pth")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(BASE, config["text_encoder_kwargs"].get("tokenizer_subpath")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(BASE, config["text_encoder_kwargs"].get("text_encoder_subpath")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=DTYPE).eval()
    sk = OmegaConf.to_container(config["scheduler_kwargs"])
    sk["shift"] = 1
    scheduler = FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler, sk))
    pipeline = WanFunPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer,
                              text_encoder=text_encoder, scheduler=scheduler)
    pipeline.to(device=DEVICE)
    return pipeline, config


# ---- cross-attn K/V cache (bit-exact): K/V depend only on the constant text context ----
def patch_kv_cache():
    from microworld.models import wan_transformer3d as M
    cls = M.WanT2VCrossAttention
    if getattr(cls, "_kv_patched", False):
        return
    orig = cls.forward

    def fwd(self, x, context, context_lens, dtype):
        if not getattr(self, "_kv_on", False):
            return orig(self, x, context, context_lens, dtype)
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        key = id(context)
        c = getattr(self, "_kv", None)
        if c is None or c[0] != key:
            k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
            v = self.v(context.to(dtype)).view(b, -1, n, d)
            self._kv = (key, k, v)
        else:
            _, k, v = c
        out = M.attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens).to(dtype)
        out = out.flatten(2)
        return self.o(out)

    cls.forward = fwd
    cls._kv_patched = True


def set_kv(pipeline, on):
    from microworld.models.wan_transformer3d import WanT2VCrossAttention
    for m in pipeline.transformer.modules():
        if isinstance(m, WanT2VCrossAttention):
            m._kv_on = on
            m._kv = None


def psnr(a, b):
    # Returns None for a bit-exact match (mse==0) so the report stays valid strict JSON.
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return None if mse == 0 else round(20 * math.log10(1.0) - 10 * math.log10(mse), 2)


def run_once(pipeline, coeffs, *, steps, teacache, cudnn_on, kv, label):
    torch.backends.cudnn.enabled = cudnn_on
    if teacache is None:
        pipeline.transformer.disable_teacache()
    else:
        pipeline.transformer.enable_teacache(coeffs, steps, teacache, num_skip_start_steps=5, offload=False)
    set_kv(pipeline, kv)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    _reset_events()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        sample = pipeline(PROMPT, num_frames=NUM_FRAMES, negative_prompt=NEG,
                          height=SAMPLE_SIZE[0], width=SAMPLE_SIZE[1], generator=gen,
                          guidance_scale=GUIDANCE, num_inference_steps=steps, shift=SHIFT).videos
    torch.cuda.synchronize()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    comp = {b: _sum_ms(b) for b in _events}
    rec = dict(label=label, steps=steps, teacache=teacache, cudnn_on=cudnn_on, kv=kv,
               wall_s=round(wall, 2), peak_gb=round(peak, 2),
               dit_s=round(comp["dit"], 2), vae_s=round(comp["vae"], 2), t5_s=round(comp["t5"], 2),
               dit_calls=len(_events["dit"]), vae_calls=len(_events["vae"]))
    print(f"[{label}] wall={wall:6.2f}s dit={comp['dit']:6.2f}s vae={comp['vae']:5.2f}s "
          f"t5={comp['t5']:4.2f}s peak={peak:4.1f}GB steps={steps} tc={teacache} "
          f"cudnn={cudnn_on} kv={kv} (dit_calls={len(_events['dit'])})", flush=True)
    return rec, sample.float().cpu()


def main():
    print(f"torch {torch.__version__} hip {torch.version.hip} | {torch.cuda.get_device_name(0)}", flush=True)
    print(f"frames={NUM_FRAMES} size={SAMPLE_SIZE} seed={SEED} run_set={sorted(RUN_SET)}", flush=True)
    pipeline, _ = load_pipeline()
    coeffs = get_teacache_coefficients(BASE)
    patch_kv_cache()
    _wrap(pipeline.transformer, "forward", "dit")
    _wrap(pipeline.vae, "decode", "vae")
    _wrap(pipeline.text_encoder, "forward", "t5")

    results = []

    # Warm up BOTH conv backends (gfx1151 MIOpen pays a large one-time autotune when cudnn is on).
    print("== warmup (conv override on) ==", flush=True)
    run_once(pipeline, coeffs, steps=6, teacache=None, cudnn_on=False, kv=False, label="warmup_override")

    # ---- reference: 30 steps, no teacache, conv override on, no kv (fidelity anchor) ----
    ref_rec, ref_px = run_once(pipeline, coeffs, steps=30, teacache=None, cudnn_on=False,
                               kv=False, label="ref_s30_override")
    ref_rec["fidelity_note"] = "reference"
    results.append(ref_rec)

    # ---- conv override A/B: same config but stock MIOpen conv (cudnn on) ----
    if "conv" in RUN_SET:
        print("== warmup (stock MIOpen conv, pays autotune) ==", flush=True)
        run_once(pipeline, coeffs, steps=6, teacache=None, cudnn_on=True, kv=False, label="warmup_stock")
        rec, px = run_once(pipeline, coeffs, steps=30, teacache=None, cudnn_on=True,
                           kv=False, label="conv_stock_s30")
        rec["max_abs_delta_vs_ref"] = round(torch.max(torch.abs(px - ref_px)).item(), 6)
        rec["psnr_vs_ref_db"] = psnr(px, ref_px)
        rec["vae_speedup_override_vs_stock"] = round(rec["vae_s"] / max(ref_rec["vae_s"], 1e-6), 2)
        rec["total_speedup_override_vs_stock"] = round(rec["wall_s"] / max(ref_rec["wall_s"], 1e-6), 2)
        results.append(rec)

    # ---- steps sweep (override on, no teacache) ----
    if "steps" in RUN_SET:
        for st in (20, 10):
            rec, px = run_once(pipeline, coeffs, steps=st, teacache=None, cudnn_on=False,
                               kv=False, label=f"steps_s{st}_override")
            rec["max_abs_delta_vs_ref"] = round(torch.max(torch.abs(px - ref_px)).item(), 6)
            rec["psnr_vs_ref_db"] = psnr(px, ref_px)
            rec["speedup_vs_ref"] = round(ref_rec["wall_s"] / max(rec["wall_s"], 1e-6), 2)
            results.append(rec)

    # ---- TeaCache A/B (override on, 30 steps) ----
    if "teacache" in RUN_SET:
        for thr in (0.10, 0.20):
            rec, px = run_once(pipeline, coeffs, steps=30, teacache=thr, cudnn_on=False,
                               kv=False, label=f"teacache_{thr}_s30")
            rec["max_abs_delta_vs_ref"] = round(torch.max(torch.abs(px - ref_px)).item(), 6)
            rec["psnr_vs_ref_db"] = psnr(px, ref_px)
            rec["speedup_vs_ref"] = round(ref_rec["wall_s"] / max(rec["wall_s"], 1e-6), 2)
            results.append(rec)

    # ---- cross-attn K/V cache A/B (override on, 30 steps, no teacache) -> expect bit-exact ----
    if "kv" in RUN_SET:
        rec, px = run_once(pipeline, coeffs, steps=30, teacache=None, cudnn_on=False,
                           kv=True, label="kvcache_s30")
        rec["max_abs_delta_vs_ref"] = round(torch.max(torch.abs(px - ref_px)).item(), 6)
        rec["psnr_vs_ref_db"] = psnr(px, ref_px)
        rec["dit_speedup_vs_ref"] = round(ref_rec["dit_s"] / max(rec["dit_s"], 1e-6), 2)
        rec["speedup_vs_ref"] = round(ref_rec["wall_s"] / max(rec["wall_s"], 1e-6), 2)
        results.append(rec)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  frames=NUM_FRAMES, sample_size=SAMPLE_SIZE, seed=SEED, guidance=GUIDANCE,
                  results=results)
    path = os.path.join(OUT_DIR, "mw_opt_ab.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {path}\nOPT_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
