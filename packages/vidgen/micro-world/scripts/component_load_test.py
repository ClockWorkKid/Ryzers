# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gate 2: load each Micro-World component INDEPENDENTLY with real (partial) weights before any
full-pipeline assembly (rule 2). Reports per-component param count + peak VRAM, and confirms the
14B I2V/I2W transformer fits on the iGPU (else the demos should fall back to model_cpu_offload).

Usage (inside the built image, after scripts/download_checkpoints.sh):
    MODE=t2w python /ryzers/scripts/component_load_test.py
MODE in {t2v, t2w, i2v, i2w}. Loads only the components that mode needs; each is loaded on its own,
moved to the GPU in bf16, measured, then freed before the next.
"""
import os
import sys

import torch
from omegaconf import OmegaConf

from microworld.models import (
    AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel,
    WanActionControlNetModel, WanActionAdaLNModel, CLIPModel,
)

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
CONFIG = os.environ.get("CONFIG", "/repos/micro-world/config/wan2.1/wan_civitai.yaml")
MODE = os.environ.get("MODE", "t2w").lower()
DEV = "cuda"
DT = torch.bfloat16

BASE = {
    "t2v": "Diffusion_Transformer/Wan2.1-T2V-1.3B",
    "t2w": "Diffusion_Transformer/Wan2.1-T2V-1.3B",
    "i2v": "Diffusion_Transformer/Wan2.1-I2V-14B-480P",
    "i2w": "Diffusion_Transformer/Wan2.1-I2V-14B-480P",
}[MODE]
base_dir = os.path.join(MODELS_DIR, BASE)


def _gb(nbytes):
    return nbytes / (1024 ** 3)


def _report(tag, model):
    torch.cuda.synchronize()
    n = sum(p.numel() for p in model.parameters())
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{tag:11s}] params={n/1e9:6.3f}B  vram_now={_gb(alloc):6.2f}GB  peak={_gb(peak):6.2f}GB")


def _free(model):
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> int:
    cfg = OmegaConf.load(CONFIG)
    print(f"MODE={MODE}  base={base_dir}")
    assert os.path.isdir(base_dir), f"missing base weights: {base_dir} (run download_checkpoints.sh)"
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    # ---- VAE (conv3d — the VERA optimization hotspot) ----
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_dir, cfg["vae_kwargs"].get("vae_subpath", "Wan2.1_VAE.pth")),
        additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"]),
    ).to(DT).to(DEV)
    _report("vae", vae)
    _free(vae)

    # ---- T5 text encoder ----
    te = WanT5EncoderModel.from_pretrained(
        os.path.join(base_dir, cfg["text_encoder_kwargs"].get("text_encoder_subpath")),
        additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=DT,
    ).eval().to(DEV)
    _report("t5", te)
    _free(te)

    # ---- CLIP image encoder (i2v / i2w only) ----
    if MODE in ("i2v", "i2w"):
        clip = CLIPModel.from_pretrained(
            os.path.join(base_dir, cfg["image_encoder_kwargs"].get("image_encoder_subpath")),
        ).to(DT).eval().to(DEV)
        _report("clip", clip)
        _free(clip)

    # ---- Transformer / action model (the 14B check for i2v/i2w) ----
    tkw = OmegaConf.to_container(cfg["transformer_additional_kwargs"])
    if MODE == "t2v" or MODE == "i2v":
        tf_path = os.path.join(base_dir, tkw.get("transformer_subpath", "./"))
        model = WanTransformer3DModel.from_pretrained(
            tf_path, transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DT)
    elif MODE == "t2w":
        # `or default`: TRANSFORMER_PATH is declared as ${TRANSFORMER_PATH:-} so it arrives as an
        # empty string (set, not unset) — .get(name, default) would return "" here.
        tf_path = os.environ.get("TRANSFORMER_PATH") or os.path.join(MODELS_DIR, "T2W/transformer")
        model = WanActionControlNetModel.from_pretrained(
            tf_path, transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DT)
    else:  # i2w
        tf_path = os.environ.get("TRANSFORMER_PATH") or os.path.join(MODELS_DIR, "I2W/transformer")
        model = WanActionAdaLNModel.from_pretrained(
            tf_path, transformer_additional_kwargs=tkw, low_cpu_mem_usage=True, torch_dtype=DT)
    print(f"  transformer path : {tf_path}")
    n_params = sum(p.numel() for p in model.parameters())
    total = torch.cuda.get_device_properties(0).total_memory
    fits = True
    try:
        model = model.to(DEV)
        _report("transformer", model)
        peak = torch.cuda.max_memory_allocated()
    except torch.OutOfMemoryError:
        # Expected for the 14B I2W adaln model on a 47GB iGPU: it cannot be fully resident. This is a
        # verdict, not a failure — the demos drive these with GPU_MEMORY_MODE=model_cpu_offload.
        fits = False
        peak = float("nan")
        torch.cuda.empty_cache()
        print(f"  [transformer] params={n_params/1e9:6.3f}B  OOM on full GPU load (does not fit resident)")
    _free(model)

    print(f"iGPU total VRAM  : {_gb(total):.1f} GB")
    if fits:
        verdict = "FITS model_full_load" if peak < 0.8 * total else "TIGHT -> use model_cpu_offload"
        print(f"transformer peak : {_gb(peak):.2f} GB ({verdict})")
    else:
        print(f"transformer      : {n_params/1e9:.3f}B params exceeds {_gb(total):.1f} GB "
              f"-> REQUIRES model_cpu_offload / sequential_cpu_offload")
    print(f"PASS: all {MODE} components loaded independently (transformer resident={fits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
