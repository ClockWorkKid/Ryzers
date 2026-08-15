#!/usr/bin/env python3
"""Curriculum merge-and-grow step for the LIBERO-plus curriculum-LoRA fine-tune.

Folds a stage-k LoRA adapter into the MolmoAct2 base and re-saves a PLAIN-base checkpoint so
stage-(k+1) can warm-start via --policy.checkpoint_path with a FRESH, higher-rank LoRA.

Why this is needed: lerobot saves a PEFT-WRAPPED checkpoint (base_model.model...lora_A/lora_B).
A fresh training launch at a different rank does a STRICT base-HF load + fresh LoRA and rejects
those PEFT keys. Merging first yields a clean plain-base key space identical to a freshly-built
base model.

This is a data-prep step, not a training-loop patch: it uses the vendored MolmoAct2Policy exactly
as training does, then the standard peft merge_and_unload() pattern.

Env (all paths are IN-CONTAINER):
  SRC      stage-k checkpoint pretrained_model dir (PEFT-wrapped)   [required]
  DST      output dir for the merged plain-base checkpoint          [required]
  HF_HOME  HF cache root (so any base snapshot resolves from cache)
"""
import os
import sys
import gc

import torch

SRC = os.environ["SRC"]
DST = os.environ["DST"]

print(f"[merge] torch {torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)
print(f"[merge] SRC={SRC}", flush=True)
print(f"[merge] DST={DST}", flush=True)

if not os.path.isfile(os.path.join(SRC, "config.json")):
    print(f"[merge] FATAL: no config.json under SRC={SRC}", flush=True)
    sys.exit(2)

from lerobot.policies.molmoact2.modeling_molmoact2 import (
    MolmoAct2Policy,
    MolmoAct2ForConditionalGeneration,
)

# from_pretrained reads the checkpoint's own config (enable_lora_vlm + lora_rank/alpha), so it
# rebuilds the PEFT wrapper at the SAME rank it was trained and strict-loads the adapter weights.
print(f"[merge] reconstructing stage-k LoRA policy from {SRC} (strict=True) ...", flush=True)
policy = MolmoAct2Policy.from_pretrained(SRC, strict=True)
policy.eval()

inner = policy.model  # peft-wrapped MolmoAct2ForConditionalGeneration
print(f"[merge] policy.model type = {type(inner).__name__}", flush=True)
if not hasattr(inner, "merge_and_unload"):
    print("[merge] FATAL: policy.model has no merge_and_unload (not a PeftModel)", flush=True)
    sys.exit(3)

print("[merge] merge_and_unload() ...", flush=True)
merged = inner.merge_and_unload()  # -> base MolmoAct2ForConditionalGeneration, plain key space
assert isinstance(merged, MolmoAct2ForConditionalGeneration), type(merged)

mad = int(getattr(merged.config, "max_action_dim", -1))
print(f"[merge] merged type={type(merged).__name__} max_action_dim={mad}", flush=True)

# Sanity: no residual PEFT keys survive the merge.
sd_keys = list(merged.state_dict().keys())
lora_left = [k for k in sd_keys if "lora_" in k or "base_model" in k]
print(f"[merge] merged state_dict: {len(sd_keys)} keys; residual lora/base_model = {len(lora_left)}",
      flush=True)
assert not lora_left, f"unexpected residual PEFT keys: {lora_left[:8]}"

# Make sure the merged config no longer advertises LoRA, so a downstream plain-base load is clean.
for attr in ("enable_lora_vlm", "lora_rank", "lora_alpha", "lora_full_model"):
    if hasattr(merged.config, attr):
        try:
            setattr(merged.config, attr, False if attr == "enable_lora_vlm" else 0)
        except Exception:
            pass

os.makedirs(DST, exist_ok=True)
print(f"[merge] save_pretrained -> {DST}", flush=True)
merged.save_pretrained(DST, safe_serialization=True)

del merged, inner, policy
gc.collect()

print("[merge] VERIFY: fresh strict base-HF load from merged dir ...", flush=True)
reloaded = MolmoAct2ForConditionalGeneration.from_pretrained(
    DST, dtype=torch.float32, low_cpu_mem_usage=True
)
print(f"[merge] VERIFY OK: reloaded {type(reloaded).__name__} "
      f"max_action_dim={int(getattr(reloaded.config, 'max_action_dim', -1))}", flush=True)

print("[merge] dir listing:", flush=True)
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f)
    print(f"   {f:44s} {os.path.getsize(p) if os.path.isfile(p) else '<dir>'}", flush=True)
print("MERGE_OK", flush=True)
