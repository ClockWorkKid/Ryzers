# X-WAM Ryzer — Port Summary (progress log)

Upstream: github.com/sharinka0715/X-WAM @ `72cfb86` · base: Wan2.2-TI2V-5B ·
target: AMD Strix Halo (`gfx1151`), ROCm 7.2.2 · branch `wam-XWAM` off `benchmark`.

## Phase 1 — image build + ROCm env sign-of-life ✅ (Gate 1)
- Direct PyTorch → ROCm port: base image ROCm torch preserved (X-WAM's `requirements.txt`
  comments out torch/flash-attn; installed under a torch constraint). flash-attn has no
  gfx1151 wheel → `modules/attention.py` falls back to torch SDPA (validated on-device).
- Build-time guard checks ROCm torch + heavy deps only (no GPU at build time); the full
  X-WAM import graph + attention are validated at run time by `test.py`.
- Result: `xwam:latest` builds; `test.py` PASS on `Radeon 8060S Graphics`
  (torch 2.10.0+rocm7.2.2, hip 7.2.5, transformers 4.51.3, diffusers 0.39.0).

## Phase 2 — full-model forward with released weights ✅ (Gate 2)
- Weights fetched at runtime (rule 8): `Wan-AI/Wan2.2-TI2V-5B` base (~32 GB: UMT5-XXL enc +
  Wan2.2 VAE + DiT shards + tokenizer) + `sharinka0715/X-WAM-checkpoints/robotwin_sft`
  (~37 GB DeepSpeed `mp_rank_00_model_states.pt` + `config.yaml`).
- `scripts/model_smoke.py` builds `XWAMRunner(robotwin_sft)`, loads the SFT state dict, and runs
  one `generate(..., early_stop=True)` (the fast async-denoising action path) on a synthetic
  3-view observation. The "newly initialized extra_blocks / TRAIN this model" note from
  `from_pretrained` is expected — those depth/action/proprio weights are then overwritten by
  the SFT `load_state_dict`.
- Result (denoise: video=50, action=10, decoupled): model 13.06B params, load 48.9s,
  action chunk (32, 14) finite; first-call 135s (one-time ROCm warmup), steady-state ~10.0s,
  peak 29.4 GB VRAM. HxW=256x320, views=3, proprio_dim=16, action_dim=14, action_num=4.

## Next
- Phase 3: open-loop eval on X-WAM-RoboTwin/RoboCasa episodes (GT-vs-pred action MAE).
- Phase 4a: closed-loop RoboTwin 2.0 via shared `simulation/robotwin` `Policy` adapter
  (`xwam_robotwin_policy`), bypassing the upstream zmq broker. Then 4b RoboCasa (new sim base).
- Post-milestone: ANS efficiency sweep (`action_denoise_steps` vs `sample_steps`) toward
  real-time closed-loop; the ~10s action path is the current control-latency baseline.
