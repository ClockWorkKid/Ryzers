# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Phase 2b - world-model rollout video generation for Cosmos3-Nano-Policy-DROID on ROCm.

Cosmos3-Nano is a *world*-action model: alongside the action chunk it imagines the future
concat-view video (samples["vision"]), which the Wan2.2 VAE decodes to RGB. This renders a
two-column clip -- ground-truth future frames on the LEFT, the model's imagined frames on the
RIGHT (rule 2.a) -- so the generation can be inspected against reality, plus the standalone
predicted MP4.

We reuse the upstream RobolabPolicyService decode path verbatim (rule 2.1); only the ROCm SDPA
attention patch + guardrails-off shim are applied (as in the Phase-2 smoke / Phase-3 eval).

HARDWARE NOTE (gfx1151 / ROCm 7.2.2): the Wan2.2 decoder is causal-3D-conv. On this unit MIOpen's
conv3d kernels HANG the GPU in bf16/fp16 for the large video shapes and fp32 is the only path that
runs (but slow, ~minutes) -- see artifacts/cosmos3_decode_bottleneck. Hence:
  * DECODE_DTYPE=fp32 (default here) + MIOPEN_FIND_MODE=2  -> runs offline, slow.
  * the recommended FAST/stable path is the tiny Conv2D decoder (taew2_2), which sidesteps the
    broken conv3d entirely (~0.5 s); wire it via TINY_DECODER=1 once its weights are staged.
Video generation is OFF the deployed action path (visualization only), so this cost never affects
the open-loop policy latency.

Run (inside the cosmos3 image, GPU passthrough):
    DROID_ROOT=/models/cosmos3_droid/success CKPT=/models/Cosmos3-Nano-Policy-DROID \
      DECODE_DTYPE=fp32 EP=0 OUT_DIR=/outputs python /work/cosmos3_videogen.py
"""
import os
import sys
import time

import numpy as np

# dataset + obs helpers are shared with the open-loop harness (single source of truth)
from openloop_replay import CAMS, FPS, OBS_KEY, _frame_at, _open_video, _read_episodes_meta


def _resize(img: np.ndarray, hw) -> np.ndarray:
    """uint8 RGB resize with no hard dep on cv2 (falls back to PIL)."""
    try:
        import cv2
        return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.BILINEAR))


def _compose_concat(wrist, left, right) -> np.ndarray:
    """Same layout the server models: wrist on top, left|right (half-size) below."""
    hh, hw = wrist.shape[0] // 2, wrist.shape[1] // 2
    bottom = np.concatenate([_resize(left, (hh, hw)), _resize(right, (hh, hw))], axis=1)
    return np.concatenate([wrist, bottom], axis=0)


def _write_video(path: str, frames: np.ndarray, fps: float) -> bool:
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)
        return True
    except Exception as exc:
        print(f"WARN: mp4 write failed ({exc!r}); trying gif", file=sys.stderr)
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(path.replace(".mp4", ".gif"), list(frames), duration=1.0 / fps)
            return True
        except Exception as exc2:
            print(f"WARN: gif write failed too ({exc2!r})", file=sys.stderr)
            return False


def main() -> int:
    import torch

    root = os.environ.get("DROID_ROOT", "/models/cosmos3_droid/success")
    ckpt = os.environ.get("CKPT", "/models/Cosmos3-Nano-Policy-DROID")
    num_steps = int(os.environ.get("NUM_STEPS", "4"))
    ep_sel = int(os.environ.get("EP", "0"))
    q_frame = os.environ.get("Q", "")  # explicit query frame; else episode start
    dtype_s = os.environ.get("DECODE_DTYPE", "fp32").lower()
    out_dir = os.environ.get("OUT_DIR", "/outputs")
    os.makedirs(out_dir, exist_ok=True)
    decode_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_s]

    print(f"DROID_ROOT : {root}\ncheckpoint : {ckpt}\ndecode dtype: {dtype_s}  episode: {ep_sel}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        return 1
    if dtype_s != "fp32":
        print("WARN: on gfx1151/ROCm 7.2.2 the Wan2.2 conv3d decode HANGS in fp16/bf16; use fp32.", file=sys.stderr)

    # --- policy (upstream path, ROCm enablement identical to smoke/eval) -------
    from cosmos_framework.scripts import action_policy_server_robolab as srv
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    _orig = RobolabPolicyService._build_setup_args

    def _no_guardrails(self, a):
        s = _orig(self, a)
        try:
            s.guardrails = False
        except Exception:
            pass
        return s

    srv.RobolabPolicyService._build_setup_args = _no_guardrails
    import cosmos3_rocm_patches
    cosmos3_rocm_patches.apply()

    t0 = time.time()
    svc = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=ckpt, decode_video=True, num_steps=num_steps,
            deterministic_seed=True, seed=0,
        )
    )
    print(f"policy ready: load={time.time() - t0:.1f}s", flush=True)

    # Best-effort: run the Wan2.2 VAE decoder in fp32 (the only conv3d path that does not hang the
    # GPU on gfx1151 / ROCm 7.2.2). We only touch the tokenizer/VAE used by model.decode, not the
    # action-path net. If the attribute layout differs across releases this is a no-op + a warning.
    if decode_dtype == torch.float32:
        for attr in ("tokenizer", "vae", "video_vae"):
            tok = getattr(svc.model, attr, None)
            if tok is not None and hasattr(tok, "to"):
                try:
                    tok.to(torch.float32)
                    print(f"decoder '{attr}' -> fp32 for stable conv3d decode", flush=True)
                except Exception as exc:
                    print(f"WARN: could not cast '{attr}' to fp32 ({exc!r})", file=sys.stderr)

    # --- pick episode + query frame -------------------------------------------
    recs = [r for r in _read_episodes_meta(root) if r["all_cams_local"]]
    recs.sort(key=lambda r: r["episode_index"])
    rec = next((r for r in recs if r["episode_index"] == ep_sel), recs[0])
    ep = rec["episode_index"]
    t = int(q_frame) if q_frame else 0
    print(f"episode {ep} (len={rec['length']})  query frame t={t}", flush=True)

    conts = {name: _open_video(root, key) for name, key in CAMS.items()}
    try:
        raw = {}
        obs = {"prompt": "complete the manipulation task"}
        for name in CAMS:
            ts = rec[f"{name}_from_ts"] + float(t) / FPS
            f = _frame_at(conts[name], ts)
            raw[name] = f
            obs[OBS_KEY[name]] = f
        obs["observation/joint_position"] = np.zeros((1, 7), dtype=np.float32)
        obs["observation/gripper_position"] = np.float32(0.0)

        # --- imagine + decode --------------------------------------------------
        tc = time.time()
        try:
            out = svc.infer(obs)
        except Exception as exc:
            print(f"FAIL: infer/decode raised {exc!r}", file=sys.stderr)
            return 4
        torch.cuda.synchronize()
        pred = out.get("video")
        if pred is None:
            print("FAIL: server returned no 'video' (decode_video path).", file=sys.stderr)
            return 5
        pred = np.asarray(pred)  # [T,H,W,3] uint8
        Tn, Hn, Wn = pred.shape[0], pred.shape[1], pred.shape[2]
        print(f"decoded imagined video {pred.shape} in {time.time() - tc:.1f}s", flush=True)

        # --- ground-truth future concat views (LEFT) --------------------------
        gt_frames = []
        for k in range(Tn):
            tk = t + k
            fr = {}
            ok = True
            for name in CAMS:
                ts = rec[f"{name}_from_ts"] + float(tk) / FPS
                try:
                    fr[name] = _frame_at(conts[name], ts)
                except Exception:
                    ok = False
                    break
            if not ok:
                break
            gt_frames.append(_resize(_compose_concat(fr["wrist"], fr["left"], fr["right"]), (Hn, Wn)))
        gt = np.stack(gt_frames) if gt_frames else np.zeros_like(pred[:1])
    finally:
        for c in conts.values():
            c.close()

    # --- two-column GT | pred montage (rule 2.a) ------------------------------
    m = min(len(gt), Tn)
    sep = np.full((m, Hn, 6, 3), 255, dtype=np.uint8)
    montage = np.concatenate([gt[:m], sep, pred[:m]], axis=2)  # [T,H,2W+6,3]

    base = os.path.join(out_dir, f"cosmos3_rollout_ep{ep}_t{t}")
    fps_out = float(getattr(svc.cfg, "conditioning_fps", FPS))
    ok1 = _write_video(base + "_gt_vs_pred.mp4", montage, fps_out)
    ok2 = _write_video(base + "_pred.mp4", pred[:m], fps_out)
    # small gif for laptop artifacts (rule 4): downscale + subsample
    thumb = montage[:: max(1, m // 24)]
    thumb = np.stack([_resize(f, (max(1, thumb.shape[1] // 2), max(1, thumb.shape[2] // 2))) for f in thumb])
    _write_video(base + "_gt_vs_pred.gif", thumb, min(fps_out, 8.0))

    print(f"wrote {base}_gt_vs_pred.mp4 ({'ok' if ok1 else 'FAILED'})", flush=True)
    print(f"wrote {base}_pred.mp4 ({'ok' if ok2 else 'FAILED'})", flush=True)
    print("VIDEOGEN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
