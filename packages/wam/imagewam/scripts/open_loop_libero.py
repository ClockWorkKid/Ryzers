# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""P5 open-loop evaluation of FLUX.2 ImageWAM on REAL LIBERO episodes (dataset replay, no
simulator) on AMD Strix Halo (gfx1151).

For each sampled real frame we feed the recorded observation (current 2-cam frame + proprio +
task prompt) to the model and, WITHOUT stepping any sim, compare against the dataset's
ground truth:
  * infer_action  -> predicted 16-step action chunk vs the recorded GT action chunk
                     (per-dim + overall MAE in de-normalized action units; overlay plot).
  * infer_video_flux2 -> ImageWAM's single "dreamed" future frame vs the recorded GT future
                     frame (two-column GT-left / prediction-right, per rules 2.a/2.b).

Also does the per-rule-2 module checks on the SAME real data before the loop: Qwen3 text
encoder (encode_prompt) and the FLUX.2 autoencoder (encode->decode PSNR on a real frame).

Everything runs through ImageWAM's own dataset/processor/model (rule 2.1) so preprocessing and
normalization exactly match training. Uses the released dataset_stats for de-norm.
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
DATA_DIR = os.environ.get("LIBERO_SUITE_DIR", "/libero_data/libero_object_no_noops_lerobot")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
NUM_SAMPLES = int(os.environ.get("OL_NUM_SAMPLES") or "5")
NUM_DREAMS = int(os.environ.get("OL_NUM_DREAMS") or "3")
SEED = int(os.environ.get("SEED") or "0")

for p in (REPO, os.path.join(FLUX2_SRC, "src"), FLUX2_SRC):
    if p and p not in sys.path:
        sys.path.insert(0, p)


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
        f"task=libero_flux2_klein_{VARIANT}_base_imagewam",
        f"ckpt={CKPT}",
        f"EVALUATION.dataset_stats_path={STATS}",
        f"model.flux2_src_path={FLUX2_SRC}",
        f"model.flux2_model_path={FLUX2_MODEL_PATH}",
        f"model.ae_model_path={FLUX2_AE_MODEL_PATH}",
        f"model.variant=klein-base-{VARIANT}",
        f"model.qwen3_model_spec={QWEN3}",
        "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true",
        "model.proprio_dim=8",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        return compose(config_name="sim_libero_omnigen2", overrides=overrides)


def _to_img(chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [-1,1] -> HxWx3 uint8."""
    a = chw.detach().to(torch.float32).cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    a = ((a.clip(-1, 1) + 1.0) * 127.5).round().astype("uint8")
    return a


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * np.log10(4.0 / mse)  # range [-1,1] -> peak^2=4


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    from hydra.utils import instantiate
    from omegaconf import open_dict

    # Inlined from experiments/libero/eval_libero_single.py::_denormalize_action -- importing
    # that module would pull in the `libero` sim package, which lives only in the
    # simulation/libero base image, not this slim policy image.
    def _denormalize_action(action, processor):
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = processor.shape_meta["action"][0]["key"]
        normalizer = processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    cfg = _compose_cfg()

    # ---- dataset (real LIBERO episodes), upstream code + released norm stats ----
    dcfg = cfg.data.train
    with open_dict(dcfg):
        dcfg.dataset_dirs = [DATA_DIR]
        dcfg.require_text_cache = False
        dcfg.qwen_text_cache_dir = None
        dcfg.text_embedding_cache_dir = None
        dcfg.pretrained_norm_stats = STATS
    ds = instantiate(dcfg, _recursive_=True)
    # RobotVideoDataset stores the (normalizer-populated) processor on its inner lerobot_dataset
    # via set_processor(); fall back across plausible attribute names.
    processor = (getattr(ds, "processor", None)
                 or getattr(getattr(ds, "lerobot_dataset", None), "processor", None)
                 or getattr(getattr(ds, "lerobot_dataset", None), "_processor", None))
    if processor is None:
        raise RuntimeError("could not locate the dataset processor (with normalizer)")
    print(f"dataset          : {type(ds).__name__} len={len(ds)} suite={os.path.basename(DATA_DIR)}")

    # ---- model ----
    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded     : {time.time()-t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    idxs = np.linspace(0, len(ds) - 1, NUM_SAMPLES).astype(int).tolist()

    # ---- rule-2 per-module checks on the FIRST real sample ----
    s0 = ds[idxs[0]]
    img0 = s0["video"][:, 0].unsqueeze(0).to("cuda", torch.bfloat16)  # [1,3,H,W]
    with torch.no_grad():
        try:
            # FLUX.2 stack encodes text via _prepare_flux2_infer_text (Qwen3), not the generic
            # encode_prompt (which targets the wan/omnigen text path).
            tc, tm = model._prepare_flux2_infer_text(s0["prompt"], None, None)
            print(f"[module] text enc: {tuple(tc.shape)} mask={tuple(tm.shape)} "
                  f"finite={bool(torch.isfinite(tc).all())} ({QWEN3})")
        except Exception as e:
            print(f"[module] text enc: FAILED {type(e).__name__}: {e}")
        rec = model.vae.decode(model.vae.encode(img0)).clamp(-1, 1)
        ae_psnr = _psnr(img0.squeeze(0), rec.squeeze(0))
        print(f"[module] AE recon: PSNR={ae_psnr:.2f} dB on a REAL LIBERO frame")

    # ---- open-loop replay ----
    per_dim = []
    overall = []
    latencies = []
    overlay_pack = None
    dream_count = 0
    for n, i in enumerate(idxs):
        s = ds[i]
        video = s["video"]                       # [3,2,H,W]
        gt_future = video[:, 1]                   # [3,H,W]
        input_image = video[:, 0].unsqueeze(0).to("cuda", torch.bfloat16)
        gt_action = s["action"]                   # [T,7] normalized
        ah = int(gt_action.shape[0])
        proprio0 = s["proprio"][0:1].to("cuda", torch.bfloat16)  # [1,8]
        prompt = s["prompt"]

        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.infer_action(prompt=prompt, input_image=input_image, action_horizon=ah,
                                     proprio=proprio0, num_inference_steps=NUM_STEPS, seed=SEED)
        torch.cuda.synchronize(); latencies.append((time.time() - t) * 1000.0)
        pred = out["action"]
        if pred.ndim == 3:
            pred = pred[0]
        pred_d = _denormalize_action(pred.cpu(), processor)[0]     # [T,7]
        gt_d = _denormalize_action(gt_action, processor)[0]        # [T,7]
        ad = np.abs(pred_d - gt_d)
        per_dim.append(ad.mean(axis=0)); overall.append(float(ad.mean()))
        print(f"sample {n} (idx {i:5d}): MAE={ad.mean():.4f}  '{prompt.split('instruction: ')[-1][:44]}'")
        if overlay_pack is None:
            overlay_pack = (pred_d.copy(), gt_d.copy(), prompt)

        if dream_count < NUM_DREAMS:
            with torch.no_grad():
                dre = model.infer_video_flux2(prompt=prompt, input_image=input_image, proprio=proprio0,
                                              num_inference_steps=NUM_STEPS, seed=SEED)["image"]
            _save_dream(_to_img(video[:, 0]), _to_img(gt_future), _to_img(dre),
                        prompt, os.path.join(OUT_DIR, f"ol_dream_{n}.png"))
            dream_count += 1

    per_dim = np.stack(per_dim).mean(axis=0)
    metrics = {
        "suite": os.path.basename(DATA_DIR), "num_samples": NUM_SAMPLES, "num_steps": NUM_STEPS,
        "action_mae_overall": float(np.mean(overall)),
        "action_mae_per_dim": [float(x) for x in per_dim],
        "ae_recon_psnr_db": round(ae_psnr, 2),
        "infer_action_ms_mean": round(float(np.mean(latencies[1:] or latencies)), 1),
        "infer_action_ms_first": round(float(latencies[0]), 1),
    }
    with open(os.path.join(OUT_DIR, "ol_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _save_action_overlay(*overlay_pack, per_dim, float(np.mean(overall)),
                         os.path.join(OUT_DIR, "ol_action_overlay.png"))
    print("METRICS:", json.dumps(metrics))
    print("PASS: open-loop on real LIBERO episodes complete")
    return 0


def _save_dream(cur, gt, dream, prompt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.2))
    for a, im, ttl in zip(ax, (cur, gt, dream),
                          ("input observation (t)", "GT future frame", "ImageWAM 'dream' (pred future)")):
        a.imshow(im); a.set_title(ttl, fontsize=10); a.axis("off")
    fig.suptitle(prompt.split("instruction: ")[-1][:90], fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  dream saved    -> {path}")


def _save_action_overlay(pred_d, gt_d, prompt, per_dim, mae, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, D = gt_d.shape
    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"][:D]
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    ax = ax.ravel()
    for d in range(D):
        ax[d].plot(range(T), gt_d[:, d], "o-", label="GT", color="tab:blue", ms=3)
        ax[d].plot(range(T), pred_d[:, d], "x--", label="pred", color="tab:red", ms=4)
        ax[d].set_title(f"{names[d]}  (MAE {per_dim[d]:.3f})", fontsize=9)
        ax[d].grid(alpha=0.3)
        if d == 0:
            ax[d].legend(fontsize=8)
    for d in range(D, len(ax)):
        ax[d].axis("off")
    fig.suptitle(f"Open-loop action chunk (sample 0)  overall MAE={mae:.4f}  |  "
                 f"{prompt.split('instruction: ')[-1][:70]}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  overlay saved  -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
