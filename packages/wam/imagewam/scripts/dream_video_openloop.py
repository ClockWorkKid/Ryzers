# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open-loop per-step "dream" VIDEO for FLUX.2 ImageWAM on REAL episodes (dataset replay,
no simulator), for LIBERO or RoboTwin.

Unlike open_loop_libero.py (which reports action MAE + a few still dream panels), this walks a
single episode frame-by-frame and, at EVERY step, runs ImageWAM's image-editing world model
(infer_video_flux2) to produce its "dreamed" future observation. Each output frame is the
two-column-style panel  [ input obs (t) | GT future | ImageWAM dream ]  (rule 2.a), assembled
into a continuous MP4 so the imagined frames can be scrubbed step by step. Works for both:
  DATASET=libero   (2-cam 224x448 stack)
  DATASET=robotwin (3-cam compact layout from the dataset's video_size)

Runs entirely through ImageWAM's own dataset/processor/model + released dataset_stats (rule 2.1).
"""
import os
import sys
import json

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
DATASET = os.environ.get("DATASET", "libero").lower()          # libero | robotwin
CKPT = os.environ.get("CKPT_PATH", "")
STATS = os.environ.get("DATASET_STATS_PATH", "")
FLUX2_MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "")
FLUX2_AE_MODEL_PATH = os.environ.get("FLUX2_AE_MODEL_PATH", "")
QWEN3 = os.environ.get("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")
DATA_DIR = os.environ.get("DATA_DIR", "")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
OUT_NAME = os.environ.get("OUT_NAME", f"ol_dreamvid_{DATASET}")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "15")           # dream diffusion steps
OL_START = int(os.environ.get("OL_START") or "0")              # first dataset index (episode start)
OL_MAX_FRAMES = int(os.environ.get("OL_MAX_FRAMES") or "100")  # cap frames (one episode)
OL_STRIDE = int(os.environ.get("OL_STRIDE") or "1")
FPS = int(os.environ.get("FPS") or "10")
SEED = int(os.environ.get("SEED") or "0")

# dataset-specific defaults
if DATASET == "robotwin":
    TASK = os.environ.get("TASK_CFG", f"robotwin_flux2_klein_{VARIANT}_base_clean_imagewam")
    CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_robotwin")
    PROPRIO_DIM = int(os.environ.get("PROPRIO_DIM") or "14")
    CKPT = CKPT or f"/models/imagewam_release/robotwin/flux2_klein_{VARIANT}/model.pt"
    STATS = STATS or f"/models/imagewam_release/robotwin/flux2_klein_{VARIANT}/dataset_stats.json"
else:
    TASK = os.environ.get("TASK_CFG", f"libero_flux2_klein_{VARIANT}_base_imagewam")
    CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero_omnigen2")
    PROPRIO_DIM = int(os.environ.get("PROPRIO_DIM") or "8")
    CKPT = CKPT or f"/models/imagewam_release/libero/flux2_klein_{VARIANT}/model.pt"
    STATS = STATS or f"/models/imagewam_release/libero/flux2_klein_{VARIANT}/dataset_stats.json"

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
        f"task={TASK}", f"ckpt={CKPT}", f"EVALUATION.dataset_stats_path={STATS}",
        f"model.flux2_src_path={FLUX2_SRC}", f"model.flux2_model_path={FLUX2_MODEL_PATH}",
        f"model.ae_model_path={FLUX2_AE_MODEL_PATH}", f"model.variant=klein-base-{VARIANT}",
        f"model.qwen3_model_spec={QWEN3}", "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true", f"model.proprio_dim={PROPRIO_DIM}",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=overrides)


def _filter_model_cfg(model_cfg):
    """Drop model-config keys unsupported by the target constructor (e.g. RoboTwin's
    skip_dit_load_from_pretrain), mirroring deploy_policy._filter_model_cfg_for_target."""
    import inspect
    import importlib
    from omegaconf import OmegaConf
    md = OmegaConf.to_container(model_cfg, resolve=True)
    target = md.get("_target_")
    if not target:
        return OmegaConf.create(md)
    mod, _, attr = str(target).rpartition(".")
    fn = getattr(importlib.import_module(mod), attr)
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return OmegaConf.create(md)
    allowed = {"_target_"} | {n for n, p in sig.parameters.items()
                              if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                            inspect.Parameter.KEYWORD_ONLY)}
    dropped = sorted(set(md) - allowed)
    if dropped:
        print(f"[dream_video] dropping unsupported model keys: {dropped}", flush=True)
    return OmegaConf.create({k: v for k, v in md.items() if k in allowed})


def _to_img(chw):
    a = chw.detach().to(torch.float32).cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    return ((a.clip(-1, 1) + 1.0) * 127.5).round().astype("uint8")


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _panel(cur, gt, dream, title, psnr):
    """Compose [cur | gt | dream] with labeled header -> HxW*3 uint8 RGB."""
    from PIL import Image, ImageDraw
    h, w, _ = cur.shape
    pad = 6
    labels = ("input obs (t)", "GT future", f"ImageWAM dream (PSNR {psnr:.1f} dB)")
    strip = 22
    canvas = Image.new("RGB", (w * 3 + pad * 2, h + strip), (15, 16, 18))
    for j, im in enumerate((cur, gt, dream)):
        canvas.paste(Image.fromarray(im), (j * (w + pad), strip))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), title, fill=(230, 230, 235))
    for j, lb in enumerate(labels):
        d.text((j * (w + pad) + 4, strip + 2), lb, fill=(200, 220, 255))
    return np.asarray(canvas)


def main() -> int:
    import imageio.v2 as imageio
    from hydra.utils import instantiate
    from omegaconf import open_dict

    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}  dataset={DATASET}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = _compose_cfg()
    dcfg = cfg.data.train
    with open_dict(dcfg):
        dcfg.dataset_dirs = [DATA_DIR]
        dcfg.require_text_cache = False
        dcfg.qwen_text_cache_dir = None
        dcfg.text_embedding_cache_dir = None
        dcfg.pretrained_norm_stats = STATS
        # DETERMINISTIC replay: kill the per-sample training augmentation (random resized crop,
        # rotate, color jitter, noise) -- otherwise consecutive frames jitter and the stitched
        # video (input AND GT future) looks "shaky". See RobotVideoDataset: aug is applied only
        # when video_augmentation is set AND is_training_set=True.
        dcfg.video_augmentation = None
        if "condition_frame_augmentation" in dcfg:
            dcfg.condition_frame_augmentation = None
    ds = instantiate(dcfg, _recursive_=True)
    print(f"dataset          : {type(ds).__name__} len={len(ds)} dir={os.path.basename(DATA_DIR)}")

    model = instantiate(_filter_model_cfg(cfg.model), model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded     : params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    end = min(len(ds), OL_START + OL_MAX_FRAMES)
    idxs = list(range(OL_START, end, OL_STRIDE))
    frames = []
    psnrs = []
    prompt0 = None
    for k, i in enumerate(idxs):
        s = ds[i]
        video = s["video"]                              # [3, T, H, W]
        cur = video[:, 0]
        gt_future = video[:, -1]
        input_image = cur.unsqueeze(0).to("cuda", torch.bfloat16)
        proprio0 = s["proprio"][0:1].to("cuda", torch.bfloat16)
        prompt = s["prompt"]
        if prompt0 is None:
            prompt0 = prompt.split("instruction: ")[-1][:70]
        with torch.no_grad():
            dre = model.infer_video_flux2(prompt=prompt, input_image=input_image, proprio=proprio0,
                                          num_inference_steps=NUM_STEPS, seed=SEED)["image"]
        cur_i, gt_i, dre_i = _to_img(cur), _to_img(gt_future), _to_img(dre)
        p = _psnr(dre_i, gt_i)
        psnrs.append(p)
        frames.append(_panel(cur_i, gt_i, dre_i, f"{DATASET} open-loop  step {k:03d}  |  {prompt0}", p))
        if k % 10 == 0:
            print(f"  step {k:03d}/{len(idxs)}  dream-vs-GT PSNR={p:.2f} dB", flush=True)

    vid_path = os.path.join(OUT_DIR, f"{OUT_NAME}.mp4")
    imageio.mimsave(vid_path, frames, fps=FPS, quality=7)
    # also drop first/mid/last stills for quick inspection
    from PIL import Image
    for tag, fr in (("first", frames[0]), ("mid", frames[len(frames)//2]), ("last", frames[-1])):
        Image.fromarray(fr).save(os.path.join(OUT_DIR, f"{OUT_NAME}_{tag}.png"))
    meta = {"dataset": DATASET, "frames": len(frames), "fps": FPS, "num_inference_steps": NUM_STEPS,
            "psnr_db_mean": round(float(np.mean(psnrs)), 2),
            "psnr_db_min": round(float(np.min(psnrs)), 2), "psnr_db_max": round(float(np.max(psnrs)), 2)}
    json.dump(meta, open(os.path.join(OUT_DIR, f"{OUT_NAME}.json"), "w"), indent=2)
    print("META:", json.dumps(meta))
    print(f"PASS: open-loop dream video -> {vid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
