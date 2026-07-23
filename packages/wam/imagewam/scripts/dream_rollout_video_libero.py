# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CONTINUOUS closed-loop dream-vs-actual VIDEO for LIBERO (one MP4 per episode).

Improves on dream_rollout_libero.py (which only stitched a few still panels at replan points,
hence the "large frame interval" look). Here we:
  * step the LIBERO MuJoCo sim every control step and record the ACTUAL composed observation
    (agentview | wrist) at EVERY step  -> the left column is continuous simulator output;
  * re-dream (infer_video_flux2) at EVERY replan and hold that predicted future on the right
    column until the next replan.
Result: a smooth two-column video [ actual sim (t) | ImageWAM dream (last replan) ] at the
sim's own step rate, with a per-frame header (step / replan# / dream-vs-actual PSNR).

Reuses the upstream eval helpers (_obs_to_model_input, _predict_action_chunk) so control +
preprocessing match the closed-loop success-rate run. Needs the combined image
(ryzers build libero imagewam): /opt/LIBERO + sim_libero on PYTHONPATH.
"""
import os
import sys
import json

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
CKPT = os.environ.get("CKPT_PATH", "/models/imagewam_release/libero/flux2_klein_4b/model.pt")
STATS = os.environ.get("DATASET_STATS_PATH", "/models/imagewam_release/libero/flux2_klein_4b/dataset_stats.json")
DIT = os.environ.get("FLUX2_MODEL_PATH", "")
AE = os.environ.get("FLUX2_AE_MODEL_PATH", "")
QWEN3 = os.environ.get("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
SUITE = os.environ.get("SUITE", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID", "0"))
SEED = int(os.environ.get("SEED", "1000"))
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "15")
ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON") or "16")
REPLAN_STEPS = int(os.environ.get("REPLAN_STEPS") or "12")
MAX_STEPS = int(os.environ.get("MAX_STEPS") or "260")
FPS = int(os.environ.get("FPS") or "20")

for p in (REPO, os.path.join(FLUX2_SRC, "src"), FLUX2_SRC,
          os.path.join(REPO, "experiments", "libero"), "/opt/LIBERO"):
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
        f"ckpt={CKPT}", f"EVALUATION.dataset_stats_path={STATS}",
        f"model.flux2_src_path={FLUX2_SRC}", f"model.flux2_model_path={DIT}",
        f"model.ae_model_path={AE}", f"model.variant=klein-base-{VARIANT}",
        f"model.qwen3_model_spec={QWEN3}", "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true", "model.proprio_dim=8",
        f"EVALUATION.action_horizon={ACTION_HORIZON}", f"EVALUATION.replan_steps={REPLAN_STEPS}",
        f"EVALUATION.num_inference_steps={NUM_STEPS}",
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        return compose(config_name="sim_libero_omnigen2", overrides=overrides)


def _to_img(chw):
    a = chw.detach().to(torch.float32).cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    return ((a.clip(-1, 1) + 1.0) * 127.5).round().astype("uint8")


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _panel(actual, dream, header):
    from PIL import Image, ImageDraw
    h, w, _ = actual.shape
    pad, strip = 6, 22
    canvas = Image.new("RGB", (w * 2 + pad, h + strip), (15, 16, 18))
    canvas.paste(Image.fromarray(actual), (0, strip))
    canvas.paste(Image.fromarray(dream), (w + pad, strip))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), header, fill=(230, 230, 235))
    d.text((4, strip + 2), "actual sim (t)", fill=(200, 255, 200))
    d.text((w + pad + 4, strip + 2), "ImageWAM dream (@last replan)", fill=(200, 220, 255))
    return np.asarray(canvas)


def main() -> int:
    import imageio.v2 as imageio
    from hydra.utils import instantiate
    import eval_libero_single as E
    from sim_libero.scene import build_scene
    from sim_libero.libero_env import get_libero_dummy_action
    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    cfg = _compose_cfg()
    device = "cuda"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    E._load_model_checkpoint(model, str(CKPT))
    model = model.to(device).eval()
    # Bake in the default-route optimizations (text-embedding cache + torch.compile[action]).
    try:
        import imagewam_opt
        imagewam_opt.patch_class()
    except Exception:  # noqa: BLE001 - optimizer must never break the demo
        pass
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(STATS)))

    video_size = list(cfg.data.train.video_size)
    H, W = int(video_size[0]), int(video_size[1])
    dtype = model.torch_dtype

    scene = build_scene(SUITE, TASK_ID, seed=SEED)
    instruction = scene.description
    prompt = DEFAULT_PROMPT.format(task=instruction)
    print(f"task             : [{SUITE}/{TASK_ID}] '{instruction}'")

    obs = scene.reset()
    for _ in range(5):
        obs, _, _, _ = scene.env.step(get_libero_dummy_action())

    def composed(o):
        x, proprio, _ = E._obs_to_model_input(o, cfg=cfg, processor=processor,
                                               width=W, height=H, device=device, dtype=dtype)
        return x, proprio

    frames = []
    replan_psnrs = []
    step = 0
    replan = 0
    done = False
    success = False
    cur_dream = None
    last_psnr = float("nan")
    while step < MAX_STEPS and not done:
        x_cur, proprio = composed(obs)
        with torch.no_grad():
            dre = model.infer_video_flux2(prompt=prompt, input_image=x_cur, proprio=proprio,
                                          num_inference_steps=NUM_STEPS, seed=0)["image"]
        cur_dream = _to_img(dre)
        action, _imgs, _ = E._predict_action_chunk(
            obs, instruction, model, processor, cfg,
            action_horizon=ACTION_HORIZON, input_w=W, input_h=H, model_device=device)
        # execute the chunk one sim step at a time, recording every actual frame
        chunk_first_actual = None
        for a in action[:REPLAN_STEPS]:
            obs, _, done, info = scene.env.step(a.tolist() if hasattr(a, "tolist") else list(a))
            step += 1
            x_step, _ = composed(obs)
            actual = _to_img(x_step[0])
            if chunk_first_actual is None:
                chunk_first_actual = actual
            frames.append(_panel(actual, cur_dream,
                                 f"{SUITE}/task{TASK_ID}  step {step:03d}  replan {replan}  "
                                 f"PSNR {last_psnr:.1f} dB  |  {instruction[:52]}"))
            if done:
                success = True
                break
        # PSNR of the dream (made at replan start) vs the actual frame reached after the chunk
        end_actual = _to_img(composed(obs)[0][0])
        last_psnr = _psnr(cur_dream, end_actual)
        replan_psnrs.append(round(last_psnr, 2))
        print(f"  replan {replan}: dream-vs-actual PSNR={last_psnr:.2f} dB (step {step})", flush=True)
        replan += 1

    print(f"episode          : success={success}  steps={step}  frames={len(frames)}")
    vid = os.path.join(OUT_DIR, f"cl_dreamvid_{SUITE}_task{TASK_ID}.mp4")
    imageio.mimsave(vid, frames, fps=FPS, quality=7)
    json.dump({"suite": SUITE, "task_id": TASK_ID, "instruction": instruction, "success": bool(success),
               "steps": step, "frames": len(frames), "fps": FPS, "replan_steps": REPLAN_STEPS,
               "dream_vs_actual_psnr_db_per_replan": replan_psnrs},
              open(os.path.join(OUT_DIR, f"cl_dreamvid_{SUITE}_task{TASK_ID}.json"), "w"), indent=2)
    print(f"PASS: continuous closed-loop dream video -> {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
