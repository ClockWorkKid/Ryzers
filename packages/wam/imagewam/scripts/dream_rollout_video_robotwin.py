# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CONTINUOUS closed-loop dream-vs-actual VIDEO for RoboTwin (one MP4 per episode).

RoboTwin analogue of dream_rollout_video_libero.py. Steps the RoboTwin (SAPIEN) sim every
control step and records the ACTUAL composed 3-cam observation (the model's compact
head-over-left|right layout) at EVERY step; re-dreams (infer_video_flux2) at EVERY replan and
holds that predicted future on the right until the next replan. Produces a smooth two-column
video [ actual sim (t) | ImageWAM dream (last replan) ] with a per-frame header.

Reuses the validated RoboTwin deploy policy (deploy_policy.get_model + _build_robotwin_image_tensor
+ _infer_action_chunk) and the sim_robotwin harness (RoboTwinScene). Needs the combined image
(ryzers build robotwin imagewam): /opt/sim + /opt/RoboTwin + the flux2 src on PYTHONPATH.
"""
import os
import sys
import json

import numpy as np
import torch

REPO = os.environ.get("IMAGEWAM_REPO", "/repos/imagewam")
FLUX2_SRC = os.environ.get("FLUX2_SRC", "/repos/flux2")
VARIANT = os.environ.get("FLUX2_VARIANT", "4b")
CKPT = os.environ.get("CKPT_PATH", f"/models/imagewam_release/robotwin/flux2_klein_4b/model.pt")
STATS = os.environ.get("DATASET_STATS_PATH", f"/models/imagewam_release/robotwin/flux2_klein_4b/dataset_stats.json")
DIT = os.environ.get("FLUX2_MODEL_PATH", "")
AE = os.environ.get("FLUX2_AE_MODEL_PATH", "")
QWEN3 = os.environ.get("FLUX2_QWEN3_MODEL_SPEC", "Qwen/Qwen3-4B")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TASK = os.environ.get("TASK", "click_bell")
TASK_CONFIG = os.environ.get("TASK_CONFIG", "demo_clean")
SEED = int(os.environ.get("SEED", "100000"))
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")
REPLAN_STEPS = int(os.environ.get("REPLAN_STEPS") or "16")
ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON") or "16")
MAX_STEPS = int(os.environ.get("MAX_STEPS") or "400")
FPS = int(os.environ.get("FPS") or "20")

for p in (REPO, os.path.join(REPO, "src"), os.path.join(FLUX2_SRC, "src"), FLUX2_SRC,
          os.path.join(REPO, "experiments", "robotwin"), "/opt/sim", "/opt/RoboTwin"):
    if p and p not in sys.path:
        sys.path.insert(0, p)


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
    import experiments.robotwin.imagewam_policy.deploy_policy as D
    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from sim_robotwin.taskenv import RoboTwinScene

    usr_args = {
        "ckpt_setting": CKPT, "dataset_stats_path": STATS, "device": "cuda", "mixed_precision": "bf16",
        "sim_task": f"robotwin_flux2_klein_{VARIANT}_base_clean_imagewam",
        "robotwin_camera_layout": os.environ.get("ROBOTWIN_CAMERA_LAYOUT", "compact_288x256"),
        "replan_steps": REPLAN_STEPS, "action_horizon": ACTION_HORIZON, "num_inference_steps": NUM_STEPS,
        "model_overrides": {
            "flux2_src_path": FLUX2_SRC, "flux2_model_path": DIT, "ae_model_path": AE,
            "variant": f"klein-base-{VARIANT}", "qwen3_model_spec": QWEN3,
            "load_text_encoder": True, "pack_proprio_after_text": True, "proprio_dim": 14,
        },
    }
    policy = D.get_model(usr_args)
    model = policy.model
    # Bake in the default-route optimizations (text-embedding cache + torch.compile[action]).
    try:
        import imagewam_opt
        imagewam_opt.patch_class()
    except Exception:  # noqa: BLE001 - optimizer must never break the demo
        pass
    print(f"policy ready     : replan={policy.replan_steps} horizon={policy.action_horizon} steps={NUM_STEPS}")

    scene = RoboTwinScene.build_stable(TASK, task_config=TASK_CONFIG, seed=SEED)
    instruction = scene.default_instruction()
    prompt = DEFAULT_PROMPT.format(task=instruction)
    print(f"task             : [{TASK}/{TASK_CONFIG}] '{instruction}'")

    def dream_from(obs):
        img = policy._build_robotwin_image_tensor(obs)
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        proprio = policy._normalize_state(state)
        with torch.no_grad():
            dre = model.infer_video_flux2(prompt=prompt, input_image=img, proprio=proprio,
                                          num_inference_steps=NUM_STEPS, seed=0)["image"]
        return _to_img(dre), _to_img(img[0])

    frames = []
    replan_psnrs = []
    step = 0
    replan = 0
    last_psnr = float("nan")
    limit = min(MAX_STEPS, scene.step_lim)
    while step < limit and not scene.success:
        obs = scene.get_obs()
        cur_dream, _actual0 = dream_from(obs)
        chunk = policy._infer_action_chunk(observation=obs, instruction=instruction)
        for a in chunk[:policy.replan_steps]:
            scene.take_action(a, action_type=getattr(policy, "action_type", "qpos"))
            step += 1
            actual = _to_img(policy._build_robotwin_image_tensor(scene.get_obs())[0])
            frames.append(_panel(actual, cur_dream,
                                 f"{TASK}  step {step:03d}  replan {replan}  PSNR {last_psnr:.1f} dB  |  {instruction[:44]}"))
            if scene.success or step >= limit:
                break
        end_actual = _to_img(policy._build_robotwin_image_tensor(scene.get_obs())[0])
        last_psnr = _psnr(cur_dream, end_actual)
        replan_psnrs.append(round(last_psnr, 2))
        print(f"  replan {replan}: dream-vs-actual PSNR={last_psnr:.2f} dB (step {step})", flush=True)
        replan += 1

    success = bool(scene.success)
    print(f"episode          : success={success}  steps={step}  frames={len(frames)}")
    vid = os.path.join(OUT_DIR, f"cl_dreamvid_robotwin_{TASK}.mp4")
    imageio.mimsave(vid, frames, fps=FPS, quality=7)
    json.dump({"task": TASK, "task_config": TASK_CONFIG, "instruction": instruction, "success": success,
               "steps": step, "frames": len(frames), "fps": FPS, "replan_steps": REPLAN_STEPS,
               "dream_vs_actual_psnr_db_per_replan": replan_psnrs},
              open(os.path.join(OUT_DIR, f"cl_dreamvid_robotwin_{TASK}.json"), "w"), indent=2)
    try:
        scene.close()
    except Exception:
        pass
    print(f"PASS: continuous closed-loop dream video -> {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
