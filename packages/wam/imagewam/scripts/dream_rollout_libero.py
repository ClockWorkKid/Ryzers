# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""P6 dream-vs-actual visualization during a CLOSED-LOOP LIBERO rollout.

Runs one ImageWAM-controlled episode in the LIBERO MuJoCo sim and, at the first few replan
steps, records ImageWAM's single "dreamed" future frame (infer_video_flux2) next to the
ACTUAL simulator frame reached after executing that action chunk. Saves two-column panels
(current obs | actual future | dream) per rule 2.a, plus dream-vs-actual PSNR and the episode
success flag.

Reuses the upstream eval helpers (_obs_to_model_input, _predict_action_chunk) and the
sim_libero Scene, so control + preprocessing match the closed-loop success-rate run. Requires
the combined image (ryzers build libero imagewam): needs /opt/LIBERO + sim_libero on PYTHONPATH.
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
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON") or "16")
REPLAN_STEPS = int(os.environ.get("REPLAN_STEPS") or "12")
NUM_DREAMS = int(os.environ.get("NUM_DREAMS") or "4")
MAX_STEPS = int(os.environ.get("MAX_STEPS") or "220")

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


def main() -> int:
    from hydra.utils import instantiate
    import eval_libero_single as E
    from sim_libero.scene import build_scene
    from sim_libero.libero_env import get_libero_dummy_action

    cfg = _compose_cfg()
    device = "cuda"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    E._load_model_checkpoint(model, str(CKPT))
    model = model.to(device).eval()
    processor = instantiate(cfg.data.train.processor).eval()
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(STATS)))

    video_size = list(cfg.data.train.video_size)
    H, W = int(video_size[0]), int(video_size[1])
    dtype = model.torch_dtype

    scene = build_scene(SUITE, TASK_ID, seed=SEED)
    instruction = scene.description
    print(f"task             : [{SUITE}/{TASK_ID}] '{instruction}'")

    obs = scene.reset()
    for _ in range(5):  # settle
        obs, _, _, _ = scene.env.step(get_libero_dummy_action())

    def model_input(o):
        x, proprio, _ = E._obs_to_model_input(o, cfg=cfg, processor=processor,
                                              width=W, height=H, device=device, dtype=dtype)
        return x, proprio

    records = []
    step = 0
    done = False
    success = False
    while step < MAX_STEPS and not done:
        x_cur, proprio = model_input(obs)
        capture = len(records) < NUM_DREAMS
        dream_img = None
        if capture:
            with torch.no_grad():
                dre = model.infer_video_flux2(prompt=_prompt(instruction),
                                              input_image=x_cur, proprio=proprio,
                                              num_inference_steps=NUM_STEPS, seed=0)
            dre = dre["image"] if isinstance(dre, dict) else dre
            if isinstance(dre, (list, tuple)):
                dre = dre[0]
            if dre.ndim == 4:
                dre = dre[0]
            dream_img = _to_img(dre)
            cur_img = _to_img(x_cur[0])
        action, _imgs, _ = E._predict_action_chunk(
            obs, instruction, model, processor, cfg,
            action_horizon=ACTION_HORIZON, input_w=W, input_h=H, model_device=device)
        for a in action[:REPLAN_STEPS]:
            obs, _, done, info = scene.env.step(a.tolist() if hasattr(a, "tolist") else list(a))
            step += 1
            if done:
                success = True
                break
        if capture:
            x_fut, _ = model_input(obs)
            fut_img = _to_img(x_fut[0])
            records.append((cur_img, fut_img, dream_img, _psnr(dream_img, fut_img)))
            print(f"  replan {len(records)-1}: dream-vs-actual PSNR={records[-1][3]:.2f} dB (step {step})")

    print(f"episode          : success={success}  steps={step}")
    _save_panels(records, instruction, success, os.path.join(OUT_DIR, f"cl_dream_{SUITE}_task{TASK_ID}.png"))
    json.dump({"suite": SUITE, "task_id": TASK_ID, "instruction": instruction, "success": bool(success),
               "steps": step, "dream_vs_actual_psnr_db": [round(r[3], 2) for r in records]},
              open(os.path.join(OUT_DIR, f"cl_dream_{SUITE}_task{TASK_ID}.json"), "w"), indent=2)
    print("PASS: closed-loop dream visualization complete")
    return 0


def _prompt(instruction):
    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    return DEFAULT_PROMPT.format(task=instruction)


def _save_panels(records, instruction, success, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(records)
    if n == 0:
        return
    fig, ax = plt.subplots(n, 3, figsize=(15, 3.1 * n))
    if n == 1:
        ax = ax[None, :]
    for i, (cur, fut, dream, psnr) in enumerate(records):
        for j, (im, ttl) in enumerate(((cur, f"obs @replan {i}"),
                                       (fut, "actual sim future"),
                                       (dream, f"ImageWAM dream (PSNR {psnr:.1f} dB)"))):
            ax[i, j].imshow(im); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(ttl, fontsize=10)
    fig.suptitle(f"Closed-loop dream vs actual  |  {instruction}  |  success={success}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=105, bbox_inches="tight"); plt.close(fig)
    print(f"  panels saved   -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
