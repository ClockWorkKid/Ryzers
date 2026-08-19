# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FlowWAM flow imagination: render the model's imagined future as RGB alongside its
PREDICTED optical-flow field, side-by-side, from a RoboTwin first-frame observation.

FlowWAM's unique output is a dense optical-flow stream: the dual-stream Wan DiT jointly
denoises a future RGB video AND a flow field (encoded as RGB by the reversible flow codec).
This runs that same dual-stream stage-1 generation (verbatim from the closed-loop
flow_action_server, minus the action expert) and writes ``imagination_robotwin_flow.gif`` =
imagined RGB (T-shape head|left|right) next to the imagined flow-color field.

Two phases avoid mixing SAPIEN (Vulkan) and torch (HIP) init in one process:
  1. ``--render-frame``: a sim-only child builds a RoboTwin scene and dumps the first-frame
     head/left/right RGB + instruction to an .npz.
  2. main: loads that .npz, builds the dual-stream pipeline, denoises, decodes, composes.

Env: TASK, TASK_CONFIG, SEED, TAG, CKPT, LOCAL_MODEL_PATH (/models/flowwam), OUT_DIR,
VIDEO_INFERENCE_STEPS (25), NUM_VIDEO_FRAMES (9), SIGMA_SHIFT (5.0).
"""
import os
import subprocess
import sys


def _env(name, default=""):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def render_first_frame(out_npz, task, task_config, seed):
    """Sim-only child: build a RoboTwin scene and save the first-frame observation."""
    import numpy as np
    from sim_robotwin.taskenv import RoboTwinScene

    scene = RoboTwinScene.build_stable(task, task_config=task_config, seed=int(seed))
    try:
        obs = scene.get_obs()
        instruction = scene.default_instruction()
        head = np.asarray(obs["observation"]["head_camera"]["rgb"], dtype=np.uint8)
        left = np.asarray(obs["observation"]["left_camera"]["rgb"], dtype=np.uint8)
        right = np.asarray(obs["observation"]["right_camera"]["rgb"], dtype=np.uint8)
    finally:
        scene.close()
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, head=head, left=left, right=right, instruction=str(instruction))
    print(f"[flowgen] first-frame obs saved -> {out_npz} (instruction: {instruction})", flush=True)


def _to_frames(video):
    """DiffSynth vae_output_to_video -> list of HxWx3 uint8 arrays."""
    import numpy as np
    from PIL import Image
    out = []
    for f in video:
        if isinstance(f, Image.Image):
            out.append(np.asarray(f.convert("RGB"), dtype=np.uint8))
        else:
            out.append(np.asarray(f, dtype=np.uint8))
    return out


def main():
    import numpy as np
    import torch
    from PIL import Image

    from pipeline_loader import build_pipeline
    from dataset_action_robotwin import RoboTwinActionFlowDataset, tshape_tile
    from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream

    task = _env("TASK", "beat_block_hammer")
    task_config = _env("TASK_CONFIG", "demo_clean")
    seed = int(_env("SEED", "0"))
    tag = _env("TAG", "robotwin")
    ckpt = _env("CKPT", "/models/flowwam/robotwin/flowwam_robotwin.safetensors")
    local_model_path = _env("LOCAL_MODEL_PATH", _env("FLOWWAM_MODEL_DIR", "/models/flowwam"))
    out_dir = _env("OUT_DIR", "/outputs")
    video_inference_steps = int(_env("VIDEO_INFERENCE_STEPS", "25"))
    num_video_frames = int(_env("NUM_VIDEO_FRAMES", "9"))
    sigma_shift = float(_env("SIGMA_SHIFT", "5.0"))
    size = (int(_env("SIZE_W", "320")), int(_env("SIZE_H", "256")))
    os.makedirs(out_dir, exist_ok=True)

    # ---- Phase 1: render the RoboTwin first frame in a sim-only child ----
    npz = os.path.join(out_dir, f"firstframe_{task}_{seed}.npz")
    if not os.path.exists(npz):
        print(f"[flowgen] rendering RoboTwin first frame ({task}/{task_config}, seed={seed}) ...",
              flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--render-frame",
                        npz, task, task_config, str(seed)], check=True)
    data = np.load(npz, allow_pickle=True)
    camera_frames = {
        "head_camera": np.asarray(data["head"], dtype=np.uint8),
        "left_camera": np.asarray(data["left"], dtype=np.uint8),
        "right_camera": np.asarray(data["right"], dtype=np.uint8),
    }
    instruction = str(data["instruction"])
    cameras = ["head_camera", "left_camera", "right_camera"]

    # ---- Phase 2: dual-stream video+flow denoise (stage-1 of flow_action_server) ----
    device = torch.device("cuda:0")
    print(f"[flowgen] loading dual-stream pipeline from {ckpt} ...", flush=True)
    pipe, flow_stream = build_pipeline(local_model_path=local_model_path, device=device,
                                       full_path=ckpt)
    dtype = pipe.torch_dtype
    vae_z_dim = getattr(pipe.vae, "z_dim", 16)
    w, h = size
    video_frames = num_video_frames

    head = camera_frames[cameras[0]]
    left = camera_frames[cameras[1]]
    right = camera_frames[cameras[2]]
    def _resize(a):
        return np.array(Image.fromarray(a).resize((w, h), Image.BICUBIC)) if a.shape[:2] != (h, w) else a
    tiled = tshape_tile(_resize(head), _resize(left), _resize(right))
    tiled_h, tiled_w = tiled.shape[:2]
    tiled_h, tiled_w, video_frames = pipe.check_resize_height_width(tiled_h, tiled_w, video_frames)
    tiled_pil = Image.fromarray(tiled).resize((tiled_w, tiled_h), Image.BICUBIC)
    flow_h, flow_w = tiled_h, tiled_w

    camera_prefix = RoboTwinActionFlowDataset.CAMERA_PREFIX
    pipe.load_models_to_device(["text_encoder"])
    context = pipe.prompter.encode_prompt(camera_prefix + instruction, positive=True, device=device)

    pipe.load_models_to_device(["vae"])
    upscale = pipe.vae.upsampling_factor
    T_lat = (video_frames - 1) // 4 + 1
    rgb_H_lat, rgb_W_lat = tiled_h // upscale, tiled_w // upscale
    flow_H_lat, flow_W_lat = flow_h // upscale, flow_w // upscale

    rgb_prefix = pipe.vae.encode(pipe.preprocess_video([tiled_pil]), device=device).to(dtype=dtype, device=device)
    zero_flow_pil = Image.new("RGB", (flow_w, flow_h), (255, 255, 255))
    flow_prefix = pipe.vae.encode(pipe.preprocess_video([zero_flow_pil]), device=device).to(dtype=dtype, device=device)

    rgb_noise = pipe.generate_noise((1, vae_z_dim, T_lat, rgb_H_lat, rgb_W_lat), seed=seed, rand_device="cpu").to(dtype=dtype, device=device)
    rgb_noise[:, :, :1] = rgb_prefix
    flow_noise = pipe.generate_noise((1, vae_z_dim, T_lat, flow_H_lat, flow_W_lat), seed=seed + 1, rand_device="cpu").to(dtype=dtype, device=device)
    flow_noise[:, :, :1] = flow_prefix

    rgb_latents, flow_latents = rgb_noise.clone(), flow_noise.clone()
    pipe.scheduler.set_timesteps(video_inference_steps, shift=sigma_shift)
    pipe.load_models_to_device(pipe.in_iteration_models)
    from tqdm import tqdm
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="dual-stream denoise")):
        t_tensor = timestep.unsqueeze(0).to(dtype=dtype, device=device)
        with torch.no_grad():
            rgb_pred, flow_pred = model_fn_wan_video_dual_stream(
                dit=pipe.dit, flow_stream=flow_stream, latents=rgb_latents,
                flow_latents=flow_latents, timestep=t_tensor, context=context,
                fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=False)
        rgb_latents = pipe.scheduler.step(rgb_pred, pipe.scheduler.timesteps[progress_id], rgb_latents)
        flow_latents = pipe.scheduler.step(flow_pred, pipe.scheduler.timesteps[progress_id], flow_latents)
        rgb_latents[:, :, :1] = rgb_prefix
        flow_latents[:, :, :1] = flow_prefix

    pipe.load_models_to_device(["vae"])
    rgb_video = _to_frames(pipe.vae_output_to_video(pipe.vae.decode(rgb_latents, device=device)))
    flow_video = _to_frames(pipe.vae_output_to_video(pipe.vae.decode(flow_latents, device=device)))

    # ---- Compose imagined RGB | predicted flow, write gif (+ best-effort mp4) ----
    n = min(len(rgb_video), len(flow_video))
    combined = []
    for i in range(n):
        r, f = rgb_video[i], flow_video[i]
        if f.shape[0] != r.shape[0]:
            f = np.array(Image.fromarray(f).resize((int(f.shape[1] * r.shape[0] / f.shape[0]), r.shape[0])))
        combined.append(np.concatenate([r, f], axis=1))

    mp4_path = os.path.join(out_dir, f"imagination_{tag}_flow.mp4")
    gif_path = os.path.join(out_dir, f"imagination_{tag}_flow.gif")
    # GIF via Pillow (always present); half-size to keep it light.
    pil_frames = [Image.fromarray(c).resize((c.shape[1] // 2, c.shape[0] // 2)) for c in combined]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=125, loop=0, optimize=True)
    print(f"[flowgen] wrote {gif_path}", flush=True)
    try:  # mp4 is a nice-to-have; the gif is the deliverable.
        import imageio
        imageio.mimsave(mp4_path, combined, fps=8)
        print(f"[flowgen] wrote {mp4_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[flowgen] mp4 skipped ({type(e).__name__}: {e})", flush=True)
    print(f"PASS: FlowWAM flow imagination ({n} frames, RGB | predicted-flow) -> {gif_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--render-frame":
        render_first_frame(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
