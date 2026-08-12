# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 2b: visualize VLA-JEPA's V-JEPA "imagination" (latent world model).

IMPORTANT: VLA-JEPA's world model is a JEPA -- it predicts future *representations*
(V-JEPA2 latents), NOT pixels. There is no pixel decoder in the upstream repo, so a
literal "imagined video" is not produced by the model. What we can show is how well
the action-conditioned predictor imagines the FUTURE LATENT STATE vs. the real future
frames' encoder latents. This mirrors exactly the world-model branch of
``VLA_JEPA.forward`` (encoder -> action-conditioned predictor -> L1 vs GT latents).

We report, over the predicted future frames:
  * cosine similarity and L1 between predicted latents and the true encoder latents
    (an "imagination accuracy" curve), and
  * a two-column latent heatmap (rule 2.a): actual future latent | imagined latent
    (PCA-projected token maps), side by side.

Env: MODEL_REPO, CKPT_REL, BASE_VLM, BASE_ENCODER, DTYPE, OUT_DIR, and the same
     GT_* / SUITE / TASK_ID / EPISODE / FLIP180 selectors as openloop_replay.py.
     T0 : start frame index in the episode (default 0).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_smoke import resolve_checkpoint, repoint_config  # noqa: E402
from openloop_replay import resolve_gt_hdf5, load_episode  # noqa: E402

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
SUITE = os.environ.get("SUITE") or "libero_object"
TASK_ID = int(os.environ.get("TASK_ID") or "0")
EPISODE = int(os.environ.get("EPISODE") or "0")
T0 = int(os.environ.get("T0") or "0")


def pca_tokenmap(latent_2d: np.ndarray, grid: int) -> np.ndarray:
    """[N_tokens, D] -> [grid, grid] scalar map via 1st principal component."""
    x = latent_2d - latent_2d.mean(0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        comp = (x @ vt[0])
    except Exception:  # noqa: BLE001
        comp = x[:, 0]
    n = grid * grid
    comp = comp[:n] if comp.shape[0] >= n else np.pad(comp, (0, n - comp.shape[0]))
    return comp.reshape(grid, grid)


def main() -> int:
    print(f"torch          : {torch.__version__} hip={torch.version.hip}")
    if not (torch.version.hip and torch.cuda.is_available()):
        print("FAIL: need a ROCm device.", file=sys.stderr)
        return 1
    print(f"device[0]      : {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = resolve_checkpoint()
    repoint_config(ckpt)
    from huggingface_hub import snapshot_download
    for repo in (os.environ.get("BASE_VLM", "Qwen/Qwen3-VL-2B-Instruct"),
                 os.environ.get("BASE_ENCODER", "facebook/vjepa2-vitl-fpc64-256")):
        snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)

    from starVLA.model.framework.base_framework import baseframework
    vla = baseframework.from_pretrained(ckpt).to("cuda:0").to(DTYPE).eval()

    num_frames = vla.config.framework.vj2_model.num_frames
    tubelet = vla.vj_encoder.config.tubelet_size
    print(f"num_frames={num_frames} tubelet={tubelet}")

    # Load episode; take num_frames consecutive frames from T0 for each view.
    primary, wrist, _state, _gt, instr = load_episode(resolve_gt_hdf5())
    T = primary.shape[0]
    if T0 + num_frames > T:
        print(f"FAIL: not enough frames (T={T}, need {T0+num_frames}).", file=sys.stderr)
        return 1
    sel = slice(T0, T0 + num_frames)
    # video: [V, T, H, W, 3]
    video = np.stack([primary[sel], wrist[sel]], axis=0)
    from PIL import Image
    first_imgs = [Image.fromarray(primary[T0]), Image.fromarray(wrist[T0])]
    print(f"instruction    : {instr}")

    # ---- Replicate VLA_JEPA.forward world-model branch (video/no-action path) ----
    bv = np.stack([video]).transpose(0, 1, 2, 5, 3, 4)  # [B,V,T,3,H,W]
    # Inference only: no autograd (the predictor tracks grad by default, which would
    # break the .numpy() calls below and waste memory).
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
            images=[first_imgs], instructions=[instr],
            prompt_replace_dict={"{actions}": vla.replace_prompt})
        aidx = torch.isin(qwen_inputs["input_ids"],
                          torch.tensor(vla.action_token_ids, device=qwen_inputs["input_ids"].device))
        aidx = aidx.nonzero(as_tuple=True)
        qout = vla.qwen_vl_interface(**qwen_inputs, output_attentions=False,
                                     output_hidden_states=True, return_dict=True)
        last_hidden = qout.hidden_states[-1]
        B, _, H = last_hidden.shape
        action_tokens = last_hidden[aidx[0], aidx[1], :].view(B, -1, H)

        Bv, V, Tv, C, Hh, Ww = bv.shape
        flat = bv.reshape(Bv * V, Tv, C, Hh, Ww)
        vids = [vla.vj_processor(videos=flat[i], return_tensors="pt")["pixel_values_videos"].to(
            vla.vj_encoder.device) for i in range(Bv * V)]
        vids = torch.cat(vids, dim=0)
        with torch.no_grad():
            emb = vla.vj_encoder.get_vision_features(pixel_values_videos=vids)
            emb = torch.cat(torch.chunk(emb, chunks=V, dim=0), dim=2)  # [B, T'*per_frame, V*D]
        Tp = Tv // tubelet
        per = emb.shape[1] // Tp
        input_states = emb[:, :per * (Tp - 1), :]
        gt_states = emb[:, per:, :]
        pred_states = vla.vj_predictor(input_states, action_tokens)

    pred = pred_states.float().cpu().numpy()[0]   # [(Tp-1)*per, Dfeat]
    gtl = gt_states.float().cpu().numpy()[0]
    n_fut = Tp - 1
    per_tok = pred.shape[0] // n_fut
    grid = int(round(per_tok ** 0.5))

    # Per-future-frame imagination accuracy (cosine + L1) in latent space.
    cos, l1 = [], []
    for k in range(n_fut):
        p = pred[k * per_tok:(k + 1) * per_tok]
        q = gtl[k * per_tok:(k + 1) * per_tok]
        cos.append(float(np.mean(np.sum(p * q, 1) /
                    (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-8))))
        l1.append(float(np.mean(np.abs(p - q))))
    print("cos per frame  : " + ", ".join(f"{c:.3f}" for c in cos))

    tag = f"{SUITE}_task{TASK_ID}_ep{EPISODE}_t{T0}"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(1, n_fut + 1), cos, "-o", color="tab:blue")
    ax[0].set_title("imagination accuracy (latent cosine)")
    ax[0].set_xlabel("future frame"); ax[0].set_ylabel("cosine sim"); ax[0].set_ylim(0, 1)
    ax[1].plot(range(1, n_fut + 1), l1, "-o", color="tab:orange")
    ax[1].set_title("latent L1 error"); ax[1].set_xlabel("future frame")
    fig.suptitle(f"VLA-JEPA latent world model imagination (ROCm gfx1151)\n{tag}  |  {instr}")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    curve_png = os.path.join(OUT_DIR, f"imagine_{tag}_curve.png")
    fig.savefig(curve_png, dpi=110); plt.close(fig)

    # Two-column latent heatmap for the last predicted frame: actual | imagined (rule 2.a).
    k = n_fut - 1
    gt_map = pca_tokenmap(gtl[k * per_tok:(k + 1) * per_tok], grid)
    im_map = pca_tokenmap(pred[k * per_tok:(k + 1) * per_tok], grid)
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.6))
    vmin = min(gt_map.min(), im_map.min()); vmax = max(gt_map.max(), im_map.max())
    ax2[0].imshow(gt_map, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2[0].set_title("actual future latent (GT)"); ax2[0].axis("off")
    ax2[1].imshow(im_map, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2[1].set_title("imagined latent (predicted)"); ax2[1].axis("off")
    fig2.suptitle(f"V-JEPA latent token map (PC1), future frame {k+1}/{n_fut}\n{tag}")
    fig2.tight_layout(rect=[0, 0, 1, 0.9])
    heat_png = os.path.join(OUT_DIR, f"imagine_{tag}_latentmap.png")
    fig2.savefig(heat_png, dpi=110); plt.close(fig2)

    np.savez(os.path.join(OUT_DIR, f"imagine_{tag}.npz"),
             cos=np.array(cos), l1=np.array(l1))
    print(f"curve          : {curve_png}")
    print(f"latent map     : {heat_png}")
    print("PASS: VLA-JEPA latent imagination viz OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
