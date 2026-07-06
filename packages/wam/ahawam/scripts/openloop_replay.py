# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open-loop replay eval for AHA-WAM on Strix Halo (gfx1151).

Feeds ground-truth observations from released RoboTwin 2.0 LeRobot episodes into the
model and compares the predicted action horizon against the dataset's ground-truth
actions (in the model's normalized action space). Produces per-dimension
GT-vs-prediction overlay plots (workspace rule 2.a) and error metrics.

AHA-WAM predicts one action *chunk* (action_chunk_size steps) per action-phase call,
conditioned on a reusable video-context prefill (OVCR). To cover the full
action_horizon open-loop, we prefill the video context once from the first frame,
then roll `num_chunks = action_horizon // action_chunk_size` action chunks, feeding
the ground-truth observation (image + proprio) at each chunk boundary -- exactly the
two-phase schedule used by deploy/server/ahawam_policy, but driven from GT frames.

Reuses the upstream `RobotVideoDataset` + `AHAWAMProcessor` so the 3-cam concat,
resize/crop, [-1,1] normalization and action/state normalization match training
exactly. The only override is the text-embedding cache, which is unused here because
`infer_action` re-encodes the prompt string on the fly.

Env: AHAWAM_REPO, CONFIG_NAME(sim_robotwin), CKPT, DATASET_STATS, DATASET_DIR,
     NUM_EPISODES(6), NUM_PLOT_EPISODES(6), NUM_STEPS(10), OUT_DIR, TAG, SEED.
"""
import os
import sys
import json
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AHAWAM_REPO = os.environ.get("AHAWAM_REPO", "/repos/ahawam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_robotwin")
CKPT = os.environ["CKPT"]
DATASET_STATS = os.environ["DATASET_STATS"]
DATASET_DIR = os.environ["DATASET_DIR"]
NUM_EPISODES = int(os.environ.get("NUM_EPISODES") or "6")
NUM_PLOT_EPISODES = int(os.environ.get("NUM_PLOT_EPISODES") or "6")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", CONFIG_NAME)
SEED = int(os.environ.get("SEED") or "0")


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
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(AHAWAM_REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])


def _build_dataset(cfg):
    """Instantiate RobotVideoDataset over the downloaded episodes, eval mode."""
    from hydra.utils import instantiate
    import ahawam.datasets.lerobot.robot_video_dataset as rvd
    from ahawam.utils import misc

    # infer_action re-encodes the prompt, so the cached text context is unused: stub it.
    def _stub_text_context(self, prompt):
        return torch.zeros(self.context_len, 8), torch.ones(self.context_len, dtype=torch.bool)
    rvd.RobotVideoDataset._get_cached_text_context = _stub_text_context
    try:
        misc.get_work_dir = lambda *a, **k: "/tmp"
    except Exception:
        pass

    ds = instantiate(
        cfg.data.train,
        dataset_dirs=[DATASET_DIR],
        is_training_set=False,
        val_set_proportion=0.0,
        pretrained_norm_stats=DATASET_STATS,
        skip_padding_as_possible=False,
    )
    return ds


def _aggregate_plots(out_dir, tag, n_ep, norm_mae_mat, ep_mse, agg):
    """Aggregate visualization graphs over all replayed episodes."""
    D = norm_mae_mat.shape[1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 0.5 * D + 2), 7))
    ax1.boxplot([norm_mae_mat[:, d] for d in range(D)], showfliers=False)
    ax1.set_xticks(range(1, D + 1))
    ax1.set_xticklabels([f"a{d}" for d in range(D)], fontsize=7)
    ax1.set_ylabel("normalized MAE")
    ax1.set_title(f"{tag}: per-dim action MAE over {n_ep} episodes (open-loop)")
    ax1.grid(True, alpha=0.3)
    ax2.bar(range(D), agg, yerr=norm_mae_mat.std(axis=0), color="tab:red", alpha=0.75, capsize=3)
    ax2.set_xticks(range(D))
    ax2.set_xticklabels([f"a{d}" for d in range(D)], fontsize=7)
    ax2.set_ylabel("mean normalized MAE")
    ax2.set_title("per-dim mean +/- std")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "agg_per_dim_mae.png"), dpi=100)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(ep_mse, bins=min(30, max(5, n_ep // 3)), color="tab:blue", alpha=0.8)
    ax1.axvline(ep_mse.mean(), color="k", linestyle="--", label=f"mean={ep_mse.mean():.4f}")
    ax1.set_xlabel("per-episode normalized MSE")
    ax1.set_ylabel("count")
    ax1.set_title(f"{tag}: episode MSE distribution (n={n_ep})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ep_mae = norm_mae_mat.mean(axis=1)
    ax2.plot(range(n_ep), ep_mae, color="tab:red", marker=".", linewidth=0.8, markersize=3)
    ax2.axhline(ep_mae.mean(), color="k", linestyle="--", label=f"mean={ep_mae.mean():.4f}")
    ax2.set_xlabel("episode index")
    ax2.set_ylabel("mean normalized MAE")
    ax2.set_title("per-episode mean MAE")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "agg_episode_error.png"), dpi=100)
    plt.close(fig)


def _rollout_horizon(model, video, proprio, prompt, action_horizon, chunk_size, freq_ratio):
    """Two-phase open-loop rollout: one video-context prefill, then num_chunks action
    chunks fed the GT observation (image + proprio) at each chunk boundary."""
    num_chunks = action_horizon // chunk_size
    input_image = video[:, 0].unsqueeze(0).to("cuda", dtype=model.torch_dtype)

    if hasattr(model, "reset_history"):
        model.reset_history()
    if hasattr(model, "_inference_state"):
        model._inference_state = None

    with torch.no_grad():
        model.infer_action(
            prompt=prompt, input_image=input_image, action_horizon=action_horizon,
            negative_prompt="", text_cfg_scale=1.0, seed=SEED, rand_device="cpu",
            tiled=False, phase="video", num_inference_steps=NUM_STEPS,
        )
        preds = []
        n_video = video.shape[1]
        for c in range(num_chunks):
            vf = min((c * chunk_size) // freq_ratio, n_video - 1)
            obs_image = video[:, vf].unsqueeze(0).to("cuda", dtype=model.torch_dtype)
            obs_proprio = proprio[c * chunk_size].unsqueeze(0).to("cuda", dtype=torch.float32)
            out = model.infer_action(
                chunk_obs_image=obs_image, chunk_proprio=obs_proprio,
                sigma_shift=None, tiled=False, phase="action", num_inference_steps=NUM_STEPS,
            )
            chunk = out["action_chunk"]
            if chunk.ndim == 3:
                chunk = chunk[0]
            preds.append(chunk.float().cpu().numpy())
    return np.concatenate(preds, axis=0)  # [num_chunks*chunk_size, D]


def main() -> int:
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"config={CONFIG_NAME}  ckpt={os.path.basename(CKPT)}  dataset_dir={DATASET_DIR}")

    if AHAWAM_REPO not in sys.path:
        sys.path.insert(0, AHAWAM_REPO)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = _compose_cfg()
    action_horizon = int(cfg.model.action_horizon)
    freq_ratio = int(cfg.data.train.action_video_freq_ratio)

    t0 = time.time()
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    chunk_size = int(model.action_chunk_size)
    print(f"model loaded: {time.time()-t0:.1f}s  proprio_dim={model.proprio_dim}  "
          f"action_horizon={action_horizon}  chunk_size={chunk_size}")

    ds = _build_dataset(cfg)
    processor = ds.lerobot_dataset.processor
    action_norm = processor.normalizer.normalizers["action"]["default"]
    starts = ds.lerobot_dataset.episode_data_index["from"].tolist()
    n_ep = min(NUM_EPISODES, len(starts))
    print(f"dataset frames={len(ds.lerobot_dataset)}  episodes={len(starts)}  replaying {n_ep}\n")

    out_dir = os.path.join(OUT_DIR, f"openloop_{TAG}")
    os.makedirs(out_dir, exist_ok=True)

    per_ep = []
    all_norm_mae = []
    for k in range(n_ep):
        idx = int(starts[k])
        sample = ds[idx]
        video = sample["video"]                      # [C, T_video, H, W] in [-1,1]
        proprio = sample["proprio"]                  # [T-1, proprio_dim] normalized
        gt = sample["action"].float().cpu().numpy()  # [T-1, D] normalized
        prompt = sample["prompt"]

        t1 = time.time()
        pr = _rollout_horizon(model, video, proprio, prompt, action_horizon, chunk_size, freq_ratio)
        latency = time.time() - t1
        T = min(gt.shape[0], pr.shape[0])
        gt, pr = gt[:T], pr[:T]

        norm_mae_dim = np.abs(pr - gt).mean(axis=0)
        norm_mse = float(((pr - gt) ** 2).mean())
        gt_raw = action_norm.backward(torch.tensor(gt)).numpy()
        pr_raw = action_norm.backward(torch.tensor(pr)).numpy()
        raw_mae_dim = np.abs(pr_raw - gt_raw).mean(axis=0)
        all_norm_mae.append(norm_mae_dim)
        per_ep.append({"episode": k, "frame_idx": idx, "norm_mse": norm_mse,
                       "norm_mae_mean": float(norm_mae_dim.mean()),
                       "raw_mae_mean": float(raw_mae_dim.mean()),
                       "latency_s": round(latency, 3), "prompt": prompt[:80]})
        print(f"ep{k:03d} idx={idx:7d}  normMSE={norm_mse:.4f}  normMAE={norm_mae_dim.mean():.4f}  "
              f"rawMAE={raw_mae_dim.mean():.4f}  {latency:.2f}s")

        if k >= NUM_PLOT_EPISODES:
            continue
        D = gt.shape[1]
        cols = 4
        rows = (D + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.0 * rows), squeeze=False)
        x = np.arange(T)
        for d in range(rows * cols):
            ax = axes[d // cols][d % cols]
            if d < D:
                ax.plot(x, gt[:, d], color="tab:blue", label="GT", linewidth=1.6)
                ax.plot(x, pr[:, d], color="tab:red", linestyle="--", label="pred", linewidth=1.4)
                ax.set_title(f"a[{d}] MAE={norm_mae_dim[d]:.3f}", fontsize=8)
                ax.tick_params(labelsize=6)
                if d == 0:
                    ax.legend(fontsize=6)
            else:
                ax.axis("off")
        fig.suptitle(f"{TAG} ep{k} (norm space)  MSE={norm_mse:.4f}\n{prompt[:70]}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(out_dir, f"ep{k:02d}.png"), dpi=90)
        plt.close(fig)

    norm_mae_mat = np.stack(all_norm_mae, axis=0)
    ep_mse = np.array([e["norm_mse"] for e in per_ep])
    agg = norm_mae_mat.mean(axis=0)
    _aggregate_plots(out_dir, TAG, n_ep, norm_mae_mat, ep_mse, agg)
    summary = {
        "tag": TAG, "config": CONFIG_NAME, "num_episodes": n_ep,
        "num_inference_steps": NUM_STEPS, "action_dim": int(agg.shape[0]),
        "action_horizon": action_horizon, "action_chunk_size": chunk_size,
        "mean_norm_mae_per_dim": [round(float(v), 4) for v in agg],
        "mean_norm_mae": round(float(agg.mean()), 4),
        "mean_norm_mse": round(float(np.mean([e["norm_mse"] for e in per_ep])), 4),
        "mean_raw_mae": round(float(np.mean([e["raw_mae_mean"] for e in per_ep])), 4),
        "mean_latency_s": round(float(np.mean([e["latency_s"] for e in per_ep])), 3),
        "episodes": per_ep,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {TAG} open-loop replay ==")
    print(f"mean normalized MAE: {summary['mean_norm_mae']:.4f}   mean normalized MSE: {summary['mean_norm_mse']:.4f}")
    print(f"mean raw-unit MAE:   {summary['mean_raw_mae']:.4f}   mean latency: {summary['mean_latency_s']:.2f}s")
    print(f"per-dim norm MAE: {summary['mean_norm_mae_per_dim']}")
    print(f"plots + summary.json -> {out_dir}")
    print("PASS: open-loop replay complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
