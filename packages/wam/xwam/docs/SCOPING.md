# X-WAM Ryzer — Scoping Notes

Scope snapshot captured during initial project scoping. Laptop-side reference only
(backup + docs; no weights/images live here per workspace rule 3). The shared laptop
root also mirrors the FastWAM port — X-WAM material is namespaced under `docs/xwam/`,
`artifacts/xwam/`, and the `wam-XWAM` branch of the `Ryzers-benchmark` clone.

## 1. Goal

Port **X-WAM** (unified 4D world-action model) into an AMD **Ryzer** package that runs
on the **Strix Halo** mini-PC (Ryzen AI Max+ 395, `gfx1151`, ROCm 7.2.2) as a **direct
PyTorch → ROCm port** — the upstream code, run on ROCm torch instead of CUDA torch, with
no module-by-module re-implementation. It is added as a slim model package on the
`benchmark` branch (models are consumers of shared `simulation/*` bases). Milestone chain:

1. Build image + **full-model smoke test** on ROCm 7.2.2 (model loads on ROCm torch; deps sign-of-life).
2. Full **forward pass with released weights** (one action chunk / one imagined future from a real observation).
3. **Open-loop evaluation** on a few dataset episodes (X-WAM-RoboCasa / X-WAM-RoboTwin).
4. **Closed-loop simulation**: RoboTwin 2.0 (existing `simulation/robotwin` base) + **RoboCasa** (new port → `simulation/robocasa`), mirroring the broker→server→client eval seam, then interactive HTTP/MJPEG demos.
5. Runtime characterization + optimization (async denoising is the headline efficiency lever).

Final deliverable: a `wam-XWAM` branch on the fork `ClockWorkKid/Ryzers` adding
`packages/wam/xwam/` (+ `packages/simulation/robocasa/`), PR upstream to `AMDResearch/Ryzers`.

## 2. Upstream X-WAM (github.com/sharinka0715/X-WAM, pinned `72cfb86b33fc5060963ef63412f16439fcfa472f`)

Paper: *Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising*
(arXiv 2604.26694). License **Apache-2.0**. HF org `sharinka0715`.

### What it is
Unified 4D WAM: takes multi-view RGB + current robot state, jointly generates future 4D
observations (**video + depth**) alongside future robot **states + actions**. Four targets
in one architecture: high-fidelity video gen, 3D spatial reconstruction, policy success,
efficient action execution. Pretrained on 5,800+ hours of robot data.

### Key features (and why they matter for us)
- **Unified 4D modeling** — video + 3D (depth) + policy in one framework.
- **Lightweight depth adaptation** — replicates the final DiT blocks as an interleaved depth
  branch (`num_extra_layers: 10`), adding spatial modeling without doubling sequence length.
- **Asynchronous Noise Sampling (ANS)** — decouples denoise budgets: **few** action denoise
  steps (`action_denoise_steps: 10`) for real-time execution, **full** steps (`sample_steps: 50`)
  for high-fidelity video. *This is the headline efficiency lever and our primary optimization target.*

### Architecture (from `configs/model/wan22_5b_sft.yaml`)
- Built on **Wan2.2-TI2V-5B**. Text encoder **UMT5-XXL** (`umt5_xxl`, bf16, `text_len 512`,
  tokenizer `google/umt5-xxl`, `models_t5_umt5-xxl-enc-bf16.pth`).
- **Wan2.2 VAE** (`Wan2.2_VAE.pth`, `vae_stride (4,16,16)`).
- Wan model DiT + **+10 extra layers** for the interleaved depth branch.
- `action_dim: 14`, `proprio_dim: 16` (dual-arm EE-pose + gripper representation).
- Flow matching (`flow_matching_num_train_timesteps 1000`, `time_shifting 5.0`).
- Inference: `sample_fps 5`, `frame_num 9`, `use_decoupled_inference/sampling: true`,
  `use_joint_distribution: true`, `use_depth: true`, cfg `cfg_list [0,2,4]`.

### Source layout (repo root)
```
modules/{attention,t5,tokenizers,vae2_2,wan_model}.py   # T5 + Wan2.2 VAE + Wan DiT
runners/xwam_runner.py                                  # runtime entrypoint
data/{robot_dataset,augmentation}.py                    # LeRobot-like JSON+mp4 dataset
scripts/{train_sft,compute_stats}.py
evaluation/{policy_broker,policy_server,robocasa_client,robotwin_client,merge_results}.py
evaluation/X-WAM/{deploy_policy}.py                     # policy plugin
configs/model/wan22_5b_sft.yaml ; configs/data/{robocasa,robotwin}.yaml
utils/{fm_solvers,fm_solvers_unipc,utils,console_logger}.py
third_party/{RoboTwin, robocasa, robosuite}             # git submodules
```

### Released assets (HuggingFace)
- Checkpoints `sharinka0715/X-WAM-checkpoints`: `wan22_5b/` (base), `pretrained/`,
  `robocasa_sft/`, `robotwin_sft/`. Wan2.2-TI2V-5B base also available from `Wan-Video/Wan2.2`.
- Datasets `sharinka0715/X-WAM-RoboCasa`, `sharinka0715/X-WAM-RoboTwin`
  (metadata.json + `data/chunk-*/episode_*.json` + `video/<cam>/…mp4` + `depth/<cam>/…mp4`).

### Dataset schema (per episode JSON)
`num_frames`, `instructions`, `observations[<cam>]{type static|dynamic, rgb_path, depth_path,
start, end, fps}`, `proprios{left/right_ee_pos, _ee_rotm, _gripper_pos}`, `actions{… , raw_actions?}`.
Cameras: `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`.
**RoboCasa uses `raw_actions`** (raw controller commands) — prefer over the decomposed action fields.

### Evaluation architecture (broker → server → client)
- `policy_broker.py` — middleware dispatching client requests to servers (`--frontend_port`, `--backend_port`).
- `policy_server.py` — loads model + inference on a GPU (`--exp_path`, `--wan_checkpoint_dir`,
  `--denoise_steps 50`, `--action_denoise_steps 10`). Multiple servers can share the broker.
- `robocasa_client.py` (24 kitchen tasks, idx 0–23) / `robotwin_client.py` (50 tasks,
  `--task_name --task_config demo_randomized`) — run the sim, send obs to the broker.
This socket seam maps cleanly onto our interactive/closed-loop demo pattern.

### Upstream env (CUDA — must be re-based to ROCm)
Python ≥3.10; `torch>=2.4` (tested 2.8.0+cu129); `numpy<1.26` (1.23.5); `diffusers>=0.31`
(0.38); `transformers 4.49–4.51.3`; **`flash-attn 2.8.3`**; `decord`, `imageio[ffmpeg]`,
`deepspeed>=0.16`, `lightning`, `omegaconf`, `einops`, `h5py`, `tyro`, `easydict`.
ROCm concerns: strip CUDA torch pins (keep base ROCm torch); flash-attn unavailable on
gfx1151 → route attention through SDPA / AOTriton (`modules/attention.py` likely has a fallback);
pin numpy to the base/sim value; decord/ffmpeg for mp4 video+depth decode.

### Benchmarks (reported)
RoboCasa (24 kitchen tasks) **79.2%** avg SR; RoboTwin 2.0 clean **89.8%** / randomized **90.7%**.

## 3. Ryzers conventions (fork ClockWorkKid/Ryzers, `benchmark` branch)

- Package layout: `packages/<category>/<name>/` with required `Dockerfile`, `config.yaml`,
  `test.py`, `README.md`; optional `scripts/`, `demos/`, `adapters/`, `patches/`, `docs/`, `assets/`.
- **Benchmark-branch model** = slim policy/model layer, ships **no simulator**; composes on the
  plain ROCm base (non-sim demos) and on shared `simulation/*` bases via a `Policy` adapter
  selected by `POLICY_FACTORY`. Simulators are the single source of truth; a model must not fork one.
- Structural template: **`packages/wam/fastwam/`** — Dockerfile (`ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`,
  git clone at pinned commit, `strip_cuda_torch.py`, `PIP_CONSTRAINT` pinning base torch+numpy,
  build-time `assert torch.version.hip`), `config.yaml` (COMMIT build arg + env knobs + volume
  mappings), `adapters/*_policy.py`, `scripts/{download_checkpoints,download_datasets,model_smoke,
  openloop_replay,strip_cuda_torch}.{sh,py}`, `demos/demo_*.sh`, `docs/UPSTREAM_PIN.commit.txt`.
- Build/run: `ryzers build xwam --name xwam`; chain on sim base: `ryzers build robotwin xwam`,
  `ryzers build robocasa xwam`. Weights/datasets fetched at runtime into a mounted HF cache (rule 8).

### Existing simulation bases (benchmark branch)
`simulation/{libero, libero-plus, robotwin, simplerenv}`. X-WAM needs **`simulation/robotwin`**
(exists — RoboTwin 2.0, SAPIEN/Vulkan) and **`simulation/robocasa`** (NEW — robosuite/MuJoCo
kitchen benchmark; the "RoboCase" in the task brief = RoboCasa).

## 4. Remote strix-halo state (as scoped, this session)

- Host `HP-Z2-Mini-G1a`, `gfx1151`, ROCm 7.2.2, ~955 GB free on `~`. **No containers running (free).**
- `~/Ryzers-benchmark` = fork clone on branch `benchmark` (origin `ClockWorkKid/Ryzers` [ssh],
  upstream `AMDResearch/Ryzers`), clean; has `packages/wam/{fastwam,ahawam}` and
  `packages/simulation/{libero,libero-plus,robotwin,simplerenv}`.
- Also present: `~/Ryzers` (upstream), `~/Ryzers-fork`, `~/Ryzers-dreamzero`, plus prior Wan2.2-on-ROCm
  work (`~/WAN-ryzers`, `~/wan22-fa-bench`, `~/patched_wan22`) — directly reusable since X-WAM's
  DiT/VAE/T5 come from Wan2.2 (same lineage as FastWAM/AHA-WAM).

## 5. Key decisions / assumptions
- New package `packages/wam/xwam/` on branch **`wam-XWAM`** cut from `benchmark`; new base
  `packages/simulation/robocasa/`. Work on remote `~/Ryzers-benchmark`; laptop mirrors for backup.
- **Direct PyTorch → ROCm port**; full-model smoke first (relaxes rule 2 as FastWAM did).
- Reuse existing `simulation/robotwin`; port RoboCasa fresh (robosuite/MuJoCo, open-source, EGL headless).
- Pin upstream `72cfb86`; bake source at that commit; weights/datasets via runtime HF download (rule 8).
- flash-attn → SDPA/AOTriton on gfx1151; pin base torch + numpy so the layer composes on plain and sim bases.
- ANS (`action_denoise_steps` vs `sample_steps`) is the primary post-milestone optimization target.
