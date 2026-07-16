# VERA Closed-Loop Control + Interactive Viewer

Closed-loop control + interactive MJPEG viewer for both VERA embodiments on Strix Halo (Ryzen AI
Max+ 395, `gfx1151`, ROCm 7.2.2), using VERA's upstream server / controller / viewer unchanged
where possible (rule 2.1). Two embodiments, two viewer modes (recorded mp4 + live port-forward).

```
client  ──websocket :PORT──▶  start_vera_server (WAN planner + tracker + Jacobian IDM)
  │                                    │
  └─ steps the sim (gym-pusht /        └─▶ MJPEG viewer @ :VIS_PORT  ──▶ save_vis_video → mp4
     robosuite+MuJoCo, offscreen EGL)      (obs | dream+tracks | dream | Jacobian)
```

The image is the `.[idm,video,eval]` build (`Dockerfile`) — the planner+IDM video stack plus the
MuJoCo sim stack (robosuite / robomimic / NVlabs-mimicgen / gym-pusht) and the EGL/OSMesa GL libs
MuJoCo needs to render headless. Weights + frozen bases are fetched at runtime into the mounted
caches (rule 8).

---

## PushT — `demos/demo_pusht.sh`

Lightest embodiment (DFoT planner + Jacobian IDM, no MuJoCo — `gym-pusht`). The demo brings up the
server (`--embodiment pusht --port 8820 --vis-port 8821`), waits for the policy port, runs the
headless `scripts/closedloop_pusht.py` client (ported from `examples/pusht_dfot_stack.ipynb`), then
dumps the viewer buffer to `pusht_vis.mp4`.

```sh
ryzers run --name vera /ryzers/demos/demo_pusht.sh
```

**Result**: 3 rollouts from replay state 3664 (horizon 200, render 252, seed 42) —
**100% success, mean max-reward 100%, mean coverage 82%**. Videos: `pusht_vis.mp4` (viewer
dashboard) + per-episode `vis_env0.mp4` (dream+tracks overlay) + `policy_env0.mp4` + a summary JSON.

## MimicGen — `demos/demo_mimicgen.sh`

robosuite/MuJoCo 2-block stack (`stack_d0`) driven by the omni **WAN-1.3B** video planner + the
MimicGen **Jacobian IDM**, all inference server-side. Uses the upstream controller
`vera.controller.run_mimicgen_eval` unchanged; MuJoCo renders offscreen via **EGL** (OSMesa
software fallback wired in). Server: `--embodiment mimicgen --port 8800 --vis-port 8801
--algo-config <cotracker-patched> --sample-steps 10`.

```sh
NUM_DEMOS=1 ROLLOUT_HORIZON=50 ryzers run --name vera /ryzers/demos/demo_mimicgen.sh   # ~15 min
# defaults: NUM_DEMOS=3 ROLLOUT_HORIZON=400 (full eval; ~4.7 h — schedule accordingly)
```

**Result**: closed-loop steps end-to-end offscreen — each server inference dreams a 120-frame WAN
rollout, runs the multiview (agentview + eye-in-hand) **cotracker** motion tracker + `draw_pts_gpu`
overlay, and the Jacobian IDM emits `H=10` actions committed to the robosuite env. A short
`NUM_DEMOS=1 ROLLOUT_HORIZON=50` run completed all **5 chunks** under EGL (env stepping
`1 → 11 → 21 → 31 → 41 → 50`) and produced `mimicgen_vis.mp4` (viewer dashboard, 50 frames @
1024×254) + per-view episode videos. cotracker (`scaled_offline.pth`, ~101 MB) is pulled once via
`torch.hub` into the mounted torch cache (rule 8). A non-fatal `libGLU.so.0` EGL warning fires at
env-close (cosmetic — the offscreen render and all video dumps succeed); add `libglu1-mesa` at the
next rebuild to silence it.

---

## Interactive live viewer (both modes)

- **Recorded**: `save_vis_video.py` writes a per-run mp4 into `OUT_DIR`, pulled small to the laptop
  (rule 4).
- **Live browser** (while a run is active): forward the viewer port over SSH and open the MJPEG
  dashboard in a browser:

```sh
ssh -L 8821:localhost:8821 <host>    # PushT   → http://localhost:8821
ssh -L 8801:localhost:8801 <host>    # MimicGen → http://localhost:8801
```

The dashboard tiles the current **observation | dream+tracks | dream | Jacobian field** as the
rollout runs.

---

## Runtime characterization (gfx1151, bf16, SDPA)

MimicGen closed-loop, `--sample-steps 10`, render 128, 2 views:

| chunk | latency | note |
|---|--:|---|
| cold start (infer #1) | **~342–350 s** | one-time: cotracker `torch.hub` fetch + model loads + kernel autotune |
| warm (infer #2+) | **~142 s / chunk** | 10 env steps per chunk (WAN 120-frame dream + cotracker 2-view tracks + Jacobian IDM) |

The per-chunk cost is dominated by the WAN DiT denoise + Wan2.1 VAE conv3d decode (the same split
profiled for video-gen: DiT ≈ 67.7 %, VAE conv3d ≈ 32.2 % of wall — see `PORT_SUMMARY.md`), plus the
VGGT Jacobian IDM and the cotracker pass on the dreamed video. A `HIPBLAS_STATUS_NOT_SUPPORTED`
hipblaslt→unfused-cublas fallback (non-fatal) inflates several linear layers on gfx1151; the first
optimization lever is fewer denoise steps / shorter horizon. At ~142 s/chunk a full `3 demos × 400
steps` eval is ~4.7 h — use the short demo for a viewer clip + smoke.

---

## Project-specific port fixes (rule 2.1)

All are hardware/version/upstream-drift patches, kept minimal and reproducible in the image build
(`scripts/patch_eval_imports.py`, applied as a cheap late Docker layer since VERA is an editable
install — iterating on them never re-triggers the dependency reinstall).

1. **NVlabs mimicgen, not the PyPI stub.** The PyPI project named `mimicgen` is unrelated (no
   `mimicgen.utils`); VERA's runner needs `mimicgen.utils.robomimic_utils.create_env`, so the
   Dockerfile unpins `mimicgen==1.0.0` and installs `NVlabs/mimicgen` from source `--no-deps`
   (+ `gdown`).
2. **torchmetrics 1.4.x LPIPS import guard.** `NoTrainLpips` / `_valid_img` moved/renamed; guarded
   so the eval-only metrics never hard-fail the policy import (metrics are off the inference path).
3. **Tracker backend field-name bug.** `tracker_backend_from_cfg` read `cfg.tracker_backend`, but
   the WAN pipeline's `MotionTrackConfig` names the field `backend`, so `MotionTrackConfig(
   backend="cotracker")` was silently ignored and the motion tracker always fell back to
   **alltracker** (a vendored-only checkpoint that VERA's own `build_policy` warns gives
   wrong-direction flow on MimicGen). Patch honors `.backend`; `demo_mimicgen.sh` layers
   `tracker.backend: cotracker` onto a **copy** of the shipped `algo_config.yaml` (never edits the
   downloaded asset). cotracker is `torch.hub`-fetched at runtime — no vendored binary.
4. **cotracker visualization dtype+device.** `draw_pts_gpu` runs on `rgbs.device` and
   boolean-indexes the visibility mask (`trajs[mask, t]`). The cotracker wrapper passed float
   visibility on CPU while the video stayed on GPU → `IndexError` (float mask) then a cuda/cpu
   device mismatch. Patch thresholds visibility to a bool mask kept on the video's device and hands
   `draw_pts_gpu` the GPU tracks (a CPU float copy is used only for the color numpy).
