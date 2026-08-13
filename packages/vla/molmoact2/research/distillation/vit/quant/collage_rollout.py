"""Feature-collage closed-loop rollout for the ViT-distillation shrink story.

Drives the pristine MolmoAct2-LIBERO policy with the W4A6 QAT student ViT (the
smallest deployable "route" is the only one that actually plans), and at every
step taps FOUR vision towers on the *same* patchified frames:

    teacher (original MolmoAct2 ViT) | distilled fp32 student | PTQ W4A6 | QAT W4A6

then renders a 2x5 collage GIF per episode:

    rows: [ 3rd-person (agentview) , wrist (eye-in-hand) ]
    cols: [ raw input , teacher , distilled fp32 , PTQ W4A6 , QAT W4A6 ]

Feature maps: each tower's 729x2304 seam (crop 0) -> 27x27 grid; a PCA(3) basis
is fit on the TEACHER seams across the episode and applied to ALL towers (shared
basis + shared colour scale, per view). Near-identical colours across the four
feature columns visually prove the features barely move as the model shrinks and
is quantized to W4A6.

Design: reuse LeRobot's ``eval_main`` (env + policy + processors) unchanged;
only ``rollout`` is monkeypatched to add capture. Teacher features reuse the
policy's own ViT via the ORIGINAL ``encode_image`` captured before the QAT swap
(no second 439M backbone load).

Env:
  QUANT_VARIANT     cnn|tinyvit
  QAT_STATE         QAT W4A6 blob  (drives the policy AND the QAT column)
  PTQ_STATE         PTQ W4A6 blob  (PTQ column)
  FP32_STUDENT      distilled fp32 student ckpt (fp32 column)
  COLLAGE_DIR       output dir for GIFs
  COLLAGE_SUITE     suite name (filenames / titles)
  COLLAGE_FPS       GIF fps (default 6)
  COLLAGE_MAX_STEPS optional cap on captured steps (validation)
  COLLAGE_PCA_SAMPLES  #frames sampled to fit the PCA basis (default 24)
All other CLI args forward verbatim to lerobot-eval.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch

# ---- module-level state shared with the monkeypatched rollout -----------------
ORIG_ENCODE = None       # original MolmoAct2VisionBackbone.encode_image (teacher)
BACKBONE_REF = None      # live policy backbone instance (captured in the patch)
TOWERS = None            # {"fp32","ptq","qat"} viz encoders (lazy, on cuda)
PATCHIFIER = None        # distill.patchify.FramePatchifier (processor mode)
VARIANT = None

COL_LABELS = ["input", "teacher (MolmoAct2 ViT)", "distilled fp32",
              "PTQ W4A6", "QAT W4A6"]
ROW_LABELS = ["3rd-person", "wrist"]


# ------------------------------------------------------------------ towers ----
def _install_qat_drive():
    """Patch encode_image so the policy plans with the QAT student; stash the
    original method (teacher) and the live backbone instance."""
    global ORIG_ENCODE, VARIANT
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M
    from quant.quant_student import quantize_student_
    from quant.variants import load_encoder, resolve

    ORIG_ENCODE = M.MolmoAct2VisionBackbone.encode_image
    VARIANT = resolve(os.environ["QUANT_VARIANT"])
    blob = torch.load(os.environ["QAT_STATE"], map_location="cpu", weights_only=False)
    wbits, abits = blob.get("weight_bits", 4), blob.get("act_bits", 6)
    cache: dict = {}

    def encode_image(self, images):
        global BACKBONE_REF
        BACKBONE_REF = self  # teacher tap needs a live backbone handle
        enc = cache.get(id(self))
        if enc is None:
            enc = load_encoder(VARIANT, None).to(torch.float32).eval()
            enc = quantize_student_(enc, weight_bits=wbits, act_bits=abits).eval()
            miss, unexp = enc.load_state_dict(blob["state_dict"], strict=False)
            if miss or unexp:
                print(f"[collage] QAT load miss={len(miss)} unexp={len(unexp)}", flush=True)
            enc = enc.to("cuda" if torch.cuda.is_available() else "cpu").float().eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            cache[id(self)] = enc
            print(f"[collage] QAT W{wbits}A{abits} {VARIANT} drives backbone {id(self)}", flush=True)
        dev = next(enc.parameters()).device
        if images.dim() == 4:
            b, cr, n, p = images.shape
            seam = enc(images.reshape(b * cr, n, p).to(dev, torch.float32))
            seam = seam.reshape(b, cr, n, seam.shape[-1])
        else:
            seam = enc(images.to(dev, torch.float32))
        return seam.to(self.dtype)

    M.MolmoAct2VisionBackbone.encode_image = encode_image
    print(f"[collage] patched encode_image (QAT drive) <- {os.environ['QAT_STATE']}", flush=True)


def _build_viz_towers():
    """fp32 / PTQ / QAT student encoders for the feature columns (on cuda)."""
    from quant.quant_student import quantize_student_
    from quant.variants import load_encoder

    dev = next(BACKBONE_REF.parameters()).device
    towers: dict = {}
    towers["fp32"] = load_encoder(VARIANT, os.environ["FP32_STUDENT"]).to(dev).float().eval()
    for name, key in (("ptq", "PTQ_STATE"), ("qat", "QAT_STATE")):
        blob = torch.load(os.environ[key], map_location="cpu", weights_only=False)
        wb, ab = blob.get("weight_bits", 4), blob.get("act_bits", 6)
        e = load_encoder(VARIANT, None).to(torch.float32).eval()
        e = quantize_student_(e, weight_bits=wb, act_bits=ab).eval()
        e.load_state_dict(blob["state_dict"], strict=False)
        towers[name] = e.to(dev).float().eval()
    for t in towers.values():
        for p in t.parameters():
            p.requires_grad_(False)
    print(f"[collage] viz towers ready: {list(towers)}", flush=True)
    return towers


def _patchifier():
    global PATCHIFIER
    if PATCHIFIER is None:
        from distill.patchify import FramePatchifier
        PATCHIFIER = FramePatchifier(mode="processor")
    return PATCHIFIER


@torch.no_grad()
def _seams(img_u8: np.ndarray) -> dict:
    """Raw HxWx3 uint8 frame -> {tower: [729,2304] float32 (crop0)} on cuda."""
    patches = _patchifier()(torch.from_numpy(np.ascontiguousarray(img_u8)))  # [crops,729,588] fp32
    dev = next(BACKBONE_REF.parameters()).device

    def _crop0(s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 4:      # [b, crops, 729, dim]
            return s[0, 0].float()
        if s.dim() == 3:      # [crops, 729, dim]
            return s[0].float()
        return s.reshape(-1, s.shape[-1])[:729].float()

    tset = ORIG_ENCODE(BACKBONE_REF, patches.unsqueeze(0).to(dev, BACKBONE_REF.dtype))
    out = {"teacher": _crop0(tset)}
    px = patches.to(dev, torch.float32)
    for name, t in TOWERS.items():
        out[name] = _crop0(t(px))
    return out


# ---------------------------------------------------------------- PCA / RGB ----
def _fit_pca3(feats: torch.Tensor):
    """feats [M,2304] -> (mean[2304], comps[3,2304])."""
    mean = feats.mean(0)
    X = feats - mean
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return mean, Vh[:3]


def _project(seam: torch.Tensor, mean: torch.Tensor, comps: torch.Tensor) -> torch.Tensor:
    return (seam - mean) @ comps.T          # [729,3]


def _rgb27(proj: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> np.ndarray:
    x = ((proj - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    x = x.reshape(27, 27, 3)
    return (x * 255).round().to(torch.uint8).cpu().numpy()


# ------------------------------------------------------------------- render ----
def _panel(arr_u8: np.ndarray, size: int = 256, smooth: bool = True):
    from PIL import Image
    im = Image.fromarray(arr_u8)
    resample = Image.BILINEAR if smooth else Image.NEAREST
    return im.resize((size, size), resample)


def _flip(a: np.ndarray) -> np.ndarray:
    return a[::-1, ::-1]                     # match LiberoEnv.render orientation


def _render_gif(frames3: list, framesw: list, task_desc: str, task_id: int):
    from PIL import Image, ImageDraw
    global TOWERS
    if BACKBONE_REF is None:
        print("[collage] backbone never captured (0 policy forwards?), skipping", flush=True)
        return
    if TOWERS is None:
        TOWERS = _build_viz_towers()
    suite = os.environ.get("COLLAGE_SUITE", "libero")
    fps = float(os.environ.get("COLLAGE_FPS", "6"))
    n_pca = int(os.environ.get("COLLAGE_PCA_SAMPLES", "24"))
    outdir = os.environ["COLLAGE_DIR"]
    os.makedirs(outdir, exist_ok=True)
    T = len(frames3)
    if T == 0:
        print("[collage] no frames captured, skipping", flush=True)
        return
    views = {"3rd-person": frames3, "wrist": framesw}

    # --- fit a shared PCA basis + colour scale on TEACHER seams, per view -----
    idx = np.linspace(0, T - 1, min(n_pca, T)).round().astype(int)
    basis = {}
    for vname, flist in views.items():
        toks = []
        for i in idx:
            toks.append(_seams(flist[i])["teacher"])
        toks = torch.cat(toks, 0)                    # [~n_pca*729, 2304]
        mean, comps = _fit_pca3(toks)
        proj = _project(toks, mean, comps)
        pn = proj.detach().float().cpu().numpy()   # torch.quantile rejects fp32 on this ROCm build
        lo = torch.tensor(np.percentile(pn, 2, axis=0), device=proj.device, dtype=proj.dtype)
        hi = torch.tensor(np.percentile(pn, 98, axis=0), device=proj.device, dtype=proj.dtype)
        basis[vname] = (mean, comps, lo, hi)
    print(f"[collage] PCA fit on {len(idx)} frames/view for {suite} task{task_id}", flush=True)

    # --- layout ---------------------------------------------------------------
    S, pad = 256, 6
    head, title, lab = 22, 26, 84            # column header / title / row-label strip
    ncol = len(COL_LABELS)
    W = lab + ncol * S + (ncol + 1) * pad
    H = title + head + 2 * S + 3 * pad
    canvas_frames = []

    for t in range(T):
        canvas = Image.new("RGB", (W, H), (17, 17, 17))
        d = ImageDraw.Draw(canvas)
        succ = "SUCCESS" if t == T - 1 else f"step {t + 1}/{T}"
        d.text((6, 6), f"{os.environ['QUANT_VARIANT'].upper()} W4A6 QAT  |  {suite} task{task_id}  |  "
                       f"{task_desc}  |  {succ}", fill=(235, 235, 235))
        # column headers
        for c, cl in enumerate(COL_LABELS):
            x = lab + pad + c * (S + pad)
            d.text((x + 4, title + 4), cl, fill=(180, 210, 255))
        # rows
        for r, vname in enumerate(ROW_LABELS):
            y = title + head + pad + r * (S + pad)
            d.text((6, y + S // 2 - 6), ROW_LABELS[r], fill=(200, 200, 200))
            raw = views[vname][t]
            mean, comps, lo, hi = basis[vname]
            seams = _seams(raw)
            cols = [_flip(raw)]                       # col0: raw input
            for tower in ("teacher", "fp32", "ptq", "qat"):
                rgb = _rgb27(_project(seams[tower], mean, comps), lo, hi)   # 27x27x3
                cols.append(_flip(np.asarray(_panel(rgb))))                 # upsample then flip
            for c, arr in enumerate(cols):
                x = lab + pad + c * (S + pad)
                canvas.paste(_panel(arr) if arr.shape[0] != S else Image.fromarray(arr), (x, y))
        canvas_frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))

    path = os.path.join(outdir, f"{os.environ['QUANT_VARIANT']}_{suite}_task{task_id}.gif")
    dur = int(round(1000.0 / fps))
    canvas_frames[0].save(path, save_all=True, append_images=canvas_frames[1:],
                          duration=dur, loop=0, disposal=2, optimize=False)
    mb = os.path.getsize(path) / 1e6
    print(f"[collage] wrote {path}  ({T} frames, {mb:.1f} MB)", flush=True)


# ------------------------------------------------------------- capture loop ----
def _capture_rollout(env, policy, env_preprocessor, env_postprocessor,
                     preprocessor, postprocessor, seeds=None, **kwargs):
    """Faithful copy of lerobot rollout() for num_envs==1 with per-step capture
    of both raw views; writes one collage GIF at the end. Returns the same
    schema eval_policy expects (action/reward/success/done)."""
    global TOWERS
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    policy.reset()
    observation, info = env.reset(seed=seeds)

    frames3, framesw = [], []
    max_steps = env.call("_max_episode_steps")[0]
    cap = int(os.environ.get("COLLAGE_MAX_STEPS", "100000"))
    max_steps = min(max_steps, cap)

    all_actions, all_rewards, all_successes, all_dones = [], [], [], []
    done = np.array([False] * env.num_envs)
    step = 0
    while not np.all(done) and step < max_steps:
        px = observation["pixels"]
        frames3.append(np.asarray(px["image"][0]).copy())
        framesw.append(np.asarray(px["wrist_image"][0]).copy())

        obs = preprocess_observation(observation)
        try:
            obs["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                obs["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                obs["task"] = [""] * env.num_envs
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocessor(action)
        at = env_postprocessor({ACTION: action})
        action_np = at[ACTION].to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        if "final_info" in info and isinstance(info["final_info"], dict):
            successes = info["final_info"]["is_success"].tolist()
        elif "is_success" in info:
            iss = info["is_success"]
            successes = iss.tolist() if hasattr(iss, "tolist") else [bool(iss)] * env.num_envs
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        all_actions.append(torch.from_numpy(action_np))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1

    task_desc = env.call("task_description")[0] if hasattr(env, "call") else ""
    task_id = env.call("task_id")[0] if hasattr(env, "call") else -1
    try:
        _render_gif(frames3, framesw, str(task_desc), int(task_id))
    except Exception as exc:  # noqa: BLE001 - never let rendering kill the eval
        import traceback
        print(f"[collage] render FAILED: {exc}\n{traceback.format_exc()}", flush=True)

    return {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }


def _run_lerobot_eval():
    import lerobot.scripts.lerobot_eval as LE
    LE.rollout = _capture_rollout            # inject capture
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:
        eps = entry_points().get("console_scripts", [])
    for ep in eps:
        if ep.name == "lerobot-eval":
            return ep.load()()
    return LE.main()


if __name__ == "__main__":
    _install_qat_drive()
    _run_lerobot_eval()
