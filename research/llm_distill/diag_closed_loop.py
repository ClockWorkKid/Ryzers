"""Closed-loop teacher-vs-student behavioral diagnostic.

For each of N episodes (fixed env seeds), runs the TEACHER and the STUDENT from the SAME
reset state and records, per rollout:
  * the agentview RGB frame at every env step,
  * the executed (env-space) action at every env step,
  * the per-layer LLM KV states handed to the flow-matching head at every action chunk
    (hooked at action_expert.prepare_context(encoder_kv_states=...) -- the exact tensors the
    flow head consumes; the first ~100 token positions).

Then it builds, per episode folder (ep_XX/):
  * rollout.mp4 -- teacher (left) | student (right) side-by-side, with a synced action plot
    below and a vertical bar sweeping along time.
  * actions.png -- teacher vs student executed action traces (7 dims).
  * kv_firstchunk.png -- token-aligned teacher-vs-student KV cosine (identical inputs at the
    first chunk), per layer x token, for K and V.
  * kv_overtime.png -- per-token KV magnitude over chunks (teacher vs student).
  * kv_raw.npz -- compact recorded arrays for deeper inspection.
  * summary.json -- seeds, success, action MSE, sizes.

Teacher = left, student = right (workspace two-column rule; teacher is the reference).
"""

from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio


# ------------------------------------------------------------------ KV capture
KV_CAP: list[dict] = []
NTOK = 100


def _install_kv_hook(policy):
    """Wrap action_expert.prepare_context to snapshot the encoder_kv_states it receives
    (the exact per-layer KV passed to the flow head)."""
    ae = policy._backbone()._require_action_expert()
    if getattr(ae, "_kv_hooked", False):
        return
    orig = ae.prepare_context

    def wrapped(*a, **k):
        kv = k.get("encoder_kv_states")
        if kv is None and a:
            kv = a[0]
        try:
            KV_CAP.append(_snapshot_kv(kv, full=(len(KV_CAP) == 0)))
        except Exception as e:  # never break the rollout
            if not getattr(wrapped, "_warned", False):
                print(f"[cl] WARN kv snapshot failed: {e!r}", flush=True)
                wrapped._warned = True
            KV_CAP.append({"error": repr(e)})
        return orig(*a, **k)

    ae.prepare_context = wrapped
    ae._kv_hooked = True


def _snapshot_kv(kv, full=False):
    """kv: sequence of (k, v) per layer. Each k/v is the per-token KV vector fed to the flow
    head, shape [B, seq, n_heads*head_dim] (teacher contract, e.g. [1, seq, 1024])."""
    L = len(kv)
    norms_k = np.zeros((L, NTOK), np.float32)
    norms_v = np.zeros((L, NTOK), np.float32)
    k_full = v_full = None
    seq = int(kv[0][0].shape[1])
    n = min(NTOK, seq)
    if full:
        Dk = int(kv[0][0].shape[2])
        k_full = np.zeros((L, NTOK, Dk), np.float16)
        v_full = np.zeros((L, NTOK, Dk), np.float16)
    for li, (k, v) in enumerate(kv):
        kk = k[0, :n, :].float()  # [n, D]
        vv = v[0, :n, :].float()
        norms_k[li, :n] = kk.norm(dim=-1).cpu().numpy()
        norms_v[li, :n] = vv.norm(dim=-1).cpu().numpy()
        if full:
            k_full[li, :n] = kk.cpu().numpy().astype(np.float16)
            v_full[li, :n] = vv.cpu().numpy().astype(np.float16)
    return {"norm_k": norms_k, "norm_v": norms_v, "seq": seq, "k_full": k_full, "v_full": v_full}


# ------------------------------------------------------------------ helpers
def _to_uint8_hwc(t):
    """[C,H,W] or [H,W,C] tensor/array -> uint8 HWC RGB."""
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[2]:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        if a.max() <= 1.0 + 1e-3:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return a


def _find_image_key(obs):
    cams = [k for k in obs if k.startswith("observation.images.")]
    agent = [k for k in cams if "wrist" not in k and "eye_in_hand" not in k]
    return (agent or cams or [None])[0]


# ------------------------------------------------------------------ rollout
def run_rollout(policy, env, env_pre, pre, post, env_post, preprocess_observation, seed, max_steps_cap=None):
    from lerobot.policies.molmoact2.modeling_molmoact2 import ACTION
    KV_CAP.clear()
    policy.reset()
    obs, info = env.reset(seed=[seed])
    frames, actions, chunk_step = [], [], []
    img_key = None
    done = np.array([False])
    max_steps = env.call("_max_episode_steps")[0]
    if max_steps_cap:
        max_steps = min(max_steps, max_steps_cap)
    step, success = 0, False
    while not bool(done.all()) and step < max_steps:
        pobs = preprocess_observation(obs)
        if img_key is None:
            img_key = _find_image_key(pobs)
        frames.append(_to_uint8_hwc(pobs[img_key][0]))
        try:
            pobs["task"] = list(env.call("task_description"))
        except Exception:
            pobs["task"] = [""]
        pobs = env_pre(pobs)
        pobs = pre(pobs)
        n_before = len(KV_CAP)
        with torch.inference_mode():
            action = policy.select_action(pobs)
        if len(KV_CAP) > n_before:
            chunk_step.append(step)
        action = post(action)
        tr = env_post({ACTION: action})
        action = tr[ACTION]
        anp = action.to("cpu").numpy()
        actions.append(anp[0].copy())
        obs, reward, terminated, truncated, info = env.step(anp)
        if "final_info" in info and isinstance(info["final_info"], dict):
            success = success or bool(info["final_info"]["is_success"][0])
        elif "is_success" in info:
            iv = info["is_success"]
            success = success or bool(iv[0] if hasattr(iv, "__len__") else iv)
        done = terminated | truncated | done
        step += 1
    return {
        "frames": frames,
        "actions": np.asarray(actions, np.float32),   # [T, adim]
        "kv": list(KV_CAP),
        "chunk_step": chunk_step,
        "success": bool(success),
        "steps": step,
    }


# ------------------------------------------------------------------ artifacts
def build_action_plot(t_act, s_act, adim_names):
    """Render the static teacher/student action figure once; return (rgb_img, x_of_step fn)."""
    T = max(len(t_act), len(s_act))
    nd = t_act.shape[1]
    fig, axes = plt.subplots(nd, 1, sharex=True, figsize=(7.5, 6.2), dpi=110)
    if nd == 1:
        axes = [axes]
    xs = np.arange(T)
    for d in range(nd):
        ax = axes[d]
        ax.plot(np.arange(len(t_act)), t_act[:, d], color="#1f77b4", lw=1.6, label="teacher")
        ax.plot(np.arange(len(s_act)), s_act[:, d], color="#d62728", lw=1.4, ls="--", label="student")
        ax.set_ylabel(adim_names[d], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max(1, T - 1))
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("env step", fontsize=9)
    fig.suptitle("executed action (env space): teacher vs student", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    ax0 = axes[0]
    x0 = ax0.transData.transform((0, 0))[0]
    x1 = ax0.transData.transform((max(1, T - 1), 0))[0]

    def x_of_step(t):
        frac = 0.0 if T <= 1 else t / (T - 1)
        return int(round(x0 + frac * (x1 - x0)))

    plt.close(fig)
    return buf, x_of_step, T


def make_video(path, teacher, student, plot_img, x_of_step, fps=20):
    tf, sf = teacher["frames"], student["frames"]
    T = max(len(tf), len(sf))
    def grab(fr, i):
        return fr[min(i, len(fr) - 1)]
    H = 300
    def tile(fr, label, ok, step_i, total):
        img = grab(fr, step_i)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * H / h), H))
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        clr = (0, 200, 0) if ok else (255, 80, 80)
        tag = "SUCCESS" if (ok and step_i >= min(len(fr), total) - 1) else f"t={step_i}"
        cv2.putText(img, tag, (img.shape[1] - 120, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
        return img
    plot_h = 360
    pw = plot_img.shape[1]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=6,
                            macro_block_size=None) as wr:
        for i in range(T):
            lt = tile(tf, "TEACHER", teacher["success"], i, len(tf))
            rt = tile(sf, "STUDENT", student["success"], i, len(sf))
            gap = np.zeros((H, 6, 3), np.uint8)
            top = np.concatenate([lt, gap, rt], axis=1)
            top_w = top.shape[1]
            p = cv2.resize(plot_img, (top_w, plot_h))
            xp = int(round(x_of_step(i) * top_w / pw))
            cv2.line(p, (xp, 0), (xp, plot_h), (255, 0, 0), 2)
            frame = np.concatenate([top, p], axis=0)
            h2, w2 = frame.shape[:2]
            frame = frame[: h2 - (h2 % 2), : w2 - (w2 % 2)]
            wr.append_data(np.ascontiguousarray(frame))


def kv_firstchunk_plot(path, teacher, student):
    tk = teacher["kv"][0] if teacher["kv"] else None
    sk = student["kv"][0] if student["kv"] else None
    if not tk or not sk or tk.get("k_full") is None or sk.get("k_full") is None:
        return False
    def cos_lt(A, B):  # [L, N, D] -> [L, N]
        A = A.astype(np.float32); B = B.astype(np.float32)
        num = (A * B).sum(-1)
        den = (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)) + 1e-8
        return num / den
    ck = cos_lt(tk["k_full"], sk["k_full"])
    cv = cos_lt(tk["v_full"], sk["v_full"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for ax, m, ttl in ((axes[0], ck, "K cosine (teacher vs student)"),
                       (axes[1], cv, "V cosine (teacher vs student)")):
        im = ax.imshow(m, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn", origin="lower")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("token position (first 100)"); ax.set_ylabel("LLM layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("First-chunk KV agreement at IDENTICAL inputs (same reset state)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path); plt.close(fig)
    return True


def kv_overtime_plot(path, teacher, student):
    def stack_norm(roll):
        arr = [c["norm_k"] for c in roll["kv"] if "norm_k" in c]  # list of [L, N]
        if not arr:
            return None
        M = np.stack(arr, 0)          # [C, L, N]
        return M.mean(1)              # [C, N] mean over layers
    T = stack_norm(teacher); S = stack_norm(student)
    if T is None or S is None:
        return False
    vmax = float(max(T.max(), S.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120, sharey=True)
    for ax, m, ttl in ((axes[0], T, "TEACHER"), (axes[1], S, "STUDENT")):
        im = ax.imshow(m.T, aspect="auto", vmin=0, vmax=vmax, cmap="viridis", origin="lower")
        ax.set_title(f"{ttl} KV |k| (mean over layers)", fontsize=10)
        ax.set_xlabel("action chunk (time)"); ax.set_ylabel("token position")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-token KV magnitude through time", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path); plt.close(fig)
    return True


def save_kv_raw(path, teacher, student):
    def norms(roll, kind):
        arr = [c[kind] for c in roll["kv"] if kind in c]
        return np.stack(arr, 0) if arr else np.zeros((0,))
    kw = dict(
        t_norm_k=norms(teacher, "norm_k"), t_norm_v=norms(teacher, "norm_v"),
        s_norm_k=norms(student, "norm_k"), s_norm_v=norms(student, "norm_v"),
        t_actions=teacher["actions"], s_actions=student["actions"],
        t_success=teacher["success"], s_success=student["success"],
        t_chunk_step=np.asarray(teacher["chunk_step"]), s_chunk_step=np.asarray(student["chunk_step"]),
    )
    if teacher["kv"] and teacher["kv"][0].get("k_full") is not None:
        kw["t_k_full0"] = teacher["kv"][0]["k_full"]; kw["t_v_full0"] = teacher["kv"][0]["v_full"]
    if student["kv"] and student["kv"][0].get("k_full") is not None:
        kw["s_k_full0"] = student["kv"][0]["k_full"]; kw["s_v_full0"] = student["kv"][0]["v_full"]
    np.savez_compressed(path, **kw)


# ------------------------------------------------------------------ main
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", default="/outputs/llm_distill/runs/qwen06w_fn/latest.pt")
    ap.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="0")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per rollout (0=env default)")
    ap.add_argument("--out", default="/outputs/llm_distill/diag_cl")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def build_policy(args):
    import data as D
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = 1; device = args.device; num_workers = 0
    _, ds_meta, _, _ = D.build_dataset_and_preprocessor(A)
    cfg = MolmoAct2Config(checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
                          action_mode="continuous", model_dtype="bfloat16", device=args.device)
    if hasattr(cfg, "inference_action_mode"):
        cfg.inference_action_mode = "continuous"
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    policy = MolmoAct2Policy(cfg).to(args.device)
    policy.eval()
    return policy, cfg, ds_meta


def swap_in_student(policy, args):
    """In-place: replace transformer with the trained student and merge the action-expert LoRA.
    Call AFTER all teacher rollouts (this permanently mutates the action-expert projections)."""
    import student as S, joint_patch as JP
    ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    keys = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
            "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")
    scfg = {k: v for k, v in ck["cfg"].items() if k in keys}
    stu, _ = S.build_student(scfg)
    stu.load_state_dict(ck["student"], strict=True)
    JP.load_student_and_merge_lora(policy, stu, ck)
    policy.eval()
    return ck.get("step")


def make_libero_env(args, cfg, ds_meta):
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    task_ids = [int(x) for x in str(args.task_ids).split(",") if x != ""]
    env_cfg = LiberoEnv(task=args.suite, task_ids=task_ids,
                        camera_name_mapping={"agentview_image": "image",
                                             "robot0_eye_in_hand_image": "wrist_image"})
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    # envs is nested {group:{task_id: vec_env}} -> take first
    vec = next(iter(next(iter(envs.values())).values()))
    return vec, env_pre, env_post, pre, post


def main():
    args = get_args()
    from lerobot.envs.utils import preprocess_observation
    os.makedirs(args.out, exist_ok=True)
    cap = args.max_steps or None
    adim_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]

    print(f"[cl] building policy (teacher)", flush=True)
    policy, cfg, ds_meta = build_policy(args)
    _install_kv_hook(policy)
    t_env, env_pre, env_post, pre, post = make_libero_env(args, cfg, ds_meta)

    summary = {"suite": args.suite, "task_ids": args.task_ids, "seed_base": args.seed,
               "student_ckpt": os.path.basename(args.student_ckpt), "episodes": []}

    # Phase 1: all teacher rollouts (original, unmerged action expert)
    teacher_rolls = {}
    for ep in range(args.n_episodes):
        seed = args.seed + ep
        print(f"[cl] === ep {ep} seed {seed}: TEACHER rollout ===", flush=True)
        t0 = time.time()
        teacher_rolls[ep] = run_rollout(policy, t_env, env_pre, pre, post, env_post,
                                        preprocess_observation, seed, cap)
        r = teacher_rolls[ep]
        print(f"[cl] ep{ep} teacher: steps={r['steps']} success={r['success']} "
              f"chunks={len(r['kv'])} ({time.time()-t0:.0f}s)", flush=True)

    # Phase 2: swap the student in, run all student rollouts (same seeds)
    step = swap_in_student(policy, args)
    print(f"[cl] swapped student (step={step}); running student rollouts", flush=True)
    for ep in range(args.n_episodes):
        seed = args.seed + ep
        epdir = os.path.join(args.out, f"ep_{ep:02d}")
        os.makedirs(epdir, exist_ok=True)
        teacher = teacher_rolls[ep]
        print(f"[cl] === ep {ep} seed {seed}: STUDENT rollout ===", flush=True)
        t1 = time.time()
        student = run_rollout(policy, t_env, env_pre, pre, post, env_post,
                              preprocess_observation, seed, cap)
        print(f"[cl] ep{ep} student: steps={student['steps']} success={student['success']} "
              f"chunks={len(student['kv'])} ({time.time()-t1:.0f}s)", flush=True)

        # action MSE over overlapping steps
        H = min(len(teacher["actions"]), len(student["actions"]))
        amse = float(np.mean((teacher["actions"][:H] - student["actions"][:H]) ** 2)) if H else None

        plot_img, x_of_step, _ = build_action_plot(teacher["actions"], student["actions"], adim_names)
        cv2.imwrite(os.path.join(epdir, "actions.png"), cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
        make_video(os.path.join(epdir, "rollout.mp4"), teacher, student, plot_img, x_of_step)
        kv1 = kv_firstchunk_plot(os.path.join(epdir, "kv_firstchunk.png"), teacher, student)
        kvo = kv_overtime_plot(os.path.join(epdir, "kv_overtime.png"), teacher, student)
        save_kv_raw(os.path.join(epdir, "kv_raw.npz"), teacher, student)

        epsum = {"ep": ep, "seed": seed,
                 "teacher_success": teacher["success"], "student_success": student["success"],
                 "teacher_steps": teacher["steps"], "student_steps": student["steps"],
                 "action_mse_overlap": amse,
                 "kv_firstchunk": kv1, "kv_overtime": kvo}
        summary["episodes"].append(epsum)
        with open(os.path.join(epdir, "summary.json"), "w") as f:
            json.dump(epsum, f, indent=2)
        print(f"[cl] ep{ep} artifacts -> {epdir} | action_mse={amse}", flush=True)

    ts = sum(e["teacher_success"] for e in summary["episodes"])
    ss = sum(e["student_success"] for e in summary["episodes"])
    summary["teacher_pc_success"] = 100.0 * ts / args.n_episodes
    summary["student_pc_success"] = 100.0 * ss / args.n_episodes
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[cl] DONE teacher={summary['teacher_pc_success']}% student={summary['student_pc_success']}% "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
