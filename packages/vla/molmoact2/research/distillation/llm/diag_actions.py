"""Open-loop action diagnostic: does the student's INFERENCE-time action chunk match the
teacher's (which scores 100% closed-loop)?

Calls the EXACT eval inference path (policy.predict_action_chunk) for teacher and student on
the SAME held-out observations. predict_action_chunk auto-seeds the ODE init noise
deterministically from the inputs, so teacher and student integrate from identical noise.
Reports action agreement (MSE teacher-vs-student, and each vs ground truth). This decouples
the low training flow loss from the 0% closed-loop success.
"""

from __future__ import annotations
import argparse
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", default="/outputs/llm_distill/runs/qwen06w_fn/step_2500.pt")
    ap.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = get_args()
    import data as D
    import student as S
    import joint_patch as JP
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy, ACTION
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = 2

    _, ds_meta, _, _ = D.build_dataset_and_preprocessor(A)
    cfg = MolmoAct2Config(checkpoint_path=args.teacher, chunk_size=10, n_action_steps=10,
                          action_mode="continuous", model_dtype="bfloat16", device=args.device)
    if hasattr(cfg, "inference_action_mode"):
        cfg.inference_action_mode = "continuous"
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    policy = MolmoAct2Policy(cfg).to(args.device); policy.eval()

    it = D.iter_full_batches(A, torch.device(args.device))
    batches = [next(it) for _ in range(args.n_batches)]

    @torch.no_grad()
    def run(pol):
        outs = []
        for b in batches:
            outs.append(pol.predict_action_chunk(b).float().cpu())
        return outs

    t_acts = run(policy)
    print(f"[diag] teacher action chunk shape: {tuple(t_acts[0].shape)}", flush=True)

    ck = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    scfg = {k: v for k, v in ck["cfg"].items()
            if k in ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                     "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")}
    stu, _ = S.build_student(scfg)
    stu.load_state_dict(ck["student"], strict=True)
    JP.load_student_and_merge_lora(policy, stu, ck)
    policy.eval()
    s_acts = run(policy)
    print(f"[diag] student ckpt step={ck.get('step')}", flush=True)

    def mse(a, b):
        return torch.mean((a.float() - b.float()) ** 2).item()

    def cos(a, b):
        a = a.reshape(-1); b = b.reshape(-1)
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    tt, gt_t, gt_s, ct = [], [], [], []
    for i, b in enumerate(batches):
        gt = b[ACTION].float().cpu()  # [B, chunk, adim]
        ta, sa = t_acts[i], s_acts[i]
        H = min(gt.shape[1], ta.shape[1]); Dm = min(gt.shape[2], ta.shape[2])
        ta_, sa_, gt_ = ta[:, :H, :Dm], sa[:, :H, :Dm], gt[:, :H, :Dm]
        tt.append(mse(ta_, sa_)); ct.append(cos(ta_, sa_))
        gt_t.append(mse(ta_, gt_)); gt_s.append(mse(sa_, gt_))

    import statistics as st
    print("\n[diag] ===== inference action agreement (eval path) =====", flush=True)
    print(f"  MSE(teacher, student)      = {st.mean(tt):.4f}   cos = {st.mean(ct):.4f}", flush=True)
    print(f"  MSE(teacher, ground-truth) = {st.mean(gt_t):.4f}", flush=True)
    print(f"  MSE(student, ground-truth) = {st.mean(gt_s):.4f}", flush=True)
    a0t = torch.cat([t[:, 0, :] for t in t_acts], 0).float()
    a0s = torch.cat([t[:, 0, :] for t in s_acts], 0).float()
    Dm = min(a0t.shape[1], a0s.shape[1])
    perdim = (a0t[:, :Dm] - a0s[:, :Dm]).abs().mean(0)
    print(f"  first-step per-dim |teacher-student|: {[round(x,3) for x in perdim.tolist()]}", flush=True)
    print(f"  teacher range [{a0t.min():.2f},{a0t.max():.2f}]  student [{a0s.min():.2f},{a0s.max():.2f}]", flush=True)


if __name__ == "__main__":
    main()
