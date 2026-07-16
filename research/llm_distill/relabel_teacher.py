"""DAgger step 2 -- teacher relabeling.

For each student-visited snapshot (from ``collect_dagger.py``), run the TEACHER's
``predict_action_chunk`` to get its normalized action chunk ``[1, n_action_steps, action_dim]`` --
which is exactly the space of ``batch[ACTION]`` consumed by the flow-matching loss (the eval
post-processor un-normalizes afterwards, so the raw chunk is the training target). We assemble
drop-in training samples ``{<model inputs>, ACTION=teacher chunk, action_dim_is_pad,
action_horizon_is_pad}`` and save them as shards.

``--validate`` first proves the relabel is in the right space/shape on a *demo* batch:
  * teacher chunk vs demo ground-truth action MSE (teacher imitates demos -> should be small),
  * flow loss with the teacher-relabeled target (should be finite and near the teacher baseline),
before touching any collected states.
"""

from __future__ import annotations
import argparse, glob, json, os, time
import torch

import diag_closed_loop as CL
import data as D


def _action_meta(policy, device):
    """Read the ACTION target shape/dtype + action_dim pad template from one demo batch."""
    class A:
        repo_id = RELABEL_ARGS.repo_id
        revision = RELABEL_ARGS.revision
        teacher = RELABEL_ARGS.teacher
        batch = 1
        device = RELABEL_ARGS.device
        num_workers = 0
    it = D.iter_full_batches(A, device)
    demo = next(it)
    from lerobot.policies.molmoact2.modeling_molmoact2 import ACTION
    act = demo[ACTION]
    meta = {
        "Dm": int(act.shape[-1]),
        "chunk": int(act.shape[1]),
        "dtype": act.dtype,
        "action_dim_is_pad": demo.get("action_dim_is_pad"),
        "ACTION": ACTION,
    }
    return demo, meta


def build_relabeled(snap, tchunk, meta, device):
    """snap: model-ready batch (B=1). tchunk: teacher chunk [1, n, adim]. Returns a training
    sample dict with the model inputs + teacher ACTION padded to Dm + pad masks."""
    ACTION = meta["ACTION"]
    Dm, chunk, dtype = meta["Dm"], meta["chunk"], meta["dtype"]
    n, adim = tchunk.shape[1], tchunk.shape[2]
    act = torch.zeros(1, chunk, Dm, dtype=dtype)
    m = min(n, chunk)
    act[:, :m, :adim] = tchunk[:, :m, :].to(dtype).cpu()
    out = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in snap.items()}
    out[ACTION] = act
    # action_dim padding: reuse the demo template (LIBERO uses a fixed real-dim count)
    adp = meta["action_dim_is_pad"]
    if adp is not None:
        out["action_dim_is_pad"] = adp[:1].detach().cpu().clone()
    # full teacher chunk -> no horizon padding
    out["action_horizon_is_pad"] = torch.zeros(1, chunk, dtype=torch.bool)
    return out


@torch.no_grad()
def teacher_chunk(policy, batch, device):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return policy.predict_action_chunk(b).detach().cpu()


@torch.no_grad()
def _flow_with_pinned(policy, batch, t, noise):
    """Deterministic flow loss on the TEACHER (native per-layer joint loss) with pinned t/noise."""
    mi = policy._model_inputs(batch)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
            batch=batch, model_inputs=mi, timesteps=t, noise=noise)
    return float(loss)


@torch.no_grad()
def validate(policy, device, n_batches=6):
    """Rigorous relabel check: over several demo batches with PINNED timesteps+noise, compare the
    teacher flow loss on (a) the ground-truth demo action vs (b) the teacher-relabeled action
    (its own current-step prediction). A well-formed relabel target should give a flow loss no
    worse than -- ideally at or below -- the demo-target loss; a large gap means a space/shape bug.
    Also reports teacher-chunk-vs-demo MSE (normalized-space alignment)."""
    from lerobot.policies.molmoact2.modeling_molmoact2 import ACTION
    _, meta = _action_meta(policy, device)
    nft = max(1, int(policy.config.num_flow_timesteps))

    class A:
        repo_id = RELABEL_ARGS.repo_id; revision = RELABEL_ARGS.revision
        teacher = RELABEL_ARGS.teacher; batch = 1; device = RELABEL_ARGS.device; num_workers = 0
    it = D.iter_full_batches(A, device)
    g = torch.Generator(device="cpu").manual_seed(1234)

    sum_demo = sum_relabel = sum_mse = 0.0
    for _ in range(n_batches):
        demo = next(it)
        a = demo[ACTION]
        B, L, Dm = a.shape[0], a.shape[1], a.shape[2]
        t = torch.rand(B, nft, generator=g).to(a.device)
        noise = torch.randn(B, nft, L, Dm, generator=g).to(a.device)
        # teacher current-step relabel
        tchunk = teacher_chunk(policy, demo, device)                  # [1, n, 7]
        gt = a[:, :1, :tchunk.shape[2]].detach().cpu()                # current-step demo action
        sum_mse += float((tchunk[:, :1, :] - gt).pow(2).mean())
        relabel = build_relabeled({k: v for k, v in demo.items()}, tchunk, meta, device)
        rb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in relabel.items()}
        sum_demo += _flow_with_pinned(policy, demo, t, noise)
        sum_relabel += _flow_with_pinned(policy, rb, t, noise)
    n = float(n_batches)
    print(f"[relabel:validate] Dm={meta['Dm']} chunk(L)={meta['chunk']} adim=7 nft={nft} "
          f"n_batches={n_batches} | teacher_chunk vs demo MSE={sum_mse/n:.4f} "
          f"| flow_loss demo_target={sum_demo/n:.4f} relabel_target={sum_relabel/n:.4f} "
          f"(teacher baseline ~0.58)", flush=True)
    return meta


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--collect-dir", default="", help="dir of collect_dagger shards (*.pt)")
    ap.add_argument("--out", default="/outputs/llm_distill/dagger/relabel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--validate", action="store_true", help="run the demo-batch sanity check")
    ap.add_argument("--limit", type=int, default=0, help="cap #states relabeled (0=all; smoke)")
    return ap.parse_args()


RELABEL_ARGS = None


def main():
    global RELABEL_ARGS
    RELABEL_ARGS = get_args()
    args = RELABEL_ARGS
    device = torch.device(args.device)
    print("[relabel] building teacher policy", flush=True)
    policy, cfg, ds_meta = CL.build_policy(args)  # teacher (no student swap)
    policy.eval()

    meta = None
    if args.validate:
        meta = validate(policy, device)

    if not args.collect_dir:
        print("[relabel] no --collect-dir; validate-only done", flush=True)
        return

    if meta is None:
        _, meta = _action_meta(policy, device)
    os.makedirs(args.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.collect_dir, "*.pt")))
    total, done = 0, 0
    t0 = time.time()
    manifest = {"collect_dir": args.collect_dir, "out": args.out, "shards": []}
    for sp in shards:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        samples = []
        for snap in d["batches"]:
            if args.limit and total >= args.limit:
                break
            tchunk = teacher_chunk(policy, snap, device)
            samples.append(build_relabeled(snap, tchunk, meta, device))
            total += 1
        name = os.path.basename(sp)
        torch.save({"samples": samples, "meta": d.get("meta", {})},
                   os.path.join(args.out, name))
        manifest["shards"].append({"file": name, "n": len(samples)})
        done += 1
        print(f"[relabel] {name}: relabeled {len(samples)} states "
              f"(total={total}, {time.time()-t0:.0f}s)", flush=True)
        if args.limit and total >= args.limit:
            break
    manifest["total_states"] = total
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[relabel] DONE relabeled {total} states from {done} shards -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
