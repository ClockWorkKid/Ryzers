"""Offline localizer for the quantized-AE closed-loop collapse.

The closed-loop sweep showed even W8A8 dropping to ~34% (FP ~99%). This isolates
*why* without paying for a 3 h sim, by comparing the FP action expert against the
QAT blob reloaded two ways:

  * ``bf16`` : matches how ae_qat.py quantized/trained the AE.
  * ``fp32`` : matches how eval_closedloop_ae.py currently reloads it.

For each variant we report, on the same held-out LIBERO batches:
  * held-out flow-matching loss (should match the QAT ~0.09-0.13 if the model is
    intact) -- a 1-step teacher-forced proxy;
  * the *integrated* action chunk (full 8-step Euler, the real closed-loop path
    via ``policy.predict_action_chunk``) MSE vs the FP chunk and vs demo actions.

If bf16 reproduces FP actions but fp32 does not  -> eval dtype mismatch is the bug.
If both match FP flow loss but both diverge on the integrated chunk -> the collapse
is real compounding error (1-step MSE is a poor proxy), not a load bug.

    python ae_diag.py --weight-bits 8 --act-bits 8 --io-bits 8 \
        --blob /outputs/ae_quant/w8a8_io8/ae_W8A8_qat_ep.pt --n-batches 8
"""
from __future__ import annotations

import argparse
import json
import types

import torch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="allenai/MolmoAct2-LIBERO")
    p.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--weight-bits", type=int, required=True)
    p.add_argument("--act-bits", type=int, required=True)
    p.add_argument("--io-bits", type=int, default=8)
    p.add_argument("--blob", required=True)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtypes", default="bf16,fp32", help="comma list: bf16,fp32")
    return p.parse_args()


def _args_ns(a):
    return types.SimpleNamespace(
        teacher=a.teacher, repo_id=a.repo_id, revision=a.revision, device=a.device,
        batch=a.batch, num_workers=a.num_workers, seed=a.seed, grad_accum=1,
    )


def _disable_cuda_graph(policy):
    """Static CUDA-graph capture of the action-flow loop is incompatible with the
    Brevitas fake-quant AE (breaks stream capture / bakes stale values). Force the
    eager path so quantized inference is correct."""
    try:
        policy.config.enable_inference_cuda_graph = False
    except Exception:  # noqa: BLE001
        pass
    for fn in ("_set_inference_cuda_graph_enabled",):
        if hasattr(policy, fn):
            try:
                getattr(policy, fn)(False)
            except Exception:  # noqa: BLE001
                pass


@torch.no_grad()
def _flow(policy, batches):
    policy.eval()
    tot = 0.0
    for b in batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(b)
        tot += float(loss)
    return tot / max(len(batches), 1)


@torch.no_grad()
def _chunks(policy, batches):
    policy.eval()
    out = []
    for b in batches:
        out.append(policy.predict_action_chunk(
            b, inference_action_mode="continuous").float().cpu())
    return out


def _masked_mse(a, b, mask):
    d = (a - b) ** 2
    if mask is not None:
        m = mask.to(d.dtype)
        while m.dim() < d.dim():
            m = m.unsqueeze(-1)
        m = m.expand_as(d)
        return float((d * m).sum() / m.sum().clamp_min(1.0))
    return float(d.mean())


def _gt(batches):
    gts, masks = [], []
    for b in batches:
        gts.append(b["action"].float().cpu() if "action" in b else None)
        mk = b.get("action_horizon_is_pad")
        # is_pad=True means invalid; valid mask = ~is_pad
        masks.append((~mk).float().cpu() if mk is not None else None)
    return gts, masks


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    blob = torch.load(a.blob, map_location="cpu", weights_only=False)
    io_bits = None if a.io_bits < 0 else a.io_bits
    dtypes = [d.strip() for d in a.dtypes.split(",") if d.strip()]

    import ae_qat as Q
    import ae_quant as AQ
    import data as D

    def log(d):
        print("[ae-diag] " + json.dumps(d), flush=True)

    # ---- FP reference (one policy build) --------------------------------- #
    ns = _args_ns(a)
    policy = Q.build_policy(ns, device)
    _disable_cuda_graph(policy)
    it = D.iter_full_batches(ns, device)
    batches = [next(it) for _ in range(a.n_batches)]
    gts, masks = _gt(batches)

    fp_flow = _flow(policy, batches)
    fp_act = _chunks(policy, batches)  # FP path is graph-free and known-good
    n = len(batches)
    print(f"[ae-diag] FP chunks ok shape={list(fp_act[0].shape)}", flush=True)

    def align(x, ref):
        """Best-effort reshape of a GT/mask tensor to (B, T, D) matching ref."""
        if x is None:
            return None
        if x.dim() == 3:
            return x[:, : ref.shape[1], : ref.shape[2]]
        if x.dim() == 2:
            B = ref.shape[0]
            if x.shape[0] == B and x.numel() % B == 0:
                try:
                    x = x.reshape(B, -1, ref.shape[2]) if x.numel() % (B * ref.shape[2]) == 0 \
                        else x.reshape(B, -1)
                    return x[:, : ref.shape[1], ...]
                except Exception:  # noqa: BLE001
                    return None
        return None

    def gt_mse(pred):
        vals = []
        for i in range(n):
            g = align(gts[i], pred[i])
            if g is None or g.dim() != pred[i].dim():
                continue
            m = align(masks[i], pred[i]) if masks[i] is not None else None
            vals.append(_masked_mse(pred[i], g, m))
        return sum(vals) / len(vals) if vals else float("nan")

    fp_gt_mse = gt_mse(fp_act)
    log({"variant": "fp", "flow_loss": round(fp_flow, 4),
         "act_mse_vs_gt": round(fp_gt_mse, 6),
         "act_shape": list(fp_act[0].shape),
         "gt_shape": list(gts[0].shape) if gts[0] is not None else None})

    # ---- quantized reloads (fresh policy per dtype) ---------------------- #
    wbits, abits = int(blob.get("weight_bits", a.weight_bits)), int(blob.get("act_bits", a.act_bits))
    for di, dname in enumerate(dtypes):
        qdt = torch.bfloat16 if dname == "bf16" else torch.float32
        pol = policy if di == 0 else Q.build_policy(ns, device)
        _disable_cuda_graph(pol)
        ae = pol._backbone()._require_action_expert()
        ae.to(dtype=qdt)
        AQ.quantize_action_expert_(ae, weight_bits=wbits, act_bits=abits, io_bits=io_bits)
        miss, unexp = ae.load_state_dict(blob["state_dict"], strict=False)
        ae.to(device=device, dtype=qdt)
        for p in ae.parameters():
            p.requires_grad_(False)
        ae.eval()

        q_flow = _flow(pol, batches)
        rec = {"variant": f"q_{dname}", "wbits": wbits, "abits": abits, "io_bits": io_bits,
               "load_miss": len(miss), "load_unexp": len(unexp),
               "flow_loss": round(q_flow, 4), "fp_flow": round(fp_flow, 4)}
        try:
            q_act = _chunks(pol, batches)
            rec["act_mse_vs_fp"] = round(sum(
                _masked_mse(q_act[i], fp_act[i], None) for i in range(n)) / max(n, 1), 6)
            rec["act_mse_vs_gt"] = round(gt_mse(q_act), 6)
        except Exception as e:  # noqa: BLE001
            rec["chunk_error"] = str(e)[:200]
        log(rec)

    log({"done": True})


if __name__ == "__main__":
    main()
