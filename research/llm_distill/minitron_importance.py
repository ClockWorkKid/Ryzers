"""Minitron activation-based importance estimation for the MolmoAct2 LLM backbone (experiment C).

Faithful to Muralidharan et al. 2024 ("Compact Language Models via Pruning and Knowledge
Distillation") + "...in Practice" 2024: purely forward-pass, activation-based importance on a
small calibration set, with the paper's best-practice aggregation **(batch=L2, seq=mean)** for
the width axes. We estimate importance for the two axes we prune first (zero action-expert
surgery): MLP intermediate **neurons** and attention **query heads**. We also compute per-layer
**Block Importance** (BI = 1 - mean cos(in, out)) for the optional depth axis later.

Calibration data = real LIBERO batches (the task distribution the backbone actually serves),
run through the FROZEN teacher backbone -- the natural analogue of Minitron's calibration set.

Hooks (no teacher-code edits):
  - MLP neurons: forward-pre-hook on ``block.mlp.ff_out`` -> its input is the intermediate
    activation ``silu(gate)*up`` of shape [B,S,intermediate] (one value per neuron).
  - Query heads: forward-pre-hook on ``block.self_attn.attn_out`` -> its input is the
    concatenated per-head attention output [B,S,num_heads*head_dim].
  - Block importance: forward-hook on each decoder block (input hidden vs output hidden).

Output: a torch save with per-layer tensors
  {mlp_neuron: [L, interm], q_head: [L, num_heads], block: [L], meta:{...}}.
"""

from __future__ import annotations
import argparse, json, os, time
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--calib-batches", type=int, default=128,
                    help="number of calibration batches (Minitron uses ~1024 samples)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/outputs/llm_distill/minitron/importance.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args()


class ImportanceAccumulator:
    """(batch=L2, seq=mean) accumulation. For each calibration batch we reduce the sequence
    dim by a masked mean of the activation magnitude, then accumulate the SQUARE across
    batches; the final importance is sqrt(sum_batches (seq_mean)^2)."""

    def __init__(self, num_layers):
        self.mlp_sq = [None] * num_layers      # [interm]
        self.head_sq = [None] * num_layers     # [num_heads]
        self.bi_sum = [0.0] * num_layers       # scalar accum of (1-cos)
        self.bi_n = 0

    def add_mlp(self, li, seq_mean):           # seq_mean: [interm] (this batch)
        v = seq_mean.detach().double() ** 2
        self.mlp_sq[li] = v if self.mlp_sq[li] is None else self.mlp_sq[li] + v

    def add_head(self, li, seq_mean):          # seq_mean: [num_heads]
        v = seq_mean.detach().double() ** 2
        self.head_sq[li] = v if self.head_sq[li] is None else self.head_sq[li] + v

    def add_bi(self, li, bi):
        self.bi_sum[li] += float(bi)

    def finalize(self):
        mlp = torch.stack([torch.sqrt(x) for x in self.mlp_sq])   # [L, interm]
        head = torch.stack([torch.sqrt(x) for x in self.head_sq]) # [L, num_heads]
        bi = torch.tensor([s / max(self.bi_n, 1) for s in self.bi_sum])
        return mlp.float(), head.float(), bi.float()


def _masked_seq_mean_mag(x, valid):
    """x: [B,S,C] activations; valid: [B,S] bool. Return [C] = mean over valid tokens of |x|,
    averaged over batch (the batch reduction here is a plain mean; the L2-over-batches happens
    across calibration BATCHES in the accumulator per the (batch=L2, seq=mean) recipe)."""
    mag = x.abs().double()
    if valid is not None:
        m = valid.to(mag.dtype)[..., None]
        num = (mag * m).sum(dim=(0, 1))
        den = m.sum(dim=(0, 1)).clamp_min(1.0)
        return num / den
    return mag.mean(dim=(0, 1))


def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import data as D
    import train_joint as TJ

    print(f"[minitron-imp] loading teacher {args.teacher}", flush=True)

    class A:
        repo_id = args.repo_id; revision = args.revision; teacher = args.teacher
        batch = args.batch; device = args.device; num_workers = args.num_workers

    policy = TJ.build_policy(A, device)
    policy.eval()
    backbone = policy._backbone()
    transformer = backbone.transformer
    blocks = transformer.blocks
    L = len(blocks)
    cfg0 = blocks[0].self_attn
    num_heads, head_dim = int(cfg0.num_heads), int(cfg0.head_dim)
    print(f"[minitron-imp] layers={L} num_heads={num_heads} head_dim={head_dim}", flush=True)

    acc = ImportanceAccumulator(L)

    # ---- hooks -----------------------------------------------------------------
    cur_valid = {"mask": None}   # per-batch [B,S] token-validity mask (set each batch)
    handles = []

    def mlp_pre_hook(li):
        def hook(mod, inp):
            x = inp[0]                                   # [B,S,interm]
            acc.add_mlp(li, _masked_seq_mean_mag(x, cur_valid["mask"]))
        return hook

    def attn_pre_hook(li):
        def hook(mod, inp):
            x = inp[0]                                   # [B,S,num_heads*head_dim]
            B, S, _ = x.shape
            xh = x.view(B, S, num_heads, head_dim)
            perhead = xh.double().norm(dim=-1)           # [B,S,num_heads] L2 over head_dim
            # per-head magnitude already positive; reuse the masked seq-mean reducer
            v = cur_valid["mask"]
            if v is not None:
                m = v.to(perhead.dtype)[..., None]
                seq_mean = (perhead * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0)
            else:
                seq_mean = perhead.mean(dim=(0, 1))
            acc.add_head(li, seq_mean)
        return hook

    def block_hook(li):
        def hook(mod, inp, out):
            # BI (depth axis) is best-effort: only if the block was called with the hidden
            # state positionally (it is, in MolmoAct2TextModel). Never crash the width run.
            if not inp or not torch.is_tensor(inp[0]):
                return
            x_in = inp[0]
            x_out = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(x_out) or x_out.shape != x_in.shape:
                return
            a = torch.nn.functional.normalize(x_in.double(), dim=-1)
            b = torch.nn.functional.normalize(x_out.double(), dim=-1)
            cos = (a * b).sum(-1)                          # [B,S]
            v = cur_valid["mask"]
            if v is not None:
                cos = (cos * v).sum() / v.sum().clamp_min(1.0)
            else:
                cos = cos.mean()
            acc.add_bi(li, 1.0 - float(cos))
        return hook

    for li, blk in enumerate(blocks):
        handles.append(blk.mlp.ff_out.register_forward_pre_hook(mlp_pre_hook(li)))
        handles.append(blk.self_attn.attn_out.register_forward_pre_hook(attn_pre_hook(li)))
        handles.append(blk.register_forward_hook(block_hook(li)))

    # ---- calibration forward passes -------------------------------------------
    it = D.iter_full_batches(A, device)

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for bi in range(args.calib_batches):
            batch = next(it)
            cap = D.capture_teacher(policy.model, batch, collect_kv=False)
            m = cap["mask"]
            cur_valid["mask"] = (m.to(torch.bool) if (m is not None and torch.is_tensor(m)
                                                      and m.dim() == 2) else None)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                transformer(inputs_embeds=cap["inputs_embeds"], attention_mask=cap["attn_bias"],
                            position_ids=cap["positions"],
                            collect_layer_kv_states=True, use_cache=False)
            acc.bi_n += 1
            done += 1
            if (bi + 1) % 16 == 0:
                print(f"[minitron-imp] calib {bi+1}/{args.calib_batches} "
                      f"({(time.time()-t0):.0f}s)", flush=True)

    for h in handles:
        h.remove()

    mlp, head, bi = acc.finalize()
    out = {
        "mlp_neuron": mlp,           # [L, interm]  higher = more important
        "q_head": head,              # [L, num_heads]
        "block": bi,                 # [L]  higher = more important (larger transform)
        "meta": {"num_layers": L, "num_heads": num_heads, "head_dim": head_dim,
                 "intermediate": int(mlp.shape[1]), "calib_batches": done,
                 "batch": args.batch, "teacher": args.teacher,
                 "agg": "batch=L2, seq=mean"},
    }
    torch.save(out, args.out)
    print(f"[minitron-imp] saved -> {args.out}", flush=True)
    # quick sanity readout
    print(f"[minitron-imp] mlp_neuron importance: shape={tuple(mlp.shape)} "
          f"min={mlp.min():.3e} max={mlp.max():.3e}", flush=True)
    print(f"[minitron-imp] q_head importance: shape={tuple(head.shape)} "
          f"per-layer spread e.g. L0={head[0].tolist()[:8]}", flush=True)
    print(f"[minitron-imp] block importance (BI) per layer: "
          f"{[round(x,4) for x in bi.tolist()]}", flush=True)


if __name__ == "__main__":
    main()
