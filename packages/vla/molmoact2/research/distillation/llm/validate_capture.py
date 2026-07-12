"""De-risk Run D Stage-1: load the frozen teacher, pull ONE real LIBERO batch, capture
per-layer hidden + per-layer KV + final hidden, and print shapes + sanity stats.

Run on GPU inside the eval SIF (mi325x). No student, no training -- purely validates the
data pipeline + teacher feature-capture API before committing to full training.
"""

from __future__ import annotations

import argparse
import os

import torch

from llm_distill_data import iter_libero_batches, capture_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default=os.environ.get("LIBERO_REPO", "allenai/MolmoAct2-LIBERO-Dataset"))
    ap.add_argument("--revision", default=os.environ.get("LIBERO_REV", "main"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dt = torch.bfloat16
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM
    print(f"[validate] loading teacher {a.teacher}", flush=True)
    model = HFM.MolmoAct2ForConditionalGeneration.from_pretrained(
        a.teacher, torch_dtype=dt).to(a.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[validate] building LIBERO batch iterator", flush=True)
    it = iter_libero_batches(a, torch.device(a.device))
    batch = next(it)
    print("[validate] model batch keys + shapes:", flush=True)
    for k, v in batch.items():
        print(f"    {k:20s} {tuple(v.shape)} {v.dtype}", flush=True)

    with torch.no_grad():
        cap = capture_teacher(model, batch)

    t = cap["teacher"]
    B, N, Dh = cap["inputs_embeds"].shape
    print(f"[validate] inputs_embeds {tuple(cap['inputs_embeds'].shape)}", flush=True)
    print(f"[validate] hidden_states: {len(t['hidden'])} entries, [0]={tuple(t['hidden'][0].shape)} "
          f"[-1]={tuple(t['hidden'][-1].shape)}", flush=True)
    print(f"[validate] kv: {len(t['kv'])} layers, k={tuple(t['kv'][0][0].shape)} v={tuple(t['kv'][0][1].shape)}", flush=True)
    print(f"[validate] last_hidden_state {tuple(t['last'].shape)}", flush=True)
    print(f"[validate] mask sum={int(cap['mask'].sum())}/{B*N} attn_bias={tuple(cap['attn_bias'].shape)}", flush=True)

    # sanity: last hidden == hidden_states[-1] ; adjacent-layer drift ; kv magnitude
    same = torch.allclose(t["last"].float(), t["hidden"][-1].float(), atol=1e-3)
    import torch.nn.functional as F
    cos01 = F.cosine_similarity(t["hidden"][0].float(), t["hidden"][-1].float(), dim=-1).mean().item()
    kmag = t["kv"][0][0].float().abs().mean().item()
    print(f"[validate] last==hidden[-1]: {same} | cos(hidden0,hiddenLast)={cos01:.3f} | mean|k[0]|={kmag:.4f}", flush=True)
    assert len(t["hidden"]) == len(t["kv"]) + 1, "expected hidden = layers+1, kv = layers"
    print(f"[validate] PASS: layers={len(t['kv'])} hidden_dim={Dh} seq={N}", flush=True)


if __name__ == "__main__":
    main()
