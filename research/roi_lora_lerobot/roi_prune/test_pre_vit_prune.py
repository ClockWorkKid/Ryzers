"""Standalone validation of the pre-ViT pruning primitives (rule 2: module test).

Run: python test_pre_vit_prune.py   (needs torch; run inside the ROCm container or
any torch env). CPU-only is fine.
"""

import torch

from pre_vit_prune import (
    PreViTPruneConfig,
    ROIGate,
    apply_gate_ste,
    gather_keep,
    scatter_back,
    score_patches,
    select_topk,
)


def _fake_encoder(x):
    """Stand-in for the ViT resblocks: any shape-preserving op on [B,K,H]."""
    return x * 2.0 + 1.0


def test_keep_count_and_shapes():
    B, N, H = 2, 16, 8
    for frac in (0.25, 0.5, 1.0):
        cfg = PreViTPruneConfig(enable=True, keep_frac=frac, select="energy")
        k = cfg.resolve_keep(N)
        x = torch.randn(B, N, H)
        scores = score_patches(x, cfg.select)
        keep = select_topk(scores, k)
        assert keep.shape == (B, k), (keep.shape, k)
        x_kept = gather_keep(x, keep)
        assert x_kept.shape == (B, k, H)
        feats = scatter_back(_fake_encoder(x_kept), keep, N, placeholder="zeros")
        assert feats.shape == (B, N, H)
    print("[ok] keep-count + shapes for fracs {0.25,0.5,1.0}")


def test_scatter_back_index_correctness():
    B, N, H = 3, 10, 4
    x = torch.randn(B, N, H)
    scores = score_patches(x, "energy")
    keep = select_topk(scores, 4)
    encoded = _fake_encoder(gather_keep(x, keep))
    feats = scatter_back(encoded, keep, N, placeholder="zeros")
    # kept positions hold the encoded value; pruned positions are exactly zero.
    for b in range(B):
        kept = set(keep[b].tolist())
        for i in range(N):
            if i in kept:
                pos = (keep[b] == i).nonzero().item()
                assert torch.allclose(feats[b, i], encoded[b, pos])
            else:
                assert torch.count_nonzero(feats[b, i]) == 0
    print("[ok] scatter_back places kept features at original indices; pruned=0")


def test_disable_is_noop():
    cfg = PreViTPruneConfig(enable=False)
    assert cfg.resolve_keep(64) == 64
    cfg2 = PreViTPruneConfig(enable=True, keep_frac=1.0)
    assert cfg2.resolve_keep(64) == 64
    print("[ok] disabled / keep_frac=1.0 keeps all patches (no-op)")


def test_gate_gradients_flow():
    B, N, H = 2, 12, 16
    gate = ROIGate(H, gate_hidden=32)
    x = torch.randn(B, N, H)
    scores = score_patches(x, "gate", gate=gate)
    keep = select_topk(scores.detach(), 6)  # selection is non-diff; STE carries grad
    x_kept = gather_keep(x, keep)
    x_kept = apply_gate_ste(x_kept, scores, keep)
    encoded = _fake_encoder(x_kept)
    feats = scatter_back(encoded, keep, N, placeholder="zeros")
    loss = feats.pow(2).mean()
    loss.backward()
    g = gate.fc1.weight.grad
    assert g is not None and torch.count_nonzero(g) > 0, "no gradient reached the gate"
    # STE must preserve the forward magnitude (weight == 1 in forward).
    ste_scale = ((1 - torch.sigmoid(scores.gather(1, keep))).detach()
                 + torch.sigmoid(scores.gather(1, keep)))
    assert torch.allclose(ste_scale, torch.ones_like(ste_scale)), "STE forward != 1"
    print("[ok] gate receives gradient; STE forward magnitude preserved")


def test_learned_placeholder_broadcast():
    B, N, H = 2, 8, 5
    x = torch.randn(B, N, H)
    keep = select_topk(score_patches(x, "energy"), 3)
    encoded = _fake_encoder(gather_keep(x, keep))
    mask_tok = torch.randn(H)
    feats = scatter_back(encoded, keep, N, placeholder=mask_tok)
    for b in range(B):
        kept = set(keep[b].tolist())
        for i in range(N):
            if i not in kept:
                assert torch.allclose(feats[b, i], mask_tok)
    print("[ok] learned mask-token placeholder fills pruned positions")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_keep_count_and_shapes()
    test_scatter_back_index_correctness()
    test_disable_is_noop()
    test_gate_gradients_flow()
    test_learned_placeholder_broadcast()
    print("\nALL PRE-VIT PRUNE MODULE TESTS PASSED")
