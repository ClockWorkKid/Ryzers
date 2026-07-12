"""Multi-level distillation losses for the thin-twin LLM student (Run D).

Stage 1 (LLM-only): match student -> teacher per-layer hidden, per-layer KV, and final
hidden. Stage 2 adds the flow-matching task loss + the ViT<->LLM interface term (assembled
in the joint trainer, reusing hidden_kd/kv_kd here).

All feature tensors are compared over valid (non-pad) positions via ``mask`` [B, N].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x, mask):
    # x: [B, N] per-position scalar; mask: [B, N] in {0,1}
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def cos_mse(pred, target, mask, cos_w=1.0, mse_w=1.0):
    """Combined (1 - cosine) + MSE over feature dim, masked over positions.
    pred/target: [B, N, D]; mask: [B, N]."""
    pred = pred.float()
    target = target.float()
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)          # [B, N]
    mse = ((pred - target) ** 2).mean(-1)                          # [B, N]
    return cos_w * _masked_mean(cos, mask) + mse_w * _masked_mean(mse, mask), \
        _masked_mean(cos, mask).detach()


def hidden_kd(student, heads, teacher_hidden, mask, layers):
    """Per-layer hidden distillation. ``student['hidden_states']`` and
    ``teacher_hidden`` are aligned lists (len num_layers+1); project student->2560."""
    total = 0.0
    cos_acc = 0.0
    n = 0
    for li in layers:
        sh = heads.project_hidden(li, student["hidden_states"][li])  # [B,N,2560]
        loss, cos = cos_mse(sh, teacher_hidden[li], mask)
        total = total + loss
        cos_acc = cos_acc + cos
        n += 1
    return total / max(n, 1), (cos_acc / max(n, 1))


def kv_kd(student, heads, teacher_kv, mask):
    """Per-layer KV distillation -- aligns with what the action expert cross-attends to.
    teacher_kv: list of (k,v), each [B, kvH, N, hd] -> flatten to [B, N, kv_dim]."""
    total = 0.0
    n = len(student["kv_states"])
    for li in range(n):
        k_raw, v_raw = student["kv_states"][li]
        kp, vp = heads.project_kv(li, k_raw, v_raw)                 # [B,N,1024]
        tk, tv = teacher_kv[li]
        B, H, N, D = tk.shape
        tk = tk.transpose(1, 2).reshape(B, N, H * D)
        tv = tv.transpose(1, 2).reshape(B, N, H * D)
        lk, _ = cos_mse(kp, tk, mask)
        lv, _ = cos_mse(vp, tv, mask)
        total = total + lk + lv
    return total / max(n, 1)


def final_kd(student, teacher_last_hidden, mask):
    """Final hidden (2560-d) distillation -- the up-projected deploy output."""
    loss, cos = cos_mse(student["last_hidden_state"], teacher_last_hidden, mask)
    return loss, cos


def stage1_loss(student, heads, teacher, mask, w):
    """teacher: dict(hidden=[...], kv=[(k,v)...], last=[B,N,2560]).
    w: dict of weights hidden/kv/final. Returns (loss, metrics)."""
    h_loss, h_cos = hidden_kd(student, heads, teacher["hidden"], mask, heads.c.hidden_distill_layers)
    kv_loss = kv_kd(student, heads, teacher["kv"], mask)
    f_loss, f_cos = final_kd(student, teacher["last"], mask)
    loss = w["hidden"] * h_loss + w["kv"] * kv_loss + w["final"] * f_loss
    return loss, {
        "loss": float(loss.detach()),
        "hidden": float(h_loss.detach()),
        "kv": float(kv_loss.detach()),
        "final": float(f_loss.detach()),
        "hidden_cos": float(h_cos),
        "final_cos": float(f_cos),
    }
