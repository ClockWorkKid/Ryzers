"""Analytic FLOP profiler for MolmoAct2 LLM distillation (teacher vs thin-twin student).

Torch-free (stdlib only) so it runs on the laptop. Counts the decoder-stack FLOPs
(the distillation-relevant compute); embedding lookup and lm_head are excluded (they
are shared/cheap relative to the 36-layer stack). The student keeps 36 layers so the
frozen action expert's one-block-per-layer / per-layer-KV conditioning still applies,
and shrinks the width to hit the ~100x compute-reduction target.

Usage:
  python -m research.llm_distill.flops                       # default student
  python flops.py --hidden 224 --heads 8 --head-dim 28 --interm 896
"""

from __future__ import annotations

import argparse

# MolmoAct2-LIBERO teacher LLM (from allenai/MolmoAct2-LIBERO config)
TEACHER = dict(hidden=2560, heads=32, kv_heads=8, head_dim=128, interm=9728, layers=36)


def layer_macs(hidden, heads, kv_heads, head_dim, interm, n):
    q = heads * head_dim
    kv = kv_heads * head_dim
    qkv = n * hidden * (q + 2 * kv)          # fused QKV projection
    attn = 2 * n * n * q                     # QK^T scores + AV
    oproj = n * q * hidden                   # output projection
    mlp = 3 * n * hidden * interm            # SwiGLU: gate+up (2) + down (1)
    return qkv + attn + oproj + mlp


def stack_flops(cfg, n):
    per = layer_macs(cfg["hidden"], cfg["heads"], cfg["kv_heads"],
                     cfg["head_dim"], cfg["interm"], n)
    return 2 * per * cfg["layers"]           # MACs -> FLOPs


def student_flops(hidden, heads, kv_heads, head_dim, interm, layers, n, teacher_hidden):
    s = dict(hidden=hidden, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
             interm=interm, layers=layers)
    core = stack_flops(s, n)
    # input down-proj (teacher_hidden->hidden) + final up-proj (hidden->teacher_hidden)
    proj = 2 * (n * teacher_hidden * hidden) * 2
    return core + proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512, help="sequence length (LIBERO ~392 image + ~100 text/state)")
    ap.add_argument("--layers", type=int, default=36, help="student layers (keep 36 for action-expert per-layer KV)")
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=28)
    ap.add_argument("--interm", type=int, default=896)
    a = ap.parse_args()

    n = a.seq
    t = stack_flops(TEACHER, n)
    s = student_flops(a.hidden, a.heads, a.kv_heads, a.head_dim, a.interm, a.layers, n, TEACHER["hidden"])
    comp = t / s

    # rough param count (stack only): per layer qkv+o + mlp(3) + norms
    q = a.heads * a.head_dim
    kv = a.kv_heads * a.head_dim
    p_attn = a.hidden * (q + 2 * kv) + q * a.hidden
    p_mlp = 3 * a.hidden * a.interm
    p_stack = a.layers * (p_attn + p_mlp)
    p_proj = 2 * TEACHER["hidden"] * a.hidden
    params = p_stack + p_proj

    print(f"Teacher LLM stack (36x{TEACHER['hidden']}, interm {TEACHER['interm']}): "
          f"{t/1e9:.1f} GFLOPs @ seq={n}")
    print(f"Student LLM  ({a.layers}x{a.hidden}, heads {a.heads}x{a.head_dim}, "
          f"kv {a.kv_heads}x{a.head_dim}, interm {a.interm}): {s/1e9:.2f} GFLOPs")
    print(f"  stack params ~ {p_stack/1e6:.1f}M + proj {p_proj/1e6:.1f}M = {params/1e6:.1f}M")
    flag = "OK  >=100x" if comp >= 100 else "!!  <100x (shrink)"
    print(f"Compression: {comp:.1f}x   [{flag}]")


if __name__ == "__main__":
    main()
