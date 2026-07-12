"""Torch-free unit tests for the FLOP profiler + config (run locally, no GPU).

    python -m pytest tests/test_flops.py       # or: python tests/test_flops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.config import StudentConfig, TEACHER_LIBERO
from distill.flops import profile, student_macs, teacher_macs


def test_teacher_matches_known_gflops():
    t = teacher_macs(TEACHER_LIBERO)
    # compute_saved.py reports ~616 GFLOPs for the resblocks; total ~617 incl patch_embed.
    assert 610.0 <= t["gflops"] <= 625.0, t["gflops"]
    assert TEACHER_LIBERO.seam_dim == 2304


def test_default_hybrid_meets_100x():
    rep = profile(StudentConfig())
    assert rep["meets_100x"], rep["compression_x"]
    assert rep["student"]["gflops"] < 6.2, rep["student"]["gflops"]


def test_student_output_dim_is_seam():
    cfg = StudentConfig()
    assert cfg.seam_dim == TEACHER_LIBERO.seam_dim
    # head must map working dim -> seam_dim
    s = student_macs(cfg)
    assert s["total"] > 0


def test_variants_are_configurable():
    for variant, kw in [
        ("hybrid", {}),
        ("cnn", {"dim": 384, "num_conv_blocks": 8}),
        ("tinyvit", {"dim": 224, "num_attn_blocks": 3, "attn_heads": 7}),
    ]:
        rep = profile(StudentConfig(variant=variant, **kw))
        assert rep["compression_x"] > 50.0, (variant, rep["compression_x"])


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
