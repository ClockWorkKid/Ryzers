"""Microbenchmark: teacher ViT seam (MolmoAct2VisionBackbone.encode_image) vs the
distilled student, ViT part ONLY, on one GPU in bf16.

Reports mean latency (ms) per encoder call for a few crop-batch sizes, plus
throughput and the measured speedup (to compare against the ~114x FLOP ratio).
"""

import importlib.util
import sys
import time

import torch


def load_student():
    spec = importlib.util.spec_from_file_location("vit_student", "/outputs/vit_student.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["vit_student"] = m
    spec.loader.exec_module(m)
    st = m.build_seam_student()
    raw = torch.load("/outputs/hybrid_full.pt", map_location="cpu", weights_only=False)
    st.load_state_dict(m.clean_sd(raw), strict=False)
    return st


def load_teacher_backbone():
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as M

    model = M.MolmoAct2ForConditionalGeneration.from_pretrained(
        "allenai/MolmoAct2-LIBERO", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # locate the MolmoAct2VisionBackbone instance regardless of nesting
    for mod in model.modules():
        if isinstance(mod, M.MolmoAct2VisionBackbone):
            return mod
    raise RuntimeError("MolmoAct2VisionBackbone not found in model")


def bench(fn, x, iters=50, warmup=15):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000.0


def main():
    dev = "cuda"
    dt = torch.bfloat16
    student = load_student().to(dev, dt).eval()
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[bench] student params={n_s/1e6:.2f}M", flush=True)
    vb = load_teacher_backbone().to(dev, dt).eval()
    n_t = sum(p.numel() for p in vb.parameters())
    print(f"[bench] teacher vision_backbone params={n_t/1e6:.2f}M", flush=True)

    print(f"{'crops':>6} {'teacher_ms':>12} {'student_ms':>12} {'speedup':>9} "
          f"{'teach_crop/s':>13} {'stud_crop/s':>12}")
    with torch.inference_mode():
        for C in (1, 2, 4, 8):
            x = torch.randn(1, C, 729, 588, device=dev, dtype=dt)
            # normalize into [-1,1]-ish range like the real forward feeds encode_image
            x = x.clamp(-1, 1)
            t_ms = bench(lambda z: vb.encode_image(z), x)
            s_ms = bench(lambda z: student(z), x)
            print(f"{C:>6} {t_ms:>12.3f} {s_ms:>12.3f} {t_ms/s_ms:>8.1f}x "
                  f"{C/(t_ms/1000):>13.1f} {C/(s_ms/1000):>12.1f}", flush=True)


if __name__ == "__main__":
    main()
