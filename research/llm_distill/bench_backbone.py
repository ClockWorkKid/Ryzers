"""Micro-benchmark: teacher-width LLM backbone vs qwen06w student backbone.

Latency + peak memory depend only on shapes/dtype, not weight values, so we
instantiate both backbones (random init) with the *exact* MolmoAct2 configs and
measure a single full-sequence prefill (the forward the flow-matching action
expert consumes per-layer KV from). bf16, single GPU, LIBERO fused length.
"""
import time, json, argparse
import torch
import student as S


def build(cfg_over):
    stu, c = S.build_student(cfg_over)
    return stu, c


def params(m):
    return sum(p.numel() for p in m.parameters())


def bench(model, N, B, dtype, device, iters=20, warmup=5):
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(B, N, model.c.teacher_hidden, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs_embeds=x, collect_layer_kv_states=True)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(inputs_embeds=x, collect_layer_kv_states=True)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device)
    ms = (t1 - t0) / iters * 1e3
    return ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=479)     # LIBERO full fused length
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    print(f"[bench] device={torch.cuda.get_device_name(0)} dtype=bf16 "
          f"seq={args.seq} batch={args.batch} iters={args.iters}", flush=True)

    teacher_cfg = dict(hidden=2560, num_heads=32, intermediate=9728)  # MolmoAct2 teacher backbone
    student_cfg = dict(preset="qwen06w")

    results = {}
    for name, cfg in [("teacher", teacher_cfg), ("qwen06w", student_cfg)]:
        m, c = build(cfg)
        n = params(m)
        # weight memory (bf16) then run
        ms, peak = bench(m, args.seq, args.batch, dtype, device, iters=args.iters)
        results[name] = dict(
            params_M=round(n / 1e6, 1),
            weight_MB_bf16=round(n * 2 / 1e6, 1),
            latency_ms=round(ms, 2),
            peak_mem_MB=round(peak / 1e6, 1),
            cfg=dict(hidden=c.hidden, num_heads=c.num_heads,
                     num_kv_heads=c.num_kv_heads, head_dim=c.head_dim,
                     intermediate=c.intermediate, num_layers=c.num_layers),
        )
        del m
        torch.cuda.empty_cache()
        print(f"[bench] {name}: {results[name]}", flush=True)

    t, s = results["teacher"], results["qwen06w"]
    results["ratios"] = dict(
        param_reduction=round(t["params_M"] / s["params_M"], 2),
        weight_reduction=round(t["weight_MB_bf16"] / s["weight_MB_bf16"], 2),
        latency_speedup=round(t["latency_ms"] / s["latency_ms"], 2),
        peak_mem_reduction=round(t["peak_mem_MB"] / s["peak_mem_MB"], 2),
        analytic_flop_reduction=round(S.student_flop_ratio(S.make_student_config(student_cfg)), 2),
    )
    print("[bench] RATIOS " + json.dumps(results["ratios"]), flush=True)
    print("[bench] RESULT_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
