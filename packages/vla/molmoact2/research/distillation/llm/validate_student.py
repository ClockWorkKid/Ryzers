"""Module-1 validation: does StudentTextModel satisfy the teacher's KV contract on REAL data?

On a real LIBERO batch (via data.capture_teacher, byte-identical to training):
  1) shapes: student last_hidden_state == [B,N,2560]; 36 KV layers; each (k,v) == teacher's
     [B, num_kv_heads, N, head_dim].
  2) collect path (collect_layer_kv_states=True -> tuple) and inference path (use_cache=True
     -> DynamicCache) return the SAME KV (so both training and rollout route identically).
  3) the student's KV is CONSUMABLE by the frozen action expert: run _extract_kv_states on
     the student's use_cache output and confirm it yields 36 x [B,N,1024] with no error, then
     project through the action expert's context_{k,v}_proj without shape errors.

This does NOT check task quality (student is untrained here) -- only the interface contract.
"""

from __future__ import annotations
import argparse, os
import torch


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.environ.get("MM2_CKPT", "allenai/MolmoAct2-LIBERO"))
    ap.add_argument("--repo-id", default="allenai/MolmoAct2-LIBERO-Dataset")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--init", default="", help="optional warm-start .pt to load before checks")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ok = True

    import data as D
    import student as S
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    print(f"[val] loading teacher {args.teacher}", flush=True)
    model = HFM.MolmoAct2ForConditionalGeneration.from_pretrained(
        args.teacher, torch_dtype=dt).to(device).eval()

    loader = D.iter_libero_batches(args, device)
    cap = D.capture_teacher(model, next(loader), collect_kv=True)
    ie = cap["inputs_embeds"]
    B, N, H = ie.shape
    tkv = cap["teacher"]["kv"]
    print(f"[val] real batch: inputs_embeds={tuple(ie.shape)} dtype={ie.dtype} "
          f"teacher_kv_layers={len(tkv)} k0={tuple(tkv[0][0].shape)}", flush=True)

    stu, c = S.build_student({"preset": args.preset})
    if args.init:
        raw = torch.load(args.init, map_location="cpu", weights_only=False)
        ms, us = stu.load_state_dict(raw["student"], strict=False)
        print(f"[val] warm-start from {args.init}: missing={len(ms)} unexpected={len(us)}", flush=True)
    stu = stu.to(device=device, dtype=dt).eval()
    n = sum(p.numel() for p in stu.parameters())
    print(f"[val] student preset={args.preset} params={n/1e6:.1f}M "
          f"approx_reduction={S.student_flop_ratio(c):.1f}x", flush=True)

    # --- 1) collect path shapes ---
    out_c = stu(inputs_embeds=ie, attention_mask=cap["attn_bias"],
                position_ids=cap["positions"], collect_layer_kv_states=True)
    skv = list(out_c.past_key_values)
    lhs = out_c.last_hidden_state
    checks = []
    checks.append(("last_hidden_state shape", tuple(lhs.shape) == (B, N, 2560)))
    checks.append(("num kv layers == teacher", len(skv) == len(tkv)))
    kshape_ok = all(tuple(skv[i][0].shape) == tuple(tkv[i][0].shape) and
                    tuple(skv[i][1].shape) == tuple(tkv[i][1].shape) for i in range(len(skv)))
    checks.append(("per-layer KV shapes == teacher", kshape_ok))
    finite_ok = torch.isfinite(lhs).all().item() and all(
        torch.isfinite(skv[i][0]).all().item() and torch.isfinite(skv[i][1]).all().item()
        for i in range(len(skv)))
    checks.append(("student outputs finite (no NaN/Inf)", finite_ok))
    print(f"[val] student collect: last={tuple(lhs.shape)} kv_layers={len(skv)} "
          f"k0={tuple(skv[0][0].shape)}", flush=True)
    # informational: KV cosine-to-teacher (NOT a gate; functional distill doesn't need it)
    import torch.nn.functional as F
    def _flat(x):
        b, h, nn_, d = x.shape
        return x.transpose(1, 2).reshape(b, nn_, h * d).float()
    if args.init:
        for li in (0, len(skv) // 2, len(skv) - 1):
            ck = F.cosine_similarity(_flat(skv[li][0]), _flat(tkv[li][0]), dim=-1).mean().item()
            cv = F.cosine_similarity(_flat(skv[li][1]), _flat(tkv[li][1]), dim=-1).mean().item()
            print(f"[val][info] warm-start KV cos vs teacher L{li:2d}: k={ck:+.3f} v={cv:+.3f} "
                  f"|sK|={_flat(skv[li][0]).norm(dim=-1).mean():.1f} "
                  f"|tK|={_flat(tkv[li][0]).norm(dim=-1).mean():.1f}", flush=True)

    # --- 2) use_cache path == collect path ---
    out_u = stu(inputs_embeds=ie, attention_mask=cap["attn_bias"],
                position_ids=cap["positions"], use_cache=True)
    pkv = out_u.past_key_values
    # extract via the REAL model helper (proves DynamicCache is consumable)
    try:
        ukv = model.model._extract_kv_states(pkv)
        checks.append(("_extract_kv_states layers==36", len(ukv) == len(tkv)))
        checks.append(("_extract_kv_states flat dim==1024",
                       tuple(ukv[0][0].shape) == (B, N, c.kv_dim)))
        # consistency: DynamicCache KV (pre-flatten) equals collect-path KV
        # rebuild [B,H,N,D] from cache to compare
        same = True
        for i in range(len(tkv)):
            kc = model.model._cache_to_sequence(skv[i][0])  # [B,N,1024] from collect
            if not torch.allclose(kc.float(), ukv[i][0].float(), atol=1e-3, rtol=1e-3):
                same = False
                break
        checks.append(("collect KV == use_cache KV", same))
    except Exception as e:
        checks.append((f"_extract_kv_states ({type(e).__name__}: {str(e)[:60]})", False))

    # --- 3) consumable by frozen action expert projections ---
    try:
        ae = model.model.action_expert
        ctx_k = ae._project_kv_tensor(ukv[0][0].to(dt), ae.context_k_proj)
        checks.append(("action-expert context_k_proj accepts student KV",
                       ctx_k.shape[0] == B and ctx_k.shape[1] == N))
        print(f"[val] action-expert projected k ctx shape={tuple(ctx_k.shape)}", flush=True)
    except Exception as e:
        checks.append((f"action-expert projection ({type(e).__name__}: {str(e)[:60]})", False))

    print("\n[val] ===== RESULTS =====", flush=True)
    for name, passed in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}", flush=True)
    print(f"\n[val] MODULE-1 {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
