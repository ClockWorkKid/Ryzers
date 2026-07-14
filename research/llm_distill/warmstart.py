"""Warm-start a StudentTextModel from pretrained Qwen3-0.6B.

The ``qwen06w`` student shares Qwen3-0.6B's WIDTH (hidden 1024, 16 Q / 8 KV heads x 128,
interm 3072) so transformer-layer weights map 1:1, with two adjustments:
  - Qwen3 splits q/k/v and gate/up into separate Linears; the student fuses them
    (qkv = [q;k;v], gate_up = [gate;up]) -> concatenate along the output dim.
  - Depth: Qwen3 has 28 layers, the student needs 36 (fixed by the 1:1 action expert).
    Map student layer i <- Qwen layer round(i * (28-1)/(36-1)) (nearest-layer stretch).

New modules with no Qwen3 counterpart (trained from scratch): ``down_proj`` (2560->hidden,
consumes the teacher's fused embeds) and ``up_proj`` (hidden->2560, feeds the discrete head).
Qwen3's token embedding and lm_head are intentionally NOT used (the student consumes fused
inputs_embeds, not token ids).
"""

from __future__ import annotations
import argparse, os
import torch

import student as S


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default=os.environ.get("QWEN_CKPT", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--preset", default="qwen06w")
    ap.add_argument("--out", default="/outputs/llm_distill/warmstart/qwen06w_init.pt")
    ap.add_argument("--cfg-json", default="{}")
    return ap.parse_args()


def _get(sd, *names):
    for n in names:
        if n in sd:
            return sd[n]
    return None


def remap_qwen_to_student(qsd, c: S.StudentConfig, qwen_layers: int):
    """Return (student_state_dict, report) mapping Qwen3 params -> student param names."""
    out = {}
    report = {"mapped": 0, "skipped_new": [], "missing_src": []}

    def put(name, tensor):
        out[name] = tensor.contiguous().clone()
        report["mapped"] += 1

    # nearest-layer stretch: student layer i <- qwen layer idx
    for i in range(c.num_layers):
        j = round(i * (qwen_layers - 1) / max(c.num_layers - 1, 1)) if c.num_layers > 1 else 0
        p = f"model.layers.{j}."
        q = _get(qsd, p + "self_attn.q_proj.weight")
        k = _get(qsd, p + "self_attn.k_proj.weight")
        v = _get(qsd, p + "self_attn.v_proj.weight")
        o = _get(qsd, p + "self_attn.o_proj.weight")
        qn = _get(qsd, p + "self_attn.q_norm.weight")
        kn = _get(qsd, p + "self_attn.k_norm.weight")
        gate = _get(qsd, p + "mlp.gate_proj.weight")
        up = _get(qsd, p + "mlp.up_proj.weight")
        down = _get(qsd, p + "mlp.down_proj.weight")
        iln = _get(qsd, p + "input_layernorm.weight")
        pln = _get(qsd, p + "post_attention_layernorm.weight")
        if any(t is None for t in (q, k, v, o, gate, up, down, iln, pln)):
            report["missing_src"].append(f"qwen layer {j} for student {i}")
            continue
        sp = f"layers.{i}."
        put(sp + "attn.qkv.weight", torch.cat([q, k, v], dim=0))
        put(sp + "attn.o.weight", o)
        if qn is not None and c.use_qk_norm:
            put(sp + "attn.q_norm.weight", qn)
        if kn is not None and c.use_qk_norm:
            put(sp + "attn.k_norm.weight", kn)
        put(sp + "mlp.gate_up.weight", torch.cat([gate, up], dim=0))
        put(sp + "mlp.down.weight", down)
        put(sp + "attn_norm.weight", iln)
        put(sp + "ff_norm.weight", pln)

    # final norm from Qwen model.norm
    mn = _get(qsd, "model.norm.weight")
    if mn is not None:
        put("final_norm.weight", mn)

    # down_proj / up_proj have no Qwen counterpart -> left to student's default init
    report["skipped_new"] = ["down_proj.weight", "up_proj.weight"]
    return out, report


def main():
    args = get_args()
    import json
    cfg = json.loads(args.cfg_json)
    cfg.setdefault("preset", args.preset)
    stu, c = S.build_student(cfg)

    print(f"[warmstart] loading {args.qwen}", flush=True)
    from transformers import AutoModelForCausalLM
    qwen = AutoModelForCausalLM.from_pretrained(args.qwen, torch_dtype=torch.float32)
    qsd = qwen.state_dict()
    qwen_layers = qwen.config.num_hidden_layers
    print(f"[warmstart] qwen layers={qwen_layers} hidden={qwen.config.hidden_size} "
          f"kv_heads={qwen.config.num_key_value_heads} head_dim={qwen.config.head_dim}", flush=True)

    sd, report = remap_qwen_to_student(qsd, c, qwen_layers)
    ms, us = stu.load_state_dict(sd, strict=False)
    # coverage: how many student params got a warm value
    stu_params = dict(stu.named_parameters())
    warm_keys = set(sd.keys())
    n_warm = sum(stu_params[k].numel() for k in stu_params if k in warm_keys)
    n_tot = sum(p.numel() for p in stu.parameters())
    print(f"[warmstart] mapped tensors={report['mapped']} "
          f"missing_src={len(report['missing_src'])} "
          f"student_load missing={len(ms)} unexpected={len(us)}", flush=True)
    print(f"[warmstart] warm-covered params={n_warm/1e6:.1f}M / {n_tot/1e6:.1f}M "
          f"({100*n_warm/n_tot:.1f}%) ; from-scratch: {sorted(set(k for k in ms))[:8]}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"student": stu.state_dict(), "cfg": c.__dict__,
                "source": args.qwen, "preset": args.preset}, args.out)
    print(f"[warmstart] saved -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
