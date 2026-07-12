"""Run D closed-loop eval launcher: swap in the jointly distilled student ViT +
thin-twin student LLM (+ optional LoRA adapters) at inference so LIBERO rollouts
measure the compressed model, not the teacher.

Three monkey-patches installed before ``lerobot-eval`` builds the policy:
  1. ``MolmoAct2VisionBackbone.encode_image`` -> route through ``student_vit``.
  2. ``MolmoAct2Policy.__init__`` -> attach ``student_llm`` + KV adapters (+ ViT
     student, + optional LoRA) matching the trained architecture BEFORE the
     ``--policy.path`` checkpoint loads, so trained weights fill in.
  3. ``MolmoAct2Model.generate_actions_from_inputs`` -> when the action expert
     asks for encoder KV, build it from the STUDENT LLM (embeds via student ViT
     -> student LLM -> per-layer KV adapters 208->1024) instead of running the
     teacher transformer. This is the exact training-time KV path, minus the
     teacher, so the action expert consumes student KV during rollouts.

Env:
  LLMD_STUDENT_CKPT   Stage-1 student_final.pt (for arch/cfg; weights overwritten)
  LLMD_STUDENT_MODULE path to student.py (default /outputs/llm_distill/student.py)
  VIT_STUDENT_MODULE  path to vit_student.py (default /outputs/vit_student.py)
  VIT_STUDENT_KWARGS  JSON build kwargs for the distilled ViT
  VIT_STUDENT_CKPT    optional ViT placeholder (weights overwritten by --policy.path)
  EVAL_LORA           "1" to inject LoRA arch (eval a LoRA-adapted checkpoint)
  LORA_MODULE         path to lora.py (default /outputs/llm_distill/lora.py)
  LORA_R/LORA_ALPHA/LORA_DROPOUT   must match the LoRA training (16/32/0.05)

All other CLI args forward verbatim to lerobot-eval.
"""

import importlib
import importlib.util
import json
import os
import sys


def _load_module(name, env_key, default):
    path = os.environ.get(env_key, default)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_joint_eval():
    import torch

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    st_ckpt = os.environ.get("LLMD_STUDENT_CKPT")
    if not st_ckpt:
        raise RuntimeError("[joint-eval] LLMD_STUDENT_CKPT unset - required for arch/cfg")
    vit_kwargs = json.loads(os.environ.get("VIT_STUDENT_KWARGS", "{}") or "{}")
    vit_ph = os.environ.get("VIT_STUDENT_CKPT")
    eval_lora = os.environ.get("EVAL_LORA", "0") == "1"
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    smod = _load_module("llmd_student", "LLMD_STUDENT_MODULE", "/outputs/llm_distill/student.py")
    vmod = _load_module("vit_student", "VIT_STUDENT_MODULE", "/outputs/vit_student.py")
    lmod = _load_module("llmd_lora", "LORA_MODULE", "/outputs/llm_distill/lora.py") if eval_lora else None

    # ---- (1) route vision through the student ViT ----
    _orig_encode = HFM.MolmoAct2VisionBackbone.encode_image

    def encode_image(self, images):
        st = getattr(self, "student_vit", None)
        return _orig_encode(self, images) if st is None else st(images)

    HFM.MolmoAct2VisionBackbone.encode_image = encode_image

    # ---- helpers (mirror the training joint loss) ----
    def _student_bias(attention_mask, seq_len, device, dtype):
        neg = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), neg, device=device, dtype=dtype), diagonal=1
        )
        bias = causal[None, None]
        if attention_mask is not None and attention_mask.ndim == 2:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * neg
            bias = bias + pad
        return bias

    def _flatten_kv(kv):
        b, h, n, d = kv.shape
        return kv.transpose(1, 2).reshape(b, n, h * d)

    # ---- (3) student-KV action generation ----
    _orig_gen = HFM.MolmoAct2Model.generate_actions_from_inputs

    def generate_actions_from_inputs(self, *, input_ids, pixel_values=None,
                                     image_token_pooling=None, image_grids=None,
                                     image_num_crops=None, pixel_values_videos=None,
                                     video_token_pooling=None, video_grids=None,
                                     attention_mask=None, token_type_ids=None,
                                     encoder_kv_states=None, encoder_attention_mask=None,
                                     **kwargs):
        student = getattr(self, "student_llm", None)
        if encoder_kv_states is None and student is not None:
            k_ad = self.kv_k_adapters
            v_ad = self.kv_v_adapters
            images, token_pooling = self.merge_visual_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )
            inputs_embeds, _ = self.build_input_embeddings(input_ids, images, token_pooling)
            seq_len = inputs_embeds.shape[1]
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            bias = _student_bias(attention_mask, seq_len, device, dtype)
            # positions default inside the student = arange(N) expanded to [B,N],
            # which is the correct full-prefill layout for a rollout step.
            sout = student(inputs_embeds, attention_bias=bias, positions=None)
            student_kv = sout["kv_states"]
            encoder_kv_states = [
                (k_ad[i](_flatten_kv(k)), v_ad[i](_flatten_kv(v)))
                for i, (k, v) in enumerate(student_kv)
            ]
            encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        return _orig_gen(
            self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

    HFM.MolmoAct2Model.generate_actions_from_inputs = generate_actions_from_inputs

    # ---- (2) attach student arch before checkpoint load ----
    _orig_init = POL.MolmoAct2Policy.__init__

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        backbone = self._backbone()
        if getattr(backbone, "student_llm", None) is not None:
            return
        import torch.nn as nn

        raw = torch.load(st_ckpt, map_location="cpu", weights_only=False)
        cfg = raw.get("cfg", {})
        student, heads, c = smod.build_student(cfg)
        n_layers = c.num_layers
        kv_in = c.num_kv_heads * c.head_dim
        teacher_kv = c.teacher_kv_dim
        k_ad = nn.ModuleList(nn.Linear(kv_in, teacher_kv, bias=False) for _ in range(n_layers))
        v_ad = nn.ModuleList(nn.Linear(kv_in, teacher_kv, bias=False) for _ in range(n_layers))

        dtype = next(self.model.parameters()).dtype
        student = student.to(dtype=dtype)
        k_ad = k_ad.to(dtype=dtype)
        v_ad = v_ad.to(dtype=dtype)
        backbone.student_llm = student
        backbone.kv_k_adapters = k_ad
        backbone.kv_v_adapters = v_ad

        vb = getattr(backbone, "vision_backbone", None)
        if vb is None:
            raise RuntimeError("[joint-eval] could not locate vision_backbone")
        st_vit = vmod.build_seam_student(**vit_kwargs)
        if vit_ph and os.path.exists(vit_ph):
            st_vit.load_state_dict(vmod.clean_sd(torch.load(vit_ph, map_location="cpu", weights_only=False)), strict=False)
        st_vit = st_vit.to(dtype=dtype)
        vb.student_vit = st_vit

        if eval_lora:
            n_llm = lmod.inject_lora(student, ("qkv", "o", "gate_up", "down"),
                                     r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            n_vit = lmod.inject_lora(st_vit, ("qkv", "proj", "pw"),
                                     r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            student.to(dtype=dtype)
            st_vit.to(dtype=dtype)
            print(f"[joint-eval] injected LoRA: {n_llm} LLM + {n_vit} ViT layers "
                  f"(r={lora_r} alpha={lora_alpha})", flush=True)

        for p in self.parameters():
            p.requires_grad_(False)
        student.eval()
        st_vit.eval()
        print(f"[joint-eval] attached student_llm + KV adapters + student_vit "
              f"(dtype={dtype}, eval_lora={eval_lora}); weights load from --policy.path", flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    print("[joint-eval] patched encode_image + generate_actions_from_inputs + __init__", flush=True)


def _run_lerobot_eval():
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group="console_scripts")
    except TypeError:
        eps = entry_points().get("console_scripts", [])
    target = None
    for ep in eps:
        if ep.name == "lerobot-eval":
            target = ep.load()
            break
    if target is None:
        for modname in ("lerobot.scripts.lerobot_eval", "lerobot.scripts.eval"):
            try:
                mod = importlib.import_module(modname)
                target = getattr(mod, "main", None) or getattr(mod, "eval_main", None)
                if target:
                    break
            except Exception:
                continue
    if target is None:
        raise RuntimeError("could not locate lerobot-eval entry point")
    return target()


if __name__ == "__main__":
    _install_joint_eval()
    _run_lerobot_eval()
