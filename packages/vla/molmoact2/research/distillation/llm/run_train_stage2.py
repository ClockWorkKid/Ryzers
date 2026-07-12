"""Run D - Stage 2 launcher: assemble the distilled thin-twin LLM (from Stage 1)
into the MolmoAct2 policy in place of the teacher text transformer, feed its
per-layer KV into the FROZEN flow-matching action expert via small learned
adapters, and jointly train with the flow-matching task loss + a multi-level
distillation term against the frozen teacher.

Design (see research/llm_distill/README):
  * The teacher text transformer is BYPASSED for action training. The student
    LLM (36 layers, hidden 208) consumes the same ``inputs_embeds`` the teacher
    would (image+text+proprio tokens) and produces 36 layers of raw KV.
  * The action expert requires one block per LLM layer and ingests LLM KV of dim
    ``llm_kv_dim = 1024`` (8 heads x 128). Student raw KV is 208 (8 x 26), so we
    insert per-layer adapters (208 -> 1024) whose weights are warm-started from
    the Stage-1 DistillHeads k/v projectors. The frozen ``context_{k,v}_proj``
    then map 1024 -> action hidden unchanged.
  * Trainable: student LLM + KV adapters (routed to the ``vlm`` optim group by
    name). Frozen: action expert, teacher transformer, connector. The ViT may
    optionally be swapped for a distilled student (VIT_STUDENT_CKPT) reusing the
    proven vit_distill encode_image patch.

Because the full student LLM replaces the teacher LLM, LoRA on the teacher
transformer is redundant and is left OFF; we train the compact student directly.

Env:
  LLMD_STUDENT_CKPT   path to Stage-1 student_final.pt (required)
  LLMD_STUDENT_MODULE path to student.py (default /outputs/llm_distill/student.py)
  LLMD_KD_W           weight for the multi-level distillation term (default 0.5;
                      set 0 to disable and train task-loss only)
  VIT_STUDENT_CKPT    distilled ViT checkpoint to swap in + train (joint ViT+LLM)
  VIT_STUDENT_MODULE  path to vit_student.py (default /outputs/vit_student.py)
  VIT_STUDENT_KWARGS  JSON build kwargs for build_seam_student (e.g. hybrid dims)
  DROID_ROOT          dir of DROID JPEGs; enables the ViT seam-retention term
  DROID_MODULE        path to droid_data.py (default /outputs/llm_distill/droid_data.py)
  DROID_W             weight for the seam-retention term (default 1.0)
  DROID_BS            DROID images per step (default 4)
  DROID_WORKERS       DROID dataloader workers (default 2)
  DROID_MAX_FRAMES    optional cap on number of DROID frames used

All other CLI args are forwarded verbatim to lerobot-train.
"""

import importlib.util
import os
import sys


def _load_module(name, env_key, default):
    path = os.environ.get(env_key, default)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _maybe_install_vit_student():
    """Optional distilled ViT swap, reusing the vit_distill pattern.

    Returns (vit_student_module, teacher_encode_fn) or (None, None). The teacher
    encode fn is the ORIGINAL (unpatched) encode_image, used by the DROID
    seam-retention term to produce teacher targets on real frames.
    """
    ckpt = os.environ.get("VIT_STUDENT_CKPT")
    if not ckpt:
        return None, None
    from lerobot.policies.molmoact2.molmoact2_hf_model import modeling_molmoact2 as HFM

    vs = _load_module("vit_student", "VIT_STUDENT_MODULE", "/outputs/vit_student.py")
    _orig_encode = HFM.MolmoAct2VisionBackbone.encode_image

    def encode_image(self, images):
        st = getattr(self, "student_vit", None)
        return _orig_encode(self, images) if st is None else st(images)

    HFM.MolmoAct2VisionBackbone.encode_image = encode_image
    return vs, _orig_encode


def _install_stage2():
    import json
    import torch
    import torch.nn as nn

    from lerobot.policies.molmoact2 import modeling_molmoact2 as POL

    ckpt = os.environ.get("LLMD_STUDENT_CKPT")
    if not ckpt:
        raise RuntimeError("[stage2] LLMD_STUDENT_CKPT unset - required for Run D Stage 2")
    kd_w = float(os.environ.get("LLMD_KD_W", "0.5"))
    vit_ckpt = os.environ.get("VIT_STUDENT_CKPT")
    vit_kwargs = json.loads(os.environ.get("VIT_STUDENT_KWARGS", "{}") or "{}")
    droid_root = os.environ.get("DROID_ROOT")
    droid_w = float(os.environ.get("DROID_W", "1.0"))
    droid_bs = int(os.environ.get("DROID_BS", "4"))
    droid_workers = int(os.environ.get("DROID_WORKERS", "2"))
    droid_max = os.environ.get("DROID_MAX_FRAMES")
    droid_max = int(droid_max) if droid_max else None
    smod = _load_module("llmd_student", "LLMD_STUDENT_MODULE", "/outputs/llm_distill/student.py")
    vit_mod, teacher_encode = _maybe_install_vit_student()
    dmod = _load_module("droid_data", "DROID_MODULE", "/outputs/llm_distill/droid_data.py") if droid_root else None

    # ---- LoRA adaptation stage (Phase 4): freeze the joint-distilled backbone and
    # train only low-rank adapters on the student submodules ----
    lora_stage = os.environ.get("LORA_STAGE", "0") == "1"
    joint_ckpt = os.environ.get("JOINT_CKPT")  # joint run's saved policy weights
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    lmod = _load_module("llmd_lora", "LORA_MODULE", "/outputs/llm_distill/lora.py") if lora_stage else None
    if lora_stage and not joint_ckpt:
        raise RuntimeError("[lora] LORA_STAGE=1 requires JOINT_CKPT (joint-distilled policy weights)")

    _expand_mask = POL._expand_mask
    ACTION = POL.ACTION

    # ---- causal additive bias for the student (decoupled from teacher mask dict) ----
    def _student_bias(attention_mask, seq_len, device, dtype):
        neg = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), neg, device=device, dtype=dtype), diagonal=1
        )
        bias = causal[None, None]  # [1,1,N,N]
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * neg  # [B,1,1,N]
            bias = bias + pad
        return bias

    def _flatten_kv(kv):
        # kv: [B, H, N, D] -> [B, N, H*D]
        b, h, n, d = kv.shape
        return kv.transpose(1, 2).reshape(b, n, h * d)

    # -------- patched joint per-layer flow loss: student KV -> frozen action expert --------
    def _stage2_joint_loss(self, *, batch, model_inputs, timesteps=None, noise=None, reduction="mean"):
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}.")
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        action_expert = backbone._require_action_expert()
        student = backbone.student_llm
        k_ad = backbone.kv_k_adapters
        v_ad = backbone.kv_v_adapters
        if transformer is None:
            raise RuntimeError("[stage2] no transformer to size the action expert against.")
        n_layers = int(transformer.config.num_hidden_layers)
        if len(action_expert.blocks) != n_layers:
            raise RuntimeError("[stage2] action expert / LLM layer count mismatch.")

        actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            timesteps=timesteps,
            noise=noise,
        )
        num_ts = max(1, int(self.config.num_flow_timesteps))
        bsz = int(actions.shape[0])
        device = actions.device
        xt_flat = xt.reshape(bsz * num_ts, actions.shape[1], actions.shape[2])
        ts_flat = timesteps.reshape(bsz * num_ts)

        inputs_embeds, _causal_map, position_ids, _cache_pos = (
            self._prepare_joint_training_backbone_inputs(model_inputs)
        )
        if inputs_embeds.shape[0] != bsz:
            raise ValueError("[stage2] backbone/action batch size mismatch.")
        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        student_bias = _student_bias(model_inputs.get("attention_mask"), seq_len, device, dtype)

        # ---- single student forward: all 36 layers' raw KV + final hidden ----
        sout = student(inputs_embeds, attention_bias=student_bias, positions=position_ids)
        student_kv = sout["kv_states"]              # list[36] of (k[B,8,N,26], v)
        student_last = sout["last_hidden_state"]    # [B,N,2560] for discrete loss

        # ---- adapt student KV 208 -> 1024 per layer (frozen context_proj unchanged) ----
        # k_seq/v_seq stay attached for the distillation term; the action expert
        # consumes detached copies when knowledge insulation is enabled.
        k_seq, v_seq = [], []
        for i in range(n_layers):
            k_raw, v_raw = student_kv[i]
            k_seq.append(k_ad[i](_flatten_kv(k_raw)))   # [B,N,1024]
            v_seq.append(v_ad[i](_flatten_kv(v_raw)))
        ki = bool(self.config.enable_knowledge_insulation)

        # ---- action-expert masks / conditioning (verbatim from stock path) ----
        enc_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        action_attention_mask = None
        if batch.get("action_horizon_is_pad") is not None:
            action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=actions.dtype).unsqueeze(-1)
            valid_action = _expand_mask(valid_action, num_ts)

        rope_cache = None
        if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
            rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
                seq_len=actions.shape[1], device=device, dtype=actions.dtype
            )
        cross_mask = _expand_mask(
            action_expert._build_cross_attention_mask(enc_mask, bsz, actions.dtype), num_ts
        )
        self_mask = _expand_mask(
            action_expert._build_self_attention_mask(
                action_attention_mask, actions.shape[1], device, actions.dtype
            ),
            num_ts,
        )
        conditioning = self._action_time_conditioning(action_expert, ts_flat)
        action_hidden = action_expert.action_embed(xt_flat)
        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        for i in range(n_layers):
            action_block = action_expert.blocks[i]
            k_in = k_seq[i].detach() if ki else k_seq[i]
            v_in = v_seq[i].detach() if ki else v_seq[i]
            k_ctx = action_expert._project_kv_tensor(k_in, action_expert.context_k_proj)
            v_ctx = action_expert._project_kv_tensor(v_in, action_expert.context_v_proj)
            k_norm = action_block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            if num_ts != 1:
                k_ctx = _expand_mask(k_ctx, num_ts)
                v_ctx = _expand_mask(v_ctx, num_ts)
            action_hidden = action_block(
                action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                action_hidden = action_hidden * valid_action

        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action
        pred_velocity = pred_velocity.reshape(bsz, num_ts, actions.shape[1], actions.shape[2])

        loss = POL.F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = POL._apply_action_chunk_padding_mask(loss, batch.get("action_horizon_is_pad"))
        if self.config.mask_action_dim_padding:
            loss = POL._apply_action_dim_padding_mask(loss, batch.get("action_dim_is_pad"))
        loss = loss.reshape(bsz, -1).mean(dim=1)
        task_loss = loss.mean() if reduction == "mean" else loss

        # ---- multi-level distillation vs frozen teacher (KV @ 1024 + final hidden) ----
        if kd_w > 0.0:
            with torch.no_grad():
                tout = transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=_causal_map,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    collect_layer_kv_states=True,
                    use_cache=False,
                )
                t_kv = list(tout.past_key_values)          # 36 x (k,v) [B,8,N,128]
                t_last = tout.last_hidden_state            # [B,N,2560]
            am = model_inputs.get("attention_mask")
            m = torch.ones(bsz, seq_len, device=device, dtype=dtype) if am is None else am.to(dtype)
            m3 = m[:, :, None]
            denom = m3.sum().clamp_min(1.0)
            kv_kd = 0.0
            for i in range(n_layers):
                tk = backbone._cache_to_sequence(t_kv[i][0])
                tv = backbone._cache_to_sequence(t_kv[i][1])
                kv_kd = kv_kd + (((k_seq[i] - tk) ** 2) * m3).sum() / (denom * k_seq[i].shape[-1])
                kv_kd = kv_kd + (((v_seq[i] - tv) ** 2) * m3).sum() / (denom * v_seq[i].shape[-1])
            kv_kd = kv_kd / n_layers
            hid_kd = (((student_last - t_last) ** 2) * m3).sum() / (denom * student_last.shape[-1])
            kd = kv_kd + hid_kd
            if reduction == "mean":
                task_loss = task_loss + kd_w * kd
            else:
                task_loss = task_loss + kd_w * kd  # scalar KD added to per-example vector

        return task_loss, student_last

    def _collect_joint_tensors(path, markers):
        """Return only the tensors whose key contains one of ``markers``, reading
        lazily from a (possibly sharded) safetensors dir/file or a torch file. This
        avoids materializing the ~7B frozen teacher into RAM."""
        out = {}

        def _keep(k, v):
            if any(m in k for m in markers):
                out[k] = v

        if os.path.isdir(path):
            idx = os.path.join(path, "model.safetensors.index.json")
            single = os.path.join(path, "model.safetensors")
            binp = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(idx):
                import json as _json
                from safetensors import safe_open
                wm = _json.load(open(idx))["weight_map"]
                shards = sorted(set(wm.values()))
                for s in shards:
                    with safe_open(os.path.join(path, s), framework="pt", device="cpu") as sf:
                        for k in sf.keys():
                            if any(m in k for m in markers):
                                out[k] = sf.get_tensor(k)
            elif os.path.exists(single):
                from safetensors import safe_open
                with safe_open(single, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        if any(m in k for m in markers):
                            out[k] = sf.get_tensor(k)
            elif os.path.exists(binp):
                for k, v in torch.load(binp, map_location="cpu").items():
                    _keep(k, v)
            else:
                raise RuntimeError(f"[lora] no model weights found under {path}")
        elif path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if any(m in k for m in markers):
                        out[k] = sf.get_tensor(k)
        else:
            for k, v in torch.load(path, map_location="cpu", weights_only=False).items():
                _keep(k, v)
        return out

    def _load_submodule_from_joint(full_sd, marker, submodule, label):
        """Slice keys of full_sd on the first occurrence of ``marker`` and load the
        resulting sub-state-dict into ``submodule`` (prefix-agnostic)."""
        sub = {}
        for k, v in full_sd.items():
            idx = k.find(marker)
            if idx >= 0:
                sub[k[idx + len(marker):]] = v
        miss, unexp = submodule.load_state_dict(sub, strict=False)
        got = len(sub)
        print(f"[lora] loaded {label}: {got} tensors from joint ckpt "
              f"(missing={len(miss)} unexpected={len(unexp)})", flush=True)
        if got == 0:
            raise RuntimeError(f"[lora] found 0 '{marker}' tensors in JOINT_CKPT - key layout mismatch")
        return got

    # -------- attach student + adapters after policy init; freeze action expert --------
    _orig_init = POL.MolmoAct2Policy.__init__

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        backbone = self._backbone()
        if getattr(backbone, "student_llm", None) is not None:
            return
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = raw.get("cfg", {})
        student, heads, c = smod.build_student(cfg)
        student.load_state_dict(raw["student"], strict=True)
        if "heads" in raw:
            heads.load_state_dict(raw["heads"], strict=True)

        n_layers = c.num_layers
        kv_in = c.num_kv_heads * c.head_dim          # 208
        teacher_kv = c.teacher_kv_dim                # 1024
        k_ad = nn.ModuleList(nn.Linear(kv_in, teacher_kv, bias=False) for _ in range(n_layers))
        v_ad = nn.ModuleList(nn.Linear(kv_in, teacher_kv, bias=False) for _ in range(n_layers))
        # warm-start adapters from Stage-1 DistillHeads k/v projectors
        for i in range(n_layers):
            k_ad[i].weight.data.copy_(heads.k_heads[i].weight.data)
            v_ad[i].weight.data.copy_(heads.v_heads[i].weight.data)

        dtype = next(self.model.parameters()).dtype
        student = student.to(dtype=dtype)
        k_ad = k_ad.to(dtype=dtype)
        v_ad = v_ad.to(dtype=dtype)
        for p in student.parameters():
            p.requires_grad_(True)
        for p in list(k_ad.parameters()) + list(v_ad.parameters()):
            p.requires_grad_(True)
        student.train()

        backbone.student_llm = student
        backbone.kv_k_adapters = k_ad
        backbone.kv_v_adapters = v_ad

        # ---- optional distilled student ViT (joint ViT+LLM) ----
        if vit_ckpt and vit_mod is not None:
            vb = getattr(backbone, "vision_backbone", None)
            if vb is None:
                raise RuntimeError("[stage2] could not locate vision_backbone for ViT swap")
            st_vit = vit_mod.build_seam_student(**vit_kwargs)
            vraw = torch.load(vit_ckpt, map_location="cpu", weights_only=False)
            miss, unexp = st_vit.load_state_dict(vit_mod.clean_sd(vraw), strict=False)
            if miss or unexp:
                print(f"[stage2] ViT load_state_dict missing={list(miss)} unexpected={list(unexp)}", flush=True)
            st_vit = st_vit.to(dtype=dtype)
            st_vit.train()
            vb.student_vit = st_vit
            n_v = sum(p.numel() for p in st_vit.parameters())
            print(f"[stage2] attached student_vit ({n_v/1e6:.2f}M) kwargs={vit_kwargs} <- {vit_ckpt}", flush=True)

        # ---- LoRA adaptation stage: load joint-trained weights, wrap adapters ----
        if lora_stage:
            full_sd = _collect_joint_tensors(
                joint_ckpt,
                ("student_llm.", "kv_k_adapters.", "kv_v_adapters.", "student_vit."),
            )
            _load_submodule_from_joint(full_sd, "student_llm.", student, "student_llm")
            _load_submodule_from_joint(full_sd, "kv_k_adapters.", k_ad, "kv_k_adapters")
            _load_submodule_from_joint(full_sd, "kv_v_adapters.", v_ad, "kv_v_adapters")
            vit_st = getattr(getattr(backbone, "vision_backbone", None), "student_vit", None)
            if vit_st is not None:
                _load_submodule_from_joint(full_sd, "student_vit.", vit_st, "student_vit")
            # inject LoRA AFTER loading base weights (LoRA renames base Linears)
            n_llm = lmod.inject_lora(student, ("qkv", "o", "gate_up", "down"),
                                     r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            n_vit = 0
            if vit_st is not None:
                n_vit = lmod.inject_lora(vit_st, ("qkv", "proj", "pw"),
                                         r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            student.to(dtype=dtype)
            if vit_st is not None:
                vit_st.to(dtype=dtype)
            n_lora, n_ten = lmod.mark_lora_only_trainable(self)
            print(f"[lora] wrapped {n_llm} LLM + {n_vit} ViT Linear layers "
                  f"(r={lora_r} alpha={lora_alpha}); trainable LoRA params="
                  f"{n_lora/1e6:.3f}M across {n_ten} tensors; base frozen; "
                  f"kd_w={kd_w} droid_w={droid_w} <- {joint_ckpt}", flush=True)
            return

        # Explicit control: freeze everything, then unfreeze ONLY the student LLM,
        # its KV adapters, and (if present) the distilled student ViT. This is
        # robust whether or not LoRA wrapped the base model.
        trainable_keys = ("student_llm", "kv_k_adapters", "kv_v_adapters", "student_vit")
        n_tr = 0
        for name, p in self.named_parameters():
            if any(k in name for k in trainable_keys):
                p.requires_grad_(True)
                n_tr += p.numel()
            else:
                p.requires_grad_(False)

        n_s = sum(p.numel() for p in student.parameters())
        n_a = sum(p.numel() for p in list(k_ad.parameters()) + list(v_ad.parameters()))
        print(f"[stage2] attached student_llm ({n_s/1e6:.2f}M) + kv adapters "
              f"({n_a/1e6:.2f}M) dtype={dtype}; total trainable={n_tr/1e6:.2f}M; "
              f"kd_w={kd_w} <- {ckpt}", flush=True)

    POL.MolmoAct2Policy.__init__ = __init__
    POL.MolmoAct2Policy._compute_flow_matching_loss_joint_per_layer = _stage2_joint_loss

    # -------- DROID seam-retention term added to the training loss each step --------
    if dmod is not None:
        _orig_forward = POL.MolmoAct2Policy.forward
        _dstate = {"batcher": None, "warned": False}

        def forward(self, batch, reduction="mean"):
            loss, metrics = _orig_forward(self, batch, reduction)
            if not self.training:
                return loss, metrics
            vb = getattr(self._backbone(), "vision_backbone", None)
            st = getattr(vb, "student_vit", None) if vb is not None else None
            if st is None or teacher_encode is None:
                if not _dstate["warned"]:
                    print("[stage2] DROID_ROOT set but no student_vit attached -> "
                          "retention term skipped (need VIT_STUDENT_CKPT)", flush=True)
                    _dstate["warned"] = True
                return loss, metrics
            if _dstate["batcher"] is None:
                _dstate["batcher"] = dmod.DroidBatcher(
                    droid_root, batch_size=droid_bs, num_workers=droid_workers, max_frames=droid_max
                )
                print(f"[stage2] DROID retention active: root={droid_root} bs={droid_bs} "
                      f"w={droid_w} frames={len(_dstate['batcher'].ds)}", flush=True)
            p = next(self.parameters())
            patches = _dstate["batcher"].next().to(device=p.device, dtype=p.dtype)
            with torch.no_grad():
                t_seam = teacher_encode(vb, patches)
            s_seam = st(patches)
            ret = dmod.seam_cosine_loss(s_seam, t_seam) + dmod.seam_norm_mse_loss(s_seam, t_seam)
            loss = loss + droid_w * ret
            metrics["vit_retention"] = float(ret.detach())
            return loss, metrics

        POL.MolmoAct2Policy.forward = forward
        print("[stage2] patched MolmoAct2Policy.forward with DROID seam-retention", flush=True)

    print("[stage2] patched MolmoAct2Policy.__init__ + joint per-layer flow loss", flush=True)


if __name__ == "__main__":
    _install_stage2()
    from lerobot.scripts.lerobot_train import main

    main()
