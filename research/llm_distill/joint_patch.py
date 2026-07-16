"""Inject the width-reduced student into MolmoAct2 for functional (flow-matching) distillation.

Two swap points (same student module, different call paths):
  * TRAINING uses the manual per-layer block loop in
    ``_compute_flow_matching_loss_joint_per_layer`` -> we monkey-patch that method with
    ``student_joint_loss`` which sources per-layer KV from the student's native forward and
    reuses ALL the action-expert machinery (projection, cross/self masks, blocks, final
    layer, flow-matching targets, padding masks).
  * EVAL uses ``generate_actions_from_inputs`` -> ``transformer.forward`` -> ``_extract_kv_states``
    -> we set ``backbone.transformer = student`` (the student mirrors that forward; validated
    in Module 1).

Action-expert co-adaptation: LoRA on ``context_{k,v}_proj`` (the only projections that see
the student's KV). Everything else (ViT, teacher transformer, rest of action expert) frozen.
"""

from __future__ import annotations
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update: y = W0 x + (B A x) * a/r."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero -> adapter starts as identity (no change to the pretrained action expert)

    def forward(self, x):
        out = self.base(x)
        upd = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return out + upd.to(out.dtype) * self.scaling

    # proxy the wrapped weight/bias so code that introspects the linear (e.g. the ViT's
    # ``device``/``dtype`` properties read patch_embedding.weight) keeps working when wrapped.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _module_dtype(m):
    for p in m.parameters():
        return p.dtype
    return torch.float32


def attach_student(policy, student, *, lora_rank=16, lora_alpha=16, lora_dropout=0.0,
                   free_teacher_transformer=True, student_dtype=None,
                   anchor_weight=0.0, anchor_mode="both", anchor_beta=1.0,
                   train_action_expert=False, adapt_ae=None):
    """Attach the student, adapt the action expert, freeze everything else.

    ``student_dtype`` controls the student's param dtype: None -> match the action expert
    (bf16); pass torch.float32 to keep fp32 master weights for stable AdamW (forward should
    then run under bf16 autocast). LoRA A/B always fp32.

    Action-expert adaptation (two modes):
      * ``train_action_expert=False`` (default): LoRA on ``context_{k,v}_proj`` only (the KV
        interface); the rest of the action expert stays frozen. This is the 30%-success recipe.
      * ``train_action_expert=True`` (Experiment D co-distillation): the WHOLE action-expert
        body becomes trainable (self/cross-attn, MLP, final, time/action embeds, context proj).
        The action expert is the teacher AE instance already loaded here, so it is exactly
        "the student action expert, initialized from the teacher AE" -> then co-adapted. No
        LoRA is inserted in this mode.

    Teacher-KV anchoring (hybrid representational term): when ``anchor_weight > 0`` the teacher
    transformer is kept RESIDENT (not freed) so ``student_joint_loss`` can source per-layer
    teacher KV on the same fused ``hidden_states`` and regress the student's KV toward it
    (all 36 layers). ``anchor_mode`` in {cos, mse, both}; ``anchor_beta`` weights the
    magnitude (normalized-MSE) term when mode='both'.

    Returns dict of trainable param groups: {"student": [...], "lora": [...],
    "action_expert": [...]} (only groups that are actually trained are non-empty).
    """
    backbone = policy._backbone()
    ae = backbone._require_action_expert()
    dev = next(ae.parameters()).device
    dt = _module_dtype(ae)
    sdt = student_dtype or dt

    # 1) freeze everything
    for p in policy.parameters():
        p.requires_grad = False

    # 2) attach student (trainable), fp32 or bf16
    student = student.to(device=dev, dtype=sdt)
    for p in student.parameters():
        p.requires_grad = True
    policy.student_llm = student

    # 3) action-expert adaptation -- adapt_ae in {full, lora, none}
    #    full : whole AE trainable (Experiment D co-adaptation)
    #    lora : LoRA on context_{k,v}_proj only (the 30% recipe / default)
    #    none : AE fully frozen; only the student LLM trains (DAgger student-only recipe)
    if adapt_ae is None:
        adapt_ae = "full" if train_action_expert else "lora"
    lora_params = []
    ae_params = []
    if adapt_ae == "full":
        # full co-adaptation: the student action expert is the loaded teacher AE, now trainable.
        ae.to(device=dev, dtype=sdt)
        for p in ae.parameters():
            p.requires_grad = True
        ae_params = [p for p in ae.parameters() if p.requires_grad]
    elif adapt_ae == "none":
        # AE stays entirely frozen (bf16); nothing added -- student LLM is the only trainable group.
        for p in ae.parameters():
            p.requires_grad = False
    else:
        # LoRA on the action-expert context projections (the KV interface); A/B fp32
        ae.context_k_proj = LoRALinear(ae.context_k_proj, lora_rank, lora_alpha, lora_dropout).to(dev)
        ae.context_v_proj = LoRALinear(ae.context_v_proj, lora_rank, lora_alpha, lora_dropout).to(dev)
        ae.context_k_proj.base.to(dt); ae.context_v_proj.base.to(dt)
        for proj in (ae.context_k_proj, ae.context_v_proj):
            proj.lora_A.requires_grad = True
            proj.lora_B.requires_grad = True
            lora_params += [proj.lora_A, proj.lora_B]

    # 4) anchoring config; keep teacher transformer resident iff we anchor to it
    anchoring = float(anchor_weight) > 0.0
    policy._anchor_cfg = {
        "enabled": anchoring, "weight": float(anchor_weight),
        "mode": str(anchor_mode), "beta": float(anchor_beta),
    }
    policy._loss_components = {}
    if free_teacher_transformer and not anchoring:
        # teacher transformer unused in training (student replaces it) -> free the weights
        try:
            for blk in backbone.transformer.blocks:
                blk.to("meta")
        except Exception:
            pass
    else:
        # keep the teacher transformer resident + frozen (bf16) as the KV anchor source
        for p in backbone.transformer.parameters():
            p.requires_grad = False
        backbone.transformer.eval()

    # 5) monkey-patch the joint loss to source KV from the student
    policy._compute_flow_matching_loss_joint_per_layer = types.MethodType(student_joint_loss, policy)

    student_params = [p for p in student.parameters() if p.requires_grad]
    return {"student": student_params, "lora": lora_params, "action_expert": ae_params}


def swap_student_for_eval(backbone, student):
    """Eval path: student replaces transformer.forward (validated in Module 1).

    The eval forward path still calls transformer.wte (token embedding, 2560-dim) inside
    build_input_embeddings, and may touch transformer.config / prepare_rope_cache. So carry
    those over from the original teacher transformer onto the student, which fully
    masquerades as the transformer while its own forward produces per-layer KV + last hidden.
    """
    dev = next(backbone.action_expert.parameters()).device
    dt = _module_dtype(backbone.action_expert)
    orig = backbone.transformer
    student = student.to(device=dev, dtype=dt).eval()
    # carry over the pieces the surrounding model still references
    student.wte = orig.wte
    student.emb_drop = getattr(orig, "emb_drop", None)
    student.config = orig.config
    student.ln_f = getattr(orig, "ln_f", None)
    student.rotary_emb = getattr(orig, "rotary_emb", None)
    student.get_input_embeddings = lambda: student.wte
    student.set_input_embeddings = lambda v: setattr(student, "wte", v)
    student.prepare_rope_cache = lambda *a, **k: None  # student builds RoPE on the fly
    backbone.transformer = student
    return backbone


@torch.no_grad()
def merge_lora_into(proj: nn.Linear, lora_A, lora_B, scaling):
    """Fold a trained LoRA update into a frozen nn.Linear (for eval): W += (B @ A) * scaling."""
    delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
    proj.weight.data.add_(delta.to(proj.weight.dtype).to(proj.weight.device))
    return proj


# --------------------------------------------------------- generic LoRA on a subtree
def wrap_linears_with_lora(root: nn.Module, *, rank=16, alpha=16, dropout=0.0):
    """Replace every nn.Linear leaf inside ``root`` with a LoRALinear (in place). Returns the
    list of new trainable LoRA params. Used for Phase-2 LoRA fine-tuning of the student LLM /
    action expert / ViT (uniform, so save+eval are symmetric)."""
    names = [n for n, m in root.named_modules()
             if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear)]
    params = []
    for name in names:
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        leaf = name.rsplit(".", 1)[-1]
        base = getattr(parent, leaf)
        dev = base.weight.device
        wrapped = LoRALinear(base, rank, alpha, dropout).to(dev)
        setattr(parent, leaf, wrapped)
        wrapped.lora_A.requires_grad = True
        wrapped.lora_B.requires_grad = True
        params += [wrapped.lora_A, wrapped.lora_B]
    return params


@torch.no_grad()
def merge_lora_linears_(root: nn.Module):
    """Fold every LoRALinear inside ``root`` back into a plain nn.Linear (in place), so the
    module's state_dict matches the original unwrapped architecture (eval-ready)."""
    names = [n for n, m in root.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        leaf = name.rsplit(".", 1)[-1]
        ll = getattr(parent, leaf)
        base = ll.base
        delta = (ll.lora_B.to(torch.float32) @ ll.lora_A.to(torch.float32)) * ll.scaling
        base.weight.data.add_(delta.to(base.weight.dtype).to(base.weight.device))
        setattr(parent, leaf, base)


def attach_student_lora_finetune(policy, student, *, targets=("llm", "ae"),
                                 lora_rank=16, lora_alpha=16, lora_dropout=0.0,
                                 student_dtype=None):
    """Phase-2 assembly: freeze the (Phase-1) student LLM + action expert + ViT bases and
    LoRA-adapt the requested subtrees, then train on the base flow objective (anchor OFF).
    ``targets`` subset of {llm, ae, vit}. Returns {"lora": [...]}.
    """
    backbone = policy._backbone()
    ae = backbone._require_action_expert()
    dev = next(ae.parameters()).device
    dt = _module_dtype(ae)
    sdt = student_dtype or dt

    for p in policy.parameters():
        p.requires_grad = False
    student = student.to(device=dev, dtype=sdt)
    for p in student.parameters():
        p.requires_grad = False
    policy.student_llm = student

    lora_params = []
    if "llm" in targets:
        lora_params += wrap_linears_with_lora(student, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    if "ae" in targets:
        lora_params += wrap_linears_with_lora(ae, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    if "vit" in targets:
        vb = getattr(backbone, "vision_backbone", None)
        if vb is not None:
            lora_params += wrap_linears_with_lora(vb, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    # anchor OFF for the data fine-tune; free the teacher transformer blocks (unused) but keep wte
    policy._anchor_cfg = {"enabled": False, "weight": 0.0, "mode": "both", "beta": 1.0}
    policy._loss_components = {}
    try:
        for blk in backbone.transformer.blocks:
            blk.to("meta")
    except Exception:
        pass
    policy._compute_flow_matching_loss_joint_per_layer = types.MethodType(student_joint_loss, policy)
    return {"lora": lora_params}


_CFG_KEYS_EVAL = ("hidden", "num_heads", "intermediate", "num_layers", "num_kv_heads",
                  "head_dim", "rope_theta", "teacher_hidden", "rms_eps", "use_qk_norm")


@torch.no_grad()
def load_student_and_merge_lora(policy, student, ckpt):
    """Legacy eval assembly (context-proj LoRA deltas): swap the (loaded) student into
    transformer + merge the action-expert context-proj LoRA. ``ckpt`` is the trainer's dict."""
    backbone = policy._backbone()
    ae = backbone._require_action_expert()
    swap_student_for_eval(backbone, student)
    scaling = ckpt.get("lora_scaling", 1.0)
    lo = ckpt["lora"]
    merge_lora_into(ae.context_k_proj, lo["ck_A"], lo["ck_B"], scaling)
    merge_lora_into(ae.context_v_proj, lo["cv_A"], lo["cv_B"], scaling)
    return policy


@torch.no_grad()
def assemble_student_for_eval(policy, ckpt):
    """Unified eval assembler across all checkpoint formats:
      * phase='lora'    -> Phase-2: wrap {llm,ae,vit} bases with LoRA, load wrapped states, merge.
      * has action_expert (no phase/lora) -> Phase-1 co-adapt: load plain student + AE states.
      * else            -> legacy: plain student + context-proj LoRA deltas.
    Returns (policy, student_module).
    """
    import student as S
    scfg = {k: v for k, v in ckpt["cfg"].items() if k in _CFG_KEYS_EVAL}
    stu, _ = S.build_student(scfg)
    backbone = policy._backbone()
    ae = backbone._require_action_expert()
    dt = _module_dtype(ae)
    phase = ckpt.get("phase", "legacy")

    if phase == "lora":
        targets = tuple(ckpt.get("lora_targets", ["llm", "ae"]))
        r = int(ckpt.get("lora_rank", 16)); a = int(ckpt.get("lora_alpha", 16))
        if "llm" in targets:
            wrap_linears_with_lora(stu, rank=r, alpha=a)
        stu.load_state_dict(ckpt["student"], strict=True)
        merge_lora_linears_(stu)
        stu = stu.to(dtype=dt)
        swap_student_for_eval(backbone, stu)
        if "ae" in targets:
            wrap_linears_with_lora(ae, rank=r, alpha=a)
            ae.load_state_dict(ckpt["action_expert"], strict=True)
            merge_lora_linears_(ae)
        elif ckpt.get("action_expert") is not None:
            ae.load_state_dict(ckpt["action_expert"], strict=True)
        vb = getattr(backbone, "vision_backbone", None)
        if vb is not None and ckpt.get("vision_backbone") is not None:
            if "vit" in targets:
                wrap_linears_with_lora(vb, rank=r, alpha=a)
                vb.load_state_dict(ckpt["vision_backbone"], strict=True)
                merge_lora_linears_(vb)
            else:
                vb.load_state_dict(ckpt["vision_backbone"], strict=True)
    elif ckpt.get("action_expert") is not None:
        stu.load_state_dict(ckpt["student"], strict=True)
        stu = stu.to(dtype=dt)
        swap_student_for_eval(backbone, stu)
        ae.load_state_dict(ckpt["action_expert"], strict=True)
        vb = getattr(backbone, "vision_backbone", None)
        if vb is not None and ckpt.get("vision_backbone") is not None:
            vb.load_state_dict(ckpt["vision_backbone"], strict=True)
    else:
        stu.load_state_dict(ckpt["student"], strict=True)
        stu = stu.to(dtype=dt)
        load_student_and_merge_lora(policy, stu, ckpt)

    policy.eval()
    return policy, stu


# --------------------------------------------------------------------------- loss
def _imports():
    from lerobot.policies.molmoact2 import modeling_molmoact2 as M
    return M


def _kv_anchor_loss(student_kv, teacher_kv, token_mask, *, mode="both", beta=1.0, eps=1e-6):
    """All-layer teacher-KV anchoring on raw pre-``_cache_to_sequence`` KV.

    Each layer's key and value tensors are ``[B, n_kv_heads, N, head_dim]``. We flatten heads
    into a per-token feature vector ``[B, N, H*D]`` and, over valid tokens only:
      * directional: mean over layers/{k,v} of ``(1 - cos(student, teacher))`` -> kills the
        orthogonal-KV shortcut (cos ~0) seen in the functional-only student,
      * magnitude: normalized MSE ``||s-t||^2 / (||t||^2 + eps)`` -> fixes drifting-norm mismatch.
    ``mode`` in {cos, mse, both}; for 'both' total = cos + beta*magnitude.
    """
    L = len(student_kv)
    ref = student_kv[0][0]
    B, _, N, _ = ref.shape
    if token_mask is None:
        tok = ref.new_ones(B, N, dtype=torch.float32)
    else:
        tok = token_mask.to(device=ref.device, dtype=torch.float32)
        if tok.dim() > 2:
            tok = tok.reshape(B, -1)
        if tok.dim() != 2 or tok.shape[0] != B or tok.shape[-1] != N:
            tok = ref.new_ones(B, N, dtype=torch.float32)
    denom = tok.sum().clamp_min(1.0)

    def _vec(x):  # [B, H, N, D] -> [B, N, H*D] fp32
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d).float()

    cos_acc = ref.new_zeros((), dtype=torch.float32)
    mag_acc = ref.new_zeros((), dtype=torch.float32)
    for li in range(L):
        for s, t in ((student_kv[li][0], teacher_kv[li][0]),
                     (student_kv[li][1], teacher_kv[li][1])):
            sv, tv = _vec(s), _vec(t)
            if mode in ("cos", "both"):
                cos = F.cosine_similarity(sv, tv, dim=-1)  # [B, N]
                cos_acc = cos_acc + ((1.0 - cos) * tok).sum() / denom
            if mode in ("mse", "both"):
                num = (sv - tv).pow(2).sum(-1)
                den = tv.pow(2).sum(-1) + eps
                mag_acc = mag_acc + ((num / den) * tok).sum() / denom
    scale = float(2 * L)
    cos_acc = cos_acc / scale
    mag_acc = mag_acc / scale
    if mode == "cos":
        return cos_acc
    if mode == "mse":
        return mag_acc
    return cos_acc + beta * mag_acc


def student_joint_loss(self, *, batch, model_inputs, timesteps=None, noise=None, reduction="mean"):
    """Faithful re-implementation of _compute_flow_matching_loss_joint_per_layer with the LLM
    half replaced by the student. Reuses the policy's flow-matching helpers and the action
    expert's projection/blocks/final layer verbatim. No FastV/ROI branches (disabled here)."""
    M = _imports()
    ACTION = M.ACTION
    _expand_mask = M._expand_mask
    _apply_chunk = M._apply_action_chunk_padding_mask
    _apply_dim = M._apply_action_dim_padding_mask

    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction={reduction!r}")
    backbone = self._backbone()
    action_expert = backbone._require_action_expert()
    student = self.student_llm

    actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
        actions=batch[ACTION],
        action_dim_is_pad=batch.get("action_dim_is_pad"),
        timesteps=timesteps,
        noise=noise,
    )
    num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))
    batch_size = int(actions.shape[0])
    device = actions.device
    xt_flat = xt.reshape(batch_size * num_flow_timesteps, actions.shape[1], actions.shape[2])
    timesteps_flat = timesteps.reshape(batch_size * num_flow_timesteps)

    hidden_states, causal_mask_mapping, position_ids, cache_position = (
        self._prepare_joint_training_backbone_inputs(model_inputs)
    )
    if hidden_states.shape[0] != batch_size:
        raise ValueError(
            f"Backbone batch {hidden_states.shape[0]} != action batch {batch_size}")

    encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
        input_ids=model_inputs.get("input_ids"),
        attention_mask=model_inputs.get("attention_mask"),
    )
    action_attention_mask = None
    if batch.get("action_horizon_is_pad") is not None:
        action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)

    valid_action = None
    if action_attention_mask is not None:
        valid_action = action_attention_mask.to(device=device, dtype=actions.dtype).unsqueeze(-1)
        valid_action = _expand_mask(valid_action, num_flow_timesteps)

    rope_cache = None
    if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
        rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
            seq_len=actions.shape[1], device=device, dtype=actions.dtype)

    cross_mask = action_expert._build_cross_attention_mask(
        encoder_attention_mask, batch_size, actions.dtype)
    cross_mask = _expand_mask(cross_mask, num_flow_timesteps)
    self_mask = action_expert._build_self_attention_mask(
        action_attention_mask, actions.shape[1], device, actions.dtype)
    self_mask = _expand_mask(self_mask, num_flow_timesteps)

    conditioning = self._action_time_conditioning(action_expert, timesteps_flat)
    action_hidden = action_expert.action_embed(xt_flat)
    if valid_action is not None:
        action_hidden = action_hidden * valid_action

    # ---- STUDENT: all per-layer KV + last hidden in one forward ----
    sout = student(
        inputs_embeds=hidden_states,
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        cache_position=cache_position,
        collect_layer_kv_states=True,
    )
    student_kv = list(sout.past_key_values)
    if len(student_kv) != len(action_expert.blocks):
        raise RuntimeError(
            f"student produced {len(student_kv)} KV layers, action expert has {len(action_expert.blocks)}")

    # ---- TEACHER KV (frozen, resident) for anchoring; training-only ----
    anchor_cfg = getattr(self, "_anchor_cfg", None) or {}
    do_anchor = bool(anchor_cfg.get("enabled")) and self.training and reduction == "mean"
    teacher_kv = None
    if do_anchor:
        with torch.no_grad():
            tout = backbone.transformer(
                inputs_embeds=hidden_states,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                cache_position=cache_position,
                collect_layer_kv_states=True,
                use_cache=False,
            )
        teacher_kv = list(tout.past_key_values)
        if len(teacher_kv) != len(student_kv):
            raise RuntimeError(
                f"teacher produced {len(teacher_kv)} KV layers, student has {len(student_kv)}")

    # ---- action-expert per-layer cross-attention on student KV ----
    for layer_idx, action_block in enumerate(action_expert.blocks):
        k_raw, v_raw = student_kv[layer_idx]
        key_states = backbone._cache_to_sequence(k_raw)
        value_states = backbone._cache_to_sequence(v_raw)
        if self.config.enable_knowledge_insulation:
            key_states = key_states.detach()
            value_states = value_states.detach()
        k_ctx = action_expert._project_kv_tensor(key_states, action_expert.context_k_proj)
        v_ctx = action_expert._project_kv_tensor(value_states, action_expert.context_v_proj)
        k_norm = action_block.cross_attn.k_norm
        if k_norm is not None:
            k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
        if num_flow_timesteps != 1:
            k_ctx = _expand_mask(k_ctx, num_flow_timesteps)
            v_ctx = _expand_mask(v_ctx, num_flow_timesteps)
        action_hidden = action_block(
            action_hidden, conditioning,
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
    pred_velocity = pred_velocity.reshape(
        batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2])

    loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
    loss = _apply_chunk(loss, batch.get("action_horizon_is_pad"))
    if self.config.mask_action_dim_padding:
        loss = _apply_dim(loss, batch.get("action_dim_is_pad"))
    loss = loss.reshape(batch_size, -1).mean(dim=1)
    if reduction == "mean":
        loss = loss.mean()

    # ---- hybrid: add teacher-KV anchoring on top of the flow-matching loss ----
    if do_anchor and teacher_kv is not None:
        flow_loss = loss
        anchor = _kv_anchor_loss(
            student_kv, teacher_kv, encoder_attention_mask,
            mode=str(anchor_cfg.get("mode", "both")),
            beta=float(anchor_cfg.get("beta", 1.0)),
        )
        w = float(anchor_cfg.get("weight", 0.0))
        loss = flow_loss + w * anchor
        self._loss_components = {
            "flow": float(flow_loss.detach()),
            "anchor": float(anchor.detach()),
            "total": float(loss.detach()),
        }
    elif reduction == "mean":
        self._loss_components = {"flow": float(loss.detach()),
                                 "anchor": 0.0, "total": float(loss.detach())}
    # second return kept for API parity (discrete path); student last hidden if needed
    return loss, sout.last_hidden_state
