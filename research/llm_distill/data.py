"""Real LIBERO batch source + teacher capture for MolmoAct2 LLM distillation.

Reuses lerobot-train's own dataset + preprocessor so batches are byte-identical to what
MolmoAct2Policy consumes during finetuning, then feeds the HF backbone. This is the same
robust path validated in the ViT distillation work.

``capture_teacher`` returns the fused ``inputs_embeds``, the native 4D attention bias, the
position ids, and (for reference/validation) the teacher's per-layer KV -- exactly the
inputs the student must consume and the KV contract it must match.
"""

from __future__ import annotations

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass


_MODEL_KEYS = (
    "input_ids", "attention_mask", "token_type_ids",
    "pixel_values", "image_token_pooling", "image_grids", "image_num_crops",
)


def build_dataset_and_preprocessor(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    repo_id = args.repo_id
    revision = getattr(args, "revision", "main") or "main"
    cfg = MolmoAct2Config(
        checkpoint_path=args.teacher,
        chunk_size=10,
        n_action_steps=10,
        action_mode="both",
        model_dtype="bfloat16",
        device=args.device,
    )
    ds_meta = LeRobotDatasetMetadata(repo_id, revision=revision)
    try:
        from lerobot.policies.factory import make_pre_post_processors
        pre, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    except Exception:
        from lerobot.policies.molmoact2.processor_molmoact2 import (
            make_molmoact2_pre_post_processors as _mk,
        )
        pre, _ = _mk(config=cfg, dataset_stats=ds_meta.stats)
    ds = LeRobotDataset(repo_id, revision=revision,
                        delta_timestamps=getattr(cfg, "delta_timestamps", None))
    return ds, ds_meta, pre, cfg


def iter_libero_batches(args, device):
    """Infinite iterator of model-ready batches on ``device``."""
    ds, ds_meta, pre, cfg = build_dataset_and_preprocessor(args)
    cam_keys = list(getattr(ds_meta, "camera_keys", []))

    def _collate(samples):
        out = {}
        for k in samples[0]:
            vals = [s[k] for s in samples]
            out[k] = torch.stack(vals, 0) if isinstance(vals[0], torch.Tensor) else vals
        return out

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=_collate, drop_last=True, persistent_workers=False,
    )
    while True:
        for raw in dl:
            for ck in cam_keys:
                if ck in raw and torch.is_tensor(raw[ck]) and raw[ck].dtype == torch.uint8:
                    raw[ck] = raw[ck].float() / 255.0
            batch = pre(raw)
            model_batch = {}
            for k in _MODEL_KEYS:
                if k in batch and torch.is_tensor(batch[k]):
                    model_batch[k] = batch[k].to(device)
            yield model_batch


def iter_full_batches(args, device):
    """Infinite iterator of FULL preprocessed batches (all tensor keys) on ``device`` --
    what MolmoAct2Policy.forward(batch) consumes (includes ACTION + pad masks)."""
    ds, ds_meta, pre, cfg = build_dataset_and_preprocessor(args)
    cam_keys = list(getattr(ds_meta, "camera_keys", []))

    def _collate(samples):
        out = {}
        for k in samples[0]:
            vals = [s[k] for s in samples]
            out[k] = torch.stack(vals, 0) if isinstance(vals[0], torch.Tensor) else vals
        return out

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=_collate, drop_last=True, persistent_workers=False,
    )
    while True:
        for raw in dl:
            for ck in cam_keys:
                if ck in raw and torch.is_tensor(raw[ck]) and raw[ck].dtype == torch.uint8:
                    raw[ck] = raw[ck].float() / 255.0
            batch = pre(raw)
            out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            yield out


@torch.no_grad()
def capture_teacher(model, batch, collect_kv=True):
    """Run the teacher backbone; return the student's inputs + (optionally) reference KV.

    Returns dict with:
      inputs_embeds [B,N,2560], attn_bias (4D), positions [B,N], mask [B,N],
      teacher = {kv: list of (k,v) [B,8,N,128], last: [B,N,2560]}  (if collect_kv)
    """
    m = model.model
    images, token_pooling = m.merge_visual_inputs(
        input_ids=batch["input_ids"],
        pixel_values=batch.get("pixel_values"),
        image_token_pooling=batch.get("image_token_pooling"),
        image_grids=batch.get("image_grids"),
        image_num_crops=batch.get("image_num_crops"),
    )
    inputs_embeds, _ = m.build_input_embeddings(batch["input_ids"], images, token_pooling)
    attn_bias = m._build_native_attention_bias(
        inputs_embeds=inputs_embeds,
        attention_mask=batch.get("attention_mask"),
        token_type_ids=batch.get("token_type_ids"),
        past_key_values=None,
    )
    B, N = inputs_embeds.shape[:2]
    positions = torch.arange(N, device=inputs_embeds.device).unsqueeze(0).expand(B, N)
    am = batch.get("attention_mask")
    mask = torch.ones(B, N, device=inputs_embeds.device) if am is None else am.float()

    out = {
        "inputs_embeds": inputs_embeds,
        "attn_bias": attn_bias,
        "positions": positions,
        "mask": mask,
    }
    if collect_kv:
        tout = m.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            position_ids=positions,
            collect_layer_kv_states=True,
            use_cache=False,
        )
        out["teacher"] = {
            "kv": list(tout.past_key_values),           # 36 x (k,v) [B,8,N,128]
            "last": tout.last_hidden_state,             # [B,N,2560]
        }
    return out


@torch.no_grad()
def capture_teacher_reduced(policy, batch, *, fastv_keep=0.25, collect_teacher_kv=True):
    """Top-down token-reduction front-end (Workstream B): produce the 25%-image-token
    reduced fused sequence via the trained ROI gate + FastV group-drop, PLUS the full-token
    teacher reference.

    Requires ``policy`` to have the ROI gate installed AND its trained weights loaded
    (``roi_prune_enable=True``, ``roi_prune_select`` in {'gate','gate_distill','gate_predict'},
    ``roi_fastv_*`` configured). Reproduces the *inference* front-end (gate-scored group drop,
    ``ROI_FASTV_INFER=1`` path) but applies the cut at the INPUT (prune-before-student), so the
    entire student LLM + action expert run on the reduced sequence -- this is the plan's
    recommended end-to-end reduced-sequence integration.

    Returns dict:
      reduced = {inputs_embeds [B,s_new,D], attention_mask [B,s_new], position_ids [B,s_new]
                 (ORIGINAL absolute positions -> RoPE geometry preserved), col_idx [B,s_new]}
      full    = {inputs_embeds [B,S,D], S}
      teacher = {kv_full: 36 x (k,v)[B,8,S,128], kv_kept: gathered at col_idx [B,8,s_new,128],
                 last_full [B,S,D]}   (when collect_teacher_kv)

    The ``reduced`` dict is a drop-in ``model_inputs`` for
    ``policy._compute_flow_matching_loss_joint_per_layer`` (the inputs_embeds path): the student
    runs on the reduced-length context and the action expert cross-attends to its reduced KV.
    """
    backbone = policy._backbone()
    vb = getattr(backbone, "vision_backbone", None)
    mi = policy._model_inputs(batch)

    # 1) gate instruction conditioning so the eval keep-set matches training
    setter = getattr(policy, "_set_gate_task_tokens", None)
    if callable(setter):
        setter(mi)
    # 2) causal-predictive carry (gate_predict): reuse the prior replan's keep-set if present;
    #    None on the first replan -> falls back to current-frame gate top-K inside encode_image
    sel = getattr(policy.config, "roi_prune_select", None)
    if sel == "gate_predict" and vb is not None:
        vb.roi_external_keep_idx = getattr(policy, "_roi_pred_prev_keep_idx", None)

    # 3) FULL fused hidden (the ViT gate prune runs inside encode_image and stashes
    #    vb.roi_last_gate_scores / roi_last_num_patches used by the FastV importance step)
    hidden, causal_mask_mapping, position_ids, cache_position = (
        policy._prepare_joint_training_backbone_inputs(mi)
    )
    B, S, D = hidden.shape

    # 4) FastV keep-set from the GATE scores (inference importance source, no teacher pass)
    if not policy._fastv_group_ctx_from_gate(mi):
        raise RuntimeError(
            "capture_teacher_reduced: gate group-ctx unavailable -- is the ROI gate installed "
            "and are gate scores populated (roi_prune_select gate*/roi_prune_enable)?")
    gd = policy._teacher_group_drop(float(fastv_keep))
    if gd is None:
        raise RuntimeError("capture_teacher_reduced: _teacher_group_drop returned None (no group ctx).")
    _new_ppi, keep_over = gd
    col_idx = policy._fastv_col_idx_from_keep(mi, keep_over)
    if col_idx is None:
        raise RuntimeError(
            "capture_teacher_reduced: ragged/non-uniform kept image-token counts -- cannot build a "
            "rectangular reduced batch.")
    s_new = int(col_idx.shape[1])

    # 5) gather the reduced sequence (prune-before-student)
    reduced_embeds = hidden.gather(1, col_idx.unsqueeze(-1).expand(B, s_new, D))
    pos2d = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0)
    if pos2d.shape[0] == 1 and B > 1:
        pos2d = pos2d.expand(B, -1)
    reduced_positions = pos2d.gather(1, col_idx)
    am = mi.get("attention_mask")
    if am is not None and torch.is_tensor(am) and am.dim() == 2:
        reduced_mask = am.gather(1, col_idx).to(hidden.dtype)
    else:
        reduced_mask = torch.ones(B, s_new, device=hidden.device, dtype=hidden.dtype)

    out = {
        "reduced": {
            "inputs_embeds": reduced_embeds,
            "attention_mask": reduced_mask,
            "position_ids": reduced_positions,
            "col_idx": col_idx,
        },
        "full": {"inputs_embeds": hidden, "S": S},
    }

    if collect_teacher_kv:
        tout = backbone.transformer(
            inputs_embeds=hidden,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            collect_layer_kv_states=True,
            use_cache=False,
        )
        kv_full = list(tout.past_key_values)            # 36 x (k,v) [B,8,S,128]
        kv_kept = []
        for k, v in kv_full:
            H, Dh = k.shape[1], k.shape[3]
            gi = col_idx[:, None, :, None].expand(B, H, s_new, Dh)
            kv_kept.append((k.gather(2, gi), v.gather(2, gi)))
        out["teacher"] = {"kv_full": kv_full, "kv_kept": kv_kept,
                          "last_full": tout.last_hidden_state}
    return out
