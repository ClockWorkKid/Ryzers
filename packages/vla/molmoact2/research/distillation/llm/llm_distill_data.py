"""Real LIBERO batch source for Run D Stage-1 LLM distillation.

Path A (robust): reuse lerobot-train's own dataset + preprocessor so batches are
byte-identical to what MolmoAct2Policy consumes during finetuning, then feed the HF
backbone. Yields model-ready dicts (input_ids, attention_mask, token_type_ids,
pixel_values, image_token_pooling, image_grids, image_num_crops, ...).

Validated on the SIF (mi325x) via validate_capture.py.
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
    # preprocessor pipeline (rename -> batch-dim -> normalize -> pack -> device)
    try:
        from lerobot.policies.factory import make_pre_post_processors
        pre, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    except Exception:  # older/newer factory name
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
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, 0)
            else:
                out[k] = vals
        return out

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=_collate, drop_last=True, persistent_workers=False,
    )
    while True:  # re-create the iterator each pass (avoids itertools.cycle FD leak)
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


def capture_teacher(model, batch):
    """Run the teacher backbone on a model-ready batch; return distillation targets.
    model: MolmoAct2ForConditionalGeneration ; batch: dict from iter_libero_batches."""
    m = model.model  # MolmoAct2Model
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
    out = m.transformer(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_bias,
        position_ids=positions,
        output_hidden_states=True,
        collect_layer_kv_states=True,
        use_cache=False,
    )
    kv = out.past_key_values  # tuple len 36 of (k,v) each [B,8,N,128]
    am = batch.get("attention_mask")
    mask = torch.ones(B, N, device=inputs_embeds.device) if am is None else am.float()
    return {
        "inputs_embeds": inputs_embeds,
        "mask": mask,
        "positions": positions,
        "attn_bias": attn_bias,
        "teacher": {"hidden": list(out.hidden_states),
                    "kv": list(kv),
                    "last": out.last_hidden_state},
    }
