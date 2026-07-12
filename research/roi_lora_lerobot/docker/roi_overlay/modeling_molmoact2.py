# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MolmoAct2 policy for LeRobot.

MolmoAct2 is a VLM-based robotics policy from Allen AI that combines a
Molmo vision-language backbone with a per-layer flow-matching action expert
for continuous action generation, plus an optional discrete action token
head. This module wraps the vendored HF model implementation
(``molmoact2_hf_model/``) into the LeRobot ``PreTrainedPolicy`` interface.

Paper:  https://allenai.org/blog/molmoact2
Code:   https://github.com/allenai/molmoact2
"""

from __future__ import annotations

import json
import logging
import os
import types
from collections import deque
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from torch.distributions import Beta

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _scipy_available, _transformers_available, require_package

from ..rtc.modeling_rtc import RTCProcessor
from .configuration_molmoact2 import MolmoAct2Config
from .roi_prune import PreViTPruneConfig, ROIGate, ROIGateXAttn

logger = logging.getLogger(__name__)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def _resolve_checkpoint_location(
    checkpoint_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> str:
    """Resolve a checkpoint path to a local directory, downloading from Hub if needed."""
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("MolmoAct2 policy requires `checkpoint_path`.")
    from pathlib import Path

    local_path = Path(checkpoint_path).expanduser()
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=checkpoint_path,
        repo_type="model",
        revision=revision,
        force_download=force_download,
        ignore_patterns=["*.py", "*.pyc", "__pycache__/*"],
        token=_hf_token(),
    )


def _torch_dtype(dtype: str) -> torch.dtype:
    """Convert a dtype name string to a torch.dtype."""
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


if TYPE_CHECKING or _transformers_available:
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

    from .molmoact2_hf_model.configuration_molmoact2 import MolmoAct2Config as HFMolmoAct2Config
    from .molmoact2_hf_model.modeling_molmoact2 import MolmoAct2ForConditionalGeneration
else:
    SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
    SAFE_WEIGHTS_NAME = "model.safetensors"
    HFMolmoAct2Config = None
    MolmoAct2ForConditionalGeneration = None

if TYPE_CHECKING or (_transformers_available and _scipy_available):
    from .molmoact2_hf_model.action_tokenizer import UniversalActionProcessor
else:
    UniversalActionProcessor = None

_MODEL_INPUT_KEYS = {
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "token_type_ids",
    "inputs_embeds",
}


def _load_hf_norm_metadata_for_tag(
    checkpoint_path: str,
    *,
    revision: str | None,
    force_download: bool,
    norm_tag: str | None,
) -> dict[str, Any]:
    """Read per-tag metadata from the checkpoint's ``norm_stats.json``."""
    norm_tag = str(norm_tag or "").strip()
    if not norm_tag:
        return {}
    from contextlib import suppress
    from pathlib import Path

    checkpoint_location = Path(
        _resolve_checkpoint_location(
            checkpoint_path,
            revision=revision,
            force_download=force_download,
        )
    )
    norm_stats_filename = "norm_stats.json"
    config_path = checkpoint_location / "config.json"
    if config_path.exists():
        with suppress(OSError, json.JSONDecodeError):
            norm_stats_filename = str(
                json.loads(config_path.read_text()).get("norm_stats_filename") or norm_stats_filename
            )
    stats_path = checkpoint_location / norm_stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(
            f"MolmoAct2 HF checkpoint is missing {norm_stats_filename!r}; cannot resolve norm_tag={norm_tag!r}."
        )
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise ValueError(f"MolmoAct2 norm stats file {stats_path} has no metadata_by_tag mapping.")
    metadata = metadata_by_tag.get(norm_tag)
    if not isinstance(metadata, dict):
        available = sorted(str(tag) for tag in metadata_by_tag)
        raise ValueError(f"Unknown MolmoAct2 norm_tag={norm_tag!r}. Available tags: {available}.")
    return metadata


def _apply_norm_tag_metadata(config: MolmoAct2Config) -> None:
    """Populate config fields from the checkpoint's norm-tag metadata."""
    if not str(config.norm_tag or "").strip():
        return
    metadata = _load_hf_norm_metadata_for_tag(
        config.checkpoint_path,
        revision=config.checkpoint_revision,
        force_download=bool(config.checkpoint_force_download),
        norm_tag=config.norm_tag,
    )
    if metadata.get("action_horizon") is not None:
        config.chunk_size = int(metadata["action_horizon"])
    if metadata.get("n_action_steps") is not None:
        config.n_action_steps = int(metadata["n_action_steps"])
    if not config.setup_type and metadata.get("setup_type") is not None:
        config.setup_type = str(metadata["setup_type"])
    if not config.control_mode and metadata.get("control_mode") is not None:
        config.control_mode = str(metadata["control_mode"])


def _saved_policy_action_mode(config: MolmoAct2Config) -> str | None:
    """Read the action mode from a LeRobot-saved checkpoint's ``config.json``."""
    from pathlib import Path

    pretrained_path = getattr(config, "pretrained_path", None)
    if pretrained_path is None:
        return None
    config_path = Path(pretrained_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        mode = json.loads(config_path.read_text()).get("action_mode")
    except (OSError, json.JSONDecodeError):
        return None
    if mode in {"continuous", "discrete", "both"}:
        return str(mode)
    return None


def _training_action_mode(config: MolmoAct2Config, saved_policy_action_mode: str | None = None) -> str:
    return saved_policy_action_mode or config.action_mode


def _validate_inference_action_mode(
    config: MolmoAct2Config, saved_policy_action_mode: str | None = None
) -> None:
    """Check that the requested inference mode is compatible with the training mode."""
    requested_mode = config.inference_action_mode
    if requested_mode is None:
        return
    training_mode = _training_action_mode(config, saved_policy_action_mode)
    if requested_mode == "continuous" and training_mode == "discrete":
        raise ValueError(
            "MolmoAct2 checkpoint was trained with action_mode='discrete' and cannot run "
            "continuous inference."
        )
    if requested_mode == "discrete" and training_mode == "continuous":
        raise ValueError(
            "MolmoAct2 checkpoint was trained with action_mode='continuous' and cannot run "
            "discrete inference. Train with action_mode='both' or action_mode='discrete' first."
        )


def _validate_checkpoint_action_mode(
    config: MolmoAct2Config,
    checkpoint_action_mode: str,
    *,
    has_action_expert: bool,
) -> None:
    """Check that the checkpoint's action mode is compatible with the config."""
    if config.action_mode == "both" and checkpoint_action_mode != "both":
        raise ValueError(
            f"action_mode='both' requires checkpoint action_mode='both', got {checkpoint_action_mode!r}."
        )
    if config.action_mode == "discrete" and checkpoint_action_mode not in {"discrete", "both"}:
        raise ValueError(
            f"action_mode='discrete' requires checkpoint action_mode in {{'discrete', 'both'}}, "
            f"got {checkpoint_action_mode!r}."
        )
    if config.action_mode in {"continuous", "both"} and not has_action_expert:
        raise ValueError("Continuous MolmoAct2 training requires an action expert checkpoint.")


def _resolve_inference_action_mode(
    config: MolmoAct2Config,
    requested_mode: str | None,
    saved_policy_action_mode: str | None = None,
) -> str:
    """Resolve the final inference action mode, validating compatibility."""
    training_mode = _training_action_mode(config, saved_policy_action_mode)
    if requested_mode is None:
        requested_mode = config.inference_action_mode
    if requested_mode is None:
        raise ValueError(
            "MolmoAct2 inference requires `inference_action_mode` to be set explicitly "
            "to either 'continuous' or 'discrete'."
        )
    if requested_mode not in {"continuous", "discrete"}:
        raise ValueError("MolmoAct2 inference_action_mode must be either 'continuous' or 'discrete'.")
    if requested_mode == "continuous" and training_mode == "discrete":
        raise ValueError("MolmoAct2 action_mode='discrete' checkpoint cannot run continuous inference.")
    if requested_mode == "discrete" and training_mode == "continuous":
        raise ValueError("MolmoAct2 action_mode='continuous' checkpoint cannot run discrete inference.")
    return requested_mode


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    index_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_INDEX_NAME)
    single_file_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        loaded_keys = set(weight_map)
        model_keys = set(model.state_dict())
        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)
        if missing_keys or unexpected_keys:
            message = ["MolmoAct2 safetensors do not match the local model implementation."]
            if missing_keys:
                message.append(f"Missing keys: {missing_keys[:8]}")
            if unexpected_keys:
                message.append(f"Unexpected keys: {unexpected_keys[:8]}")
            raise RuntimeError(" ".join(message))
        for shard_file in sorted(set(weight_map.values())):
            state_dict = load_safetensors_file(os.path.join(checkpoint_location, shard_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        return
    if os.path.isfile(single_file_path):
        state_dict = load_safetensors_file(single_file_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        return
    raise FileNotFoundError(
        f"MolmoAct2 checkpoint at {checkpoint_location} must contain {SAFE_WEIGHTS_NAME} "
        f"or {SAFE_WEIGHTS_INDEX_NAME}."
    )


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> Tensor:
    if cutoff < time_offset:
        raise ValueError(f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}")
    if time_scale <= 0:
        raise ValueError(f"flow-matching time_scale must be > 0, got {time_scale}")
    upper = min(cutoff, time_offset + time_scale)
    dist = Beta(torch.tensor(alpha, device=device), torch.tensor(beta, device=device))
    samples = dist.sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _mask_discrete_action_spans(
    *,
    input_ids: Tensor,
    mask: Tensor,
    start_token_id: int | None,
    end_token_id: int | None,
) -> Tensor:
    if start_token_id is None or end_token_id is None:
        return mask
    mask = mask.clone()
    for batch_idx in range(input_ids.shape[0]):
        row = input_ids[batch_idx]
        starts = (row == int(start_token_id)).nonzero(as_tuple=False).flatten().tolist()
        ends = (row == int(end_token_id)).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start in starts:
            while end_ptr < len(ends) and ends[end_ptr] < start:
                end_ptr += 1
            if end_ptr >= len(ends):
                mask[batch_idx, start:] = False
                break
            end = int(ends[end_ptr])
            mask[batch_idx, start : end + 1] = False
            end_ptr += 1
    return mask


def _drop_trivial_attention_mask(model_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    attention_mask = model_inputs.get("attention_mask")
    if torch.is_tensor(attention_mask) and bool(attention_mask.to(dtype=torch.bool).all().item()):
        model_inputs = dict(model_inputs)
        model_inputs.pop("attention_mask", None)
    return model_inputs


def _expand_mask(mask: Tensor | None, num_flow_timesteps: int) -> Tensor | None:
    if mask is None:
        return None
    return (
        mask.unsqueeze(1)
        .expand(-1, num_flow_timesteps, *([-1] * (mask.ndim - 1)))
        .reshape(mask.shape[0] * num_flow_timesteps, *mask.shape[1:])
    )


def _action_dim_valid_mask(target: Tensor, action_dim_is_pad: Tensor | None) -> Tensor | None:
    if action_dim_is_pad is None:
        return None
    mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"action_dim_is_pad width {mask.shape[-1]} does not match target width {target.shape[-1]}."
        )
    if mask.shape[0] == 1 and target.shape[0] != 1:
        mask = mask.expand(target.shape[0], -1)
    if mask.shape[0] != target.shape[0]:
        raise ValueError(
            f"action_dim_is_pad batch {mask.shape[0]} does not match target batch {target.shape[0]}."
        )
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask


def _mask_action_dim_tensor(tensor: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
    if action_dim_is_pad is None:
        return tensor
    valid_mask = _action_dim_valid_mask(tensor, action_dim_is_pad)
    if valid_mask is None:
        return tensor
    return tensor.masked_fill(~valid_mask, 0)


def _apply_action_dim_padding_mask(loss: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
    valid_mask = _action_dim_valid_mask(loss, action_dim_is_pad)
    if valid_mask is None:
        return loss
    valid = valid_mask.to(dtype=loss.dtype)
    denom = valid.sum(dim=-1).clamp_min(1.0)
    return (loss * valid).sum(dim=-1) / denom


def _apply_action_chunk_padding_mask(loss: Tensor, action_horizon_is_pad: Tensor | None) -> Tensor:
    if action_horizon_is_pad is None:
        return loss
    valid_action = (
        (~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)).unsqueeze(1).unsqueeze(-1)
    )
    return loss * valid_action


def _combine_rollout_seeds(first_seed: int, batch_size: int) -> int:
    seed = 0
    for idx in range(batch_size):
        seed = (seed + (idx + 1) * (first_seed + idx)) % (2**63 - 1)
    return seed


def _rollout_task_signature(batch: dict[str, Any]) -> tuple[Any, ...] | None:
    task = batch.get("task")
    if task is None:
        task = batch.get("observation.language")
    if task is None:
        return None
    if isinstance(task, str):
        return (task,)
    if isinstance(task, (list, tuple)):
        return tuple(str(item) for item in task)
    return (str(task),)


def _extract_discrete_token_bins(
    generated_ids: list[int],
    start_token_id: int,
    end_token_id: int,
    token_id_to_bin: dict[int, int],
) -> list[int]:
    start_idx = None
    end_idx = None
    for idx, token_id in enumerate(generated_ids):
        if token_id == start_token_id:
            start_idx = idx
            break
    if start_idx is not None:
        for idx in range(start_idx + 1, len(generated_ids)):
            if generated_ids[idx] == end_token_id:
                end_idx = idx
                break
    span_start = 0 if start_idx is None else start_idx + 1
    span_end = len(generated_ids) if end_idx is None else end_idx
    return [
        int(token_id_to_bin[token_id])
        for token_id in generated_ids[span_start:span_end]
        if token_id in token_id_to_bin
    ]


def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return torch.dot(values, weights) / weights.sum().clamp_min(1.0)


def _weighted_per_example(
    values: Tensor,
    weights: Tensor | None,
    example_indices: Tensor,
    batch_size: int,
) -> Tensor:
    values = values.float()
    if weights is None:
        weights = torch.ones_like(values)
    else:
        weights = weights.to(device=values.device, dtype=values.dtype)
    loss_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
    weight_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
    loss_sum.scatter_add_(0, example_indices, values * weights)
    weight_sum.scatter_add_(0, example_indices, weights)
    global_weight_sum = weight_sum.sum().clamp_min(1.0)
    return loss_sum * float(batch_size) / global_weight_sum


class MolmoAct2Policy(PreTrainedPolicy):
    """MolmoAct2 policy wrapping the vendored HF model for LeRobot.

    Supports three training modes via ``config.action_mode``:
    ``"continuous"`` (flow-matching only), ``"discrete"`` (autoregressive
    token prediction only), or ``"both"`` (joint loss). At inference,
    ``config.inference_action_mode`` selects which head generates actions.
    """

    config_class = MolmoAct2Config
    name = "molmoact2"

    def __init__(
        self,
        config: MolmoAct2Config,
        *inputs,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta: Any | None = None,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)
        _apply_norm_tag_metadata(self.config)
        self.config.validate_features()
        del inputs, kwargs, dataset_stats, dataset_meta
        self._checkpoint_action_mode = _saved_policy_action_mode(self.config)
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self._rollout_action_generator: torch.Generator | None = None
        self._rollout_task_key: tuple[Any, ...] | None = None
        self._rollout_index_for_task = -1
        self.rtc_processor: RTCProcessor | None = None
        self.action_tokenizer: Any | None = None
        self._load_hf_model()
        _validate_inference_action_mode(self.config, self._checkpoint_action_mode)
        if self.config.enable_lora_vlm:
            self._apply_lora_adapters()
        self._install_roi_prune()
        self.init_rtc_processor()

    def _load_hf_model(self) -> None:
        require_package("transformers", extra="molmoact2")

        checkpoint_location = _resolve_checkpoint_location(
            self.config.checkpoint_path,
            revision=self.config.checkpoint_revision,
            force_download=bool(self.config.checkpoint_force_download),
        )
        model_dtype = _torch_dtype(self.config.model_dtype)
        if HFMolmoAct2Config is None or MolmoAct2ForConditionalGeneration is None:
            raise RuntimeError("transformers is required to load MolmoAct2 checkpoints.")
        hf_config = HFMolmoAct2Config.from_pretrained(
            checkpoint_location,
            token=_hf_token(),
        )
        self.model = MolmoAct2ForConditionalGeneration.from_pretrained(
            checkpoint_location,
            config=hf_config,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            token=_hf_token(),
        )
        # Keep Hub loading limited to local code plus safetensors, and verify the
        # local implementation exactly matches the checkpoint key space.
        _strict_load_safetensors_weights(self.model, checkpoint_location)
        hf_max_action_dim = int(getattr(self.model.config, "max_action_dim", -1))
        if hf_max_action_dim != int(self.config.expected_max_action_dim):
            raise ValueError(
                "MolmoAct2 checkpoint max_action_dim mismatch: "
                f"checkpoint={hf_max_action_dim}, expected={self.config.expected_max_action_dim}."
            )
        if hf_max_action_dim != 32:
            raise ValueError(
                f"MolmoAct2 released checkpoints must have max_action_dim=32, got {hf_max_action_dim}."
            )

        if not hasattr(self.model.config, "max_action_horizon"):
            raise ValueError("MolmoAct2 HF checkpoints must define `max_action_horizon`.")
        self._override_loaded_max_action_horizon(int(self.config.chunk_size))

        if not hasattr(self.model.config, "action_mode"):
            raise ValueError(
                "MolmoAct2 HF checkpoints must define `action_mode`. If this is a released "
                "MolmoAct2 checkpoint, refresh the local Hub cache with "
                "`policy.checkpoint_force_download=true` after the updated files are pushed."
            )
        checkpoint_action_mode = str(self.model.config.action_mode)
        _validate_checkpoint_action_mode(
            self.config,
            checkpoint_action_mode,
            has_action_expert=bool(getattr(self.model.config, "add_action_expert", False)),
        )

        if self.config.freeze_embedding:
            self._freeze_input_embeddings()
        if self.config.train_action_expert_only:
            self._freeze_non_action_expert_parameters()
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        if int(getattr(self.config, "roi_unfreeze_last_decoder", 0)) > 0:
            self._unfreeze_last_decoder_layers(int(self.config.roi_unfreeze_last_decoder))
        self.train(self.training)

    def reset(self) -> None:
        """Clear the action queue and rollout generator between episodes."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._rollout_action_generator = None
        # Causal predictive gating: drop the gate keep-set carried from the previous
        # episode's last replan so a new episode's first frame prunes on its own gate.
        self._roi_pred_prev_keep_idx = None

    def _set_inference_cuda_graph_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "model"):
            return
        hf_model = self._hf_model()
        enabled = bool(enabled and getattr(self.config, "enable_inference_cuda_graph", True))
        managers = [
            getattr(self._backbone(), "action_cuda_graph_manager", None),
            getattr(hf_model, "action_cuda_graph_manager", None),
            getattr(hf_model, "depth_decode_cuda_graph_manager", None),
        ]
        seen: set[int] = set()
        for manager in managers:
            if manager is None or id(manager) in seen:
                continue
            seen.add(id(manager))
            set_enabled = getattr(manager, "set_enabled", None)
            if callable(set_enabled):
                set_enabled(enabled)

    def init_rtc_processor(self) -> None:
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _action_expert(self) -> torch.nn.Module:
        return self._backbone()._require_action_expert()

    def _enable_gradient_checkpointing(self) -> None:
        enable_gradient_checkpointing = getattr(self._hf_model(), "gradient_checkpointing_enable", None)
        if callable(enable_gradient_checkpointing):
            try:
                enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable_gradient_checkpointing()
        else:
            transformer = getattr(self._backbone(), "transformer", None)
            if transformer is None:
                raise RuntimeError("gradient_checkpointing=true, but MolmoAct2 exposes no text transformer.")
            transformer.gradient_checkpointing = True

        transformer = getattr(self._backbone(), "transformer", None)
        if transformer is not None:
            transformer.gradient_checkpointing = True
        vision_backbone = getattr(self._backbone(), "vision_backbone", None)
        if vision_backbone is not None:
            vision_backbone.gradient_checkpointing = True

    def _freeze_non_action_expert_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            param.requires_grad = "action_expert" in name
            if param.requires_grad:
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("train_action_expert_only=true, but no action_expert parameters were found.")

    def _unfreeze_action_expert_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            if "action_expert" in name:
                param.requires_grad_(True)
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("enable_lora_vlm=true, but no action_expert parameters were found.")

    def _unfreeze_last_decoder_layers(self, n: int) -> None:
        """Full-FT the last `n` VLM decoder blocks (extra capacity so the model can
        adapt to operating on the FastV-pruned token set in its deep layers). These
        params land in the `vlm_params` optimizer group via requires_grad."""
        transformer = getattr(self._backbone(), "transformer", None)
        blocks = getattr(transformer, "blocks", None) if transformer is not None else None
        if blocks is None:
            raise RuntimeError("roi_unfreeze_last_decoder>0 but MolmoAct2 exposes no transformer.blocks.")
        total = len(blocks)
        n = min(int(n), total)
        unfrozen = 0
        for blk in list(blocks)[-n:]:
            for param in blk.parameters():
                param.requires_grad_(True)
                unfrozen += param.numel()
        print(
            f"[roi-fastv] unfroze last {n}/{total} decoder layers ({unfrozen / 1e6:.1f}M params)",
            flush=True,
        )

    def _current_fastv_keep(self) -> float:
        """Curriculum-annealed FastV keep fraction: linearly interpolate from
        `roi_curriculum_start` down to `roi_fastv_keep_frac` over `roi_curriculum_steps`
        forward passes. Returns `roi_fastv_keep_frac` when the curriculum is disabled."""
        end = float(self.config.roi_fastv_keep_frac)
        steps = int(getattr(self.config, "roi_curriculum_steps", 0))
        if steps <= 0:
            return end
        start = float(getattr(self.config, "roi_curriculum_start", end))
        t = min(1.0, max(0, int(getattr(self, "_roi_step", 0))) / float(steps))
        return start + (end - start) * t

    def _fastv_col_idx_from_keep(self, model_inputs: dict[str, Tensor], keep_over: Tensor):
        """Build the per-example column index [B, s_new] that keeps every non-image
        column plus the surviving image placeholders (as marked by `keep_over`, a bool
        over image placeholders in batch-major/in-sequence order). Mirrors
        `_apply_group_drop_to_inputs` but does NOT modify the inputs -- the full grid
        still enters the LLM; the reduction happens mid-forward at layer L0. Returns
        None if the batch is ragged (kept-per-example not uniform)."""
        input_ids = model_inputs.get("input_ids")
        img_id = self._resolve_image_patch_id()
        if input_ids is None or img_id is None:
            return None
        B, S = input_ids.shape
        is_img = input_ids == int(img_id)
        counts = is_img.sum(dim=1)
        n_tok = int(counts[0].item()) if counts.numel() else 0
        if n_tok == 0 or not bool((counts == n_tok).all()):
            return None
        if int(keep_over.numel()) != B * n_tok:
            return None
        keep_bt = keep_over.reshape(B, n_tok).to(device=input_ids.device)
        if not bool((keep_bt.sum(dim=1) == int(keep_bt[0].sum())).all()):
            return None  # kept-per-example must be uniform to keep a rectangular batch
        keep_cols = torch.ones(B, S, dtype=torch.bool, device=input_ids.device)
        keep_cols[is_img] = keep_bt.reshape(-1)
        idx = torch.stack(
            [keep_cols[b].nonzero(as_tuple=False).flatten() for b in range(B)], dim=0
        )  # [B, s_new]
        return idx

    def _fastv_reduce(
        self,
        idx: Tensor,
        hidden: Tensor,
        position_ids: Tensor,
        cache_position: Tensor,
        encoder_attention_mask: Tensor | None,
        model_inputs: dict[str, Tensor],
        transformer,
        action_expert,
        batch_size: int,
        num_flow_timesteps: int,
        dtype,
    ):
        """Reduce the VLM sequence to the FastV-kept columns and rebuild every
        seq-dependent tensor the deeper layers + action-expert cross-attention consume:
        hidden states, position ids (ORIGINAL positions kept -> RoPE geometry preserved),
        causal mask, cross-attention mask, rotary embeddings, cache positions."""
        B, S, D = hidden.shape
        s_new = int(idx.shape[1])
        hidden_r = hidden.gather(1, idx.unsqueeze(-1).expand(B, s_new, D))
        if os.environ.get("ROI_PRUNE_DEBUG") and not getattr(self, "_roi_fastv_dbg", False):
            self._roi_fastv_dbg = True
            print(
                f"[roi-fastv] in-LLM cut @L0={int(self.config.roi_fastv_layer)} | "
                f"seq {S}->{s_new} | keep~{self._current_fastv_keep():.2f}",
                flush=True,
            )

        if torch.is_tensor(position_ids) and position_ids.dim() == 2 and position_ids.shape[0] == B:
            pid_r = position_ids.gather(1, idx)
        else:
            pid_full = torch.as_tensor(position_ids, device=hidden.device).reshape(1, -1).expand(B, S)
            pid_r = pid_full.gather(1, idx)

        cache_r = torch.arange(s_new, device=hidden.device)

        am = model_inputs.get("attention_mask")
        am_r = am.gather(1, idx) if (torch.is_tensor(am) and am.dim() == 2 and am.shape[1] == S) else None
        tt = model_inputs.get("token_type_ids")
        tt_r = tt.gather(1, idx) if (torch.is_tensor(tt) and tt.dim() == 2 and tt.shape[1] == S) else None
        backbone = self._backbone()
        causal_r = backbone._build_native_attention_bias(
            inputs_embeds=hidden_r, attention_mask=am_r, token_type_ids=tt_r, past_key_values=None
        )

        enc_r = encoder_attention_mask
        if (
            torch.is_tensor(encoder_attention_mask)
            and encoder_attention_mask.dim() == 2
            and encoder_attention_mask.shape[1] == S
        ):
            enc_r = encoder_attention_mask.gather(1, idx)
        cross_r = action_expert._build_cross_attention_mask(enc_r, batch_size, dtype)
        cross_r = _expand_mask(cross_r, num_flow_timesteps)

        pe = None
        pem = None
        if transformer.config.rope_scaling_layers is not None:
            pem = {
                "default": transformer.rotary_embs["default"](hidden_r, pid_r),
                "scaling": transformer.rotary_embs["scaling"](hidden_r, pid_r),
            }
        else:
            pe = transformer.rotary_emb(hidden_r, pid_r)
        return hidden_r, pid_r, causal_r, cache_r, cross_r, pe, pem

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.config, "train_action_expert_only", False) and hasattr(self, "model"):
            self._hf_model().eval()
            self._action_expert().train(mode)
        self._set_inference_cuda_graph_enabled(not mode)
        return self

    def _freeze_input_embeddings(self) -> None:
        embedding_modules: list[torch.nn.Module] = []
        seen_module_ids: set[int] = set()
        hf_model = self._hf_model()
        for module in (hf_model, self._backbone()):
            get_input_embeddings = getattr(module, "get_input_embeddings", None)
            if not callable(get_input_embeddings):
                continue
            embeddings = get_input_embeddings()
            if embeddings is None or id(embeddings) in seen_module_ids:
                continue
            embedding_modules.append(embeddings)
            seen_module_ids.add(id(embeddings))

        if not embedding_modules:
            raise RuntimeError("freeze_embedding=true, but MolmoAct2 checkpoint exposes no input embeddings.")

        lm_head = getattr(hf_model, "lm_head", None)
        lm_head_params = {id(param) for param in lm_head.parameters()} if lm_head is not None else set()
        embedding_params = [param for embeddings in embedding_modules for param in embeddings.parameters()]
        if any(id(param) in lm_head_params for param in embedding_params):
            raise RuntimeError(
                "freeze_embedding=true would also freeze lm_head because input embeddings and lm_head "
                "share parameters in this checkpoint."
            )
        for param in embedding_params:
            param.requires_grad = False

    def _install_roi_prune(self) -> None:
        """Attach stage-D pre-ViT pruning (config + learnable gate + mask token) to
        the vision backbone. Called after LoRA wrapping so the gate/mask-token
        parameters are freshly created (trainable) instead of frozen by peft.
        A no-op (only records a disabled config) when ``roi_prune_enable`` is False.
        """
        teacher_layers = tuple(
            int(x) for x in str(getattr(self.config, "roi_teacher_layers", "9,20,21")).split(",") if x != ""
        )
        cfg = PreViTPruneConfig(
            enable=bool(getattr(self.config, "roi_prune_enable", False)),
            keep_frac=float(getattr(self.config, "roi_prune_keep_frac", 1.0)),
            select=str(getattr(self.config, "roi_prune_select", "energy")),
            gate_hidden=int(getattr(self.config, "roi_prune_gate_hidden", 256)),
            placeholder=str(getattr(self.config, "roi_prune_placeholder", "mask")),
            teacher_layers=teacher_layers,
            teacher_debias=bool(getattr(self.config, "roi_teacher_debias", True)),
            distill_weight=float(getattr(self.config, "roi_distill_weight", 1.0)),
            gate_seam=int(getattr(self.config, "roi_prune_gate_seam", 0)),
            group_drop=bool(getattr(self.config, "roi_prune_group_drop", False)),
        )
        vision_backbone = getattr(self._backbone(), "vision_backbone", None)
        if vision_backbone is None:
            if cfg.enable:
                raise RuntimeError("roi_prune_enable=true, but MolmoAct2 exposes no vision_backbone.")
            return

        vision_backbone.roi_cfg = cfg
        vision_backbone.roi_external_keep_idx = None
        if not cfg.enable:
            return

        hidden_size = int(vision_backbone.image_vit.config.hidden_size)
        feat_dim = hidden_size * len(vision_backbone.vit_layers)
        ref = vision_backbone.image_vit.patch_embedding.weight
        device, dtype = ref.device, ref.dtype

        if cfg.uses_gate:
            # Task-aware gate: condition the keep-logit on a pooled instruction
            # embedding so the gate can localise the *task-relevant* object (a plain
            # content gate cannot recover the teacher ROI; see artifacts/actionattn).
            task_dim = None
            _emb = self._token_embedding()
            if _emb is not None:
                _w = getattr(_emb, "weight", None)
                if _w is not None and _w.dim() >= 2:
                    task_dim = int(_w.shape[-1])
                else:
                    for _p in _emb.parameters():
                        if _p.dim() >= 2:
                            task_dim = int(_p.shape[-1])
                            break
            logging.getLogger("lerobot").info(
                "ROI task-aware gate: resolved task_dim=%s (embedding=%s)",
                task_dim,
                type(_emb).__name__ if _emb is not None else None,
            )
            if task_dim:
                # patch->instruction cross-attention: real task interaction (an
                # additive task term is inert; see artifacts/actionattn ablation).
                gate = ROIGateXAttn(hidden_size, task_dim, cfg.gate_hidden)
            else:
                gate = ROIGate(hidden_size, cfg.gate_hidden)
            gate = gate.to(device=device, dtype=dtype)
            for p in gate.parameters():
                p.requires_grad_(True)
            vision_backbone.roi_gate = gate
            vision_backbone.roi_task_dim = task_dim
        if cfg.placeholder == "mask":
            mask = torch.nn.Parameter(torch.zeros(feat_dim, device=device, dtype=dtype))
            vision_backbone.register_parameter("roi_mask_token", mask)

        n_roi_bb = sum(p.numel() for n, p in vision_backbone.named_parameters() if "roi_" in n)
        n_roi_self = sum(
            p.numel() for n, p in self.named_parameters() if "roi_" in n and p.requires_grad
        )
        logging.getLogger("lerobot").info(
            f"ROI pre-ViT prune installed: select={cfg.select} keep_frac={cfg.keep_frac} "
            f"placeholder={cfg.placeholder} feat_dim={feat_dim} "
            f"roi_params_backbone={n_roi_bb} roi_params_optimized={n_roi_self}"
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Return optimizer param groups with per-component learning rates."""
        vit_params: list[Tensor] = []
        connector_params: list[Tensor] = []
        action_expert_params: list[Tensor] = []
        roi_params: list[Tensor] = []
        vlm_params: list[Tensor] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "roi_gate" in name or "roi_mask_token" in name:
                roi_params.append(param)
            elif "action_expert" in name:
                action_expert_params.append(param)
            elif any(part in name for part in ("image_pooling_2d", "image_projector")):
                connector_params.append(param)
            elif any(part in name for part in ("vision", "image_encoder", "vit")):
                vit_params.append(param)
            elif any(part in name for part in ("multi_modal_projector", "connector", "mm_projector")):
                connector_params.append(param)
            else:
                vlm_params.append(param)

        vlm_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_lr
        vit_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_vit_lr
        connector_lr = 5e-5 if self.config.enable_lora_vlm else self.config.optimizer_connector_lr

        groups: list[dict[str, Any]] = []
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        if vit_params:
            groups.append({"params": vit_params, "lr": vit_lr})
        if connector_params:
            groups.append({"params": connector_params, "lr": connector_lr})
        if action_expert_params:
            groups.append({"params": action_expert_params, "lr": self.config.optimizer_action_expert_lr})
        if roi_params:
            groups.append({"params": roi_params, "lr": self.config.roi_prune_gate_lr})
        return groups

    def _model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        compute_dtype = _torch_dtype(self.config.model_dtype)
        return {
            key: value.to(dtype=compute_dtype) if value.is_floating_point() else value
            for key, value in batch.items()
            if key in _MODEL_INPUT_KEYS and value is not None
        }

    def _output_action_dim(self, batch: dict[str, Tensor]) -> int:
        action_feature = self.config.output_features.get(ACTION)
        if action_feature is not None and action_feature.shape:
            action_dim = int(action_feature.shape[0])
            if action_dim > 0:
                return action_dim

        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            valid_counts = (~action_dim_is_pad.to(dtype=torch.bool)).sum(dim=-1)
            if bool((valid_counts == valid_counts[0]).all()) and int(valid_counts[0]) > 0:
                return int(valid_counts[0])

        raise RuntimeError("MolmoAct2 inference requires a positive action dimension in output_features.")

    def _hf_model(self):
        base_model = getattr(self.model, "base_model", None)
        wrapped_model = getattr(base_model, "model", None) if base_model is not None else None
        return wrapped_model if wrapped_model is not None else self.model

    def _backbone(self):
        return self._hf_model().model

    def _token_embedding(self):
        """Return the input token embedding module, trying the HF model then the
        backbone (mirrors _freeze_input_embeddings). Used to build the task-aware
        gate's instruction vector."""
        for module in (self._hf_model(), self._backbone()):
            get = getattr(module, "get_input_embeddings", None)
            if callable(get):
                try:
                    emb = get()
                except Exception:
                    emb = None
                if emb is not None:
                    return emb
        return None

    def _override_loaded_max_action_horizon(self, action_horizon: int) -> None:
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}.")
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is not None:
                cfg.max_action_horizon = int(action_horizon)

    def _generation_action_horizon(self) -> int:
        chunk_size = getattr(self.config, "chunk_size", None)
        if chunk_size is not None:
            return int(chunk_size)
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is None:
                continue
            value = getattr(cfg, "max_action_horizon", None)
            if value is not None:
                return int(value)
        raise RuntimeError("MolmoAct2 could not resolve an action generation horizon.")

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        backbone = self._backbone()
        get_encoder_attention_mask = getattr(backbone, "_get_encoder_attention_mask", None)
        if callable(get_encoder_attention_mask):
            mask = get_encoder_attention_mask(input_ids, attention_mask)
        elif attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
        elif input_ids is not None:
            mask = input_ids != -1
        else:
            return None

        if getattr(self.config, "action_mode", None) != "both" or input_ids is None or mask is None:
            return mask

        mask = mask.to(dtype=torch.bool).clone()
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is not None:
            mask &= input_ids != int(eos_token_id)
        return _mask_discrete_action_spans(
            input_ids=input_ids,
            mask=mask,
            start_token_id=getattr(self.model.config, "action_start_token_id", None),
            end_token_id=getattr(self.model.config, "action_end_token_id", None),
        )

    def _load_discrete_action_tokenizer(self) -> Any:
        if self.action_tokenizer is None:
            require_package("transformers", extra="molmoact2")
            require_package("scipy", extra="molmoact2")

            if UniversalActionProcessor is None:
                raise RuntimeError("transformers and scipy are required to load MolmoAct2 action tokenizer.")
            self.action_tokenizer = UniversalActionProcessor.from_pretrained_local(
                self.config.discrete_action_tokenizer,
            )
        return self.action_tokenizer

    def _resolve_inference_action_mode(self, requested_mode: str | None) -> str:
        return _resolve_inference_action_mode(self.config, requested_mode, self._checkpoint_action_mode)

    def _rollout_generator_for_inputs(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Generator | None:
        if not bool(getattr(self.config, "per_episode_seed", False)):
            return None
        if self._rollout_action_generator is not None:
            return self._rollout_action_generator

        task_signature = _rollout_task_signature(batch)
        if task_signature != self._rollout_task_key:
            self._rollout_task_key = task_signature
            self._rollout_index_for_task = 0
        else:
            self._rollout_index_for_task += 1

        base_seed = int(getattr(self.config, "eval_seed", None) or 0)
        first_seed = base_seed + self._rollout_index_for_task * batch_size
        generator_device = (
            device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(_combine_rollout_seeds(first_seed, batch_size))
        self._rollout_action_generator = generator
        return generator

    def _prepare_flow_matching_tensors(
        self,
        *,
        actions: Tensor,
        action_dim_is_pad: Tensor | None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_expert = self._backbone()._require_action_expert()
        action_dtype = next(action_expert.parameters()).dtype
        actions = actions.to(dtype=action_dtype)
        batch_size = int(actions.shape[0])
        device = actions.device
        num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))

        if timesteps is None:
            timesteps = (
                _sample_beta_timesteps(
                    batch_size=batch_size * num_flow_timesteps,
                    device=device,
                    cutoff=self.config.flow_matching_cutoff,
                    time_offset=self.config.flow_matching_time_offset,
                    time_scale=self.config.flow_matching_time_scale,
                    alpha=self.config.flow_matching_beta_alpha,
                    beta=self.config.flow_matching_beta_beta,
                )
                .to(dtype=action_dtype)
                .view(batch_size, num_flow_timesteps)
            )
        else:
            expected_timesteps_shape = (batch_size, num_flow_timesteps)
            timesteps = timesteps.to(device=device, dtype=action_dtype)
            if tuple(timesteps.shape) != expected_timesteps_shape:
                raise ValueError(
                    f"flow timesteps must have shape {expected_timesteps_shape}, got {tuple(timesteps.shape)}."
                )

        if self.config.mask_action_dim_padding:
            actions = _mask_action_dim_tensor(actions, action_dim_is_pad)

        expected_noise_shape = (batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2])
        if noise is None:
            noise = torch.randn(*expected_noise_shape, device=device, dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
            if tuple(noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"flow noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}."
                )
        if self.config.mask_action_dim_padding:
            noise = _mask_action_dim_tensor(noise, action_dim_is_pad)

        t_broadcast = timesteps.view(batch_size, num_flow_timesteps, 1, 1)
        actions_expanded = actions.unsqueeze(1).expand(-1, num_flow_timesteps, -1, -1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions_expanded
        target_velocity = actions_expanded - noise
        return actions, timesteps, xt, target_velocity

    def _prepare_joint_training_backbone_inputs(
        self,
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | dict[str, Any], Tensor, Tensor]:
        backbone = self._backbone()
        input_ids = model_inputs.get("input_ids")
        inputs_embeds = model_inputs.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "MolmoAct2 joint flow training requires exactly one of input_ids or inputs_embeds."
            )

        images = None
        token_pooling = None
        merge_visual_inputs = getattr(backbone, "merge_visual_inputs", None)
        if callable(merge_visual_inputs):
            images, token_pooling = merge_visual_inputs(
                input_ids=input_ids,
                pixel_values=model_inputs.get("pixel_values"),
                image_token_pooling=model_inputs.get("image_token_pooling"),
                image_grids=model_inputs.get("image_grids"),
                image_num_crops=model_inputs.get("image_num_crops"),
                pixel_values_videos=model_inputs.get("pixel_values_videos"),
                video_token_pooling=model_inputs.get("video_token_pooling"),
                video_grids=model_inputs.get("video_grids"),
            )
        elif (
            model_inputs.get("pixel_values") is not None
            or model_inputs.get("pixel_values_videos") is not None
        ):
            raise RuntimeError("MolmoAct2 checkpoint does not expose merge_visual_inputs for joint training.")

        if images is not None and inputs_embeds is not None:
            raise ValueError("MolmoAct2 joint flow training cannot combine inputs_embeds with visual inputs.")
        if inputs_embeds is None:
            inputs_embeds, _image_features = backbone.build_input_embeddings(input_ids, images, token_pooling)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = model_inputs.get("attention_mask")
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            causal_mask_mapping = backbone._build_native_attention_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=model_inputs.get("token_type_ids"),
                past_key_values=None,
            )
        return inputs_embeds, causal_mask_mapping, position_ids, cache_position

    @staticmethod
    def _decoder_layer_kv_outputs(
        layer_outputs: tuple[Any, ...], *, output_attentions: bool
    ) -> tuple[Tensor, Tensor]:
        output_idx = 2 if output_attentions else 1
        return layer_outputs[output_idx], layer_outputs[output_idx + 1]

    @staticmethod
    def _action_time_conditioning(action_expert: torch.nn.Module, timesteps: Tensor) -> Tensor:
        time_conditioning = getattr(action_expert, "_time_conditioning", None)
        if callable(time_conditioning):
            return time_conditioning(timesteps)
        return action_expert.time_embed(timesteps)

    def _compute_flow_matching_loss_joint_per_layer(
        self,
        *,
        batch: dict[str, Tensor],
        model_inputs: dict[str, Tensor],
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        action_expert = backbone._require_action_expert()
        if transformer is None:
            raise RuntimeError("MolmoAct2 joint flow training requires a patchable text transformer.")
        if len(action_expert.blocks) != int(transformer.config.num_hidden_layers):
            raise RuntimeError(
                "MolmoAct2 joint flow training requires one action expert block per text transformer layer."
            )

        # ROI action-attention teacher capture (Phase-0 validation): when enabled,
        # record each cross-attn block's action->encoder attention so we can inspect
        # what the action expert looks at over the image tokens. No-op unless the
        # ROI_DUMP_TEACHER env flag is set.
        _teacher_dump = bool(int(os.environ.get("ROI_DUMP_TEACHER", "0") or "0"))
        _capture = _teacher_dump or bool(getattr(self, "_roi_capture_attn", False))
        if _capture:
            for _blk in action_expert.blocks:
                _blk.cross_attn.capture_attn = True

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
                f"Backbone batch size {hidden_states.shape[0]} does not match action batch size {batch_size}."
            )

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
                seq_len=actions.shape[1],
                device=device,
                dtype=actions.dtype,
            )

        cross_mask = action_expert._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            actions.dtype,
        )
        cross_mask = _expand_mask(cross_mask, num_flow_timesteps)
        self_mask = action_expert._build_self_attention_mask(
            action_attention_mask,
            actions.shape[1],
            device,
            actions.dtype,
        )
        self_mask = _expand_mask(self_mask, num_flow_timesteps)

        conditioning = self._action_time_conditioning(action_expert, timesteps_flat)
        action_hidden = action_expert.action_embed(xt_flat)
        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        use_gradient_checkpointing = bool(
            getattr(self.config, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
        )

        def run_layer(
            layer_idx: int, layer_hidden: Tensor, layer_action_hidden: Tensor
        ) -> tuple[Tensor, Tensor]:
            decoder_block = transformer.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                layer_hidden,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                collect_layer_kv_states=True,
            )
            next_hidden = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(layer_outputs, output_attentions=False)
            key_states = backbone._cache_to_sequence(key_states)
            value_states = backbone._cache_to_sequence(value_states)
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

            next_action_hidden = action_block(
                layer_action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                next_action_hidden = next_action_hidden * valid_action
            return next_hidden, next_action_hidden

        # FastV: drop the low-importance image columns AFTER layer L0 so every deeper
        # layer -- and the action expert's per-layer cross-attention -- runs on the
        # reduced set. `_roi_fastv_col_idx` is consumed here (cleared) so a later pass
        # can't reuse a stale index. Requires the rebuildable (tensor) causal-mask path.
        _fastv_idx = getattr(self, "_roi_fastv_col_idx", None)
        self._roi_fastv_col_idx = None
        _fastv_L0 = (
            int(self.config.roi_fastv_layer)
            if (
                self.config.roi_fastv_enable
                and _fastv_idx is not None
                and not getattr(self, "_roi_fastv_force_off", False)
            )
            else -1
        )
        if _fastv_L0 >= int(transformer.config.num_hidden_layers) or isinstance(causal_mask_mapping, dict):
            _fastv_L0 = -1
        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            if _fastv_L0 >= 0 and layer_idx == _fastv_L0:
                (
                    hidden_states,
                    position_ids,
                    causal_mask_mapping,
                    cache_position,
                    cross_mask,
                    _fastv_pe,
                    _fastv_pem,
                ) = self._fastv_reduce(
                    _fastv_idx,
                    hidden_states,
                    position_ids,
                    cache_position,
                    encoder_attention_mask,
                    model_inputs,
                    transformer,
                    action_expert,
                    batch_size,
                    num_flow_timesteps,
                    actions.dtype,
                )
                if transformer.config.rope_scaling_layers is not None:
                    position_embeddings_mapping = _fastv_pem
                else:
                    position_embeddings = _fastv_pe
                # Discrete loss (action_mode='both') reads the (now shortened)
                # last_hidden_state; reuse the group-drop realignment so `labels` are
                # gathered to the same kept columns. All dropped columns are <image>
                # placeholders (label == ignore_index), so no action-token label is lost.
                self._roi_gd_col_idx = _fastv_idx
            if use_gradient_checkpointing:
                hidden_states, action_hidden = torch.utils.checkpoint.checkpoint(
                    lambda layer_hidden, layer_action_hidden, idx=layer_idx: run_layer(
                        idx,
                        layer_hidden,
                        layer_action_hidden,
                    ),
                    hidden_states,
                    action_hidden,
                    use_reentrant=False,
                )
            else:
                hidden_states, action_hidden = run_layer(layer_idx, hidden_states, action_hidden)

        if _capture:
            self._collect_teacher_attention(
                action_expert, model_inputs, num_flow_timesteps, batch_size, dump=_teacher_dump
            )

        hidden_states = transformer.ln_f(hidden_states)
        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action
        pred_velocity = pred_velocity.reshape(
            batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2]
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = _apply_action_chunk_padding_mask(loss, batch.get("action_horizon_is_pad"))
        if self.config.mask_action_dim_padding:
            loss = _apply_action_dim_padding_mask(loss, batch.get("action_dim_is_pad"))
        loss = loss.reshape(batch_size, -1).mean(dim=1)
        if reduction == "mean":
            loss = loss.mean()
        return loss, hidden_states

    def _collect_teacher_attention(
        self,
        action_expert: torch.nn.Module,
        model_inputs: dict[str, Tensor],
        num_flow_timesteps: int,
        batch_size: int,
        dump: bool = False,
    ) -> None:
        """Collect captured action->encoder cross-attention.

        Stashes the per-layer attention `[B, n_layers, src]` (mean over heads,
        action queries and flow timesteps) on `self._roi_captured` for in-training
        use by the action-attention teacher. When `dump` is set, also writes an npz
        with the pieces needed to map attention back to the 729 ViT patches offline
        (input_ids, pooled->patch index, raw pixels) for diagnostics.
        """
        caps = [
            b.cross_attn.captured_attn
            for b in action_expert.blocks
            if getattr(b.cross_attn, "captured_attn", None) is not None
        ]
        caps_hmax = [
            b.cross_attn.captured_attn_hmax
            for b in action_expert.blocks
            if getattr(b.cross_attn, "captured_attn_hmax", None) is not None
        ]
        for b in action_expert.blocks:
            b.cross_attn.capture_attn = False
            b.cross_attn.captured_attn = None
            b.cross_attn.captured_attn_hmax = None
        if not caps:
            return

        def _per_layer(items):
            # list[n_layers] of [B*T, src] -> [B, n_layers, src] (mean over flow timesteps)
            stacked = torch.stack(items, dim=1)  # [B*T, n_layers, src]
            nl, s = stacked.shape[1], stacked.shape[2]
            return stacked.reshape(batch_size, num_flow_timesteps, nl, s).mean(dim=1)

        attn_layers = _per_layer(caps)                     # [B, n_layers, src]
        self._roi_captured = attn_layers                   # in-training teacher signal
        if not dump:
            return

        attn = attn_layers.mean(dim=1)                     # [B, src] (block-mean, back-compat)
        src = int(attn.shape[-1])

        img_id = None
        for cfg in (getattr(self._backbone(), "config", None), getattr(self._hf_model(), "config", None)):
            if cfg is not None and getattr(cfg, "image_patch_id", None) is not None:
                img_id = int(cfg.image_patch_id)
                break

        payload: dict[str, np.ndarray] = {
            "attn": attn.float().cpu().numpy(),
            "attn_layers": attn_layers.float().cpu().numpy(),
            "input_ids": model_inputs["input_ids"].cpu().numpy(),
        }
        if caps_hmax:
            payload["attn_layers_hmax"] = _per_layer(caps_hmax).float().cpu().numpy()
        if img_id is not None:
            payload["image_patch_id"] = np.asarray(img_id, dtype=np.int64)
        ppi = model_inputs.get("image_token_pooling")
        if ppi is not None:
            payload["image_token_pooling"] = ppi.cpu().numpy()
        pv = model_inputs.get("pixel_values")
        if pv is not None:
            payload["pixel_values"] = pv[: min(2, pv.shape[0])].float().cpu().numpy()
        grids = model_inputs.get("image_grids")
        if grids is not None:
            payload["image_grids"] = grids.cpu().numpy()

        path = os.environ.get("ROI_DUMP_PATH", "/outputs/teacher_dump.npz")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **payload)
        logging.getLogger("lerobot").warning(
            "[teacher-dump] wrote %s attn=%s src=%d img_id=%s",
            path,
            tuple(attn.shape),
            src,
            img_id,
        )

    def _resolve_image_patch_id(self) -> int | None:
        for cfg in (getattr(self._backbone(), "config", None), getattr(self._hf_model(), "config", None)):
            if cfg is not None and getattr(cfg, "image_patch_id", None) is not None:
                return int(cfg.image_patch_id)
        return None

    def _action_teacher_patch_scores(
        self, batch: dict[str, Tensor], model_inputs: dict[str, Tensor]
    ) -> tuple[Tensor | None, int, int]:
        """Run a full (no-grad) teacher pass and return per-image per-patch ROI scores.

        Returns (scores [n_images, num_patches], num_images_per_sample, num_patches).
        `scores` rows are ordered as the ViT sees them: sample-major, then crop
        (image index = b * num_crops + crop), matching `encode_image`'s
        `images.view(batch*num_crops, ...)`.
        """
        backbone = self._backbone()
        vb = getattr(backbone, "vision_backbone", None)
        cfg = getattr(vb, "roi_cfg", None) if vb is not None else None
        pv = model_inputs.get("pixel_values")
        if pv is not None:
            num_patches = int(pv.shape[-2])         # [.., num_patches, patch_dim]
        else:
            inp = vb.image_vit.config.image_num_patch
            num_patches = int(inp) if isinstance(inp, (int, float)) else int(np.prod(inp))
        if cfg is None:
            return None, 1, num_patches

        # --- teacher pass: full ViT (no external keep), capture cross-attention ---
        # The teacher must be a DETERMINISTIC function of the frame: run it in eval
        # mode (dropout off) so the selected keep-set is stable across steps for a
        # given image. A churning keep-set makes the trained pass chase a moving
        # target and the loss never descends. Use a shallow copy of model_inputs so
        # any per-pass scratch keys cannot leak into the trained forward.
        prev_capture = getattr(self, "_roi_capture_attn", False)
        prev_ext = getattr(vb, "roi_external_keep_idx", None)
        self._roi_capture_attn = True
        self._roi_captured = None
        vb.roi_external_keep_idx = None
        teacher_inputs = dict(model_inputs)
        # Run the teacher on the FROZEN base model (adapters disabled) so the keep-set
        # is a stationary function of the frame, and — critically — use peft's own
        # disable_adapter() context so the LoRA adapters are guaranteed re-enabled for
        # the trained pass. (A leaked disable/merge silently drops every LoRA tensor
        # from the loss graph, so the VLM never trains.)
        pm = getattr(self, "model", None)
        _disable_ctx = pm.disable_adapter() if hasattr(pm, "disable_adapter") else nullcontext()
        try:
            with torch.no_grad(), _disable_ctx:
                self._compute_flow_matching_loss_joint_per_layer(
                    batch=batch, model_inputs=teacher_inputs
                )
        finally:
            # capture is enabled per-block inside the flow method; make sure it is
            # OFF for the (trained) pass so it does not add a redundant softmax.
            action_expert = getattr(self._backbone(), "_require_action_expert", None)
            if callable(action_expert):
                for _blk in action_expert().blocks:
                    _blk.cross_attn.capture_attn = False
        self._roi_capture_attn = prev_capture
        vb.roi_external_keep_idx = prev_ext
        attn_layers = self._roi_captured  # [B, n_layers, src]
        self._roi_captured = None
        if attn_layers is None:
            return None, 1, num_patches

        # --- sink-debias + select teacher layers -> per-token score [B, src] ---
        if cfg.teacher_debias:
            attn_layers = torch.clamp(attn_layers - attn_layers.mean(dim=0, keepdim=True), min=0.0)
        layers = [layer for layer in cfg.teacher_layers if 0 <= layer < attn_layers.shape[1]]
        if not layers:
            layers = list(range(attn_layers.shape[1]))
        teacher_tok = attn_layers[:, layers, :].sum(dim=1)  # [B, src]

        img_id = self._resolve_image_patch_id()
        input_ids = model_inputs["input_ids"]
        B, src = input_ids.shape
        img_mask = input_ids == img_id
        counts = img_mask.sum(dim=1)
        n_tok = int(counts[0].item())
        if n_tok == 0 or not bool((counts == n_tok).all()):
            return None, 1, num_patches

        pv = model_inputs.get("pixel_values")
        total_images = int(pv.shape[0]) if pv is not None else B
        num_crops = max(1, total_images // B)
        tpc = max(1, n_tok // num_crops)

        tok_all = teacher_tok[img_mask].reshape(B, n_tok).float()      # [B, n_tok]
        ppi = model_inputs["image_token_pooling"]
        pool = ppi.shape[-1]
        rows = ppi.reshape(B, n_tok, pool)                            # [B, n_tok, pool]

        device = tok_all.device
        token_crop = (torch.arange(n_tok, device=device) // tpc).clamp_max(num_crops - 1)  # [n_tok]
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_tok, pool)
        crop_idx = token_crop.view(1, n_tok, 1).expand(B, n_tok, pool)
        global_img = b_idx * num_crops + crop_idx                     # [B, n_tok, pool]
        local = (rows % num_patches)
        valid = rows >= 0
        target = (global_img * num_patches + local)                  # [B, n_tok, pool]
        vals = tok_all.view(B, n_tok, 1).expand(B, n_tok, pool)

        n_images = B * num_crops
        score_flat = torch.zeros(n_images * num_patches, device=device, dtype=torch.float32)
        count_flat = torch.zeros(n_images * num_patches, device=device, dtype=torch.float32)
        tv = target[valid].reshape(-1)
        score_flat.index_add_(0, tv, vals[valid].reshape(-1).float())
        count_flat.index_add_(0, tv, torch.ones_like(tv, dtype=torch.float32))
        scores = (score_flat / count_flat.clamp_min(1.0)).view(n_images, num_patches)
        # Stash the native per-pooled-token (per-2x2-group) teacher scores + layout so
        # the group-drop path can rank whole groups without a second teacher pass.
        self._roi_group_ctx = {
            "tok_all": tok_all,          # [B, n_tok] teacher score per pooled group
            "ppi": ppi,                  # original image_token_pooling tensor
            "token_crop": token_crop,    # [n_tok] crop id of each pooled group
            "tpc": tpc,                  # pooled groups per crop
            "n_tok": n_tok,
            "num_crops": num_crops,
            "batch": B,
        }
        return scores, num_crops, num_patches

    def _roi_predict_gate_scores(
        self, batch: dict[str, Tensor], model_inputs: dict[str, Tensor]
    ) -> tuple[Tensor | None, Tensor | None]:
        """Causal predictive gate: score the ROI gate on the PAST frame
        (`past_pixel_values`, `roi_predict_horizon` steps earlier) so its top-K can
        prune the CURRENT frame's ViT. The keep-set is therefore a function of the
        past observation only -- no look-ahead -- which is exactly what the inference
        loop can supply (the gate predicted at t-H drives the single-pass prune at t).

        Returns (scores [n_images, num_patches] or None, valid [n_images] bool or
        None). `valid` is False for images whose past frame was padded at an episode
        boundary (LeRobot repeats the boundary frame + sets `*_is_pad`); the predict
        loss masks those rows so the gate is never supervised on a non-causal pair.
        Row order matches `_action_teacher_patch_scores` (image = b*num_crops+crop).
        """
        backbone = self._backbone()
        vb = getattr(backbone, "vision_backbone", None)
        if vb is None or not hasattr(vb, "roi_gate_scores_at_seam"):
            return None, None
        past = batch.get("past_pixel_values")
        if past is None or not torch.is_tensor(past):
            return None, None
        try:
            device = next(vb.parameters()).device
        except StopIteration:
            device = past.device
        past = past.to(device=device)
        compute_dtype = _torch_dtype(self.config.model_dtype)
        if past.is_floating_point():
            past = past.to(dtype=compute_dtype)
        # `past_pixel_values` arrives flat [total_crops, n_patches, pixels] (exactly
        # the layout of `pixel_values`). Rebuild the padded 4D grid
        # [B, num_crops, n_patches, pixels] with the CURRENT frame's image metadata --
        # the past frame shares the identical crop tiling -- so the seam gate scores
        # the same per-image grid it will at causal inference.
        if past.dim() == 3:
            try:
                past, _ = backbone.build_batched_images(
                    input_ids=model_inputs.get("input_ids"),
                    pixel_values=past,
                    image_token_pooling=model_inputs.get("image_token_pooling"),
                    image_grids=model_inputs.get("image_grids"),
                    image_num_crops=model_inputs.get("image_num_crops"),
                )
            except Exception:
                return None, None
        if not torch.is_tensor(past) or past.dim() != 4:
            return None, None
        B, num_crops = int(past.shape[0]), int(past.shape[1])

        # Reuse the pooled instruction tokens already stashed for encode_image so the
        # predicted keep-set is conditioned on the same task context as training.
        task_img = None
        task_tokens = getattr(vb, "roi_task_tokens", None)
        task_mask = getattr(vb, "roi_task_mask", None)
        if task_tokens is not None and int(task_tokens.shape[0]) == B:
            tok = task_tokens.to(device=past.device).repeat_interleave(num_crops, dim=0)
            msk = (
                task_mask.to(device=past.device).repeat_interleave(num_crops, dim=0)
                if task_mask is not None
                else None
            )
            task_img = (tok, msk)

        scores = vb.roi_gate_scores_at_seam(past, task_img=task_img)  # [B*num_crops, N]

        valid = None
        is_pad = batch.get("past_pixel_values_is_pad")
        if is_pad is not None and torch.is_tensor(is_pad):
            v = (~is_pad.to(dtype=torch.bool)).reshape(B)
            valid = v.repeat_interleave(num_crops).to(device=scores.device)
        return scores, valid

    def _compute_action_attention_keep_idx(
        self, batch: dict[str, Tensor], model_inputs: dict[str, Tensor]
    ) -> Tensor | None:
        """Variant A selection: top-K patches per image by the action-attention teacher."""
        backbone = self._backbone()
        vb = getattr(backbone, "vision_backbone", None)
        cfg = getattr(vb, "roi_cfg", None) if vb is not None else None
        if cfg is None:
            return None
        scores, _num_crops, num_patches = self._action_teacher_patch_scores(batch, model_inputs)
        if scores is None:
            return None
        keep = cfg.resolve_keep(num_patches)
        keep_idx = scores.topk(keep, dim=1).indices
        return keep_idx.sort(dim=1).values  # [n_images, keep], patch order preserved

    def _teacher_group_drop(self, keep_frac: float):
        """Group-drop selection: rank WHOLE 2x2 pooling groups by the teacher's native
        per-pooled-token score and keep round(keep_frac*groups) per crop/view. Returns
        (new_ppi, keep_over) where new_ppi is `image_token_pooling` with dropped groups
        set to -1 (the pooler then emits only kept tokens, shrinking the LLM prefill)
        and keep_over is a bool over the ORIGINAL valid pooled tokens (batch-major, i.e.
        the order the <image> placeholders appear in), or None if unavailable.
        """
        ctx = getattr(self, "_roi_group_ctx", None)
        if ctx is None:
            return None
        tok_all = ctx["tok_all"]                       # [B, n_tok]
        ppi = ctx["ppi"]
        token_crop = ctx["token_crop"]                 # [n_tok]
        n_tok = int(ctx["n_tok"])
        num_crops = int(ctx["num_crops"])
        B = int(ctx["batch"])
        device = tok_all.device
        rows = ppi.reshape(B, n_tok, -1)
        valid_token = (rows >= 0).any(-1)              # [B, n_tok]
        keep_group = torch.zeros(B, n_tok, dtype=torch.bool, device=device)
        # Per-crop uniform keep budget so a busy view can't starve another of tokens.
        for c in range(num_crops):
            m_c = valid_token & (token_crop.view(1, n_tok).to(device) == c)
            if int(m_c.sum()) == 0:
                continue
            nv_c = m_c.sum(-1, keepdim=True)
            k_c = torch.clamp((float(keep_frac) * nv_c.float()).round().long(), min=1)
            sc_c = torch.where(m_c, tok_all.float(), torch.full_like(tok_all, float("-inf")))
            rank_c = torch.argsort(torch.argsort(sc_c, dim=-1, descending=True), dim=-1)
            keep_group |= (rank_c < k_c) & m_c
        keep_over = keep_group.flatten()[valid_token.flatten()]
        new_ppi = torch.where(
            keep_group[:, :, None], rows, torch.full_like(rows, -1)
        ).reshape_as(ppi)
        return new_ppi, keep_over

    def _apply_group_drop_to_inputs(self, model_inputs: dict[str, Tensor], keep_over: Tensor):
        """Drop the pruned <image> placeholder columns from the LLM sequence so it is
        physically shorter (real prefill saving) and stays consistent with the pooler
        (which now emits only the kept pooled tokens). `keep_over` is a bool over the
        image placeholders in batch-major, in-sequence order. Returns a new model_inputs
        dict, or None if the per-example image-token counts are ragged (can't group-drop
        with a rectangular batch, so the caller falls back to the unpruned path).
        """
        input_ids = model_inputs.get("input_ids")
        img_id = self._resolve_image_patch_id()
        if input_ids is None or img_id is None:
            return None
        B, S = input_ids.shape
        is_img = input_ids == int(img_id)
        counts = is_img.sum(dim=1)
        n_tok = int(counts[0].item()) if counts.numel() else 0
        if n_tok == 0 or not bool((counts == n_tok).all()):
            return None
        if int(keep_over.numel()) != B * n_tok:
            return None
        keep_bt = keep_over.reshape(B, n_tok).to(device=input_ids.device)
        if not bool((keep_bt.sum(dim=1) == int(keep_bt[0].sum())).all()):
            return None  # kept-per-example must be uniform to keep a rectangular batch
        # Column keep-mask: keep every non-image column; at image columns keep the
        # surviving placeholders (in order) as dictated by keep_bt.
        keep_cols = torch.ones(B, S, dtype=torch.bool, device=input_ids.device)
        keep_cols[is_img] = keep_bt.reshape(-1)
        s_new = int(keep_cols[0].sum().item())
        idx = torch.stack(
            [keep_cols[b].nonzero(as_tuple=False).flatten() for b in range(B)], dim=0
        )  # [B, s_new]
        new_inputs = dict(model_inputs)
        new_inputs["input_ids"] = input_ids.gather(1, idx)
        for key in ("attention_mask", "token_type_ids", "position_ids"):
            t = model_inputs.get(key)
            if t is not None and torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == S:
                new_inputs[key] = t.gather(1, idx)
        # `labels` live in the raw batch (not model_inputs); stash the column index so
        # the discrete loss can gather them to the shortened length. Dropped columns are
        # always <image> placeholders (label == ignore_index), so no action-token label
        # is removed and the causal shift for action tokens is preserved.
        self._roi_gd_col_idx = idx
        if os.environ.get("ROI_PRUNE_DEBUG") and not getattr(self, "_roi_gd_dbg", False):
            self._roi_gd_dbg = True
            _kept = int(keep_bt[0].sum().item())
            print(
                f"[roi-groupdrop] seq {S}->{s_new} | img tok {n_tok}->{_kept}/crop-set "
                f"(keep_frac~{_kept / max(1, n_tok):.2f})",
                flush=True,
            )
        return new_inputs

    def _discrete_token_weights(self, valid_positions: Tensor) -> Tensor | None:
        mode = self.config.discrete_loss_token_weighting
        if mode in {"none", "token", "root_subsegments"}:
            return None
        if mode != "root_subsegments_root_tokens" and mode != "root_tokens":
            raise ValueError(f"Unsupported discrete_loss_token_weighting={mode!r}.")

        token_counts = valid_positions.sum(dim=1).to(dtype=torch.float32)
        example_weights = torch.zeros_like(token_counts)
        nonempty = token_counts > 0
        example_weights[nonempty] = 2.0 / torch.sqrt(token_counts[nonempty])
        return example_weights[:, None].expand_as(valid_positions)[valid_positions].to(dtype=torch.float32)

    def _discrete_loss_from_backbone_outputs(
        self,
        batch: dict[str, Tensor],
        outputs: Any,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor | None]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        labels = batch.get("labels")
        if labels is None:
            raise RuntimeError("MolmoAct2 discrete training requires labels.")
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise RuntimeError("MolmoAct2 backbone did not return last_hidden_state.")

        # Group-drop shortened the LLM sequence (and thus hidden_states); realign the
        # labels to the same kept columns so the masked index matches. Consume the index
        # so a subsequent non-group-drop step can't reuse it.
        gd_idx = getattr(self, "_roi_gd_col_idx", None)
        self._roi_gd_col_idx = None
        if (
            gd_idx is not None
            and torch.is_tensor(labels)
            and labels.dim() == 2
            and gd_idx.dim() == 2
            and gd_idx.shape[0] == labels.shape[0]
            and gd_idx.shape[1] <= labels.shape[1]
            and hidden_states.shape[1] == gd_idx.shape[1]
        ):
            labels = labels.gather(1, gd_idx.to(labels.device))

        ignore_index = -100
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        valid_positions = shift_labels != ignore_index
        if not bool(valid_positions.any()):
            raise RuntimeError("MolmoAct2 discrete training labels contain no valid action tokens.")

        hidden_size = hidden_states.shape[-1]
        selected_hidden = hidden_states.reshape(-1, hidden_size)[valid_positions.reshape(-1)]
        selected_labels = shift_labels.reshape(-1)[valid_positions.reshape(-1)].to(
            device=hidden_states.device
        )
        logits = F.linear(selected_hidden, self.model.lm_head.weight).float()
        log_z = logits.logsumexp(dim=-1)
        target_logits = logits.gather(dim=-1, index=selected_labels[:, None]).squeeze(-1)
        token_ce_loss = log_z - target_logits
        token_weights = self._discrete_token_weights(valid_positions)
        if reduction == "none":
            example_indices = valid_positions.nonzero(as_tuple=False)[:, 0].to(device=hidden_states.device)
            ce_loss = _weighted_per_example(
                token_ce_loss,
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            ce_loss = _weighted_mean(token_ce_loss, token_weights)
        if not self.config.softmax_auxiliary_loss:
            return ce_loss, None

        if reduction == "none":
            z_loss = self.config.softmax_auxiliary_loss_scale * _weighted_per_example(
                log_z.pow(2),
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            z_loss = self.config.softmax_auxiliary_loss_scale * _weighted_mean(log_z.pow(2), token_weights)
        return ce_loss, z_loss

    def _action_token_id_to_bin(self) -> dict[int, int]:
        method = getattr(self.model, "_action_token_id_to_bin", None)
        if callable(method):
            return dict(method())
        start = getattr(self.model.config, "action_token_start_id", None)
        num_tokens = int(getattr(self.model.config, "num_action_tokens", 0) or 0)
        if start is None or num_tokens <= 0:
            return {}
        return {int(start) + idx: idx for idx in range(num_tokens)}

    def _require_discrete_eos_token_id(self) -> int:
        method = getattr(self.model, "_require_eos_token_id", None)
        if callable(method):
            return int(method())
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is None and getattr(self.model, "generation_config", None) is not None:
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise RuntimeError("Discrete action generation requires eos_token_id in the checkpoint config.")
        return int(eos_token_id)

    def _discrete_generation_max_steps(self) -> int:
        if self.config.discrete_generation_max_steps is not None:
            return int(self.config.discrete_generation_max_steps)
        return max(1, self._generation_action_horizon() * 16)

    def _continue_discrete_generation_from_output(
        self,
        initial_output: Any,
        *,
        past_key_values: Any | None,
        attention_mask: Tensor | None,
        end_token_id: int,
        max_steps: int,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        consume_generation_tokens = getattr(self.model, "_consume_generation_tokens", None)
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        if attention_bias is None and not callable(consume_generation_tokens):
            raise RuntimeError("MolmoAct2 checkpoint does not expose discrete token generation helpers.")
        if attention_bias is not None and not callable(ar_decode_step):
            raise RuntimeError("MolmoAct2 checkpoint does not expose graph-backed AR decode helpers.")

        generated_tokens: list[Tensor] = []
        current_output = initial_output
        current_past_key_values = past_key_values
        current_attention_mask = attention_mask
        hit_end = False
        for _ in range(int(max_steps)):
            next_token = torch.argmax(current_output.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token)
            if bool((next_token == int(end_token_id)).all()):
                hit_end = True
                break
            if attention_bias is None:
                current_output, current_attention_mask = consume_generation_tokens(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_mask=current_attention_mask,
                )
                current_past_key_values = current_output.past_key_values
            else:
                last_hidden, current_past_key_values = ar_decode_step(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_bias=attention_bias,
                )
                current_output = types.SimpleNamespace(
                    logits=self.model.lm_head(last_hidden),
                    past_key_values=current_past_key_values,
                )
        if not generated_tokens:
            raise RuntimeError("Discrete continuation generated no tokens.")
        if not hit_end:
            raise RuntimeError(
                f"Discrete continuation did not emit end token {int(end_token_id)} within {int(max_steps)} steps."
            )
        return torch.stack(generated_tokens, dim=1)

    def _make_discrete_ar_graph_decode_inputs(
        self,
        model_inputs: dict[str, Tensor],
        *,
        max_steps: int,
    ) -> tuple[Any | None, Tensor | None]:
        if not bool(getattr(self.config, "enable_inference_cuda_graph", False)):
            return None, None
        if self.training or self.model.training:
            return None, None
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        make_attention_bias = getattr(self.model, "_make_depth_decode_attention_bias", None)
        if not callable(ar_decode_step) or not callable(make_attention_bias):
            return None, None

        make_static_cache = getattr(self.model, "_make_ar_decode_static_cache", None)
        if callable(make_static_cache):
            static_cache = make_static_cache(model_inputs, max_steps=max_steps)
        else:
            graph_manager = getattr(self.model, "depth_decode_cuda_graph_manager", None)
            make_manager_static_cache = getattr(graph_manager, "make_static_cache", None)
            if not callable(make_manager_static_cache):
                return None, None
            prompt_len = int(model_inputs["input_ids"].shape[1])
            static_cache = make_manager_static_cache(max_cache_len=prompt_len + max(1, int(max_steps)))

        attention_bias = make_attention_bias(model_inputs, static_cache)
        return static_cache, attention_bias

    def _decode_discrete_action_chunk(self, generated_token_ids: Tensor, *, action_dim: int) -> Tensor:
        if (
            getattr(self.model.config, "action_start_token_id", None) is None
            or getattr(self.model.config, "action_end_token_id", None) is None
        ):
            raise RuntimeError("Discrete action generation requires <action_start>/<action_end> token IDs.")
        token_id_to_bin = self._action_token_id_to_bin()
        if not token_id_to_bin:
            raise RuntimeError(
                "Discrete action generation requires indexed action tokens in the checkpoint config."
            )

        action_tokenizer = self._load_discrete_action_tokenizer()
        if generated_token_ids.ndim == 1:
            generated_token_ids = generated_token_ids.unsqueeze(0)
        if generated_token_ids.ndim == 3:
            generated_token_ids = generated_token_ids[:, 0, :]
        if generated_token_ids.ndim != 2:
            raise ValueError(f"Unexpected generated token tensor shape {tuple(generated_token_ids.shape)}.")

        chunks: list[Tensor] = []
        for token_row in generated_token_ids:
            generated_ids = [int(token_id) for token_id in token_row.detach().cpu().tolist()]
            discrete_token_ids = _extract_discrete_token_bins(
                generated_ids,
                int(self.model.config.action_start_token_id),
                int(self.model.config.action_end_token_id),
                token_id_to_bin,
            )
            if not discrete_token_ids:
                raise RuntimeError(
                    "Model generated no decodable action tokens between <action_start>/<action_end>."
                )
            try:
                decoded = action_tokenizer.decode(
                    [discrete_token_ids],
                    time_horizon=self._generation_action_horizon(),
                    action_dim=int(action_dim),
                )
            except TypeError:
                decoded = action_tokenizer.decode([discrete_token_ids])
            action_chunk = np.asarray(decoded, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            elif action_chunk.ndim == 3:
                if int(action_chunk.shape[0]) != 1:
                    action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
                else:
                    action_chunk = action_chunk[0]
            elif action_chunk.ndim > 3:
                action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
            if action_chunk.ndim != 2:
                raise RuntimeError(f"Decoded action chunk has unexpected shape {action_chunk.shape}.")
            chunks.append(torch.as_tensor(action_chunk, device=token_row.device, dtype=torch.float32))
        return torch.stack(chunks, dim=0)

    def _generate_discrete_actions_from_inputs(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim: int,
    ) -> Tensor:
        model_inputs = _drop_trivial_attention_mask(model_inputs)
        max_steps = self._discrete_generation_max_steps()
        static_cache, attention_bias = self._make_discrete_ar_graph_decode_inputs(
            model_inputs,
            max_steps=max_steps,
        )
        prefill_kwargs: dict[str, Any] = {}
        if static_cache is not None:
            prefill_kwargs["past_key_values"] = static_cache
        prefill_output = self.model(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            **prefill_kwargs,
        )
        generated_token_ids = self._continue_discrete_generation_from_output(
            prefill_output,
            past_key_values=prefill_output.past_key_values,
            attention_mask=model_inputs.get("attention_mask"),
            end_token_id=self._require_discrete_eos_token_id(),
            max_steps=max_steps,
            attention_bias=attention_bias,
        )
        return self._decode_discrete_action_chunk(generated_token_ids, action_dim=action_dim)

    def _generate_actions_from_inputs_with_rtc(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim_is_pad: Tensor | None,
        num_steps: int | None,
        generator: torch.Generator | None,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
        execution_horizon: int | None,
    ) -> Tensor:
        backbone = self._backbone()
        action_expert = self._action_expert()
        outputs = backbone(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )

        steps = int(num_steps or backbone.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        source_tensor = encoder_kv_states[0][0]
        batch_size = int(source_tensor.shape[0])
        device = source_tensor.device
        trajectory = torch.randn(
            batch_size,
            self._generation_action_horizon(),
            int(backbone.config.max_action_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.config.mask_action_dim_padding:
            trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)

        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=trajectory.dtype)
            for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )

        dt = 1.0 / steps
        mask_enabled = self.config.mask_action_dim_padding
        for idx, flow_timestep in enumerate(flow_timesteps):
            modulation = modulation_cache[idx]

            def denoise_step(input_trajectory: Tensor, step_modulation=modulation) -> Tensor:
                velocity = action_expert.forward_with_context(
                    input_trajectory,
                    step_modulation.conditioning,
                    context=action_context,
                    modulation=step_modulation,
                )
                if mask_enabled:
                    velocity = _mask_action_dim_tensor(velocity, action_dim_is_pad)
                return velocity

            if self._rtc_enabled():
                if self.rtc_processor is None:
                    raise RuntimeError("RTC is enabled but rtc_processor is not initialized.")

                def rtc_denoise_step(input_trajectory: Tensor) -> Tensor:
                    return -denoise_step(input_trajectory)

                rtc_time = 1.0 - float(flow_timestep[0].item())
                rtc_velocity = self.rtc_processor.denoise_step(
                    x_t=trajectory,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=int(inference_delay or 0),
                    time=rtc_time,
                    original_denoise_step_partial=rtc_denoise_step,
                    execution_horizon=execution_horizon,
                )
                velocity = -rtc_velocity
            else:
                velocity = denoise_step(trajectory)

            trajectory = trajectory + dt * velocity
            if mask_enabled:
                trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=float(flow_timestep[0].item()), x_t=trajectory, v_t=velocity)

        return trajectory

    # ------------------------------------------------------------------ #
    # FastV in-LLM pruning at INFERENCE (eval parity with training).       #
    # Opt-in via ROI_FASTV_INFER=1. Default off -> stock (gate-only) path, #
    # so training and existing eval are byte-for-byte unaffected.          #
    # ------------------------------------------------------------------ #
    def _fastv_infer_enabled(self) -> bool:
        if not bool(getattr(self.config, "roi_fastv_enable", False)):
            return False
        return bool(int(os.environ.get("ROI_FASTV_INFER", "0") or "0"))

    def _set_gate_task_tokens(self, model_inputs: dict[str, Tensor]) -> None:
        """Feed the ROI gate the same pooled-instruction conditioning it saw in
        training so the eval keep-set matches. No-op when the gate is task-free."""
        vb = getattr(self._backbone(), "vision_backbone", None)
        cfg = getattr(vb, "roi_cfg", None) if vb is not None else None
        if (
            cfg is not None
            and getattr(cfg, "enable", False)
            and getattr(cfg, "uses_gate", False)
            and getattr(vb, "roi_task_dim", None)
        ):
            _tt = self._compute_task_tokens(model_inputs)
            vb.roi_task_tokens = _tt[0] if _tt is not None else None
            vb.roi_task_mask = _tt[1] if _tt is not None else None

    def _fastv_group_ctx_from_gate(self, model_inputs: dict[str, Tensor]) -> bool:
        """Build `_roi_group_ctx` (per-pooled-token importance) from the ROI gate's
        per-patch scores captured during the ViT encode, so the training-time
        `_teacher_group_drop` -> `_fastv_col_idx_from_keep` path can drive the in-LLM
        FastV cut at inference (no action-attention teacher is run at eval). Mirrors
        `_action_teacher_patch_scores`'s pooled-token layout. Returns True on success."""
        self._roi_group_ctx = None
        vb = getattr(self._backbone(), "vision_backbone", None)
        gate_scores = getattr(vb, "roi_last_gate_scores", None) if vb is not None else None
        num_patches = getattr(vb, "roi_last_num_patches", None) if vb is not None else None
        input_ids = model_inputs.get("input_ids")
        ppi = model_inputs.get("image_token_pooling")
        if gate_scores is None or num_patches is None or input_ids is None or ppi is None:
            return False
        img_id = self._resolve_image_patch_id()
        if img_id is None:
            return False
        gate_scores = gate_scores.float()                       # [n_images, num_patches]
        num_patches = int(num_patches)
        B, _S = input_ids.shape
        img_mask = input_ids == int(img_id)
        counts = img_mask.sum(dim=1)
        n_tok = int(counts[0].item()) if counts.numel() else 0
        if n_tok == 0 or not bool((counts == n_tok).all()):
            return False
        n_images = int(gate_scores.shape[0])
        num_crops = max(1, n_images // B)
        if num_crops * B != n_images:
            return False
        tpc = max(1, n_tok // num_crops)
        device = gate_scores.device
        pool = ppi.shape[-1]
        rows = ppi.reshape(B, n_tok, pool).to(device)
        token_crop = (torch.arange(n_tok, device=device) // tpc).clamp_max(num_crops - 1)
        crop_idx = token_crop.view(1, n_tok, 1).expand(B, n_tok, pool)
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_tok, pool)
        global_img = b_idx * num_crops + crop_idx
        local = rows % num_patches
        valid = rows >= 0
        target = (global_img * num_patches + local).clamp_min(0)
        gate_flat = gate_scores.reshape(-1)                     # [n_images*num_patches]
        gathered = gate_flat[target.reshape(-1)].reshape(B, n_tok, pool)
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
        denom = valid.sum(-1).clamp_min(1).to(gathered.dtype)
        tok_all = gathered.sum(-1) / denom                     # [B, n_tok]
        self._roi_group_ctx = {
            "tok_all": tok_all,
            "ppi": ppi,
            "token_crop": token_crop,
            "tpc": tpc,
            "n_tok": n_tok,
            "num_crops": num_crops,
            "batch": B,
        }
        return True

    def _generate_actions_fastv(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim_is_pad: Tensor | None,
        num_steps: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Continuous flow-matching generation with the FastV in-LLM cut applied at eval
        (matches training): run decoder layers 0..L0 on the full sequence, drop the
        low-importance image columns (ranked by the distilled gate), then run the deeper
        layers + per-layer action cross-attention on the reduced set. Falls back to the
        stock backbone path whenever the cut cannot be applied safely."""
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        action_expert = backbone._require_action_expert()

        def _stock() -> Tensor:
            return backbone.generate_actions_from_inputs(
                **model_inputs,
                action_dim_is_pad=action_dim_is_pad,
                action_horizon=self._generation_action_horizon(),
                num_steps=num_steps,
                generator=generator,
            )

        # Depth-gating scales KV by a full-length token mask; it is incompatible with the
        # per-layer variable-length KV the cut produces. It is off for these runs, but
        # guard anyway so a depth-gated checkpoint stays correct (gate-only) instead of
        # silently wrong.
        if (
            transformer is None
            or getattr(backbone, "action_expert_depth_gate", None) is not None
            or len(action_expert.blocks) != int(transformer.config.num_hidden_layers)
        ):
            return _stock()

        hidden_states, causal_mask_mapping, position_ids, cache_position = (
            self._prepare_joint_training_backbone_inputs(model_inputs)
        )
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch_size = int(hidden_states.shape[0])

        # Build the FastV keep-set from the gate (single-pass, no teacher at eval).
        _fk = float(self.config.roi_fastv_keep_frac)
        col_idx = None
        if (
            batch_size == 1
            and _fk < 1.0
            and not isinstance(causal_mask_mapping, dict)
            and int(self.config.roi_fastv_layer) < int(transformer.config.num_hidden_layers)
            and self._fastv_group_ctx_from_gate(model_inputs)
        ):
            _gd = self._teacher_group_drop(_fk)
            if _gd is not None:
                _, keep_over = _gd
                col_idx = self._fastv_col_idx_from_keep(model_inputs, keep_over)
        self._roi_group_ctx = None
        if col_idx is None:
            return _stock()  # nothing to cut -> equivalent to gate-only

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
            position_embeddings = None
        else:
            position_embeddings_mapping = None
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        _fastv_L0 = int(self.config.roi_fastv_layer)
        encoder_kv_states: list[tuple[Tensor, Tensor]] = []
        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            if layer_idx == _fastv_L0:
                (
                    hidden_states,
                    position_ids,
                    causal_mask_mapping,
                    cache_position,
                    _cross_r,
                    _pe,
                    _pem,
                ) = self._fastv_reduce(
                    col_idx,
                    hidden_states,
                    position_ids,
                    cache_position,
                    None,
                    model_inputs,
                    transformer,
                    action_expert,
                    batch_size,
                    1,
                    dtype,
                )
                if transformer.config.rope_scaling_layers is not None:
                    position_embeddings_mapping = _pem
                else:
                    position_embeddings = _pe
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings
            layer_outputs = transformer.blocks[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                collect_layer_kv_states=True,
            )
            hidden_states = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(
                layer_outputs, output_attentions=False
            )
            key_states = backbone._cache_to_sequence(key_states)
            value_states = backbone._cache_to_sequence(value_states)
            encoder_kv_states.append((key_states, value_states))

        # Denoise against the per-layer (FastV-reduced) KV. cross_mask is None at eval
        # (batch=1, no encoder padding), so per-layer KV of differing lengths (full
        # before L0, reduced after) is handled natively by the cross-attention.
        horizon = self._generation_action_horizon()
        max_action_dim = int(backbone.config.max_action_dim)
        trajectory_dtype = action_expert.action_embed.weight.dtype
        trajectory = torch.randn(
            (batch_size, horizon, max_action_dim),
            device=device,
            dtype=trajectory_dtype,
            generator=generator,
        )
        if self.config.mask_action_dim_padding:
            trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
        context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=None,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        steps = int(num_steps or backbone.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=torch.float32)
            for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )
        dt = 1.0 / steps
        mask_enabled = bool(self.config.mask_action_dim_padding)
        for idx in range(steps):
            step_modulation = modulation_cache[idx]
            velocity = action_expert.forward_with_context(
                trajectory,
                step_modulation.conditioning,
                context=context,
                modulation=step_modulation,
            )
            if mask_enabled:
                velocity = _mask_action_dim_tensor(velocity, action_dim_is_pad)
            trajectory = trajectory + dt * velocity
            if mask_enabled:
                trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
        return trajectory

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute training loss (flow-matching and/or discrete token loss)."""
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        model_inputs = self._model_inputs(batch)
        # Cleared every step; set only when group-drop physically shortens the LLM
        # sequence, so the discrete loss can realign `labels` to the shortened hidden
        # states (labels are NOT in _MODEL_INPUT_KEYS, so they are not shortened here).
        self._roi_gd_col_idx = None
        losses: list[Tensor] = []
        metrics: dict[str, Any] = {}

        # Variant A (dual-pass select-by-teacher): run a full no-grad forward to get
        # the sink-debiased action-attention teacher, pick the top-K patches per
        # image, and stash them so the (trained) pass below prunes the ViT to that
        # keep-set. The keep-idx MUST persist through the caller's backward() (the
        # gradient-checkpointed ViT recompute needs the same pruned shape), so we
        # overwrite it each step and rely on encode_image's shape-guard to ignore
        # any stale index (e.g. at inference).
        _vb = getattr(self._backbone(), "vision_backbone", None)
        _roi_cfg = getattr(_vb, "roi_cfg", None) if _vb is not None else None
        _dual_pass = (
            self.training
            and _roi_cfg is not None
            and getattr(_roi_cfg, "enable", False)
            and _roi_cfg.uses_action_teacher
            and _roi_cfg.keep_frac < 1.0
        )
        # A task-aware gate needs the instruction context at the (image-only) pre-ViT
        # seam, so pool the instruction-token embeddings here and stash them for
        # encode_image. Available at train and inference -> single-pass preserved.
        if (
            _roi_cfg is not None
            and getattr(_roi_cfg, "enable", False)
            and _roi_cfg.uses_gate
            and getattr(_vb, "roi_task_dim", None)
        ):
            _tt = self._compute_task_tokens(model_inputs)
            _vb.roi_task_tokens = _tt[0] if _tt is not None else None
            _vb.roi_task_mask = _tt[1] if _tt is not None else None

        if _dual_pass:
            # One no-grad teacher forward (frozen base, adapters disabled) yields the
            # sink-debiased action-attention per patch. Variant A prunes the ViT to
            # the teacher top-K; variants B/C additionally stash the full per-patch
            # scores so the cheap gate can be distilled toward them (single-pass at
            # inference). Clear stale state first so a failed teacher pass can't leak
            # a previous step's keep-set / target into this forward.
            _vb.roi_external_keep_idx = None
            _vb.roi_teacher_scores = None
            scores, _num_crops, num_patches = self._action_teacher_patch_scores(batch, model_inputs)
            if scores is not None:
                if bool(getattr(_roi_cfg, "group_drop", False)):
                    # Prune whole 2x2 pooling groups: the pooler emits only kept tokens
                    # (fewer image tokens -> shorter LLM prefill = real backbone saving).
                    # ViT runs full here; the gate is still scored at the seam + distilled
                    # so a single-pass gate can drive group selection at inference (v2).
                    _gd = self._teacher_group_drop(float(_roi_cfg.keep_frac))
                    if _gd is not None:
                        new_ppi, keep_over = _gd
                        # Physically shorten the sequence (drop the dropped <image>
                        # placeholder columns) so the backbone hidden states, the
                        # action-expert encoder mask, and the pooled tokens all stay
                        # consistent and the LLM prefill actually shrinks. If the image
                        # placeholders are ragged across the batch, skip group-drop for
                        # this step rather than half-applying it.
                        _short = self._apply_group_drop_to_inputs(model_inputs, keep_over)
                        if _short is not None:
                            model_inputs = _short
                            model_inputs["image_token_pooling"] = new_ppi
                            if _roi_cfg.uses_gate:
                                _vb.roi_teacher_scores = scores
                else:
                    keep = _roi_cfg.resolve_keep(num_patches)
                    if _roi_cfg.uses_gate:
                        # teacher on the CURRENT frame -> BCE target for the gate
                        _vb.roi_teacher_scores = scores
                    _vb.roi_predicted_gate_scores = None
                    _vb.roi_predicted_valid = None
                    if (
                        _roi_cfg.select == "gate_predict"
                        and int(getattr(self.config, "roi_predict_horizon", 0)) > 0
                    ):
                        # Causal predictive gating: the keep-set that prunes the
                        # CURRENT frame is the top-K of the gate scored on the PAST
                        # frame (roi_predict_horizon steps earlier). The teacher above
                        # only supervises the prediction; it never drives the prune.
                        pred, valid = self._roi_predict_gate_scores(batch, model_inputs)
                        if pred is not None and pred.shape == scores.shape:
                            _vb.roi_predicted_gate_scores = pred
                            _vb.roi_predicted_valid = valid
                            _vb.roi_external_keep_idx = (
                                pred.detach().topk(keep, dim=1).indices.sort(dim=1).values
                            )
                        else:
                            # No usable past frame this step -> fall back to the
                            # teacher keep-set (non-causal) so the encode still trains;
                            # the predict loss is skipped (roi_predicted_gate_scores=None).
                            _vb.roi_external_keep_idx = (
                                scores.topk(keep, dim=1).indices.sort(dim=1).values
                            )
                    else:
                        _vb.roi_external_keep_idx = scores.topk(keep, dim=1).indices.sort(dim=1).values

        # FastV in-LLM pruning: reuse the teacher's per-pooled-token action-attention
        # scores (stashed in _roi_group_ctx by the pass above) to mark which image
        # columns survive the mid-forward cut at layer L0. The full grid still entered
        # the LLM; only deeper layers + the action-expert cross-attention see the
        # reduced set. Cleared each step; consumed once in the flow forward.
        self._roi_fastv_col_idx = None
        if self.training:
            self._roi_step = int(getattr(self, "_roi_step", 0)) + 1
        if (
            self.config.roi_fastv_enable
            and self.training
            and getattr(self, "_roi_group_ctx", None) is not None
        ):
            _fk = self._current_fastv_keep()
            if _fk < 1.0:
                _gd = self._teacher_group_drop(_fk)
                if _gd is not None:
                    _, _fv_keep_over = _gd
                    self._roi_fastv_col_idx = self._fastv_col_idx_from_keep(model_inputs, _fv_keep_over)

        return self._forward_losses(batch, model_inputs, reduction, losses, metrics)

    def _compute_task_tokens(self, model_inputs: dict[str, Tensor]):
        """Gather the instruction token embeddings (non-image, attention-valid) into a
        padded [B, T, D] tensor + [B, T] bool mask for the cross-attention gate. Uses
        the frozen input embedding and detaches: the gate learns from fixed
        instruction features, not by backprop into the LLM embedding. Returns
        (tokens, mask) or None."""
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            return None
        embed = self._token_embedding()
        if embed is None:
            return None
        img_id = self._resolve_image_patch_id()
        with torch.no_grad():
            keep = torch.ones_like(input_ids, dtype=torch.bool)
            if img_id is not None:
                keep &= input_ids != int(img_id)
            am = model_inputs.get("attention_mask")
            if am is not None:
                keep &= am.to(dtype=torch.bool)
            counts = keep.sum(dim=1)
            t_max = int(counts.max().item()) if counts.numel() else 0
            if t_max == 0:
                return None
            emb = embed(input_ids)  # [B, L, D]
            b, _, d = emb.shape
            tokens = emb.new_zeros(b, t_max, d)
            mask = torch.zeros(b, t_max, dtype=torch.bool, device=emb.device)
            for i in range(b):
                idx = keep[i].nonzero(as_tuple=False).squeeze(-1)
                n = int(idx.numel())
                if n:
                    tokens[i, :n] = emb[i, idx]
                    mask[i, :n] = True
        return tokens.detach(), mask

    def _forward_losses(self, batch, model_inputs, reduction, losses, metrics):
        if self.config.action_mode == "discrete":
            outputs = self._backbone()(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()

        elif self.config.action_mode == "continuous":
            flow_loss, _ = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        else:
            flow_loss, hidden_states = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            outputs = types.SimpleNamespace(last_hidden_state=hidden_states)
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        distill_loss = self._roi_gate_distill_loss()
        if distill_loss is not None:
            losses.append(distill_loss)
            metrics["roi_distill_loss"] = distill_loss.detach().float().item()

        predict_loss = self._roi_gate_predict_loss()
        if predict_loss is not None:
            losses.append(predict_loss)
            metrics["roi_predict_loss"] = predict_loss.detach().float().item()

        loss = torch.stack(losses).sum(dim=0)
        metrics["loss"] = loss.detach().float().mean().item()
        return loss, metrics

    def _roi_gate_distill_loss(self) -> Tensor | None:
        """Variant B (gate_distill): pull the cheap pre-ViT gate toward the frozen
        action-attention teacher. The teacher supplies a per-patch keep-target (its
        top-K = 1, rest = 0) and the gate is trained as a binary ROI classifier via
        BCE, so at inference the gate reproduces the teacher's keep-set in a single
        pass (no teacher forward). Returns None when not applicable."""
        if not self.training:
            return None
        vb = getattr(self._backbone(), "vision_backbone", None)
        cfg = getattr(vb, "roi_cfg", None) if vb is not None else None
        if cfg is None or cfg.select != "gate_distill":
            return None
        gate_scores = getattr(vb, "roi_last_gate_scores", None)
        teacher_scores = getattr(vb, "roi_teacher_scores", None)
        # consume so a skipped step can't reuse a stale target/logit pair
        vb.roi_last_gate_scores = None
        vb.roi_teacher_scores = None
        if gate_scores is None or teacher_scores is None:
            return None
        if gate_scores.shape != teacher_scores.shape:
            return None
        keep = cfg.resolve_keep(teacher_scores.shape[1])
        with torch.no_grad():
            target = torch.zeros_like(teacher_scores)
            target.scatter_(1, teacher_scores.topk(keep, dim=1).indices, 1.0)
        loss = F.binary_cross_entropy_with_logits(
            gate_scores.float(), target.to(dtype=torch.float32)
        )
        _dump = os.environ.get("ROI_DUMP_AGREEMENT")
        if _dump:
            with torch.no_grad():
                gk = gate_scores.topk(keep, dim=1).indices
                # precision@K: fraction of the gate's top-K that are in the teacher's
                # top-K (== recall@K == IoU here since both sets have size K).
                prec = target.gather(1, gk).sum(dim=1).div(float(keep)).mean().item()
                base = float(keep) / float(teacher_scores.shape[1])  # random-pick baseline
                rec = {
                    "keep_frac": round(float(cfg.keep_frac), 3),
                    "n_patches": int(teacher_scores.shape[1]),
                    "keep_k": int(keep),
                    "bce": float(loss.item()),
                    "precision_at_k": float(prec),
                    "random_baseline": float(base),
                    "batch_images": int(gate_scores.shape[0]),
                    "gate_scores_requires_grad": bool(gate_scores.requires_grad),
                }
            if os.environ.get("ROI_DUMP_GRAD"):
                try:
                    _gp = vb.roi_gate.fc2.weight
                    _g = torch.autograd.grad(
                        loss, _gp, retain_graph=True, allow_unused=True
                    )[0]
                    rec["gate_grad_norm"] = (
                        float(_g.float().norm().item()) if _g is not None else -1.0
                    )
                except Exception as _e:
                    rec["gate_grad_err"] = repr(_e)[:120]
            try:
                with open(_dump, "a") as _fh:
                    _fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        return float(getattr(cfg, "distill_weight", 1.0)) * loss

    def _roi_gate_predict_loss(self) -> Tensor | None:
        """Causal predictive gating (select='gate_predict'): train the gate scored on
        the PAST frame to reproduce the action-attention teacher's keep-set on the
        CURRENT frame. Same BCE-ROI-classifier objective as gate_distill, but the
        logit/target come from DIFFERENT timesteps (t-H vs t), so at inference the
        gate predicted one step ahead can prune without any teacher / dual pass.
        Rows whose past frame was padded (episode start) are masked out. Returns None
        when not applicable."""
        if not self.training:
            return None
        vb = getattr(self._backbone(), "vision_backbone", None)
        cfg = getattr(vb, "roi_cfg", None) if vb is not None else None
        if cfg is None or cfg.select != "gate_predict":
            return None
        gate_scores = getattr(vb, "roi_predicted_gate_scores", None)
        teacher_scores = getattr(vb, "roi_teacher_scores", None)
        valid = getattr(vb, "roi_predicted_valid", None)
        # consume so a skipped step can't reuse a stale target/logit pair
        vb.roi_predicted_gate_scores = None
        vb.roi_teacher_scores = None
        vb.roi_predicted_valid = None
        if gate_scores is None or teacher_scores is None:
            return None
        if gate_scores.shape != teacher_scores.shape:
            return None
        keep = cfg.resolve_keep(teacher_scores.shape[1])
        with torch.no_grad():
            target = torch.zeros_like(teacher_scores)
            target.scatter_(1, teacher_scores.topk(keep, dim=1).indices, 1.0)
        loss_per = F.binary_cross_entropy_with_logits(
            gate_scores.float(), target.to(dtype=torch.float32), reduction="none"
        ).mean(dim=1)  # [n_images]
        if valid is not None:
            v = valid.to(device=loss_per.device, dtype=torch.bool)
            if not bool(v.any()):
                return None
            loss = loss_per[v].mean()
        else:
            loss = loss_per.mean()
        _dump = os.environ.get("ROI_DUMP_AGREEMENT")
        if _dump:
            with torch.no_grad():
                gk = gate_scores.topk(keep, dim=1).indices
                prec = target.gather(1, gk).sum(dim=1).div(float(keep))
                if valid is not None:
                    vv = valid.to(device=prec.device, dtype=torch.bool)
                    prec = prec[vv] if bool(vv.any()) else prec
                rec = {
                    "mode": "gate_predict",
                    "horizon": int(getattr(self.config, "roi_predict_horizon", 0)),
                    "keep_frac": round(float(cfg.keep_frac), 3),
                    "n_patches": int(teacher_scores.shape[1]),
                    "keep_k": int(keep),
                    "bce": float(loss.item()),
                    "precision_at_k": float(prec.mean().item()),
                    "random_baseline": float(keep) / float(teacher_scores.shape[1]),
                    "n_valid": int(valid.to(dtype=torch.bool).sum().item()) if valid is not None else int(gate_scores.shape[0]),
                    "batch_images": int(gate_scores.shape[0]),
                    "gate_scores_requires_grad": bool(gate_scores.requires_grad),
                }
            try:
                with open(_dump, "a") as _fh:
                    _fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass
        return float(getattr(cfg, "distill_weight", 1.0)) * loss

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Generate an action chunk via continuous flow matching or discrete AR decoding."""
        if "action_mode" in kwargs:
            raise TypeError(
                "MolmoAct2 predict_action_chunk got unexpected keyword argument 'action_mode'; "
                "use 'inference_action_mode'."
            )
        model_inputs = self._model_inputs(batch)
        # Give the ROI gate its instruction conditioning so the eval keep-set matches
        # training (no-op for a task-free gate). Applies to every inference mode.
        self._set_gate_task_tokens(model_inputs)
        # Causal predictive gating (select='gate_predict'): prune THIS replan's ViT with
        # the keep-set the gate predicted at the PREVIOUS replan (H=n_action_steps frames
        # earlier, i.e. this observation's past). The gate never sees the frame it prunes.
        # The first replan of an episode has no carried set -> the gate falls back to the
        # current frame for that single step (reset() clears the carry between episodes).
        _causal_vb = None
        _causal_roi_cfg = None
        if int(getattr(self.config, "roi_predict_horizon", 0)) > 0:
            _causal_vb = getattr(self._backbone(), "vision_backbone", None)
            _causal_roi_cfg = getattr(_causal_vb, "roi_cfg", None) if _causal_vb is not None else None
            if _causal_vb is not None and getattr(_causal_roi_cfg, "select", None) == "gate_predict":
                _causal_vb.roi_external_keep_idx = getattr(self, "_roi_pred_prev_keep_idx", None)
                _causal_vb.roi_last_gate_scores = None
            else:
                _causal_vb = None
        inference_action_mode = self._resolve_inference_action_mode(kwargs.get("inference_action_mode"))
        num_steps = kwargs.get("num_steps", getattr(self.config, "num_inference_steps", None))
        generator = kwargs.get("generator")
        model_dtype = _torch_dtype(self.config.model_dtype)
        device = next(self.parameters()).device
        batch_size = int(next(iter(model_inputs.values())).shape[0])
        if generator is None:
            generator = self._rollout_generator_for_inputs(
                batch,
                batch_size=batch_size,
                device=device,
            )
        action_dim = self._output_action_dim(batch)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=model_dtype)
            if device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            if inference_action_mode == "discrete":
                if self._rtc_enabled():
                    raise ValueError("RTC is only supported for continuous MolmoAct2 inference.")
                actions = self._generate_discrete_actions_from_inputs(
                    model_inputs=model_inputs,
                    action_dim=action_dim,
                )
            elif self._rtc_enabled():
                actions = self._generate_actions_from_inputs_with_rtc(
                    model_inputs=model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    num_steps=num_steps,
                    generator=generator,
                    inference_delay=kwargs.get("inference_delay"),
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            elif self._fastv_infer_enabled():
                actions = self._generate_actions_fastv(
                    model_inputs=model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    num_steps=num_steps,
                    generator=generator,
                )
            else:
                actions = self._backbone().generate_actions_from_inputs(
                    **model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    action_horizon=self._generation_action_horizon(),
                    num_steps=num_steps,
                    generator=generator,
                )
        # Carry THIS replan's predicted gate scores forward: their top-K becomes the
        # keep-set that prunes the NEXT replan's frame (t+H), keeping inference causal.
        if _causal_vb is not None:
            _scores = getattr(_causal_vb, "roi_last_gate_scores", None)
            if _scores is not None and torch.is_tensor(_scores):
                _keep = _causal_roi_cfg.resolve_keep(int(_scores.shape[1]))
                self._roi_pred_prev_keep_idx = (
                    _scores.detach().topk(_keep, dim=1).indices.sort(dim=1).values
                )
            _causal_vb.roi_external_keep_idx = None
        return actions[:, : self.config.n_action_steps, :action_dim].to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Pop one action step from the queue, regenerating the chunk when empty."""
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action, use it with predict_action_chunk")
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(prefix=r"model\.model")
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
        }

    def _get_inner_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(prefix="model")
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
        }

    def _lora_target_modules(self, *, prefix: str) -> str:
        vlm_linear_leaves = "w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding"
        target_modules = rf"{prefix}\.(transformer|vision_backbone)\.(?:.*\.)?({vlm_linear_leaves})$"
        if self.config.enable_lora_action_expert:
            action_expert_linear_paths = (
                r"time_embed\.(1|3)|"
                r"action_embed|context_k_proj|context_v_proj|"
                r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
                r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
                r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
                r"blocks\.\d+\.modulation\.linear|"
                r"final_layer\.(modulation\.linear|linear)"
            )
            target_modules = (
                f"({target_modules}|"
                rf"{prefix}\.action_expert\.({action_expert_linear_paths})$)"
            )
        return target_modules

    def _build_inner_lora_config(self):
        require_package("peft", extra="molmoact2")
        from peft import LoraConfig

        return LoraConfig(**self._get_inner_peft_targets())

    def _apply_lora_adapters(self) -> None:
        require_package("peft", extra="molmoact2")
        from peft import get_peft_model

        peft_config = self._build_inner_lora_config()
        self._validate_peft_config(peft_config)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model = get_peft_model(self.model, peft_config)
        if not self.config.enable_lora_action_expert:
            self._unfreeze_action_expert_parameters()
        self.train(self.training)

    def _validate_peft_config(self, peft_config) -> None:
        del peft_config
        if not self.config.checkpoint_path:
            raise ValueError("MolmoAct2 LoRA fine-tuning requires `policy.checkpoint_path`.")
