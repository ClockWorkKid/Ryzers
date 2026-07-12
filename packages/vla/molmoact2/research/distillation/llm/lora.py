"""Self-contained LoRA for the Run D LoRA-adaptation stage.

We inject low-rank adapters directly into the custom student submodules (thin-twin
LLM + distilled ViT) rather than using PEFT, because those modules are attached by
monkey-patch and PEFT's name-mangling would break lerobot's save/resume of the
composed policy. A LoRALinear wraps a frozen base nn.Linear with a trainable
``B @ A`` low-rank update; only the adapter params (names containing ``lora_``)
train. Adapters register as normal submodules so lerobot checkpoints them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> adapter is identity at init (no perturbation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lx = self.lora_dropout(x)
        update = (lx @ self.lora_A.t().to(lx.dtype)) @ self.lora_B.t().to(lx.dtype)
        return out + self.scaling * update


def inject_lora(root: nn.Module, target_suffixes: tuple[str, ...], r: int = 16,
                alpha: int = 32, dropout: float = 0.05) -> int:
    """Replace every nn.Linear child whose attribute name is in target_suffixes
    with a LoRALinear. Returns the number of layers wrapped."""
    n = 0
    for module in root.modules():
        for attr, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr in target_suffixes:
                setattr(module, attr, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                n += 1
    return n


def mark_lora_only_trainable(root: nn.Module) -> tuple[int, int]:
    """Freeze everything, then unfreeze only LoRA adapter params. Returns
    (num_trainable_params, num_lora_tensors)."""
    n_params = 0
    n_tensors = 0
    for name, p in root.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n_params += p.numel()
            n_tensors += 1
        else:
            p.requires_grad_(False)
    return n_params, n_tensors
