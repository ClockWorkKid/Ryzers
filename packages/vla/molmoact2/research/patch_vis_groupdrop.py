# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Route-A "stage-D-groups" pruner: drop WHOLE 2x2 pooling groups before the ViT.

Unlike stage-D (`patch_vis_preenc.py`, drops patches *within* a group, keeps token
count) and stage-B (`patch_vis_reduce.py`, drops pooled tokens *after* the ViT),
this drops entire pooling groups *before* the encoder. Effect: BOTH the ViT FLOPs
AND the LLM prefill shrink ~linearly with `VIS_GROUPDROP_KEEP_FRAC`, while every
surviving group stays a full 2x2 group -> the learned attention-pooler stays
in-distribution.

Mechanism (env-gated by VIS_GROUPDROP_KEEP_FRAC < 1.0):
  * Backbone keeps round(keep*Ngroups) whole groups (>=1), encodes ONLY their
    patches through the ViT (correct per-position pos-emb via `forward_subset`),
    -1s the dropped groups in pooled_patches_idx (pooler emits only kept tokens),
    and stashes `_groupdrop_keep` (bool over original valid pooled tokens).
  * build_input_embeddings scatters the (now fewer) vision tokens into ONLY the
    kept <image> placeholders (no assert blow-up).
  * Model.forward drops the dropped image columns from the prefill sequence and
    sets `self._vis_keep_cols`, so the stage-B `VIS_REDUCE_GEN` action-path block
    subsamples the encoder-attention mask to match. (Requires patch_vis_reduce.py
    applied for that action-path block; revert patch_vis_preenc.py first to free
    the encode_image anchor.)

Idempotent + reversible. Usage:
    python patch_vis_groupdrop.py <hf_cache_dir>          # apply
    python patch_vis_groupdrop.py <hf_cache_dir> --revert # remove
"""
import glob
import os
import sys

# --- Edit 1: ViT subset-encode helpers (before VisionTransformer.forward) -----
ANCHOR1 = (
    "    def forward(self, x: torch.Tensor, patch_num: int = None) -> list[torch.Tensor]:\n"
)
BLOCK1 = r'''    # === VIS_GROUPDROP_VIT START ===
    def get_pos_emb_grid(self, patch_num=None) -> torch.Tensor:
        """Return the (N, hidden) positional-embedding grid (interpolated to
        `patch_num` if needed) so a pruned subset can gather only its rows."""
        if patch_num is None:
            patch_num = self.config.image_num_patch
        pos_emb = self.positional_embedding
        pos_emb = pos_emb.reshape(
            (
                int(math.sqrt(pos_emb.shape[0])),
                int(math.sqrt(pos_emb.shape[0])),
                pos_emb.shape[1],
            )
        )
        (patch_num_0, patch_num_1) = patch_num
        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)
        return pos_emb.reshape(-1, pos_emb.shape[-1])

    def forward_subset(
        self, x: torch.Tensor, keep_idx: torch.Tensor, patch_num: int = None
    ) -> list[torch.Tensor]:
        """Encode a gathered subset of patches (one crop).
        :param x: (1, K, n_pixels) already-gathered kept patches
        :param keep_idx: (K,) original grid positions in [0, N)"""
        x = self.patch_embedding(x)
        pos_emb = self.get_pos_emb_grid(patch_num)
        x = x + pos_emb[keep_idx][None, :, :].to(x.dtype)
        return self.transformer(x)
    # === VIS_GROUPDROP_VIT END ===

'''
EDIT1 = (ANCHOR1, BLOCK1 + ANCHOR1)

# --- Edit 2: backbone group-drop encode method (after encode_image) -----------
ANCHOR2 = (
    "        image_features = image_features.view(B, T, N, -1)\n"
    "        return image_features\n"
    "\n"
    "    @property\n"
    "    def dtype(self) -> torch.dtype:\n"
)
BLOCK2 = (
    "        image_features = image_features.view(B, T, N, -1)\n"
    "        return image_features\n"
    "\n"
    "    # === VIS_GROUPDROP_BACKBONE START ===\n"
    r'''    def _encode_image_groupdrop(self, images, pooled_patches_idx, keep_frac):
        """Keep round(keep_frac*Ngroups) whole pooling groups (>=1), encode ONLY
        their patches through the ViT, -1 the dropped groups in pooled_patches_idx,
        and return (features, modified_ppi, keep_over) where keep_over is a bool
        over the ORIGINAL valid pooled tokens (in flattened batch-major order)."""
        import os as _os

        B, T, N, D = images.shape
        device = images.device
        bt = B * T
        flat = T * N
        P = pooled_patches_idx.shape[1]

        ppi = pooled_patches_idx.clone()
        valid_token = (ppi >= 0).any(-1)  # (B, P): groups that are real
        scores = torch.rand(B, P, device=device)
        scores = torch.where(valid_token, scores, torch.full_like(scores, float("inf")))
        order = torch.argsort(scores, dim=-1)
        rank = torch.argsort(order, dim=-1)
        nvalid = valid_token.sum(-1, keepdim=True)
        k = torch.clamp((keep_frac * nvalid.float()).round().long(), min=1)
        keep_group = (rank < k) & valid_token  # (B, P)
        keep_over = keep_group.flatten()[valid_token.flatten()]

        ppi = torch.where(keep_group[:, :, None], ppi, torch.full_like(ppi, -1))

        keep_slot = ppi >= 0
        keep_mask = torch.zeros(B, flat, dtype=torch.bool, device=device)
        kept_idx = ppi[keep_slot]
        batch_of = torch.arange(B, device=device).view(B, 1, 1).expand_as(ppi)[keep_slot]
        keep_mask[batch_of, kept_idx] = True
        keep_mask = keep_mask.view(bt, N)

        images = images.view(bt, N, D)
        out_dim = self.vit_config.hidden_size * len(self.vit_layers)
        image_features = images.new_zeros(bt, N, out_dim)
        for c in range(bt):
            idx = keep_mask[c].nonzero().flatten()
            if idx.numel() == 0:
                continue
            xc = images[c : c + 1, idx, :]
            hs = self.image_vit.forward_subset(xc, idx)
            feats = torch.cat([hs[layer] for layer in self.vit_layers], dim=-1)
            image_features[c, idx] = feats[0].to(image_features.dtype)
        image_features = image_features.view(B, T, N, out_dim)

        if _os.environ.get("VIS_GROUPDROP_DEBUG"):
            import sys as _sys
            print(
                "[VIS_GROUPDROP] keep=%.2f groups %d->%d patches/crop~%d/%d"
                % (keep_frac, int(valid_token.sum()), int(keep_group.sum()),
                   int(keep_mask[0].sum()), N),
                file=_sys.stderr, flush=True,
            )
        return image_features, ppi, keep_over
    # === VIS_GROUPDROP_BACKBONE END ===

'''
    "    @property\n"
    "    def dtype(self) -> torch.dtype:\n"
)
EDIT2 = (ANCHOR2, BLOCK2)

# --- Edit 3: env-gated branch at the encode_image call site -------------------
ANCHOR3 = "        image_features = self.encode_image(images)\n"
BLOCK3 = (
    "        # === VIS_GROUPDROP_CALL START ===\n"
    "        import os as _os\n"
    "        _gkf = float(_os.environ.get(\"VIS_GROUPDROP_KEEP_FRAC\", \"1.0\"))\n"
    "        if _gkf < 1.0:\n"
    "            image_features, pooled_patches_idx, self._groupdrop_keep = self._encode_image_groupdrop(\n"
    "                images, pooled_patches_idx, _gkf\n"
    "            )\n"
    "        else:\n"
    "            self._groupdrop_keep = None\n"
    "            image_features = self.encode_image(images)\n"
    "        # === VIS_GROUPDROP_CALL END ===\n"
)
EDIT3 = (ANCHOR3, BLOCK3)

# --- Edit 4: group-drop-aware scatter in build_input_embeddings ---------------
ANCHOR4 = (
    "            image_features = self.vision_backbone(images, token_pooling).to(x.device)\n"
    "            is_image_patch = input_ids.view(-1) == self.config.image_patch_id\n"
    "            assert is_image_patch.sum() == len(image_features)\n"
    "            x.view(-1, x.shape[-1])[is_image_patch] += image_features\n"
)
BLOCK4 = (
    "            image_features = self.vision_backbone(images, token_pooling).to(x.device)\n"
    "            is_image_patch = input_ids.view(-1) == self.config.image_patch_id\n"
    "            # === VIS_GROUPDROP_SCATTER START ===\n"
    "            _gd_keep = getattr(self.vision_backbone, \"_groupdrop_keep\", None)\n"
    "            if _gd_keep is not None and int(image_features.shape[0]) != int(is_image_patch.sum()):\n"
    "                _gpos = is_image_patch.nonzero().flatten()\n"
    "                _gkept = _gpos[_gd_keep.to(_gpos.device)]\n"
    "                x.view(-1, x.shape[-1])[_gkept] += image_features\n"
    "            else:\n"
    "                assert is_image_patch.sum() == len(image_features)\n"
    "                x.view(-1, x.shape[-1])[is_image_patch] += image_features\n"
    "            # === VIS_GROUPDROP_SCATTER END ===\n"
)
EDIT4 = (ANCHOR4, BLOCK4)

# --- Edit 5: shorten prefill sequence in Model.forward (after stage-B block) --
ANCHOR5 = "            # === VIS_REDUCE_PATCH END ===\n"
BLOCK5 = (
    "            # === VIS_REDUCE_PATCH END ===\n"
    "            # === VIS_GROUPDROP_SEQ START ===\n"
    "            import os as _os\n"
    "            _gd = getattr(self.vision_backbone, \"_groupdrop_keep\", None)\n"
    "            if _gd is not None and input_ids is not None and inputs_embeds is not None:\n"
    "                _ipid2 = int(self.config.image_patch_id)\n"
    "                _S2 = int(input_ids.shape[1])\n"
    "                _imgpos2 = (input_ids[0] == _ipid2).nonzero().flatten()\n"
    "                if int(_imgpos2.numel()) > 0:\n"
    "                    _keepimg2 = _imgpos2[_gd.to(_imgpos2.device)]\n"
    "                    _keep2 = torch.ones(_S2, dtype=torch.bool, device=input_ids.device)\n"
    "                    _keep2[_imgpos2] = False\n"
    "                    _keep2[_keepimg2] = True\n"
    "                    _cols2 = _keep2.nonzero().flatten()\n"
    "                    inputs_embeds = inputs_embeds[:, _cols2, :]\n"
    "                    input_ids = input_ids[:, _cols2]\n"
    "                    if token_type_ids is not None:\n"
    "                        token_type_ids = token_type_ids[:, _cols2]\n"
    "                    if attention_mask is not None and hasattr(attention_mask, \"ndim\") and attention_mask.ndim == 2:\n"
    "                        attention_mask = attention_mask[:, _cols2]\n"
    "                    position_ids = None\n"
    "                    cache_position = None\n"
    "                    self._vis_keep_cols = _cols2\n"
    "                    if _os.environ.get(\"VIS_GROUPDROP_DEBUG\"):\n"
    "                        import sys as _sys\n"
    "                        print(\"[VIS_GROUPDROP] seq %d->%d (kept %d/%d img tok)\" % (_S2, int(_cols2.numel()), int(_keepimg2.numel()), int(_imgpos2.numel())), file=_sys.stderr, flush=True)\n"
    "            # === VIS_GROUPDROP_SEQ END ===\n"
)
EDIT5 = (ANCHOR5, BLOCK5)

EDITS = [EDIT1, EDIT2, EDIT3, EDIT4, EDIT5]


def apply(text):
    statuses = []
    for find, repl in EDITS:
        if repl in text:
            statuses.append("already")
        elif find in text:
            text = text.replace(find, repl, 1)
            statuses.append("ok")
        else:
            statuses.append("no-anchor")
    return text, ",".join(statuses)


def revert(text):
    statuses = []
    for find, repl in EDITS:
        if repl in text:
            text = text.replace(repl, find, 1)
            statuses.append("reverted")
        else:
            statuses.append("clean")
    return text, ",".join(statuses)


def main():
    if len(sys.argv) < 2:
        print("usage: patch_vis_groupdrop.py <hf_cache_dir> [--revert]")
        return 2
    root = sys.argv[1]
    do_revert = "--revert" in sys.argv[2:]
    files = glob.glob(os.path.join(root, "**", "modeling_molmoact2.py"), recursive=True)
    if not files:
        print("no modeling_molmoact2.py found under", root)
        return 1
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read()
        new, status = (revert(text) if do_revert else apply(text))
        if new != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
        print(f"{status:>40}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
