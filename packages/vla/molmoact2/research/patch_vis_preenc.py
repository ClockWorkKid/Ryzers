# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Insert an env-gated PRE-ENCODER (stage-D) vision-token pruner into MolmoAct2.

Companion to `patch_vis_reduce.py` (stage B: drops POOLED tokens *after* the ViT,
so the ViT still runs full). This patch drops spatial patches *before* the ViT
transformer blocks, so the SigLIP2 encoder itself runs on fewer tokens -- the
~234 ms `vision_ms` that stage B never touches.

When `VIS_PREENC_KEEP_FRAC` < 1.0, `MolmoAct2VisionBackbone.forward` keeps only
`round(keep_frac * pool_size)` patches (>=1) per pooling group, chosen randomly
among the group's valid patches, and encodes ONLY those through the ViT. Kept
patches carry their correct positional embedding (gathered by original grid
position -- the ViT blocks are plain, permutation-equivariant self-attention, so
pos-emb is the only spatial signal). Dropped patches are set to -1 in
`pooled_patches_idx` so the pooling step ignores them. The pooled-token count is
preserved (>=1 patch per group), so `input_ids` / prefill are unchanged -- this is
purely an encoder-FLOP cut and composes with stage B (`VIS_KEEP_FRAC`) for the
LLM-side cut.

Idempotent + reversible. Usage:
    python patch_vis_preenc.py <hf_cache_dir>          # apply
    python patch_vis_preenc.py <hf_cache_dir> --revert # remove
"""
import glob
import os
import sys

# --- Edit 1: new ViT subset-encode helpers (before VisionTransformer.forward) ---
ANCHOR1 = (
    "    def forward(self, x: torch.Tensor, patch_num: int = None) -> list[torch.Tensor]:\n"
)
BLOCK1 = r'''    # === VIS_PREENC_VIT START ===
    def get_pos_emb_grid(self, patch_num=None) -> torch.Tensor:
        """Return the (N, hidden) positional-embedding grid, interpolated to
        `patch_num` if needed -- same math as `add_pos_emb`, but returned (not added)
        so a pruned subset can gather only the rows it keeps."""
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
        """Encode a gathered subset of patches.

        :param x: (1, K, n_pixels) already-gathered kept patches for one crop
        :param keep_idx: (K,) original grid positions of those patches in [0, N)
        """
        x = self.patch_embedding(x)
        pos_emb = self.get_pos_emb_grid(patch_num)
        x = x + pos_emb[keep_idx][None, :, :].to(x.dtype)
        return self.transformer(x)
    # === VIS_PREENC_VIT END ===

'''
EDIT1 = (ANCHOR1, BLOCK1 + ANCHOR1)

# --- Edit 2: new backbone subset-encode method (after encode_image) ----------
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
    "    # === VIS_PREENC_BACKBONE START ===\n"
    r'''    def _encode_image_preenc(self, images, pooled_patches_idx, keep_frac):
        """Stage-D pre-encoder pruning. Keep round(keep_frac*pool) patches per
        pooling group (>=1), encode ONLY kept patches through the ViT with correct
        positional embeddings, scatter back to the full patch layout, and -1 the
        dropped patches in pooled_patches_idx so pooling ignores them. Returns the
        (B, T, N, out_dim) feature tensor and the modified pooled_patches_idx."""
        import os as _os

        B, T, N, D = images.shape
        device = images.device
        bt = B * T
        flat = T * N
        pool = pooled_patches_idx.shape[-1]
        k = max(1, int(round(keep_frac * pool)))

        ppi = pooled_patches_idx.clone()
        valid = ppi >= 0
        scores = torch.rand(ppi.shape, device=device)
        scores = torch.where(valid, scores, torch.full_like(scores, float("inf")))
        order = torch.argsort(scores, dim=-1)
        rank = torch.argsort(order, dim=-1)
        keep_slot = (rank < k) & valid
        ppi = torch.where(keep_slot, ppi, torch.full_like(ppi, -1))

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

        if _os.environ.get("VIS_PREENC_DEBUG"):
            import sys as _sys
            print(
                "[VIS_PREENC] keep_frac=%.2f k=%d/%d kept/crop~%d/%d"
                % (keep_frac, k, pool, int(keep_mask[0].sum()), N),
                file=_sys.stderr,
                flush=True,
            )
        return image_features, ppi
    # === VIS_PREENC_BACKBONE END ===

'''
    "    @property\n"
    "    def dtype(self) -> torch.dtype:\n"
)
EDIT2 = (ANCHOR2, BLOCK2)

# --- Edit 3: env-gated branch at the encode_image call site -------------------
ANCHOR3 = "        image_features = self.encode_image(images)\n"
BLOCK3 = (
    "        # === VIS_PREENC_CALL START ===\n"
    "        import os as _os\n"
    "        _pkf = float(_os.environ.get(\"VIS_PREENC_KEEP_FRAC\", \"1.0\"))\n"
    "        if _pkf < 1.0:\n"
    "            image_features, pooled_patches_idx = self._encode_image_preenc(\n"
    "                images, pooled_patches_idx, _pkf\n"
    "            )\n"
    "        else:\n"
    "            image_features = self.encode_image(images)\n"
    "        # === VIS_PREENC_CALL END ===\n"
)
EDIT3 = (ANCHOR3, BLOCK3)

EDITS = [EDIT1, EDIT2, EDIT3]


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
        print("usage: patch_vis_preenc.py <hf_cache_dir> [--revert]")
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
        print(f"{status:>28}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
