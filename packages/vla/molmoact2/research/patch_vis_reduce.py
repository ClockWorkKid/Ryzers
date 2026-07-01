# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Insert an env-gated random vision-token reducer into MolmoAct2 modeling files.

When `VIS_KEEP_FRAC` < 1.0, MolmoAct2Model.forward keeps only that fraction of
the image-patch columns (chosen randomly) right after `build_input_embeddings`,
dropping the rest from inputs_embeds / input_ids / attention_mask /
token_type_ids together, so the LLM prefill sequence truly shortens. The ViT
(and its positional embeddings) still run fully; we only subsample the pooled
tokens that flow into the backbone.

The kept column indices are stashed on the model so the action-expert's
cross-attention mask (`_get_encoder_attention_mask`, built from the full
input_ids) is subsampled to match the reduced prefill KV.

Idempotent + reversible. Usage:
    python patch_vis_reduce.py <hf_cache_dir>          # apply
    python patch_vis_reduce.py <hf_cache_dir> --revert # remove
"""
import glob
import os
import sys

START = "            # === VIS_REDUCE_PATCH START ==="
END = "            # === VIS_REDUCE_PATCH END ==="

# --- Edit 1: reduction block after build_input_embeddings -------------------
ANCHOR1 = (
    "            inputs_embeds, image_features = self.build_input_embeddings(\n"
    "                input_ids,\n"
    "                images,\n"
    "                token_pooling,\n"
    "            )\n"
)
BLOCK = START + "\n" + r"""            import os as _os
            _kf = float(_os.environ.get("VIS_KEEP_FRAC", "1.0"))
            self._vis_keep_cols = None
            if _kf < 1.0 and images is not None and input_ids is not None and inputs_embeds is not None:
                _ipid = int(self.config.image_patch_id)
                _S = int(input_ids.shape[1])
                _imgpos = (input_ids[0] == _ipid).nonzero().flatten()
                _nimg = int(_imgpos.numel())
                if _nimg > 0:
                    _k = max(1, int(round(_kf * _nimg)))
                    _perm = torch.randperm(_nimg, device=_imgpos.device)[:_k]
                    _keepimg = _imgpos[_perm]
                    _keep = torch.ones(_S, dtype=torch.bool, device=input_ids.device)
                    _keep[_imgpos] = False
                    _keep[_keepimg] = True
                    _cols = _keep.nonzero().flatten()
                    inputs_embeds = inputs_embeds[:, _cols, :]
                    input_ids = input_ids[:, _cols]
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids[:, _cols]
                    if attention_mask is not None and hasattr(attention_mask, "ndim") and attention_mask.ndim == 2:
                        attention_mask = attention_mask[:, _cols]
                    position_ids = None
                    cache_position = None
                    self._vis_keep_cols = _cols
                    if _os.environ.get("VIS_REDUCE_DEBUG"):
                        import sys as _sys
                        print("[VIS_REDUCE] kept %d/%d image tokens, seq %d->%d" % (_k, _nimg, _S, int(_cols.numel())), file=_sys.stderr, flush=True)
""" + END + "\n"
EDIT1 = (ANCHOR1, ANCHOR1 + BLOCK)

# --- Edit 2: subsample input_ids/masks after prefill in the action path -----
# After the LLM prefill (which stashed _vis_keep_cols and shortened the KV),
# input_ids/attention_mask/token_type_ids are still full-length. Subsample them
# to the kept columns so the encoder-attention mask and the depth gate (built
# from input_ids) line up with the reduced prefill KV.
ANCHOR2 = (
    "            encoder_kv_states = self._extract_kv_states(outputs.past_key_values)\n"
    "            encoder_attention_mask = self._get_encoder_attention_mask(\n"
    "                input_ids, attention_mask\n"
    "            )\n"
)
INJECT2 = (
    "            encoder_kv_states = self._extract_kv_states(outputs.past_key_values)\n"
    "            # === VIS_REDUCE_GEN START ===\n"
    "            _c = getattr(self, \"_vis_keep_cols\", None)\n"
    "            if _c is not None and input_ids is not None and input_ids.shape[1] != int(_c.numel()):\n"
    "                input_ids = input_ids[:, _c]\n"
    "                if attention_mask is not None and hasattr(attention_mask, \"ndim\") and attention_mask.ndim == 2:\n"
    "                    attention_mask = attention_mask[:, _c]\n"
    "                if token_type_ids is not None:\n"
    "                    token_type_ids = token_type_ids[:, _c]\n"
    "            # === VIS_REDUCE_GEN END ===\n"
    "            encoder_attention_mask = self._get_encoder_attention_mask(\n"
    "                input_ids, attention_mask\n"
    "            )\n"
)
EDIT2 = (ANCHOR2, INJECT2)

EDITS = [EDIT1, EDIT2]


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
        print("usage: patch_vis_reduce.py <hf_cache_dir> [--revert]")
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
