# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Attention-feedback DETERMINISTIC group-drop pruner for MolmoAct2 (research).

Builds ON TOP of `patch_vis_groupdrop.py` (apply that first). Where group-drop
chooses which 2x2 pooling groups to keep at *random* (`torch.rand`), this patch
chooses them from a per-group saliency map harvested from the LLM self-attention
of a *previous* observation -- "tunnel vision": survey the whole scene once, then
focus compute on the task-relevant groups on subsequent frames.

Mechanism (env-gated by VIS_ATTNFB=1, keep fraction reuses VIS_GROUPDROP_KEEP_FRAC):
  * Survey frame (no saliency yet): forces the LLM attention to eager + turns on
    output_attentions, runs the FULL encoder + LLM, then `_attn_feedback_gather`
    turns the self-attention received by each <image> token into a debiased
    per-group saliency vector stashed on vision_backbone._attn_feedback_saliency_out.
  * Pruned frame (saliency present): the group-drop scorer uses (-saliency) as its
    keep score (group-drop keeps the smallest score -> we keep highest saliency).
    No eager / no output_attentions -> normal fast SDPA path.
  * Carry: by default (VIS_ATTNFB_SELFCARRY!=0) the model copies _out -> _in at the
    top of each generate_actions_from_inputs call so direct predict_action loops
    (host_server / bench / validate) get cross-frame behaviour for free. A
    multi-env policy wrapper can set VIS_ATTNFB_SELFCARRY=0 and manage
    `_attn_feedback_saliency_{in,out}` per env (see patch_lerobot_attnfb.py).

Env knobs:
  VIS_ATTNFB=1                  enable attention-feedback scoring + harvest
  VIS_GROUPDROP_KEEP_FRAC=0.5   keep fraction on pruned frames (reused)
  VIS_ATTNFB_SELFCARRY=0        disable model self-carry (wrapper-managed)
  VIS_ATTNFB_LAYER_LO/HI        LLM layer band to average (default: last 8)
  VIS_ATTNFB_DEBIAS=clip|zmedian|none   sink debiasing (default clip @ p95)
  VIS_ATTNFB_DEBUG=1            stderr diagnostics

Idempotent + reversible. Usage:
    python patch_vis_attnfeedback.py <hf_cache_dir>          # apply
    python patch_vis_attnfeedback.py <hf_cache_dir> --revert # remove
"""
import glob
import os
import sys

# --- Edit 1: deterministic score source inside _encode_image_groupdrop --------
# Group-drop samples `scores = torch.rand(...)`; we override with (-saliency) when
# attention feedback is available so the highest-saliency groups are kept.
ANCHOR1 = (
    "        scores = torch.rand(B, P, device=device)\n"
    "        scores = torch.where(valid_token, scores, torch.full_like(scores, float(\"inf\")))\n"
)
BLOCK1 = (
    "        scores = torch.rand(B, P, device=device)\n"
    "        # === VIS_ATTNFB_SCORE START ===\n"
    "        import os as _os_afb\n"
    "        if _os_afb.environ.get(\"VIS_ATTNFB\"):\n"
    "            _afb_sal = getattr(self, \"_attn_feedback_saliency_in\", None)\n"
    "            if _afb_sal is not None:\n"
    "                _afb_s = _afb_sal.to(device=device, dtype=torch.float32).reshape(-1)\n"
    "                _afb_nv = int(valid_token.sum().item())\n"
    "                if int(_afb_s.numel()) == _afb_nv and _afb_nv > 0:\n"
    "                    _afb_full = torch.zeros(B, P, device=device, dtype=torch.float32)\n"
    "                    _afb_full[valid_token] = (-_afb_s)\n"
    "                    scores = _afb_full\n"
    "                    if _os_afb.environ.get(\"VIS_ATTNFB_DEBUG\"):\n"
    "                        import sys as _sys_afb\n"
    "                        print(\"[VIS_ATTNFB] deterministic scores from saliency nv=%d\" % _afb_nv, file=_sys_afb.stderr, flush=True)\n"
    "        # === VIS_ATTNFB_SCORE END ===\n"
    "        scores = torch.where(valid_token, scores, torch.full_like(scores, float(\"inf\")))\n"
)
EDIT1 = (ANCHOR1, BLOCK1)

# --- Edit 2: survey-frame gate at the group-drop call site --------------------
ANCHOR2 = (
    "        _gkf = float(_os.environ.get(\"VIS_GROUPDROP_KEEP_FRAC\", \"1.0\"))\n"
    "        if _gkf < 1.0:\n"
)
BLOCK2 = (
    "        _gkf = float(_os.environ.get(\"VIS_GROUPDROP_KEEP_FRAC\", \"1.0\"))\n"
    "        _afb_survey = bool(_os.environ.get(\"VIS_ATTNFB\")) and getattr(self, \"_attn_feedback_saliency_in\", None) is None\n"
    "        if _gkf < 1.0 and not _afb_survey:\n"
)
EDIT2 = (ANCHOR2, BLOCK2)

# --- Edit 3: harvest hook around the continuous-prefill self(...) call --------
ANCHOR3 = (
    "            outputs = self(\n"
    "                input_ids=input_ids,\n"
    "                pixel_values=pixel_values,\n"
    "                image_token_pooling=image_token_pooling,\n"
    "                image_grids=image_grids,\n"
    "                image_num_crops=image_num_crops,\n"
    "                pixel_values_videos=pixel_values_videos,\n"
    "                video_token_pooling=video_token_pooling,\n"
    "                video_grids=video_grids,\n"
    "                attention_mask=attention_mask,\n"
    "                token_type_ids=token_type_ids,\n"
    "                use_cache=True,\n"
    "            )\n"
    "            encoder_kv_states = self._extract_kv_states(outputs.past_key_values)\n"
)
BLOCK3 = (
    "            # === VIS_ATTNFB_CARRY START ===\n"
    "            import os as _os_afb\n"
    "            _afb_on = bool(_os_afb.environ.get(\"VIS_ATTNFB\"))\n"
    "            _afb_vb = getattr(self, \"vision_backbone\", None)\n"
    "            if _afb_on and _afb_vb is not None and _os_afb.environ.get(\"VIS_ATTNFB_SELFCARRY\", \"1\") != \"0\":\n"
    "                _afb_vb._attn_feedback_saliency_in = getattr(_afb_vb, \"_attn_feedback_saliency_out\", None)\n"
    "            _afb_survey = _afb_on and (_afb_vb is not None) and (getattr(_afb_vb, \"_attn_feedback_saliency_in\", None) is None)\n"
    "            _afb_prev_impl = self._afb_set_attn_impl(\"eager\") if _afb_survey else None\n"
    "            # === VIS_ATTNFB_CARRY END ===\n"
    "            outputs = self(\n"
    "                input_ids=input_ids,\n"
    "                pixel_values=pixel_values,\n"
    "                image_token_pooling=image_token_pooling,\n"
    "                image_grids=image_grids,\n"
    "                image_num_crops=image_num_crops,\n"
    "                pixel_values_videos=pixel_values_videos,\n"
    "                video_token_pooling=video_token_pooling,\n"
    "                video_grids=video_grids,\n"
    "                attention_mask=attention_mask,\n"
    "                token_type_ids=token_type_ids,\n"
    "                use_cache=True,\n"
    "                output_attentions=_afb_survey,\n"
    "            )\n"
    "            encoder_kv_states = self._extract_kv_states(outputs.past_key_values)\n"
    "            # === VIS_ATTNFB_GATHER START ===\n"
    "            if _afb_prev_impl is not None:\n"
    "                self._afb_restore_attn_impl(_afb_prev_impl)\n"
    "            if _afb_survey:\n"
    "                self._attn_feedback_gather(outputs, input_ids)\n"
    "            # === VIS_ATTNFB_GATHER END ===\n"
)
EDIT3 = (ANCHOR3, BLOCK3)

# --- Edit 4: harvest + eager-toggle helper methods ----------------------------
ANCHOR4 = "    def generate_actions_from_inputs(\n"
BLOCK4 = r'''    # === VIS_ATTNFB_METHOD START ===
    def _afb_set_attn_impl(self, impl):
        """Force `impl` (e.g. "eager") on every distinct config that carries
        `_attn_implementation`, returning a restore token. Needed because the SDPA
        path returns attn_weights=None -- only eager surfaces attentions."""
        prev = []
        seen = set()
        for mod in self.modules():
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation") and id(cfg) not in seen:
                seen.add(id(cfg))
                prev.append((cfg, cfg._attn_implementation))
                try:
                    cfg._attn_implementation = impl
                except Exception:
                    pass
        return prev

    def _afb_restore_attn_impl(self, prev):
        for cfg, old in prev:
            try:
                cfg._attn_implementation = old
            except Exception:
                pass

    def _attn_feedback_gather(self, outputs, input_ids):
        """Turn LLM self-attention received by each <image> token into a debiased
        per-group saliency vector, stashed on vision_backbone._attn_feedback_saliency_out.
        Only fires on FULL (survey) frames -- attn-seq == input_ids len -- so pruned
        frames never clobber the last full-resolution saliency map."""
        import os as _os
        attns = getattr(outputs, "attentions", None)
        if not attns or input_ids is None:
            return
        if getattr(self.vision_backbone, "_groupdrop_keep", None) is not None:
            return  # a prune happened this frame -> not a full survey
        try:
            S = int(input_ids.shape[1])
            a0 = attns[0]
            if a0 is None or int(a0.shape[-1]) != S:
                return
            ipid = int(self.config.image_patch_id)
            img = (input_ids[0] == ipid)
            n_img = int(img.sum())
            if n_img == 0:
                return
            nL = len(attns)
            lo = int(_os.environ.get("VIS_ATTNFB_LAYER_LO", str(max(0, nL - 8))))
            hi = int(_os.environ.get("VIS_ATTNFB_LAYER_HI", str(nL)))
            lo = max(0, min(lo, nL)); hi = max(lo + 1, min(hi, nL))
            txt = ~img
            if int(txt.sum()) == 0:
                txt = torch.ones_like(img)
            recv = None
            for li in range(lo, hi):
                a = attns[li]
                if a is None:
                    continue
                a = a.float()[0]                          # (H, S, S)
                col = a[:, txt, :][:, :, img].sum(dim=1)  # (H, n_img): attn paid by text rows to image keys
                col = col.mean(dim=0)                     # (n_img,)
                recv = col if recv is None else recv + col
            if recv is None:
                return
            sal = recv
            # Sink suppression. Selection is argsort-based, so a *monotonic* debias
            # cannot change which groups are kept -- only DEMOTING the task-independent
            # attention sink (the extreme MAD-outlier column) changes the keep set,
            # while preserving strict (deterministic) order among the remaining groups.
            mode = _os.environ.get("VIS_ATTNFB_DEBIAS", "sink")
            if mode in ("sink", "clip"):
                k_mad = float(_os.environ.get("VIS_ATTNFB_SINK_K", "6.0"))
                med = sal.median()
                mad = (sal - med).abs().median() + 1e-6
                outlier = sal > (med + k_mad * mad)
                if bool(outlier.any()) and bool((~outlier).any()):
                    floor = sal[~outlier].min() - (sal[~outlier].max() - sal[~outlier].min() + 1e-6)
                    sal = sal.masked_fill(outlier, floor)
            elif mode == "zmedian":
                med = sal.median()
                mad = (sal - med).abs().median() + 1e-6
                sal = (sal - med) / mad
            sal = sal - sal.min()
            self.vision_backbone._attn_feedback_saliency_out = sal.detach().to("cpu")
            if _os.environ.get("VIS_ATTNFB_DEBUG"):
                import sys as _sys
                print("[VIS_ATTNFB] harvested saliency n_img=%d layers=[%d,%d) max=%.4g"
                      % (n_img, lo, hi, float(sal.max())), file=_sys.stderr, flush=True)
        except Exception as _e:
            if _os.environ.get("VIS_ATTNFB_DEBUG"):
                import sys as _sys
                print("[VIS_ATTNFB] gather failed:", repr(_e), file=_sys.stderr, flush=True)

    # === VIS_ATTNFB_METHOD END ===
    def generate_actions_from_inputs(
'''
EDIT4 = (ANCHOR4, BLOCK4)

EDITS = [EDIT1, EDIT2, EDIT3, EDIT4]


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
        print("usage: patch_vis_attnfeedback.py <hf_cache_dir> [--revert]")
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
