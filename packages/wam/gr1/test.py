# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the GR-1 vision-language-action image on Strix Halo (gfx1151).

Runs inside the built image with NO model weights and NO simulator: proves (1) a working ROCm
torch on the iGPU, (2) GR-1's model-path deps import cleanly, (3) each submodule runs a random-
input forward on device independently (rule 2 — MAE ViT, Perceiver resampler, GPT-2 backbone),
and (4) the full GR-1 policy forward produces action + future-frame predictions of the expected
shapes. CLIP's text encoder is stubbed (random features) so the smoke needs no weight download;
the real CLIP ViT-B/32 is exercised in the phase-2 open-loop demo. Exits non-zero on any failure
so `ryzers run` / CI catches a broken image early.
"""
import sys

# GR-1 hyperparameters (upstream logs/configs.json).
CFG = dict(
    embed_dim=384, n_layer=12, n_head=12, activation_function="relu", dropout=0.1,
    n_positions=1024, resampler_depth=3, resampler_dim_head=128, resampler_heads=4,
    resampler_num_media_embeds=1, resampler_num_latents=9, seq_len=10, act_dim=7,
    state_dim=7, use_hand_rgb=True, without_norm_pix_loss=False,
    img_feat_dim=768, patch_feat_dim=768, lang_feat_dim=512,
)


class _StubCLIP:
    """Stand-in for clip.load(...)[0]: only encode_text + named_parameters are used by GR1."""
    def __init__(self, lang_feat_dim):
        self.lang_feat_dim = lang_feat_dim

    def encode_text(self, tokens):
        import torch
        return torch.randn(tokens.shape[0], self.lang_feat_dim, device=tokens.device)

    def named_parameters(self):
        return iter(())


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible. Check --device=/dev/kfd, /dev/dri.", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    import clip                                          # noqa: F401  (openai CLIP)
    import transformers
    import einops                                        # noqa: F401
    from flamingo_pytorch import PerceiverResampler
    import models.vision_transformer as vits
    from models.trajectory_gpt2 import GPT2Model
    from models.gr1 import GR1
    print(f"transformers     : {transformers.__version__}")
    print("imports ok       : clip, flamingo_pytorch, models.{vision_transformer,trajectory_gpt2,gr1}")

    dev = "cuda"
    B, L = 1, CFG["seq_len"]

    # (1) MAE ViT-B/16 image encoder: (N,3,224,224) -> (N,768) cls + (N,196,768) patches.
    mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(dev).eval()
    with torch.no_grad():
        cls_feat, patch_feat = mae(torch.randn(2, 3, 224, 224, device=dev))
    assert tuple(cls_feat.shape) == (2, 768), cls_feat.shape
    assert tuple(patch_feat.shape) == (2, 196, 768), patch_feat.shape
    print(f"MAE ViT ok       : cls={tuple(cls_feat.shape)} patch={tuple(patch_feat.shape)}")

    # (2) Perceiver resampler: 196 patch tokens -> num_latents=9.
    resampler = PerceiverResampler(
        dim=CFG["patch_feat_dim"], depth=CFG["resampler_depth"],
        dim_head=CFG["resampler_dim_head"], heads=CFG["resampler_heads"],
        num_latents=CFG["resampler_num_latents"],
        num_media_embeds=CFG["resampler_num_media_embeds"]).to(dev).eval()
    with torch.no_grad():
        r = resampler(patch_feat.unsqueeze(1)).squeeze(1)
    assert tuple(r.shape) == (2, CFG["resampler_num_latents"], CFG["patch_feat_dim"]), r.shape
    print(f"resampler ok     : out={tuple(r.shape)}")

    # (3) GPT-2 trajectory backbone: random embeds -> last_hidden_state.
    gpt_cfg = transformers.GPT2Config(
        vocab_size=1, n_embd=CFG["embed_dim"], n_layer=2, n_head=CFG["n_head"],
        n_inner=4 * CFG["embed_dim"], activation_function=CFG["activation_function"],
        n_positions=CFG["n_positions"], resid_pdrop=CFG["dropout"], attn_pdrop=CFG["dropout"])
    gpt = GPT2Model(gpt_cfg).to(dev).eval()
    with torch.no_grad():
        g = gpt(inputs_embeds=torch.randn(1, 16, CFG["embed_dim"], device=dev),
                attention_mask=torch.ones(1, 16, dtype=torch.long, device=dev))
    assert tuple(g["last_hidden_state"].shape) == (1, 16, CFG["embed_dim"]), g["last_hidden_state"].shape
    print(f"GPT-2 backbone ok: out={tuple(g['last_hidden_state'].shape)}")

    # (4) Full GR-1 policy forward (all heads), CLIP stubbed to avoid a weight download.
    policy = GR1(
        model_clip=_StubCLIP(CFG["lang_feat_dim"]),
        model_mae=mae,
        state_dim=CFG["state_dim"], act_dim=CFG["act_dim"], hidden_size=CFG["embed_dim"],
        sequence_length=CFG["seq_len"], training_target=["act_pred", "fwd_pred", "fwd_pred_hand"],
        img_feat_dim=CFG["img_feat_dim"], patch_feat_dim=CFG["patch_feat_dim"],
        lang_feat_dim=CFG["lang_feat_dim"],
        resampler_params=dict(
            depth=CFG["resampler_depth"], dim_head=CFG["resampler_dim_head"],
            heads=CFG["resampler_heads"], num_latents=CFG["resampler_num_latents"],
            num_media_embeds=CFG["resampler_num_media_embeds"]),
        without_norm_pixel_loss=CFG["without_norm_pix_loss"], use_hand_rgb=CFG["use_hand_rgb"],
        n_layer=CFG["n_layer"], n_head=CFG["n_head"], n_inner=4 * CFG["embed_dim"],
        activation_function=CFG["activation_function"], n_positions=CFG["n_positions"],
        resid_pdrop=CFG["dropout"], attn_pdrop=CFG["dropout"]).to(dev).eval()

    rgb = torch.randn(B, L, 3, 224, 224, device=dev)
    hand_rgb = torch.randn(B, L, 3, 224, 224, device=dev)
    state = {"arm": torch.randn(B, L, CFG["act_dim"] - 1, device=dev),
             "gripper": torch.randn(B, L, 2, device=dev)}
    language = torch.randint(0, 100, (B, 77), dtype=torch.long, device=dev)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = policy(rgb=rgb, hand_rgb=hand_rgb, state=state, language=language,
                      attention_mask=attention_mask)

    n_patches = (224 // 16) ** 2  # 196
    p2c = 16 * 16 * 3             # 768
    assert tuple(pred["arm_action_preds"].shape) == (B, L, CFG["act_dim"] - 1), pred["arm_action_preds"].shape
    assert tuple(pred["gripper_action_preds"].shape) == (B, L, 1), pred["gripper_action_preds"].shape
    assert tuple(pred["obs_preds"].shape) == (B, L, n_patches, p2c), pred["obs_preds"].shape
    assert tuple(pred["obs_hand_preds"].shape) == (B, L, n_patches, p2c), pred["obs_hand_preds"].shape
    print(f"GR-1 forward ok  : arm={tuple(pred['arm_action_preds'].shape)} "
          f"grip={tuple(pred['gripper_action_preds'].shape)} "
          f"fwd={tuple(pred['obs_preds'].shape)} hand={tuple(pred['obs_hand_preds'].shape)}")

    print("PASS: GR-1 ROCm env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
