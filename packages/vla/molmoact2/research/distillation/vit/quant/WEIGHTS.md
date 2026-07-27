# Weights — full dev flow (distill+ft → PTQ → QAT), both variants

Per project policy, model weights are **not committed to git**. They are hosted
externally and pulled on demand into `resource/ckpt/vit_distill/` (fp32 students)
and `resource/ckpt/vit_distill/quant/` (quantized blobs).

- **Host:** `sayeedmd320/molmoact2-vit-distill-w4a6` (private Hugging Face model repo; request access).
- **Fetch:** `bash download_weights.sh` (login first: `huggingface-cli login` or export `HF_TOKEN`).

## Deployable winners (W4A6 QAT)
| variant | checkpoint | closed-loop mean |
|---|---|---|
| conv (`cnn`) | `quant/cnn_w4a6_qat.pt` | **95.0** |
| nano (`tinyvit`) | `quant/tinyvit_w4a6_qat.pt` | **93.8** |

## Full manifest (sha256, bytes)

### conv (`cnn`) lineage
| stage | file | bytes | sha256 |
|---|---|---|---|
| distill+ft (fp32 student) | `cnn_fpga_full.pt` | 46124115 | `cf29614a14e56003069cf070022d70638ec7e700ccab08a3ee53cb6de0bb2dd6` |
| W8A8 PTQ | `quant/cnn_w8a8_ptq.pt` | 15653560 | `565cf24ce19e6edf2cd1646eb30ca46a7aa95fc989dcc4ca9b04644a54adb3d1` |
| W4A6 PTQ | `quant/cnn_w4a6_ptq.pt` | 15653560 | `f812fe0f6a1a6c5d390d32dcd7adf52ae007a338b8dcfa98f3625fb04c834f31` |
| **W4A6 QAT (winner)** | `quant/cnn_w4a6_qat.pt` | 15653496 | `57857c9f886227d68448647587b3ed2b829e73a6b38f8aa4edc256cdb1630d8e` |
| W4A6 re-distill (3ep, DROID mix) — history | `quant/cnn_w4a6_redistill.pt` | 15654911 | `5f82e7c786aa3dadcd1282ea73d177fcfc2aac0f45708cc714847b00be97980f` |
| W4A6 re-distill + QAT-ft (3ep) — history | `quant/cnn_w4a6_redistill_qatft.pt` | 15655833 | `7607dfe97562c0811c06e1dd4cf086375c329870ab1e3b2f950d2ef101283795` |

### nano (`tinyvit`) lineage
| stage | file | bytes | sha256 |
|---|---|---|---|
| distill+ft (fp32 student) | `siglip_nano_full.pt` | 32778617 | `bad17331010e35d5454fc5ce677c32eb8fe7de184971f1dd38acb643ba622e0b` |
| W8A8 PTQ | `quant/tinyvit_w8a8_ptq.pt` | 11195103 | `30051aa9a98220e0ca3b844274c9c3834fbca7d29c592ba4bbe335ea4d183d73` |
| W4A6 PTQ | `quant/tinyvit_w4a6_ptq.pt` | 11195103 | `e57f2c739acfac600fb9d9abeb62caa5e57ef87af76e1f507c6809b6da3540cf` |
| **W4A6 QAT (winner)** | `quant/tinyvit_w4a6_qat.pt` | 11195039 | `ab01558e995a51401437885ae90a495fe3bcafc06d3dea58c9f27a079a8b500f` |
| W4A6 re-distill (3ep, DROID mix) — history | `quant/tinyvit_w4a6_redistill.pt` | 11196222 | `be70e2ed77b5fa6d2fb0b8bb8a7cc1540f4d453d387b5778c5c68f15dbd7afe6` |
| W4A6 re-distill + QAT-ft (3ep) — history | `quant/tinyvit_w4a6_redistill_qatft.pt` | 11196840 | `37bfc3dccbe19eabd2ba45cc636e57229cf6a566d0639dc4db65097c54756c25` |

Verify after download: `sha256sum -c` against this manifest.

## How each blob loads
- fp32 students: `variants.load_encoder("cnn"|"tinyvit", ckpt=...)`.
- quant blobs: `quantize_student_(encoder, weight_bits, act_bits)` then load the
  saved fake-quant `state_dict` (see `quant/eval_closedloop_quant.py` and
  `quant/eval_seam.py` for the exact load path).
