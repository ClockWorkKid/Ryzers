# Release TODO: gallery assets to regenerate

Some packages ship README galleries with `PLACEHOLDER` images because their demo
outputs were pruned and no source clips survive on the laptop or the remote. These
must be regenerated on strix-halo (run the demos, capture the outputs, convert to
small gifs/pngs, replace the placeholders) before the final upstream release.

Everything else in each README (structure, build, demos, numbers, knobs) is final.

| Package | Placeholder slots to regenerate | Notes |
|---|---|---|
| `wam/imagewam` | open-loop, dreamed-frame, closed-loop (LIBERO + RoboTwin), interactive | No source assets anywhere; full render pass needed. |
| `wam/vlajepa` | closed-loop LIBERO, closed-loop LIBERO-Plus, closed-loop SimplerEnv, latent imagine, open-loop | No source assets anywhere; full render pass needed (all 5 gallery slots are placeholders). |
| `wam/ahawam` | open-loop plots, closed-loop RoboTwin beyond `beat_block_hammer`, interactive/real-time RoboTwin capture | Only `beat_block_hammer` rollouts survive; open-loop/perf plots gone. Widen the closed-loop gallery to one success-true gif per remaining suite task (`click_bell`, `handover_block`, `place_object_basket`, `lift_pot`). |
| `simulation/ithor` | Example rollout (`ithor_rollout.png`, AVDC closed-loop ObjectNav) | No source clips anywhere; render on strix-halo (ai2thor CloudRendering) with AVDC chained. |
| `simulation/libero-plus` | paired example: closed-loop LIBERO-Plus (VLA-JEPA) | No source clip anywhere; render the VLA-JEPA closed-loop rollout on strix-halo. |
| `simulation/simplerenv` | worked-example visual `assets/simplerenv_vlajepa.png` (VLA-JEPA closed-loop) | No source clip on laptop or remote; render the VLA-JEPA closed-loop demo, save as `assets/simplerenv_vlajepa.gif`. |

## How to regenerate

1. Build/refresh the package image on strix-halo (ROCm 7.14 base) and its sim base.
2. Run the demos that back each placeholder slot (open-loop, videogen/dream, closed-loop).
3. Convert clips with `agent_scripts/mp4_to_gif.py` (keep each gif well under ~1 MB).
4. Drop the results into the package `assets/` under the placeholder filenames and delete
   the placeholder PNGs and their `<!-- TODO(release) -->` comments in the README.

## Pre-release checks (non-gallery)

- [ ] `simulation/libero-plus`: pin the exact upstream commit. The Dockerfile clones
  `sylvestf/LIBERO-plus` with `--depth 1` (default-branch HEAD), so `docs/UPSTREAM_PIN.commit.txt`
  records the repo but not a SHA. Resolve and pin it on strix-halo before release.
- [ ] Audit every package's Dockerfile for other unpinned `git clone --depth 1` / branch clones
  and pin them for reproducibility.
- [ ] `wam/imagewam`: reconcile the checkpoint source. `config.yaml` derives the autoencoder
  from the public klein VAE via `convert_klein_vae.py`, while `download_checkpoints.sh` fetches
  the gated FLUX.2-dev `ae.safetensors`. Pick one default and align the README.
- [ ] `simulation/simplerenv`: confirm SAPIEN 2 / ManiSkill2-real2sim is the intended shipped
  backend (the README now documents what the Dockerfile ships, dropping the old SAPIEN 3 note).
- [ ] `wam/vera`: docs (`PLAN.md`, `PORT_SUMMARY.md`, `SCOPING.md`) still name
  `scripts/strip_cuda_torch.py`, but the package moved to a `PIP_CONSTRAINT` approach and ships no
  such script. Update the doc wording (or restore the script) so the reference resolves.
