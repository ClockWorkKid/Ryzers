# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Guard VERA's eval-only LPIPS metric imports against the pinned torchmetrics 1.4.0.post0.

Two files in `vera/idm/common/metrics/video/` import symbols from `torchmetrics.image.lpip`
that no longer exist under the installed torchmetrics (probed on the built image):
  * `shared_registry.py`: `NoTrainLpips`  -> renamed to the private `_NoTrainLpips`.
  * `lpips.py`         : `_valid_img`     -> no longer exported from that module.

These are video-quality metrics, used only for offline eval — they are NOT on the
closed-loop inference path (planner dream + Jacobian IDM). But the DFoT/PushT planner
import chain pulls `shared_registry`, so a hard `ImportError` there blocks the whole
policy import. Per rule 2.1 we patch only the version-specific breakage: rewrite the two
imports into resilient try/except blocks (prefer the real symbol, degrade to a stub) so
the package imports cleanly while keeping the metric working wherever torchmetrics still
provides it. Idempotent (sentinel-guarded); never fails the build.
"""
import sys
from pathlib import Path

SENTINEL = "[ryzers-eval-patch]"

SHARED_REGISTRY_OLD = "from torchmetrics.image.lpip import NoTrainLpips\n"
SHARED_REGISTRY_NEW = (
    "# " + SENTINEL + " NoTrainLpips was renamed to the private _NoTrainLpips in\n"
    "# torchmetrics 1.4.x. Metrics are eval-only (off the inference path); prefer the\n"
    "# real class, else fall back to the public LPIPS so the import never hard-fails.\n"
    "try:\n"
    "    from torchmetrics.image.lpip import NoTrainLpips  # older torchmetrics\n"
    "except ImportError:  # torchmetrics >= 1.4\n"
    "    try:\n"
    "        from torchmetrics.image.lpip import _NoTrainLpips as NoTrainLpips\n"
    "    except ImportError:\n"
    "        from torchmetrics.image.lpip import (\n"
    "            LearnedPerceptualImagePatchSimilarity as NoTrainLpips,\n"
    "        )\n"
)

LPIPS_OLD = (
    "from torchmetrics.image.lpip import (\n"
    "    LearnedPerceptualImagePatchSimilarity as _LearnedPerceptualImagePatchSimilarity,\n"
    "    _valid_img,\n"
    ")\n"
)
LPIPS_NEW = (
    "# " + SENTINEL + " torchmetrics 1.4.x no longer exports `_valid_img` from\n"
    "# torchmetrics.image.lpip. Import the class normally and guard the private helper\n"
    "# (eval-only) with a permissive fallback so the metrics package still imports.\n"
    "from torchmetrics.image.lpip import (\n"
    "    LearnedPerceptualImagePatchSimilarity as _LearnedPerceptualImagePatchSimilarity,\n"
    ")\n"
    "try:\n"
    "    from torchmetrics.image.lpip import _valid_img\n"
    "except ImportError:  # pragma: no cover - eval-only helper\n"
    "    def _valid_img(img, normalize):\n"
    "        return True\n"
)

# Field-name bug: `tracker_backend_from_cfg` reads `cfg.tracker_backend`, but the WAN pipeline's
# `MotionTrackConfig` (the motion tracker used on the dreamed video) names the field `backend`.
# So `MotionTrackConfig(backend="cotracker")` is silently ignored and the WAN motion tracker
# always resolves to "alltracker" — which the MimicGen build_policy explicitly warns produces
# wrong-direction flow ("arm flees blocks"). Honor `.backend` as a fallback so the algo_config's
# `tracker.backend: cotracker` (and cotracker feedback) actually takes effect. Off the vendored-
# alltracker path entirely; cotracker is the intended MimicGen tracker.
TRACKER_OLD = (
    "    backend = getattr(cfg, \"tracker_backend\", None)\n"
    "    if backend is None:\n"
    "        return \"alltracker\"\n"
)
TRACKER_NEW = (
    "    backend = getattr(cfg, \"tracker_backend\", None)\n"
    "    if backend is None:  # " + SENTINEL + " MotionTrackConfig names the field `backend`\n"
    "        backend = getattr(cfg, \"backend\", None)\n"
    "    if backend is None:\n"
    "        return \"alltracker\"\n"
)

# Two bugs on the cotracker visualization path, both in draw_pts_gpu's contract. draw_pts_gpu runs
# on `rgbs.device` and boolean-indexes the visibility mask (`trajs[mask, t]`, `colors[mask]`):
#   1) dtype   — `visibs` must be bool, but CoTracker returns float visibility probabilities
#                -> `IndexError: tensors used as indices must be long, int, byte or bool`.
#   2) device  — the wrapper moved tracks/visibility to CPU while `pixel_video` stayed on GPU
#                -> `RuntimeError: expected all tensors on the same device (cuda vs cpu)`.
# alltracker's wrapper avoids both (bool mask + all tensors on the video's device). Mirror that:
# threshold visibility to a bool mask kept on the video's device, and hand draw_pts_gpu the GPU
# tracks; a CPU float copy is retained only to compute per-point colors (numpy). The control-path
# RuntimeMotionTracks keeps its own float visibility copy (built earlier from pred_visibility).
COTRACKER_VIS_OLD = (
    "            visibility = visibility.detach().cpu().float()\n"
    "            pred_tracks_cpu = pred_tracks.detach().cpu().float()\n"
    "            for batch_idx in range(pixel_video.shape[0]):\n"
    "                xy0 = pred_tracks_cpu[batch_idx, 0].numpy()\n"
    "                colors = _get_2d_colors(xy0, image_size[0], image_size[1])\n"
    "                vis_np = draw_pts_gpu(\n"
    "                    pixel_video[batch_idx],\n"
    "                    pred_tracks_cpu[batch_idx],\n"
    "                    visibility[batch_idx],\n"
)
COTRACKER_VIS_NEW = (
    "            # " + SENTINEL + " keep the tracks viz on the video's device with a bool mask\n"
    "            # (draw_pts_gpu boolean-indexes visibs on rgbs.device); CPU float copy is only\n"
    "            # used to compute per-point colors (numpy).\n"
    "            visibility = visibility.detach() > 0.5\n"
    "            pred_tracks_cpu = pred_tracks.detach().cpu().float()\n"
    "            for batch_idx in range(pixel_video.shape[0]):\n"
    "                xy0 = pred_tracks_cpu[batch_idx, 0].numpy()\n"
    "                colors = _get_2d_colors(xy0, image_size[0], image_size[1])\n"
    "                vis_np = draw_pts_gpu(\n"
    "                    pixel_video[batch_idx],\n"
    "                    pred_tracks[batch_idx].float(),\n"
    "                    visibility[batch_idx],\n"
)


def _patch(path: Path, old: str, new: str) -> str:
    if not path.exists():
        return f"SKIP (missing): {path}"
    text = path.read_text()
    if SENTINEL in text:
        return f"SKIP (already patched): {path}"
    if old not in text:
        return f"WARN (import block not found, upstream drift?): {path}"
    path.write_text(text.replace(old, new, 1))
    return f"PATCHED: {path}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/repos/vera")
    metrics = root / "vera" / "idm" / "common" / "metrics" / "video"
    print(_patch(metrics / "shared_registry.py", SHARED_REGISTRY_OLD, SHARED_REGISTRY_NEW))
    print(_patch(metrics / "lpips.py", LPIPS_OLD, LPIPS_NEW))
    tracker = root / "vera" / "policy" / "world_models" / "tracker_backends.py"
    print(_patch(tracker, TRACKER_OLD, TRACKER_NEW))
    cotracker = root / "vera" / "policy" / "world_models" / "cotracker_inference.py"
    print(_patch(cotracker, COTRACKER_VIS_OLD, COTRACKER_VIS_NEW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
