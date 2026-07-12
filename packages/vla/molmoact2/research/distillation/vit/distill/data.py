"""Image data for distillation: LIBERO frames (direct parquet) + heavy augmentation.

Vision-encoder distillation needs only images (no actions). The MolmoAct2-LIBERO
dataset stores each camera frame as PNG-encoded bytes inside its ``data/*.parquet``
files (HF ``Image`` feature: ``struct<bytes, path>``, 256x256x3) -- there are no
separate video files. We therefore read frames straight from parquet with pyarrow +
PIL, which is far lighter and faster than LeRobot's episode/video index (that path
also trips a lerobot<->huggingface_hub version bug; see ``_lerobot_compat.py``).

Frames are augmented (input-space perturbation) then patchified+normalized with the
teacher processor, so the *identical* tensor drives teacher and student. A dummy
target keeps torchdistill's ``(batch, target)`` contract.
"""

from __future__ import annotations

import glob
import io
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AugConfig:
    enable: bool = True
    color_jitter: float = 0.3        # brightness/contrast/saturation strength
    hue: float = 0.05
    blur_p: float = 0.3
    gray_p: float = 0.1
    noise_std: float = 0.02          # gaussian noise on [0,1] frames
    erase_p: float = 0.25
    hflip_p: float = 0.0             # off by default (spatial semantics matter)


def build_augment(aug: AugConfig):
    """Return a callable transform on a uint8 HWC tensor -> uint8 HWC tensor."""
    import torchvision.transforms.v2 as T

    ops: list = []
    if aug.hflip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=aug.hflip_p))
    if aug.color_jitter > 0:
        ops.append(
            T.ColorJitter(
                brightness=aug.color_jitter,
                contrast=aug.color_jitter,
                saturation=aug.color_jitter,
                hue=aug.hue,
            )
        )
    if aug.gray_p > 0:
        ops.append(T.RandomGrayscale(p=aug.gray_p))
    if aug.blur_p > 0:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=5)], p=aug.blur_p))

    color = T.Compose(ops) if ops else None

    def _apply(frame_u8: torch.Tensor) -> torch.Tensor:
        x = frame_u8.permute(2, 0, 1)  # CHW
        if color is not None:
            x = color(x)
        if aug.noise_std > 0:
            xf = x.float() / 255.0
            xf = (xf + torch.randn_like(xf) * aug.noise_std).clamp(0, 1)
            x = (xf * 255.0).round().to(torch.uint8)
        if aug.erase_p > 0:
            x = T.RandomErasing(p=aug.erase_p, value="random")(x)
        return x.permute(1, 2, 0).contiguous()

    return _apply


@dataclass
class DataConfig:
    repo_id: str = "allenai/MolmoAct2-LIBERO-Dataset"
    revision: str = "main"
    image_keys: list[str] = field(default_factory=list)  # empty -> both cameras
    # data source
    cache_dir: str | None = None      # HF cache holding the snapshot (reuse the 33GB)
    snapshot_dir: str | None = None   # explicit snapshot path (skips download)
    subsample_stride: int = 1         # keep every Nth frame (thin the 320k frames)
    max_frames: int | None = None     # hard cap (smoke runs)
    val_fraction: float = 0.05        # last fraction of files held out for eval
    cache_files: int = 3              # per-worker LRU of decoded parquet files
    seed: int = 0
    aug: AugConfig = field(default_factory=AugConfig)
    # patchifier
    patchify_mode: str = "processor"                 # "processor" | "simple"
    checkpoint_path: str = "allenai/MolmoAct2-LIBERO"


_DEFAULT_KEYS = ["observation.images.image", "observation.images.wrist_image"]


def _find_cached_snapshot(cfg: DataConfig) -> str | None:
    """Locate an already-downloaded snapshot dir directly (no network / ref resolution).

    Works offline regardless of how the cache was populated (LeRobot stores the ref by
    commit hash, so offline ``snapshot_download(revision='main')`` can't resolve it).
    """
    flat = "datasets--" + cfg.repo_id.replace("/", "--")
    roots = [cfg.cache_dir, os.environ.get("HF_LEROBOT_HOME")]
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    roots += [os.path.join(hf_home, "lerobot"), hf_home]
    for root in roots:
        if not root:
            continue
        for snap in sorted(glob.glob(os.path.join(root, "hub", flat, "snapshots", "*"))):
            if os.path.isdir(os.path.join(snap, "data")):
                return snap
    return None


def _resolve_snapshot(cfg: DataConfig) -> str:
    """Return a local dir with the dataset's data/ + meta/ parquets (download if absent)."""
    if cfg.snapshot_dir:
        return cfg.snapshot_dir
    cached = _find_cached_snapshot(cfg)
    if cached:
        return cached
    from huggingface_hub import snapshot_download

    cache_dir = cfg.cache_dir or os.environ.get("HF_LEROBOT_HOME")
    return snapshot_download(
        cfg.repo_id, repo_type="dataset", revision=cfg.revision,
        allow_patterns=["data/**", "meta/**"], cache_dir=cache_dir,
    )


class LiberoFrameDataset(Dataset):
    """Direct-parquet LIBERO frames -> (normalized teacher-ready patches, dummy target).

    Files are shuffled once (seeded) and the flat frame index is built file-major so a
    small per-file LRU cache keeps parquet reads sequential and cheap even under a
    plain (shuffle=False) DataLoader.
    """

    def __init__(self, cfg: DataConfig | None = None, train: bool = True, **kwargs) -> None:
        import pyarrow.parquet as pq

        from .patchify import FramePatchifier

        if cfg is None:
            cfg = DataConfig(**kwargs)
        self.cfg = cfg
        self._pq = pq
        self.keys = cfg.image_keys or _DEFAULT_KEYS

        root = _resolve_snapshot(cfg)
        files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
        if not files:
            raise RuntimeError(f"No data parquet files under {root}/data")

        # deterministic file-level train/val split (hold out the LAST val_fraction files)
        n_val = max(1, int(round(len(files) * cfg.val_fraction))) if cfg.val_fraction > 0 else 0
        files = files[: len(files) - n_val] if train else files[len(files) - n_val:]

        # seeded file-order permutation -> file-major flat index (locality-friendly)
        rng = np.random.default_rng(cfg.seed + (0 if train else 1))
        order = rng.permutation(len(files))
        self.files = [files[i] for i in order]
        self._rows = [pq.ParquetFile(f).metadata.num_rows for f in self.files]

        index: list[tuple[int, int, str]] = []
        stride = max(1, cfg.subsample_stride)
        for fi, nrows in enumerate(self._rows):
            local = rng.permutation(nrows)  # randomize row order within each file
            for r in range(0, nrows, stride):
                for k in self.keys:
                    index.append((fi, int(local[r]), k))
        if cfg.max_frames is not None:
            index = index[: cfg.max_frames]
        self._index = index

        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._augment = build_augment(cfg.aug) if (train and cfg.aug.enable) else None
        self._patchify = FramePatchifier(mode=cfg.patchify_mode, checkpoint_path=cfg.checkpoint_path)

    def __len__(self) -> int:
        return len(self._index)

    def _file_columns(self, fi: int) -> dict:
        cache = self._cache
        if fi in cache:
            cache.move_to_end(fi)
            return cache[fi]
        tbl = self._pq.read_table(self.files[fi], columns=self.keys)
        cols = {k: tbl.column(k).to_pylist() for k in self.keys}  # list of {'bytes','path'}
        cache[fi] = cols
        cache.move_to_end(fi)
        while len(cache) > max(1, self.cfg.cache_files):
            cache.popitem(last=False)
        return cols

    def _decode(self, cell) -> torch.Tensor:
        from PIL import Image

        b = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # HWC (writable copy)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        fi, r, key = self._index[idx]
        cell = self._file_columns(fi)[key][r]
        img = self._decode(cell)  # HWC uint8
        if self._augment is not None:
            img = self._augment(img)
        patches = self._patchify(img)  # [num_crops, N, in_pixels] float32, normalized
        return patches, 0


class ImageFolderFrameDataset(Dataset):
    """Real-image frames from a flat directory (jpg/png) -> teacher-ready patches.

    Used for the DROID real-robot frames (pre-extracted to JPEG by the prep step),
    yielding the identical ``(patches, dummy)`` contract as ``LiberoFrameDataset`` so
    the two can be concatenated for a mixed distillation set. The teacher processor
    resizes arbitrary input resolutions (e.g. DROID 180x320) to the 378x378 crop.
    """

    def __init__(self, root: str, train: bool = True, aug: "AugConfig | None" = None,
                 max_frames: int | None = None, val_fraction: float = 0.05,
                 patchify_mode: str = "processor",
                 checkpoint_path: str = "allenai/MolmoAct2-LIBERO", seed: int = 0,
                 **kwargs) -> None:
        from PIL import Image  # noqa: F401  (import cost paid once)

        from .patchify import FramePatchifier

        exts = ("*.jpg", "*.jpeg", "*.png")
        files: list[str] = []
        for e in exts:
            files += glob.glob(os.path.join(root, "**", e), recursive=True)
        files = sorted(files)
        if not files:
            raise RuntimeError(f"No image frames (jpg/png) under {root}")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        files = [files[i] for i in order]
        n_val = max(1, int(round(len(files) * val_fraction))) if val_fraction > 0 else 0
        files = files[: len(files) - n_val] if train else files[len(files) - n_val:]
        if max_frames is not None:
            files = files[:max_frames]
        self.files = files
        self._augment = build_augment(aug or AugConfig()) if train else None
        self._patchify = FramePatchifier(mode=patchify_mode, checkpoint_path=checkpoint_path)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.files[idx]).convert("RGB")
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))  # HWC uint8
        if self._augment is not None:
            arr = self._augment(arr)
        patches = self._patchify(arr)
        return patches, 0


class LiberoDroidMixDataset(Dataset):
    """Concatenation of LIBERO (sim, parquet) + DROID (real, image-folder) frames.

    Registry entrypoint for the hybrid_droid experiment: builds both source datasets
    and exposes them as one dataset so torchdistill's data loader mixes them each
    epoch (RandomSampler over the concatenation). Both branches emit the identical
    teacher-ready ``(patches, dummy)`` tensors.
    """

    def __init__(self, train: bool = True,
                 repo_id: str = "allenai/MolmoAct2-LIBERO-Dataset",
                 checkpoint_path: str = "allenai/MolmoAct2-LIBERO",
                 patchify_mode: str = "processor",
                 subsample_stride: int = 1, max_frames: int | None = None,
                 droid_root: str = "/cache/droid_frames",
                 droid_max_frames: int | None = None, **kwargs) -> None:
        from torch.utils.data import ConcatDataset

        libero = LiberoFrameDataset(
            train=train, repo_id=repo_id, checkpoint_path=checkpoint_path,
            patchify_mode=patchify_mode, subsample_stride=subsample_stride,
            max_frames=max_frames,
        )
        droid = ImageFolderFrameDataset(
            root=droid_root, train=train, max_frames=droid_max_frames,
            patchify_mode=patchify_mode, checkpoint_path=checkpoint_path,
        )
        self._ds = ConcatDataset([libero, droid])
        self.n_libero = len(libero)
        self.n_droid = len(droid)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]
