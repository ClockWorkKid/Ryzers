#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stage 3 -- Path B: numeric validation vs ground-truth actions across N
held-out DROID episodes from ``lerobot/droid_100``.

For each (episode, chunk_start) pair:
  1. Load 4 anchor frames per camera at offsets [c-23, c-16, c-8, c] (matching
     the AR_droid frame schedule).
  2. Load the episode's task text as the prompt.
  3. Run the model, get the predicted (24, 8) action chunk.
  4. Read ground-truth actions from the parquet at frames [c+1, c+2, ..., c+24].
  5. Compute MSE (per dim) + cosine similarity (per step) between predicted
     and ground-truth.

IMPORTANT alignment note (logged in the report):
  - Model emits action.joint_position (7) + action.gripper_position (1) = 8 dims.
  - lerobot/droid_100 GT action is (7) where dim[6] is binary gripper.
  - The model's first 6 dims and GT's first 6 dims are both likely Cartesian
    xyz_rpy end-effector deltas (DROID native action space is Cartesian; the
    "joint_position" name in the model output is a legacy field name).
  - The 7th model dim's semantics are unclear without a confirmed cross-check
    against original DROID HDF5; for v1 of Path B we report MSE on the first 6
    overlap dims + gripper accuracy, and save raw arrays for offline analysis.

Inputs (read-only):
  - $MODEL_PATH                  DreamZero-DROID snapshot dir
  - $WAN21_DIR                   Wan2.1 snapshot mount point
  - $DROID_DATASET_DIR           lerobot/droid_100 local dir (default
                                 /data/lerobot_droid_100)
  - $EPISODES                    comma-sep list of episode_index to use
                                 (default "5,10,20")
  - $CHUNKS_PER_EPISODE          int K (default 3)
  - $OUTPUT_DIR                  where to dump artifacts

Outputs:
  - per_chunk.csv                one row per (episode, chunk) inference
  - per_chunk_actions.npz        all predicted + GT action arrays keyed by
                                 ep<NN>_c<NN>
  - aggregate.json               aggregate MSE/cosine numbers
  - episode_summary.csv          per-episode aggregate
  - plot_*.png                   per-(episode,chunk) overlay plots
  - manifest.json                env + result + pass/fail

Exit codes mirror Path A.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any


def banner(msg: str) -> None:
    bar = "=" * max(len(msg), 60)
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def cuda_mem_gib(reset: bool = False) -> dict[str, float]:
    import torch
    if not torch.cuda.is_available():
        return {"alloc_gib": 0.0, "max_alloc_gib": 0.0, "reserved_gib": 0.0}
    torch.cuda.synchronize()
    out = {
        "alloc_gib":     torch.cuda.memory_allocated() / 1024**3,
        "max_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gib":  torch.cuda.memory_reserved() / 1024**3,
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def host_mem_gib() -> dict[str, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        return {"used_gib": (m.total - m.available) / 1024**3, "free_gib": m.available / 1024**3}
    except Exception:
        return {"used_gib": -1.0, "free_gib": -1.0}


# ---------------------------------------------------------------------------
# AR_droid native camera-key mapping (matches what the model expects)
# ---------------------------------------------------------------------------
# LeRobot droid_1.0.1 uses slightly different image keys than droid_100.
# Auto-detect which dataset variant we're pointing at by checking the
# video directory layout.
CAMERA_KEY_MAP_DROID_100 = {
    "observation.images.exterior_image_1_left": "video.exterior_image_1_left",
    "observation.images.exterior_image_2_left": "video.exterior_image_2_left",
    "observation.images.wrist_image_left":      "video.wrist_image_left",
}
CAMERA_KEY_MAP_DROID_101 = {
    "observation.images.exterior_1_left": "video.exterior_image_1_left",
    "observation.images.exterior_2_left": "video.exterior_image_2_left",
    "observation.images.wrist_left":      "video.wrist_image_left",
}

RELATIVE_OFFSETS = (-23, -16, -8, 0)
ACTION_HORIZON = 24


# ---------------------------------------------------------------------------
# Dataset loader -- handles BOTH lerobot/droid_100 and lerobot/droid_1.0.1
# ---------------------------------------------------------------------------
class DroidDataset:
    """Lazy-loading view onto a downloaded LeRobot DROID dataset.

    Auto-detects between two variants:
      * droid_100 -- single-file layout (1 parquet, 1 mp4 per camera),
        7-dim composite ``action`` (Cartesian + gripper).
      * droid_1.0.1 -- sharded layout (156 data parquets, multi-file videos
        with timestamp-based access), 8-dim composite ``action`` (7 joint +
        1 gripper). This is what the DreamZero-DROID model actually emits.

    Per-episode metadata in droid_1.0.1 carries ``data/file_index`` and
    ``videos/<cam>/{file_index,from_timestamp,to_timestamp}`` columns that
    map an episode_index to its (parquet, video, timestamp window).
    """

    def __init__(self, root: Path) -> None:
        import pyarrow.parquet as pq
        import numpy as np
        import glob

        self.root = root
        self._gear = False
        # ---- GEAR per-episode layout (GEAR-Dreams/DreamZero-DROID-Data) --
        # data/chunk-*/episode_NNNNNN.parquet + videos/**/<cam>/episode_NNNNNN.mp4
        # + meta/episodes.jsonl + meta/modality.json. One episode per file, so
        # frame_index == local index. Detected before the droid_1.0.1 sharded
        # layout below.
        gear_data = sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet")))
        if gear_data and (root / "meta" / "episodes.jsonl").exists():
            self._init_gear(root, gear_data)
            return
        # ---- layout detection -------------------------------------------
        ep_meta_glob = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
        data_glob = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
        if not ep_meta_glob:
            raise FileNotFoundError(f"no meta/episodes parquets under {root}")
        if not data_glob:
            raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {root}")

        # Detect droid_101 if camera dirs are without "_image_" infix.
        v101_cams = list(CAMERA_KEY_MAP_DROID_101.keys())
        v100_cams = list(CAMERA_KEY_MAP_DROID_100.keys())
        v101_present = all((root / "videos" / k).exists() for k in v101_cams)
        v100_present = all((root / "videos" / k).exists() for k in v100_cams)
        if v101_present:
            self.variant = "droid_101"
            self.camera_key_map = CAMERA_KEY_MAP_DROID_101
        elif v100_present:
            self.variant = "droid_100"
            self.camera_key_map = CAMERA_KEY_MAP_DROID_100
        else:
            raise FileNotFoundError(f"could not detect camera layout under {root}/videos")
        print(f"[dataset] variant={self.variant}  meta_files={len(ep_meta_glob)}  data_files={len(data_glob)}")

        # ---- read + concat ALL episode-meta parquets --------------------
        ep_tables = [pq.read_table(p) for p in ep_meta_glob]
        import pyarrow as pa
        self.ep_table = pa.concat_tables(ep_tables)
        ep_index = self.ep_table.column("episode_index").to_numpy()
        from_idx = self.ep_table.column("dataset_from_index").to_numpy()
        to_idx = self.ep_table.column("dataset_to_index").to_numpy()
        length = self.ep_table.column("length").to_numpy()
        tasks_col = self.ep_table.column("tasks")
        schema_names = set(self.ep_table.schema.names)
        has_data_file_idx = "data/file_index" in schema_names
        has_data_chunk_idx = "data/chunk_index" in schema_names
        if has_data_file_idx:
            data_chunk = self.ep_table.column("data/chunk_index").to_numpy() if has_data_chunk_idx else np.zeros(len(ep_index), np.int32)
            data_file = self.ep_table.column("data/file_index").to_numpy()
        else:
            data_chunk = np.zeros(len(ep_index), np.int32)
            data_file = np.zeros(len(ep_index), np.int32)

        # collect per-camera video file + timestamps if present
        video_col_keys = {}
        for cam in self.camera_key_map:
            prefix = f"videos/{cam}"
            if f"{prefix}/file_index" in schema_names:
                video_col_keys[cam] = {
                    "chunk_idx": self.ep_table.column(f"{prefix}/chunk_index").to_numpy(),
                    "file_idx":  self.ep_table.column(f"{prefix}/file_index").to_numpy(),
                    "from_ts":   self.ep_table.column(f"{prefix}/from_timestamp").to_numpy(),
                    "to_ts":     self.ep_table.column(f"{prefix}/to_timestamp").to_numpy(),
                }

        self._ep_info: dict[int, dict[str, Any]] = {}
        for i in range(len(ep_index)):
            tasks_list = tasks_col[i].as_py() or []
            ep_dict: dict[str, Any] = {
                "global_from": int(from_idx[i]),
                "global_to": int(to_idx[i]),
                "length": int(length[i]),
                "task": tasks_list[0] if tasks_list else "",
                "data_chunk_idx": int(data_chunk[i]),
                "data_file_idx": int(data_file[i]),
            }
            per_cam = {}
            for cam, cols in video_col_keys.items():
                per_cam[cam] = {
                    "chunk_idx": int(cols["chunk_idx"][i]),
                    "file_idx":  int(cols["file_idx"][i]),
                    "from_ts":   float(cols["from_ts"][i]),
                    "to_ts":     float(cols["to_ts"][i]),
                }
            ep_dict["videos"] = per_cam
            self._ep_info[int(ep_index[i])] = ep_dict
        print(f"[dataset] {len(self._ep_info)} episodes indexed from {root}")

        # cached open handles (lazy)
        self._data_cache: dict[tuple[int, int], Any] = {}     # (chunk,file) -> pyarrow Table
        self._action_dim_cache: dict[tuple[int, int], int] = {}

        # for backward compat (droid_100 path that reads from one parquet)
        # we always go through _get_data_table now.

    def list_episodes(self) -> list[int]:
        return sorted(self._ep_info.keys())

    def episode_info(self, episode_index: int) -> dict[str, Any]:
        if episode_index not in self._ep_info:
            raise KeyError(f"episode {episode_index} not found")
        return self._ep_info[episode_index]

    def _get_data_table(self, chunk_idx: int, file_idx: int):
        import pyarrow.parquet as pq
        key = (chunk_idx, file_idx)
        if key in self._data_cache:
            return self._data_cache[key]
        p = self.root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        cols = [
            "episode_index", "frame_index", "index",
            "action", "observation.state",
        ]
        # only request action.joint_position / action.gripper_position when they exist
        try:
            schema = pq.read_schema(p)
            schema_names = set(schema.names)
            for extra in ("action.joint_position", "action.gripper_position",
                          "observation.state.joint_position",
                          "observation.state.gripper_position"):
                if extra in schema_names:
                    cols.append(extra)
        except Exception:
            pass
        tbl = pq.read_table(p, columns=cols)
        self._data_cache[key] = tbl
        # cap cache to 4 tables to bound RAM (each is ~70-90 MB)
        if len(self._data_cache) > 4:
            # evict oldest
            evict_key = next(iter(self._data_cache))
            if evict_key != key:
                self._data_cache.pop(evict_key, None)
        return tbl

    def read_actions(self, episode_index: int, local_start: int, count: int):
        """Return (count, action_dim) numpy array. action_dim is 7 for
        droid_100 (Cartesian + gripper) or 8 for droid_101 (7 joint + 1
        gripper). The `action` column in each parquet already has the right
        composite layout for the dataset variant.
        """
        if self._gear:
            return self._gear_read_actions(episode_index, local_start, count)
        import numpy as np
        info = self.episode_info(episode_index)
        if local_start < 0 or local_start + count > info["length"]:
            raise IndexError(
                f"episode {episode_index} length={info['length']}, requested "
                f"local frames [{local_start}, {local_start+count})"
            )
        tbl = self._get_data_table(info["data_chunk_idx"], info["data_file_idx"])
        # rows for this episode are contiguous in [global_from, global_to)
        # but the parquet's `index` column is global, so we can filter / slice
        idx_arr = tbl.column("index").to_numpy()
        gstart = info["global_from"] + local_start
        # find rows where idx in [gstart, gstart+count)
        mask = (idx_arr >= gstart) & (idx_arr < gstart + count)
        sel = np.nonzero(mask)[0]
        if len(sel) != count:
            raise RuntimeError(
                f"ep {episode_index}: requested {count} action rows starting "
                f"global={gstart}, got {len(sel)} from "
                f"chunk-{info['data_chunk_idx']:03d}/file-{info['data_file_idx']:03d}.parquet"
            )
        # rebuild in order of `index`
        order = np.argsort(idx_arr[sel])
        sel = sel[order]
        actions = []
        for i in sel:
            actions.append(np.asarray(tbl.column("action")[int(i)].as_py(), dtype=np.float32))
        return np.stack(actions, axis=0)

    def read_state(self, episode_index: int, local_index: int):
        if self._gear:
            return self._gear_read_state(episode_index, local_index)
        import numpy as np
        info = self.episode_info(episode_index)
        if local_index < 0 or local_index >= info["length"]:
            raise IndexError(f"state local index {local_index} out of range for ep {episode_index}")
        tbl = self._get_data_table(info["data_chunk_idx"], info["data_file_idx"])
        idx_arr = tbl.column("index").to_numpy()
        target = info["global_from"] + local_index
        rows = np.nonzero(idx_arr == target)[0]
        if not len(rows):
            raise RuntimeError(f"state for ep {episode_index} local {local_index} not found")
        return np.asarray(tbl.column("observation.state")[int(rows[0])].as_py(), dtype=np.float32)

    def read_frames(self, episode_index: int, local_indices: list[int]) -> dict[str, "np.ndarray"]:
        """Read 1+ frames per camera. Returns dict mapping AR_droid native key
        to (T, H, W, 3) uint8 array (RGB).

        For droid_100: all episodes in one big mp4 -> decode-from-start to
        target global frame.
        For droid_101: each video file holds many episodes; we use the
        per-episode video timestamp window from the meta table to seek to
        the correct PTS, then decode to the desired sub-frame within the
        episode (        frame_rate=15).
        """
        if self._gear:
            return self._gear_read_frames(episode_index, local_indices)
        import av  # PyAV
        import numpy as np
        info = self.episode_info(episode_index)
        out: dict[str, "np.ndarray"] = {}
        fps = 15.0

        for lerobot_key, native_key in self.camera_key_map.items():
            cam_info = info["videos"].get(lerobot_key, {})
            file_idx = cam_info.get("file_idx", 0)
            chunk_idx = cam_info.get("chunk_idx", 0)
            from_ts = cam_info.get("from_ts", None)
            video_path = (self.root / "videos" / lerobot_key /
                          f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
            if not video_path.exists():
                raise FileNotFoundError(f"video {video_path} for ep {episode_index} cam {lerobot_key}")

            container = av.open(str(video_path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            if self.variant == "droid_100":
                # decode whole stream up to needed global frames
                global_indices = sorted(set(info["global_from"] + li for li in local_indices))
                max_g = max(global_indices)
                idx_set = set(global_indices)
                cur_g = -1
                collected: dict[int, "np.ndarray"] = {}
                try:
                    for frame in container.decode(stream):
                        cur_g += 1
                        if cur_g in idx_set:
                            collected[cur_g] = frame.to_ndarray(format="rgb24")
                        if cur_g >= max_g:
                            break
                finally:
                    container.close()
                requested = [info["global_from"] + li for li in local_indices]
                missing = [g for g in requested if g not in collected]
                if missing:
                    raise RuntimeError(f"missing global frames {missing} from {video_path}")
                out[native_key] = np.stack([collected[g] for g in requested], axis=0)
                continue

            # droid_101 path: each video file holds many episodes back-to-back.
            # The episode's video timestamps are [from_ts, to_ts) in seconds.
            # Strategy: seek to ~0.5s before from_ts, then decode forward; pick
            # the first frame with pts*time_base >= (from_ts + li/fps) - tol.
            if from_ts is None:
                container.close()
                raise RuntimeError(f"ep {episode_index} cam {lerobot_key}: no from_ts in meta")
            targets_sorted = sorted(set(local_indices))
            seek_ts = max(0.0, from_ts + targets_sorted[0] / fps - 0.5)
            seek_pts = int(seek_ts / float(stream.time_base))
            try:
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            except Exception as e:
                container.close()
                raise RuntimeError(f"seek to {seek_ts:.3f}s in {video_path} failed: {e!r}")
            tol = 0.5 / fps  # half a frame tolerance
            target_li_remaining = list(targets_sorted)
            collected_li: dict[int, "np.ndarray"] = {}
            try:
                for frame in container.decode(stream):
                    if frame.pts is None or not target_li_remaining:
                        if not target_li_remaining:
                            break
                        continue
                    ts_s = frame.pts * float(stream.time_base)
                    # consume every target whose target_ts <= current frame ts (+tol)
                    while target_li_remaining:
                        next_target_ts = from_ts + target_li_remaining[0] / fps
                        if ts_s + tol >= next_target_ts:
                            li = target_li_remaining.pop(0)
                            collected_li[li] = frame.to_ndarray(format="rgb24")
                        else:
                            break
            finally:
                container.close()
            if target_li_remaining:
                # try to be lenient: clip to last decoded frame if any
                if collected_li:
                    last_li = max(collected_li.keys())
                    last_arr = collected_li[last_li]
                    for li in target_li_remaining:
                        collected_li[li] = last_arr
                else:
                    raise RuntimeError(
                        f"ep {episode_index} cam {lerobot_key}: failed to decode "
                        f"any of local frames {targets_sorted} (from_ts={from_ts})")
            out[native_key] = np.stack([collected_li[li] for li in local_indices], axis=0)
        return out

    # -----------------------------------------------------------------------
    # GEAR per-episode layout (GEAR-Dreams/DreamZero-DROID-Data)
    # -----------------------------------------------------------------------
    def _init_gear(self, root, gear_data) -> None:
        import json
        import glob

        self._gear = True
        self.variant = "droid_gear"
        self.camera_key_map = CAMERA_KEY_MAP_DROID_100  # folders carry the _image_ infix

        # episode_index -> parquet path (from episode_NNNNNN.parquet filename)
        self.gear_files: dict[int, str] = {}
        for p in gear_data:
            stem = os.path.basename(p)
            try:
                ep = int(stem.split("_")[1].split(".")[0])
            except Exception:
                continue
            self.gear_files[ep] = p

        # per-episode meta (length, task) from episodes.jsonl, only for present eps
        self._ep_info: dict[int, dict[str, Any]] = {}
        with open(root / "meta" / "episodes.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ep = int(d["episode_index"])
                if ep not in self.gear_files:
                    continue
                tasks = d.get("tasks") or []
                ln = int(d.get("length", 0))
                self._ep_info[ep] = {
                    "length": ln,
                    "task": tasks[0] if tasks else "",
                    "global_from": 0,
                    "global_to": ln,
                    "data_chunk_idx": 0,
                    "data_file_idx": ep,
                    "videos": {},
                }

        # resolve per-camera video dirs (videos/**/<original_key folder>)
        vroot = root / "videos"
        self.gear_video_dirs: dict[str, str] = {}
        for lerobot_key in self.camera_key_map:
            cands = sorted(glob.glob(str(vroot / "**" / lerobot_key), recursive=True))
            if cands:
                self.gear_video_dirs[lerobot_key] = cands[0]

        # 8-dim model state = [joint_position(7), gripper_position(1)] sliced
        # out of the 14-dim observation.state via modality.json.
        self._gear_state_slice = {"joint": (7, 14), "gripper": (6, 7)}
        # 8-dim GT action = [joint_position(7), gripper_position(1)] out of the
        # 28-dim action vector.
        self._gear_action_slice = {"joint": (14, 21), "gripper": (12, 13)}
        try:
            with open(root / "meta" / "modality.json") as f:
                mod = json.load(f)
            st = mod.get("state", {})
            if "joint_position" in st:
                self._gear_state_slice["joint"] = (int(st["joint_position"]["start"]), int(st["joint_position"]["end"]))
            if "gripper_position" in st:
                self._gear_state_slice["gripper"] = (int(st["gripper_position"]["start"]), int(st["gripper_position"]["end"]))
            ac = mod.get("action", {})
            if "joint_position" in ac:
                self._gear_action_slice["joint"] = (int(ac["joint_position"]["start"]), int(ac["joint_position"]["end"]))
            if "gripper_position" in ac:
                self._gear_action_slice["gripper"] = (int(ac["gripper_position"]["start"]), int(ac["gripper_position"]["end"]))
        except Exception:
            pass

        self._gear_pq_cache: dict[int, Any] = {}
        print(f"[dataset] variant=droid_gear  episodes={len(self._ep_info)}  "
              f"cams={list(self.gear_video_dirs)}  "
              f"state_slice={self._gear_state_slice}")

    def _gear_table(self, ep: int):
        import pyarrow.parquet as pq
        if ep in self._gear_pq_cache:
            return self._gear_pq_cache[ep]
        if ep not in self.gear_files:
            raise KeyError(f"episode {ep} not present in GEAR data dir")
        t = pq.read_table(self.gear_files[ep])
        if len(self._gear_pq_cache) > 4:
            self._gear_pq_cache.pop(next(iter(self._gear_pq_cache)), None)
        self._gear_pq_cache[ep] = t
        return t

    def _gear_read_state(self, episode_index: int, local_index: int):
        import numpy as np
        t = self._gear_table(episode_index)
        st = np.asarray(t.column("observation.state")[int(local_index)].as_py(), dtype=np.float32)
        j0, j1 = self._gear_state_slice["joint"]
        g0, g1 = self._gear_state_slice["gripper"]
        return np.concatenate([st[j0:j1], st[g0:g1]]).astype(np.float32)  # (8,)

    def _gear_read_actions(self, episode_index: int, local_start: int, count: int):
        import numpy as np
        t = self._gear_table(episode_index)
        j0, j1 = self._gear_action_slice["joint"]
        g0, g1 = self._gear_action_slice["gripper"]
        rows = []
        for i in range(local_start, local_start + count):
            a = np.asarray(t.column("action")[int(i)].as_py(), dtype=np.float32)
            rows.append(np.concatenate([a[j0:j1], a[g0:g1]]))
        return np.stack(rows, axis=0)  # (count, 8)

    def _gear_read_frames(self, episode_index: int, local_indices):
        import av
        import numpy as np
        out: dict[str, "np.ndarray"] = {}
        want = sorted({int(i) for i in local_indices})
        maxi = max(want)
        for lerobot_key, native_key in self.camera_key_map.items():
            vdir = self.gear_video_dirs.get(lerobot_key)
            if vdir is None:
                continue
            mp4 = os.path.join(vdir, f"episode_{episode_index:06d}.mp4")
            if not os.path.exists(mp4):
                raise FileNotFoundError(f"video {mp4} for ep {episode_index} cam {lerobot_key}")
            container = av.open(mp4)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            collected: dict[int, "np.ndarray"] = {}
            cur = -1
            try:
                for frame in container.decode(stream):
                    cur += 1
                    if cur in want:
                        collected[cur] = frame.to_ndarray(format="rgb24")
                    if cur >= maxi:
                        break
            finally:
                container.close()
            if not collected:
                raise RuntimeError(f"no frames decoded from {mp4}")
            last = collected[max(collected)]
            out[native_key] = np.stack(
                [collected.get(int(li), last) for li in local_indices], axis=0)
        return out


# ---------------------------------------------------------------------------
# Stage-2 / Path-A patch suite (re-applied here). Keep in sync.
# ---------------------------------------------------------------------------
def apply_amd_patches(policy):
    import torch
    import tree as _tree

    banner("Applying AMD ROCm patches (10 patches; same set as Stage 2 v16)")

    vae = policy.trained_model.action_head.vae
    _orig_vae_encode_unbound = type(vae).encode

    def _tiled_vae_encode(self_vae, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        return _orig_vae_encode_unbound(self_vae, videos, tiled=True,
                                        tile_size=tile_size, tile_stride=tile_stride)
    vae.encode = types.MethodType(_tiled_vae_encode, vae)
    print("[patch] vae.encode -> always tiled=True")

    _orig_lazy_causal = type(policy.trained_model).lazy_joint_video_action_causal

    def _cuda_hoisted_lazy_causal(self_vla, inputs, latent_video=None):
        moved = 0
        if isinstance(inputs, dict):
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.device.type == "cpu":
                    inputs[k] = v.to("cuda", non_blocking=True)
                    moved += 1
        if latent_video is not None and torch.is_tensor(latent_video) and latent_video.device.type == "cpu":
            latent_video = latent_video.to("cuda", non_blocking=True)
            moved += 1
        if moved:
            print(f"  [patch] outer hoist: moved {moved} input tensor(s) cpu->cuda")
        return _orig_lazy_causal(self_vla, inputs, latent_video=latent_video)

    type(policy.trained_model).lazy_joint_video_action_causal = _cuda_hoisted_lazy_causal
    print("[patch] VLA.lazy_joint_video_action_causal wrapped")

    def _cuda_prepare_input(self_vla, inputs):
        self_vla.validate_inputs(inputs)
        backbone_inputs = self_vla.backbone.prepare_input(inputs)
        action_inputs = self_vla.action_head.prepare_input(inputs)
        target_device = torch.device("cuda")
        target_dtype = self_vla.action_head.dtype

        def to_cuda(x):
            if torch.is_floating_point(x):
                return x.to(target_device, dtype=target_dtype)
            return x.to(target_device)

        backbone_inputs = _tree.map_structure(to_cuda, backbone_inputs)
        action_inputs = _tree.map_structure(to_cuda, action_inputs)
        return backbone_inputs, action_inputs

    type(policy.trained_model).prepare_input = _cuda_prepare_input
    print("[patch] VLA.prepare_input overridden -> forced CUDA")

    action_head = policy.trained_model.action_head
    _orig_encode_prompt = type(action_head).encode_prompt
    _orig_encode_image = type(action_head).encode_image

    state: dict[str, Any] = {"text_encode_calls": 0, "image_encode_calls": 0}
    TEXT_PER_CHUNK = 2

    def _encode_prompt_wrapped(self_ah, input_ids, attention_mask):
        try:
            te_dev = next(self_ah.text_encoder.parameters()).device
        except StopIteration:
            te_dev = torch.device("cuda")
        if te_dev.type == "cpu":
            self_ah.text_encoder.to("cuda")
        if torch.is_tensor(input_ids) and input_ids.device.type == "cpu":
            input_ids = input_ids.to("cuda", non_blocking=True)
        if torch.is_tensor(attention_mask) and attention_mask.device.type == "cpu":
            attention_mask = attention_mask.to("cuda", non_blocking=True)
        state["text_encode_calls"] += 1
        out = _orig_encode_prompt(self_ah, input_ids, attention_mask)
        if state["text_encode_calls"] % TEXT_PER_CHUNK == 0:
            try:
                self_ah.text_encoder.to("cpu")
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            except Exception as e:
                print(f"  [offload] text_encoder offload failed: {e!r}")
        return out

    def _encode_image_wrapped(self_ah, image, num_frames, height, width):
        try:
            ie_dev = next(self_ah.image_encoder.parameters()).device
        except StopIteration:
            ie_dev = torch.device("cuda")
        if ie_dev.type == "cpu":
            self_ah.image_encoder.to("cuda")
        if torch.is_tensor(image) and image.device.type == "cpu":
            image = image.to("cuda", non_blocking=True)
        state["image_encode_calls"] += 1
        print(f"  [encode_image #{state['image_encode_calls']}] image.shape={tuple(image.shape)}")
        out = _orig_encode_image(self_ah, image, num_frames, height, width)
        try:
            self_ah.image_encoder.to("cpu")
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        except Exception as e:
            print(f"  [offload] image_encoder offload failed: {e!r}")
        return out

    type(action_head).encode_prompt = _encode_prompt_wrapped
    type(action_head).encode_image = _encode_image_wrapped
    print("[patch] encode_prompt/encode_image wrapped")
    return state


# ---------------------------------------------------------------------------
# unpack_action_chunk -- shared with Path A
# ---------------------------------------------------------------------------
def unpack_action_chunk(act):
    import numpy as np
    import torch
    from tianshou.data import Batch

    def to_np(v):
        if hasattr(v, "detach"):
            t = v.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(torch.float32)
            return t.numpy()
        return np.asarray(v)

    if isinstance(act, Batch) or hasattr(act, "keys"):
        flat: dict[str, Any] = {}
        for k in act.keys():
            v = act[k]
            if isinstance(v, Batch) and hasattr(v, "keys"):
                for k2 in v.keys():
                    flat[f"{k}.{k2}"] = v[k2]
            else:
                flat[k] = v
        per_key = {k: to_np(v) for k, v in flat.items()}
    elif isinstance(act, np.ndarray) and act.dtype == object and act.size > 0:
        first = act.flat[0]
        per_key = {}
        if hasattr(first, "keys"):
            for elem in act.flat:
                for k in elem.keys():
                    per_key.setdefault(k, []).append(to_np(elem[k]))
            per_key = {k: np.stack(lst, axis=0) for k, lst in per_key.items()}
        else:
            per_key = {"action": to_np(act)}
    else:
        per_key = {"action": to_np(act)}

    joint = next((v for k, v in per_key.items() if "joint_position" in k and isinstance(v, np.ndarray)), None)
    gripper = next((v for k, v in per_key.items() if "gripper_position" in k and isinstance(v, np.ndarray)), None)
    if joint is not None:
        j = joint.astype(np.float32)
        while j.ndim > 2 and j.shape[0] == 1:
            j = j[0]
        if j.ndim == 1:
            j = j.reshape(-1, 1)
        if gripper is not None:
            g = gripper.astype(np.float32)
            while g.ndim > 2 and g.shape[0] == 1:
                g = g[0]
            if g.ndim == 1:
                g = g.reshape(-1, 1)
        else:
            g = np.zeros(j.shape[:-1] + (1,), dtype=np.float32)
        if g.shape[:-1] != j.shape[:-1]:
            g = np.broadcast_to(g, j.shape[:-1] + (g.shape[-1],)).copy()
        primary = np.concatenate([j, g], axis=-1)
    else:
        primary = next((v for v in per_key.values() if isinstance(v, np.ndarray) and v.dtype.kind == "f"), None)
        if primary is None:
            raise RuntimeError(f"no float action array found in {list(per_key.keys())}")
        primary = primary.astype(np.float32)
    return primary, per_key


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_chunk_metrics(pred: "np.ndarray", gt: "np.ndarray", *, variant: str) -> dict[str, Any]:
    """Compute MSE/cosine between predicted action chunk and ground truth.

    For ``variant == "droid_101"`` both arrays are (24, 8) in the same action
    space: dims 0..6 are Franka joint positions (radians), dim 7 is gripper
    [0, 1]. Direct element-wise comparison is meaningful.

    For ``variant == "droid_100"`` pred is (24, 8) but GT is (24, 7) and the
    first 6 dims live in different physical spaces (joint vs Cartesian), so
    only the gripper has a meaningful comparison. We still report the legacy
    "arm" RMSE (pred[:, :6] vs gt[:, :6]) but flag it as space-mismatched.
    """
    import numpy as np
    assert pred.shape == (24, 8), f"pred shape {pred.shape}"

    if variant == "droid_101":
        assert gt.shape == (24, 8), f"gt shape {gt.shape} (expected (24,8) for droid_101)"
        n_arm = 7
        pred_arm = pred[:, :n_arm]
        gt_arm = gt[:, :n_arm]
        pred_gripper = pred[:, 7]
        gt_gripper = gt[:, 7]
        space_label = "joint_position (rad)"
    else:
        assert gt.shape == (24, 7), f"gt shape {gt.shape}"
        n_arm = 6
        pred_arm = pred[:, :n_arm]   # NOTE: pred is joint, gt is Cartesian
        gt_arm = gt[:, :n_arm]
        pred_gripper = pred[:, 7]
        gt_gripper = gt[:, 6]
        space_label = "MIXED: pred=joint vs GT=Cartesian (incommensurable)"

    err = pred_arm - gt_arm
    mse_per_dim = (err ** 2).mean(axis=0)                 # (n_arm,)
    mse_total = float((err ** 2).mean())
    rmse_per_dim = np.sqrt(mse_per_dim)

    # per-step cosine similarity over the arm dims
    cos = np.zeros(24, dtype=np.float32)
    for t in range(24):
        a = pred_arm[t]; b = gt_arm[t]
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            cos[t] = float(np.dot(a, b) / (na * nb))
        else:
            cos[t] = 1.0 if na == nb else 0.0

    # also report per-dim Pearson correlation across the 24-step trajectory
    # (catches phase / scale issues that MSE can hide)
    pearson_per_dim = []
    for d in range(n_arm):
        p = pred_arm[:, d]; g = gt_arm[:, d]
        if p.std() < 1e-9 or g.std() < 1e-9:
            pearson_per_dim.append(0.0)
        else:
            pearson_per_dim.append(float(np.corrcoef(p, g)[0, 1]))

    pred_gripper_bin = (pred_gripper > 0.5).astype(np.int32)
    gt_gripper_bin = (gt_gripper > 0.5).astype(np.int32)
    gripper_acc = float((pred_gripper_bin == gt_gripper_bin).mean())
    gripper_mse = float(((pred_gripper - gt_gripper) ** 2).mean())

    # raw signal stats for debugging normalization issues
    pred_arm_range = [float(pred_arm.min()), float(pred_arm.max())]
    gt_arm_range = [float(gt_arm.min()), float(gt_arm.max())]
    pred_arm_per_dim_std = pred_arm.std(0).tolist()
    gt_arm_per_dim_std = gt_arm.std(0).tolist()

    return {
        "space_label": space_label,
        "n_arm_dims": n_arm,
        "mse_arm_total": mse_total,
        "rmse_arm_total": float(np.sqrt(mse_total)),
        "mse_per_dim_arm": mse_per_dim.tolist(),
        "rmse_per_dim_arm": rmse_per_dim.tolist(),
        "pearson_per_dim_arm": pearson_per_dim,
        "cos_per_step_mean": float(cos.mean()),
        "cos_per_step_min": float(cos.min()),
        "cos_per_step_max": float(cos.max()),
        "gripper_accuracy": gripper_acc,
        "gripper_mse": gripper_mse,
        "pred_arm_range": pred_arm_range,
        "gt_arm_range": gt_arm_range,
        "pred_arm_per_dim_std": pred_arm_per_dim_std,
        "gt_arm_per_dim_std": gt_arm_per_dim_std,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    import numpy as np
    t_start = time.perf_counter()

    output_dir = Path(os.environ.get("OUTPUT_DIR", "/artifacts/stage3_path_b"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.environ["MODEL_PATH"]
    embodiment = os.environ.get("EMBODIMENT_TAG", "OXE_DROID")
    dataset_dir = Path(os.environ.get("DROID_DATASET_DIR", "/data/lerobot_droid_100"))
    episodes_str = os.environ.get("EPISODES", "5,10,20")
    K = int(os.environ.get("CHUNKS_PER_EPISODE", "3"))
    episode_list = [int(s) for s in episodes_str.split(",") if s.strip()]

    banner("Stage 3 Path B -- numeric validation vs GT actions")
    print(f"[cfg] MODEL_PATH       = {model_path}")
    print(f"[cfg] DROID_DATASET    = {dataset_dir}")
    print(f"[cfg] EPISODES         = {episode_list}")
    print(f"[cfg] CHUNKS_PER_EP    = {K}")
    print(f"[cfg] OUTPUT_DIR       = {output_dir}")

    mem_trace: list[dict[str, Any]] = []

    def probe(label: str) -> None:
        h = host_mem_gib(); c = cuda_mem_gib()
        print(f"[mem] {label:<30} RAM used={h['used_gib']:.1f} GiB  "
              f"VRAM alloc={c['alloc_gib']:.2f} GiB  max={c['max_alloc_gib']:.2f} GiB  "
              f"reserved={c['reserved_gib']:.2f} GiB", flush=True)
        mem_trace.append({"label": label, "ts": time.perf_counter() - t_start, **h, **c})

    probe("00_script_start")

    # ---- Disable torch.compile / dynamo BEFORE importing dreamzero ---------
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch
        import torch.distributed as dist
        import torch._dynamo
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.reset()
        print("[dynamo] disabled")
    except Exception as exc:
        print(f"[FATAL] base import failed: {exc!r}")
        return 2

    if not os.path.isdir(model_path):
        print(f"[FATAL] checkpoint dir not found: {model_path}")
        return 11
    if not dataset_dir.is_dir():
        print(f"[FATAL] dataset dir not found: {dataset_dir}")
        return 12

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[gpu] {torch.cuda.get_device_name(0)}  arch={p.gcnArchName}  total={p.total_memory/1024**3:.1f} GiB")
    else:
        print("[FATAL] CUDA not available")
        return 13
    probe("01_post_torch_init")

    # ---- Open dataset early so we fail fast if it's broken -----------------
    banner("Opening DROID dataset")
    try:
        ds = DroidDataset(dataset_dir)
    except Exception as exc:
        print(f"[FATAL] dataset open failed: {exc!r}")
        traceback.print_exc()
        return 14

    for e in episode_list:
        info = ds.episode_info(e)
        print(f"  ep {e:>3}  frames=[{info['global_from']}, {info['global_to']})  "
              f"len={info['length']}  task={info['task']!r}")

    # ---- Model imports + patches that must run BEFORE load ------------------
    banner("Importing model classes")
    try:
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
            WANPolicyHead as _WANPolicyHead,
        )
        from tianshou.data import Batch
        from safetensors.torch import load_file as _load_safetensors
    except Exception as exc:
        print(f"[FATAL] model import failed: {exc!r}")
        traceback.print_exc()
        return 3
    probe("02_post_model_imports")

    @classmethod
    def _amd_low_mem_from_pretrained(cls, pretrained_model_name_or_path, config=None):
        del config
        cfg_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        if isinstance(cfg_dict.get("action_head_cfg", {}).get("config"), dict):
            cfg_dict["action_head_cfg"]["config"]["defer_lora_injection"] = False
        cfg = VLAConfig(**cfg_dict)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            t0 = time.perf_counter()
            model = cls(cfg)
            print(f"[low-mem] empty bf16 model built in {time.perf_counter()-t0:.1f} s")
        finally:
            torch.set_default_dtype(prev_dtype)
        st_idx = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
        st_one = os.path.join(pretrained_model_name_or_path, "model.safetensors")

        def _apply(sd):
            if any(".base_layer." in k for k in sd):
                sd = {k.replace(".base_layer.", "."): v for k, v in sd.items()}
            return model.load_state_dict(sd, strict=False)

        if os.path.exists(st_idx):
            with open(st_idx) as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            for i, shard in enumerate(shards):
                t0 = time.perf_counter()
                sd = _load_safetensors(os.path.join(pretrained_model_name_or_path, shard))
                _apply(sd); del sd; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[low-mem] shard {i+1}/{len(shards)} {shard} in {time.perf_counter()-t0:.1f}s")
        elif os.path.exists(st_one):
            sd = _load_safetensors(st_one)
            _apply(sd); del sd; gc.collect()
        else:
            raise FileNotFoundError(f"No safetensors at {pretrained_model_name_or_path}")
        return model

    VLA.from_pretrained = _amd_low_mem_from_pretrained

    def _amd_post_initialize(self_ah):
        print("[patch] post_initialize: move to cuda+bf16, skip torch.compile")
        self_ah.model.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.text_encoder.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.image_encoder.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.vae.to(device=self_ah._device, dtype=torch.bfloat16)
        self_ah.trt_engine = None
    _WANPolicyHead.post_initialize = _amd_post_initialize

    # ---- Policy load -------------------------------------------------------
    banner("Loading GrootSimPolicy")
    probe("03_pre_policy_load")
    t_load = time.perf_counter()
    try:
        policy = GrootSimPolicy(
            embodiment_tag=getattr(EmbodimentTag, embodiment),
            model_path=model_path,
            device="cuda",
        )
    except Exception as exc:
        print(f"[FATAL] policy load failed: {exc!r}")
        traceback.print_exc()
        return 4
    load_dt = time.perf_counter() - t_load
    print(f"[ok] policy loaded in {load_dt:.1f} s")
    probe("04_post_policy_load")

    patch_state = apply_amd_patches(policy)
    probe("05_post_patches")

    # ---- Stage 4 perf overlay (optional) ----------------------------------
    # ENABLE_FLASH_SDPA=1 turns on aotriton flash + mem-efficient SDPA (2.6x
    # DiT speedup proven in Stage 4). DENOISE_STEPS overrides the video
    # branch num_inference_steps (default 16; Stage 4 verified 4 and 2 still
    # produce a valid output shape). Both default to off so Stage 3 keeps
    # bit-exact behaviour unless explicitly opted in.
    if os.environ.get("ENABLE_FLASH_SDPA", "0") == "1":
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[perf] flash + mem-efficient SDPA enabled (Stage 4 overlay)")
        except Exception as e:
            print(f"[perf] enable_flash_sdp failed: {e!r}")
    denoise_override = os.environ.get("DENOISE_STEPS", "").strip()
    if denoise_override:
        try:
            n = int(denoise_override)
            old = int(getattr(policy.trained_model.action_head, "num_inference_steps", 16))
            policy.trained_model.action_head.num_inference_steps = n
            print(f"[perf] num_inference_steps {old} -> {n} (DENOISE_STEPS env)")
        except Exception as e:
            print(f"[perf] DENOISE_STEPS override failed: {e!r}")

    # ---- Inference loop ----------------------------------------------------
    banner(f"Running validation: episodes={episode_list}, K={K} chunks each")
    per_chunk_rows: list[dict[str, Any]] = []
    all_actions_dict: dict[str, "np.ndarray"] = {}

    cuda_mem_gib(reset=True)
    n_successful = 0
    n_skipped = 0

    for ep_idx, ep in enumerate(episode_list):
        # Aggressive between-episode VRAM purge. Each new episode triggers
        # a language change in the model, which re-encodes the prompt and
        # forces the text_encoder back to CUDA (~9 GiB transient). Without
        # an explicit cleanup the residual KV cache + intermediate buffers
        # from the prior episode push us over the 47 GiB iGPU ceiling
        # (observed: 46.46 GiB allocated, 8 MiB free at ep 13 entry).
        if ep_idx > 0:
            try:
                ah = policy.trained_model.action_head
                # Only purge text_encoder -- our encode_prompt wrapper will
                # bring it back to CUDA on the next call. image_encoder and
                # vae have their own offload/rehome lifecycles already
                # (or no rehome wrapper at all -- safer to leave alone).
                if hasattr(ah, "text_encoder"):
                    try:
                        ah.text_encoder.cpu()
                    except Exception:
                        pass
                # nuke per-session state that pins KV cache references.
                # IMPORTANT: do NOT clear `current_start_frame` -- the model
                # uses it as an int (>=0). Setting None breaks comparison.
                # Setting language=None forces re-encode (intended for new ep).
                # Correct upstream attribute names from
                # wan_flow_matching_action_tf.WANPolicyHead.__init__:
                # kv_cache1, kv_cache_neg, clip_feas, ys, language,
                # current_start_frame. The older list (kv_cache /
                # _kv_cache / cached_*) was a no-op.
                for _attr in ("language", "clip_feas", "ys",
                              "kv_cache1", "kv_cache_neg"):
                    if hasattr(ah, _attr):
                        try:
                            setattr(ah, _attr, None)
                        except Exception:
                            pass
                if hasattr(ah, "current_start_frame"):
                    try:
                        ah.current_start_frame = 0
                    except Exception:
                        pass
            except Exception as e:
                print(f"[mem-purge] partial failure: {e!r}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cm = torch.cuda.memory_allocated() / 1024**3
                print(f"[mem-purge] post ep {episode_list[ep_idx-1]} -> ep {ep}: "
                      f"VRAM allocated = {cm:.2f} GiB", flush=True)

        info = ds.episode_info(ep)
        ep_len = info["length"]
        prompt = info["task"] or "Perform the demonstrated task."
        # Need >= 24 frames after the chunk start for GT, and >= 23 frames
        # before for the input window. So valid local chunk_start positions:
        #   chunk_start in [23, ep_len - 24]
        valid_start_min = 23
        valid_start_max = ep_len - 24 - 1  # need 24 GT frames AFTER
        if valid_start_max < valid_start_min:
            print(f"[skip] ep {ep} too short (len={ep_len})")
            n_skipped += K
            continue
        # K evenly-spaced anchor positions in [valid_start_min, valid_start_max]
        if K == 1:
            anchors = [valid_start_min]
        else:
            step = max(1, (valid_start_max - valid_start_min) // (K - 1))
            anchors = [valid_start_min + i * step for i in range(K)]
            anchors = [min(a, valid_start_max) for a in anchors]
        print(f"\n[ep {ep}] len={ep_len}  prompt={prompt!r}\n[ep {ep}] anchors (local frames) = {anchors}")

        for k_idx, anchor in enumerate(anchors):
            chunk_tag = f"ep{ep:03d}_c{k_idx:02d}"
            input_indices = [max(anchor + off, 0) for off in RELATIVE_OFFSETS]
            try:
                frames_dict = ds.read_frames(ep, input_indices)
            except Exception as exc:
                print(f"[skip] {chunk_tag}: read_frames failed: {exc!r}")
                n_skipped += 1
                continue

            obs: dict[str, Any] = dict(frames_dict)
            # CRITICAL: pass the ACTUAL joint+gripper state at the anchor
            # frame. The model's Untransform step un-normalizes its output
            # using `last_state`, so passing zeros biases predictions toward
            # the model's mean-centered normalized space (offset by ~ -state
            # on every dim). For droid_101 we read the state from
            # obs.state.joint_position; for droid_100 the combined 8-dim
            # state is split: dims 0..6 are joint, dim 7 is gripper.
            try:
                anchor_state_8 = ds.read_state(ep, anchor)
                if anchor_state_8.shape == (8,):
                    js = anchor_state_8[:7].reshape(1, 7).astype(np.float64)
                    gs = anchor_state_8[7:8].reshape(1, 1).astype(np.float64)
                elif anchor_state_8.shape == (7,):
                    js = anchor_state_8.reshape(1, 7).astype(np.float64)
                    gs = np.zeros((1, 1), dtype=np.float64)
                else:
                    raise RuntimeError(f"unexpected state shape {anchor_state_8.shape}")
            except Exception as exc:
                print(f"[warn] {chunk_tag}: anchor state read failed: {exc!r}; using zeros")
                js = np.zeros((1, 7), dtype=np.float64)
                gs = np.zeros((1, 1), dtype=np.float64)
            obs["state.joint_position"] = js
            obs["state.gripper_position"] = gs
            obs["annotation.language.action_text"] = prompt
            print(f"    [obs] state.joint_position = {[round(x,3) for x in js[0].tolist()]}", flush=True)

            t0 = time.perf_counter()
            try:
                batch_in = Batch(obs=obs)
                result = policy.lazy_joint_forward_causal(batch_in)
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[FATAL] {chunk_tag}: inference failed: {exc!r}")
                traceback.print_exc()
                # save partials
                _save_partial(output_dir, per_chunk_rows, all_actions_dict)
                return 5
            chunk_dt = time.perf_counter() - t0
            batch_out = result[0] if isinstance(result, tuple) else result
            act = getattr(batch_out, "act", None)
            if act is None:
                print(f"[FATAL] {chunk_tag}: no .act attr")
                return 6
            try:
                pred, _ = unpack_action_chunk(act)
            except Exception as exc:
                print(f"[FATAL] {chunk_tag}: unpack failed: {exc!r}")
                return 6
            if pred.shape != (24, 8):
                print(f"[FATAL] {chunk_tag}: unexpected shape {pred.shape}")
                return 6

            # ground truth: actions for frames [anchor+1, anchor+24] inclusive
            try:
                gt = ds.read_actions(ep, anchor + 1, 24)
            except Exception as exc:
                print(f"[FATAL] {chunk_tag}: GT read failed: {exc!r}")
                return 14
            expected_gt_dim = 8 if ds.variant == "droid_101" else 7
            if gt.shape != (24, expected_gt_dim):
                print(f"[FATAL] {chunk_tag}: GT shape {gt.shape} (expected (24,{expected_gt_dim}))")
                return 14

            metrics = compute_chunk_metrics(pred, gt, variant=ds.variant)
            all_actions_dict[f"{chunk_tag}_pred"] = pred
            all_actions_dict[f"{chunk_tag}_gt"] = gt
            row = {
                "episode": ep,
                "chunk_index": k_idx,
                "local_anchor": anchor,
                "input_frames": list(input_indices),
                "gt_frames": list(range(anchor + 1, anchor + 25)),
                "dt_s": round(chunk_dt, 2),
                "prompt": prompt,
                **metrics,
            }
            per_chunk_rows.append(row)
            n_successful += 1
            print(f"  [{chunk_tag}] dt={chunk_dt:.1f}s  anchor={anchor}  "
                  f"MSE_arm={metrics['mse_arm_total']:.4f}  "
                  f"RMSE_arm={metrics['rmse_arm_total']:.4f}  "
                  f"cos_mean={metrics['cos_per_step_mean']:.4f}  "
                  f"gripper_acc={metrics['gripper_accuracy']:.4f}", flush=True)
            print(f"    pred_arm range={metrics['pred_arm_range']}  "
                  f"gt_arm range={metrics['gt_arm_range']}", flush=True)
            print(f"    rmse_per_dim={[round(v,4) for v in metrics['rmse_per_dim_arm']]}", flush=True)
            print(f"    pearson_per_dim={[round(v,4) for v in metrics['pearson_per_dim_arm']]}", flush=True)
            c = cuda_mem_gib()
            print(f"  [mem] VRAM alloc={c['alloc_gib']:.2f}/{c['max_alloc_gib']:.2f} GiB", flush=True)

    probe("06_post_validation")

    if not per_chunk_rows:
        print("[FATAL] no successful chunks; nothing to aggregate")
        return 6

    # ---- Aggregate ---------------------------------------------------------
    banner("Aggregating")
    arm_mse_total = float(np.mean([r["mse_arm_total"] for r in per_chunk_rows]))
    arm_rmse_total = float(np.mean([r["rmse_arm_total"] for r in per_chunk_rows]))
    cos_mean_mean = float(np.mean([r["cos_per_step_mean"] for r in per_chunk_rows]))
    cos_mean_min = float(np.min([r["cos_per_step_mean"] for r in per_chunk_rows]))
    grip_acc_mean = float(np.mean([r["gripper_accuracy"] for r in per_chunk_rows]))

    per_dim_mse = np.stack([np.asarray(r["mse_per_dim_arm"]) for r in per_chunk_rows]).mean(0).tolist()

    # per-episode summary
    ep_summary: dict[int, dict[str, Any]] = {}
    for ep in episode_list:
        rows = [r for r in per_chunk_rows if r["episode"] == ep]
        if not rows:
            continue
        ep_summary[ep] = {
            "n_chunks": len(rows),
            "arm_mse_mean": float(np.mean([r["mse_arm_total"] for r in rows])),
            "arm_rmse_mean": float(np.mean([r["rmse_arm_total"] for r in rows])),
            "cos_mean": float(np.mean([r["cos_per_step_mean"] for r in rows])),
            "gripper_acc": float(np.mean([r["gripper_accuracy"] for r in rows])),
            "mean_chunk_dt_s": float(np.mean([r["dt_s"] for r in rows])),
            "prompt": rows[0]["prompt"],
        }

    # ---- Save artifacts ----------------------------------------------------
    banner("Saving artifacts")
    np.savez(output_dir / "per_chunk_actions.npz", **all_actions_dict)
    print(f"[save] per_chunk_actions.npz ({len(all_actions_dict)} arrays)")

    # per_chunk.csv
    if per_chunk_rows:
        keys = ["episode", "chunk_index", "local_anchor", "input_frames", "gt_frames",
                "dt_s", "space_label", "n_arm_dims", "mse_arm_total", "rmse_arm_total",
                "cos_per_step_mean", "cos_per_step_min", "cos_per_step_max",
                "gripper_accuracy", "gripper_mse", "prompt"]
        with open(output_dir / "per_chunk.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in per_chunk_rows:
                vals = []
                for k in keys:
                    v = r.get(k, "")
                    if isinstance(v, str):
                        vals.append(f"\"{v.replace(chr(34), chr(39))}\"")
                    elif isinstance(v, (list, tuple)):
                        vals.append(f"\"{v}\"")
                    elif isinstance(v, float):
                        vals.append(f"{v:.6f}")
                    else:
                        vals.append(str(v))
                f.write(",".join(vals) + "\n")
        print(f"[save] per_chunk.csv")

    # episode_summary.csv
    with open(output_dir / "episode_summary.csv", "w") as f:
        f.write("episode,n_chunks,arm_mse_mean,arm_rmse_mean,cos_mean,gripper_acc,mean_chunk_dt_s,prompt\n")
        for ep, s in ep_summary.items():
            f.write(f"{ep},{s['n_chunks']},{s['arm_mse_mean']:.6f},{s['arm_rmse_mean']:.6f},"
                    f"{s['cos_mean']:.6f},{s['gripper_acc']:.6f},{s['mean_chunk_dt_s']:.2f},"
                    f"\"{s['prompt']}\"\n")
    print(f"[save] episode_summary.csv")

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        is_v101 = ds.variant == "droid_101"
        n_arm_dims = 7 if is_v101 else 6
        gt_gripper_dim = 7 if is_v101 else 6
        for r in per_chunk_rows:
            tag = f"ep{r['episode']:03d}_c{r['chunk_index']:02d}"
            pred = all_actions_dict[f"{tag}_pred"]
            gt = all_actions_dict[f"{tag}_gt"]
            fig, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
            for d in range(n_arm_dims):
                ax = axes[d // 4][d % 4]
                ax.plot(pred[:, d], label="pred", linewidth=1.0)
                ax.plot(gt[:, d], label="GT", linewidth=1.0, linestyle="--")
                ax.set_title(f"dim {d}", fontsize=9)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=7)
                if d == 0:
                    ax.legend(fontsize=7)
            ax = axes[1][3]
            ax.plot(pred[:, 7], label=f"pred gripper(pred[7])", linewidth=1.0)
            ax.plot(gt[:, gt_gripper_dim], label=f"GT gripper(gt[{gt_gripper_dim}])",
                    linewidth=1.0, linestyle="--")
            ax.set_title(f"gripper (pred[7] vs GT[{gt_gripper_dim}])", fontsize=9)
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
            fig.suptitle(f"{tag} | {ds.variant} | RMSE={r['rmse_arm_total']:.3f} | {r['prompt'][:48]}", fontsize=10)
            fig.tight_layout()
            fig.savefig(output_dir / f"plot_{tag}.png", dpi=110)
            plt.close(fig)
        print(f"[save] per-chunk plots ({len(per_chunk_rows)} files)")
    except Exception as e:
        print(f"[warn] plots failed: {e!r}")

    # mem_trace.csv
    with open(output_dir / "mem_trace.csv", "w") as f:
        if mem_trace:
            keys = list(mem_trace[0].keys())
            f.write(",".join(keys) + "\n")
            for row in mem_trace:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"[save] mem_trace.csv")

    # manifest
    final_mem = cuda_mem_gib()
    aggregate = {
        "arm_mse_total_mean": arm_mse_total,
        "arm_rmse_total_mean": arm_rmse_total,
        "cos_per_step_mean_mean": cos_mean_mean,
        "cos_per_step_mean_min": cos_mean_min,
        "gripper_acc_mean": grip_acc_mean,
        "per_dim_arm_mse_mean": per_dim_mse,
    }
    # PASS criteria for v1: every chunk got shape (24,8) and finite. We do NOT
    # gate on numeric error here -- this is the first run and we want to see
    # the numbers before setting a threshold.
    result = "PASS" if (n_successful > 0) else "FAIL"
    manifest = {
        "stage": "stage3_path_b_gt_validation",
        "result": result,
        "model_path": model_path,
        "embodiment_tag": embodiment,
        "dataset_dir": str(dataset_dir),
        "episodes": episode_list,
        "chunks_per_episode_requested": K,
        "n_successful_chunks": n_successful,
        "n_skipped_chunks": n_skipped,
        "load_seconds": round(load_dt, 2),
        "mean_chunk_seconds": round(float(np.mean([r["dt_s"] for r in per_chunk_rows])), 2),
        "total_seconds": round(time.perf_counter() - t_start, 2),
        "aggregate": aggregate,
        "per_episode": {str(k): v for k, v in ep_summary.items()},
        "vram_peak_alloc_gib": round(final_mem["max_alloc_gib"], 3),
        "host_ram_peak_used_gib": round(max(r.get("used_gib", 0.0) for r in mem_trace), 2),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch_version": torch.__version__,
        "encoder_call_counts": dict(patch_state),
        "dataset_variant": ds.variant,
        "alignment_note": (
            "droid_101: Model emits action.joint_position (7) + "
            "action.gripper_position (1) = 8 dims in the SAME space as "
            "LeRobot droid_1.0.1's combined 'action' (8) field. We compare "
            "all 7 arm joint dims directly + gripper as binary." if ds.variant == "droid_101"
            else
            "droid_100: Model emits joint_position (7)+gripper(1); GT is "
            "Cartesian (6)+gripper(1). Arm dims are INCOMMENSURABLE; only "
            "gripper accuracy is interpretable. Use droid_101 variant for "
            "proper numeric validation."
        ),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] manifest.json")

    banner("STAGE 3 PATH B RESULT")
    print(f"  n_successful_chunks      : {n_successful}  ({n_skipped} skipped)")
    print(f"  mean chunk seconds       : {manifest['mean_chunk_seconds']} s")
    print(f"  AGGREGATE arm RMSE       : {arm_rmse_total:.4f}")
    print(f"  AGGREGATE cos per-step   : {cos_mean_mean:.4f}  (min over chunks: {cos_mean_min:.4f})")
    print(f"  AGGREGATE gripper acc    : {grip_acc_mean:.4f}")
    print(f"  per-dim arm MSE (mean)   : {[f'{v:.4f}' for v in per_dim_mse]}")
    print(f"  peak VRAM                : {manifest['vram_peak_alloc_gib']} GiB")
    print(f"  result                   : {result}")
    return 0 if n_successful > 0 else 6


def _save_partial(output_dir, per_chunk_rows, all_actions_dict):
    """Best-effort artifact dump on failure."""
    try:
        import numpy as np
        if all_actions_dict:
            np.savez(output_dir / "per_chunk_actions_partial.npz", **all_actions_dict)
        with open(output_dir / "per_chunk_partial.json", "w") as f:
            json.dump(per_chunk_rows, f, indent=2, default=str)
        print(f"[save-partial] {len(per_chunk_rows)} chunks dumped")
    except Exception as e:
        print(f"[save-partial] failed: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
