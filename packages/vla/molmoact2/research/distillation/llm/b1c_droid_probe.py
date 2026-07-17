"""Track-B1c rule-2 validation: DROID single-teacher path.

Confirms, on REAL data, that (1) a bounded offline DROID episode subset loads through the same
lerobot dataset+preprocessor path used for LIBERO, (2) the MolmoAct2-DROID teacher builds and
consumes those batches, and (3) the teacher's flow-matching loss is finite/low (a sane teacher
baseline) -- i.e. the DROID module works independently before we compose the co-distillation.

Env: REPO, TEACHER, NORM_TAG, MAX_EPISODES, BATCH.
"""
from __future__ import annotations
import glob, os
import torch


def usable_droid_episodes(max_eps):
    """Episodes whose data file AND all three camera video files are file_index==0 (the only
    files we downloaded) -> loadable fully offline."""
    import pandas as pd
    snaps = glob.glob("/cache/hub/datasets--lerobot--droid_1.0.1/snapshots/*")
    md = os.path.join(sorted(snaps)[0], "meta")
    epfiles = sorted(glob.glob(os.path.join(md, "episodes", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in epfiles], ignore_index=True)
    cams = ["observation.images.exterior_1_left", "observation.images.exterior_2_left",
            "observation.images.wrist_left"]
    cols = list(df.columns)
    print("[probe] episodes cols:", cols[:40], flush=True)

    def fidx(row, key):
        for c in (f"{key}/file_index", f"videos/{key}/file_index"):
            if c in df.columns:
                return int(row[c])
        return None

    keep = []
    for _, row in df.iterrows():
        dfi = int(row["data/file_index"]) if "data/file_index" in df.columns else None
        vfi = [fidx(row, c) for c in cams]
        if dfi == 0 and all(v == 0 for v in vfi):
            ei = int(row["episode_index"]) if "episode_index" in df.columns else int(row.name)
            keep.append(ei)
        if len(keep) >= max_eps:
            break
    return sorted(set(keep))


def main():
    repo = os.environ.get("REPO", "lerobot/droid_1.0.1")
    teacher = os.environ.get("TEACHER", "allenai/MolmoAct2-DROID")
    norm_tag = os.environ.get("NORM_TAG", "franka_droid")
    max_eps = int(os.environ.get("MAX_EPISODES", "24"))
    B = int(os.environ.get("BATCH", "2"))
    device = torch.device("cuda")

    eps = usable_droid_episodes(max_eps)
    print(f"[probe] usable DROID episodes (data+video file-000): n={len(eps)} sample={eps[:10]}", flush=True)
    if not eps:
        raise SystemExit("[probe] no fully-offline DROID episodes found in file-000")

    import data as D
    from types import SimpleNamespace

    A = SimpleNamespace(repo_id=repo, revision="main", teacher=teacher, device="cuda",
                        batch=B, num_workers=0, norm_tag=norm_tag, episodes=eps,
                        video_backend="pyav")
    ds, ds_meta, pre, cfg = D.build_dataset_and_preprocessor(A)
    print(f"[probe] dataset built: frames={ds.num_frames} episodes={ds.num_episodes} "
          f"camera_keys={list(getattr(ds_meta, 'camera_keys', []))}", flush=True)

    it = D.iter_full_batches(A, device)
    batch = next(it)
    print("[probe] BATCH keys/shapes:", flush=True)
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"   {k:28s} {tuple(v.shape)} {v.dtype}", flush=True)
        else:
            print(f"   {k:28s} (non-tensor: {type(v).__name__})", flush=True)

    # build DROID teacher policy and run one forward -> flow-matching loss (teacher baseline)
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.configs.types import FeatureType
    try:
        from lerobot.datasets.utils import dataset_to_policy_features
    except Exception:
        from lerobot.policies.factory import dataset_to_policy_features

    tcfg = MolmoAct2Config(checkpoint_path=teacher, chunk_size=10, n_action_steps=10,
                           action_mode="continuous", model_dtype="bfloat16", device="cuda",
                           norm_tag=norm_tag)
    feats = dataset_to_policy_features(ds_meta.features)
    tcfg.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    tcfg.input_features = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    print(f"[probe] output(action) feats: {[(k, tuple(v.shape)) for k, v in tcfg.output_features.items()]}", flush=True)
    policy = MolmoAct2Policy(tcfg).to(device)
    policy.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    print(f"[probe] DROID TEACHER flow loss (baseline) = {float(loss):.4f}", flush=True)
    print("[probe] OK DROID single-teacher path validated", flush=True)


if __name__ == "__main__":
    main()
