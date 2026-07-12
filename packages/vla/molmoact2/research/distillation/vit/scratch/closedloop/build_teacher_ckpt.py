"""Build a PRISTINE MolmoAct2-LIBERO lerobot checkpoint that carries dataset
normalization stats, so closed-loop eval produces correctly-scaled actions.

Both the teacher run and the student run load this identical checkpoint via
--policy.path; the student run additionally monkey-patches encode_image. This
makes the teacher-vs-student comparison a pure vision-encoder swap (same
pool / projector / LLM / action head / normalization stats).

No training step is taken: make_policy(ds_meta) loads the released weights and
initializes Normalize buffers from dataset stats; we then save_pretrained the
policy and the pre/post processors (which hold the stats used at eval).
"""

import os


def _patch_safe_version() -> None:
    """Bypass the broken get_safe_version (HF Hub v1.x raises on RevisionNotFound
    with a signature mismatch). Fall back to the requested revision."""
    def _safe(repo_id, version, *a, **k):
        return version or "main"
    for modname in ("lerobot.datasets.utils", "lerobot.datasets.dataset_metadata"):
        try:
            import importlib
            m = importlib.import_module(modname)
            if hasattr(m, "get_safe_version"):
                m.get_safe_version = _safe
        except Exception:
            pass


def _load_ds_meta(repo_id: str):
    _patch_safe_version()
    try:
        from lerobot.datasets import LeRobotDatasetMetadata
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    return LeRobotDatasetMetadata(repo_id, revision="main")


def main() -> None:
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    out = os.environ.get(
        "TEACHER_CKPT_DIR",
        "/outputs/base_teacher/checkpoints/000000/pretrained_model",
    )
    os.makedirs(out, exist_ok=True)

    cfg = MolmoAct2Config(
        checkpoint_path="allenai/MolmoAct2-LIBERO",
        inference_action_mode="continuous",
        chunk_size=10,
        n_action_steps=10,
        model_dtype="bfloat16",
        device="cuda",
    )
    print("[build] config:", cfg.type, "chunk", cfg.chunk_size, "ckpt", cfg.checkpoint_path, flush=True)

    print("[build] loading dataset metadata (stats) ...", flush=True)
    ds_meta = _load_ds_meta("allenai/MolmoAct2-LIBERO-Dataset")

    print("[build] building policy (loads released weights + stats) ...", flush=True)
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)

    print("[build] saving policy ...", flush=True)
    policy.save_pretrained(out)

    print("[build] building + saving pre/post processors with stats ...", flush=True)
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
    pre.save_pretrained(out)
    post.save_pretrained(out)

    print("[build] DONE ->", out, flush=True)
    print("[build] files:", sorted(os.listdir(out)), flush=True)


if __name__ == "__main__":
    main()
