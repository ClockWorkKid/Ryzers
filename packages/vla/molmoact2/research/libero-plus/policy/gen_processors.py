"""Generate the lerobot pre/post processors (normalization) for the policy dir. norm_tag must be
set (e.g. "libero") so the QUANTILES stats load. Run INSIDE the eval image with the policy dir at
/policy."""
import os, json, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# Import policy module so @PreTrainedConfig.register_subclass("molmoact2") runs before parsing.
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

D = "/policy"
cfg = PreTrainedConfig.from_pretrained(D)
print("norm_tag=", getattr(cfg, "norm_tag", None), "| normalization_mapping=", cfg.normalization_mapping)
assert str(getattr(cfg, "norm_tag", "") or "").strip(), "norm_tag empty -> stats will NOT load"

pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=None)
pre.save_pretrained(D)
post.save_pretrained(D)
print("SAVED processors into", D)
print(sorted(os.listdir(D)))
