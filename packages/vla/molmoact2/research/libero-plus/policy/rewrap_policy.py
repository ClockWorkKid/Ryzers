"""Re-wrap the merged HF checkpoint as a lerobot policy checkpoint.

lerobot-eval's make_policy -> from_pretrained requires a lerobot-format model.safetensors in the
--policy.path dir (the policy wrapper state_dict incl. normalization buffers). A freshly filtered
policy dir only has config.json. Build the policy from that config (which loads the merged HF
weights from checkpoint_path=/ckpt + norm_stats), then save_pretrained a lerobot model.safetensors
+ config.json into the SAME policy dir so eval can load it.

Run INSIDE the eval image with the merged HF checkpoint mounted at /ckpt and the policy dir at
/policy.
"""
import os, sys, traceback
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import torch
from lerobot.configs.policies import PreTrainedConfig
# Import the policy module FIRST so its @PreTrainedConfig.register_subclass("molmoact2") decorator
# runs and the choice class is registered before we parse the config.
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2 import configuration_molmoact2  # noqa: F401

POLICY_DIR = "/policy"   # has config.json (type=molmoact2, checkpoint_path=/ckpt)

cfg = PreTrainedConfig.from_pretrained(POLICY_DIR)
cfg.device = "cuda"
# The training config template ships norm_tag=None; eval-time normalization needs the LIBERO
# QUANTILES preset. save_pretrained (below) persists cfg into config.json, so set it HERE so the
# saved policy config carries norm_tag=libero and gen_processors can load the stats.
cfg.norm_tag = "libero"
print("Loaded config type=", getattr(cfg, "type", None), "checkpoint_path=", cfg.checkpoint_path,
      "enable_lora_vlm=", getattr(cfg, "enable_lora_vlm", None), "norm_tag=", getattr(cfg, "norm_tag", None))


def build():
    for attempt in ("cfg_only", "cfg_stats_none"):
        try:
            if attempt == "cfg_only":
                return MolmoAct2Policy(cfg)
            else:
                return MolmoAct2Policy(cfg, dataset_stats=None)
        except TypeError as e:
            print(f"  build attempt {attempt} TypeError: {e}")
    raise RuntimeError("could not construct MolmoAct2Policy")


policy = build()
policy.eval()
n_params = sum(p.numel() for p in policy.parameters())
print(f"Built MolmoAct2Policy | params={n_params/1e9:.2f}B")

policy.save_pretrained(POLICY_DIR)
print("Saved lerobot policy checkpoint into", POLICY_DIR)
print("DIR:", sorted(os.listdir(POLICY_DIR)))
