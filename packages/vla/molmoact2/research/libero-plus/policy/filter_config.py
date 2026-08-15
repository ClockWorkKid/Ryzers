"""Filter a training lerobot config down to fields valid in the EVAL image's MolmoAct2Config,
repoint checkpoint_path at the merged HF dir (/ckpt), and disable LoRA/PEFT (weights already
merged). Rank-agnostic: the same config works for every merged curriculum stage.

Run INSIDE the eval image. Env: SRC_CFG (default /src/_lerobot_cfg_src.json),
OUT_CFG (default /policy/config.json)."""
import json, dataclasses, os
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

SRC = os.environ.get("SRC_CFG", "/src/_lerobot_cfg_src.json")
OUT = os.environ.get("OUT_CFG", "/policy/config.json")

valid = {f.name for f in dataclasses.fields(MolmoAct2Config)}
src = json.load(open(SRC))
dropped = sorted(k for k in src if k not in valid and k != "type")
kept = {k: v for k, v in src.items() if k in valid or k == "type"}
kept["type"] = "molmoact2"
kept["checkpoint_path"] = "/ckpt"
kept["checkpoint_revision"] = None
kept["enable_lora_vlm"] = False
kept["enable_lora_action_expert"] = False
if "use_peft" in valid:
    kept["use_peft"] = False
# Eval-time normalization: use the LIBERO QUANTILES preset (training template ships None).
if "norm_tag" in valid:
    kept["norm_tag"] = "libero"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(kept, open(OUT, "w"), indent=4)
print("VALID_FIELDS", len(valid))
print("DROPPED", dropped)
print("checkpoint_path=", kept["checkpoint_path"], "enable_lora_vlm=", kept["enable_lora_vlm"],
      "norm_tag=", kept.get("norm_tag"))
print("type=", kept.get("type"), "| setup_type=", kept.get("setup_type"), "| control_mode=", kept.get("control_mode"))
print("nkeys=", len(kept), "-> WROTE", OUT)
