# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-env attention-feedback saliency carrier for the lerobot MolmoAct2 policy.

The model patch (patch_vis_attnfeedback.py) can self-carry saliency across
predict_action calls, which is correct for a single uninterrupted episode. For
closed-loop eval that runs MANY episodes/tasks with one shared model, the carry
must be keyed per env and CLEARED on reset() so every new episode re-surveys
(full first frame) instead of inheriting the previous task's tunnel.

This patches the installed lerobot policy to mirror `self._depth_caches`:
  * __init__ : add self._saliency_caches and force VIS_ATTNFB_SELFCARRY=0
  * reset    : clear self._saliency_caches  (-> first frame of each episode surveys)
  * select_action: set vision_backbone._attn_feedback_saliency_in before predict,
                   store ._attn_feedback_saliency_out after.

Idempotent + reversible. Runs inside the libero venv (site-packages are baked
into the image, so apply it at run time in the same container before lerobot-eval).

    python patch_lerobot_attnfb.py [policy_file]            # apply
    python patch_lerobot_attnfb.py [policy_file] --revert   # remove
"""
import glob
import os
import sys

DEFAULT_GLOBS = [
    "/opt/*/lib/python*/site-packages/lerobot/policies/molmoact2/modeling_molmoact2.py",
    "/opt/*/lib/python*/site-packages/lerobot/policies/molmoact2/policy_molmoact2.py",
]

# --- Edit 1: __init__ state + disable model self-carry ------------------------
ANCHOR1 = "        self._depth_caches: dict[int, Any] = {}\n"
BLOCK1 = (
    "        self._depth_caches: dict[int, Any] = {}\n"
    "        # === ATTNFB_CARRIER START ===\n"
    "        self._saliency_caches: dict[int, Any] = {}\n"
    "        self._afb_steps: dict[int, int] = {}\n"
    "        import os as _os_afb\n"
    "        _os_afb.environ.setdefault(\"VIS_ATTNFB_SELFCARRY\", \"0\")\n"
    "        # === ATTNFB_CARRIER END ===\n"
)
EDIT1 = (ANCHOR1, BLOCK1)

# --- Edit 2: reset clears the per-env saliency cache --------------------------
ANCHOR2 = (
    "        self._action_queues = defaultdict(lambda: deque())\n"
    "        self._depth_caches = {}\n"
)
BLOCK2 = (
    "        self._action_queues = defaultdict(lambda: deque())\n"
    "        self._depth_caches = {}\n"
    "        self._saliency_caches = {}  # ATTNFB_CARRIER: re-survey each episode\n"
    "        self._afb_steps = {}\n"
)
EDIT2 = (ANCHOR2, BLOCK2)

# --- Edit 3: backbone resolver + reset hook (method inserted before reset) ----
ANCHOR3 = "    def reset(self) -> None:\n"
BLOCK3 = (
    "    # === ATTNFB_CARRIER_METHOD START ===\n"
    "    def _afb_vision_backbone(self):\n"
    "        m = getattr(self, \"model\", None)\n"
    "        for path in ((\"model\", \"vision_backbone\"), (\"vision_backbone\",)):\n"
    "            o = m\n"
    "            ok = True\n"
    "            for a in path:\n"
    "                o = getattr(o, a, None)\n"
    "                if o is None:\n"
    "                    ok = False\n"
    "                    break\n"
    "            if ok and o is not None:\n"
    "                return o\n"
    "        return None\n"
    "    # === ATTNFB_CARRIER_METHOD END ===\n"
    "\n"
    "    def reset(self) -> None:\n"
)
EDIT3 = (ANCHOR3, BLOCK3)

# --- Edit 4: set saliency_in before predict -----------------------------------
ANCHOR4 = "                depth_cache = self._depth_caches.get(idx)\n"
BLOCK4 = (
    "                depth_cache = self._depth_caches.get(idx)\n"
    "                _afb_vb = self._afb_vision_backbone()\n"
    "                if _afb_vb is not None:\n"
    "                    import os as _os_afb2\n"
    "                    _afb_every = int(_os_afb2.environ.get(\"VIS_ATTNFB_SURVEY_EVERY\", \"0\") or 0)\n"
    "                    _afb_step = self._afb_steps.get(idx, 0)\n"
    "                    if _afb_every > 0 and (_afb_step % _afb_every) == 0:\n"
    "                        _afb_vb._attn_feedback_saliency_in = None  # forced re-survey -> refresh saliency to track motion\n"
    "                    else:\n"
    "                        _afb_vb._attn_feedback_saliency_in = self._saliency_caches.get(idx)\n"
    "                    self._afb_steps[idx] = _afb_step + 1\n"
)
EDIT4 = (ANCHOR4, BLOCK4)

# --- Edit 5: store saliency_out after predict ---------------------------------
ANCHOR5 = (
    "                if output.depth_cache is not None:\n"
    "                    self._depth_caches[idx] = output.depth_cache\n"
)
BLOCK5 = (
    "                if output.depth_cache is not None:\n"
    "                    self._depth_caches[idx] = output.depth_cache\n"
    "                if _afb_vb is not None:\n"
    "                    _afb_out = getattr(_afb_vb, \"_attn_feedback_saliency_out\", None)\n"
    "                    if _afb_out is not None:\n"
    "                        self._saliency_caches[idx] = _afb_out\n"
)
EDIT5 = (ANCHOR5, BLOCK5)

EDITS = [EDIT1, EDIT2, EDIT3, EDIT4, EDIT5]


def apply(text):
    statuses = []
    for find, repl in EDITS:
        if repl in text:
            statuses.append("already")
        elif find in text:
            text = text.replace(find, repl, 1)
            statuses.append("ok")
        else:
            statuses.append("no-anchor")
    return text, ",".join(statuses)


def revert(text):
    statuses = []
    for find, repl in EDITS:
        if repl in text:
            text = text.replace(repl, find, 1)
            statuses.append("reverted")
        else:
            statuses.append("clean")
    return text, ",".join(statuses)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_revert = "--revert" in sys.argv[1:]
    if args:
        files = args
    else:
        files = []
        for g in DEFAULT_GLOBS:
            files += glob.glob(g)
    if not files:
        print("no lerobot molmoact2 policy file found")
        return 1
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read()
        new, status = (revert(text) if do_revert else apply(text))
        if new != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
        print(f"{status:>45}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
