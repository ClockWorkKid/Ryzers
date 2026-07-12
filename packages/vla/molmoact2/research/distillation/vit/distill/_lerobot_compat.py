"""Compat shim: lerobot 0.5.2 dataset loading under huggingface_hub 1.x.

lerobot's ``get_safe_version()`` refuses to load a dataset that has no PEP440
*version tag* on the Hub, raising ``RevisionNotFoundError``. Under huggingface_hub
1.x that error's ``__init__`` requires a keyword-only ``response=`` argument, which
lerobot 0.5.2 does not pass, so the ``raise`` itself throws
``TypeError: HfHubHTTPError.__init__() missing 1 required keyword-only argument:
'response'`` and masks the real (helpful) message.

``allenai/MolmoAct2-LIBERO-Dataset`` is a valid public v3.0 dataset that simply
isn't tagged with a version (its ``codebase_version`` lives in ``meta/info.json``).
We fall back to the ``main`` branch, which carries the data. Idempotent; call
``apply()`` before instantiating ``LeRobotDataset``.
"""

from __future__ import annotations

import logging

_PATCHED = False
_FALLBACK_REVISION = "main"


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        import lerobot.datasets.utils as lru
    except Exception:  # lerobot not installed (e.g. laptop) -> nothing to patch
        _PATCHED = True
        return

    orig = lru.get_safe_version

    def get_safe_version(repo_id, version):
        try:
            return orig(repo_id, version)
        except Exception as exc:  # untagged repo / hf_hub ctor mismatch -> use main
            logging.getLogger(__name__).warning(
                "lerobot get_safe_version(%s, %s) failed (%s: %s); falling back to '%s'.",
                repo_id, version, type(exc).__name__, exc, _FALLBACK_REVISION,
            )
            return _FALLBACK_REVISION

    lru.get_safe_version = get_safe_version
    # Patch modules that imported the symbol by value (e.g. dataset_metadata).
    for mod_name in ("lerobot.datasets.dataset_metadata", "lerobot.datasets.lerobot_dataset"):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            if getattr(mod, "get_safe_version", None) is orig:
                mod.get_safe_version = get_safe_version
        except Exception:
            pass
    _PATCHED = True
