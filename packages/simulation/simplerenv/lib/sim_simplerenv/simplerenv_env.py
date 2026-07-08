# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""SimplerEnv Gym-API env loader: build/reset/step/image/instruction/horizon helpers.

Wraps ``simpler_env.make`` so the rollout loop, sanity runner, and interactive servers stay
model-agnostic. Handles both embodiments (Google Robot / WidowX+Bridge), long-horizon
subtask advancement, and the SAPIEN offscreen-render config (denoiser off -- OIDN is
CUDA-only). Images come from SimplerEnv's ``get_image_from_maniskill2_obs_dict`` as HxWx3
uint8. The env's ``step`` returns ``(obs, reward, terminated, truncated, info)`` where
``terminated``/``info['success']`` encode task success.
"""
import numpy as np

# The 8 visual-matching tasks (example.ipynb) grouped by embodiment. The VLA-JEPA paper
# reports pick_coke_can / move_near / drawer (google) and the widowx put/place tasks.
GOOGLE_TASKS = [
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
]
WIDOWX_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]
ALL_TASKS = GOOGLE_TASKS + WIDOWX_TASKS

# Fallback per-task horizons (SimplerEnv sets max_episode_steps on the env spec; we prefer
# that at runtime and only fall back to these if unavailable).
_DEFAULT_MAX_STEPS = {"google_robot": 113, "widowx": 60}

ACTION_DIM = 7


def policy_setup_for(task):
    """Return the policy setup name SimplerEnv expects for a task ('google_robot'|'widowx_bridge')."""
    return "google_robot" if "google" in task else "widowx_bridge"


def embodiment_for(task):
    return "google_robot" if "google" in task else "widowx"


def _disable_denoiser():
    """OIDN ray-trace denoiser is CUDA-only; disable for AMD/ROCm offscreen rendering."""
    try:
        import sapien.core as sapien
        sapien.render_config.rt_use_denoiser = False
    except Exception:  # noqa: BLE001 - older/newer sapien may expose this differently
        try:
            import sapien
            sapien.render.set_ray_tracing_denoiser("none")
        except Exception:  # noqa: BLE001
            pass


def build_env(task, seed=0, **make_kwargs):
    """Create a SimplerEnv task; return (env, obs, instruction)."""
    import simpler_env

    _disable_denoiser()
    env = simpler_env.make(task, **make_kwargs)
    obs, _reset_info = env.reset(seed=seed) if _accepts_seed(env) else env.reset()
    instruction = env.get_language_instruction()
    return env, obs, instruction


def _accepts_seed(env):
    try:
        import inspect
        return "seed" in inspect.signature(env.reset).parameters
    except (TypeError, ValueError):
        return False


def get_image(env, obs):
    """HxWx3 uint8 third-person view from the maniskill2 obs dict."""
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return np.asarray(get_image_from_maniskill2_obs_dict(env, obs))


def get_instruction(env):
    return env.get_language_instruction()


def get_max_steps(env_or_task):
    """Prefer the env spec's max_episode_steps; else fall back per embodiment."""
    spec = getattr(env_or_task, "spec", None)
    n = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if n:
        return int(n)
    task = env_or_task if isinstance(env_or_task, str) else ""
    return _DEFAULT_MAX_STEPS["google_robot" if "google" in task else "widowx"]


def get_dummy_action():
    """Zero delta + open gripper (harmless settle/HOLD action)."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[6] = -1.0
    return a


def is_success(info, terminated):
    """SimplerEnv encodes success in info['success'] (and terminated on success)."""
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    return bool(terminated)
