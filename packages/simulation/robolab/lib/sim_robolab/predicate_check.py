# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gate G3: validate the object_in_container(banana, bowl) predicate on the REAL YCB assets.

This is a predicate UNIT TEST, not a policy rollout: it places the real YCB banana + bowl into
four controlled configurations using genuine gripper contact and real MuJoCo physics settling
(no faked flags -- the predicate reads real body poses + real contacts), each isolating one
gating condition:

  POSITIVE       : banana grasped (real finger contact) then released, settles inside the bowl,
                   gripper retracted  -> success = True.
  NEG never_grasp: banana settles in the bowl but the gripper never touches it
                   -> False (require_contact_with).
  NEG still_held : banana held in the bowl footprint, gripper still in contact
                   -> False (require_gripper_detached).
  NEG missed_bowl: banana grasped then released OUTSIDE the bowl
                   -> False (containment).

PASS iff all four match expectation. Saves an annotated 4-panel image to $OUT_DIR.
Env: TASK, SEED, OUT_DIR.
"""
import os
from datetime import datetime

import numpy as np

from sim_robolab.envutil import env_int, env_str
from sim_robolab.render import banner_frame, compose_view
from sim_robolab.robolab_env import arm_qpos, abs_joints_to_action, eef_pos
from sim_robolab.scene import build_scene

CLOSE, OPEN = -1.0, 1.0
QUAT = (1.0, 0.0, 0.0, 0.0)


def _set_pose(env, obj, pos, quat=QUAT):
    j = obj.joints[0]
    env.sim.data.set_joint_qpos(j, np.concatenate([np.asarray(pos, float), np.asarray(quat, float)]))
    env.sim.data.set_joint_qvel(j, np.zeros(6))
    env.sim.forward()


def _step(env, n, gripper, pin=None):
    """Step n control steps holding the arm (zero joint delta); optionally re-pin objects.

    pin: list of (obj, pos) re-asserted every step (a stable test fixture, e.g. a bowl held
    around the grasped banana). predicate.update() runs inside env.step (_post_action).
    """
    for _ in range(n):
        if pin:
            for obj, pos in pin:
                _set_pose(env, obj, pos)
        env.step(abs_joints_to_action(env, arm_qpos(env), gripper))


def _table_top(env):
    return float(env.table_offset[2] + env.table_full_size[2] / 2.0)


def _bowl_center(env):
    return np.array(env.sim.data.body_xpos[env.bowl_body_id])


def _setup(scene, bowl_xy):
    """Reset, drop the real bowl at bowl_xy and let it settle on the table."""
    scene.reset()
    env = scene.env
    top = _table_top(env)
    _set_pose(env, env.bowl, [bowl_xy[0], bowl_xy[1], top + 0.06])
    _step(env, 25, OPEN)
    return env, top


def _grasp_at_eef(env, steps=15):
    """Pin the real banana at the eef and close the gripper -> genuine finger-banana contact."""
    p = eef_pos(env)
    for _ in range(steps):
        _set_pose(env, env.banana, p)
        env.step(abs_joints_to_action(env, arm_qpos(env), CLOSE))
    return p


def _frame(scene, title, info, success):
    obs = scene.observe()
    tag = f"contain={int(info['contained'])} grasp={int(info['was_grasped'])} " \
          f"detach={int(info['detached'])} settle={int(info['settled'])} -> " \
          f"{'SUCCESS' if success else 'False'}"
    return banner_frame(compose_view(obs["view"]), f"{title} | {tag}", 720)


def scenario_positive(scene):
    env, top = _setup(scene, (0.0, 0.0))
    c = _bowl_center(env)
    _grasp_at_eef(env)                                   # real contact -> was_grasped
    _set_pose(env, env.banana, [c[0], c[1], top + 0.05])  # release into bowl
    _step(env, 45, OPEN)                                 # settle; arm at eef -> detached
    return env._predicate.info, bool(env._check_success())


def scenario_never_grasped(scene):
    env, top = _setup(scene, (0.0, 0.0))
    c = _bowl_center(env)
    _set_pose(env, env.banana, [c[0], c[1], top + 0.05])  # into bowl, never touched
    _step(env, 45, OPEN)
    return env._predicate.info, bool(env._check_success())


def scenario_still_held(scene):
    env, top = _setup(scene, (0.0, 0.0))
    p = _grasp_at_eef(env)                               # real contact -> held at eef
    # bring the bowl up around the held banana (pinned fixture) so it IS contained while held
    bowl_at = [p[0], p[1], p[2]]
    _step(env, 12, CLOSE, pin=[(env.banana, p), (env.bowl, bowl_at)])
    return env._predicate.info, bool(env._check_success())


def scenario_missed_bowl(scene):
    env, top = _setup(scene, (0.0, 0.0))
    _grasp_at_eef(env)                                   # real contact -> was_grasped
    _set_pose(env, env.banana, [0.30, 0.0, top + 0.05])  # release OUTSIDE bowl
    _step(env, 45, OPEN)
    return env._predicate.info, bool(env._check_success())


SCENARIOS = [
    ("POSITIVE", scenario_positive, True, None),
    ("NEG never_grasped", scenario_never_grasped, False, "was_grasped"),
    ("NEG still_held", scenario_still_held, False, "detached"),
    ("NEG missed_bowl", scenario_missed_bowl, False, "contained"),
]


def main():
    task = env_str("TASK", "BananaInBowl")
    seed = env_int("SEED", 0)
    out_dir = env_str("OUT_DIR", "/sim_outputs")
    os.makedirs(out_dir, exist_ok=True)

    import imageio

    print(f"[g3] validating object_in_container on real YCB assets ({task})", flush=True)
    scene = build_scene(task, seed=seed)
    frames, all_ok = [], True
    for name, fn, expect_success, isolates in SCENARIOS:
        info, success = fn(scene)
        ok = (success == expect_success)
        # for negatives, also require the isolated condition to be the blocking one
        if not expect_success and isolates is not None:
            blocking = (not info[isolates])
            ok = ok and blocking
        all_ok = all_ok and ok
        frames.append(_frame(scene, name, info, success))
        pr = scene.env._predicate
        o, c = pr._obj_pos(), pr._container_pos()
        xy = float(np.linalg.norm(o[:2] - c[:2]))
        print(f"[g3] {name:18s} success={success} expected={expect_success} "
              f"info={info} -> {'OK' if ok else 'MISMATCH'}", flush=True)
        print(f"       geom: banana={np.round(o,3)} bowl={np.round(c,3)} "
              f"xy={xy:.3f}/rim={pr.rim_radius:.3f}  z={o[2]:.3f} in "
              f"[{c[2]+pr.base_z_offset:.3f},{c[2]+pr.rim_z_offset+pr.protrusion_tol:.3f}]",
              flush=True)

    scene.close()
    ts = datetime.now().strftime("%H%M%S")
    path = os.path.join(out_dir, f"predicate_{task}_{ts}.png")
    imageio.imwrite(path, np.ascontiguousarray(np.vstack(frames)))
    print(f"[g3] {'PASS' if all_ok else 'FAIL'}: saved {path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
