# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""MuJoCo-state reimplementation of RoboLab success conditionals.

Upstream RoboLab's `object_in_container` (robolab/core/task/conditionals) runs on Isaac
Sim contact/pose state. Here we recompute the same *semantics* directly from robosuite's
MuJoCo `sim.data` (body/geom world poses + contact list) — no renderer, no Isaac. Only the
condition used by the pilot (`object_in_container`) is implemented; more are added as tasks
scale.

`object_in_container(object, container, gripper, require_contact_with, require_gripper_detached)`
is True iff ALL hold:
  - containment : object COM is within the container's rim footprint (XY) and its Z sits
                  between the container base and rim (+ protrusion_tol, since an object larger
                  than the container rests with its COM proud of the rim).
  - was_grasped : a gripper finger geom touched the object at some point this episode
                  (sticky flag; only enforced when require_contact_with=True).
  - detached    : no gripper finger geom currently contacts the object
                  (only enforced when require_gripper_detached=True).
  - settled     : object nearly stationary for `settle_steps` consecutive checks (avoids
                  counting a transient mid-air frame).
"""
import numpy as np


def _body_id(sim, name):
    return sim.model.body_name2id(name)


def _geom_ids_under_body(sim, body_name):
    """All geom ids whose owning body name starts with `body_name` (robosuite name-mangles)."""
    ids = []
    for gid in range(sim.model.ngeom):
        bid = sim.model.geom_bodyid[gid]
        bname = sim.model.body_id2name(bid)
        if bname is not None and body_name in bname:
            ids.append(gid)
    return set(ids)


def geoms_in_contact(sim, geoms_a, geoms_b):
    """True if any active contact pairs a geom in set A with a geom in set B."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
            return True
    return False


class ObjectInContainer:
    """Stateful episode checker for object_in_container (see module docstring)."""

    def __init__(self, sim, object_body, container_body, gripper_finger_geoms,
                 rim_radius, base_z_offset, rim_z_offset, protrusion_tol=0.0,
                 require_contact_with=True, require_gripper_detached=True,
                 settle_steps=5, settle_vel=0.02):
        self.sim = sim
        self.object_body = object_body
        self.container_body = container_body
        self.obj_geoms = _geom_ids_under_body(sim, object_body)
        self.finger_geoms = set(gripper_finger_geoms)
        self.rim_radius = float(rim_radius)
        self.base_z_offset = float(base_z_offset)
        self.rim_z_offset = float(rim_z_offset)
        # An object larger than the container rests with its COM proud of the rim; allow it to
        # protrude by up to `protrusion_tol` (tied to the object's own size) while still counting
        # as contained. The XY rim-radius test remains the primary guard (see _contained).
        self.protrusion_tol = float(protrusion_tol)
        self.require_contact_with = bool(require_contact_with)
        self.require_gripper_detached = bool(require_gripper_detached)
        self.settle_steps = int(settle_steps)
        self.settle_vel = float(settle_vel)
        self.reset()

    def reset(self):
        self._was_grasped = False
        self._settle_count = 0
        self._last_obj_pos = None

    def _obj_pos(self):
        return np.array(self.sim.data.body_xpos[_body_id(self.sim, self.object_body)])

    def _container_pos(self):
        return np.array(self.sim.data.body_xpos[_body_id(self.sim, self.container_body)])

    def update(self):
        """Advance sticky/settle state one check. Call once per env step before success()."""
        if self.require_contact_with and not self._was_grasped:
            if geoms_in_contact(self.sim, self.finger_geoms, self.obj_geoms):
                self._was_grasped = True
        pos = self._obj_pos()
        if self._last_obj_pos is not None:
            moved = np.linalg.norm(pos - self._last_obj_pos)
            self._settle_count = self._settle_count + 1 if moved < self.settle_vel else 0
        self._last_obj_pos = pos

    def _contained(self):
        obj = self._obj_pos()
        cont = self._container_pos()
        xy = np.linalg.norm(obj[:2] - cont[:2])
        z_ok = ((cont[2] + self.base_z_offset) <= obj[2]
                <= (cont[2] + self.rim_z_offset + self.protrusion_tol))
        return (xy <= self.rim_radius) and z_ok

    def success(self):
        if not self._contained():
            return False
        if self.require_contact_with and not self._was_grasped:
            return False
        if self.require_gripper_detached and geoms_in_contact(
            self.sim, self.finger_geoms, self.obj_geoms
        ):
            return False
        if self._settle_count < self.settle_steps:
            return False
        return True

    @property
    def info(self):
        return {
            "contained": self._contained(),
            "was_grasped": self._was_grasped,
            "detached": not geoms_in_contact(self.sim, self.finger_geoms, self.obj_geoms),
            "settled": self._settle_count >= self.settle_steps,
        }
