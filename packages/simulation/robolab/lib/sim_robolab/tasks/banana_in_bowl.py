# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""BananaInBowl — AMD-native reimplementation of RoboLab's BananaInBowlTask.

Upstream (NVlabs/RoboLab, banana_in_bowl_task.py): banana + bowl on a table, DROID Franka
(Robotiq 2F-85), success = object_in_container(banana, bowl) with require_contact_with +
require_gripper_detached. Here the same task is built on robosuite/MuJoCo (v1.5.1) so it
runs headless on ROCm/gfx1151.

Objects are the REAL YCB meshes upstream RoboLab uses (fetched + coacd-decomposed to MJCF at
runtime by ycb_to_mjcf.py; see docs/PILOT_PLAN.md):
  - banana : YCB 011_banana (graspable by the Robotiq 2F-85).
  - bowl   : YCB 024_bowl (concave; collision kept hollow via coacd so objects sit inside).
A primitive fallback (yellow capsule + red hollow cylinder) is used only when YCB assets are
unavailable (ROBOLAB_USE_YCB=0). Both are movable rigid objects (upstream contact_object_list).

Success is delegated to sim_robolab.predicates.ObjectInContainer (MuJoCo-state reimpl of the
upstream conditional). The env is modelled on robosuite.environments.manipulation.lift.Lift.
"""
import os

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CapsuleObject, HollowCylinderObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

INSTRUCTION = "Pick up the banana and place it in the bowl"

# Primitive-fallback bowl geometry (metres), used only when YCB assets are unavailable.
BOWL_OUTER_R = 0.075
BOWL_INNER_R = 0.065
BOWL_HEIGHT = 0.05


def _ycb_assets_root():
    """Return the YCB assets root with real banana+bowl MJCFs ready, else None.

    If enabled but the MJCFs are missing, lazily fetch+convert the YCB meshes (needs network
    + coacd). Any failure falls back to primitives so the sim still runs.
    """
    if os.environ.get("ROBOLAB_USE_YCB", "1") not in ("1", "true", "True"):
        return None
    root = os.environ.get("ROBOLAB_ASSETS_DIR", "/models/robolab_assets")

    def _ready():
        return (os.path.exists(os.path.join(root, "banana", "banana.xml"))
                and os.path.exists(os.path.join(root, "bowl", "bowl.xml")))

    if _ready():
        return root
    try:
        from sim_robolab.ycb_to_mjcf import ensure_assets
        ensure_assets(root)
        return root if _ready() else None
    except Exception as e:  # noqa: BLE001
        print(f"[robolab] YCB asset prep failed ({type(e).__name__}: {e}); "
              f"falling back to primitive banana/bowl.", flush=True)
        return None


class BananaInBowl(ManipulationEnv):
    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self._predicate = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ----- model -----
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        root = _ycb_assets_root()
        self._bowl_meta = None
        self._banana_meta = None
        if root is not None:
            # Real YCB 011_banana + 024_bowl meshes (same source as upstream RoboLab).
            from sim_robolab.ycb_to_mjcf import load_meta
            # duplicate_collision_geoms=False: our MJCF already ships an explicit textured
            # visual geom (group 1); duplicating the coacd collision hulls would add grey
            # (untextured) group-1 geoms that render on top and hide the YCB texture.
            self.banana = MujocoXMLObject(
                os.path.join(root, "banana", "banana.xml"), name="banana",
                joints=[dict(type="free", damping="0.0005")], obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.bowl = MujocoXMLObject(
                os.path.join(root, "bowl", "bowl.xml"), name="bowl",
                joints=[dict(type="free", damping="0.0005")], obj_type="all",
                duplicate_collision_geoms=False,
            )
            self._bowl_meta = load_meta(root, "bowl")
            self._banana_meta = load_meta(root, "banana")
        else:
            # Primitive fallback (yellow capsule + red hollow cylinder) for asset-less smoke.
            self.banana = CapsuleObject(
                name="banana", size=[0.018, 0.05],
                rgba=[0.95, 0.85, 0.15, 1.0], density=200.0,
            )
            self.bowl = HollowCylinderObject(
                name="bowl", outer_radius=BOWL_OUTER_R, inner_radius=BOWL_INNER_R,
                height=BOWL_HEIGHT, rgba=[0.85, 0.15, 0.15, 1.0],
            )

        objects = [self.banana, self.bowl]
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=objects,
                x_range=[-0.12, 0.12],
                y_range=[-0.18, 0.18],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.banana_body_id = self.sim.model.body_name2id(self.banana.root_body)
        self.bowl_body_id = self.sim.model.body_name2id(self.bowl.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def banana_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.banana_body_id])

            @sensor(modality=modality)
            def bowl_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bowl_body_id])

            sensors = [banana_pos, bowl_pos]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )
        return observables

    # ----- reset / success -----
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for pos, quat, obj in placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)])
                )
        self._build_predicate()

    def _finger_geom_ids(self):
        ids = []
        gripper = self.robots[0].gripper
        names = []
        if isinstance(gripper, dict):
            for g in gripper.values():
                names += list(getattr(g, "contact_geoms", []) or [])
        else:
            names += list(getattr(gripper, "contact_geoms", []) or [])
        for n in names:
            try:
                ids.append(self.sim.model.geom_name2id(n))
            except Exception:  # noqa: BLE001
                pass
        return ids

    def _build_predicate(self):
        from sim_robolab.predicates import ObjectInContainer

        if self._bowl_meta is not None:
            rim_radius = self._bowl_meta["inner_radius"]
            base_z_offset = self._bowl_meta["base_z_offset"]
            rim_z_offset = self._bowl_meta["rim_z_offset"]
            # Allow the banana to protrude above the rim by up to its own thickness (its
            # smallest extent); it is larger than the bowl so it rests proud of the rim.
            protrusion_tol = float(min(self._banana_meta["extent"]))
        else:
            rim_radius, base_z_offset, rim_z_offset = BOWL_INNER_R, -BOWL_HEIGHT / 2, BOWL_HEIGHT
            protrusion_tol = 0.0

        self._predicate = ObjectInContainer(
            sim=self.sim,
            object_body=self.banana.root_body,
            container_body=self.bowl.root_body,
            gripper_finger_geoms=self._finger_geom_ids(),
            rim_radius=rim_radius,
            base_z_offset=base_z_offset,
            rim_z_offset=rim_z_offset,
            protrusion_tol=protrusion_tol,
            require_contact_with=True,
            require_gripper_detached=True,
        )

    def reward(self, action=None):
        return float(self._check_success()) * self.reward_scale

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._predicate is not None:
            self._predicate.update()
        return ret

    def _check_success(self):
        if self._predicate is None:
            return False
        return bool(self._predicate.success())
