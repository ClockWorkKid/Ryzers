# RoboTwin on ROCm — deviations

curobo (upstream's default planner and expert seed validator) is CUDA-only, and
SAPIEN's ray-tracing OIDN denoiser needs CUDA. The following changes make the
closed-loop eval run on ROCm/Vulkan. They are the only functional deviations.

- **Planner**: `curobo` → `mplib` (`mplib_RRT`). `patches/robotwin_rocm.patch`
  stubs `CuroboPlanner = None` and routes `set_planner` to `MplibPlanner`;
  `setup_robotwin.sh` sets `planner: "mplib_RRT"` in the aloha-agilex asset config.
- **Expert seed validation**: `expert_check = False` (it relies on curobo). Seeds
  are used as-is; instruction text falls back to the task `full_description`
  (the released RoboTwin checkpoint is unconditional, so text is unused).
- **Renderer**: ray-tracing denoiser `oidn` → `none` (OIDN is CUDA-only).
- **numpy**: pinned to `1.26.4` for the `mplib` C-extension ABI.
- **open3d**: import made optional (only used for point-cloud export, not needed here).
