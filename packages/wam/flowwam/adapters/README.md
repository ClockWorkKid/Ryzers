# FlowWAM sim Policy adapters

Runtime adapters that wrap the FlowWAM dual-stream world model behind the simulation packages'
model-agnostic `Policy` seam, selected via `POLICY_FACTORY=flowwam_robotwin_policy:build_policy`.

Only imported when driving a sim base (chained build `ryzers build simulation/robotwin flowwam`); harmless
(unused) otherwise. Added in the closed-loop phase (P6) once the flow -> action seam is resolved
(see `docs/flowwam/PLAN.md`, open Q1: FlowWAM emits video + optical flow, not joint actions).

- `flowwam_robotwin_policy.py` - RoboTwin 2.0 closed-loop adapter (added at P6).
