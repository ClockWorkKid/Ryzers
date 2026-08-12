# iTHOR simulator base - demos

Model-agnostic demos for the AI2-THOR (iTHOR) ObjectNav simulator base. All render headless via
Vulkan (ai2thor `platform=CloudRendering`) on gfx1151. Select a model at runtime with
`POLICY_FACTORY=module:function`; with none set, the built-in `ScriptedPolicy` runs (no model).

## Sign-of-life / gate
```sh
ryzers build simulation/ithor --name sim-ithor
ryzers run --name sim-ithor            # test.py: headless Vulkan render spike (the GATE)
```

## `demo_sim_sanity.sh` - scripted ObjectNav rollout (no model)
Brings up the controller, resets to a random reachable pose, steps the ScriptedPolicy, renders
offscreen and writes a rollout video. Proves the harness is alive end to end.
```sh
SCENE=FloorPlan1 TARGET=Toaster ryzers run --name sim-ithor /ryzers/demos/demo_sim_sanity.sh
```

## `demo_benchmark.sh` - ObjectNav benchmark (model-agnostic)
Runs the seam policy across tasks x seeds in one process (writes `result_dict.json` + `summary.json`
+ per-task sample videos). Point it at a model adapter to benchmark it:
```sh
TASKS=all N_SEEDS=20 POLICY_FACTORY=avdc_ithor_policy:build_policy \
  ryzers run --name avdc-ithor /ryzers/demos/demo_benchmark.sh
```

## Knobs (env)
`SCENE`, `TARGET`, `N_SEEDS`, `RESOLUTION` (64), `MAX_EPLEN` (50), `RENDER_RESOLUTION`,
`THOR_PLATFORM` (CloudRendering), `THOR_GPU_DEVICE`, `VK_ICD_FILENAMES`, `POLICY_FACTORY`.

## Tasks (4 scenes x 3 targets)
- FloorPlan1: Toaster, Spatula, Bread
- FloorPlan201: Painting, Laptop, Television
- FloorPlan301: Blinds, DeskLamp, Pillow
- FloorPlan401: Mirror, ToiletPaper, SoapBar
