# AVDC demos

Run with `VAR=... ryzers run --name avdc /ryzers/demos/<demo>.sh`; outputs land in
`workspace/avdc/outputs`. See the package `README.md` for build + weight fetch.

- **phase 2 - open-loop video prediction** (`demo_openloop.py` / `.sh`): load a real AVDC snapshot
  + CLIP, feed one initial frame + a task string, and generate the predicted future video. A
  reference frame exists, so per rule 2.b the initial frame is shown on the left and the generated
  video on the right (when a ground-truth future clip is available, a two-column GT-vs-pred layout
  per rule 2.a).
- **phase 3 - closed-loop Meta-World** (`demo_metaworld.sh`): drive AVDC closed-loop in the
  Meta-World MuJoCo simulator (video model -> UniMatch dense flow -> actions) once the sim layer is
  added; writes rollout GIFs + a success rate.

The weight-free image entrypoint is the smoke `test.py` (`ryzers run --name avdc`).
