# GR-1 demos

Run with `VAR=... ryzers run --name gr1 /ryzers/demos/<demo>.sh`; outputs land in
`workspace/gr1/outputs`. See the package `README.md` for build + weight/dataset fetch.

- **phase 2 — open-loop prediction** (`demo_openloop.py` / `.sh`): load real MAE + GR-1 weights,
  run the policy forward on a real CALVIN-format episode window (no simulator), and dump a GT-vs-pred
  action overlay + a two-column future-frame comparison (rule 2.a).
- **phase 3 — closed-loop CALVIN** (`demo_calvin.py` / `.sh`): drive GR-1 in the headless PyBullet
  sim (EGL on gfx1151) across the CALVIN debug validation tasks; writes rollout GIFs + a success rate.
- **phase 4 — benchmark** (`demo_bench.py` / `.sh`): time the closed-loop step path and report
  latency + arm MAE for the env-selected optimization config.
- **optimization layer** (`gr1_optim.py`): the opt-in, env-toggled fp16/bf16 + ROCm-flash SDPA +
  `torch.compile` wrapper shared by the closed-loop and benchmark runners (see `../OPTIMIZATIONS.md`).

The weight-free image entrypoint is the smoke `test.py` (`ryzers run --name gr1`).
