# Docker / Apptainer for the LLM-backbone distillation pipeline

The whole distillation + DAgger + eval pipeline runs inside **one image** — the MolmoAct2 ×
LeRobot **closed-loop LIBERO eval** image (`molmoact2-lerobot-eval:rocm942`) — converted to an
Apptainer **SIF** and executed with `apptainer exec` (see the `../slurm/*.sbatch` scripts for the
exact invocations). Running training *and* eval in the same image guarantees train/deploy
consistency: the teacher modeling code the student is distilled against is byte-for-byte the code
used at rollout.

## Image chain

```
Dockerfile.instinct-gfx942        (ROCm gfx942 torch + lerobot@052d329 + ROI overlay)   -> molmoact2-lerobot-train:rocm942
        │  FROM
        ▼
Dockerfile.instinct-gfx942-eval   (+ LIBERO sim: robosuite + mujoco + bddl + GL/EGL)     -> molmoact2-lerobot-eval:rocm942
```

- The **canonical** Dockerfiles for the base/train image, the eval image, and the Strix-Halo
  (gfx1151) variant live in **`../../roi_lora_lerobot/docker/`**, together with the model overlay
  (`roi_overlay/`: `configuration_molmoact2.py`, `modeling_molmoact2.py`,
  `molmoact2_hf_model/modeling_molmoact2.py`, `processor_molmoact2.py`, `roi_prune.py`).
- `Dockerfile.instinct-gfx942-eval` here is a **snapshot copy** so this folder is a self-contained
  backup; if it ever diverges, the `roi_lora_lerobot/docker/` copy is source of truth.

> Per repo policy, large binary assets (model weights, datasets) are **not** vendored — the image
> pulls the public MolmoAct2 weights / LIBERO assets at build/run time; only code is committed here.

## Build the SIF

```bash
# 1) build (or already have) the train base image, then the eval image, then convert to SIF.
#    Runs on a node with a root docker daemon + the train base image present.
WORK=/path/to/workspace \
OUTSIF=$WORK/images/molmoact2-lerobot-eval.sif \
bash build_eval_sif.sh
```

## The Apptainer exec contract (what every sbatch does)

```bash
apptainer exec --rocm --cleanenv --home "$CACHE":/cache \
  --env HF_HOME=/cache --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True --env TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --env HIP_VISIBLE_DEVICES=$GPU --env PYTHONPATH=/work \
  --bind "$CODE":/work --bind "$OUTPUTS":/outputs --pwd /work \
  --bind "$OV/roi_prune.py":"$MP/roi_prune.py":ro \
  --bind "$OV/modeling_molmoact2.py":"$MP/modeling_molmoact2.py":ro \
  --bind "$OV/configuration_molmoact2.py":"$MP/configuration_molmoact2.py":ro \
  --bind "$OV/hf_modeling_molmoact2.py":"$MP/molmoact2_hf_model/modeling_molmoact2.py":ro \
  "$SIF" python /work/<script>.py ...
```

- `CODE` (`/work`) = this `research/llm_distill/` folder (the training/eval code).
- `OUTPUTS` (`/outputs`) = where checkpoints/rollouts are written.
- `OV` = the model overlay dir (copy `roi_overlay/*` from `../../roi_lora_lerobot/docker/roi_overlay/`;
  note the eval scripts bind `hf_modeling_molmoact2.py` -> `.../molmoact2_hf_model/modeling_molmoact2.py`).
- `MP` = `/opt/lerobot/src/lerobot/policies/molmoact2` (the in-image policy package the binds override).
- For closed-loop eval add `--env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa` (CPU render
  fallback; EGL is the fast path when available).

Runs used a single **AMD Instinct MI210** GPU, bf16.
