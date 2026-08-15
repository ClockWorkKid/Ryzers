#!/usr/bin/env bash
# Runs INSIDE the eval container for ONE shard: an explicit comma-separated list of task_ids of a
# suite. Uses the /policy + /ckpt mounts, osmesa software render, continuous flow-matching actions.
#   args: SUITE  IDLIST(comma-separated)  OUTDIR
set -e
SUITE="$1"; IDLIST="$2"; OUTDIR="$3"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Two known LIBERO-plus runtime patches, applied defensively per container (idempotent):
#  (1) NumPy 2.0 removed np.float_ -> np.float64 (base ships numpy>=2).
#  (2) imagenet-c fog() calls plasma_fractal() with hardcoded mapsize=256, then slices [:H,:W];
#      the lerobot libero_plus env renders larger frames -> broadcast error. Pass next pow2>=max(H,W).
sed -i 's/np\.float_/np.float64/g' /opt/libero-plus/libero/libero/envs/env_wrapper.py 2>/dev/null || true
SITE=$(python3 -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null || echo /opt/venv/lib/python3.12/site-packages)
echo 'import numpy as _n; [setattr(_n,k,getattr(_n,v)) for k,v in {"float_":"float64","complex_":"complex128","unicode_":"str_"}.items() if not hasattr(_n,k)]' > "$SITE/zz_np2compat.pth" 2>/dev/null || true
python3 - <<'PY' 2>/dev/null || true
f="/opt/libero-plus/libero/libero/envs/env_wrapper.py"
s=open(f).read()
old="x += c[0] * plasma_fractal(wibbledecay=c[1])[:height_x, :weight_x][..., np.newaxis]"
new="x += c[0] * plasma_fractal(mapsize=1<<(max(height_x, weight_x)-1).bit_length(), wibbledecay=c[1])[:height_x, :weight_x][..., np.newaxis]"
if old in s:
    open(f,"w").write(s.replace(old,new))
PY

IDS="[$IDLIST]"
rm -rf "$OUTDIR"   # launcher only calls us when eval_info.json is absent -> any content is partial
mkdir -p "$OUTDIR"
python -m lerobot.scripts.lerobot_eval \
  --policy.path=/policy \
  --policy.inference_action_mode=continuous \
  --env.type=libero_plus \
  --env.task="$SUITE" \
  --env.task_ids="$IDS" \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --rename_map='{"observation.images.image":"observation.images.front","observation.images.image2":"observation.images.wrist"}' \
  --output_dir="$OUTDIR"
