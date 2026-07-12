#!/usr/bin/env bash
# Runs INSIDE the eval SIF. Tests whether a PRISTINE MolmoAct2-LIBERO teacher can be
# evaluated directly from --policy.type (no saved lerobot checkpoint), and whether it
# produces sane actions (nonzero success on an easy task) with the LIBERO env
# pre/post processors handling action normalization.
set -u
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_BLAS_PREFER_HIPBLASLT=0
LE=lerobot-eval; [ -x /opt/venv/bin/lerobot-eval ] && LE=/opt/venv/bin/lerobot-eval
CAM='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
"$LE" \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.inference_action_mode=continuous \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.model_dtype=bfloat16 --policy.use_amp=true \
  --policy.enable_inference_cuda_graph=false \
  --policy.device=cuda \
  --policy.per_episode_seed=true --policy.eval_seed=1000 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --env.camera_name_mapping="$CAM" \
  --eval.batch_size=1 --eval.n_episodes=1 --seed=1000 \
  --output_dir=/outputs/vd_smoke/teacher_type_test
echo "INNER_RC=$?"
