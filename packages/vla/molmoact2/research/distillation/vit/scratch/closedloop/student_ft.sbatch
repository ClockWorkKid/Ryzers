#!/usr/bin/env bash
#SBATCH --job-name=student_ft
#SBATCH -p mi300x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/sft_%x_%j.log
# Finetune the distilled student ViT + LoRA(VLM) + action expert on LIBERO.
#   sbatch student_ft.sbatch                         # full 12k
#   sbatch --export=ALL,RUN=student_ft_smoke,STEPS=4,SAVE_FREQ=4 -t 00:30:00 student_ft.sbatch
set -u
U=$USER
OUT=/shared_nobackup/$U/molmoact2/outputs
# stage launcher + student module from $HOME into /outputs (bind target)
for f in run_train_student.py vit_student.py student_ft_seg.sh; do
  [ -f "$HOME/$f" ] && { cp -f "$HOME/$f" "$OUT/$f"; sed -i 's/\r$//' "$OUT/$f"; }
done
sed -i 's/\r$//' "$HOME/student_ft_seg.sh" 2>/dev/null || true
echo "[sft] host=$(hostname) job=${SLURM_JOB_ID:-none} rocr=${ROCR_VISIBLE_DEVICES:-} run=${RUN:-student_ft_lora} steps=${STEPS:-12000} t=$(date -Is)"
bash "$HOME/student_ft_seg.sh"
echo "[sft] DONE rc=$? t=$(date -Is)"
