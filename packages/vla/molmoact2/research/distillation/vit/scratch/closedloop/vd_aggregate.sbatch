#!/usr/bin/env bash
#SBATCH --job-name=vd_agg
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH --output=/shared_nobackup/%u/molmoact2/vd_agg_%j.log
# N-way closed-loop comparison across all eval arms present under outputs/.
set -u
U=$USER
OUT=/shared_nobackup/$U/molmoact2/outputs
[ -f "$HOME/aggregate_nway.py" ] && { cp -f "$HOME/aggregate_nway.py" "$OUT/aggregate_nway.py"; sed -i 's/\r$//' "$OUT/aggregate_nway.py"; }
echo "[vd_agg] host=$(hostname) t=$(date -Is)"
python3 "$OUT/aggregate_nway.py" "$OUT" 2>&1 || python "$OUT/aggregate_nway.py" "$OUT"
echo "[vd_agg] DONE csv=$OUT/compare_nway.csv t=$(date -Is)"
