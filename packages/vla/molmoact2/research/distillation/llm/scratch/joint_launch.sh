#!/usr/bin/env bash
set -u
U=$USER
OUT=/shared_nobackup/$U/molmoact2/outputs
# fresh start: remove any stale joint dirs so the run begins at step 0
rm -rf "$OUT/llmd_joint" "$OUT/llmd_joint_smoke" 2>/dev/null || true
echo "wiped stale llmd_joint dirs"
# full joint run + 2 afterany resume segments (24k steps ~ 2h05m at ~3.2 step/s)
J1=$(sbatch --parsable --job-name=llmd_joint ~/llmd_joint.sbatch)
echo "seg1=$J1"
J2=$(sbatch --parsable --job-name=llmd_joint --dependency=afterany:$J1 ~/llmd_joint.sbatch)
echo "seg2=$J2 (afterany:$J1)"
J3=$(sbatch --parsable --job-name=llmd_joint --dependency=afterany:$J2 ~/llmd_joint.sbatch)
echo "seg3=$J3 (afterany:$J2)"
squeue -u "$U" -o '%.7i %.9P %.11j %.2t %.6M %R' | grep -E 'llmd_joint|JOBID'
