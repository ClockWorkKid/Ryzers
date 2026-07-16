#!/usr/bin/env bash
# Build the LIBERO eval image (FROM the train base) then convert it to an Apptainer SIF.
#
# The docker BUILD (apt + pip) needs a root docker daemon AND the train base image
# (molmoact2-lerobot-train:rocm942), so it must run on a node that holds the train base.
# apptainer fakeroot is typically unavailable on shared compute nodes, so a free-node
# def-file build is usually not possible. Prefer running when node load is low
# (the build is CPU-heavy: egl_probe/mujoco compiles).
#
# All paths are env-overridable; defaults assume a WORK workspace root.
set -u
WORK="${WORK:-$HOME/molmoact2}"
DOCKERFILE="${DOCKERFILE:-$(dirname "$0")/Dockerfile.instinct-gfx942-eval}"
BCTX="${BCTX:-$WORK/eval_build}"
EVAL_IMG="${EVAL_IMG:-molmoact2-lerobot-eval:rocm942}"
OUTSIF="${OUTSIF:-$WORK/images/molmoact2-lerobot-eval.sif}"

TMPBASE="${APPTAINER_TMPDIR:-$WORK/apptmp}"; mkdir -p "$TMPBASE"
export APPTAINER_TMPDIR="$TMPBASE" APPTAINER_CACHEDIR="$TMPBASE/cache"; mkdir -p "$APPTAINER_CACHEDIR"

docker image inspect molmoact2-lerobot-train:rocm942 >/dev/null 2>&1 \
  || { echo "[build-eval-sif] train base image not on this node -> build it first (see ../../roi_lora_lerobot/docker/Dockerfile.instinct-gfx942)"; exit 2; }

mkdir -p "$BCTX"; cp -f "$DOCKERFILE" "$BCTX/Dockerfile"
echo "[build-eval-sif] host=$(hostname) docker build $EVAL_IMG  $(date '+%F %T')"
( cd "$BCTX" && docker build -t "$EVAL_IMG" -f Dockerfile . )
rc=$?; echo "[build-eval-sif] docker build rc=$rc $(date '+%F %T')"
[ "$rc" = "0" ] || { echo "[build-eval-sif] DOCKER BUILD FAILED"; exit "$rc"; }

mkdir -p "$(dirname "$OUTSIF")"
STAGE="$TMPBASE/$(basename "$OUTSIF")"
nice -n 19 ionice -c 3 apptainer build --force "$STAGE" "docker-daemon://$EVAL_IMG"
rc=$?; echo "[build-eval-sif] apptainer build rc=$rc $(date '+%F %T')"
[ "$rc" = "0" ] || { echo "[build-eval-sif] SIF CONVERT FAILED"; exit "$rc"; }
nice -n 19 ionice -c 3 cp -f "$STAGE" "$OUTSIF"; rm -f "$STAGE"
ls -la "$OUTSIF"; echo BUILD_EVAL_SIF_DONE
