#!/bin/bash
# SLURM job: one SASE shot with the full (t, x, y, z) history streamed to HDF5 for movies/3D
# renders (scripts/run_movie.py, XLO_sim/movie.py). One serial process -- the z/t marching can't be
# split across cores -- so a single task is enough.
#
# Default config/base/Cu-seed-SASE-movie.yaml (t x y z = 1000 x 31 x 31 x 61) writes <= 14.6 GB
# (uncompressed estimate; gzip makes it smaller). Mind the home-directory quota: point OUT_DIR at
# larger storage if data/ can't take it, e.g.
#   OUT_DIR=/path/to/big/storage/movie sbatch scripts/submit_movie.sh
# Other overrides: YAML=..., SEED=... (SASE shot; default is the config's random_seed).
# Progress: one "z plane N/61 written" line per plane in logs/xlo-movie_<jobid>.out.
#
# Before submitting:
#   1. mkdir -p logs
#   2. python scripts/run_movie.py --check-only   (size estimate + stale-code check, no simulation)
#   3. sbatch scripts/submit_movie.sh

#SBATCH --partition=allcpu
#SBATCH --job-name=xlo-movie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
mkdir -p logs

YAML=${YAML:-config/base/Cu-seed-SASE-movie.yaml}
OUT_DIR=${OUT_DIR:-data/movie_${SLURM_JOB_ID}}
SEED_ARGS=()
if [[ -n "${SEED:-}" ]]; then
    SEED_ARGS=(--seed "$SEED")
fi

echo "movie run: $YAML -> $OUT_DIR/movie.h5"

# ${arr[@]+...} form: an empty array trips `set -u` on bash < 4.4
python scripts/run_movie.py --yaml "$YAML" --out "$OUT_DIR/movie.h5" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"}
