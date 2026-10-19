#!/bin/bash
# submit_wiki.sh — DistTGL on WIKI, single node, N x 2080 Ti.
#
# Reproduces the 1 x j x 1 configurations from the DistTGL paper (Fig. 9a).
# 2080 Ti on Unity allows 3 cores and ~46 GB RAM per GPU.
#
#   cd /work/pi_mserafini_umass_edu/aniketmishra/disttgl
#   mkdir -p logs                                   # must exist before sbatch
#   sbatch --gpus=1 -c 3  --mem=40G  submit_wiki.sh
#   sbatch --gpus=2 -c 6  --mem=80G  submit_wiki.sh
#   sbatch --gpus=4 -c 12 --mem=160G submit_wiki.sh
#   SEED=1 sbatch --gpus=4 -c 12 --mem=160G submit_wiki.sh
#
# Paper targets (Wikipedia, test MRR vs 49 negatives): ~0.853 on 1 GPU,
# trending to ~0.81 on 8 GPUs.
#
#SBATCH --job-name=disttgl-wiki
#SBATCH --partition=gpu
#SBATCH --constraint=2080ti
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# Slurm copies this script to its spool dir, so use the submit directory.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate disttgl     # activate.d hook adds the env's CUDA 11.3 libs to LD_LIBRARY_PATH

NGPU="${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)}"
SEED="${SEED:-0}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-3} / NGPU ))
(( THREADS < 1 )) && THREADS=1

export OMP_NUM_THREADS="${THREADS}"
export PYTHONUNBUFFERED=1

mkdir -p logs results
LOG="results/WIKI_1x${NGPU}x1_seed${SEED}.log"

echo "[$(date)] DistTGL WIKI | ${NGPU} GPU(s) | seed ${SEED} | ${THREADS} threads/proc | $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --standalone handles single-node rendezvous; no head-node address needed.
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  train.py \
    --data WIKI \
    --group 1 \
    --minibatch_parallelism 1 \
    --seed "${SEED}" \
    --omp_num_threads 1 \
    --log "${LOG}"

echo "[$(date)] finished. Final metrics:"
grep -i "test" "${LOG}" | tail -1
