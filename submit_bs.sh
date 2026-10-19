#!/bin/bash -l
# submit_bs.sh — DistTGL batch-size sensitivity on WIKI, single node, N x 2080 Ti.
#
# BS is the per-GPU training batch size. EVAL_BS is the validation/test batch
# size and defaults to BS (matched, as in PRISM's evaluation); set EVAL_BS=600
# to hold evaluation fixed instead.
# Each combination gets its own working dir (runs/bs<BS>_eval<EVAL_BS>) because
# DistTGL's minibatch paths don't include the batch size. Minibatches are
# generated on first use.
#
#   BS=512 sbatch -p gpu,gpu-preempt --nodes=1 --gpus=1 -c 3 --mem=40G submit_bs.sh
#
# Never run two DistTGL jobs on one node at once (fixed /dev/shm names):
# chain with --dependency=afterany:<jobid>.
#
#SBATCH --job-name=disttgl-bs
#SBATCH --constraint=2080ti
#SBATCH --time=03:00:00
#SBATCH --output=/work/pi_mserafini_umass_edu/aniketmishra/disttgl/logs/%x_%j.out
#SBATCH --error=/work/pi_mserafini_umass_edu/aniketmishra/disttgl/logs/%x_%j.err

set -eo pipefail

REPO=/work/pi_mserafini_umass_edu/aniketmishra/disttgl
BS="${BS:?set BS}"
EVAL_BS="${EVAL_BS:-${BS}}"
SEED="${SEED:-0}"
RUN="${REPO}/runs/bs${BS}_eval${EVAL_BS}"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate disttgl
export PYTHONUNBUFFERED=1

mkdir -p "${RUN}" "${REPO}/results" "${REPO}/logs"
cd "${RUN}"
ln -sfn "${REPO}/DATA" DATA

NGPU="${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)}"
LOG="${REPO}/results/WIKI_bs${BS}_eval${EVAL_BS}_1x${NGPU}x1_seed${SEED}.log"

echo "[$(date)] DistTGL WIKI | train bs ${BS}/GPU | eval bs ${EVAL_BS} | ${NGPU} GPU(s) | seed ${SEED} | $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -f minibatches/.done ]]; then
  echo "[$(date)] generating minibatches (train bs ${BS}, eval bs ${EVAL_BS})"
  rm -rf minibatches
  python "${REPO}/gen_minibatch.py" --data WIKI --gen_eval --minibatch_parallelism 1 --batchsize "${BS}" --eval_batchsize "${EVAL_BS}"
  touch minibatches/.done
fi
echo "train batches: $(ls minibatches/WIKI_1_49_32 | grep -c train_pos) | val: $(ls minibatches/WIKI_49_eval | grep -c val_pos) | test: $(ls minibatches/WIKI_49_eval | grep -c test_pos) | $(du -sh minibatches | cut -f1)"

# --omp_num_threads 1: with more threads the forked mailbox daemon deadlocks.
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  "${REPO}/train.py" \
    --data WIKI \
    --group 1 \
    --minibatch_parallelism 1 \
    --batchsize "${BS}" \
    --seed "${SEED}" \
    --omp_num_threads 1 \
    --log "${LOG}"

echo "[$(date)] finished. Final metrics:"
tail -n 1 "${LOG}"
