#!/bin/bash -l
# submit_tgb.sh — DistTGL on a TGB dataset with TGB-style evaluation (TGB's
# fixed negatives, TGB's MRR), single node, N x 2080 Ti.
#
# Prerequisite: the dataset converted into DATA/<DATA>/ with tgb_export.py and
# prep_tgb.py (see their headers).
#
#   DATA=tgbl-wiki BS=600 sbatch -p gpu,gpu-preempt --nodes=1 --gpus=1 -c 3 --mem=40G submit_tgb.sh
#
# BS       per-GPU training batch size
# EVAL_BS  val/test memory-ingestion batch size (default: BS)
# EPOCHS   override the number of epochs (default: DistTGL's, 100 / #GPUs)
# SEED     default 0; SEED>0 needs the seed-0 run dir (reuses its TGB eval minibatches)
# EVAL_EVERY  validate every N epochs (default 1; last epoch always validated)
# STRICT   1 = eval neighbours only from edges before each eval batch (TGB protocol)
# STATIC   1 (default) keeps DistTGL static memory; 0 = plain TGN (--no_static_memory)
# Each (DATA, BS, EVAL_BS) gets its own working dir runs/<DATA>_tgb_bs<BS>_eval<EVAL_BS>
# because DistTGL's minibatch paths don't include the batch size; minibatches
# are generated on first use. Never run two DistTGL jobs on one node at once
# (fixed /dev/shm names): chain with --dependency=afterany:<jobid>.
#
#SBATCH --job-name=disttgl-tgb
#SBATCH --constraint=2080ti
#SBATCH --time=12:00:00
#SBATCH --output=/work/pi_mserafini_umass_edu/aniketmishra/disttgl/logs/%x_%j.out
#SBATCH --error=/work/pi_mserafini_umass_edu/aniketmishra/disttgl/logs/%x_%j.err

set -eo pipefail

REPO=/work/pi_mserafini_umass_edu/aniketmishra/disttgl
DATA="${DATA:?set DATA, e.g. tgbl-wiki}"
BS="${BS:?set BS}"
EVAL_BS="${EVAL_BS:-${BS}}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-0}"
EVAL_EVERY="${EVAL_EVERY:-1}"
STATIC="${STATIC:-1}"
STRICT="${STRICT:-0}"
BASE_RUN="${REPO}/runs/${DATA}_tgb_bs${BS}_eval${EVAL_BS}"
EVAL_DIR="${DATA}_tgb_eval"
# STRICT=1: own run dirs, eval neighbours limited to edges before each eval batch
if (( STRICT == 1 )); then BASE_RUN="${BASE_RUN}_strict"; EVAL_DIR="${EVAL_DIR}_strict"; fi
RUN="${BASE_RUN}"
# SEED>0: own training minibatches (negatives drawn with --seed SEED); the TGB
# eval minibatches don't depend on the seed and are shared from the seed-0 dir.
(( SEED > 0 )) && RUN="${BASE_RUN}_s${SEED}"

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate disttgl
export PYTHONUNBUFFERED=1

mkdir -p "${RUN}" "${REPO}/results" "${REPO}/logs"
cd "${RUN}"
ln -sfn "${REPO}/DATA" DATA

NGPU="${SLURM_GPUS_ON_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)}"
TAG=""
(( EPOCHS > 0 )) && TAG="_ep${EPOCHS}"
(( STATIC == 0 )) && TAG="${TAG}_nostatic"
(( STRICT == 1 )) && TAG="${TAG}_strict"
LOG="${REPO}/results/${DATA}_tgb_bs${BS}_eval${EVAL_BS}_1x${NGPU}x1_seed${SEED}${TAG}.log"

echo "[$(date)] DistTGL ${DATA} (TGB eval) | train bs ${BS}/GPU | eval bs ${EVAL_BS} | ${NGPU} GPU(s) | seed ${SEED} | val every ${EVAL_EVERY} | static memory ${STATIC} | strict ${STRICT} | $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -f minibatches/.done ]]; then
  echo "[$(date)] generating minibatches (train bs ${BS}, eval bs ${EVAL_BS}, TGB negatives)"
  rm -rf minibatches
  STRICT_FLAG=()
  (( STRICT == 1 )) && STRICT_FLAG=(--tgb_strict)
  if (( SEED > 0 )); then
    [[ -f "${BASE_RUN}/minibatches/.done" ]] || { echo "run seed 0 first: ${BASE_RUN}/minibatches missing"; exit 1; }
    mkdir minibatches
    ln -s "${BASE_RUN}/minibatches/${EVAL_DIR}" "minibatches/${EVAL_DIR}"
    python "${REPO}/gen_minibatch.py" --data "${DATA}" --minibatch_parallelism 1 \
      --batchsize "${BS}" --seed "${SEED}"
  else
    python "${REPO}/gen_minibatch.py" --data "${DATA}" --gen_eval --tgb_eval --minibatch_parallelism 1 \
      --batchsize "${BS}" --eval_batchsize "${EVAL_BS}" --seed "${SEED}" "${STRICT_FLAG[@]}"
  fi
  touch minibatches/.done
  echo "[$(date)] generation done"
fi
echo "minibatches: $(du -sh minibatches | cut -f1) | train batches: $(ls minibatches/${DATA}_1_49_32 | grep -c train_pos)"

EXTRA=()
(( EPOCHS > 0 )) && EXTRA+=(--epochs "${EPOCHS}")
(( STATIC == 0 )) && EXTRA+=(--no_static_memory)
(( STRICT == 1 )) && EXTRA+=(--tgb_strict)

# --omp_num_threads 1: with more threads the forked mailbox daemon deadlocks.
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  "${REPO}/train.py" \
    --data "${DATA}" \
    --group 1 \
    --minibatch_parallelism 1 \
    --batchsize "${BS}" \
    --tgb_eval \
    --eval_every "${EVAL_EVERY}" \
    --seed "${SEED}" \
    --omp_num_threads 1 \
    "${EXTRA[@]}" \
    --log "${LOG}"

echo "[$(date)] finished. Final metrics:"
tail -n 1 "${LOG}"
