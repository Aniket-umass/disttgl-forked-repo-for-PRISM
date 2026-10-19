# DistTGL with TGB-style evaluation (UMass changes)

Changes on top of amazon-science/disttgl `e36777d`, used to run DistTGL on Unity
(single node, RTX 2080 Ti) as a baseline.

## Running
- `--omp_num_threads 1` is required: with more threads the forked mailbox daemon
  deadlocks (0% GPU hang). The submit scripts pass it.
- `submit_tgb.sh`: one run on a TGB dataset; settings via env
  (`DATA BS EVAL_BS EPOCHS SEED EVAL_EVERY STATIC STRICT`, see its header).
  Minibatches are generated per (dataset, batch size) under `runs/`.
- Never run two DistTGL jobs on one node at once (fixed `/dev/shm` names).

## Data
- `tgb_export.py` (PRISM's py-tgb env): TGB dataset to `DATA/<name>/tgb_raw.npz`
  (node ids +1, fixed val/test negatives).
- `prep_tgb.py` (disttgl env): `edges.csv`, `ext_full.npz`, `edge_features.pt`
  in DistTGL's format. For tgbl-wiki the result is identical to DistTGL's WIKI.

## New options
- `gen_minibatch.py --batchsize/--eval_batchsize`: per-GPU training batch and
  the batch size for memory ingestion during val/test.
- `--tgb_eval` (gen_minibatch.py + train.py): rank each val/test edge against
  TGB's fixed negatives (~1000) with TGB's MRR (mean of optimistic and
  pessimistic rank). Scoring runs in chunks of `--eval_chunk` edges from the
  pre-batch state; memory is ingested once per eval batch, after scoring.
- `--tgb_strict`: eval neighbours only from edges before the eval batch, as in
  TGB, where the neighbour store is updated once per batch. Without it, an edge
  can see earlier edges of its own eval batch (~40% of neighbours on tgbl-wiki).
  Time deltas stay relative to each edge's own timestamp.
- `--no_static_memory` (train.py): drop DistTGL's static node memory
  (`learned_node_feats.pt`), i.e. plain TGN memory.
- `--epochs`, `--eval_every N` (last epoch is always validated), `val time` in the log.
- `layers.py`: eval-only chunked attention for very large blocks (8192 batches
  with TGB negatives); equal to the original within ~1e-6.
- The mailbox daemon is `daemon=True`, so a crash exits instead of hanging.
