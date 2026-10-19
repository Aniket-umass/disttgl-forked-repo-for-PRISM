# tgb_export.py — step 1 of converting a TGB link-prediction dataset for DistTGL.
#
# Run in the PRISM env (prism-pyg) from the PRISM repo root, so it uses PRISM's
# own TGB loader and dataset copy (the installed tgb package resolves the
# dataset root inside its install dir instead):
#
#   cd /work/pi_mserafini_umass_edu/aniketmishra/PRISM
#   python ../disttgl/tgb_export.py --name tgbl-wiki
#
# Writes <disttgl>/DATA/<name>/tgb_raw.npz with plain numpy arrays (node ids
# shifted by +1, as in DistTGL's own datasets, where node 0 is unused):
#   src, dst, ts, msg (float32 edge features), split (0 train / 1 val / 2 test),
#   {val,test}_neg  [n_edges_in_split, K] TGB's fixed negative destinations,
#                   short lists padded with their first entry,
#   {val,test}_cnt  [n_edges_in_split] number of real negatives per edge.
# Step 2 is prep_tgb.py in the disttgl env.
import argparse
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='TGB dataset name, e.g. tgbl-wiki')
parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA'))
args = parser.parse_args()

sys.path.insert(0, os.getcwd())
from modules.dataset import LinkPropPredDataset

ds = LinkPropPredDataset(name=args.name, root='datasets', preprocess=True)
d = ds.full_data
src = np.asarray(d['sources'], dtype=np.int64)
dst = np.asarray(d['destinations'], dtype=np.int64)
ts = np.asarray(d['timestamps'])
msg = np.asarray(d['edge_feat'], dtype=np.float32)
if msg.ndim == 1:
    msg = msg.reshape(-1, 1)
train_mask, val_mask, test_mask = (np.asarray(m) for m in (ds.train_mask, ds.val_mask, ds.test_mask))

# DistTGL assumes chronologically sorted edges with contiguous train/val/test.
assert (np.diff(ts) >= 0).all(), 'timestamps not sorted'
split = np.where(train_mask, 0, np.where(val_mask, 1, 2)).astype(np.int8)
assert (np.diff(split) >= 0).all(), 'splits are not contiguous'


def fixed_negatives(mode, mask):
    if mode == 'val':
        ds.load_val_ns()
    else:
        ds.load_test_ns()
    lists = ds.negative_sampler.query_batch(src[mask], dst[mask], ts[mask], split_mode=mode)
    cnt = np.array([len(x) for x in lists], dtype=np.int64)
    neg = np.empty((len(lists), int(cnt.max())), dtype=np.int64)
    for i, x in enumerate(lists):
        x = np.asarray(x, dtype=np.int64)
        neg[i, :len(x)] = x
        neg[i, len(x):] = x[0]
    return neg + 1, cnt


val_neg, val_cnt = fixed_negatives('val', val_mask)
test_neg, test_cnt = fixed_negatives('test', test_mask)

out_dir = os.path.join(args.out, args.name)
os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir, 'tgb_raw.npz'),
         src=src + 1, dst=dst + 1, ts=ts.astype(np.float64), msg=msg, split=split,
         val_neg=val_neg, val_cnt=val_cnt, test_neg=test_neg, test_cnt=test_cnt)
print('{}: {} edges, {} nodes, edge feat dim {}, split {} / {} / {}'.format(
    args.name, len(src), int(max(src.max(), dst.max())) + 2, msg.shape[1],
    int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())))
print('negatives per edge: val {}-{}, test {}-{}'.format(val_cnt.min(), val_cnt.max(), test_cnt.min(), test_cnt.max()))
print('wrote', os.path.join(out_dir, 'tgb_raw.npz'))
