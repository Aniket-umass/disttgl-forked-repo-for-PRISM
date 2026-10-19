# prep_tgb.py — step 2 of converting a TGB dataset for DistTGL (disttgl env).
#
#   cd /work/pi_mserafini_umass_edu/aniketmishra/disttgl
#   python prep_tgb.py --name tgbl-wiki [--node_feats_from WIKI] [--compare WIKI]
#
# Reads DATA/<name>/tgb_raw.npz (from tgb_export.py) and writes the files
# DistTGL expects: edges.csv, ext_full.npz (temporal CSR with both edge
# directions, neighbours sorted by time), edge_features.pt.
# --node_feats_from copies another dataset's learned_node_feats.pt (only valid
# when node ids are identical, e.g. tgbl-wiki and DistTGL's WIKI).
# --compare checks the output against an existing DistTGL dataset.
import argparse
import os

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--node_feats_from', type=str, default='')
parser.add_argument('--compare', type=str, default='')
args = parser.parse_args()

d = 'DATA/{}'.format(args.name)
raw = np.load('{}/tgb_raw.npz'.format(d))
src, dst, ts, split = raw['src'], raw['dst'], raw['ts'], raw['split'].astype(np.int64)
num_edges = len(src)
num_nodes = int(max(src.max(), dst.max())) + 1

df = pd.DataFrame({'src': src.astype(np.float64), 'dst': dst.astype(np.float64), 'time': ts.astype(np.float64),
                   'int_roll': split, 'ext_roll': split})
df.to_csv('{}/edges.csv'.format(d))

# Temporal CSR: every edge appears in both endpoints' lists, sorted by (time, eid).
node = np.concatenate([src, dst])
nbr = np.concatenate([dst, src])
eid = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
ets = np.concatenate([ts, ts])
order = np.lexsort((eid, ets, node))
indptr = np.zeros(num_nodes + 1, dtype=np.int64)
indptr[1:] = np.cumsum(np.bincount(node, minlength=num_nodes))
np.savez('{}/ext_full.npz'.format(d), indptr=indptr, indices=nbr[order].astype(np.int64),
         ts=ets[order].astype(np.float64), eid=eid[order].astype(np.int64))

# load_feat() caches edge_features_e0.pt (features + a zero row); drop any stale copy.
torch.save(torch.from_numpy(np.ascontiguousarray(raw['msg'])), '{}/edge_features.pt'.format(d))
if os.path.exists('{}/edge_features_e0.pt'.format(d)):
    os.remove('{}/edge_features_e0.pt'.format(d))

if args.node_feats_from:
    nf = torch.load('DATA/{}/learned_node_feats.pt'.format(args.node_feats_from))
    assert nf.shape[0] == num_nodes, (tuple(nf.shape), num_nodes)
    torch.save(nf, '{}/learned_node_feats.pt'.format(d))

print('{}: {} nodes (incl. unused node 0), {} edges, split {}'.format(
    args.name, num_nodes, num_edges, np.bincount(split).tolist()))

if args.compare:
    ref = pd.read_csv('DATA/{}/edges.csv'.format(args.compare))
    for c in ('src', 'dst', 'time', 'ext_roll'):
        print('edges.csv {:8s} identical: {}'.format(c, np.array_equal(ref[c].values, df[c].values)))
    g = np.load('DATA/{}/ext_full.npz'.format(args.compare))
    print('ext_full indptr identical:', np.array_equal(g['indptr'], indptr))
    rnode = np.repeat(np.arange(len(g['indptr']) - 1), np.diff(g['indptr']))
    ro = np.lexsort((g['eid'], g['ts'], rnode))
    for k, ours in (('indices', nbr[order]), ('eid', eid[order]), ('ts', ets[order])):
        print('ext_full {:7s} identical (ref re-sorted by time, eid): {}'.format(k, np.array_equal(g[k][ro], ours)))
    ef = torch.load('DATA/{}/edge_features.pt'.format(args.compare)).numpy()
    print('edge features max |diff|: {:.2e}'.format(float(np.abs(ef - raw['msg']).max())))
