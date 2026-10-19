import torch
import dgl
import math
import numpy as np

# In eval mode, blocks with more edges than this go through the attention layer
# in chunks of destination nodes (same result, bounded GPU memory). Needed when
# the eval batch size is large (e.g. 8192 edges x 49 negatives on an 11 GB GPU).
EVAL_CHUNK_EDGES = 500000

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, pos_rst, neg_rst):
        num_edge = pos_rst.shape[0] // 2
        h_src = self.src_fc(pos_rst[:num_edge])
        h_pos_dst = self.dst_fc(pos_rst[num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        prob_pos = self.out_fc(h_pos_edge)
        prob_neg = None
        if neg_rst is not None:
            h_neg_dst = self.dst_fc(neg_rst)
            neg_samples = h_neg_dst.shape[0] // h_src.shape[0]
            h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
            prob_neg = self.out_fc(h_neg_edge)
        return prob_pos, prob_neg


class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def _forward_chunked(self, b, chunk_edges):
        # Same computation as forward(), but K/V/attention are built for a slice of
        # destination nodes at a time instead of for every edge at once.
        device = b.src_idx_cuda.device
        h = b.srcdata['h']
        idx = b.src_idx_cuda
        n_dst = b.num_dst_nodes()
        if b.combined:
            q_idx = torch.cat([idx[:b.num_pos_dst], idx[b.num_pos_idx:b.num_pos_idx + b.num_neg_dst]], dim=0)
            kv_idx = torch.cat([idx[b.num_pos_dst:b.num_pos_idx], idx[b.num_pos_idx + b.num_neg_dst:]], dim=0)
        else:
            q_idx = idx[:n_dst]
            kv_idx = idx[n_dst:]
        q_data = h[q_idx]
        zero_time_feat = self.time_enc(torch.zeros(n_dst, dtype=torch.float32, device=device))
        Qn = self.w_q(torch.cat([q_data, zero_time_feat], dim=1))
        row = b.edges()[1]
        row_s, order = torch.sort(row)
        num_e = row_s.shape[0]
        agg = torch.zeros((n_dst, self.dim_out), device=device)
        start = 0
        while start < num_e:
            end = min(start + chunk_edges, num_e)
            if end < num_e:
                # never split one destination node's edges across chunks
                end = int(torch.searchsorted(row_s, row_s[end - 1], right=True))
            e = order[start:end]
            r = row_s[start:end]
            d0 = int(r[0])
            d1 = int(r[-1]) + 1
            time_feat = self.time_enc(b.edata['dt'][e])
            if self.dim_edge_feat == 0:
                kv = torch.cat([h[kv_idx[e]], time_feat], dim=1)
            else:
                kv = torch.cat([h[kv_idx[e]], b.edata['f'][e], time_feat], dim=1)
            Q = Qn[r]
            K = self.w_k(kv)
            V = self.w_v(kv)
            del kv
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            g = dgl.create_block((torch.arange(end - start, device=device), r - d0), num_src_nodes=end - start, num_dst_nodes=d1 - d0)
            att = dgl.ops.edge_softmax(g, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            agg[d0:d1] += dgl.ops.copy_e_sum(g, V)
            start = end
        rst = torch.cat([agg, q_data], dim=1)
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

    def forward(self, b):
        if (not self.training) and (not self.combined) and self.dim_time > 0 and b.num_edges() > EVAL_CHUNK_EDGES:
            return self._forward_chunked(b, EVAL_CHUNK_EDGES)
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=b.src_idx_cuda.device))
        src_data = b.srcdata['h'][b.src_idx_cuda]
        if b.combined:
            q_data = torch.cat([src_data[:b.num_pos_dst], src_data[b.num_pos_idx:b.num_pos_idx + b.num_neg_dst]], dim=0)
            kv_data = torch.cat([src_data[b.num_pos_dst:b.num_pos_idx], src_data[b.num_pos_idx + b.num_neg_dst:]], dim=0)
        else:
            q_data = src_data[:b.num_dst_nodes()]
            kv_data = src_data[b.num_dst_nodes():]
        if b.num_edges() > 0:
            if self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([q_data, zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([kv_data, time_feat], dim=1))
                V = self.w_v(torch.cat([kv_data, time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([q_data, zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([kv_data, b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([kv_data, b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            b.dstdata['h'] = torch.zeros((b.num_dst_nodes(), self.dim_out), device=b.src_idx_cuda.device)
        if b.combined:
            rst_pos = torch.cat([b.dstdata['h'][:b.num_pos_dst], src_data[:b.num_pos_dst]], dim=1)
            rst_neg = torch.cat([b.dstdata['h'][b.num_pos_dst:], src_data[b.num_pos_idx:b.num_pos_idx + b.num_neg_dst]], dim=1)
            rst = torch.cat([rst_pos, rst_neg])
        else:
            rst = torch.cat([b.dstdata['h'], src_data[:b.num_dst_nodes()]], dim=1)
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            