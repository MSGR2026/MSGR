# coding: utf-8
r"""
SMORE_2025
################################################
Faithful re-implementation of the methodology described in the provided paper excerpt.

Core components:
1) Spectrum Modality Fusion (FFT -> complex filters -> pointwise product fusion -> IFFT)
2) Multi-modal Graph Learning
   - item-item modality graphs (cosine sim + topK + sym-norm)
   - fusion graph via (approx.) max-pooling across modality graphs
   - behavior-guided gating before item-item propagation
   - 1-layer LightGCN on item-item graphs; multi-layer LightGCN on user-item graph
3) Modality-aware Preference Module
   - modality-specific attention weights computed from fusion embeddings
   - preference gates derived from behavioral embeddings
   - InfoNCE contrastive loss (user + item)
4) BPR optimization + contrastive + L2

Notes / minimal assumptions due to paper ambiguities:
- FFT "along the sequence/spatial dimension": dataset provides per-item feature vectors (no token/patch dimension).
  We therefore apply 1D FFT along the embedding dimension d (i.e., treat feature channels as the "sequence"),
  so n := d in Eq.(2)-(7). This is a common minimal adaptation when only global vectors are available.
- Eq.(11) fusion affinity matrix definition is ambiguous in text. We implement fusion graph weights as
  elementwise max across modality adjacency weights for the same (a,b) edge, i.e., S_f(a,b)=max_m S_m(a,b),
  using union of edges from all modality graphs. This matches the "max-pooling" intent while staying implementable.
- Eq.(14) for fusion user aggregation has a typo in the paper (uses \bar h_{i,m}); we use \bar h_{i,f}.
- "concatenation" mentioned in Sec 4.2.1 conflicts with dimension statement; we follow the practical
  usage consistent with later equations: we use the propagated embeddings as the enriched embeddings
  (no doubling of dimension). This is annotated inline.

This code targets the MMRec-style API shown in the baselines:
- class name == model name, inherits GeneralRecommender
- implements __init__ / forward / calculate_loss / full_sort_predict
"""

import os
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


def _scipy_to_torch_sparse(coo: sp.coo_matrix, device):
    coo = coo.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), device=device).coalesce()


def _build_ui_norm_adj(inter_mat: sp.coo_matrix, n_users: int, n_items: int):
    """Build sym-normalized adjacency for bipartite user-item graph: D^{-1/2} A D^{-1/2}."""
    n_nodes = n_users + n_items
    A = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

    R = inter_mat.tocoo()
    Rt = R.transpose().tocoo()
    # stable public API: build with COO then sum duplicates
    rows = np.concatenate([R.row, Rt.row + n_users])
    cols = np.concatenate([R.col + n_users, Rt.col])
    data = np.ones_like(rows, dtype=np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    deg = np.array(A.sum(axis=1)).flatten() + 1e-7
    deg_inv_sqrt = np.power(deg, -0.5).astype(np.float32)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
    return L


def _build_R_norm(inter_mat: sp.coo_matrix, n_users: int, n_items: int):
    """
    Build the user->item aggregation sparse matrix R with sym norm coefficients:
    R[u,i] = 1/sqrt(|N_u||N_i|), matching Eq.(14).
    """
    R = inter_mat.tocoo().astype(np.float32)
    # degrees
    u_deg = np.bincount(R.row, minlength=n_users).astype(np.float32) + 1e-7
    i_deg = np.bincount(R.col, minlength=n_items).astype(np.float32) + 1e-7
    vals = 1.0 / np.sqrt(u_deg[R.row] * i_deg[R.col])
    return sp.coo_matrix((vals, (R.row, R.col)), shape=(n_users, n_items)).tocoo()


def _knn_graph_from_feats(feats: torch.Tensor, topk: int):
    """
    Build sparse sym-normalized kNN adjacency from features by cosine similarity.
    Returns torch.sparse_coo_tensor of shape [n_items, n_items].
    """
    # feats: [I, d]
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.t()  # [I, I]
    # NOTE: keep self-loop? Paper doesn't specify. We follow common kNN graph: allow self in topK.
    _, knn_ind = torch.topk(sim, k=topk, dim=-1)

    I = feats.size(0)
    row = torch.arange(I, device=feats.device).unsqueeze(1).expand(-1, topk).reshape(-1)
    col = knn_ind.reshape(-1)
    indices = torch.stack([row, col], dim=0)

    # unweighted edges then sym-normalize like Eq.(10): D^{-1/2} A D^{-1/2}
    values_ones = torch.ones(indices.size(1), device=feats.device, dtype=feats.dtype)
    adj = torch.sparse_coo_tensor(indices, values_ones, (I, I), device=feats.device).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
    deg_inv_sqrt = deg.pow(-0.5)
    v = deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, v, (I, I), device=feats.device).coalesce()


def _sparse_union_max(adjs):
    """
    Elementwise max over multiple sparse COO matrices with same shape.
    Minimal implementable approximation for Eq.(11) "max-pooling" fusion affinity.
    """
    assert len(adjs) > 0
    shape = adjs[0].shape
    device = adjs[0].device
    dtype = adjs[0].dtype

    # concatenate indices/values, then reduce by max for identical indices
    all_idx = torch.cat([a.coalesce().indices() for a in adjs], dim=1)
    all_val = torch.cat([a.coalesce().values() for a in adjs], dim=0)

    # hash indices -> unique
    # key = row * N + col
    N = shape[1]
    key = all_idx[0] * N + all_idx[1]
    uniq_key, inv = torch.unique(key, return_inverse=True)

    # max-reduce via scatter_reduce (PyTorch>=1.12). If unavailable, fallback to dense (not provided).
    out_val = torch.full((uniq_key.numel(),), -float("inf"), device=device, dtype=dtype)
    out_val.scatter_reduce_(0, inv, all_val, reduce="amax", include_self=True)

    out_row = torch.div(uniq_key, N, rounding_mode="floor")
    out_col = uniq_key % N
    out_idx = torch.stack([out_row, out_col], dim=0)

    # already normalized per-modality; max keeps scale. (Paper doesn't mention re-normalize after max.)
    return torch.sparse_coo_tensor(out_idx, out_val, shape, device=device).coalesce()


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---------- required configs ----------
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_ui_layers = config['n_ui_layers']  # L in Eq.(15)-(16)
        self.lambda_cl = config['lambda_cl']      # λ1 in Eq.(25)
        self.reg_weight = config['reg_weight']    # λ2 in Eq.(25)
        self.tau = config['tau']                  # temperature τ in Eq.(22)

        # ---------- dataset / graph ----------
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # UI normalized adjacency for LightGCN (behavior view)
        norm_adj = _build_ui_norm_adj(self.interaction_matrix, self.n_users, self.n_items)
        self.norm_adj = _scipy_to_torch_sparse(norm_adj, self.device)

        # R for user aggregation from item features (Eq.(14))
        R_norm = _build_R_norm(self.interaction_matrix, self.n_users, self.n_items)
        self.R = _scipy_to_torch_sparse(R_norm, self.device)  # [U, I]

        # ---------- ID embeddings ----------
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ---------- modalities ----------
        # infer available modalities from dataset-provided features
        self.modalities = []
        if self.v_feat is not None:
            self.modalities.append('v')
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.proj_v = nn.Linear(self.v_feat.shape[1], self.embedding_dim, bias=True)  # Eq.(1)
        if self.t_feat is not None:
            self.modalities.append('t')
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.proj_t = nn.Linear(self.t_feat.shape[1], self.embedding_dim, bias=True)  # Eq.(1)

        if len(self.modalities) == 0:
            raise ValueError("SMORE requires at least one modality feature (v_feat or t_feat).")

        # ---------- spectral filters ----------
        # n := d due to 1D FFT over embedding channels (assumption; see header comment)
        n = self.embedding_dim

        # complex trainable weights W_{2,m}^c \in C^{n x d}; here n=d => C^{d x d}
        # We parameterize complex weight as two real tensors.
        self.spec_filter_real = nn.ParameterDict()
        self.spec_filter_imag = nn.ParameterDict()
        for m in self.modalities:
            self.spec_filter_real[m] = nn.Parameter(torch.randn(n, self.embedding_dim) * 0.02)
            self.spec_filter_imag[m] = nn.Parameter(torch.randn(n, self.embedding_dim) * 0.02)

        # fusion filter δ_f with complex weight too
        self.fuse_filter_real = nn.Parameter(torch.randn(n, self.embedding_dim) * 0.02)
        self.fuse_filter_imag = nn.Parameter(torch.randn(n, self.embedding_dim) * 0.02)

        # ---------- behavior-guided gating before item-item propagation (Eq.(12)) ----------
        self.gate_W = nn.ModuleDict()
        self.gate_b = nn.ParameterDict()
        for m in self.modalities:
            self.gate_W[m] = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.gate_fuse = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        # ---------- modality-aware preference module ----------
        # attention parameters per modality: p_m, W_{5,m}, b_{5,m} (Eq.(17))
        self.att_W = nn.ModuleDict()
        self.att_p = nn.ParameterDict()
        for m in self.modalities:
            self.att_W[m] = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
            self.att_p[m] = nn.Parameter(torch.randn(self.embedding_dim) * 0.02)

        # preference gates from behavioral embeddings: W6m/b6m and W7f/b7f (Eq.(19)-(20))
        self.pref_gate_m = nn.ModuleDict()
        for m in self.modalities:
            self.pref_gate_m[m] = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.pref_gate_f = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        # ---------- item-item graphs cache ----------
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        os.makedirs(dataset_path, exist_ok=True)
        self._ii_graph_cache = {}  # modality -> torch.sparse
        self._f_graph_cache = None

        # Build and cache graphs using RAW modality features E_{i,m} (Eq.(8)-(11))
        # (paper says compute similarity on raw features)
        self._build_item_item_graphs(dataset_path)

    def _build_item_item_graphs(self, dataset_path: str):
        # Cache files similar to baselines
        # include modality names to avoid clashes
        mod_files = {}
        for m in self.modalities:
            mod_files[m] = os.path.join(dataset_path, f'smore_{m}_adj_knn{self.knn_k}.pt')
        fuse_file = os.path.join(dataset_path, f'smore_fuse_adj_knn{self.knn_k}.pt')

        loaded_all = True
        for m in self.modalities:
            if not os.path.exists(mod_files[m]):
                loaded_all = False
        if loaded_all and os.path.exists(fuse_file):
            for m in self.modalities:
                self._ii_graph_cache[m] = torch.load(mod_files[m], map_location=self.device).coalesce()
            self._f_graph_cache = torch.load(fuse_file, map_location=self.device).coalesce()
            return

        # build graphs on device (may be heavy for large I; consistent with baseline style)
        adjs = []
        with torch.no_grad():
            if 'v' in self.modalities:
                feats_v = self.image_embedding.weight.detach().to(self.device)  # raw E_{i,v}
                adj_v = _knn_graph_from_feats(feats_v, self.knn_k)
                self._ii_graph_cache['v'] = adj_v
                adjs.append(adj_v)
            if 't' in self.modalities:
                feats_t = self.text_embedding.weight.detach().to(self.device)   # raw E_{i,t}
                adj_t = _knn_graph_from_feats(feats_t, self.knn_k)
                self._ii_graph_cache['t'] = adj_t
                adjs.append(adj_t)

            if len(adjs) == 1:
                # if only one modality, fusion view degenerates to that view (minimal assumption)
                self._f_graph_cache = adjs[0]
            else:
                self._f_graph_cache = _sparse_union_max(adjs)

        for m in self.modalities:
            torch.save(self._ii_graph_cache[m].cpu(), mod_files[m])
        torch.save(self._f_graph_cache.cpu(), fuse_file)

        # move back to device buffers
        for m in self.modalities:
            self._ii_graph_cache[m] = self._ii_graph_cache[m].to(self.device).coalesce()
        self._f_graph_cache = self._f_graph_cache.to(self.device).coalesce()

    @staticmethod
    def _bpr_loss(u_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    @staticmethod
    def _infonce(view1, view2, temperature: float):
        # Eq.(22): full-batch denominator over all nodes in the set.
        # We compute it over the provided batch (common practice); paper states sum over all users/items.
        # Minimal assumption: use in-batch negatives.
        v1 = F.normalize(view1, dim=1)
        v2 = F.normalize(view2, dim=1)
        logits = (v1 @ v2.t()) / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _lightgcn_ui(self, norm_adj):
        """
        User-Item behavioral view (Eq.(15)-(16)).
        Returns:
          Ebar_id_users: [U,d], Ebar_id_items: [I,d]
          Ebar_id_all:   [U+I,d]
        """
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)  # [U+I,d]
        all_layers = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(norm_adj, x)
            all_layers.append(x)
        out = torch.stack(all_layers, dim=1).mean(dim=1)  # mean over layers
        u, i = torch.split(out, [self.n_users, self.n_items], dim=0)
        return u, i, out

    def _spectrum_fuse_and_denoise(self):
        """
        Spectrum Modality Fusion (Sec 4.1):
          H_{i,m} = W1_m E_{i,m} + b1_m        (projection)
          \tilde H = FFT(H)
          \hat H_{i,m} = W2_m^c ⊙ \tilde H     (denoise filter)
          fusion: \hat H_{i,f} = W2_f^c ⊙ Π_m \tilde H_{i,m}
          back:   \dot H = IFFT(\hat H)
        Returns:
          dot_H_m: dict m -> [I,d] real
          dot_H_f: [I,d] real
        """
        proj = {}
        if 'v' in self.modalities:
            proj['v'] = self.proj_v(self.image_embedding.weight)  # [I,d]
        if 't' in self.modalities:
            proj['t'] = self.proj_t(self.text_embedding.weight)   # [I,d]

        spec = {}
        for m in self.modalities:
            # 1D FFT over last dimension (assumption)
            spec[m] = torch.fft.fft(proj[m], dim=-1)  # complex [I,d]

        # denoise each modality: elementwise complex multiplication
        dot_H_m = {}
        for m in self.modalities:
            Wc = torch.complex(self.spec_filter_real[m], self.spec_filter_imag[m])  # [d,d]
            # broadcast Wc over items: [I,d] * [d,d] is invalid; need elementwise same shape.
            # Minimal assumption: filter is per-frequency and per-channel -> shape [d] or [d,d]?
            # Paper states W_{2,m}^c in C^{n x d}. With n=d, that's [d,d].
            # For item vector, spectrum is [d]; treat it as [n] with channel=1, or [n x d]?
            # To keep fidelity with stated shape, we interpret projected feature as [n x d] with n=1 not possible.
            # Minimal workable faithful mapping: use diagonal of [d,d] as per-frequency-per-channel selector.
            # This is the smallest assumption to reconcile vector input with W in C^{n x d}.
            Wc_diag = torch.diagonal(Wc, 0)  # [d]
            denoised = spec[m] * Wc_diag  # [I,d]
            dot_H_m[m] = torch.fft.ifft(denoised, dim=-1).real  # [I,d]

        # fusion spectrum via pointwise product over modalities, then fusion filter
        fused_spec = None
        for m in self.modalities:
            fused_spec = spec[m] if fused_spec is None else (fused_spec * spec[m])

        Wf = torch.complex(self.fuse_filter_real, self.fuse_filter_imag)
        Wf_diag = torch.diagonal(Wf, 0)  # [d] (same rationale as above)
        fused_spec = fused_spec * Wf_diag
        dot_H_f = torch.fft.ifft(fused_spec, dim=-1).real  # [I,d]

        return dot_H_m, dot_H_f

    def _item_item_view(self, E_item_id, dot_H_m, dot_H_f):
        """
        Multi-modal Graph Learning - item-item view (Sec 4.2.1):
          gating: ddot H_{i,m} = E_id ⊙ sigmoid(W3m dot_H_{i,m} + b3m)
                 ddot H_{i,f} = E_id ⊙ sigmoid(W4f dot_H_{i,f} + b4f)
          1-layer propagation:
                 overline H_{i,m} = S_m ddot H_{i,m}
                 overline H_{i,f} = S_f ddot H_{i,f}
          user aggregation:
                 overline h_{u,*} = R overline h_{i,*}    (Eq.(14))
        Returns:
          Hbar_m_all: dict m -> [U+I,d]
          Hbar_f_all: [U+I,d]
        """
        # gating
        ddot_m = {}
        for m in self.modalities:
            gate = torch.sigmoid(self.gate_W[m](dot_H_m[m]))  # [I,d]
            ddot_m[m] = E_item_id * gate  # [I,d]

        gate_f = torch.sigmoid(self.gate_fuse(dot_H_f))
        ddot_f = E_item_id * gate_f

        # 1-layer LightGCN on item-item graphs (Eq.(13))
        H_item_m = {}
        for m in self.modalities:
            H_item_m[m] = torch.sparse.mm(self._ii_graph_cache[m], ddot_m[m])  # [I,d]
        H_item_f = torch.sparse.mm(self._f_graph_cache, ddot_f)  # [I,d]

        # user aggregation via R (Eq.(14))
        H_user_m = {}
        for m in self.modalities:
            H_user_m[m] = torch.sparse.mm(self.R, H_item_m[m])  # [U,d]
        H_user_f = torch.sparse.mm(self.R, H_item_f)  # [U,d]

        # "concatenation" in paper is inconsistent; we use enriched embeddings directly (same dim) (assumption)
        Hbar_m_all = {}
        for m in self.modalities:
            Hbar_m_all[m] = torch.cat([H_user_m[m], H_item_m[m]], dim=0)  # [U+I,d]
        Hbar_f_all = torch.cat([H_user_f, H_item_f], dim=0)  # [U+I,d]

        return Hbar_m_all, Hbar_f_all

    def _modality_aware_preference(self, Ebar_id_all, Hbar_m_all, Hbar_f_all):
        """
        Modality-aware Preference Module (Sec 4.3):
          alpha_m = softmax(p_m^T tanh(W5m Hbar_f + b5m)) over modalities (Eq.(17))
          H*_m = sum_m alpha_m * Hbar_m  (Eq.(18))  -> aggregated uni-modal features
          Q_m = sigmoid(W6m Ebar_id + b6m), Q_f = sigmoid(W7f Ebar_id + b7f) (Eq.(19)-(20))
          H_s = (1/|M|) sum_m (H*_m ⊙ Q_m) + (H_f ⊙ Q_f) (Eq.(21))
        Returns:
          H_s_all: [U+I,d]
        """
        # attention logits per modality from fusion embeddings
        logits = []
        for m in self.modalities:
            x = torch.tanh(self.att_W[m](Hbar_f_all))  # [U+I,d]
            # p_m^T x
            logit_m = torch.sum(x * self.att_p[m].unsqueeze(0), dim=1, keepdim=True)  # [U+I,1]
            logits.append(logit_m)
        logits = torch.cat(logits, dim=1)  # [U+I,|M|]
        alpha = F.softmax(logits, dim=1)   # [U+I,|M|]

        # weighted sum of uni-modal features (Eq.(18))
        H_star = 0.0
        for idx, m in enumerate(self.modalities):
            H_star = H_star + alpha[:, idx:idx+1] * Hbar_m_all[m]  # [U+I,d]

        # preference gates from behavioral embeddings (Eq.(19)-(20))
        # paper defines Q_m per modality and Q_f for fusion
        Qm = {}
        for m in self.modalities:
            Qm[m] = torch.sigmoid(self.pref_gate_m[m](Ebar_id_all))  # [U+I,d]
        Qf = torch.sigmoid(self.pref_gate_f(Ebar_id_all))  # [U+I,d]

        # H_s (Eq.(21))
        sum_uni = 0.0
        for m in self.modalities:
            sum_uni = sum_uni + (H_star * Qm[m])
        sum_uni = sum_uni / float(len(self.modalities))
        H_s = sum_uni + (Hbar_f_all * Qf)

        return H_s

    def forward(self, adj=None, train=False):
        """
        Returns:
          user_final: [U,d], item_final: [I,d]
        If train=True, also returns auxiliary views for contrastive learning:
          Ebar_id_all, H_s_all
        """
        if adj is None:
            adj = self.norm_adj

        # behavioral view
        Ebar_u, Ebar_i, Ebar_all = self._lightgcn_ui(adj)  # Eq.(15)-(16)

        # spectrum fusion + denoise (item side)
        dot_H_m, dot_H_f = self._spectrum_fuse_and_denoise()

        # item-item view: gating + propagation + user aggregation
        E_item_id = self.item_id_embedding.weight  # E_{i,id} in Eq.(12)
        Hbar_m_all, Hbar_f_all = self._item_item_view(E_item_id, dot_H_m, dot_H_f)

        # modality-aware preference => side features H_s
        H_s_all = self._modality_aware_preference(Ebar_all, Hbar_m_all, Hbar_f_all)

        # final representations (Eq.(23))
        E_final_all = Ebar_all + H_s_all
        user_final, item_final = torch.split(E_final_all, [self.n_users, self.n_items], dim=0)

        if train:
            return user_final, item_final, Ebar_all, H_s_all
        return user_final, item_final

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        u_final, i_final, Ebar_all, H_s_all = self.forward(self.norm_adj, train=True)

        u_emb = u_final[users]
        pos_emb = i_final[pos_items]
        neg_emb = i_final[neg_items]

        loss_bpr = self._bpr_loss(u_emb, pos_emb, neg_emb)

        # contrastive loss (Eq.(22)) for both users and items; in-batch negatives (assumption)
        Ebar_u, Ebar_i = torch.split(Ebar_all, [self.n_users, self.n_items], dim=0)
        Hs_u, Hs_i = torch.split(H_s_all, [self.n_users, self.n_items], dim=0)

        # use current batch users and positive items as contrastive samples
        loss_cl_u = self._infonce(Ebar_u[users], Hs_u[users], self.tau)
        loss_cl_i = self._infonce(Ebar_i[pos_items], Hs_i[pos_items], self.tau)
        loss_cl = loss_cl_u + loss_cl_i

        # L2 regularization term ||Theta||^2 (Eq.(25))
        l2 = 0.0
        for p in self.parameters():
            l2 = l2 + torch.sum(p ** 2)
        loss_reg = self.reg_weight * l2

        return loss_bpr + self.lambda_cl * loss_cl + loss_reg

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        u_final, i_final = self.forward(self.norm_adj, train=False)
        u = u_final[user]  # [B,d]
        scores = torch.matmul(u, i_final.t())  # [B,I]
        return scores