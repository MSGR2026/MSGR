# coding: utf-8
r"""
PGL_2025
################################################
Paper: PGL (2025): Principal Graph Learning for Multimedia Recommendation

Core idea:
- Principal Graph Learning on user-item graph:
  - Training: message passing on extracted principal subgraph F(\tilde{A})
  - Inference: message passing on complete graph \tilde{A}
  - Two extraction operators:
    * global-aware: randomized SVD + truncated reconstruction + sparsification
    * local-aware: degree-sensitive multinomial edge sampling
- Latent Graph Learning on fixed item-item similarity graphs from modality features:
  - build kNN graph (k=10) with Laplacian normalization; weighted sum across modalities
- Optimization:
  - BPR loss + lambda_ssl * SSL loss (feature masking + in-batch InfoNCE) on final fused embeddings

Assumptions (minimal, due to paper ambiguity):
1) Message passing H is implemented as LightGCN-style propagation with mean pooling across layers
   (common for graph CF; paper does not specify layer aggregation).
2) For global-aware extraction, we approximate randomized SVD with torch.pca_lowrank on dense
   normalized adjacency; this is only feasible for small graphs. For large datasets, user should
   switch to 'local' extraction (recommended).
3) Truncation keeps eigenvalues in [0, gamma*d) and [(1-gamma)*d, d) as described; remaining set to 0.
4) Principal subgraph is symmetrically normalized before message passing (D^-1/2 A D^-1/2),
   consistent with typical GCN/LightGCN stability, though not explicitly stated.
5) Item raw representations: concatenate projected modalities (v/t) ONLY (no item ID embedding),
   as paper states content features are more informative than item ID.
   If only one modality exists, use that single projected feature.

Framework dependency:
- Must run inside MMRec-like framework providing GeneralRecommender, dataset.inter_matrix, self.v_feat/self.t_feat.
"""

import os
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class PGL(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---- required config ----
        self.embedding_dim = config['embedding_size']          # also used as d in SVD
        self.feat_embed_dim = config['feat_embed_dim']         # per-modality projected dim
        self.n_ui_layers = config['n_ui_layers']               # principal graph MP layers (user-item)
        self.n_mm_layers = config['n_mm_layers']               # latent item-item MP layers
        self.knn_k = config['knn_k']                           # kNN for item-item, paper uses k=10
        self.mm_image_weight = config['mm_image_weight']       # modality fusion weight (if both exist)
        self.principal_extractor = config['principal_extractor']  # 'global' or 'local'

        # principal graph hyperparams
        # global-aware
        self.gamma = config['gamma']       # truncation ratio in (0,1)
        self.epsilon = config['epsilon']   # sparsification threshold, typically 1e-3
        # local-aware
        self.p = config['p']               # percentage of edges kept, typically 0.3

        # SSL hyperparams
        self.lambda_ssl = config['lambda_ssl']
        self.ssl_rho = config['ssl_rho']   # feature masking ratio
        self.ssl_tau = config['ssl_tau']   # temperature

        self.reg_weight = config['reg_weight']  # L2 reg on embeddings (not in paper explicitly; keep minimal)

        # ---- numeric constants ----
        # NOTE: define before any helper that uses it (e.g., _get_edge_info/_normalize_bipartite_edges).
        self._eps = 1e-7

        # ---- data / graphs ----
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # complete normalized adjacency (user-item bipartite) for inference
        norm_adj = self._get_norm_adj_mat(self.interaction_matrix)
        self.register_buffer('norm_adj', norm_adj)

        # edge info for local-aware sampling
        edge_indices, edge_probs = self._get_edge_info(self.interaction_matrix)
        self.register_buffer('edge_indices', edge_indices)  # [2, E] with (u, i) (item in [0,n_items))
        self.register_buffer('edge_probs', edge_probs)      # [E] sampling probability

        # principal subgraph cache for training (updated each epoch)
        self._principal_adj = None  # torch.sparse.FloatTensor on device

        # ---- embeddings / feature MLP ----
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # item content features -> item raw representation
        # paper: "map content features using MLP to low dim and concatenate to obtain item raw representations"
        self.has_v = (self.v_feat is not None)
        self.has_t = (self.t_feat is not None)
        assert self.has_v or self.has_t, "PGL requires at least one modality feature (v_feat or t_feat)."

        if self.has_v:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_mlp = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            )
        if self.has_t:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_mlp = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            )

        # item raw dim after concat
        self.item_raw_dim = (int(self.has_v) + int(self.has_t)) * self.feat_embed_dim
        # project concat raw -> embedding_dim for graph CF space (necessary for user-item message passing)
        self.item_raw_to_embed = nn.Linear(self.item_raw_dim, self.embedding_dim, bias=True)

        # ---- latent item-item graph (fixed) ----
        # Build and cache a sparse normalized Laplacian similarity matrix S (items x items).
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(
            dataset_path,
            f"pgl_mm_adj_knn{self.knn_k}_w{int(100*self.mm_image_weight)}.pt"
        )
        if os.path.exists(mm_adj_file):
            mm_adj = torch.load(mm_adj_file, map_location='cpu')
        else:
            with torch.no_grad():
                mm_adj = self._build_mm_item_graph()
            torch.save(mm_adj, mm_adj_file)
        self.register_buffer('mm_adj', mm_adj.coalesce())

        # ---- numeric constants ----
        # NOTE: must be defined before any helper that uses it (e.g., _get_edge_info in __init__).
        self._eps = 1e-7

        # ---- losses ----
        # (no external loss classes; keep faithful and minimal)

    # -------------------------------------------------------------------------
    # graph builders
    # -------------------------------------------------------------------------
    def _get_norm_adj_mat(self, inter_coo: sp.coo_matrix) -> torch.Tensor:
        """Symmetric normalized adjacency for bipartite graph:
           A = [[0, R],[R^T,0]],  L = D^{-1/2} A D^{-1/2}
        """
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        R = inter_coo
        Rt = inter_coo.transpose()
        data_dict = dict(zip(zip(R.row, R.col + self.n_users), [1.0] * R.nnz))
        data_dict.update(dict(zip(zip(Rt.row + self.n_users, Rt.col), [1.0] * Rt.nnz)))
        # stable public API: update via dict items
        for (r, c), v in data_dict.items():
            A[r, c] = v
        A = A.tocsr()

        rowsum = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(rowsum, -0.5).astype(np.float32)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = D_inv_sqrt @ A @ D_inv_sqrt
        L = L.tocoo()

        indices = torch.from_numpy(np.vstack([L.row, L.col]).astype(np.int64))
        values = torch.from_numpy(L.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(L.shape)).coalesce()

    def _normalize_bipartite_edges(self, edges_ui: torch.Tensor, n_users: int, n_items: int) -> torch.Tensor:
        """Given bipartite edges (u, i) with i in [0,n_items),
        compute symmetric normalization values: 1/sqrt(deg_u*deg_i).
        """
        device = edges_ui.device
        u = edges_ui[0]
        i = edges_ui[1]

        deg_u = torch.bincount(u, minlength=n_users).float().to(device) + self._eps
        deg_i = torch.bincount(i, minlength=n_items).float().to(device) + self._eps
        vals = torch.pow(deg_u[u], -0.5) * torch.pow(deg_i[i], -0.5)
        return vals

    def _get_edge_info(self, inter_coo: sp.coo_matrix):
        """Return edge indices [2, E] (u, i) and multinomial sampling probs for local-aware extraction:
           prob(u,i) ∝ 1/sqrt(|N_u|*|N_i|)
        """
        u = torch.from_numpy(inter_coo.row.astype(np.int64))
        i = torch.from_numpy(inter_coo.col.astype(np.int64))
        edges = torch.stack([u, i], dim=0)  # [2, E]

        # prob = 1/sqrt(deg_u*deg_i)
        with torch.no_grad():
            vals = self._normalize_bipartite_edges(edges, self.n_users, self.n_items)
            probs = vals / (vals.sum() + self._eps)
        return edges.long(), probs.float()

    def _compute_normalized_laplacian_from_indices(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Build symmetric normalized adjacency (D^-1/2 A D^-1/2) for item-item graph from indices [2, nnz]."""
        device = indices.device
        values_ones = torch.ones(indices.size(1), device=device, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values_ones, torch.Size([n, n])).coalesce()
        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + self._eps
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        vals = d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, vals, torch.Size([n, n])).coalesce()

    def _build_mm_item_graph(self) -> torch.Tensor:
        """Pre-construct item-item latent graph S by kNN on raw modality features.
        Paper: build per-modality graphs, Laplacian normalization, weighted add.
        """
        device = torch.device('cpu')  # build on CPU, then buffer will move with model.to(device)

        def knn_graph_from_feats(feats: torch.Tensor, topk: int) -> torch.Tensor:
            # cosine similarity kNN
            feats = feats.to(device).float()
            feats = feats / (torch.norm(feats, p=2, dim=1, keepdim=True) + self._eps)
            sim = feats @ feats.t()  # [n_items, n_items], dense
            _, knn_ind = torch.topk(sim, k=topk, dim=-1)
            rows = torch.arange(knn_ind.size(0), device=device).unsqueeze(1).expand_as(knn_ind)
            indices = torch.stack([rows.reshape(-1), knn_ind.reshape(-1)], dim=0).long()
            # NOTE: keep directed edges as in many kNN graph impls; normalization will handle.
            return self._compute_normalized_laplacian_from_indices(indices, feats.size(0))

        with torch.no_grad():
            mats = []
            weights = []
            if self.has_v:
                v_raw = self.image_embedding.weight.detach().cpu()
                mats.append(knn_graph_from_feats(v_raw, self.knn_k))
                weights.append(self.mm_image_weight)
            if self.has_t:
                t_raw = self.text_embedding.weight.detach().cpu()
                w = (1.0 - self.mm_image_weight) if self.has_v else 1.0
                mats.append(knn_graph_from_feats(t_raw, self.knn_k))
                weights.append(w)

            # weighted sum of sparse matrices
            # (coalesce after sum)
            assert len(mats) >= 1
            out = None
            for mat, w in zip(mats, weights):
                mat = mat.coalesce()
                if out is None:
                    out = torch.sparse_coo_tensor(mat.indices(), mat.values() * float(w), mat.size())
                else:
                    out = out + torch.sparse_coo_tensor(mat.indices(), mat.values() * float(w), mat.size())
            return out.coalesce()

    # -------------------------------------------------------------------------
    # principal subgraph extraction
    # -------------------------------------------------------------------------
    def pre_epoch_processing(self):
        """Called by framework each epoch. Build principal subgraph for training."""
        if self.principal_extractor == 'local':
            self._principal_adj = self._extract_local_principal_adj()
        elif self.principal_extractor == 'global':
            self._principal_adj = self._extract_global_principal_adj()
        else:
            raise ValueError("config['principal_extractor'] must be 'local' or 'global'")

    def _extract_local_principal_adj(self) -> torch.Tensor:
        """Local-aware extraction: multinomial sampling of edges with prob ∝ 1/sqrt(deg_u*deg_i),
        keep p percentage of edges. Then build symmetric normalized adjacency (bipartite) on kept edges.
        """
        E = self.edge_probs.numel()
        keep_len = max(1, int(E * float(self.p)))

        # multinomial sampling without replacement
        idx = torch.multinomial(self.edge_probs, keep_len, replacement=False)
        keep_edges = self.edge_indices[:, idx].to(self.device)  # [2, keep_len] (u, i)

        # compute symmetric norm values for kept edges
        vals = self._normalize_bipartite_edges(keep_edges, self.n_users, self.n_items).to(self.device)

        # build bipartite adjacency in (n_users+n_items, n_users+n_items)
        u = keep_edges[0]
        i = keep_edges[1] + self.n_users
        idx_ui = torch.stack([u, i], dim=0)
        idx_iu = torch.stack([i, u], dim=0)

        indices = torch.cat([idx_ui, idx_iu], dim=1)
        values = torch.cat([vals, vals], dim=0)
        adj = torch.sparse_coo_tensor(indices, values, torch.Size([self.n_nodes, self.n_nodes])).coalesce()
        return adj

    def _extract_global_principal_adj(self) -> torch.Tensor:
        """Global-aware extraction via approximate SVD on dense normalized adjacency.

        WARNING (assumption): This densifies the (n_users+n_items)^2 matrix and is infeasible for large datasets.
        Use local extractor for large graphs.

        Steps:
        1) ApproxSVD(\tilde{A}, d) -> U, S, V (here use torch.pca_lowrank on dense \tilde{A})
        2) Truncation with gamma: keep first gamma*d and last (1-gamma)*d singular values
        3) Reconstruct A_hat = U S_trunc V^T
        4) Sparsification: set entries < epsilon to 0; keep sparse; then symmetric normalize.
        """
        A = self.norm_adj.to_dense()  # assumption; expensive
        d = int(self.embedding_dim)

        # torch.pca_lowrank returns U, S, V such that A ≈ U diag(S) V^T
        # Use center=False to approximate SVD-like decomposition (assumption).
        U, S, V = torch.pca_lowrank(A, q=d, center=False)

        k1 = int(max(0, min(d, round(self.gamma * d))))
        k2_start = int(max(0, min(d, round((1.0 - self.gamma) * d))))

        S_trunc = torch.zeros_like(S)
        if k1 > 0:
            S_trunc[:k1] = S[:k1]
        if k2_start < d:
            S_trunc[k2_start:] = S[k2_start:]

        A_hat = (U * S_trunc.unsqueeze(0)) @ V.t()

        # sparsify
        A_hat = torch.where(A_hat.abs() >= float(self.epsilon), A_hat, torch.zeros_like(A_hat))

        # build sparse from non-zeros
        nz = A_hat.nonzero(as_tuple=False).t().contiguous()
        if nz.numel() == 0:
            # fallback: if fully sparsified, use original norm_adj (minimal safe fallback)
            # (assumption: avoid empty graph breaking training)
            return self.norm_adj

        values = A_hat[nz[0], nz[1]].float()

        # symmetric normalize reconstructed adjacency (assumption for stability)
        adj = torch.sparse_coo_tensor(nz, values, torch.Size([self.n_nodes, self.n_nodes])).coalesce()
        deg = torch.sparse.sum(adj.abs(), dim=1).to_dense() + self._eps
        d_inv_sqrt = torch.pow(deg, -0.5)
        norm_values = d_inv_sqrt[adj.indices()[0]] * adj.values() * d_inv_sqrt[adj.indices()[1]]
        adj = torch.sparse_coo_tensor(adj.indices(), norm_values, adj.size()).coalesce()
        return adj.to(self.device)

    # -------------------------------------------------------------------------
    # embeddings and message passing
    # -------------------------------------------------------------------------
    def _item_raw_embeddings(self) -> torch.Tensor:
        """Item raw representations from modality features then concat; returns [n_items, item_raw_dim]."""
        feats = []
        if self.has_v:
            v = self.image_mlp(self.image_embedding.weight)
            feats.append(v)
        if self.has_t:
            t = self.text_mlp(self.text_embedding.weight)
            feats.append(t)
        x = torch.cat(feats, dim=1)
        return x

    def _principal_message_passing(self, adj: torch.Tensor, user_e0: torch.Tensor, item_e0: torch.Tensor):
        """LightGCN-style propagation on bipartite graph, mean pooling across layers."""
        ego = torch.cat([user_e0, item_e0], dim=0)  # [n_nodes, dim]
        outs = [ego]
        x = ego
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            outs.append(x)
        out = torch.stack(outs, dim=1).mean(dim=1)
        u, i = torch.split(out, [self.n_users, self.n_items], dim=0)
        return u, i

    def _latent_item_message_passing(self, item_e0: torch.Tensor) -> torch.Tensor:
        """Message passing on fixed item-item similarity graph S (sparse)."""
        x = item_e0
        outs = [x]
        for _ in range(self.n_mm_layers):
            x = torch.sparse.mm(self.mm_adj, x)
            outs.append(x)
        return torch.stack(outs, dim=1).mean(dim=1)

    def forward(self, adj=None, train_mode: bool = False):
        """
        If train_mode=True: use principal subgraph (extracted) for principal message passing.
        If train_mode=False: use complete graph (self.norm_adj) for principal message passing.
        """
        if adj is None:
            adj = self.norm_adj

        # raw user/item base embeddings
        user_e0 = self.user_embedding.weight
        item_raw = self._item_raw_embeddings()
        item_e0 = self.item_raw_to_embed(item_raw)

        # principal graph learning
        if train_mode:
            if self._principal_adj is None:
                # if framework didn't call pre_epoch_processing, fall back to local extraction with p (minimal)
                # (assumption)
                self._principal_adj = self._extract_local_principal_adj()
            u_pri, i_pri = self._principal_message_passing(self._principal_adj, user_e0, item_e0)
        else:
            u_pri, i_pri = self._principal_message_passing(adj, user_e0, item_e0)

        # latent graph learning (item-item)
        i_lat = self._latent_item_message_passing(item_e0)
        # paper: e_u^latent exists; not defined for user from item-item graph.
        # Minimal faithful choice: set user latent to 0 (since latent graph is item-item only).
        # (assumption documented)
        u_lat = torch.zeros_like(u_pri)

        # final embeddings for prediction
        u_hat = u_pri + u_lat
        i_hat = i_pri + i_lat
        return u_hat, i_hat

    # -------------------------------------------------------------------------
    # losses
    # -------------------------------------------------------------------------
    def _bpr_loss(self, u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _l2_reg(self, *tensors) -> torch.Tensor:
        reg = 0.0
        for t in tensors:
            reg = reg + (t.pow(2).sum())
        return reg

    def _feature_mask(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """Feature masking: randomly mask (set to 0) a ratio rho of dimensions per element."""
        if rho <= 0.0:
            return x
        keep_prob = 1.0 - float(rho)
        mask = (torch.rand_like(x) < keep_prob).float()
        return x * mask

    def _inbatch_infonce(self, z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
        """Standard in-batch InfoNCE with positives on diagonal."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = (z1 @ z2.t()) / float(tau)
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, interaction):
        users = interaction[0].long()
        pos_items = interaction[1].long()
        neg_items = interaction[2].long()

        u_hat, i_hat = self.forward(train_mode=True)  # training uses principal subgraph

        u = u_hat[users]
        pos = i_hat[pos_items]
        neg = i_hat[neg_items]

        loss_bpr = self._bpr_loss(u, pos, neg)

        # SSL on involved nodes (paper: sum over all users/items; in practice, use in-batch for efficiency)
        # Minimal assumption: apply SSL on current batch users and batch (pos+neg) items unique.
        batch_users = torch.unique(users)
        batch_items = torch.unique(torch.cat([pos_items, neg_items], dim=0))

        u_view1 = self._feature_mask(u_hat[batch_users], self.ssl_rho)
        u_view2 = self._feature_mask(u_hat[batch_users], self.ssl_rho)
        i_view1 = self._feature_mask(i_hat[batch_items], self.ssl_rho)
        i_view2 = self._feature_mask(i_hat[batch_items], self.ssl_rho)

        loss_ssl = self._inbatch_infonce(u_view1, u_view2, self.ssl_tau) + \
                   self._inbatch_infonce(i_view1, i_view2, self.ssl_tau)

        # reg (minimal; not described in paper; keep controlled by reg_weight)
        reg = self._l2_reg(
            self.user_embedding.weight[batch_users],
            u, pos, neg
        ) / (users.size(0) + self._eps)

        return loss_bpr + float(self.lambda_ssl) * loss_ssl + float(self.reg_weight) * reg

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = interaction[0].long()
        u_hat, i_hat = self.forward(adj=self.norm_adj, train_mode=False)  # inference uses complete graph
        u = u_hat[user]  # [B, dim]
        scores = u @ i_hat.t()
        return scores