# -*- coding: utf-8 -*-
"""
PGL_2025: Principal Graph Learning for Multimedia Recommendation (core implementation)

This implementation is written in a "RecBole-like" style consistent with the provided baselines:
- __init__(config, dataset) loads interaction matrix and multimodal features
- pre_epoch_processing() builds the training-time principal subgraph (global-aware or local-aware)
- forward(adj_ui, use_principal: bool) performs:
    (1) Principal graph learning on user-item graph (training: principal subgraph; inference: full graph)
    (2) Latent graph learning on fixed multimodal item-item graph (pre-constructed KNN Laplacian)
    (3) Fuse: e_hat = e_principal + e_latent
- calculate_loss(interaction) = BPR + lambda_ssl * SSL(InfoNCE with feature masking)
- full_sort_predict(interaction) uses full graph (complete) for inference

Notes / engineering choices (due to paper ambiguity):
1) Item raw representations: project each modality feature to `feat_embed_dim`, then concatenate and
   linearly project to `embedding_dim` to match user embedding dim.
2) Principal graph learning message passing H(·): LightGCN-style propagation with layer-wise mean pooling.
3) Global-aware extraction via (randomized) SVD is implemented using torch.pca_lowrank as a practical
   randomized SVD surrogate on the (dense) normalized adjacency. This is only suitable for small graphs.
   For large graphs, please implement a true sparse randomized SVD or compute it offline.
4) Truncation keeps top gamma*d and bottom (1-gamma)*d singular values (as described).
5) Sparsification threshold eps is applied to reconstructed adjacency before converting to sparse.
6) Local-aware extraction uses degree-sensitive multinomial sampling on bipartite edges.

Required config keys (with typical defaults):
- embedding_size (int)
- feat_embed_dim (int)
- n_ui_layers (int)          # layers for user-item message passing
- n_mm_layers (int)          # layers for item-item message passing
- knn_k (int)                # kNN for item-item graph
- mm_image_weight (float)    # if both image/text, fuse weight for image
- principal_type (str)       # 'local' or 'global'
- local_p (float)            # edge keep ratio for local-aware extraction, e.g., 0.3
- global_gamma (float)       # truncation ratio gamma in (0,1), e.g., 0.5
- global_eps (float)         # sparsification eps, e.g., 1e-3
- ssl_weight (float)         # lambda_ssl
- ssl_mask_ratio (float)     # rho
- ssl_tau (float)            # temperature tau
- reg_weight (float)         # optional L2 reg on embeddings (small), default 0.

Dataset is expected to provide:
- dataset.inter_matrix(form='coo') -> scipy.sparse.coo_matrix
- self.v_feat / self.t_feat are handled by GeneralRecommender parent in baselines; here we use dataset features
  via self.v_feat and self.t_feat as in LATTICE/FREEDOM.
"""

import os
import math
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

# If you have the same project structure as baselines:
from common.abstract_recommender import GeneralRecommender


def _coo_to_torch_sparse(coo: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    coo = coo.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def _build_sym_norm_adj_from_bipartite_coo(inter_coo: sp.coo_matrix, n_users: int, n_items: int) -> sp.coo_matrix:
    """
    Build symmetric normalized adjacency for bipartite graph:
        A = [[0, R],
             [R^T, 0]]
        L = D^{-1/2} A D^{-1/2}
    No self-loop (consistent with LightGCN style in many recsys papers).
    """
    n_nodes = n_users + n_items
    R = inter_coo.tocoo()
    rows_u = R.row
    cols_i = R.col + n_users
    data = np.ones(R.nnz, dtype=np.float32)

    # bi-directional edges
    row = np.concatenate([rows_u, cols_i])
    col = np.concatenate([cols_i, rows_u])
    val = np.concatenate([data, data])

    A = sp.coo_matrix((val, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    deg = np.array(A.sum(axis=1)).flatten() + 1e-7
    deg_inv_sqrt = np.power(deg, -0.5).astype(np.float32)
    D_inv = sp.diags(deg_inv_sqrt)
    L = D_inv.dot(A).dot(D_inv).tocoo()
    return L


def _normalize_bipartite_edges(u_idx: torch.Tensor, i_idx: torch.Tensor, n_users: int, n_items: int) -> torch.Tensor:
    """
    Compute symmetric normalization values for bipartite edges (u,i) in R:
        val(u,i) = 1/sqrt(deg_u * deg_i)
    where deg_i is degree on item side in the bipartite graph.
    """
    device = u_idx.device
    ones = torch.ones_like(u_idx, dtype=torch.float32, device=device)
    deg_u = torch.zeros(n_users, dtype=torch.float32, device=device).scatter_add_(0, u_idx, ones)
    deg_i = torch.zeros(n_items, dtype=torch.float32, device=device).scatter_add_(0, i_idx, ones)

    val = 1.0 / torch.sqrt((deg_u[u_idx] + 1e-7) * (deg_i[i_idx] + 1e-7))
    return val


def _feature_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Feature masking for SSL: randomly zero-out a ratio of feature dimensions for each node.
    """
    if mask_ratio <= 0.0:
        return x
    keep_prob = 1.0 - mask_ratio
    mask = (torch.rand_like(x) < keep_prob).float()
    # scale to keep expectation
    return x * mask / (keep_prob + 1e-7)


def _infonce_inbatch(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Standard in-batch InfoNCE:
      L = - mean_i log exp(sim(i,i)/tau) / sum_j exp(sim(i,j)/tau)
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / max(tau, 1e-6)  # [B,B]
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class PGL(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_dim = int(config["embedding_size"])
        # RecBole's Config behaves like a dict for __getitem__ but doesn't have .get()
        def cfg(key, default=None):
            try:
                return config[key]
            except Exception:
                return default

        self.feat_embed_dim = int(cfg("feat_embed_dim", self.embedding_dim))
        self.n_ui_layers = int(cfg("n_ui_layers", 2))
        self.n_mm_layers = int(cfg("n_mm_layers", 1))

        self.knn_k = int(cfg("knn_k", 10))
        self.mm_image_weight = float(cfg("mm_image_weight", 0.5))

        self.principal_type = str(cfg("principal_type", "local")).lower()
        assert self.principal_type in ["local", "global"]

        self.local_p = float(cfg("local_p", 0.3))

        self.global_gamma = float(cfg("global_gamma", 0.5))
        self.global_eps = float(cfg("global_eps", 1e-3))

        self.ssl_weight = float(cfg("ssl_weight", 0.1))
        self.ssl_mask_ratio = float(cfg("ssl_mask_ratio", 0.1))
        self.ssl_tau = float(cfg("ssl_tau", 0.2))

        self.reg_weight = float(cfg("reg_weight", 0.0))

        # --------- interaction graph ----------
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # full graph (inference)
        full_norm_adj_sp = _build_sym_norm_adj_from_bipartite_coo(self.interaction_matrix, self.n_users, self.n_items)
        self.register_buffer("full_norm_adj", _coo_to_torch_sparse(full_norm_adj_sp, self.device))

        # store bipartite edge list (u,i) for local-aware sampling
        self._ui_u = torch.from_numpy(self.interaction_matrix.row.astype(np.int64)).to(self.device)
        self._ui_i = torch.from_numpy(self.interaction_matrix.col.astype(np.int64)).to(self.device)

        # principal (training) graph buffer, created each epoch
        self.register_buffer("principal_norm_adj", self.full_norm_adj)  # initialized as full graph

        # --------- embeddings ----------
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # Item raw representation: content features -> embedding_dim
        # Following paper: prefer content features over item ID for convergence.
        # We thus do NOT create a learnable item ID embedding by default.
        self.use_item_id_fallback = (self.v_feat is None) and (self.t_feat is None)
        if self.use_item_id_fallback:
            self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
            nn.init.xavier_uniform_(self.item_id_embedding.weight)
        else:
            self.item_id_embedding = None

        # modality feature adapters
        self.has_v = self.v_feat is not None
        self.has_t = self.t_feat is not None

        if self.has_v:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            )
        if self.has_t:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            )

        # concatenate modalities -> embedding_dim
        in_dim = 0
        if self.has_v:
            in_dim += self.feat_embed_dim
        if self.has_t:
            in_dim += self.feat_embed_dim
        self.item_raw_proj = None
        if not self.use_item_id_fallback:
            self.item_raw_proj = nn.Linear(in_dim, self.embedding_dim)

        # --------- latent item-item graph (fixed) ----------
        # Pre-construct S from raw modality features (paper: avoid classical LATTICE dynamic latent graph learning)
        self.register_buffer("mm_adj", None)
        self._build_fixed_mm_graph(config, dataset)

    @torch.no_grad()
    def _build_fixed_mm_graph(self, config, dataset):
        """
        Build fixed item-item similarity graph S (sparse normalized Laplacian) with kNN.
        Cache to disk for reproducibility.
        """
        dataset_path = os.path.abspath(os.path.join(config["data_path"], config["dataset"]))
        os.makedirs(dataset_path, exist_ok=True)
        flag = f"k{self.knn_k}_w{int(self.mm_image_weight * 100):03d}_d{self.embedding_dim}_fd{self.feat_embed_dim}"
        mm_adj_file = os.path.join(dataset_path, f"pgl_mm_adj_{flag}.pt")

        if os.path.exists(mm_adj_file):
            mm_adj = torch.load(mm_adj_file, map_location=self.device)
            self.mm_adj = mm_adj.coalesce()
            return

        if self.use_item_id_fallback:
            # no content features: use identity-like sparse graph (self-loop) to make latent branch no-op
            idx = torch.arange(self.n_items, device=self.device)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(self.n_items, device=self.device)
            self.mm_adj = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items), device=self.device).coalesce()
            torch.save(self.mm_adj.cpu(), mm_adj_file)
            return

        # compute item raw modality embeddings (for similarity)
        feats = []
        if self.has_v:
            v = self.image_trs(self.image_embedding.weight.to(self.device))
            feats.append(F.normalize(v, dim=-1))
        if self.has_t:
            t = self.text_trs(self.text_embedding.weight.to(self.device))
            feats.append(F.normalize(t, dim=-1))

        # build knn graph per modality, then weighted add (paper: weighted added; we use mm_image_weight)
        mm_adj = None
        if self.has_v and not self.has_t:
            mm_adj = self._knn_laplacian_from_dense(feats[0], self.knn_k)
        elif self.has_t and not self.has_v:
            mm_adj = self._knn_laplacian_from_dense(feats[0], self.knn_k)
        else:
            img_adj = self._knn_laplacian_from_dense(feats[0], self.knn_k)
            txt_adj = self._knn_laplacian_from_dense(feats[1], self.knn_k)
            mm_adj = (self.mm_image_weight * img_adj + (1.0 - self.mm_image_weight) * txt_adj).coalesce()

        self.mm_adj = mm_adj.coalesce()
        torch.save(self.mm_adj.cpu(), mm_adj_file)

    @torch.no_grad()
    def _knn_laplacian_from_dense(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build sparse normalized Laplacian from dense features:
            sim = cos(x_i, x_j); take top-k (excluding self is not enforced; harmless).
            adjacency A is unweighted (1), then L = D^{-1/2} A D^{-1/2}.
        """
        x = F.normalize(x, dim=-1)
        sim = x @ x.t()  # [n,n]
        _, knn = torch.topk(sim, k=k, dim=-1)
        n = x.size(0)
        row = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = knn.reshape(-1)
        indices = torch.stack([row, col], dim=0)

        # unweighted adjacency then normalize
        adj = torch.sparse_coo_tensor(indices, torch.ones(indices.size(1), device=x.device), (n, n), device=x.device).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        deg_inv_sqrt = torch.pow(deg, -0.5)
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(indices, values, (n, n), device=x.device).coalesce()

    def pre_epoch_processing(self):
        """
        Called each epoch to build training-time principal subgraph from complete graph.
        - local-aware: multinomial sampling on edges with degree-sensitive probabilities.
        - global-aware: SVD truncation reconstruction + sparsification (small graphs only).
        """
        if self.principal_type == "local":
            self.principal_norm_adj = self._local_aware_principal_adj(self.local_p)
        else:
            self.principal_norm_adj = self._global_aware_principal_adj(self.global_gamma, self.global_eps)

    @torch.no_grad()
    def _local_aware_principal_adj(self, p: float) -> torch.Tensor:
        """
        Multinomial sampling edges (u,i) with prob proportional to:
            w(u,i) = 1/sqrt(deg_u)*1/sqrt(deg_i)
        Keep ratio p of edges.
        Then build symmetric normalized adjacency on the induced bipartite graph.
        """
        assert 0.0 < p <= 1.0
        u = self._ui_u
        i = self._ui_i
        n_edges = u.numel()
        keep = max(1, int(n_edges * p))

        # sampling weights
        # compute degrees on original graph (could also use current, but paper implies original)
        ones = torch.ones(n_edges, dtype=torch.float32, device=self.device)
        deg_u = torch.zeros(self.n_users, dtype=torch.float32, device=self.device).scatter_add_(0, u, ones)
        deg_i = torch.zeros(self.n_items, dtype=torch.float32, device=self.device).scatter_add_(0, i, ones)
        w = 1.0 / (torch.sqrt(deg_u[u] + 1e-7) * torch.sqrt(deg_i[i] + 1e-7))

        idx = torch.multinomial(w, num_samples=keep, replacement=False)
        u_k = u[idx]
        i_k = i[idx]

        # build symmetric normalized adjacency from sampled bipartite edges
        # normalize values for bipartite edges
        val_ui = _normalize_bipartite_edges(u_k, i_k, self.n_users, self.n_items)  # [keep]
        # expand to full (n_users+n_items) symmetric
        row1 = u_k
        col1 = i_k + self.n_users
        row2 = col1
        col2 = row1
        indices = torch.stack([torch.cat([row1, row2]), torch.cat([col1, col2])], dim=0)
        values = torch.cat([val_ui, val_ui], dim=0)

        adj = torch.sparse_coo_tensor(indices, values, (self.n_nodes, self.n_nodes), device=self.device).coalesce()
        return adj

    @torch.no_grad()
    def _global_aware_principal_adj(self, gamma: float, eps: float) -> torch.Tensor:
        """
        Global-aware extraction:
        1) ApproxSVD(full_norm_adj, d)  (d = embedding_dim)
        2) Truncation: keep top gamma*d and bottom (1-gamma)*d singular values, set middle to 0
        3) Reconstruct A_hat = U * S_trunc * V^T
        4) Sparsify: set |A_hat| < eps to 0
        5) Convert to sparse and sym-normalize again (optional). Here we directly use A_hat and then sym-normalize.

        WARNING: This densifies the matrix; only feasible for small graphs.
        """
        d = self.embedding_dim
        # dense adjacency
        A = self.full_norm_adj.to_dense()  # [N,N]
        N = A.size(0)
        q = min(d, N - 1)
        if q <= 0:
            return self.full_norm_adj

        # randomized low-rank decomposition: torch.pca_lowrank is a practical randomized SVD-like routine
        # It returns U, S, V such that A ≈ U diag(S) V^T
        U, S, V = torch.pca_lowrank(A, q=q, center=False)

        # truncation pattern: keep top and bottom
        k_top = int(max(0, min(q, math.floor(gamma * q))))
        k_bottom = q - k_top  # keep bottom (1-gamma)*q
        S_trunc = torch.zeros_like(S)
        if k_top > 0:
            S_trunc[:k_top] = S[:k_top]
        if k_bottom > 0:
            S_trunc[-k_bottom:] = S[-k_bottom:]

        A_hat = (U * S_trunc.unsqueeze(0)) @ V.t()

        # sparsification
        A_hat = torch.where(A_hat.abs() >= eps, A_hat, torch.zeros_like(A_hat))

        # remove diagonal (optional; bipartite original has no self-loop)
        A_hat.fill_diagonal_(0.0)

        # convert to sparse
        A_sp = A_hat.to_sparse_coo().coalesce()

        # sym-normalize again to stabilize message passing
        deg = torch.sparse.sum(A_sp, dim=1).to_dense().abs() + 1e-7
        deg_inv_sqrt = torch.pow(deg, -0.5)
        idx = A_sp.indices()
        val = A_sp.values()
        val = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        A_norm = torch.sparse_coo_tensor(idx, val, A_sp.shape, device=self.device).coalesce()
        return A_norm

    def _build_item_raw(self) -> torch.Tensor:
        """
        Item raw representation (paper):
          - project each modality feature via MLP to low-dimensional repr
          - concatenate, then map to embedding_dim
        """
        if self.use_item_id_fallback:
            return self.item_id_embedding.weight

        feats = []
        if self.has_v:
            v = self.image_trs(self.image_embedding.weight)
            feats.append(v)
        if self.has_t:
            t = self.text_trs(self.text_embedding.weight)
            feats.append(t)
        x = torch.cat(feats, dim=-1)
        x = self.item_raw_proj(x)
        return x

    def _lightgcn_propagate(self, adj: torch.Tensor, ego: torch.Tensor, n_layers: int) -> torch.Tensor:
        """
        LightGCN propagation with layer-wise mean pooling.
        """
        all_emb = [ego]
        x = ego
        for _ in range(n_layers):
            x = torch.sparse.mm(adj, x)
            all_emb.append(x)
        out = torch.stack(all_emb, dim=1).mean(dim=1)
        return out

    def forward(self, use_principal: bool = True):
        """
        Returns:
            u_hat: [n_users, dim]
            i_hat: [n_items, dim]
        """
        # principal graph learning on user-item graph
        adj_ui = self.principal_norm_adj if use_principal else self.full_norm_adj

        user_raw = self.user_embedding.weight
        item_raw = self._build_item_raw()

        ego = torch.cat([user_raw, item_raw], dim=0)  # [n_nodes, dim]
        principal_out = self._lightgcn_propagate(adj_ui, ego, self.n_ui_layers)
        u_principal, i_principal = torch.split(principal_out, [self.n_users, self.n_items], dim=0)

        # latent graph learning on item-item graph (fixed)
        # message passing: h <- S h (n_mm_layers)
        h = item_raw
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        i_latent = h

        # user latent part: paper defines e_u^latent as output of latent graph learning too.
        # But latent graph is item-item; a common practice is to derive user latent by aggregating latent items
        # through the full user-item graph (or same adj used). Here we use one-step aggregation from principal adj.
        # This matches "collaborative signals under modality information".
        # u_latent = sum_{i in N(u)} norm_adj(u,i) * i_latent
        # We can do: u_latent = (adj_ui[:U, U:]) @ i_latent, but we only have full adj over nodes.
        # So compute via sparse mm on full node embedding where item part is replaced by i_latent.
        zero_users = torch.zeros_like(user_raw)
        ego_lat = torch.cat([zero_users, i_latent], dim=0)
        lat_node = torch.sparse.mm(adj_ui, ego_lat)
        u_latent, _ = torch.split(lat_node, [self.n_users, self.n_items], dim=0)

        u_hat = u_principal + u_latent
        i_hat = i_principal + i_latent
        return u_hat, i_hat

    def _bpr_loss(self, u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos_scores = (u * pos).sum(dim=-1)
        neg_scores = (u * neg).sum(dim=-1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _l2_reg(self, *tensors) -> torch.Tensor:
        if self.reg_weight <= 0:
            return torch.zeros((), device=self.device)
        reg = torch.zeros((), device=self.device)
        for x in tensors:
            reg = reg + x.pow(2).sum() / x.size(0)
        return self.reg_weight * reg

    def calculate_loss(self, interaction):
        """
        interaction: [users, pos_items, neg_items]
        """
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]

        # training uses principal graph
        u_all, i_all = self.forward(use_principal=True)

        u = u_all[users]
        pos = i_all[pos_items]
        neg = i_all[neg_items]
        loss_bpr = self._bpr_loss(u, pos, neg)

        # SSL: feature masking + in-batch InfoNCE for users and items (sampled batch nodes)
        if self.ssl_weight > 0:
            u_mask1 = _feature_mask(u, self.ssl_mask_ratio)
            u_mask2 = _feature_mask(u, self.ssl_mask_ratio)
            i_pos = pos  # use positive items as item nodes in SSL batch
            i_mask1 = _feature_mask(i_pos, self.ssl_mask_ratio)
            i_mask2 = _feature_mask(i_pos, self.ssl_mask_ratio)

            loss_ssl = _infonce_inbatch(u_mask1, u_mask2, self.ssl_tau) + _infonce_inbatch(i_mask1, i_mask2, self.ssl_tau)
            loss = loss_bpr + self.ssl_weight * loss_ssl
        else:
            loss_ssl = torch.zeros((), device=self.device)
            loss = loss_bpr

        # optional reg (small)
        loss = loss + self._l2_reg(u, pos, neg)

        return loss

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        inference uses the complete user-item graph (paper).
        interaction: [user] (single user id tensor)
        """
        user = interaction[0]
        u_all, i_all = self.forward(use_principal=False)
        u = u_all[user]  # [B, dim]
        scores = u @ i_all.t()
        return scores