# coding: utf-8
"""
PGL
################################################
Paper: PGL_2025 (Principal Graph Learning for Multimedia Recommendation)

Core ideas implemented:
- User raw reps: user ID embeddings.
- Item raw reps: project each available modality feature by MLP to low-dim, then concatenate (no item ID embedding used).
- Principal Graph Learning on user-item bipartite graph:
  - Training: message passing on extracted principal subgraph (global-aware via randomized SVD + truncation + sparsification
             OR local-aware via degree-aware multinomial edge sampling).
  - Inference: message passing on the complete graph (to avoid fragmentation).
- Latent Graph Learning on item-item similarity graph:
  - Pre-construct kNN item-item graphs for each modality from raw modality features, Laplacian normalize, then weighted-add.
  - Message passing on fixed item-item graph.
- Prediction: inner product on fused reps: e_hat = e_principal + e_latent (both for user and item).
- Optimization: BPR + lambda_ssl * InfoNCE( feature masking, in-batch ).

Notes / minimal assumptions due to paper ambiguity:
- Message passing H(.) uses LightGCN-style propagation with layer-wise mean pooling.
- For global-aware extraction, we approximate on dense adjacency (n_nodes x n_nodes). This can be expensive; intended for
  smaller benchmarks. Kept faithful to paper; if too large, use local-aware extraction.
- Truncation: keep top gamma*d and bottom (1-gamma)*d eigen-components (as described), zero-out the middle part.
  We assume singular values are sorted descending (torch.svd_lowrank does that).
- Sparsification threshold epsilon is applied on reconstructed dense adjacency then converted to sparse COO.
- Local-aware extraction samples undirected UI edges; builds symmetric normalized adjacency for propagation.

Framework contract:
- class name == model name, inherits GeneralRecommender.
- implements __init__/forward/calculate_loss/full_sort_predict.
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_neighbourhood, compute_normalized_laplacian


class PGL(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ----- required configs (no config.get) -----
        self.embedding_dim = config['embedding_size']          # also equals SVD rank d per paper
        self.feat_embed_dim = config['feat_embed_dim']         # per-modality projected dim BEFORE concat
        self.n_ui_layers = config['n_ui_layers']               # message passing layers for user-item graph
        self.n_mm_layers = config['n_mm_layers']               # message passing layers for item-item graph
        self.knn_k = config['knn_k']                           # kNN sparsification for item-item graphs (paper: k=10)
        self.ssl_weight = config['lambda_ssl']
        self.ssl_temp = config['ssl_tau']
        self.ssl_mask_ratio = config['ssl_rho']

        self.principal_mode = config['principal_mode']         # 'global' or 'local'
        # local-aware
        self.local_p = config['local_p']                       # percentage of original edges kept (paper typical 0.3)
        # global-aware
        self.global_gamma = config['global_gamma']             # truncation ratio gamma in (0,1)
        self.global_epsilon = config['global_epsilon']         # sparsification threshold epsilon (paper typical 1e-3)

        # modality weights for latent item-item graph fusion
        # paper says "weighted added", but does not specify learnable or fixed
        # minimal assumption: use fixed weights provided by config (to avoid extra learnable parameters).
        self.mm_image_weight = config['mm_image_weight']       # used if both v and t exist; else ignored

        # ----- dataset -----
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # ----- raw embeddings -----
        # user raw reps = user ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # item raw reps = concat(projected modality features); no item ID embedding per paper rationale
        self.has_v = self.v_feat is not None
        self.has_t = self.t_feat is not None
        if (not self.has_v) and (not self.has_t):
            raise ValueError("PGL requires at least one modality feature (v_feat or t_feat).")

        # item projection MLPs: minimal 1-layer MLP (Linear) to match common baselines; paper says MLP.
        # If you want deeper MLP, extend here, but that would be extra assumption.
        if self.has_v:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_mlp = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.has_t:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_mlp = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # final item raw dim after concat
        self.item_raw_dim = (self.feat_embed_dim if self.has_v else 0) + (self.feat_embed_dim if self.has_t else 0)
        # to make user/item be in same space for dot-product & graph propagation, map item_raw -> embedding_dim
        # minimal assumption: linear projection after concat.
        self.item_fuse = nn.Linear(self.item_raw_dim, self.embedding_dim)

        # ----- build complete UI normalized adjacency (for inference; also for global-aware base) -----
        self.complete_ui_adj = self._build_ui_norm_adj_from_interaction(self.interaction_matrix).to(self.device)
        self.register_buffer('_complete_ui_adj_cached', self.complete_ui_adj)  # keep device consistent

        # edge info for local-aware sampling
        self.edge_u_i, self.edge_prob = self._get_ui_edge_info(self.interaction_matrix)
        self.edge_u_i = self.edge_u_i.to(self.device)
        self.edge_prob = self.edge_prob.to(self.device)

        # ----- latent item-item graph (fixed) -----
        # Pre-construct based on similarity of item raw modality features; kNN + Laplacian normalization.
        self.mm_adj = self._build_fused_mm_adj(config, dataset).to(self.device)
        self.register_buffer('_mm_adj_cached', self.mm_adj)

        # runtime graph used for training principal message passing
        self.masked_ui_adj = None  # set in pre_epoch_processing()

    # --------- graph building utilities ---------
    def _build_ui_norm_adj_from_interaction(self, inter_coo: sp.coo_matrix):
        """Symmetric normalized adjacency for bipartite graph, shape (n_users+n_items, n_users+n_items)."""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = inter_coo
        inter_M_t = inter_M.transpose().tocoo()
        # stable public update
        data = np.ones(inter_M.nnz, dtype=np.float32)
        A_u_i = sp.coo_matrix((data, (inter_M.row, inter_M.col + self.n_users)), shape=(self.n_nodes, self.n_nodes))
        A_i_u = sp.coo_matrix((data, (inter_M_t.row + self.n_users, inter_M_t.col)), shape=(self.n_nodes, self.n_nodes))
        A = (A_u_i + A_i_u).tocsr()

        # symmetric normalization D^{-1/2} A D^{-1/2}
        rowsum = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(rowsum, -0.5)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = D_inv_sqrt @ A @ D_inv_sqrt
        L = sp.coo_matrix(L)

        indices = torch.from_numpy(np.vstack([L.row, L.col]).astype(np.int64))
        values = torch.from_numpy(L.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(L.shape)).coalesce()

    def _normalize_bipartite_edges(self, ui_edges: torch.LongTensor):
        """
        Compute symmetric normalized values for bipartite edges (u,i) in user-item space.
        ui_edges: [2, E] with (user_index, item_index in [0,n_items))
        Return: values [E] = 1/sqrt(deg_u * deg_i)
        """
        device = ui_edges.device
        u = ui_edges[0]
        i = ui_edges[1]

        # degrees from interaction matrix (complete graph degrees)
        # (faithful to paper "probability uses |N_u| and |N_i|" from original graph)
        # compute once via scipy for correctness; then move to torch
        # Note: This function may be called often; we keep degrees cached.
        if not hasattr(self, '_deg_u'):
            inter = self.interaction_matrix.tocsr()
            deg_u = np.asarray(inter.sum(axis=1)).flatten().astype(np.float32) + 1e-7
            deg_i = np.asarray(inter.sum(axis=0)).flatten().astype(np.float32) + 1e-7
            self.register_buffer('_deg_u', torch.from_numpy(deg_u).to(self.device))
            self.register_buffer('_deg_i', torch.from_numpy(deg_i).to(self.device))

        val = torch.pow(self._deg_u[u], -0.5) * torch.pow(self._deg_i[i], -0.5)
        return val.to(device)

    def _get_ui_edge_info(self, inter_coo: sp.coo_matrix):
        """
        Return:
            edge_u_i: [2,E] (u, i) with i in [0,n_items)
            edge_prob: [E] sampling prob proportional to 1/sqrt(|Nu|*|Ni|)
        """
        rows = torch.from_numpy(inter_coo.row.astype(np.int64))
        cols = torch.from_numpy(inter_coo.col.astype(np.int64))
        edges = torch.stack([rows, cols], dim=0)  # [2,E]

        # sampling weights per paper: 1/sqrt(|Nu|)*sqrt(|Ni|)
        inter = inter_coo.tocsr()
        deg_u = np.asarray(inter.sum(axis=1)).flatten().astype(np.float32) + 1e-7
        deg_i = np.asarray(inter.sum(axis=0)).flatten().astype(np.float32) + 1e-7
        prob = (1.0 / np.sqrt(deg_u[inter_coo.row])) * (1.0 / np.sqrt(deg_i[inter_coo.col]))
        prob = torch.from_numpy(prob.astype(np.float32))
        return edges, prob

    def _edges_to_ui_adj(self, ui_edges: torch.LongTensor, n_users: int, n_items: int):
        """
        Build symmetric normalized adjacency for chosen bipartite edges.
        ui_edges: [2,E] (u, i) with i in [0,n_items)
        returns sparse adj [n_nodes, n_nodes]
        """
        device = ui_edges.device
        values = self._normalize_bipartite_edges(ui_edges)  # [E]
        # build (u, i+n_users) and symmetric (i+n_users, u)
        u = ui_edges[0]
        i = ui_edges[1] + n_users
        idx_ui = torch.stack([u, i], dim=0)
        idx_iu = torch.stack([i, u], dim=0)

        indices = torch.cat([idx_ui, idx_iu], dim=1)
        vals = torch.cat([values, values], dim=0)
        adj = torch.sparse_coo_tensor(indices, vals, torch.Size((n_users + n_items, n_users + n_items)),
                                      device=device).coalesce()
        return adj

    def _build_fused_mm_adj(self, config, dataset):
        """
        Build fixed fused item-item adjacency S (sparse) with kNN and Laplacian normalization.
        Uses utils.utils: build_sim, build_knn_neighbourhood, compute_normalized_laplacian (as in LATTICE).
        """
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        cache_name = f"pgl_mm_adj_knn{self.knn_k}_w{int(100*self.mm_image_weight)}.pt"
        cache_path = os.path.join(dataset_path, cache_name)

        if os.path.exists(cache_path):
            mm_adj = torch.load(cache_path, map_location='cpu')
            return mm_adj

        with torch.no_grad():
            mm_adj = None
            if self.has_v:
                v_raw = self.image_embedding.weight.detach().cpu()
                v_sim = build_sim(v_raw)
                v_knn = build_knn_neighbourhood(v_sim, topk=self.knn_k)
                v_adj = compute_normalized_laplacian(v_knn)
                mm_adj = v_adj

            if self.has_t:
                t_raw = self.text_embedding.weight.detach().cpu()
                t_sim = build_sim(t_raw)
                t_knn = build_knn_neighbourhood(t_sim, topk=self.knn_k)
                t_adj = compute_normalized_laplacian(t_knn)
                mm_adj = t_adj if mm_adj is None else mm_adj

            if self.has_v and self.has_t:
                # fixed weighted add (paper: weighted added; ambiguous if learnable)
                mm_adj = self.mm_image_weight * v_adj + (1.0 - self.mm_image_weight) * t_adj

        torch.save(mm_adj, cache_path)
        return mm_adj

    # --------- principal subgraph extraction ---------
    def pre_epoch_processing(self):
        """
        Called by framework each epoch (if supported). We update masked_ui_adj for training:
        - global-aware: SVD-based reconstruction + truncation + sparsification.
        - local-aware: degree-aware multinomial sampling to keep p% edges.
        """
        if self.principal_mode == 'local':
            self.masked_ui_adj = self._local_aware_extraction()
        elif self.principal_mode == 'global':
            self.masked_ui_adj = self._global_aware_extraction()
        else:
            raise ValueError("config['principal_mode'] must be 'local' or 'global'.")

    def _local_aware_extraction(self):
        # multinomial sampling on original edges with prob weights
        E = self.edge_prob.numel()
        keep_len = int(E * self.local_p)
        # torch.multinomial requires non-negative and sum>0
        prob = self.edge_prob
        # if keep_len==0, fallback to keep at least 1 edge (minimal safety)
        if keep_len < 1:
            keep_len = 1  # minimal necessary assumption
        idx = torch.multinomial(prob, keep_len, replacement=False)
        keep_edges = self.edge_u_i[:, idx]  # [2, keep_len]
        adj = self._edges_to_ui_adj(keep_edges, self.n_users, self.n_items).to(self.device)
        return adj

    def _global_aware_extraction(self):
        """
        Global-aware truncated reconstruction using randomized SVD approximation.
        Implementation:
        - Take complete normalized adjacency (sparse) -> dense (assumption; costly).
        - Compute low-rank SVD with rank d=embedding_dim using torch.svd_lowrank.
        - Truncate singular values by keeping head gamma*d and tail (1-gamma)*d (paper description).
        - Reconstruct dense matrix and sparsify by epsilon threshold.
        - Convert to sparse adjacency and coalesce.
        """
        # WARNING: Dense conversion is expensive. Kept faithful to described SVD reconstruction.
        A = self._complete_ui_adj_cached.to_dense()  # [N,N]
        d = self.embedding_dim

        # randomized SVD approximation
        # torch.svd_lowrank returns U [N,q], S [q], V [N,q]
        # choose q=d; for numerical stability, we could use q=d+5 but that is extra assumption -> keep q=d.
        U, S, V = torch.svd_lowrank(A, q=d)

        # truncation
        k_head = int(self.global_gamma * d)
        k_tail = d - k_head
        # paper: keep [1..gamma d] and [(1-gamma)d..d] in eigenvalues; middle set to 0.
        # With descending S, "tail" refers to smallest components; we keep them too.
        S_trunc = S.clone()
        if k_head < 0 or k_head > d:
            raise ValueError("global_gamma out of range leads to invalid truncation.")
        # zero-out middle: [k_head : d-k_tail] == [k_head : k_head] if k_tail==d-k_head -> ok
        mid_l = k_head
        mid_r = d - k_tail
        if mid_r > mid_l:
            S_trunc[mid_l:mid_r] = 0.0

        # reconstruct: A_hat = U diag(S_trunc) V^T
        A_hat = (U * S_trunc.unsqueeze(0)) @ V.t()

        # sparsification: set abs(val) < epsilon to 0
        eps = self.global_epsilon
        mask = A_hat.abs() >= eps
        # avoid empty
        if mask.sum() == 0:
            # minimal fallback: use complete graph if everything pruned
            return self._complete_ui_adj_cached

        idx = mask.nonzero(as_tuple=False).t().contiguous()  # [2, nnz]
        val = A_hat[idx[0], idx[1]]
        adj = torch.sparse_coo_tensor(idx, val, torch.Size((self.n_nodes, self.n_nodes)),
                                      device=self.device).coalesce()

        # Minimal necessary assumption:
        # reconstructed adjacency may not be symmetric; enforce symmetry as it is an undirected interaction graph.
        # (paper implies subgraph from complete user-item graph, which is undirected bipartite).
        adj_t = torch.sparse_coo_tensor(adj.indices().flip(0), adj.values(), adj.shape, device=self.device).coalesce()
        # average (A + A^T)/2
        adj = (adj + adj_t).coalesce()
        # Scale values by 0.5 if duplicates? coalesce sums duplicates; divide by 2 for exact averaging
        adj = torch.sparse_coo_tensor(adj.indices(), adj.values() * 0.5, adj.shape, device=self.device).coalesce()
        return adj

    # --------- embeddings & message passing ---------
    def _get_item_raw_embeddings(self):
        feats = []
        if self.has_v:
            v = self.image_mlp(self.image_embedding.weight)  # [n_items, feat_embed_dim]
            feats.append(v)
        if self.has_t:
            t = self.text_mlp(self.text_embedding.weight)     # [n_items, feat_embed_dim]
            feats.append(t)
        item_raw = torch.cat(feats, dim=-1)                   # [n_items, item_raw_dim]
        item_raw = self.item_fuse(item_raw)                   # [n_items, embedding_dim]
        return item_raw

    def _lightgcn_propagate(self, adj: torch.Tensor, ego_embeddings: torch.Tensor, n_layers: int):
        """
        LightGCN propagation with layer-wise mean pooling.
        adj: sparse [N,N]
        ego_embeddings: dense [N,D]
        return: dense [N,D]
        """
        all_embeddings = [ego_embeddings]
        x = ego_embeddings
        for _ in range(n_layers):
            x = torch.sparse.mm(adj, x)
            all_embeddings.append(x)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        return all_embeddings

    def forward(self, adj=None, is_training=False):
        """
        Returns:
            user_hat [n_users, D], item_hat [n_items, D]
        If is_training:
            principal uses masked_ui_adj (prepared by pre_epoch_processing) if available, else complete.
        Else:
            principal uses complete graph.
        Latent item-item propagation always uses fixed mm_adj.
        """
        # raw embeddings
        u0 = self.user_embedding.weight  # [U,D]
        i0 = self._get_item_raw_embeddings()  # [I,D]
        ego = torch.cat([u0, i0], dim=0)  # [N,D]

        # principal graph message passing on UI graph
        if is_training:
            ui_adj = self.masked_ui_adj if self.masked_ui_adj is not None else self._complete_ui_adj_cached
        else:
            ui_adj = self._complete_ui_adj_cached
        if adj is not None:
            # keep compatibility: if caller passes adj explicitly, use it (framework/official variations)
            ui_adj = adj

        principal_all = self._lightgcn_propagate(ui_adj, ego, self.n_ui_layers)
        u_principal, i_principal = torch.split(principal_all, [self.n_users, self.n_items], dim=0)

        # latent graph message passing on item-item graph (items only)
        h = i0
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self._mm_adj_cached, h)
        i_latent = h
        u_latent = torch.zeros_like(u_principal)  # paper defines e_u^latent; not specified.
        # Minimal assumption: latent graph is item-item only -> no user latent; set to zeros.

        # fusion for final prediction
        u_hat = u_principal + u_latent
        i_hat = i_principal + i_latent
        return u_hat, i_hat

    # --------- losses ---------
    def _bpr_loss(self, u_emb, pos_emb, neg_emb):
        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _feature_mask(self, x: torch.Tensor, rho: float):
        """
        Feature masking: randomly zero-out features with ratio rho.
        Returns two masked views x', x''.
        """
        if rho <= 0.0:
            return x, x
        keep_prob = 1.0 - rho
        m1 = (torch.rand_like(x) < keep_prob).type_as(x)
        m2 = (torch.rand_like(x) < keep_prob).type_as(x)
        return x * m1, x * m2

    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor, tau: float):
        """
        In-batch InfoNCE (symmetric not required by paper; it presents one direction).
        We follow paper equation style: for each sample, positive is diagonal, negatives are all in batch.
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = (z1 @ z2.t()) / tau  # [B,B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        u_hat, i_hat = self.forward(is_training=True)

        u_b = u_hat[users]
        pos_b = i_hat[pos_items]
        neg_b = i_hat[neg_items]
        loss_bpr = self._bpr_loss(u_b, pos_b, neg_b)

        # SSL: feature masking + InfoNCE on (all users in batch) and (all items in batch)
        u1, u2 = self._feature_mask(u_b, self.ssl_mask_ratio)
        i1, i2 = self._feature_mask(pos_b, self.ssl_mask_ratio)  # paper sums over all items; batch approximation.

        loss_ssl_u = self._infonce(u1, u2, self.ssl_temp)
        loss_ssl_i = self._infonce(i1, i2, self.ssl_temp)
        loss_ssl = loss_ssl_u + loss_ssl_i

        return loss_bpr + self.ssl_weight * loss_ssl

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        u_hat, i_hat = self.forward(is_training=False)
        u = u_hat[user]  # [B,D]
        scores = u @ i_hat.t()  # [B, I]
        return scores