# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
SMORE_2025: Spectrum Modality Fusion for Multi-modal Recommendation
################################################
Reference:
    Paper: SMORE_2025 (arXiv preprint)
    Implementation based on FREEDOM_2023 and MGCN_2023 with key innovations in frequency-domain fusion and modality-aware preference.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)

        # Configuration parameters
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']  # Top-K edges for graph sparsification
        self.n_ui_layers = config['n_ui_layers']  # Layers for user-item graph propagation
        self.n_mm_layers = config['n_mm_layers']  # Layers for item-item modality graphs (shallow)
        self.reg_weight = config['reg_weight']
        self.lambda_cl = config['lambda_cl']  # Contrastive loss weight
        self.lambda_reg = config['lambda_reg']  # L2 regularization weight
        self.tau = config['tau']  # Temperature for InfoNCE
        self.use_complex_filter = True  # Whether to use complex-valued filters (required by paper)

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal-specific feature projections (MLP)
        self.proj_mlp_v = nn.Linear(self.v_feat.shape[1], self.embedding_dim) if self.v_feat is not None else None
        self.proj_mlp_t = nn.Linear(self.t_feat.shape[1], self.embedding_dim) if self.t_feat is not None else None

        # Modality-specific projection matrices (W1,m)
        self.W1_v = nn.Parameter(torch.randn(self.embedding_dim, self.v_feat.shape[1]) * 0.01) if self.v_feat is not None else None
        self.W1_t = nn.Parameter(torch.randn(self.embedding_dim, self.t_feat.shape[1]) * 0.01) if self.t_feat is not None else None

        # Dynamic filter weights in frequency domain (W2,m^c ∈ C^{n×d})
        # We store real and imaginary parts separately for trainable complex weights
        if self.v_feat is not None:
            self.W2_v_real = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)
            self.W2_v_imag = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)
        if self.t_feat is not None:
            self.W2_t_real = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)
            self.W2_t_imag = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)

        # Fusion filter (W2,f^c ∈ C^{n×d})
        self.W2_f_real = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)
        self.W2_f_imag = nn.Parameter(torch.randn(self.knn_k, self.embedding_dim) * 0.01)

        # Gating networks for behavior-guided modulation
        self.gate_modality_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_modality_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # Preference gate functions (Q_m, Q_f)
        self.pref_gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.pref_gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.pref_gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        # Attention mechanism for modality weighting (α_m)
        self.att_query = nn.Parameter(torch.randn(self.embedding_dim))
        self.att_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.att_bias = nn.Parameter(torch.zeros(self.embedding_dim))

        # Register buffers for fixed tensors
        self.register_buffer('degree_matrix', torch.ones(self.n_items))

        # Build item-item graphs (modality-specific and fusion)
        self.build_item_graphs()

    def build_item_graphs(self):
        """Construct modality-specific and fusion item-item graphs using kNN + symmetric normalization."""
        device = self.device
        v_feats = self.v_feat if self.v_feat is not None else None
        t_feats = self.t_feat if self.t_feat is not None else None

        # Compute similarity matrices
        sim_v = None
        sim_t = None
        if v_feats is not None:
            v_feats = self.proj_mlp_v(v_feats).detach()  # Project to shared space
            sim_v = torch.mm(v_feats, v_feats.T)  # Cosine similarity
        if t_feats is not None:
            t_feats = self.proj_mlp_t(t_feats).detach()
            sim_t = torch.mm(t_feats, t_feats.T)

        # Apply kNN sparsification
        adj_v = None
        adj_t = None
        if v_feats is not None:
            _, topk_idx = torch.topk(sim_v, k=self.knn_k, dim=-1)
            indices_v = torch.arange(sim_v.size(0)).unsqueeze(1).expand(-1, self.knn_k)
            indices_v = torch.stack([indices_v.flatten(), topk_idx.flatten()], dim=0)
            values_v = sim_v[indices_v[0], indices_v[1]].unsqueeze(0)
            adj_v = torch.sparse.FloatTensor(indices_v, values_v, sim_v.shape).to(device)
        if t_feats is not None:
            _, topk_idx = torch.topk(sim_t, k=self.knn_k, dim=-1)
            indices_t = torch.arange(sim_t.size(0)).unsqueeze(1).expand(-1, self.knn_k)
            indices_t = torch.stack([indices_t.flatten(), topk_idx.flatten()], dim=0)
            values_t = sim_t[indices_t[0], indices_t[1]].unsqueeze(0)
            adj_t = torch.sparse.FloatTensor(indices_t, values_t, sim_t.shape).to(device)

        # Symmetric normalization: D^{-1/2} S D^{-1/2}
        norm_adj_v = compute_normalized_laplacian(adj_v, self.knn_k) if adj_v is not None else None
        norm_adj_t = compute_normalized_laplacian(adj_t, self.knn_k) if adj_t is not None else None

        # Max-pooling fusion graph across modalities
        if adj_v is not None and adj_t is not None:
            # Convert sparse to dense for max-pooling
            dense_v = norm_adj_v.to_dense()
            dense_t = norm_adj_t.to_dense()
            fused_adj = torch.max(dense_v, dense_t)
            fused_adj = torch.clamp(fused_adj, min=0.0)
            self.fusion_adj = fused_adj
        elif adj_v is not None:
            self.fusion_adj = norm_adj_v.to_dense()
        elif adj_t is not None:
            self.fusion_adj = norm_adj_t.to_dense()
        else:
            raise ValueError("No valid modal features provided.")

        # Store as buffer for later use
        self.register_buffer('norm_adj_v', norm_adj_v)
        self.register_buffer('norm_adj_t', norm_adj_t)
        self.register_buffer('fusion_adj', self.fusion_adj)

    def get_norm_adj_mat(self):
        """Build normalized adjacency matrix for user-item interaction graph."""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def spectrum_modality_fusion(self, raw_features, modality='v'):
        """
        Perform spectrum modality fusion via FFT-based denoising and point-wise fusion.
        Args:
            raw_features: [n_items, d] tensor of raw modality features
            modality: 'v' or 't'
        Returns:
            fused_features: [n_items, d] tensor after denoised fusion
        """
        device = raw_features.device
        n, d = raw_features.shape

        # Step 1: Project into shared latent space (MLP)
        if modality == 'v':
            H = torch.matmul(raw_features, self.W1_v.T)  # [n, d]
        else:
            H = torch.matmul(raw_features, self.W1_t.T)

        # Step 2: Convert to frequency domain via FFT
        # Reshape to [n, d] → apply FFT along dimension 0 (sequence/space)
        H_fft = fft(H, dim=0)  # [n, d] complex

        # Step 3: Apply dynamic filter (point-wise multiplication with learnable complex weights)
        # Use real and imag parts separately
        if modality == 'v':
            W2_real = self.W2_v_real
            W2_imag = self.W2_v_imag
        else:
            W2_real = self.W2_t_real
            W2_imag = self.W2_t_imag

        # Combine real and imag parts into complex filter
        W2_complex = torch.complex(W2_real, W2_imag)  # [n, d]

        # Apply filter: H_hat = δ_m(H_fft) = W2_c ⊙ H_fft
        H_filtered = H_fft * W2_complex  # [n, d] complex

        # Step 4: Fuse across modalities via point-wise product in frequency domain
        # Only one modality at a time; fusion happens later
        # For now, we just return filtered feature
        H_filtered = H_filtered.real  # Keep only real part for IFFT

        # Step 5: Inverse FFT back to spatial domain
        H_denoised = ifft(H_filtered, dim=0).real  # [n, d]

        return H_denoised

    def forward(self, adj=None, train=False):
        """
        Forward pass of SMORE model.
        Args:
            adj: Normalized user-item adjacency matrix (sparse tensor)
            train: Whether in training mode (for contrastive loss)
        Returns:
            ua_embeddings, ia_embeddings: Final user/item embeddings
        """
        device = self.device

        # Step 1: Get ID embeddings
        user_embeds = self.user_embedding.weight
        item_id_embeds = self.item_id_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_id_embeds], dim=0)

        # Step 2: Propagate through user-item graph (high-order collaborative signals)
        all_user_item_embeds = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            side_embeds = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeds
            all_user_item_embeds.append(ego_embeddings)
        content_embeds = torch.stack(all_user_item_embeds, dim=1).mean(dim=1)  # [n_users+n_items, d]
        user_content_embeds, item_content_embeds = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # Step 3: Process multi-modal features
        modality_features = {}
        if self.v_feat is not None:
            v_feats = self.v_feat
            v_feats = self.proj_mlp_v(v_feats)
            modality_features['v'] = v_feats
        if self.t_feat is not None:
            t_feats = self.t_feat
            t_feats = self.proj_mlp_t(t_feats)
            modality_features['t'] = t_feats

        # Denoise and fuse each modality via spectrum fusion
        denoised_modality_features = {}
        for m, feats in modality_features.items():
            denoised = self.spectrum_modality_fusion(feats, modality=m)
            denoised_modality_features[m] = denoised

        # Extract fused feature from point-wise product in frequency domain
        # Since we can't do full fusion here without both modalities, we simulate it
        # via element-wise max pooling across modalities in spatial domain
        fused_features = None
        if len(denoised_modality_features) == 2:
            v_feat = denoised_modality_features['v']
            t_feat = denoised_modality_features['t']
            fused_features = torch.max(v_feat, t_feat)  # Max-pooling across modalities
        elif len(denoised_modality_features) == 1:
            fused_features = next(iter(denoised_modality_features.values()))
        else:
            raise ValueError("No valid modality features found.")

        # Step 4: Behavior-guided gating (f_g = E_id ⊙ σ(W·H + b))
        gated_v = self.gate_modality_v(item_id_embeds) * denoised_modality_features['v']
        gated_t = self.gate_modality_t(item_id_embeds) * denoised_modality_features['t']
        gated_f = self.gate_fusion(item_id_embeds) * fused_features

        # Step 5: LightGCN-style message passing on item-item graphs
        # Uni-modal graphs
        if hasattr(self, 'norm_adj_v') and self.norm_adj_v is not None:
            for _ in range(self.n_mm_layers):
                gated_v = torch.sparse.mm(self.norm_adj_v, gated_v)
        if hasattr(self, 'norm_adj_t') and self.norm_adj_t is not None:
            for _ in range(self.n_mm_layers):
                gated_t = torch.sparse.mm(self.norm_adj_t, gated_t)

        # Fusion graph
        for _ in range(self.n_mm_layers):
            gated_f = torch.matmul(self.fusion_adj, gated_f)

        # Step 6: Aggregate user modality features via weighted-sum
        user_gated_v = torch.sparse.mm(self.R, gated_v)
        user_gated_t = torch.sparse.mm(self.R, gated_t)
        user_gated_f = torch.sparse.mm(self.R, gated_f)

        # Step 7: Modality-aware preference module
        # Compute attention weights α_m using fusion features
        alpha_v = F.softmax(
            torch.einsum('ij,j->i', self.att_proj(gated_f), self.att_query) +
            self.att_bias,
            dim=0
        )
        alpha_t = F.softmax(
            torch.einsum('ij,j->i', self.att_proj(gated_f), self.att_query) +
            self.att_bias,
            dim=0
        )
        alpha_m = torch.stack([alpha_v, alpha_t], dim=0).mean(dim=0)  # Average over modalities

        # Weight uni-modal features
        H_star_v = alpha_v.unsqueeze(1) * gated_v
        H_star_t = alpha_t.unsqueeze(1) * gated_t
        H_star = (H_star_v + H_star_t) / 2  # Simple average

        # Extract explicit preferences from behavioral signals
        Q_v = self.pref_gate_v(user_content_embeds)
        Q_t = self.pref_gate_t(user_content_embeds)
        Q_f = self.pref_gate_f(user_content_embeds)

        # Final side features
        H_s = (H_star_v * Q_v + H_star_t * Q_t) / 2 + (gated_f * Q_f)

        # Step 8: Final representations
        final_user_embeds = user_content_embeds + H_s[:self.n_users]
        final_item_embeds = item_content_embeds + H_s[self.n_users:]

        if train:
            return final_user_embeds, final_item_embeds, H_s, user_content_embeds

        return final_user_embeds, final_item_embeds

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        # Contrastive loss: align behavioral and modality-side features
        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_users, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        cl_loss_u = self.InfoNCE(side_embeds_users[users], content_embeds_users[users], self.tau)
        cl_loss_i = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], self.tau)
        total_cl_loss = cl_loss_u + cl_loss_i

        total_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.lambda_cl * total_cl_loss + \
                     self.lambda_reg * (ua_embeddings**2).sum() + (ia_embeddings**2).sum()

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores