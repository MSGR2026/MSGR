# coding: utf-8
r"""
CM3: Calibrated Multimodal Recommendation
Implementation based on the paper description with:
- Spherical Bézier Multimodal Fusion
- Graph Learning on User-Item and Item-Item Graphs
- Alignment and Calibrated Uniformity Losses
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class CM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CM3, self).__init__(config, dataset)

        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config.get('feat_embed_dim', self.embedding_dim)
        self.hidden_dim = config.get('hidden_dim', 256)  # d_1 in paper
        self.n_ui_layers = config.get('n_ui_layers', 3)  # L_ui layers for user-item graph
        self.n_ii_layers = config.get('n_ii_layers', 1)  # L_ii layers for item-item graph
        self.knn_k = config.get('knn_k', 10)  # k for item-item graph construction
        self.temperature = config.get('temperature', 0.2)  # t in uniformity loss
        self.gamma = config.get('gamma', 1.0)  # weight for uniformity losses
        self.alpha = config.get('alpha', 0.5)  # Beta distribution parameter for Mixup
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.sim_clamp_min = config.get('sim_clamp_min', 0.0)
        self.sim_clamp_max = config.get('sim_clamp_max', 1.0)

        self.n_nodes = self.n_users + self.n_items

        # Number of modalities (vision and text)
        self.n_modalities = 0
        if self.v_feat is not None:
            self.n_modalities += 1
        if self.t_feat is not None:
            self.n_modalities += 1
        
        # Total dimension: (|M| + 1) * d for modalities + mixed feature
        self.total_dim = (self.n_modalities + 1) * self.feat_embed_dim

        # Build normalized adjacency matrix for user-item graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # User embeddings: separate parameters for each modality pathway
        # Following the paper's description of partitioning user embeddings
        self.user_embedding_v = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.user_embedding_t = nn.Embedding(self.n_users, self.feat_embed_dim)
        self.user_embedding_mixed = nn.Embedding(self.n_users, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.user_embedding_v.weight)
        nn.init.xavier_uniform_(self.user_embedding_t.weight)
        nn.init.xavier_uniform_(self.user_embedding_mixed.weight)

        # User preference weights: W_3 in paper, shape [n_users, n_modalities+1, 1]
        self.user_pref_weights = nn.Parameter(
            torch.ones(self.n_users, self.n_modalities + 1, 1) / (self.n_modalities + 1)
        )

        # Multimodal feature projection layers (Equation 1)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(self.hidden_dim, self.feat_embed_dim)
            )
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(self.hidden_dim, self.feat_embed_dim)
            )

        # Build and store item-item graph based on multimodal features
        self.mm_adj = self._build_item_item_graph()

        # Pre-compute similarity scores for calibrated uniformity loss
        self.item_similarity = None  # Will be computed in forward

    def get_norm_adj_mat(self):
        """Build normalized adjacency matrix for user-item bipartite graph"""
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        for (r, c), v in data_dict.items():
            A[r, c] = v
        
        # Symmetric normalization: D^{-0.5} A D^{-0.5}
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
        
        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes))).coalesce()

    def _build_item_item_graph(self):
        """Build item-item graph based on multimodal feature similarity"""
        # Use original multimodal features for building item-item graph
        mm_features = []
        if self.v_feat is not None:
            mm_features.append(F.normalize(self.v_feat, p=2, dim=-1))
        if self.t_feat is not None:
            mm_features.append(F.normalize(self.t_feat, p=2, dim=-1))
        
        if len(mm_features) == 0:
            return None
        
        # Average normalized features across modalities
        combined_feat = torch.stack(mm_features, dim=0).mean(dim=0)
        combined_feat = F.normalize(combined_feat, p=2, dim=-1)
        
        # Compute similarity and get top-k neighbors
        sim = torch.mm(combined_feat, combined_feat.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        # Construct sparse adjacency matrix
        indices0 = torch.arange(knn_ind.shape[0], device=self.device)
        indices0 = indices0.unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((indices0.flatten(), knn_ind.flatten()), dim=0)
        
        # Symmetric normalization
        adj_size = (self.n_items, self.n_items)
        adj = torch.sparse_coo_tensor(
            indices, 
            torch.ones(indices.shape[1], device=self.device),
            adj_size
        ).coalesce()
        
        row_sum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-7
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        
        return torch.sparse_coo_tensor(indices, values, adj_size).coalesce().to(self.device)

    def spherical_interpolation(self, a, b, lambda_val):
        """
        Spherical interpolation (slerp) between two vectors (Equation 3)
        Args:
            a: tensor of shape [..., d]
            b: tensor of shape [..., d]
            lambda_val: interpolation parameter, tensor of shape [...] or scalar
        """
        # Normalize inputs to unit sphere
        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        
        # Compute angle between vectors
        cos_theta = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        
        # Handle small angles (use linear interpolation for numerical stability)
        sin_theta = torch.sin(theta).clamp(min=1e-7)
        
        if isinstance(lambda_val, (int, float)):
            lambda_val = torch.tensor(lambda_val, device=a.device, dtype=a.dtype)
        if lambda_val.dim() == 1:
            lambda_val = lambda_val.unsqueeze(-1)
        
        # Spherical interpolation formula
        coef_a = torch.sin(lambda_val * theta) / sin_theta
        coef_b = torch.sin((1 - lambda_val) * theta) / sin_theta
        
        result = coef_a * a_norm + coef_b * b_norm
        
        # For very small angles, fall back to linear interpolation
        small_angle_mask = (theta.abs() < 1e-6).squeeze(-1)
        if small_angle_mask.any():
            linear_result = lambda_val * a_norm + (1 - lambda_val) * b_norm
            result = torch.where(small_angle_mask.unsqueeze(-1), linear_result, result)
        
        return F.normalize(result, p=2, dim=-1)

    def spherical_bezier_fusion(self, features_list):
        """
        Spherical Bézier Multimodal Fusion using De Casteljau's algorithm (Equation 2)
        Args:
            features_list: list of tensors, each of shape [n_items, d]
        Returns:
            mixed features of shape [n_items, d]
        """
        if len(features_list) == 1:
            return F.normalize(features_list[0], p=2, dim=-1)
        
        # Sample lambda from Beta distribution for each item independently
        n_items = features_list[0].shape[0]
        
        # Iteratively apply spherical interpolation
        result = F.normalize(features_list[0], p=2, dim=-1)
        for i in range(1, len(features_list)):
            # Sample lambda independently for each item
            lambda_val = torch.distributions.Beta(self.alpha, self.alpha).sample((n_items,)).to(self.device)
            result = self.spherical_interpolation(features_list[i], result, lambda_val)
        
        return result

    def project_multimodal_features(self):
        """Project multimodal features to low-dimensional space (Equation 1)"""
        projected_features = []
        
        if self.v_feat is not None:
            v_proj = self.image_proj(self.image_embedding.weight)
            projected_features.append(v_proj)
        
        if self.t_feat is not None:
            t_proj = self.text_proj(self.text_embedding.weight)
            projected_features.append(t_proj)
        
        return projected_features

    def forward(self):
        """
        Forward pass to compute user and item representations
        """
        # Project multimodal features
        projected_features = self.project_multimodal_features()
        
        # Spherical Bézier Multimodal Fusion (Equation 2)
        mixed_features = self.spherical_bezier_fusion(projected_features)
        
        # Concatenate modality features with mixed features (Equation 4)
        # Order: [visual, textual, mixed] when both modalities present
        item_features_list = projected_features + [mixed_features]
        item_features = torch.cat(item_features_list, dim=-1)  # [n_items, (|M|+1)*d]
        
        # Concatenate user embeddings in same order
        user_features_list = []
        if self.v_feat is not None:
            user_features_list.append(self.user_embedding_v.weight)
        if self.t_feat is not None:
            user_features_list.append(self.user_embedding_t.weight)
        user_features_list.append(self.user_embedding_mixed.weight)
        user_features = torch.cat(user_features_list, dim=-1)  # [n_users, (|M|+1)*d]
        
        # Graph learning on user-item graph (Equations 5-6)
        ego_embeddings = torch.cat([user_features, item_features], dim=0)
        all_embeddings = [ego_embeddings]
        
        for layer in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # READOUT: sum aggregation
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.sum(dim=1)
        
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        # User preference mining (Equation 7)
        # Reshape user embeddings to [n_users, n_modalities+1, d]
        user_emb_reshaped = user_embeddings.view(
            self.n_users, self.n_modalities + 1, self.feat_embed_dim
        )
        # Apply user-specific preference weights
        pref_weights = F.softmax(self.user_pref_weights, dim=1)  # Normalize weights
        user_emb_weighted = (pref_weights * user_emb_reshaped).view(self.n_users, -1)
        
        # Graph learning on item-item graph (Section 3.4.3)
        item_emb_initial = item_embeddings.clone()
        if self.mm_adj is not None:
            item_emb_ii = item_embeddings
            for layer in range(self.n_ii_layers):
                item_emb_ii = torch.sparse.mm(self.mm_adj, item_emb_ii)
            # Residual connection (Equation 8)
            item_embeddings = item_emb_ii + item_emb_initial
        
        # Store mixed features for calibrated uniformity loss
        self.current_mixed_features = mixed_features
        
        return user_emb_weighted, item_embeddings, mixed_features

    def compute_alignment_loss(self, user_emb, item_emb, users, pos_items):
        """
        Alignment loss: bring positive pairs closer (Equation 9)
        """
        u_emb = F.normalize(user_emb[users], p=2, dim=-1)
        i_emb = F.normalize(item_emb[pos_items], p=2, dim=-1)
        
        return torch.mean(torch.sum((u_emb - i_emb) ** 2, dim=-1))

    def compute_uniformity_loss(self, embeddings):
        """
        Standard uniformity loss (Equation 9)
        """
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        
        # Pairwise squared distances
        sq_dist = torch.cdist(emb_norm, emb_norm, p=2) ** 2
        
        # Uniformity loss
        loss = torch.log(torch.exp(-self.temperature * sq_dist).mean() + 1e-8)
        
        return loss

    def compute_calibrated_uniformity_loss(self, item_emb, mixed_features, items):
        """
        Calibrated uniformity loss for items (Equation 10)
        """
        # Get embeddings for sampled items
        i_emb = F.normalize(item_emb[items], p=2, dim=-1)
        i_mixed = F.normalize(mixed_features[items], p=2, dim=-1)
        
        # Compute similarity scores using mixed features
        sim_scores = torch.mm(i_mixed, i_mixed.t())
        sim_scores = sim_scores.clamp(self.sim_clamp_min, self.sim_clamp_max)
        
        # Compute pairwise squared distances
        sq_dist = torch.cdist(i_emb, i_emb, p=2) ** 2
        
        # Calibrated uniformity: subtract 2 - 2*similarity from squared distance
        calibrated_dist = sq_dist - 2 + 2 * sim_scores
        
        # Uniformity loss with calibration
        loss = torch.log(torch.exp(-self.temperature * calibrated_dist).mean() + 1e-8)
        
        return loss

    def calculate_loss(self, interaction):
        """
        Calculate total loss (Equation 12)
        """
        users = interaction[0]
        pos_items = interaction[1]
        
        # Forward pass
        user_emb, item_emb, mixed_features = self.forward()
        
        # Alignment loss for positive pairs
        align_loss = self.compute_alignment_loss(user_emb, item_emb, users, pos_items)
        
        # Standard uniformity loss for users
        unique_users = torch.unique(users)
        user_uniform_loss = self.compute_uniformity_loss(user_emb[unique_users])
        
        # Calibrated uniformity loss for items
        unique_items = torch.unique(pos_items)
        item_uniform_loss = self.compute_calibrated_uniformity_loss(
            item_emb, mixed_features, unique_items
        )
        
        # Total loss (Equation 12)
        total_loss = align_loss + self.gamma * (user_uniform_loss + item_uniform_loss)
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            self.user_embedding_v.weight.norm(2) +
            self.user_embedding_t.weight.norm(2) +
            self.user_embedding_mixed.weight.norm(2)
        )
        
        return total_loss + reg_loss

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users (Equation 13)
        """
        users = interaction[0]
        
        user_emb, item_emb, _ = self.forward()
        
        u_emb = user_emb[users]
        
        # Inner product for preference scores
        scores = torch.matmul(u_emb, item_emb.t())
        
        return scores