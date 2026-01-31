# coding: utf-8
r"""
CM3: Calibrated Multimodal Recommendation with Spherical Bézier Fusion
Paper: CM3_2025
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

        # Store config for later use
        self.config = config

        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim'] if 'feat_embed_dim' in config else self.embedding_dim
        self.hidden_dim = config['hidden_dim'] if 'hidden_dim' in config else 256
        self.n_ui_layers = config['n_ui_layers'] if 'n_ui_layers' in config else 2
        self.n_ii_layers = config['n_ii_layers'] if 'n_ii_layers' in config else 1
        self.knn_k = config['knn_k'] if 'knn_k' in config else 10
        self.gamma = config['gamma'] if 'gamma' in config else 1.0
        self.temperature = config['temperature'] if 'temperature' in config else 2.0
        self.alpha = config['alpha'] if 'alpha' in config else 0.4
        self.mm_image_weight = config['mm_image_weight'] if 'mm_image_weight' in config else 0.5

        self.n_nodes = self.n_users + self.n_items
        
        # Determine number of modalities
        self.n_modalities = 0
        if self.v_feat is not None:
            self.n_modalities += 1
        if self.t_feat is not None:
            self.n_modalities += 1
        
        # Total dimension: |M| modalities + 1 mixed feature
        self.total_dim = (self.n_modalities + 1) * self.feat_embed_dim

        # Load and build adjacency matrices
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # User embeddings: separate for each modality pathway + mixed
        # Following the paper's Section 3.4.2: partition user embeddings into |M|+1 segments
        self.user_embedding = nn.Embedding(self.n_users, self.total_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.feat_embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # User preference weight matrix W_3 from Equation (7)
        # Shape: [n_users, n_modalities+1, 1] for element-wise weighting
        self.user_pref_weight = nn.Parameter(
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

        # Build item-item graph
        self.mm_adj = self._build_item_item_graph()

        # Pre-compute similarity scores for calibrated uniformity loss
        self.item_similarity = None  # Will be computed during forward pass

    def get_norm_adj_mat(self):
        """Build normalized adjacency matrix for user-item graph."""
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
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        indices = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        
        return torch.sparse_coo_tensor(indices, data, torch.Size((self.n_nodes, self.n_nodes))).coalesce()

    def _build_item_item_graph(self):
        """Build item-item graph based on multimodal feature similarity."""
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_cm3_{self.knn_k}_{int(10*self.mm_image_weight)}.pt')
        
        if os.path.exists(mm_adj_file):
            return torch.load(mm_adj_file, map_location=self.device)
        
        mm_adj = None
        
        if self.v_feat is not None:
            image_adj = self._get_knn_adj_mat(self.v_feat)
            mm_adj = image_adj
        
        if self.t_feat is not None:
            text_adj = self._get_knn_adj_mat(self.t_feat)
            if mm_adj is None:
                mm_adj = text_adj
            else:
                mm_adj = self.mm_image_weight * mm_adj + (1.0 - self.mm_image_weight) * text_adj
        
        torch.save(mm_adj, mm_adj_file)
        return mm_adj

    def _get_knn_adj_mat(self, features):
        """Build k-NN adjacency matrix from features."""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        
        # Compute similarity matrix
        sim = torch.mm(features_norm, features_norm.transpose(0, 1))
        
        # Get top-k neighbors
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        # Build sparse adjacency matrix
        n_items = features.shape[0]
        indices0 = torch.arange(n_items, device=features.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
        
        # Symmetric normalization
        adj = torch.sparse_coo_tensor(
            indices, 
            torch.ones(indices.shape[1], device=features.device),
            (n_items, n_items)
        ).coalesce()
        
        row_sum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-7
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        
        return torch.sparse_coo_tensor(indices, values, (n_items, n_items)).coalesce().to(self.device)

    def spherical_interpolation(self, a, b, lambda_val):
        """
        Spherical interpolation (Equation 3).
        f(a, b) = sin(λθ)/sin(θ) * a + sin((1-λ)θ)/sin(θ) * b
        """
        # Normalize vectors to unit sphere
        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        
        # Compute angle between vectors
        cos_theta = torch.clamp(torch.sum(a_norm * b_norm, dim=-1, keepdim=True), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        
        sin_theta = torch.sin(theta) + 1e-7
        
        # Spherical interpolation
        coef_a = torch.sin(lambda_val * theta) / sin_theta
        coef_b = torch.sin((1 - lambda_val) * theta) / sin_theta
        
        result = coef_a * a_norm + coef_b * b_norm
        return result

    def spherical_bezier_fusion(self, modality_features):
        """
        Spherical Bézier Multimodal Fusion (Equation 2).
        Iteratively applies spherical interpolation using De Casteljau's algorithm.
        """
        if len(modality_features) == 1:
            return F.normalize(modality_features[0], p=2, dim=-1)
        
        # Sample lambda from Beta distribution
        if self.training:
            lambda_val = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lambda_val = 0.5  # Use mean during evaluation
        
        # Iterative fusion using De Casteljau's algorithm
        result = modality_features[0]
        for i in range(1, len(modality_features)):
            result = self.spherical_interpolation(result, modality_features[i], lambda_val)
        
        return result

    def project_modality_features(self):
        """Project multimodal features to low-dimensional space (Equation 1)."""
        modality_features = []
        
        if self.v_feat is not None:
            image_feat = self.image_proj(self.image_embedding.weight)
            modality_features.append(image_feat)
        
        if self.t_feat is not None:
            text_feat = self.text_proj(self.text_embedding.weight)
            modality_features.append(text_feat)
        
        return modality_features

    def forward(self):
        """Forward pass to compute user and item representations."""
        # Project multimodal features
        modality_features = self.project_modality_features()
        
        # Spherical Bézier fusion
        mixed_features = self.spherical_bezier_fusion(modality_features)
        
        # Concatenate all features (Equation 4)
        # item_features: [n_items, (|M|+1) * d]
        all_item_features = modality_features + [mixed_features]
        item_features = torch.cat(all_item_features, dim=-1)
        
        # Graph learning on user-item graph (Equations 5, 6)
        ego_embeddings = torch.cat([self.user_embedding.weight, item_features], dim=0)
        all_embeddings = [ego_embeddings]
        
        for layer in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # READOUT: sum aggregation
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        # User preference mining (Equation 7)
        # Reshape user embeddings to [n_users, |M|+1, d]
        user_embeddings_reshaped = user_embeddings.view(
            self.n_users, self.n_modalities + 1, self.feat_embed_dim
        )
        
        # Apply preference weights
        user_pref_weight_softmax = F.softmax(self.user_pref_weight, dim=1)
        user_embeddings_weighted = user_embeddings_reshaped * user_pref_weight_softmax
        
        # Reshape back to [n_users, total_dim]
        user_final = user_embeddings_weighted.view(self.n_users, self.total_dim)
        
        # Graph learning on item-item graph (Equation 8)
        item_mm_embeddings = mixed_features
        for layer in range(self.n_ii_layers):
            item_mm_embeddings = torch.sparse.mm(self.mm_adj, item_mm_embeddings)
        
        # Residual connection
        item_mm_final = item_mm_embeddings + mixed_features
        
        # Combine item representations
        # Use the mixed features after item-item graph for final prediction
        item_final = item_embeddings.clone()
        # Replace the mixed feature portion with the enhanced version
        item_final[:, -self.feat_embed_dim:] = item_mm_final
        
        # Store mixed features for calibrated uniformity loss
        self.current_mixed_features = mixed_features
        
        return user_final, item_final

    def compute_similarity_scores(self, mixed_features):
        """Compute clamped similarity scores for calibrated uniformity loss."""
        # Normalize features
        features_norm = F.normalize(mixed_features, p=2, dim=-1)
        # Similarity matrix
        sim = torch.mm(features_norm, features_norm.transpose(0, 1))
        # Clamp to [0, 1]
        sim = torch.clamp(sim, 0.0, 1.0)
        return sim

    def alignment_loss(self, user_emb, item_emb):
        """Alignment loss (Equation 9): bring positive pairs closer."""
        # L2 distance
        return torch.mean(torch.sum((user_emb - item_emb) ** 2, dim=-1))

    def uniformity_loss(self, embeddings, t=2.0):
        """Standard uniformity loss (Equation 9)."""
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sq_dist = torch.cdist(embeddings, embeddings, p=2) ** 2
        # Exclude diagonal
        mask = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        sq_dist = sq_dist[mask].view(embeddings.shape[0], -1)
        return torch.log(torch.mean(torch.exp(-t * sq_dist)))

    def calibrated_uniformity_loss(self, item_emb, mixed_features, t=2.0):
        """
        Calibrated uniformity loss (Equation 10).
        Amplifies repulsion between dissimilar items.
        """
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        
        # Compute similarity scores from mixed features
        sim_scores = self.compute_similarity_scores(mixed_features)
        
        # Compute squared distances
        sq_dist = torch.cdist(item_emb, item_emb, p=2) ** 2
        
        # Calibrated term: ||i - i'||^2 - 2 + 2*s(i, i')
        calibrated_dist = sq_dist - 2 + 2 * sim_scores
        
        # Exclude diagonal
        mask = ~torch.eye(item_emb.shape[0], dtype=torch.bool, device=item_emb.device)
        calibrated_dist = calibrated_dist[mask].view(item_emb.shape[0], -1)
        
        return torch.log(torch.mean(torch.exp(-t * calibrated_dist)))

    def calculate_loss(self, interaction):
        """Calculate the total loss (Equation 12)."""
        users = interaction[0]
        pos_items = interaction[1]
        
        # Forward pass
        user_embeddings, item_embeddings = self.forward()
        
        # Get embeddings for batch
        user_emb = user_embeddings[users]
        pos_item_emb = item_embeddings[pos_items]
        
        # For alignment, we need to match dimensions
        # Sum over modality segments to get unified representation for scoring
        user_emb_unified = user_emb.view(-1, self.n_modalities + 1, self.feat_embed_dim).sum(dim=1)
        pos_item_emb_unified = pos_item_emb.view(-1, self.n_modalities + 1, self.feat_embed_dim).sum(dim=1)
        
        # Normalize for alignment loss
        user_emb_norm = F.normalize(user_emb_unified, p=2, dim=-1)
        pos_item_emb_norm = F.normalize(pos_item_emb_unified, p=2, dim=-1)
        
        # Alignment loss
        l_align = self.alignment_loss(user_emb_norm, pos_item_emb_norm)
        
        # Uniformity loss for users (standard)
        batch_user_emb = F.normalize(user_emb_unified, p=2, dim=-1)
        l_uniform_user = self.uniformity_loss(batch_user_emb, t=self.temperature)
        
        # Calibrated uniformity loss for items
        batch_item_emb = pos_item_emb_unified
        batch_mixed_features = self.current_mixed_features[pos_items]
        l_cal_uniform_item = self.calibrated_uniformity_loss(
            batch_item_emb, batch_mixed_features, t=self.temperature
        )
        
        # Total loss (Equation 12)
        loss = l_align + self.gamma * (l_uniform_user + l_cal_uniform_item)
        
        return loss

    def full_sort_predict(self, interaction):
        """Predict scores for all items (Equation 13)."""
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        
        # Get user embeddings
        user_emb = user_embeddings[user]
        
        # Sum over modality segments for unified representation
        user_emb_unified = user_emb.view(-1, self.n_modalities + 1, self.feat_embed_dim).sum(dim=1)
        item_emb_unified = item_embeddings.view(-1, self.n_modalities + 1, self.feat_embed_dim).sum(dim=1)
        
        # Compute scores via inner product
        scores = torch.matmul(user_emb_unified, item_emb_unified.transpose(0, 1))
        
        return scores