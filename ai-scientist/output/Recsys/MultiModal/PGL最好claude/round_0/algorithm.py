# coding: utf-8
r"""
PGL: Principal Graph Learning for Multimedia Recommendation
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
        super(PGL, self).__init__(config, dataset)

        # Basic configuration
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_mm_layers = config['n_mm_layers'] if 'n_mm_layers' in config else 1
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        
        # PGL specific parameters
        self.extraction_type = config['extraction_type'] if 'extraction_type' in config else 'local'  # 'global' or 'local'
        self.truncation_ratio = config['truncation_ratio'] if 'truncation_ratio' in config else 0.3  # gamma for global-aware
        self.sampling_ratio = config['sampling_ratio'] if 'sampling_ratio' in config else 0.3  # p for local-aware
        self.sparsification_threshold = config['sparsification_threshold'] if 'sparsification_threshold' in config else 1e-3  # epsilon
        self.svd_rank = config['svd_rank'] if 'svd_rank' in config else self.embedding_dim  # d for SVD
        
        # SSL parameters
        self.ssl_weight = config['ssl_weight'] if 'ssl_weight' in config else 0.1  # lambda_SSL
        self.ssl_temp = config['ssl_temp'] if 'ssl_temp' in config else 0.2  # tau
        self.mask_ratio = config['mask_ratio'] if 'mask_ratio' in config else 0.5  # rho
        
        # Modality weight
        self.mm_image_weight = config['mm_image_weight'] if 'mm_image_weight' in config else 0.5

        self.n_nodes = self.n_users + self.n_items

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Get normalized adjacency matrix for user-item graph
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # For principal graph learning during training
        self.principal_adj = None
        
        # Edge information for local-aware extraction
        self.edge_indices, self.edge_weights = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_weights = self.edge_weights.to(self.device)

        # User embedding (ID-based)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # Item embeddings - using content features through MLP
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Item ID embedding for combining with content features
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Build or load pre-computed item-item similarity graphs
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'pgl_mm_adj_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            self.mm_adj = self._build_mm_adj()
            torch.save(self.mm_adj, mm_adj_file)
        
        self.mm_adj = self.mm_adj.to(self.device)

    def _build_mm_adj(self):
        """Build multimodal item-item similarity graph"""
        mm_adj = None
        
        if self.v_feat is not None:
            image_adj = self._build_knn_adj(self.v_feat)
            mm_adj = image_adj
            
        if self.t_feat is not None:
            text_adj = self._build_knn_adj(self.t_feat)
            if mm_adj is None:
                mm_adj = text_adj
            else:
                mm_adj = self.mm_image_weight * mm_adj + (1.0 - self.mm_image_weight) * text_adj
        
        return mm_adj

    def _build_knn_adj(self, features):
        """Build kNN adjacency matrix from features"""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Compute similarity
        sim = torch.mm(features_norm, features_norm.t())
        # Get top-k neighbors
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        # Build sparse adjacency matrix
        n_items = features.shape[0]
        indices0 = torch.arange(n_items, device=features.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
        
        # Compute normalized Laplacian
        adj = torch.sparse_coo_tensor(
            indices, 
            torch.ones(indices.shape[1], device=features.device),
            (n_items, n_items)
        ).coalesce()
        
        # Symmetric normalization D^{-1/2} A D^{-1/2}
        row_sum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-7
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        
        return torch.sparse_coo_tensor(indices, values, (n_items, n_items)).coalesce()

    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix for user-item bipartite graph"""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        # Symmetric normalization
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to torch sparse tensor
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes))).coalesce()

    def get_edge_info(self):
        """Get edge indices and weights for local-aware extraction"""
        rows = torch.from_numpy(self.interaction_matrix.row).long()
        cols = torch.from_numpy(self.interaction_matrix.col).long()
        edges = torch.stack([rows, cols], dim=0)
        
        # Compute degree-based weights: 1/sqrt(|N_u| * |N_i|)
        user_degrees = torch.from_numpy(np.array(self.interaction_matrix.sum(axis=1)).flatten()).float()
        item_degrees = torch.from_numpy(np.array(self.interaction_matrix.sum(axis=0)).flatten()).float()
        
        weights = 1.0 / (torch.sqrt(user_degrees[rows] + 1e-7) * torch.sqrt(item_degrees[cols] + 1e-7))
        weights = weights / weights.sum()  # Normalize to probability distribution
        
        return edges, weights

    def pre_epoch_processing(self):
        """Extract principal subgraph before each epoch during training"""
        if self.extraction_type == 'global':
            self.principal_adj = self._global_aware_extraction()
        else:
            self.principal_adj = self._local_aware_extraction()

    def _global_aware_extraction(self):
        """Global-aware extraction using randomized SVD and truncated reconstruction"""
        # Convert sparse adj to dense for SVD (only for user-item part)
        # For efficiency, we work with the bipartite structure
        adj_dense = self.norm_adj.to_dense()
        
        # Use randomized SVD for efficiency
        try:
            # Use torch.svd_lowrank for randomized SVD approximation
            U, S, Vh = torch.svd_lowrank(adj_dense, q=self.svd_rank)
            V = Vh.t()
        except:
            # Fallback to standard SVD if lowrank fails
            U, S, V = torch.svd(adj_dense)
            U = U[:, :self.svd_rank]
            S = S[:self.svd_rank]
            V = V[:, :self.svd_rank]
        
        # Truncation: keep top gamma*d and bottom (1-gamma)*d eigenvalues
        d = S.shape[0]
        top_k = int(self.truncation_ratio * d)
        bottom_k = int(self.truncation_ratio * d)
        
        # Create mask for eigenvalues to keep
        mask = torch.zeros_like(S)
        if top_k > 0:
            mask[:top_k] = 1.0
        if bottom_k > 0:
            mask[-bottom_k:] = 1.0
        
        # Truncated reconstruction
        S_truncated = S * mask
        reconstructed = torch.mm(torch.mm(U, torch.diag(S_truncated)), V.t())
        
        # Sparsification: set elements below threshold to 0
        reconstructed[torch.abs(reconstructed) < self.sparsification_threshold] = 0
        
        # Convert back to sparse
        indices = torch.nonzero(reconstructed, as_tuple=False).t()
        values = reconstructed[indices[0], indices[1]]
        
        # Re-normalize
        sparse_adj = torch.sparse_coo_tensor(indices, values, reconstructed.shape).coalesce()
        
        return self._normalize_sparse_adj(sparse_adj)

    def _local_aware_extraction(self):
        """Local-aware extraction using multinomial sampling"""
        n_edges = self.edge_weights.shape[0]
        n_keep = int(n_edges * self.sampling_ratio)
        
        # Multinomial sampling based on edge weights
        sampled_indices = torch.multinomial(self.edge_weights, n_keep, replacement=False)
        
        # Get sampled edges
        sampled_edges = self.edge_indices[:, sampled_indices]
        
        # Build sparse adjacency matrix
        # User-item edges
        user_indices = sampled_edges[0]
        item_indices = sampled_edges[1] + self.n_users
        
        # Create symmetric adjacency (user-item and item-user)
        all_rows = torch.cat([user_indices, item_indices])
        all_cols = torch.cat([item_indices, user_indices])
        indices = torch.stack([all_rows, all_cols], dim=0)
        values = torch.ones(indices.shape[1], device=self.device)
        
        sparse_adj = torch.sparse_coo_tensor(indices, values, (self.n_nodes, self.n_nodes)).coalesce()
        
        return self._normalize_sparse_adj(sparse_adj)

    def _normalize_sparse_adj(self, adj):
        """Apply symmetric normalization to sparse adjacency matrix"""
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        
        # Compute degree
        degree = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        d_inv_sqrt = torch.pow(degree, -0.5)
        
        # Normalize values
        new_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]
        
        return torch.sparse_coo_tensor(indices, new_values, adj.shape).coalesce()

    def get_item_raw_embeddings(self):
        """Get item raw representations by concatenating/combining content features"""
        item_embeds = []
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            item_embeds.append(image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            item_embeds.append(text_feats)
        
        if len(item_embeds) > 1:
            # Average multimodal features
            item_content = torch.stack(item_embeds, dim=0).mean(dim=0)
        elif len(item_embeds) == 1:
            item_content = item_embeds[0]
        else:
            item_content = None
        
        return item_content

    def principal_graph_learning(self, adj):
        """Message passing on principal/complete user-item graph"""
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return user_embeddings, item_embeddings

    def latent_graph_learning(self):
        """Message passing on latent item-item graph"""
        item_content = self.get_item_raw_embeddings()
        
        if item_content is None:
            return torch.zeros(self.n_items, self.embedding_dim, device=self.device)
        
        h = item_content
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        
        return h

    def forward(self, adj=None, training=True):
        """
        Forward pass
        During training: use principal subgraph
        During inference: use complete graph
        """
        if training and self.principal_adj is not None:
            adj_to_use = self.principal_adj
        else:
            adj_to_use = self.norm_adj
        
        # Principal graph learning on user-item graph
        user_principal, item_principal = self.principal_graph_learning(adj_to_use)
        
        # Latent graph learning on item-item graph
        item_latent = self.latent_graph_learning()
        
        # Combine representations
        user_embeddings = user_principal
        item_embeddings = item_principal + F.normalize(item_latent, p=2, dim=1)
        
        return user_embeddings, item_embeddings

    def feature_masking(self, embeddings, mask_ratio):
        """Apply random feature masking"""
        mask = torch.bernoulli(torch.ones_like(embeddings) * (1 - mask_ratio))
        return embeddings * mask

    def ssl_loss(self, user_embeddings, item_embeddings, users, pos_items):
        """Self-supervised contrastive loss with feature masking"""
        # Get embeddings for batch
        user_emb = user_embeddings[users]
        item_emb = item_embeddings[pos_items]
        
        # Generate two views through masking
        user_view1 = self.feature_masking(user_emb, self.mask_ratio)
        user_view2 = self.feature_masking(user_emb, self.mask_ratio)
        item_view1 = self.feature_masking(item_emb, self.mask_ratio)
        item_view2 = self.feature_masking(item_emb, self.mask_ratio)
        
        # Normalize
        user_view1 = F.normalize(user_view1, p=2, dim=1)
        user_view2 = F.normalize(user_view2, p=2, dim=1)
        item_view1 = F.normalize(item_view1, p=2, dim=1)
        item_view2 = F.normalize(item_view2, p=2, dim=1)
        
        # User contrastive loss (in-batch negatives)
        user_sim = torch.mm(user_view1, user_view2.t()) / self.ssl_temp
        user_labels = torch.arange(user_sim.shape[0], device=self.device)
        user_loss = F.cross_entropy(user_sim, user_labels)
        
        # Item contrastive loss (in-batch negatives)
        item_sim = torch.mm(item_view1, item_view2.t()) / self.ssl_temp
        item_labels = torch.arange(item_sim.shape[0], device=self.device)
        item_loss = F.cross_entropy(item_sim, item_labels)
        
        return user_loss + item_loss

    def bpr_loss(self, users, pos_items, neg_items, user_embeddings, item_embeddings):
        """BPR loss for recommendation"""
        user_emb = user_embeddings[users]
        pos_emb = item_embeddings[pos_items]
        neg_emb = item_embeddings[neg_items]
        
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization
        reg_loss = (1/2) * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return loss + self.reg_weight * reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        # Forward pass (training mode)
        user_embeddings, item_embeddings = self.forward(training=True)
        
        # BPR loss
        bpr_loss = self.bpr_loss(users, pos_items, neg_items, user_embeddings, item_embeddings)
        
        # SSL loss
        ssl_loss = self.ssl_loss(user_embeddings, item_embeddings, users, pos_items)
        
        # Total loss
        total_loss = bpr_loss + self.ssl_weight * ssl_loss
        
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        # Forward pass (inference mode - use complete graph)
        user_embeddings, item_embeddings = self.forward(training=False)
        
        u_embeddings = user_embeddings[user]
        
        # Compute scores
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        
        return scores