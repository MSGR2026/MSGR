# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
SMORE: Spectral Modality Fusion for Multi-modal Recommendation
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config.get('feat_embed_dim', self.embedding_dim)
        self.knn_k = config['knn_k']
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layers = config.get('n_mm_layers', 1)
        self.reg_weight = config['reg_weight']
        self.cl_weight = config.get('cl_weight', 0.1)
        self.temperature = config.get('temperature', 0.2)
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.R = self.get_R_mat().to(self.device)
        
        # User and Item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modality embeddings and projections
        self.modalities = []
        if self.v_feat is not None:
            self.modalities.append('image')
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.modalities.append('text')
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        
        # Spectrum filters (complex-valued weights for FFT filtering)
        # Using real and imaginary parts separately for compatibility
        self.spectrum_filters = nn.ModuleDict()
        self.spectrum_filters_imag = nn.ModuleDict()
        for m in self.modalities:
            self.spectrum_filters[m] = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
            self.spectrum_filters_imag[m] = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        
        # Fusion filter
        self.fusion_filter_real = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        self.fusion_filter_imag = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        
        # Behavior-guided gating for uni-modal features
        self.modal_gates = nn.ModuleDict()
        for m in self.modalities:
            self.modal_gates[m] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Modality-aware preference attention
        self.modal_attention = nn.ModuleDict()
        for m in self.modalities:
            self.modal_attention[m] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, 1, bias=False)
            )
        
        # Preference gates
        self.pref_gates = nn.ModuleDict()
        for m in self.modalities:
            self.pref_gates[m] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        self.pref_gate_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Build item-item modality graphs
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.modal_adjs = {}
        
        for m in self.modalities:
            adj_file = os.path.join(dataset_path, f'smore_{m}_adj_{self.knn_k}.pt')
            if os.path.exists(adj_file):
                self.modal_adjs[m] = torch.load(adj_file).to(self.device)
            else:
                if m == 'image':
                    feat = self.image_embedding.weight.detach()
                else:
                    feat = self.text_embedding.weight.detach()
                adj = self.build_knn_graph(feat)
                torch.save(adj.cpu(), adj_file)
                self.modal_adjs[m] = adj.to(self.device)
        
        # Build fusion graph
        fusion_adj_file = os.path.join(dataset_path, f'smore_fusion_adj_{self.knn_k}.pt')
        if os.path.exists(fusion_adj_file):
            self.fusion_adj = torch.load(fusion_adj_file).to(self.device)
        else:
            self.fusion_adj = self.build_fusion_graph()
            torch.save(self.fusion_adj.cpu(), fusion_adj_file)
            self.fusion_adj = self.fusion_adj.to(self.device)

    def build_knn_graph(self, features):
        """Build kNN graph from features with symmetric normalization"""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Compute similarity
        sim = torch.mm(features_norm, features_norm.t())
        
        # Get top-K neighbors
        _, knn_indices = torch.topk(sim, self.knn_k, dim=-1)
        
        # Build sparse adjacency matrix
        row_indices = torch.arange(self.n_items, device=features.device).unsqueeze(1).expand(-1, self.knn_k).flatten()
        col_indices = knn_indices.flatten()
        
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.ones(indices.shape[1], device=features.device)
        
        adj = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items))
        adj = adj.coalesce()
        
        # Symmetric normalization
        adj = self.normalize_sparse_adj(adj)
        return adj
    
    def normalize_sparse_adj(self, adj):
        """Symmetric normalization D^{-1/2} A D^{-1/2}"""
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        
        # Compute degree
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        row_sum = row_sum + 1e-7
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        
        # Normalize
        row_indices = indices[0]
        col_indices = indices[1]
        new_values = values * d_inv_sqrt[row_indices] * d_inv_sqrt[col_indices]
        
        return torch.sparse_coo_tensor(indices, new_values, adj.shape).coalesce()
    
    def build_fusion_graph(self):
        """Build fusion graph using max-pooling across modality graphs"""
        if len(self.modalities) == 1:
            return self.modal_adjs[self.modalities[0]]
        
        # Convert sparse to dense for max operation, then back to sparse
        # For efficiency, we use union of edges and take max values
        all_indices = []
        all_values = []
        
        for m in self.modalities:
            adj = self.modal_adjs[m].coalesce()
            all_indices.append(adj.indices())
            all_values.append(adj.values())
        
        # Concatenate all indices and values
        combined_indices = torch.cat(all_indices, dim=1)
        combined_values = torch.cat(all_values, dim=0)
        
        # Create combined sparse tensor and coalesce (this will sum duplicates)
        # We need max, so we'll use a different approach
        # Convert to dense, take max, convert back to sparse
        dense_adjs = []
        for m in self.modalities:
            adj = self.modal_adjs[m]
            dense_adj = adj.to_dense()
            dense_adjs.append(dense_adj)
        
        # Max pooling
        stacked = torch.stack(dense_adjs, dim=0)
        fusion_dense = torch.max(stacked, dim=0)[0]
        
        # Convert back to sparse with top-K sparsification
        _, knn_indices = torch.topk(fusion_dense, self.knn_k, dim=-1)
        row_indices = torch.arange(self.n_items, device=fusion_dense.device).unsqueeze(1).expand(-1, self.knn_k).flatten()
        col_indices = knn_indices.flatten()
        
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.ones(indices.shape[1], device=fusion_dense.device)
        
        fusion_adj = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items))
        fusion_adj = self.normalize_sparse_adj(fusion_adj)
        
        return fusion_adj
    
    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix for user-item graph"""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        # Normalize
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes))).coalesce()
    
    def get_R_mat(self):
        """Get normalized user-item interaction matrix for aggregating user features"""
        R = self.interaction_matrix.tocsr()
        
        # Normalize: 1/sqrt(|N_u||N_i|)
        user_degrees = np.array(R.sum(axis=1)).flatten() + 1e-7
        item_degrees = np.array(R.sum(axis=0)).flatten() + 1e-7
        
        user_d_inv_sqrt = np.power(user_degrees, -0.5)
        item_d_inv_sqrt = np.power(item_degrees, -0.5)
        
        R_coo = R.tocoo()
        row = R_coo.row
        col = R_coo.col
        data = user_d_inv_sqrt[row] * item_d_inv_sqrt[col]
        
        indices = torch.LongTensor(np.array([row, col]))
        values = torch.FloatTensor(data)
        
        return torch.sparse_coo_tensor(indices, values, (self.n_users, self.n_items)).coalesce()
    
    def spectrum_modality_fusion(self):
        """
        Spectrum Modality Fusion: FFT -> Filter -> Fusion -> IFFT
        """
        projected_feats = {}
        
        # Project modality features to shared space
        for m in self.modalities:
            if m == 'image':
                raw_feat = self.image_embedding.weight
                proj_feat = self.image_proj(raw_feat)
            else:
                raw_feat = self.text_embedding.weight
                proj_feat = self.text_proj(raw_feat)
            projected_feats[m] = proj_feat
        
        # FFT transformation and filtering
        spectrum_feats = {}
        filtered_feats = {}
        
        for m in self.modalities:
            # Apply FFT along item dimension
            feat = projected_feats[m]
            spectrum = torch.fft.fft(feat, dim=0)
            spectrum_feats[m] = spectrum
            
            # Apply modality-specific filter (complex multiplication)
            filter_real = self.spectrum_filters[m]
            filter_imag = self.spectrum_filters_imag[m]
            filter_complex = torch.complex(filter_real, filter_imag)
            
            filtered = spectrum * filter_complex
            filtered_feats[m] = filtered
        
        # Fusion in frequency domain via point-wise product
        if len(self.modalities) > 1:
            fusion_spectrum = spectrum_feats[self.modalities[0]]
            for m in self.modalities[1:]:
                fusion_spectrum = fusion_spectrum * spectrum_feats[m]
            
            # Apply fusion filter
            fusion_filter = torch.complex(self.fusion_filter_real, self.fusion_filter_imag)
            fusion_spectrum = fusion_spectrum * fusion_filter
        else:
            fusion_spectrum = filtered_feats[self.modalities[0]]
        
        # IFFT to get back to spatial domain
        denoised_uni_feats = {}
        for m in self.modalities:
            denoised = torch.fft.ifft(filtered_feats[m], dim=0).real
            denoised_uni_feats[m] = denoised
        
        denoised_fusion_feat = torch.fft.ifft(fusion_spectrum, dim=0).real
        
        return denoised_uni_feats, denoised_fusion_feat
    
    def forward(self, train=False):
        # Spectrum Modality Fusion
        denoised_uni_feats, denoised_fusion_feat = self.spectrum_modality_fusion()
        
        # Behavior-guided gating
        item_id_emb = self.item_id_embedding.weight
        
        gated_uni_feats = {}
        for m in self.modalities:
            gate = self.modal_gates[m](denoised_uni_feats[m])
            gated_uni_feats[m] = item_id_emb * gate
        
        gated_fusion_feat = item_id_emb * self.fusion_gate(denoised_fusion_feat)
        
        # Item-Item Graph Propagation (shallow, single layer)
        propagated_uni_feats = {}
        for m in self.modalities:
            adj = self.modal_adjs[m]
            prop_feat = torch.sparse.mm(adj, gated_uni_feats[m])
            propagated_uni_feats[m] = prop_feat
        
        propagated_fusion_feat = torch.sparse.mm(self.fusion_adj, gated_fusion_feat)
        
        # Aggregate user modality features
        user_uni_feats = {}
        for m in self.modalities:
            user_feat = torch.sparse.mm(self.R, propagated_uni_feats[m])
            user_uni_feats[m] = user_feat
        
        user_fusion_feat = torch.sparse.mm(self.R, propagated_fusion_feat)
        
        # Concatenate user and item features
        enriched_uni_feats = {}
        for m in self.modalities:
            enriched_uni_feats[m] = torch.cat([user_uni_feats[m], propagated_uni_feats[m]], dim=0)
        
        enriched_fusion_feat = torch.cat([user_fusion_feat, propagated_fusion_feat], dim=0)
        
        # User-Item Behavioral View (LightGCN-style)
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        behavioral_embeddings = all_embeddings.mean(dim=1)
        
        # Modality-Aware Preference Module
        # Compute attention weights using fusion features
        attention_weights = {}
        for m in self.modalities:
            att = self.modal_attention[m](enriched_fusion_feat)
            attention_weights[m] = att
        
        # Stack and softmax
        att_stack = torch.cat([attention_weights[m] for m in self.modalities], dim=-1)
        att_softmax = F.softmax(att_stack, dim=-1)
        
        # Weighted aggregation of uni-modal features
        aggregated_uni_feat = torch.zeros_like(enriched_fusion_feat)
        for i, m in enumerate(self.modalities):
            aggregated_uni_feat = aggregated_uni_feat + att_softmax[:, i:i+1] * enriched_uni_feats[m]
        
        # Extract preferences from behavioral embeddings
        uni_preferences = {}
        for m in self.modalities:
            uni_preferences[m] = self.pref_gates[m](behavioral_embeddings)
        
        fusion_preference = self.pref_gate_fusion(behavioral_embeddings)
        
        # Compute final side features
        side_feat = torch.zeros_like(behavioral_embeddings)
        for m in self.modalities:
            side_feat = side_feat + aggregated_uni_feat * uni_preferences[m]
        side_feat = side_feat / len(self.modalities)
        side_feat = side_feat + enriched_fusion_feat * fusion_preference
        
        # Final representations
        final_embeddings = behavioral_embeddings + side_feat
        
        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items], dim=0)
        
        if train:
            behavioral_user, behavioral_item = torch.split(behavioral_embeddings, [self.n_users, self.n_items], dim=0)
            side_user, side_item = torch.split(side_feat, [self.n_users, self.n_items], dim=0)
            return user_embeddings, item_embeddings, behavioral_user, behavioral_item, side_user, side_item
        
        return user_embeddings, item_embeddings
    
    def bpr_loss(self, users, pos_items, neg_items):
        """BPR loss"""
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss
    
    def info_nce_loss(self, view1, view2, temperature):
        """InfoNCE contrastive loss"""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        pos_score = torch.sum(view1 * view2, dim=1)
        pos_score = torch.exp(pos_score / temperature)
        
        all_score = torch.mm(view1, view2.t())
        all_score = torch.exp(all_score / temperature).sum(dim=1)
        
        loss = -torch.log(pos_score / all_score + 1e-8)
        return loss.mean()
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, beh_user, beh_item, side_user, side_item = self.forward(train=True)
        
        u_emb = user_emb[users]
        pos_i_emb = item_emb[pos_items]
        neg_i_emb = item_emb[neg_items]
        
        # BPR loss
        bpr_loss = self.bpr_loss(u_emb, pos_i_emb, neg_i_emb)
        
        # Contrastive loss
        cl_loss_user = self.info_nce_loss(beh_user[users], side_user[users], self.temperature)
        cl_loss_item = self.info_nce_loss(beh_item[pos_items], side_item[pos_items], self.temperature)
        cl_loss = cl_loss_user + cl_loss_item
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            self.user_embedding.weight.norm(2).pow(2) +
            self.item_id_embedding.weight.norm(2).pow(2)
        ) / self.n_users
        
        total_loss = bpr_loss + self.cl_weight * cl_loss + reg_loss
        
        return total_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_emb, item_emb = self.forward(train=False)
        u_emb = user_emb[user]
        
        scores = torch.matmul(u_emb, item_emb.t())
        return scores
    
    def pre_epoch_processing(self):
        """Called before each epoch - can be used for preprocessing"""
        pass