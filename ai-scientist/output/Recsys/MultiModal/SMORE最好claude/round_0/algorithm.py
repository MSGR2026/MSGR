# coding: utf-8
r"""
SMORE: Spectrum Modality Fusion for Multimodal Recommendation
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
        
        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.temperature = config['temperature']
        self.dropout = config['dropout']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.R = self.get_R_mat().to(self.device)
        
        # User and Item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modality feature projections
        self.modalities = []
        if self.v_feat is not None:
            self.modalities.append('image')
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            # Complex filter for image modality (real and imaginary parts)
            self.image_filter_real = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
            self.image_filter_imag = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
            
        if self.t_feat is not None:
            self.modalities.append('text')
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            # Complex filter for text modality
            self.text_filter_real = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
            self.text_filter_imag = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        
        # Fusion filter (complex)
        self.fusion_filter_real = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        self.fusion_filter_imag = nn.Parameter(torch.randn(self.n_items, self.embedding_dim) * 0.01)
        
        # Behavior-guided gating for uni-modal features
        if self.v_feat is not None:
            self.gate_image = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        if self.t_feat is not None:
            self.gate_text = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        
        # Behavior-guided gating for fusion features
        self.gate_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Modality-aware attention using fusion features
        if self.v_feat is not None:
            self.attention_image = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, 1, bias=False)
            )
        if self.t_feat is not None:
            self.attention_text = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, 1, bias=False)
            )
        
        # Preference gates from behavioral signals
        if self.v_feat is not None:
            self.pref_gate_image = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        if self.t_feat is not None:
            self.pref_gate_text = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        self.pref_gate_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Build item-item graphs
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.modal_adjs = {}
        
        if self.v_feat is not None:
            image_adj_file = os.path.join(dataset_path, f'smore_image_adj_{self.knn_k}.pt')
            if os.path.exists(image_adj_file):
                self.modal_adjs['image'] = torch.load(image_adj_file).to(self.device)
            else:
                image_adj = self.build_knn_graph(self.image_embedding.weight.detach())
                torch.save(image_adj, image_adj_file)
                self.modal_adjs['image'] = image_adj.to(self.device)
                
        if self.t_feat is not None:
            text_adj_file = os.path.join(dataset_path, f'smore_text_adj_{self.knn_k}.pt')
            if os.path.exists(text_adj_file):
                self.modal_adjs['text'] = torch.load(text_adj_file).to(self.device)
            else:
                text_adj = self.build_knn_graph(self.text_embedding.weight.detach())
                torch.save(text_adj, text_adj_file)
                self.modal_adjs['text'] = text_adj.to(self.device)
        
        # Build fusion graph via max-pooling
        fusion_adj_file = os.path.join(dataset_path, f'smore_fusion_adj_{self.knn_k}.pt')
        if os.path.exists(fusion_adj_file):
            self.fusion_adj = torch.load(fusion_adj_file).to(self.device)
        else:
            self.fusion_adj = self.build_fusion_graph().to(self.device)
            torch.save(self.fusion_adj.cpu(), fusion_adj_file)
    
    def build_knn_graph(self, features):
        """Build kNN graph from features with symmetric normalization."""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Compute similarity
        sim = torch.mm(features_norm, features_norm.t())
        
        # Get top-K neighbors (excluding self)
        sim.fill_diagonal_(float('-inf'))
        _, knn_indices = torch.topk(sim, self.knn_k, dim=-1)
        
        # Build sparse adjacency matrix
        row_indices = torch.arange(self.n_items, device=features.device).unsqueeze(1).expand(-1, self.knn_k).flatten()
        col_indices = knn_indices.flatten()
        
        # Get similarity values for edges
        edge_values = sim[row_indices, col_indices]
        
        # Create sparse matrix
        indices = torch.stack([row_indices, col_indices], dim=0)
        adj = torch.sparse_coo_tensor(indices, edge_values, (self.n_items, self.n_items))
        
        # Make symmetric
        adj = adj + adj.t()
        adj = adj.coalesce()
        
        # Symmetric normalization
        adj = self.normalize_sparse_adj(adj)
        
        return adj.cpu()
    
    def normalize_sparse_adj(self, adj):
        """Apply symmetric normalization to sparse adjacency matrix."""
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        
        # Compute degree
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree = torch.clamp(degree, min=1e-7)
        d_inv_sqrt = torch.pow(degree, -0.5)
        
        # Normalize values
        row_norm = d_inv_sqrt[indices[0]]
        col_norm = d_inv_sqrt[indices[1]]
        normalized_values = values * row_norm * col_norm
        
        return torch.sparse_coo_tensor(indices, normalized_values, adj.shape).coalesce()
    
    def build_fusion_graph(self):
        """Build fusion graph using max-pooling across modality graphs."""
        if len(self.modal_adjs) == 0:
            return None
        
        if len(self.modal_adjs) == 1:
            key = list(self.modal_adjs.keys())[0]
            return self.modal_adjs[key].clone()
        
        # Convert sparse tensors to dense for max operation, then back to sparse
        # For efficiency, we work with the union of edges
        all_indices = []
        all_values = []
        
        for name, adj in self.modal_adjs.items():
            adj = adj.coalesce()
            all_indices.append(adj.indices())
            all_values.append(adj.values())
        
        # Concatenate all edges
        combined_indices = torch.cat(all_indices, dim=1)
        combined_values = torch.cat(all_values, dim=0)
        
        # Create combined sparse tensor and take max for duplicate edges
        fusion_adj = torch.sparse_coo_tensor(
            combined_indices, combined_values, (self.n_items, self.n_items)
        )
        
        # Coalesce will sum duplicates, but we want max
        # Convert to dense, apply max logic, then back to sparse
        fusion_dense = fusion_adj.to_dense()
        
        # For each modality, update with max
        for name, adj in self.modal_adjs.items():
            adj_dense = adj.to_dense()
            fusion_dense = torch.max(fusion_dense, adj_dense)
        
        # Convert back to sparse
        fusion_adj = fusion_dense.to_sparse().coalesce()
        
        # Normalize
        fusion_adj = self.normalize_sparse_adj(fusion_adj)
        
        return fusion_adj
    
    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix for user-item bipartite graph."""
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
        
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes))).coalesce()
    
    def get_R_mat(self):
        """Get normalized user-item interaction matrix for aggregating user features."""
        R = self.interaction_matrix.tocsr()
        
        # Compute normalization: 1 / sqrt(|N_u| * |N_i|)
        user_degree = np.array(R.sum(axis=1)).flatten() + 1e-7
        item_degree = np.array(R.sum(axis=0)).flatten() + 1e-7
        
        user_d_inv_sqrt = np.power(user_degree, -0.5)
        item_d_inv_sqrt = np.power(item_degree, -0.5)
        
        R_coo = R.tocoo()
        norm_values = user_d_inv_sqrt[R_coo.row] * item_d_inv_sqrt[R_coo.col]
        
        indices = torch.LongTensor(np.array([R_coo.row, R_coo.col]))
        values = torch.FloatTensor(norm_values)
        
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_users, self.n_items))).coalesce()
    
    def spectrum_modality_fusion(self):
        """
        Perform spectrum modality fusion using FFT.
        Returns denoised uni-modal features and fused features.
        """
        modal_features = {}
        modal_spectrums = {}
        
        # Project features and apply FFT
        if self.v_feat is not None:
            image_feat = self.image_proj(self.image_embedding.weight)  # [n_items, d]
            image_spectrum = torch.fft.fft(image_feat, dim=0)  # Complex tensor
            
            # Apply modality-specific filter (complex multiplication)
            filter_complex = torch.complex(self.image_filter_real, self.image_filter_imag)
            image_spectrum_filtered = image_spectrum * filter_complex
            
            modal_features['image'] = image_feat
            modal_spectrums['image'] = image_spectrum_filtered
            
        if self.t_feat is not None:
            text_feat = self.text_proj(self.text_embedding.weight)  # [n_items, d]
            text_spectrum = torch.fft.fft(text_feat, dim=0)
            
            filter_complex = torch.complex(self.text_filter_real, self.text_filter_imag)
            text_spectrum_filtered = text_spectrum * filter_complex
            
            modal_features['text'] = text_feat
            modal_spectrums['text'] = text_spectrum_filtered
        
        # Fusion in frequency domain via point-wise product
        if len(modal_spectrums) > 1:
            fusion_spectrum = None
            for name, spectrum in modal_spectrums.items():
                if fusion_spectrum is None:
                    fusion_spectrum = spectrum
                else:
                    fusion_spectrum = fusion_spectrum * spectrum
            
            # Apply fusion filter
            fusion_filter_complex = torch.complex(self.fusion_filter_real, self.fusion_filter_imag)
            fusion_spectrum_filtered = fusion_spectrum * fusion_filter_complex
        elif len(modal_spectrums) == 1:
            fusion_spectrum_filtered = list(modal_spectrums.values())[0]
        else:
            fusion_spectrum_filtered = None
        
        # IFFT to convert back to spatial domain
        denoised_modal_features = {}
        for name, spectrum in modal_spectrums.items():
            denoised_feat = torch.fft.ifft(spectrum, dim=0).real  # Take real part
            denoised_modal_features[name] = denoised_feat
        
        if fusion_spectrum_filtered is not None:
            fusion_features = torch.fft.ifft(fusion_spectrum_filtered, dim=0).real
        else:
            fusion_features = None
        
        return denoised_modal_features, fusion_features
    
    def behavior_guided_gating(self, modal_features, fusion_features):
        """Apply behavior-guided gating to filter preference-relevant features."""
        item_id_emb = self.item_id_embedding.weight
        
        gated_modal_features = {}
        
        if 'image' in modal_features:
            gate = self.gate_image(modal_features['image'])
            gated_modal_features['image'] = item_id_emb * gate
            
        if 'text' in modal_features:
            gate = self.gate_text(modal_features['text'])
            gated_modal_features['text'] = item_id_emb * gate
        
        if fusion_features is not None:
            gate = self.gate_fusion(fusion_features)
            gated_fusion_features = item_id_emb * gate
        else:
            gated_fusion_features = None
        
        return gated_modal_features, gated_fusion_features
    
    def item_item_propagation(self, gated_modal_features, gated_fusion_features):
        """Propagate features through item-item graphs."""
        propagated_modal_features = {}
        
        for name, features in gated_modal_features.items():
            if name in self.modal_adjs:
                adj = self.modal_adjs[name]
                propagated = torch.sparse.mm(adj, features)
                propagated_modal_features[name] = propagated
            else:
                propagated_modal_features[name] = features
        
        if gated_fusion_features is not None and self.fusion_adj is not None:
            propagated_fusion_features = torch.sparse.mm(self.fusion_adj, gated_fusion_features)
        else:
            propagated_fusion_features = gated_fusion_features
        
        return propagated_modal_features, propagated_fusion_features
    
    def aggregate_user_modal_features(self, item_modal_features, item_fusion_features):
        """Aggregate user modality features from interacted items."""
        user_modal_features = {}
        
        for name, features in item_modal_features.items():
            user_feat = torch.sparse.mm(self.R, features)
            user_modal_features[name] = user_feat
        
        if item_fusion_features is not None:
            user_fusion_features = torch.sparse.mm(self.R, item_fusion_features)
        else:
            user_fusion_features = None
        
        return user_modal_features, user_fusion_features
    
    def user_item_propagation(self):
        """Propagate on user-item bipartite graph (LightGCN style)."""
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for layer in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return user_emb, item_emb
    
    def modality_aware_preference(self, user_modal_features, item_modal_features,
                                   user_fusion_features, item_fusion_features,
                                   user_behavioral, item_behavioral):
        """
        Compute modality-aware preferences and aggregate features.
        """
        # Concatenate user and item features for each modality
        modal_features_all = {}
        for name in item_modal_features.keys():
            user_feat = user_modal_features[name]
            item_feat = item_modal_features[name]
            modal_features_all[name] = torch.cat([user_feat, item_feat], dim=0)
        
        if user_fusion_features is not None and item_fusion_features is not None:
            fusion_features_all = torch.cat([user_fusion_features, item_fusion_features], dim=0)
        else:
            fusion_features_all = None
        
        behavioral_all = torch.cat([user_behavioral, item_behavioral], dim=0)
        
        # Compute attention weights using fusion features
        if fusion_features_all is not None:
            attention_scores = {}
            if 'image' in modal_features_all:
                attention_scores['image'] = self.attention_image(fusion_features_all)
            if 'text' in modal_features_all:
                attention_scores['text'] = self.attention_text(fusion_features_all)
            
            # Softmax over modalities
            if len(attention_scores) > 1:
                scores_cat = torch.cat(list(attention_scores.values()), dim=-1)
                attention_weights = F.softmax(scores_cat, dim=-1)
                
                # Weight uni-modal features
                weighted_modal = None
                for i, name in enumerate(attention_scores.keys()):
                    weight = attention_weights[:, i:i+1]
                    if weighted_modal is None:
                        weighted_modal = weight * modal_features_all[name]
                    else:
                        weighted_modal = weighted_modal + weight * modal_features_all[name]
            else:
                weighted_modal = list(modal_features_all.values())[0]
        else:
            # Simple average if no fusion features
            weighted_modal = None
            for name, feat in modal_features_all.items():
                if weighted_modal is None:
                    weighted_modal = feat
                else:
                    weighted_modal = weighted_modal + feat
            if len(modal_features_all) > 0:
                weighted_modal = weighted_modal / len(modal_features_all)
        
        # Extract preferences from behavioral signals
        pref_modal = {}
        if 'image' in modal_features_all:
            pref_modal['image'] = self.pref_gate_image(behavioral_all)
        if 'text' in modal_features_all:
            pref_modal['text'] = self.pref_gate_text(behavioral_all)
        
        pref_fusion = self.pref_gate_fusion(behavioral_all)
        
        # Compute final side features
        # H_s = (1/|M|) * sum(H*_m ⊙ Q_m) + (H_f ⊙ Q_f)
        side_features = None
        num_modalities = len(modal_features_all)
        
        if weighted_modal is not None and num_modalities > 0:
            # Average preference-weighted modal features
            modal_side = None
            for name in modal_features_all.keys():
                pref = pref_modal.get(name, torch.ones_like(behavioral_all))
                weighted = modal_features_all[name] * pref
                if modal_side is None:
                    modal_side = weighted
                else:
                    modal_side = modal_side + weighted
            modal_side = modal_side / num_modalities
            side_features = modal_side
        
        if fusion_features_all is not None:
            fusion_side = fusion_features_all * pref_fusion
            if side_features is None:
                side_features = fusion_side
            else:
                side_features = side_features + fusion_side
        
        return side_features
    
    def forward(self, train=False):
        # 1. Spectrum Modality Fusion
        denoised_modal_features, fusion_features = self.spectrum_modality_fusion()
        
        # 2. Behavior-guided Gating
        gated_modal_features, gated_fusion_features = self.behavior_guided_gating(
            denoised_modal_features, fusion_features
        )
        
        # 3. Item-Item Graph Propagation
        prop_modal_features, prop_fusion_features = self.item_item_propagation(
            gated_modal_features, gated_fusion_features
        )
        
        # 4. Aggregate User Modal Features
        user_modal_features, user_fusion_features = self.aggregate_user_modal_features(
            prop_modal_features, prop_fusion_features
        )
        
        # 5. User-Item Behavioral Propagation
        user_behavioral, item_behavioral = self.user_item_propagation()
        
        # 6. Modality-Aware Preference Module
        side_features = self.modality_aware_preference(
            user_modal_features, prop_modal_features,
            user_fusion_features, prop_fusion_features,
            user_behavioral, item_behavioral
        )
        
        # 7. Final representations
        behavioral_all = torch.cat([user_behavioral, item_behavioral], dim=0)
        
        if side_features is not None:
            final_embeddings = behavioral_all + side_features
        else:
            final_embeddings = behavioral_all
        
        user_emb, item_emb = torch.split(final_embeddings, [self.n_users, self.n_items], dim=0)
        
        if train:
            return user_emb, item_emb, side_features, behavioral_all
        
        return user_emb, item_emb
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, side_features, behavioral_all = self.forward(train=True)
        
        u_embeddings = user_emb[users]
        pos_i_embeddings = item_emb[pos_items]
        neg_i_embeddings = item_emb[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_i_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_i_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss (InfoNCE)
        cl_loss = 0.0
        if side_features is not None:
            user_side, item_side = torch.split(side_features, [self.n_users, self.n_items], dim=0)
            user_behavioral, item_behavioral = torch.split(behavioral_all, [self.n_users, self.n_items], dim=0)
            
            # User contrastive loss
            user_cl_loss = self.info_nce_loss(
                user_behavioral[users], user_side[users], self.temperature
            )
            
            # Item contrastive loss
            item_cl_loss = self.info_nce_loss(
                item_behavioral[pos_items], item_side[pos_items], self.temperature
            )
            
            cl_loss = user_cl_loss + item_cl_loss
        
        # L2 Regularization
        reg_loss = self.reg_weight * (
            torch.norm(self.user_embedding.weight[users]) ** 2 +
            torch.norm(self.item_id_embedding.weight[pos_items]) ** 2 +
            torch.norm(self.item_id_embedding.weight[neg_items]) ** 2
        ) / len(users)
        
        total_loss = bpr_loss + self.cl_weight * cl_loss + reg_loss
        
        return total_loss
    
    def info_nce_loss(self, view1, view2, temperature):
        """Compute InfoNCE contrastive loss."""
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        
        pos_score = torch.sum(view1 * view2, dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        
        ttl_score = torch.matmul(view1, view2.t())
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-8))
        return cl_loss.mean()
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_emb, item_emb = self.forward(train=False)
        u_embeddings = user_emb[user]
        
        scores = torch.matmul(u_embeddings, item_emb.t())
        return scores
    
    def pre_epoch_processing(self):
        """Called before each training epoch."""
        pass