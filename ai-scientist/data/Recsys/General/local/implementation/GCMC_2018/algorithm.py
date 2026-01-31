# -*- coding: utf-8 -*-
# @Time   : 2025-01-15
# @Author : GCMC Implementation
# @Email  : gcmc@recbole.com

r"""
GCMC
################################################
Reference:
    Rianne van den Berg et al. "Graph Convolutional Matrix Completion" in KDD 2018.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class GCMC(GeneralRecommender):
    r"""GCMC is a graph convolutional matrix completion model for recommendation.
    
    It treats matrix completion as link prediction on a bipartite user-item graph,
    using graph convolutional encoder and bilinear decoder.
    """
    
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(GCMC, self).__init__(config, dataset)
        
        # Load configuration parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']  # Number of graph conv layers
        self.dropout_prob = config['dropout_prob']
        self.node_dropout_prob = config['node_dropout_prob']
        self.accum = config['accum']  # 'stack' or 'sum'
        self.normalization = config['normalization']  # 'left' or 'symmetric'
        self.num_basis = config['num_basis']  # Number of basis matrices for decoder
        self.reg_weight = config['reg_weight']
        
        # Rating levels (assume ratings are 1-5)
        self.rating_values = dataset.inter_feat[self.RATING].unique().sort()[0].tolist()
        self.n_ratings = len(self.rating_values)
        
        # Build rating-specific adjacency matrices
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._build_rating_matrices(dataset)
        
        # Initialize encoder parameters
        # For each rating type, we have a weight matrix
        if self.accum == 'stack':
            # Stacking increases dimension
            conv_out_dim = self.embedding_size * self.n_ratings
        else:  # sum
            conv_out_dim = self.embedding_size
        
        # Ordinal weight sharing: W_r = sum_{s=1}^r T_s
        self.T_matrices = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(self.embedding_size, self.embedding_size))
            for _ in range(self.n_ratings)
        ])
        
        # Dense layer after graph convolution
        self.dense_weight = nn.Parameter(torch.FloatTensor(conv_out_dim, self.embedding_size))
        self.dense_bias = nn.Parameter(torch.FloatTensor(self.embedding_size))
        
        # Bilinear decoder with basis matrices
        self.basis_matrices = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(self.embedding_size, self.embedding_size))
            for _ in range(self.num_basis)
        ])
        
        # Coefficients for linear combination of basis matrices
        self.decoder_weights = nn.Parameter(
            torch.FloatTensor(self.n_ratings, self.num_basis)
        )
        
        # User and item embeddings (one-hot encoded initially)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # Apply initialization
        self.apply(xavier_normal_initialization)
        
        # Cache for evaluation
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
    
    def _build_rating_matrices(self, dataset):
        """Build rating-specific adjacency matrices M_r for each rating level."""
        inter_feat = dataset.inter_feat
        user_np = inter_feat[self.USER_ID].numpy()
        item_np = inter_feat[self.ITEM_ID].numpy()
        rating_np = inter_feat[self.RATING].numpy()
        
        # Create adjacency matrices for each rating
        self.rating_matrices = []
        for r_val in self.rating_values:
            # Find interactions with rating = r_val
            mask = (rating_np == r_val)
            users_r = user_np[mask]
            items_r = item_np[mask]
            
            # Build bipartite adjacency matrix
            # Shape: (n_users + n_items, n_users + n_items)
            # Structure: [[0, M_r], [M_r^T, 0]]
            data = np.ones(len(users_r))
            
            # User-to-item part (top-right block)
            row_ui = users_r
            col_ui = items_r + self.n_users
            
            # Item-to-user part (bottom-left block)
            row_iu = items_r + self.n_users
            col_iu = users_r
            
            # Combine
            row = np.concatenate([row_ui, row_iu])
            col = np.concatenate([col_ui, col_iu])
            data_full = np.concatenate([data, data])
            
            # Create sparse matrix
            adj = sp.coo_matrix(
                (data_full, (row, col)),
                shape=(self.n_users + self.n_items, self.n_users + self.n_items)
            )
            
            # Normalize
            adj = self._normalize_adjacency(adj)
            
            # Convert to PyTorch sparse tensor
            adj = self._convert_sp_mat_to_sp_tensor(adj).to(self.device)
            self.rating_matrices.append(adj)
    
    def _normalize_adjacency(self, adj):
        """Normalize adjacency matrix with left or symmetric normalization."""
        adj = adj.tocoo()
        
        if self.normalization == 'left':
            # Left normalization: D^{-1} * A
            rowsum = np.array(adj.sum(1)).flatten()
            rowsum[rowsum == 0] = 1e-10  # Avoid division by zero
            d_inv = sp.diags(1.0 / rowsum)
            norm_adj = d_inv.dot(adj)
        else:  # symmetric
            # Symmetric normalization: D^{-0.5} * A * D^{-0.5}
            rowsum = np.array(adj.sum(1)).flatten()
            rowsum[rowsum == 0] = 1e-10
            d_inv_sqrt = sp.diags(1.0 / np.sqrt(rowsum))
            norm_adj = d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
        
        return norm_adj.tocoo()
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = X.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def _get_ordinal_weights(self):
        """Compute ordinal weight sharing: W_r = sum_{s=1}^r T_s."""
        W_list = []
        for r in range(self.n_ratings):
            W_r = sum(self.T_matrices[s] for s in range(r + 1))
            W_list.append(W_r)
        return W_list
    
    def _graph_convolution(self, embeddings, dropout=True):
        """Perform graph convolution with message passing.
        
        Args:
            embeddings: Initial node embeddings [n_users + n_items, embedding_size]
            dropout: Whether to apply node dropout
        
        Returns:
            Convolved embeddings [n_users + n_items, conv_out_dim]
        """
        W_list = self._get_ordinal_weights()
        
        # Message passing for each rating type
        messages = []
        for r, adj_r in enumerate(self.rating_matrices):
            # Apply node dropout
            if dropout and self.training and self.node_dropout_prob > 0:
                # Randomly drop nodes
                n_nodes = embeddings.shape[0]
                keep_prob = 1 - self.node_dropout_prob
                keep_mask = torch.rand(n_nodes, device=self.device) < keep_prob
                keep_mask = keep_mask.float().unsqueeze(1)
                
                # Drop messages from dropped nodes
                dropped_embeddings = embeddings * keep_mask
                # Rescale
                dropped_embeddings = dropped_embeddings / keep_prob
            else:
                dropped_embeddings = embeddings
            
            # Transform: W_r * x_j
            transformed = torch.matmul(dropped_embeddings, W_list[r].t())
            
            # Message passing: adj_r * transformed
            message = torch.sparse.mm(adj_r, transformed)
            messages.append(message)
        
        # Accumulate messages
        if self.accum == 'stack':
            # Concatenate along feature dimension
            h = torch.cat(messages, dim=1)
        else:  # sum
            h = sum(messages)
        
        # Apply activation
        h = F.relu(h)
        
        return h
    
    def _dense_layer(self, h):
        """Apply dense transformation."""
        # h: [n_users + n_items, conv_out_dim]
        # Output: [n_users + n_items, embedding_size]
        out = torch.matmul(h, self.dense_weight) + self.dense_bias
        out = F.relu(out)
        
        # Apply standard dropout
        if self.training:
            out = F.dropout(out, p=self.dropout_prob)
        
        return out
    
    def forward(self):
        """Graph encoder: generate user and item embeddings."""
        # Initial embeddings (one-hot encoded as identity)
        # Use learned embeddings instead of one-hot for efficiency
        user_emb_init = self.user_embedding.weight
        item_emb_init = self.item_embedding.weight
        
        # Combine into single embedding matrix
        all_embeddings = torch.cat([user_emb_init, item_emb_init], dim=0)
        
        # Graph convolution
        h = self._graph_convolution(all_embeddings, dropout=True)
        
        # Dense layer
        embeddings = self._dense_layer(h)
        
        # Split into user and item embeddings
        user_embeddings, item_embeddings = torch.split(
            embeddings, [self.n_users, self.n_items], dim=0
        )
        
        return user_embeddings, item_embeddings
    
    def _bilinear_decoder(self, user_emb, item_emb):
        """Bilinear decoder with basis matrices.
        
        Args:
            user_emb: [batch_size, embedding_size]
            item_emb: [batch_size, embedding_size]
        
        Returns:
            Rating probabilities: [batch_size, n_ratings]
        """
        batch_size = user_emb.shape[0]
        
        # Compute Q_r = sum_s a_rs * P_s for each rating
        logits = []
        for r in range(self.n_ratings):
            # Linear combination of basis matrices
            Q_r = sum(
                self.decoder_weights[r, s] * self.basis_matrices[s]
                for s in range(self.num_basis)
            )
            
            # Bilinear: u_i^T * Q_r * v_j
            # [batch_size, embedding_size] * [embedding_size, embedding_size] * [batch_size, embedding_size]
            logit = (user_emb * torch.matmul(item_emb, Q_r.t())).sum(dim=1)
            logits.append(logit)
        
        # Stack logits: [batch_size, n_ratings]
        logits = torch.stack(logits, dim=1)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        return probs
    
    def calculate_loss(self, interaction):
        """Calculate loss for training."""
        # Clear cache
        self.restore_user_e = None
        self.restore_item_e = None
        
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.RATING]
        
        # Get embeddings
        user_embeddings, item_embeddings = self.forward()
        
        # Select embeddings for current batch
        u_emb = user_embeddings[user]
        i_emb = item_embeddings[item]
        
        # Get rating probabilities
        probs = self._bilinear_decoder(u_emb, i_emb)
        
        # Compute negative log likelihood
        # Convert ratings to class indices (0-based)
        rating_indices = torch.zeros_like(rating, dtype=torch.long)
        for idx, r_val in enumerate(self.rating_values):
            rating_indices[rating == r_val] = idx
        
        # Cross entropy loss
        loss = F.cross_entropy(probs, rating_indices)
        
        # Add L2 regularization
        reg_loss = 0
        for param in self.T_matrices:
            reg_loss += param.norm(2)
        for param in self.basis_matrices:
            reg_loss += param.norm(2)
        reg_loss += self.decoder_weights.norm(2)
        reg_loss += self.dense_weight.norm(2)
        
        loss += self.reg_weight * reg_loss
        
        return loss
    
    def predict(self, interaction):
        """Predict ratings for user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # Get embeddings
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_emb = self.restore_user_e[user]
        i_emb = self.restore_item_e[item]
        
        # Get rating probabilities
        probs = self._bilinear_decoder(u_emb, i_emb)
        
        # Expected rating: E[r] = sum_r r * p(r)
        rating_values_tensor = torch.FloatTensor(self.rating_values).to(self.device)
        predicted_ratings = (probs * rating_values_tensor).sum(dim=1)
        
        return predicted_ratings
    
    def full_sort_predict(self, interaction):
        """Predict ratings for all items for given users."""
        user = interaction[self.USER_ID]
        
        # Get embeddings
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_emb = self.restore_user_e[user]
        
        # Compute scores for all items
        # For efficiency, we compute logits for all items at once
        batch_size = u_emb.shape[0]
        all_scores = []
        
        for i in range(batch_size):
            user_emb_i = u_emb[i:i+1].expand(self.n_items, -1)
            item_emb_all = self.restore_item_e
            
            # Get rating probabilities
            probs = self._bilinear_decoder(user_emb_i, item_emb_all)
            
            # Expected rating
            rating_values_tensor = torch.FloatTensor(self.rating_values).to(self.device)
            predicted_ratings = (probs * rating_values_tensor).sum(dim=1)
            
            all_scores.append(predicted_ratings)
        
        # Concatenate all scores
        scores = torch.cat(all_scores, dim=0)
        
        return scores.view(-1)