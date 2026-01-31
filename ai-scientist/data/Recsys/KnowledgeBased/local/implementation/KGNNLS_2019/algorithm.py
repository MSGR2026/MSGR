# -*- coding: utf-8 -*-
# @Time   : 2024/12/19
# @Author : KGNNLS Implementation
# @Paper  : Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class KGNNLS(KnowledgeRecommender):
    r"""KGNNLS is a knowledge-aware graph neural network model with label smoothness regularization.
    
    It transforms the heterogeneous KG into user-personalized weighted graphs and uses GNN layers
    to aggregate neighbor representations. Label smoothness regularization is applied to help
    learn better edge weights through leave-one-out label propagation.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(KGNNLS, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.ls_weight = config["ls_weight"]
        
        self.LABEL = config["LABEL_FIELD"]

        # Build KG adjacency structure
        self._build_kg_structure(dataset)

        # Define embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # Define GNN transformation matrices for each layer
        self.gnn_layers = nn.ModuleList()
        input_dim = self.embedding_size
        for _ in range(self.n_layers):
            self.gnn_layers.append(nn.Linear(input_dim, self.embedding_size, bias=False))
            input_dim = self.embedding_size

        # Define loss functions
        self.rec_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = EmbLoss()

        # Apply initialization
        self.apply(xavier_normal_initialization)
        
        # Move precomputed structures to device
        self.other_parameter_name = ["kg_dict", "user_history"]

    def _build_kg_structure(self, dataset):
        r"""Build KG adjacency structure for efficient neighbor lookup."""
        # Get KG triples
        kg_data = dataset.kg_graph(form="coo", value_field="relation_id")
        head_entities = kg_data.row
        tail_entities = kg_data.col
        relations = kg_data.data

        # Build adjacency dictionary: entity -> [(neighbor, relation)]
        kg_dict = {}
        for h, t, r in zip(head_entities, tail_entities, relations):
            if h not in kg_dict:
                kg_dict[h] = []
            kg_dict[h].append((t, r))
        
        self.kg_dict = kg_dict

        # Build user interaction history
        inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        users = inter_matrix.row
        items = inter_matrix.col
        
        user_history = {}
        for u, i in zip(users, items):
            if u not in user_history:
                user_history[u] = []
            user_history[u].append(i)
        
        self.user_history = user_history

    def _compute_user_relation_scores(self, user_emb, relation_emb):
        r"""Compute user-specific relation scores: s_u(r) = g(u, r).
        
        Args:
            user_emb: [batch_size, dim]
            relation_emb: [n_relations, dim]
        
        Returns:
            scores: [batch_size, n_relations]
        """
        # Use inner product as scoring function
        scores = torch.matmul(user_emb, relation_emb.t())  # [batch_size, n_relations]
        return scores

    def _build_user_adjacency_matrix(self, user_ids):
        r"""Build user-personalized adjacency matrices.
        
        Args:
            user_ids: [batch_size]
        
        Returns:
            adj_matrices: list of sparse matrices, one per user
        """
        batch_size = len(user_ids)
        user_emb = self.user_embedding(user_ids)  # [batch_size, dim]
        relation_emb = self.relation_embedding.weight  # [n_relations, dim]
        
        # Compute relation scores for all users
        relation_scores = self._compute_user_relation_scores(user_emb, relation_emb)  # [batch_size, n_relations]
        relation_scores = torch.sigmoid(relation_scores)  # Normalize to [0, 1]
        
        adj_matrices = []
        for idx, user_id in enumerate(user_ids.cpu().numpy()):
            # Build adjacency matrix for this user
            row_indices = []
            col_indices = []
            values = []
            
            for entity in range(self.n_entities):
                if entity in self.kg_dict:
                    for neighbor, relation in self.kg_dict[entity]:
                        row_indices.append(entity)
                        col_indices.append(neighbor)
                        # Use user-specific relation score as edge weight
                        values.append(relation_scores[idx, relation].item())
            
            # Create sparse adjacency matrix
            if len(row_indices) > 0:
                adj = sp.coo_matrix(
                    (values, (row_indices, col_indices)),
                    shape=(self.n_entities, self.n_entities)
                )
            else:
                adj = sp.coo_matrix((self.n_entities, self.n_entities))
            
            # Add self-connections
            adj = adj + sp.eye(self.n_entities)
            
            adj_matrices.append(adj)
        
        return adj_matrices, relation_scores

    def _normalize_adjacency_matrix(self, adj):
        r"""Normalize adjacency matrix: D^{-1/2} A D^{-1/2}.
        
        Args:
            adj: scipy sparse matrix
        
        Returns:
            normalized_adj: scipy sparse matrix
        """
        # Compute degree matrix
        adj = adj.tocsr()
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        
        # D^{-1/2}
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        
        # D^{-1/2} A D^{-1/2}
        normalized_adj = d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat
        
        return normalized_adj

    def _gnn_forward(self, entity_emb, adj_matrix):
        r"""Forward pass through GNN layers.
        
        Args:
            entity_emb: [n_entities, dim]
            adj_matrix: normalized adjacency matrix (scipy sparse)
        
        Returns:
            final_emb: [n_entities, dim]
        """
        # Convert sparse matrix to torch sparse tensor
        adj_coo = adj_matrix.tocoo()
        indices = torch.LongTensor([adj_coo.row, adj_coo.col]).to(self.device)
        values = torch.FloatTensor(adj_coo.data).to(self.device)
        shape = adj_coo.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        
        h = entity_emb
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            # Aggregate neighbors: D^{-1/2} A D^{-1/2} H
            h_aggregated = torch.sparse.mm(adj_tensor, h)
            
            # Transform: W_l
            h = gnn_layer(h_aggregated)
            
            # Activation (ReLU for intermediate layers, no activation for last layer)
            if layer_idx < self.n_layers - 1:
                h = F.relu(h)
        
        return h

    def _label_propagation(self, adj_matrix, labels, item_indices):
        r"""Perform label propagation for label smoothness regularization.
        
        Args:
            adj_matrix: normalized adjacency matrix (scipy sparse)
            labels: [n_items] true labels for items
            item_indices: list of item indices
        
        Returns:
            predicted_labels: [n_items] predicted labels after propagation
        """
        # Initialize label vector for all entities
        n_entities = adj_matrix.shape[0]
        label_vec = np.zeros(n_entities)
        
        # Set labels for items
        for idx, item_id in enumerate(item_indices):
            label_vec[item_id] = labels[idx]
        
        # Compute transition matrix P = D^{-1} A
        adj_csr = adj_matrix.tocsr()
        degrees = np.array(adj_csr.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        d_inv = sp.diags(1.0 / degrees)
        P = d_inv @ adj_csr
        
        # Iterative label propagation (simplified version, few iterations)
        n_iterations = 3
        for _ in range(n_iterations):
            # Propagate: l(E) <- P l(E)
            label_vec = P @ label_vec
            
            # Reset item labels
            for idx, item_id in enumerate(item_indices):
                label_vec[item_id] = labels[idx]
        
        # Extract predicted labels for items
        predicted_labels = np.array([label_vec[item_id] for item_id in item_indices])
        
        return predicted_labels

    def forward(self, user, item):
        r"""Forward pass to compute prediction scores.
        
        Args:
            user: [batch_size]
            item: [batch_size]
        
        Returns:
            scores: [batch_size]
        """
        batch_size = len(user)
        
        # Get user embeddings
        user_emb = self.user_embedding(user)  # [batch_size, dim]
        
        # Build user-personalized adjacency matrices
        adj_matrices, _ = self._build_user_adjacency_matrix(user)
        
        # Get initial entity embeddings
        entity_emb = self.entity_embedding.weight  # [n_entities, dim]
        
        # Process each user separately (due to personalized adjacency)
        item_embs = []
        for idx in range(batch_size):
            # Normalize adjacency matrix
            adj_norm = self._normalize_adjacency_matrix(adj_matrices[idx])
            
            # GNN forward pass
            final_entity_emb = self._gnn_forward(entity_emb, adj_norm)
            
            # Get item embedding (user-specific)
            item_id = item[idx]
            item_emb = final_entity_emb[item_id]  # [dim]
            item_embs.append(item_emb)
        
        item_embs = torch.stack(item_embs, dim=0)  # [batch_size, dim]
        
        # Compute scores: inner product
        scores = torch.sum(user_emb * item_embs, dim=1)  # [batch_size]
        
        return scores

    def calculate_loss(self, interaction):
        r"""Calculate the total loss including recommendation loss, LS regularization, and L2 regularization.
        
        Args:
            interaction: interaction batch
        
        Returns:
            loss: total loss
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        
        # Recommendation loss
        scores = self.forward(user, item)
        rec_loss = self.rec_loss_fn(scores, label.float())
        
        # Label smoothness regularization
        ls_loss = self._calculate_ls_loss(user, item, label)
        
        # L2 regularization
        user_emb = self.user_embedding(user)
        item_emb = self.entity_embedding(item)
        reg_loss = self.reg_loss_fn(user_emb, item_emb)
        
        # Total loss
        total_loss = rec_loss + self.ls_weight * ls_loss + self.reg_weight * reg_loss
        
        return total_loss

    def _calculate_ls_loss(self, user, item, label):
        r"""Calculate label smoothness regularization loss using leave-one-out.
        
        Args:
            user: [batch_size]
            item: [batch_size]
            label: [batch_size]
        
        Returns:
            ls_loss: scalar
        """
        batch_size = len(user)
        
        # Build adjacency matrices
        adj_matrices, _ = self._build_user_adjacency_matrix(user)
        
        ls_losses = []
        for idx in range(batch_size):
            user_id = user[idx].item()
            item_id = item[idx].item()
            true_label = label[idx].item()
            
            # Get user's interaction history
            if user_id not in self.user_history:
                continue
            
            history_items = self.user_history[user_id]
            if len(history_items) < 2:  # Need at least 2 items for leave-one-out
                continue
            
            # Hold out current item
            other_items = [i for i in history_items if i != item_id]
            if len(other_items) == 0:
                continue
            
            # Create labels for other items (all positive)
            other_labels = np.ones(len(other_items))
            
            # Normalize adjacency matrix
            adj_norm = self._normalize_adjacency_matrix(adj_matrices[idx])
            
            # Label propagation to predict held-out item
            try:
                predicted_labels = self._label_propagation(
                    adj_norm.tocsr(), 
                    other_labels, 
                    other_items + [item_id]
                )
                predicted_label = predicted_labels[-1]  # Last one is the held-out item
                
                # Compute cross-entropy loss
                predicted_label = np.clip(predicted_label, 1e-7, 1 - 1e-7)  # Avoid log(0)
                if true_label == 1:
                    loss = -np.log(predicted_label)
                else:
                    loss = -np.log(1 - predicted_label)
                
                ls_losses.append(loss)
            except:
                continue
        
        if len(ls_losses) == 0:
            return torch.tensor(0.0).to(self.device)
        
        return torch.tensor(np.mean(ls_losses)).to(self.device)

    def predict(self, interaction):
        r"""Predict scores for given user-item pairs.
        
        Args:
            interaction: interaction batch
        
        Returns:
            scores: [batch_size]
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.
        
        Args:
            interaction: interaction batch
        
        Returns:
            scores: [batch_size * n_items]
        """
        user = interaction[self.USER_ID]
        batch_size = len(user)
        
        # Get user embeddings
        user_emb = self.user_embedding(user)  # [batch_size, dim]
        
        # Build user-personalized adjacency matrices
        adj_matrices, _ = self._build_user_adjacency_matrix(user)
        
        # Get initial entity embeddings
        entity_emb = self.entity_embedding.weight  # [n_entities, dim]
        
        all_scores = []
        for idx in range(batch_size):
            # Normalize adjacency matrix
            adj_norm = self._normalize_adjacency_matrix(adj_matrices[idx])
            
            # GNN forward pass
            final_entity_emb = self._gnn_forward(entity_emb, adj_norm)
            
            # Get all item embeddings (first n_items entities are items)
            item_embs = final_entity_emb[:self.n_items]  # [n_items, dim]
            
            # Compute scores
            scores = torch.matmul(user_emb[idx:idx+1], item_embs.t()).squeeze(0)  # [n_items]
            all_scores.append(scores)
        
        all_scores = torch.cat(all_scores, dim=0)  # [batch_size * n_items]
        return all_scores