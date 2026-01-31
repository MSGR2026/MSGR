# -*- coding: utf-8 -*-
# @Time   : 2024/01/01
# @Author : MCCLK Implementation
# @Email  : mcclk@example.com

r"""
MCCLK
##################################################
Reference:
    Ding Zou et al. "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." in SIGIR 2022.

Note: This implementation focuses on the multi-view contrastive learning framework with:
    1. Structural view: Knowledge graph propagation with relation-aware attention
    2. Collaborative view: LightGCN-style user-item propagation
    3. Semantic view: kNN item-item graph based on entity features
    4. Multi-level contrastive learning across views
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """
    Relational Path-aware Aggregator for KG propagation with relation-aware attention.
    """

    def __init__(self, embedding_size):
        super(Aggregator, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, entity_emb, edge_index, edge_type, relation_emb, n_entities):
        """
        Aggregate neighbor information with relation-aware attention.
        
        Args:
            entity_emb: [n_entities, embedding_size]
            edge_index: [2, n_edges] - (head, tail) pairs
            edge_type: [n_edges] - relation types
            relation_emb: [n_relations, embedding_size]
            n_entities: number of entities
            
        Returns:
            entity_agg: [n_entities, embedding_size]
        """
        from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

        head, tail = edge_index
        
        # Get relation embeddings for each edge
        edge_relation_emb = relation_emb[edge_type]  # [n_edges, embedding_size]
        
        # Get tail entity embeddings
        tail_emb = entity_emb[tail]  # [n_edges, embedding_size]
        head_emb = entity_emb[head]  # [n_edges, embedding_size]
        
        # Calculate attention scores: (h * r * t).sum(-1)
        # Relation-aware attention mechanism
        attention_scores = torch.sum(head_emb * edge_relation_emb * tail_emb, dim=-1)  # [n_edges]
        
        # Normalize attention scores per head entity
        attention_weights = scatter_softmax(attention_scores, head, dim=0)  # [n_edges]
        
        # Weighted aggregation of neighbor embeddings
        weighted_neighbor_emb = tail_emb * attention_weights.unsqueeze(-1)  # [n_edges, embedding_size]
        
        # Aggregate to head entities
        entity_agg = scatter_sum(weighted_neighbor_emb, head, dim=0, dim_size=n_entities)
        
        return entity_agg


class LightGCNConv(nn.Module):
    """
    LightGCN convolution layer for collaborative filtering view.
    """

    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, user_emb, item_emb, interact_mat):
        """
        Perform one layer of LightGCN propagation.
        
        Args:
            user_emb: [n_users, embedding_size]
            item_emb: [n_items, embedding_size]
            interact_mat: normalized sparse interaction matrix
            
        Returns:
            user_emb_new: [n_users, embedding_size]
            item_emb_new: [n_items, embedding_size]
        """
        # User aggregation: aggregate from items
        user_emb_new = torch.sparse.mm(interact_mat, item_emb)
        
        # Item aggregation: aggregate from users
        item_emb_new = torch.sparse.mm(interact_mat.t(), user_emb)
        
        return user_emb_new, item_emb_new


class MCCLK(KnowledgeRecommender):
    r"""MCCLK is a knowledge-aware recommendation model that leverages multi-level 
    cross-view contrastive learning to capture user preferences from multiple perspectives:
    structural (KG), collaborative (user-item), and semantic (item-item similarity).
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MCCLK, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.temperature = config["temperature"]
        self.alpha = config["alpha"]  # Weight for local contrastive loss
        self.beta = config["beta"]  # Weight for global contrastive loss
        self.knn_k = config["knn_k"]  # Number of neighbors for semantic graph
        self.node_dropout_rate = config["node_dropout_rate"]
        self.mess_dropout_rate = config["mess_dropout_rate"]

        # Load dataset info
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")

        # Build graphs
        self._build_graphs()

        # Define embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        # Structural view aggregators
        self.kg_aggregators = nn.ModuleList([
            Aggregator(self.embedding_size) for _ in range(self.n_layers)
        ])

        # Collaborative view (LightGCN)
        self.lightgcn_layers = nn.ModuleList([
            LightGCNConv() for _ in range(self.n_layers)
        ])

        # Semantic view aggregation layer
        self.item_agg_layer = nn.Linear(self.embedding_size, self.embedding_size)

        # Projection heads for contrastive learning
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size)  # Structural projection
        self.fc2 = nn.Linear(self.embedding_size, self.embedding_size)  # Collaborative projection
        self.fc3 = nn.Linear(self.embedding_size, self.embedding_size)  # Semantic projection

        # Dropout
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)

        # Loss functions
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Cache for evaluation
        self.restore_user_e = None
        self.restore_item_e = None

        # Parameters initialization
        self.apply(xavier_uniform_initialization)

    def _build_graphs(self):
        """Build all required graph structures."""
        # 1. Build normalized user-item interaction matrix for LightGCN
        self.interact_mat = self._build_interact_matrix()

        # 2. Build KG edge index and type
        self.edge_index, self.edge_type = self._build_kg_edges()

        # 3. Build semantic kNN item-item graph (initialized later with embeddings)
        self.semantic_adj = None

    def _build_interact_matrix(self):
        """Build normalized user-item interaction matrix."""
        # D^{-1/2} A D^{-1/2} normalization
        row = self.inter_matrix.row
        col = self.inter_matrix.col
        
        # Compute degree
        user_degree = np.array(self.inter_matrix.sum(axis=1)).flatten()
        item_degree = np.array(self.inter_matrix.sum(axis=0)).flatten()
        
        # D^{-1/2}
        user_d_inv_sqrt = np.power(user_degree + 1e-8, -0.5)
        item_d_inv_sqrt = np.power(item_degree + 1e-8, -0.5)
        
        # Normalize
        data = user_d_inv_sqrt[row] * item_d_inv_sqrt[col]
        
        # Create sparse tensor
        indices = torch.LongTensor(np.array([row, col]))
        values = torch.FloatTensor(data)
        shape = torch.Size([self.n_users, self.n_items])
        
        interact_mat = torch.sparse.FloatTensor(indices, values, shape)
        
        return interact_mat.to(self.device)

    def _build_kg_edges(self):
        """Build KG edge index and edge type tensors."""
        edge_index = torch.LongTensor(np.array([self.kg_graph.row, self.kg_graph.col]))
        edge_type = torch.LongTensor(np.array(self.kg_graph.data))
        
        return edge_index.to(self.device), edge_type.to(self.device)

    def _build_semantic_adj(self, item_emb):
        """Build kNN item-item semantic graph based on item embeddings."""
        # Compute cosine similarity
        item_emb_norm = F.normalize(item_emb, p=2, dim=1)  # [n_items, embedding_size]
        sim_matrix = torch.mm(item_emb_norm, item_emb_norm.t())  # [n_items, n_items]
        
        # Set diagonal to -inf to exclude self-loops in topk
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # Get top-k neighbors
        _, topk_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)  # [n_items, knn_k]
        
        # Build sparse adjacency matrix
        row_indices = torch.arange(self.n_items, device=self.device).unsqueeze(1).expand(-1, self.knn_k).reshape(-1)
        col_indices = topk_indices.reshape(-1)
        
        # Create normalized adjacency (uniform weights)
        values = torch.ones(row_indices.shape[0], device=self.device) / self.knn_k
        
        indices = torch.stack([row_indices, col_indices])
        shape = torch.Size([self.n_items, self.n_items])
        
        semantic_adj = torch.sparse.FloatTensor(indices, values, shape)
        
        return semantic_adj

    def _structural_view_forward(self, entity_emb):
        """
        Structural view: KG propagation with relation-aware attention.
        
        Args:
            entity_emb: [n_entities, embedding_size]
            
        Returns:
            entity_emb_list: list of entity embeddings at each layer
        """
        entity_emb_list = [entity_emb]
        relation_emb = self.relation_embedding.weight
        
        for layer in range(self.n_layers):
            # KG aggregation
            entity_agg = self.kg_aggregators[layer](
                entity_emb_list[-1],
                self.edge_index,
                self.edge_type,
                relation_emb,
                self.n_entities
            )
            
            # Combine with residual
            entity_emb_new = entity_emb_list[-1] + entity_agg
            
            # Message dropout
            if self.training and self.mess_dropout_rate > 0:
                entity_emb_new = self.mess_dropout(entity_emb_new)
            
            # Normalize
            entity_emb_new = F.normalize(entity_emb_new, p=2, dim=1)
            
            entity_emb_list.append(entity_emb_new)
        
        return entity_emb_list

    def _collaborative_view_forward(self, user_emb, item_emb):
        """
        Collaborative view: LightGCN propagation on user-item graph.
        
        Args:
            user_emb: [n_users, embedding_size]
            item_emb: [n_items, embedding_size]
            
        Returns:
            user_emb_list: list of user embeddings at each layer
            item_emb_list: list of item embeddings at each layer
        """
        user_emb_list = [user_emb]
        item_emb_list = [item_emb]
        
        for layer in range(self.n_layers):
            user_emb_new, item_emb_new = self.lightgcn_layers[layer](
                user_emb_list[-1],
                item_emb_list[-1],
                self.interact_mat
            )
            
            # Message dropout
            if self.training and self.mess_dropout_rate > 0:
                user_emb_new = self.mess_dropout(user_emb_new)
                item_emb_new = self.mess_dropout(item_emb_new)
            
            # Normalize
            user_emb_new = F.normalize(user_emb_new, p=2, dim=1)
            item_emb_new = F.normalize(item_emb_new, p=2, dim=1)
            
            user_emb_list.append(user_emb_new)
            item_emb_list.append(item_emb_new)
        
        return user_emb_list, item_emb_list

    def _semantic_view_forward(self, item_emb):
        """
        Semantic view: kNN item-item graph propagation.
        
        Args:
            item_emb: [n_items, embedding_size]
            
        Returns:
            item_emb_semantic: [n_items, embedding_size]
        """
        # Build semantic adjacency based on current item embeddings
        semantic_adj = self._build_semantic_adj(item_emb)
        
        # Propagate on semantic graph
        item_emb_agg = torch.sparse.mm(semantic_adj, item_emb)
        
        # Transform
        item_emb_semantic = self.item_agg_layer(item_emb_agg)
        item_emb_semantic = F.relu(item_emb_semantic)
        
        # Combine with original
        item_emb_semantic = item_emb + item_emb_semantic
        item_emb_semantic = F.normalize(item_emb_semantic, p=2, dim=1)
        
        return item_emb_semantic

    def _compute_contrastive_loss(self, emb1, emb2, batch_indices=None):
        """
        Compute InfoNCE contrastive loss between two views.
        
        Args:
            emb1: [batch_size, embedding_size] or [n_nodes, embedding_size]
            emb2: [batch_size, embedding_size] or [n_nodes, embedding_size]
            batch_indices: indices to sample from (if None, use all)
            
        Returns:
            loss: scalar contrastive loss
        """
        if batch_indices is not None:
            emb1 = emb1[batch_indices]
            emb2 = emb2[batch_indices]
        
        # Normalize
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Positive pairs: same index
        pos_score = torch.sum(emb1 * emb2, dim=1) / self.temperature  # [batch_size]
        
        # Negative pairs: all other indices
        neg_score = torch.mm(emb1, emb2.t()) / self.temperature  # [batch_size, batch_size]
        
        # InfoNCE loss
        pos_score_exp = torch.exp(pos_score)
        neg_score_exp = torch.exp(neg_score).sum(dim=1)
        
        loss = -torch.log(pos_score_exp / (neg_score_exp + 1e-8)).mean()
        
        return loss

    def forward(self):
        """
        Forward pass to compute user and item representations from all views.
        
        Returns:
            user_emb_final: [n_users, embedding_size]
            item_emb_final: [n_items, embedding_size]
            structural_item_emb: [n_items, embedding_size]
            collaborative_item_emb: [n_items, embedding_size]
            semantic_item_emb: [n_items, embedding_size]
            collaborative_user_emb: [n_users, embedding_size]
        """
        # Get initial embeddings
        user_emb_init = self.user_embedding.weight
        entity_emb_init = self.entity_embedding.weight
        item_emb_init = entity_emb_init[:self.n_items]
        
        # Structural view (KG propagation)
        entity_emb_list = self._structural_view_forward(entity_emb_init)
        structural_entity_emb = torch.stack(entity_emb_list, dim=1).mean(dim=1)
        structural_item_emb = structural_entity_emb[:self.n_items]
        
        # Collaborative view (LightGCN)
        user_emb_list, item_emb_list = self._collaborative_view_forward(user_emb_init, item_emb_init)
        collaborative_user_emb = torch.stack(user_emb_list, dim=1).mean(dim=1)
        collaborative_item_emb = torch.stack(item_emb_list, dim=1).mean(dim=1)
        
        # Semantic view (kNN item graph)
        semantic_item_emb = self._semantic_view_forward(item_emb_init)
        
        # Final representations: combine all views
        user_emb_final = collaborative_user_emb
        item_emb_final = (structural_item_emb + collaborative_item_emb + semantic_item_emb) / 3.0
        
        return (user_emb_final, item_emb_final, 
                structural_item_emb, collaborative_item_emb, semantic_item_emb,
                collaborative_user_emb)

    def calculate_loss(self, interaction):
        """
        Calculate the total training loss including:
        1. BPR recommendation loss
        2. Multi-level contrastive losses
        3. L2 regularization
        
        Args:
            interaction: batch interaction data
            
        Returns:
            loss: total loss
        """
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Forward pass
        (user_emb_final, item_emb_final,
         structural_item_emb, collaborative_item_emb, semantic_item_emb,
         collaborative_user_emb) = self.forward()

        # Get batch embeddings for recommendation
        u_emb = user_emb_final[user]
        pos_emb = item_emb_final[pos_item]
        neg_emb = item_emb_final[neg_item]

        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Regularization loss
        reg_loss = self.reg_loss(
            self.user_embedding(user),
            self.entity_embedding(pos_item),
            self.entity_embedding(neg_item)
        )

        # Multi-level contrastive losses
        # Local contrastive: item-level alignment between views
        # Project embeddings
        struct_proj = self.fc1(structural_item_emb)
        collab_proj = self.fc2(collaborative_item_emb)
        semantic_proj = self.fc3(semantic_item_emb)

        # Sample items for contrastive learning (use batch items)
        batch_items = torch.unique(torch.cat([pos_item, neg_item]))
        
        # Local contrastive: structural vs collaborative
        local_cl_loss_1 = self._compute_contrastive_loss(
            struct_proj, collab_proj, batch_items
        )
        
        # Local contrastive: collaborative vs semantic
        local_cl_loss_2 = self._compute_contrastive_loss(
            collab_proj, semantic_proj, batch_items
        )
        
        # Local contrastive: structural vs semantic
        local_cl_loss_3 = self._compute_contrastive_loss(
            struct_proj, semantic_proj, batch_items
        )
        
        local_cl_loss = (local_cl_loss_1 + local_cl_loss_2 + local_cl_loss_3) / 3.0

        # Global contrastive: user-item alignment
        # Sample users for contrastive learning
        batch_users = torch.unique(user)
        
        # User view projections (use collaborative user embeddings)
        user_collab_proj = self.fc2(collaborative_user_emb)
        
        # Global contrastive: user vs item (positive pairs from interactions)
        # This aligns user representations with their interacted items
        global_cl_loss = self._compute_contrastive_loss(
            user_collab_proj[user], collab_proj[pos_item]
        )

        # Total loss
        loss = mf_loss + self.reg_weight * reg_loss + self.alpha * local_cl_loss + self.beta * global_cl_loss

        return loss

    def predict(self, interaction):
        """
        Predict scores for user-item pairs.
        
        Args:
            interaction: interaction data
            
        Returns:
            scores: predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        (user_emb_final, item_emb_final, _, _, _, _) = self.forward()

        u_emb = user_emb_final[user]
        i_emb = item_emb_final[item]

        scores = torch.sum(u_emb * i_emb, dim=1)

        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        
        Args:
            interaction: interaction data with user IDs
            
        Returns:
            scores: [batch_size * n_items] predicted scores
        """
        user = interaction[self.USER_ID]

        if self.restore_user_e is None or self.restore_item_e is None:
            (self.restore_user_e, self.restore_item_e, _, _, _, _) = self.forward()

        u_emb = self.restore_user_e[user]
        i_emb = self.restore_item_e

        scores = torch.matmul(u_emb, i_emb.t())

        return scores.view(-1)