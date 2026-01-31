# -*- coding: utf-8 -*-
# @Time   : 2024/01/01
# @Author : KGAT Implementation
# @Email  : kgat@example.com

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in KDD 2019.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class KGAT(KnowledgeRecommender):
    r"""KGAT is a knowledge-aware recommendation model that exploits high-order relations 
    through attentive embedding propagation layers on the collaborative knowledge graph.
    
    The model consists of three main components:
    1) Embedding layer with TransR-based knowledge graph embedding
    2) Attentive embedding propagation layers with knowledge-aware attention
    3) Prediction layer with layer-wise representation concatenation
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.aggregator_type = config["aggregator_type"]  # 'gcn', 'graphsage', 'bi-interaction'
        
        # TransR transformation size
        self.relation_dim = self.kg_embedding_size

        # Build Collaborative Knowledge Graph (CKG)
        self._build_ckg(dataset)

        # Define layers
        # Entity and relation embeddings (TransR-based)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.relation_dim)
        
        # TransR projection matrices for each relation
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.relation_dim
        )

        # Aggregator weight matrices for each layer
        self.aggregator_layers = nn.ModuleList()
        for i in range(self.n_layers):
            input_dim = self.embedding_size if i == 0 else self.embedding_size
            if self.aggregator_type == 'gcn':
                self.aggregator_layers.append(
                    nn.Linear(input_dim, self.embedding_size)
                )
            elif self.aggregator_type == 'graphsage':
                self.aggregator_layers.append(
                    nn.Linear(input_dim * 2, self.embedding_size)
                )
            elif self.aggregator_type == 'bi-interaction':
                self.aggregator_layers.append(nn.ModuleDict({
                    'w1': nn.Linear(input_dim, self.embedding_size),
                    'w2': nn.Linear(input_dim, self.embedding_size)
                }))

        # Loss functions
        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['adj_entity', 'adj_relation']

    def _build_ckg(self, dataset):
        r"""Build the Collaborative Knowledge Graph (CKG) by combining user-item interactions
        and item-entity relations. Users are treated as special entities in the graph.
        """
        # Get interaction matrix
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Get knowledge graph
        kg_graph = dataset.kg_graph(form='coo', value_field='relation_id')
        
        # Build adjacency lists for entities
        # adj_entity: [n_entities, neighbor_size] stores neighbor entity IDs
        # adj_relation: [n_entities, neighbor_size] stores relation IDs
        
        self.neighbor_sample_size = 8  # Fixed neighbor sampling size
        
        # Initialize adjacency structures
        adj_entity = {}
        adj_relation = {}
        
        # Add user-item interactions as special relations
        # User nodes: entity IDs from n_items to n_items + n_users - 1
        users = interaction_matrix.row
        items = interaction_matrix.col
        
        # Special relation ID for user-item interaction
        interact_relation_id = self.n_relations  # Use a new relation ID
        
        for user, item in zip(users, items):
            user_entity_id = self.n_items + user  # Map user to entity space
            
            # Add edge: user -> item
            if user_entity_id not in adj_entity:
                adj_entity[user_entity_id] = []
                adj_relation[user_entity_id] = []
            adj_entity[user_entity_id].append(item)
            adj_relation[user_entity_id].append(interact_relation_id)
            
            # Add reverse edge: item -> user
            if item not in adj_entity:
                adj_entity[item] = []
                adj_relation[item] = []
            adj_entity[item].append(user_entity_id)
            adj_relation[item].append(interact_relation_id)
        
        # Add KG triplets
        head_entities = kg_graph.row
        tail_entities = kg_graph.col
        relations = kg_graph.data
        
        for h, r, t in zip(head_entities, relations, tail_entities):
            if h not in adj_entity:
                adj_entity[h] = []
                adj_relation[h] = []
            adj_entity[h].append(t)
            adj_relation[h].append(r)
        
        # Convert to fixed-size arrays with padding
        max_entity_id = max(self.n_entities, self.n_items + self.n_users)
        self.adj_entity = np.zeros((max_entity_id, self.neighbor_sample_size), dtype=np.int64)
        self.adj_relation = np.zeros((max_entity_id, self.neighbor_sample_size), dtype=np.int64)
        
        for entity_id in range(max_entity_id):
            if entity_id in adj_entity:
                neighbors = adj_entity[entity_id]
                relations = adj_relation[entity_id]
                
                # Sample or pad neighbors
                if len(neighbors) >= self.neighbor_sample_size:
                    sampled_indices = np.random.choice(
                        len(neighbors), self.neighbor_sample_size, replace=False
                    )
                else:
                    sampled_indices = np.random.choice(
                        len(neighbors), self.neighbor_sample_size, replace=True
                    )
                
                self.adj_entity[entity_id] = [neighbors[i] for i in sampled_indices]
                self.adj_relation[entity_id] = [relations[i] for i in sampled_indices]
            else:
                # Use self-loop for entities without neighbors
                self.adj_entity[entity_id] = [entity_id] * self.neighbor_sample_size
                self.adj_relation[entity_id] = [0] * self.neighbor_sample_size
        
        # Convert to tensors
        self.adj_entity = torch.LongTensor(self.adj_entity).to(self.device)
        self.adj_relation = torch.LongTensor(self.adj_relation).to(self.device)

    def _get_ego_embeddings(self):
        r"""Get initial entity embeddings (layer 0)."""
        return self.entity_embedding.weight

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        r"""Compute TransR-based embeddings for knowledge graph triplets.
        
        Args:
            h: head entity IDs [batch_size]
            r: relation IDs [batch_size]
            pos_t: positive tail entity IDs [batch_size]
            neg_t: negative tail entity IDs [batch_size]
            
        Returns:
            Projected embeddings in relation-specific space
        """
        # Get embeddings
        h_e = self.entity_embedding(h)  # [batch_size, embedding_size]
        pos_t_e = self.entity_embedding(pos_t)
        neg_t_e = self.entity_embedding(neg_t)
        r_e = self.relation_embedding(r)  # [batch_size, relation_dim]
        
        # Get projection matrices
        r_trans_w = self.trans_w(r).view(
            -1, self.embedding_size, self.relation_dim
        )  # [batch_size, embedding_size, relation_dim]
        
        # Project entities to relation space
        h_e_projected = torch.bmm(
            h_e.unsqueeze(1), r_trans_w
        ).squeeze(1)  # [batch_size, relation_dim]
        pos_t_e_projected = torch.bmm(
            pos_t_e.unsqueeze(1), r_trans_w
        ).squeeze(1)
        neg_t_e_projected = torch.bmm(
            neg_t_e.unsqueeze(1), r_trans_w
        ).squeeze(1)
        
        return h_e_projected, r_e, pos_t_e_projected, neg_t_e_projected

    def _compute_attention(self, entity_emb, neighbor_emb, relation_emb, trans_w):
        r"""Compute knowledge-aware attention scores.
        
        Args:
            entity_emb: [batch_size, embedding_size]
            neighbor_emb: [batch_size, neighbor_size, embedding_size]
            relation_emb: [batch_size, neighbor_size, relation_dim]
            trans_w: [batch_size, neighbor_size, embedding_size, relation_dim]
            
        Returns:
            attention: [batch_size, neighbor_size] normalized attention weights
        """
        batch_size = entity_emb.size(0)
        
        # Project entity and neighbors to relation space
        # entity_projected: [batch_size, neighbor_size, relation_dim]
        entity_projected = torch.matmul(
            entity_emb.unsqueeze(1).expand(-1, self.neighbor_sample_size, -1).unsqueeze(2),
            trans_w
        ).squeeze(2)
        
        # neighbor_projected: [batch_size, neighbor_size, relation_dim]
        neighbor_projected = torch.matmul(
            neighbor_emb.unsqueeze(2), trans_w
        ).squeeze(2)
        
        # Compute attention score: (W_r * e_t)^T * tanh(W_r * e_h + e_r)
        # [batch_size, neighbor_size]
        attention_scores = torch.sum(
            neighbor_projected * torch.tanh(entity_projected + relation_emb),
            dim=-1
        )
        
        # Softmax normalization
        attention = F.softmax(attention_scores, dim=1)
        
        return attention

    def _aggregate_neighbors(self, entity_emb, neighbor_emb, attention):
        r"""Aggregate neighbor information with attention weights.
        
        Args:
            entity_emb: [batch_size, embedding_size]
            neighbor_emb: [batch_size, neighbor_size, embedding_size]
            attention: [batch_size, neighbor_size]
            
        Returns:
            aggregated: [batch_size, embedding_size]
        """
        # Weighted sum of neighbors
        # [batch_size, embedding_size]
        ego_network_emb = torch.sum(
            neighbor_emb * attention.unsqueeze(-1), dim=1
        )
        
        return ego_network_emb

    def _apply_aggregator(self, entity_emb, ego_network_emb, layer_idx):
        r"""Apply aggregation function to combine entity and neighbor embeddings.
        
        Args:
            entity_emb: [batch_size, embedding_size]
            ego_network_emb: [batch_size, embedding_size]
            layer_idx: current layer index
            
        Returns:
            aggregated: [batch_size, embedding_size]
        """
        if self.aggregator_type == 'gcn':
            # GCN aggregator: LeakyReLU(W(e_h + e_N_h))
            aggregated = self.aggregator_layers[layer_idx](
                entity_emb + ego_network_emb
            )
            aggregated = F.leaky_relu(aggregated)
            
        elif self.aggregator_type == 'graphsage':
            # GraphSage aggregator: LeakyReLU(W(e_h || e_N_h))
            concatenated = torch.cat([entity_emb, ego_network_emb], dim=1)
            aggregated = self.aggregator_layers[layer_idx](concatenated)
            aggregated = F.leaky_relu(aggregated)
            
        elif self.aggregator_type == 'bi-interaction':
            # Bi-Interaction aggregator: 
            # LeakyReLU(W1(e_h + e_N_h)) + LeakyReLU(W2(e_h ⊙ e_N_h))
            sum_part = self.aggregator_layers[layer_idx]['w1'](
                entity_emb + ego_network_emb
            )
            product_part = self.aggregator_layers[layer_idx]['w2'](
                entity_emb * ego_network_emb
            )
            aggregated = F.leaky_relu(sum_part) + F.leaky_relu(product_part)
        
        return aggregated

    def _propagate(self, entity_ids):
        r"""Perform L-layer attentive embedding propagation.
        
        Args:
            entity_ids: [batch_size] entity IDs to propagate
            
        Returns:
            embeddings: list of [batch_size, embedding_size] for each layer
        """
        batch_size = entity_ids.size(0)
        
        # Layer 0: initial embeddings
        entity_emb_list = [self.entity_embedding(entity_ids)]
        
        # Propagate through L layers
        for layer in range(self.n_layers):
            # Get current entity embeddings
            entity_emb = entity_emb_list[-1]  # [batch_size, embedding_size]
            
            # Get neighbor entities and relations
            # [batch_size, neighbor_size]
            neighbor_entities = self.adj_entity[entity_ids]
            neighbor_relations = self.adj_relation[entity_ids]
            
            # Get neighbor embeddings
            # [batch_size, neighbor_size, embedding_size]
            neighbor_emb = self.entity_embedding(neighbor_entities)
            
            # Get relation embeddings and projection matrices
            # [batch_size, neighbor_size, relation_dim]
            relation_emb = self.relation_embedding(neighbor_relations)
            
            # [batch_size, neighbor_size, embedding_size, relation_dim]
            trans_w = self.trans_w(neighbor_relations).view(
                batch_size, self.neighbor_sample_size, 
                self.embedding_size, self.relation_dim
            )
            
            # Compute attention scores
            attention = self._compute_attention(
                entity_emb, neighbor_emb, relation_emb, trans_w
            )
            
            # Aggregate neighbor information
            ego_network_emb = self._aggregate_neighbors(
                entity_emb, neighbor_emb, attention
            )
            
            # Apply aggregator
            new_entity_emb = self._apply_aggregator(
                entity_emb, ego_network_emb, layer
            )
            
            entity_emb_list.append(new_entity_emb)
        
        return entity_emb_list

    def forward(self, user, item):
        r"""Forward propagation for user-item pairs.
        
        Args:
            user: [batch_size] user IDs
            item: [batch_size] item IDs
            
        Returns:
            scores: [batch_size] predicted scores
        """
        # Map users to entity space
        user_entity_ids = user + self.n_items
        
        # Propagate embeddings
        user_emb_list = self._propagate(user_entity_ids)
        item_emb_list = self._propagate(item)
        
        # Concatenate embeddings from all layers
        user_emb = torch.cat(user_emb_list, dim=1)  # [batch_size, (L+1)*embedding_size]
        item_emb = torch.cat(item_emb_list, dim=1)
        
        # Compute scores via inner product
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores

    def calculate_rs_loss(self, interaction):
        r"""Calculate recommendation loss (BPR loss).
        
        Args:
            interaction: interaction data
            
        Returns:
            rec_loss: BPR loss for recommendation
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        pos_score = self.forward(user, pos_item)
        neg_score = self.forward(user, neg_item)
        
        rec_loss = self.rec_loss(pos_score, neg_score)
        
        return rec_loss

    def calculate_kg_loss(self, interaction):
        r"""Calculate knowledge graph embedding loss (TransR loss).
        
        Args:
            interaction: interaction data with KG triplets
            
        Returns:
            kg_loss: TransR loss for knowledge graph
        """
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]
        
        # Get TransR embeddings
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        
        # Compute scores: ||W_r * e_h + e_r - W_r * e_t||^2
        pos_score = torch.sum((h_e + r_e - pos_t_e) ** 2, dim=1)
        neg_score = torch.sum((h_e + r_e - neg_t_e) ** 2, dim=1)
        
        # BPR loss (lower score is better, so we use neg - pos)
        kg_loss = self.kg_loss(neg_score, pos_score)
        
        return kg_loss

    def calculate_loss(self, interaction):
        r"""Calculate total loss combining recommendation and KG losses.
        
        Args:
            interaction: interaction data
            
        Returns:
            loss: total loss
        """
        # Recommendation loss
        rec_loss = self.calculate_rs_loss(interaction)
        
        # KG embedding loss
        kg_loss = self.calculate_kg_loss(interaction)
        
        # Regularization loss
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_entity_ids = user + self.n_items
        user_emb = self.entity_embedding(user_entity_ids)
        pos_item_emb = self.entity_embedding(pos_item)
        neg_item_emb = self.entity_embedding(neg_item)
        
        reg_loss = self.reg_loss(user_emb, pos_item_emb, neg_item_emb)
        
        # Add regularization for relation embeddings and transformation matrices
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]
        
        h_emb = self.entity_embedding(h)
        pos_t_emb = self.entity_embedding(pos_t)
        neg_t_emb = self.entity_embedding(neg_t)
        r_emb = self.relation_embedding(r)
        trans_w = self.trans_w(r)
        
        reg_loss = reg_loss + self.reg_loss(h_emb, pos_t_emb, neg_t_emb, r_emb, trans_w)
        
        # Total loss
        loss = rec_loss + kg_loss + self.reg_weight * reg_loss
        
        return loss

    def predict(self, interaction):
        r"""Predict scores for user-item pairs.
        
        Args:
            interaction: interaction data
            
        Returns:
            scores: predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.
        
        Args:
            interaction: interaction data with user IDs
            
        Returns:
            scores: [batch_size * n_items] predicted scores
        """
        user = interaction[self.USER_ID]
        batch_size = user.size(0)
        
        # Map users to entity space
        user_entity_ids = user + self.n_items
        
        # Propagate user embeddings
        user_emb_list = self._propagate(user_entity_ids)
        user_emb = torch.cat(user_emb_list, dim=1)  # [batch_size, (L+1)*embedding_size]
        
        # Propagate all item embeddings
        all_item_ids = torch.arange(self.n_items, dtype=torch.long, device=self.device)
        all_item_emb_list = self._propagate(all_item_ids)
        all_item_emb = torch.cat(all_item_emb_list, dim=1)  # [n_items, (L+1)*embedding_size]
        
        # Compute scores via matrix multiplication
        # [batch_size, n_items]
        scores = torch.matmul(user_emb, all_item_emb.t())
        
        return scores.view(-1)