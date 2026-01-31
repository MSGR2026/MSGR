# -*- coding: utf-8 -*-
# @Time   : 2024/12/19
# @Author : KGCN Implementation
# @Email  : kgcn@example.com

r"""
KGCN
##################################################
Reference:
    Hongwei Wang et al. "Knowledge Graph Convolutional Networks for Recommender Systems." in WWW 2019.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class KGCN(KnowledgeRecommender):
    r"""KGCN is a knowledge-aware recommendation model that captures high-order structural proximity
    in knowledge graphs through graph convolutional networks with user-specific neighbor aggregation.
    
    The model propagates entity representations through multiple layers, where each layer aggregates
    information from sampled neighbors with user-relation attention weights.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(KGCN, self).__init__(config, dataset)

        # Load parameters info
        self.LABEL = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.n_iter = config["n_iter"]  # Number of iterations (H in paper)
        self.neighbor_sample_size = config["neighbor_sample_size"]  # K in paper
        self.aggregator_type = config["aggregator"]  # 'sum', 'concat', or 'neighbor'
        self.reg_weight = config["reg_weight"]
        
        # Build knowledge graph adjacency structure
        self._build_kg_adjacency(dataset)
        
        # Define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        
        # Aggregator layers for each iteration
        self.aggregator_layers = nn.ModuleList()
        for i in range(self.n_iter):
            if self.aggregator_type == 'sum':
                self.aggregator_layers.append(
                    nn.Linear(self.embedding_size, self.embedding_size, bias=True)
                )
            elif self.aggregator_type == 'concat':
                self.aggregator_layers.append(
                    nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
                )
            elif self.aggregator_type == 'neighbor':
                self.aggregator_layers.append(
                    nn.Linear(self.embedding_size, self.embedding_size, bias=True)
                )
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['adj_entity', 'adj_relation']

    def _build_kg_adjacency(self, dataset):
        r"""Build adjacency lists for entities and relations in the knowledge graph.
        For each entity, sample a fixed number of neighbors. If an entity has fewer neighbors
        than neighbor_sample_size, pad with itself.
        """
        # Get KG triples
        kg = dataset.kg_graph(form='coo', value_field='relation_id')
        head_entities = kg.row
        tail_entities = kg.col
        relations = kg.data
        
        # Build neighbor dictionary: entity -> list of (tail, relation) pairs
        neighbor_dict = {}
        for h, t, r in zip(head_entities, tail_entities, relations):
            if h not in neighbor_dict:
                neighbor_dict[h] = []
            neighbor_dict[h].append((t, r))
        
        # Sample neighbors for each entity
        adj_entity = np.zeros((self.n_entities, self.neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros((self.n_entities, self.neighbor_sample_size), dtype=np.int64)
        
        for entity in range(self.n_entities):
            if entity in neighbor_dict and len(neighbor_dict[entity]) > 0:
                neighbors = neighbor_dict[entity]
                n_neighbors = len(neighbors)
                
                if n_neighbors >= self.neighbor_sample_size:
                    # Sample without replacement
                    sampled_indices = np.random.choice(
                        n_neighbors, size=self.neighbor_sample_size, replace=False
                    )
                else:
                    # Sample with replacement
                    sampled_indices = np.random.choice(
                        n_neighbors, size=self.neighbor_sample_size, replace=True
                    )
                
                for i, idx in enumerate(sampled_indices):
                    adj_entity[entity][i] = neighbors[idx][0]
                    adj_relation[entity][i] = neighbors[idx][1]
            else:
                # No neighbors, use self-loop
                adj_entity[entity] = entity
                adj_relation[entity] = 0  # Use relation 0 as default
        
        # Convert to torch tensors
        self.adj_entity = torch.LongTensor(adj_entity).to(self.device)
        self.adj_relation = torch.LongTensor(adj_relation).to(self.device)

    def _get_receptive_field(self, entities):
        r"""Get the receptive field for given entities across all layers.
        
        Args:
            entities: [batch_size] tensor of entity IDs
            
        Returns:
            receptive_field: list of tensors, where receptive_field[h] contains
                            all entities at layer h. Shape grows as:
                            [batch_size] -> [batch_size * K] -> [batch_size * K^2] -> ...
        """
        receptive_field = [entities]
        
        for h in range(self.n_iter):
            # Get neighbors of all entities at current layer
            current_entities = receptive_field[-1]
            # [current_size, neighbor_sample_size]
            neighbors = torch.index_select(self.adj_entity, 0, current_entities)
            # Flatten to [current_size * neighbor_sample_size]
            neighbors = neighbors.view(-1)
            receptive_field.append(neighbors)
        
        return receptive_field

    def _compute_user_relation_score(self, user_emb, relation_emb):
        r"""Compute user-relation scores using inner product.
        
        Args:
            user_emb: [batch_size, dim] or [batch_size, 1, dim]
            relation_emb: [batch_size, neighbor_sample_size, dim]
            
        Returns:
            scores: [batch_size, neighbor_sample_size]
        """
        if len(user_emb.shape) == 2:
            user_emb = user_emb.unsqueeze(1)  # [batch_size, 1, dim]
        
        # Inner product: [batch_size, 1, dim] * [batch_size, neighbor_sample_size, dim]
        # -> [batch_size, neighbor_sample_size]
        scores = torch.sum(user_emb * relation_emb, dim=-1)
        return scores

    def _aggregate_neighbors(self, user_emb, entity_emb, neighbor_entity_emb, 
                            neighbor_relation_emb, layer_idx):
        r"""Aggregate entity and its neighborhood representations.
        
        Args:
            user_emb: [batch_size, dim]
            entity_emb: [batch_size, dim] - current entity representations
            neighbor_entity_emb: [batch_size, neighbor_sample_size, dim]
            neighbor_relation_emb: [batch_size, neighbor_sample_size, dim]
            layer_idx: current layer index
            
        Returns:
            aggregated_emb: [batch_size, dim]
        """
        batch_size = entity_emb.shape[0]
        
        # Compute user-relation attention scores
        # [batch_size, neighbor_sample_size]
        scores = self._compute_user_relation_score(user_emb, neighbor_relation_emb)
        
        # Normalize scores with softmax
        # [batch_size, neighbor_sample_size]
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute neighborhood representation as weighted sum
        # [batch_size, neighbor_sample_size, 1] * [batch_size, neighbor_sample_size, dim]
        # -> [batch_size, dim]
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        neighborhood_emb = torch.sum(attention_weights_expanded * neighbor_entity_emb, dim=1)
        
        # Apply aggregator
        if self.aggregator_type == 'sum':
            # Sum aggregator: σ(W · (v + v_N) + b)
            aggregated = self.aggregator_layers[layer_idx](entity_emb + neighborhood_emb)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        elif self.aggregator_type == 'concat':
            # Concat aggregator: σ(W · concat(v, v_N) + b)
            concatenated = torch.cat([entity_emb, neighborhood_emb], dim=1)
            aggregated = self.aggregator_layers[layer_idx](concatenated)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        elif self.aggregator_type == 'neighbor':
            # Neighbor aggregator: σ(W · v_N + b)
            aggregated = self.aggregator_layers[layer_idx](neighborhood_emb)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        return aggregated

    def _propagate(self, user_emb, entities):
        r"""Propagate entity representations through multiple layers.
        
        Args:
            user_emb: [batch_size, dim]
            entities: [batch_size] - starting entities (items)
            
        Returns:
            final_entity_emb: [batch_size, dim] - H-order entity representations
        """
        batch_size = entities.shape[0]
        
        # Get receptive field
        receptive_field = self._get_receptive_field(entities)
        
        # Initialize entity representations for each layer
        # entity_representations[h] stores h-order representations
        entity_representations = {}
        
        # 0-order: initial embeddings
        for h in range(self.n_iter + 1):
            entities_at_layer = receptive_field[h]
            entity_representations[h] = self.entity_embedding(entities_at_layer)
        
        # Propagate through layers (from layer H-1 to 0)
        for h in range(self.n_iter - 1, -1, -1):
            # Entities at current layer
            entities_at_layer = receptive_field[h]
            num_entities = entities_at_layer.shape[0]
            
            # Get neighbors for these entities
            # [num_entities, neighbor_sample_size]
            neighbor_entities = torch.index_select(self.adj_entity, 0, entities_at_layer)
            neighbor_relations = torch.index_select(self.adj_relation, 0, entities_at_layer)
            
            # Get embeddings
            # Current entity embeddings: [num_entities, dim]
            current_entity_emb = entity_representations[h]
            
            # Neighbor embeddings from next layer: [num_entities, neighbor_sample_size, dim]
            neighbor_entity_emb = entity_representations[h + 1].view(
                num_entities, self.neighbor_sample_size, self.embedding_size
            )
            
            # Relation embeddings: [num_entities, neighbor_sample_size, dim]
            neighbor_relation_emb = self.relation_embedding(neighbor_relations)
            
            # Expand user embeddings to match batch
            # For entities beyond batch_size, we need to replicate user embeddings
            if num_entities > batch_size:
                # Calculate how many times to replicate each user
                repeat_factor = num_entities // batch_size
                user_emb_expanded = user_emb.repeat_interleave(repeat_factor, dim=0)
                if num_entities % batch_size != 0:
                    # Handle remainder
                    remainder = num_entities % batch_size
                    user_emb_expanded = torch.cat([
                        user_emb_expanded,
                        user_emb[:remainder]
                    ], dim=0)
            else:
                user_emb_expanded = user_emb
            
            # Aggregate
            aggregated_emb = self._aggregate_neighbors(
                user_emb_expanded, current_entity_emb, 
                neighbor_entity_emb, neighbor_relation_emb, h
            )
            
            # Update representations for this layer
            entity_representations[h] = aggregated_emb
        
        # Return the final representation of the starting entities (first batch_size entities)
        final_entity_emb = entity_representations[0][:batch_size]
        
        return final_entity_emb

    def forward(self, user, item):
        r"""Forward propagation.
        
        Args:
            user: [batch_size]
            item: [batch_size]
            
        Returns:
            scores: [batch_size] - predicted scores
        """
        # Get user embeddings
        user_emb = self.user_embedding(user)
        
        # Propagate item (entity) representations
        item_emb = self._propagate(user_emb, item)
        
        # Compute scores as inner product
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores

    def calculate_loss(self, interaction):
        r"""Calculate the training loss.
        
        Args:
            interaction: interaction data
            
        Returns:
            loss: total loss including BCE loss and L2 regularization
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        
        # Forward pass
        scores = self.forward(user, item)
        
        # BCE loss
        bce_loss = self.bce_loss(scores, label.float())
        
        # L2 regularization
        user_emb = self.user_embedding(user)
        item_entity_emb = self.entity_embedding(item)
        reg_loss = self.reg_weight * self.l2_loss(user_emb, item_entity_emb)
        
        # Total loss
        loss = bce_loss + reg_loss
        
        return loss

    def predict(self, interaction):
        r"""Predict scores for given user-item pairs.
        
        Args:
            interaction: interaction data
            
        Returns:
            scores: predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        scores = self.forward(user, item)
        
        return torch.sigmoid(scores)

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.
        
        Args:
            interaction: interaction data containing users
            
        Returns:
            scores: [batch_size * n_items] predicted scores for all items
        """
        user = interaction[self.USER_ID]
        batch_size = user.shape[0]
        
        # Get user embeddings
        user_emb = self.user_embedding(user)
        
        # For efficiency, we propagate all items at once
        # Items are the first n_items entities
        all_items = torch.arange(self.n_items, dtype=torch.long, device=self.device)
        
        # We need to handle this in batches to avoid OOM
        # For simplicity, we'll compute item representations for all items at once
        # But replicate user embeddings appropriately
        
        # Expand user embeddings: [batch_size, dim] -> [batch_size, n_items, dim]
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, self.n_items, -1)
        user_emb_expanded = user_emb_expanded.reshape(-1, self.embedding_size)
        
        # Replicate items for each user: [n_items] -> [batch_size * n_items]
        items_expanded = all_items.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        
        # Propagate
        item_emb = self._propagate(user_emb_expanded, items_expanded)
        
        # Compute scores
        scores = torch.sum(user_emb_expanded * item_emb, dim=1)
        scores = torch.sigmoid(scores)
        
        return scores