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

    # DIFFERENT: KGCN and MCCLK both use PAIRWISE input in RecBole; do not switch MCCLK to POINTWISE.
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
        
        # DIFFERENT: KGCN uses fixed neighbor sampling and receptive-field expansion; MCCLK uses full KG edges with scatter ops.
        self._build_kg_adjacency(dataset)
        
        # REUSE: Same embedding layers for users, entities, and relations
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        
        # DIFFERENT: KGCN uses simple linear aggregators, MCCLK uses attention-based Aggregator class
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
        
        # REUSE: Same xavier initialization
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

    # DIFFERENT: KGCN aggregation uses user-relation attention; MCCLK uses head-tail-relation attention
    def _aggregate_neighbors(self, user_emb, entity_emb, neighbor_entity_emb, 
                            neighbor_relation_emb, layer_idx):
        r"""Aggregate entity and its neighborhood representations.
        """
        batch_size = entity_emb.shape[0]
        
        # Compute user-relation attention scores
        # [batch_size, neighbor_sample_size]
        scores = self._compute_user_relation_score(user_emb, neighbor_relation_emb)
        
        # Normalize scores with softmax
        # [batch_size, neighbor_sample_size]
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute neighborhood representation as weighted sum
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        neighborhood_emb = torch.sum(attention_weights_expanded * neighbor_entity_emb, dim=1)
        
        # Apply aggregator
        if self.aggregator_type == 'sum':
            aggregated = self.aggregator_layers[layer_idx](entity_emb + neighborhood_emb)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        elif self.aggregator_type == 'concat':
            concatenated = torch.cat([entity_emb, neighborhood_emb], dim=1)
            aggregated = self.aggregator_layers[layer_idx](concatenated)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        elif self.aggregator_type == 'neighbor':
            aggregated = self.aggregator_layers[layer_idx](neighborhood_emb)
            aggregated = F.relu(aggregated) if layer_idx < self.n_iter - 1 else torch.tanh(aggregated)
        
        return aggregated

    def _propagate(self, user_emb, entities):
        r"""Propagate entity representations through multiple layers.
        """
        batch_size = entities.shape[0]
        
        # Get receptive field
        receptive_field = self._get_receptive_field(entities)
        
        entity_representations = {}
        
        # 0-order: initial embeddings
        for h in range(self.n_iter + 1):
            entities_at_layer = receptive_field[h]
            entity_representations[h] = self.entity_embedding(entities_at_layer)
        
        # Propagate through layers (from layer H-1 to 0)
        for h in range(self.n_iter - 1, -1, -1):
            entities_at_layer = receptive_field[h]
            num_entities = entities_at_layer.shape[0]
            
            neighbor_entities = torch.index_select(self.adj_entity, 0, entities_at_layer)
            neighbor_relations = torch.index_select(self.adj_relation, 0, entities_at_layer)
            
            current_entity_emb = entity_representations[h]
            
            neighbor_entity_emb = entity_representations[h + 1].view(
                num_entities, self.neighbor_sample_size, self.embedding_size
            )
            
            neighbor_relation_emb = self.relation_embedding(neighbor_relations)
            
            if num_entities > batch_size:
                repeat_factor = num_entities // batch_size
                user_emb_expanded = user_emb.repeat_interleave(repeat_factor, dim=0)
                if num_entities % batch_size != 0:
                    remainder = num_entities % batch_size
                    user_emb_expanded = torch.cat([
                        user_emb_expanded,
                        user_emb[:remainder]
                    ], dim=0)
            else:
                user_emb_expanded = user_emb
            
            aggregated_emb = self._aggregate_neighbors(
                user_emb_expanded, current_entity_emb, 
                neighbor_entity_emb, neighbor_relation_emb, h
            )
            
            entity_representations[h] = aggregated_emb
        
        final_entity_emb = entity_representations[0][:batch_size]
        
        return final_entity_emb

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self._propagate(user_emb, item)
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    # DIFFERENT: KGCN uses BCE loss only; MCCLK adds multi-level contrastive losses (local + global)
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        
        scores = self.forward(user, item)
        
        bce_loss = self.bce_loss(scores, label.float())
        
        user_emb = self.user_embedding(user)
        item_entity_emb = self.entity_embedding(item)
        reg_loss = self.reg_weight * self.l2_loss(user_emb, item_entity_emb)
        
        loss = bce_loss + reg_loss
        
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        scores = self.forward(user, item)
        
        return torch.sigmoid(scores)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_size = user.shape[0]
        
        user_emb = self.user_embedding(user)
        
        all_items = torch.arange(self.n_items, dtype=torch.long, device=self.device)
        
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, self.n_items, -1)
        user_emb_expanded = user_emb_expanded.reshape(-1, self.embedding_size)
        
        items_expanded = all_items.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        
        item_emb = self._propagate(user_emb_expanded, items_expanded)
        
        scores = torch.sum(user_emb_expanded * item_emb, dim=1)
        scores = torch.sigmoid(scores)
        
        return scores
