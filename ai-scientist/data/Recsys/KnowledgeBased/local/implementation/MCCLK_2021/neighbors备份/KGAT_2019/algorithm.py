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
    # REUSE: Both models inherit from KnowledgeRecommender and use InputType.PAIRWISE

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

        # SIMILAR: Both models use entity and relation embeddings for KG-aware recommendation
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.relation_dim)
        
        # DIFFERENT: KGAT uses TransR projection matrices; MCCLK uses relation embeddings directly in attention.
        # NOTE: Do not add TransR/CKG triple loss branches unless the target paper explicitly requires them.
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

        # REUSE: Both models use BPRLoss and EmbLoss for training
        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['adj_entity', 'adj_relation']

    def _build_ckg(self, dataset):
        # DIFFERENT: KGAT builds CKG with sampled neighbors; MCCLK uses full graph with edge_index/edge_type
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        kg_graph = dataset.kg_graph(form='coo', value_field='relation_id')
        
        self.neighbor_sample_size = 8  # Fixed neighbor sampling size
        
        adj_entity = {}
        adj_relation = {}
        
        users = interaction_matrix.row
        items = interaction_matrix.col
        
        interact_relation_id = self.n_relations
        
        for user, item in zip(users, items):
            user_entity_id = self.n_items + user
            
            if user_entity_id not in adj_entity:
                adj_entity[user_entity_id] = []
                adj_relation[user_entity_id] = []
            adj_entity[user_entity_id].append(item)
            adj_relation[user_entity_id].append(interact_relation_id)
            
            if item not in adj_entity:
                adj_entity[item] = []
                adj_relation[item] = []
            adj_entity[item].append(user_entity_id)
            adj_relation[item].append(interact_relation_id)
        
        head_entities = kg_graph.row
        tail_entities = kg_graph.col
        relations = kg_graph.data
        
        for h, r, t in zip(head_entities, relations, tail_entities):
            if h not in adj_entity:
                adj_entity[h] = []
                adj_relation[h] = []
            adj_entity[h].append(t)
            adj_relation[h].append(r)
        
        max_entity_id = max(self.n_entities, self.n_items + self.n_users)
        self.adj_entity = np.zeros((max_entity_id, self.neighbor_sample_size), dtype=np.int64)
        self.adj_relation = np.zeros((max_entity_id, self.neighbor_sample_size), dtype=np.int64)
        
        for entity_id in range(max_entity_id):
            if entity_id in adj_entity:
                neighbors = adj_entity[entity_id]
                relations = adj_relation[entity_id]
                
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
                self.adj_entity[entity_id] = [entity_id] * self.neighbor_sample_size
                self.adj_relation[entity_id] = [0] * self.neighbor_sample_size
        
        self.adj_entity = torch.LongTensor(self.adj_entity).to(self.device)
        self.adj_relation = torch.LongTensor(self.adj_relation).to(self.device)

    def _get_ego_embeddings(self):
        return self.entity_embedding.weight

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h)
        pos_t_e = self.entity_embedding(pos_t)
        neg_t_e = self.entity_embedding(neg_t)
        r_e = self.relation_embedding(r)
        
        r_trans_w = self.trans_w(r).view(
            -1, self.embedding_size, self.relation_dim
        )
        
        h_e_projected = torch.bmm(
            h_e.unsqueeze(1), r_trans_w
        ).squeeze(1)
        pos_t_e_projected = torch.bmm(
            pos_t_e.unsqueeze(1), r_trans_w
        ).squeeze(1)
        neg_t_e_projected = torch.bmm(
            neg_t_e.unsqueeze(1), r_trans_w
        ).squeeze(1)
        
        return h_e_projected, r_e, pos_t_e_projected, neg_t_e_projected

    def _compute_attention(self, entity_emb, neighbor_emb, relation_emb, trans_w):
        # DIFFERENT: KGAT uses TransR-based attention; MCCLK uses simpler h*r*t attention in Aggregator
        batch_size = entity_emb.size(0)
        
        entity_projected = torch.matmul(
            entity_emb.unsqueeze(1).expand(-1, self.neighbor_sample_size, -1).unsqueeze(2),
            trans_w
        ).squeeze(2)
        
        neighbor_projected = torch.matmul(
            neighbor_emb.unsqueeze(2), trans_w
        ).squeeze(2)
        
        attention_scores = torch.sum(
            neighbor_projected * torch.tanh(entity_projected + relation_emb),
            dim=-1
        )
        
        attention = F.softmax(attention_scores, dim=1)
        
        return attention

    def _aggregate_neighbors(self, entity_emb, neighbor_emb, attention):
        ego_network_emb = torch.sum(
            neighbor_emb * attention.unsqueeze(-1), dim=1
        )
        
        return ego_network_emb

    def _apply_aggregator(self, entity_emb, ego_network_emb, layer_idx):
        # DIFFERENT: KGAT has multiple aggregator types; MCCLK uses scatter_mean aggregation
        if self.aggregator_type == 'gcn':
            aggregated = self.aggregator_layers[layer_idx](
                entity_emb + ego_network_emb
            )
            aggregated = F.leaky_relu(aggregated)
            
        elif self.aggregator_type == 'graphsage':
            concatenated = torch.cat([entity_emb, ego_network_emb], dim=1)
            aggregated = self.aggregator_layers[layer_idx](concatenated)
            aggregated = F.leaky_relu(aggregated)
            
        elif self.aggregator_type == 'bi-interaction':
            sum_part = self.aggregator_layers[layer_idx]['w1'](
                entity_emb + ego_network_emb
            )
            product_part = self.aggregator_layers[layer_idx]['w2'](
                entity_emb * ego_network_emb
            )
            aggregated = F.leaky_relu(sum_part) + F.leaky_relu(product_part)
        
        return aggregated

    def _propagate(self, entity_ids):
        batch_size = entity_ids.size(0)
        
        entity_emb_list = [self.entity_embedding(entity_ids)]
        
        for layer in range(self.n_layers):
            entity_emb = entity_emb_list[-1]
            
            neighbor_entities = self.adj_entity[entity_ids]
            neighbor_relations = self.adj_relation[entity_ids]
            
            neighbor_emb = self.entity_embedding(neighbor_entities)
            
            relation_emb = self.relation_embedding(neighbor_relations)
            
            trans_w = self.trans_w(neighbor_relations).view(
                batch_size, self.neighbor_sample_size, 
                self.embedding_size, self.relation_dim
            )
            
            attention = self._compute_attention(
                entity_emb, neighbor_emb, relation_emb, trans_w
            )
            
            ego_network_emb = self._aggregate_neighbors(
                entity_emb, neighbor_emb, attention
            )
            
            new_entity_emb = self._apply_aggregator(
                entity_emb, ego_network_emb, layer
            )
            
            entity_emb_list.append(new_entity_emb)
        
        return entity_emb_list

    def forward(self, user, item):
        user_entity_ids = user + self.n_items
        
        user_emb_list = self._propagate(user_entity_ids)
        item_emb_list = self._propagate(item)
        
        # SIMILAR: Both models concatenate/aggregate multi-layer embeddings for final representation
        user_emb = torch.cat(user_emb_list, dim=1)
        item_emb = torch.cat(item_emb_list, dim=1)
        
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores

    def calculate_rs_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        pos_score = self.forward(user, pos_item)
        neg_score = self.forward(user, neg_item)
        
        rec_loss = self.rec_loss(pos_score, neg_score)
        
        return rec_loss

    def calculate_kg_loss(self, interaction):
        # DIFFERENT: KGAT has explicit KG loss with TransR; MCCLK has no separate KG loss
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]
        
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        
        pos_score = torch.sum((h_e + r_e - pos_t_e) ** 2, dim=1)
        neg_score = torch.sum((h_e + r_e - neg_t_e) ** 2, dim=1)
        
        kg_loss = self.kg_loss(neg_score, pos_score)
        
        return kg_loss

    def calculate_loss(self, interaction):
        # DIFFERENT: KGAT combines rec_loss + kg_loss; MCCLK uses rec_loss + contrastive losses
        rec_loss = self.calculate_rs_loss(interaction)
        
        kg_loss = self.calculate_kg_loss(interaction)
        
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_entity_ids = user + self.n_items
        user_emb = self.entity_embedding(user_entity_ids)
        pos_item_emb = self.entity_embedding(pos_item)
        neg_item_emb = self.entity_embedding(neg_item)
        
        reg_loss = self.reg_loss(user_emb, pos_item_emb, neg_item_emb)
        
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
        
        loss = rec_loss + kg_loss + self.reg_weight * reg_loss
        
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_size = user.size(0)
        
        user_entity_ids = user + self.n_items
        
        user_emb_list = self._propagate(user_entity_ids)
        user_emb = torch.cat(user_emb_list, dim=1)
        
        all_item_ids = torch.arange(self.n_items, dtype=torch.long, device=self.device)
        all_item_emb_list = self._propagate(all_item_ids)
        all_item_emb = torch.cat(all_item_emb_list, dim=1)
        
        scores = torch.matmul(user_emb, all_item_emb.t())
        
        return scores.view(-1)
