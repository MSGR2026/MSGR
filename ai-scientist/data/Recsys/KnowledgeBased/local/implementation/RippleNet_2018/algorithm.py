# -*- coding: utf-8 -*-
# @Time   : 2024/01/01
# @Author : RippleNet Implementation
# @Email  : example@example.com

r"""
RippleNet
##################################################
Reference:
    Hongwei Wang et al. "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems." in CIKM 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class RippleNet(KnowledgeRecommender):
    r"""RippleNet is a knowledge-aware recommendation model that propagates user preferences on the knowledge graph.
    
    It uses a memory network-style approach to iteratively expand user interests through K-hop ripple sets,
    where each hop aggregates tail entities using attention weights computed from user and relation embeddings.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RippleNet, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_hop = config["n_hop"]  # Number of hops (K)
        self.n_memory = config["n_memory"]  # Max size of ripple set per hop
        self.reg_weight = config["reg_weight"]
        self.use_st_gumbel = config.get("use_st_gumbel", False)  # Straight-through Gumbel-Softmax
        
        # Define embeddings
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        
        # Transform matrices for each hop
        self.transform_matrix = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        
        # Define loss functions
        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Build ripple sets for all users
        self._build_ripple_sets(dataset)
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)
        
    def _build_ripple_sets(self, dataset):
        """Build ripple sets for each user by K-hop expansion from their historical items."""
        # Get user-item interaction matrix
        inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        
        # Get KG in COO format: (head, tail, relation)
        kg_graph = dataset.kg_graph(form='coo', value_field='relation_id')
        heads = kg_graph.row
        tails = kg_graph.col
        relations = kg_graph.data
        
        # Build adjacency list for efficient neighbor lookup
        # kg_dict: {head_entity: [(tail_entity, relation), ...]}
        kg_dict = {}
        for h, t, r in zip(heads, tails, relations):
            if h not in kg_dict:
                kg_dict[h] = []
            kg_dict[h].append((t, r))
        
        # Build ripple sets for each user
        # ripple_sets[user][hop] = (heads, relations, tails)
        ripple_sets = {}
        
        for user_id in range(self.n_users):
            # Get user's historical items (seed set)
            user_items = inter_matrix[user_id].indices.tolist()
            
            if len(user_items) == 0:
                # If user has no history, use empty ripple sets
                ripple_sets[user_id] = [
                    ([], [], []) for _ in range(self.n_hop)
                ]
                continue
            
            ripple_sets[user_id] = []
            
            # Start from user's historical items
            current_entities = user_items
            
            for hop in range(self.n_hop):
                hop_heads = []
                hop_relations = []
                hop_tails = []
                
                # For each entity in current hop, get its neighbors
                for entity in current_entities:
                    if entity in kg_dict:
                        neighbors = kg_dict[entity]
                        for tail, relation in neighbors:
                            hop_heads.append(entity)
                            hop_relations.append(relation)
                            hop_tails.append(tail)
                
                # Sample if exceeds n_memory
                if len(hop_heads) > self.n_memory:
                    indices = np.random.choice(len(hop_heads), self.n_memory, replace=False)
                    hop_heads = [hop_heads[i] for i in indices]
                    hop_relations = [hop_relations[i] for i in indices]
                    hop_tails = [hop_tails[i] for i in indices]
                
                # If no neighbors found, use self-loop
                if len(hop_heads) == 0:
                    hop_heads = current_entities[:self.n_memory]
                    hop_relations = [0] * len(hop_heads)
                    hop_tails = current_entities[:self.n_memory]
                
                ripple_sets[user_id].append((hop_heads, hop_relations, hop_tails))
                
                # Update current entities for next hop
                current_entities = list(set(hop_tails))
        
        # Convert to tensors and store
        self.ripple_sets = {}
        for user_id in range(self.n_users):
            self.ripple_sets[user_id] = []
            for hop in range(self.n_hop):
                heads, relations, tails = ripple_sets[user_id][hop]
                
                # Pad to n_memory if needed
                if len(heads) < self.n_memory:
                    pad_size = self.n_memory - len(heads)
                    heads = heads + [heads[0] if heads else 0] * pad_size
                    relations = relations + [0] * pad_size
                    tails = tails + [tails[0] if tails else 0] * pad_size
                
                self.ripple_sets[user_id].append((
                    torch.LongTensor(heads),
                    torch.LongTensor(relations),
                    torch.LongTensor(tails)
                ))
        
        # Move to device
        for user_id in range(self.n_users):
            for hop in range(self.n_hop):
                heads, relations, tails = self.ripple_sets[user_id][hop]
                self.ripple_sets[user_id][hop] = (
                    heads.to(self.device),
                    relations.to(self.device),
                    tails.to(self.device)
                )
    
    def _get_ripple_set_batch(self, users):
        """Get ripple sets for a batch of users.
        
        Args:
            users: [batch_size]
        
        Returns:
            List of tuples for each hop: (heads, relations, tails)
            Each element has shape [batch_size, n_memory]
        """
        batch_size = users.size(0)
        ripple_set_batch = []
        
        for hop in range(self.n_hop):
            hop_heads = []
            hop_relations = []
            hop_tails = []
            
            for user_id in users.cpu().numpy():
                heads, relations, tails = self.ripple_sets[int(user_id)][hop]
                hop_heads.append(heads)
                hop_relations.append(relations)
                hop_tails.append(tails)
            
            # Stack to [batch_size, n_memory]
            hop_heads = torch.stack(hop_heads, dim=0)
            hop_relations = torch.stack(hop_relations, dim=0)
            hop_tails = torch.stack(hop_tails, dim=0)
            
            ripple_set_batch.append((hop_heads, hop_relations, hop_tails))
        
        return ripple_set_batch
    
    def _key_addressing(self, user_emb, ripple_set):
        """Compute attention weights and aggregate tail entities.
        
        Args:
            user_emb: [batch_size, embedding_size]
            ripple_set: (heads, relations, tails)
                heads: [batch_size, n_memory]
                relations: [batch_size, n_memory]
                tails: [batch_size, n_memory]
        
        Returns:
            aggregated_emb: [batch_size, embedding_size]
        """
        heads, relations, tails = ripple_set
        batch_size = heads.size(0)
        
        # Get embeddings
        # [batch_size, n_memory, embedding_size]
        head_emb = self.entity_embedding(heads)
        relation_emb = self.relation_embedding(relations)
        tail_emb = self.entity_embedding(tails)
        
        # Compute attention scores using relation-user interaction
        # Expand user_emb to [batch_size, n_memory, embedding_size]
        user_emb_expanded = user_emb.unsqueeze(1).expand_as(relation_emb)
        
        # Compute attention: R^T * u
        # [batch_size, n_memory]
        attention_scores = torch.sum(relation_emb * user_emb_expanded, dim=2)
        
        # Apply softmax
        if self.use_st_gumbel:
            # Gumbel-Softmax for better gradient flow
            attention_weights = F.gumbel_softmax(attention_scores, tau=1.0, hard=False, dim=1)
        else:
            attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of tail embeddings
        # [batch_size, n_memory, 1]
        attention_weights_expanded = attention_weights.unsqueeze(2)
        
        # [batch_size, embedding_size]
        aggregated_emb = torch.sum(tail_emb * attention_weights_expanded, dim=1)
        
        return aggregated_emb
    
    def forward(self, users, items):
        """Forward propagation.
        
        Args:
            users: [batch_size]
            items: [batch_size]
        
        Returns:
            scores: [batch_size]
        """
        batch_size = users.size(0)
        
        # Get item embeddings (items are entities)
        item_emb = self.entity_embedding(items)  # [batch_size, embedding_size]
        
        # Get ripple sets for users
        ripple_set_batch = self._get_ripple_set_batch(users)
        
        # Initialize user representation as item embedding
        # This serves as the initial "query" for the memory network
        user_repr = item_emb
        
        # Iteratively propagate through K hops
        for hop in range(self.n_hop):
            # Get aggregated embedding from current ripple set
            aggregated_emb = self._key_addressing(user_repr, ripple_set_batch[hop])
            
            # Update user representation
            # Apply transformation and add to previous representation
            user_repr = self.transform_matrix(user_repr + aggregated_emb)
        
        # Compute final scores
        # [batch_size]
        scores = torch.sum(user_repr * item_emb, dim=1)
        
        return scores
    
    def calculate_loss(self, interaction):
        """Calculate loss for training.
        
        Args:
            interaction: Interaction object containing user, pos_item, neg_item
        
        Returns:
            total_loss: Combined recommendation loss and regularization loss
        """
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        
        # Get scores
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        
        # Compute BPR loss
        rec_loss = self.rec_loss(pos_scores, neg_scores)
        
        # Compute regularization loss
        # Regularize embeddings used in this batch
        pos_item_emb = self.entity_embedding(pos_items)
        neg_item_emb = self.entity_embedding(neg_items)
        
        # Get all relations and entities used in ripple sets
        ripple_set_batch = self._get_ripple_set_batch(users)
        
        reg_embeddings = [pos_item_emb, neg_item_emb]
        
        for hop in range(self.n_hop):
            heads, relations, tails = ripple_set_batch[hop]
            
            # Sample a subset for regularization to avoid memory issues
            sample_size = min(10, self.n_memory)
            sample_indices = torch.randint(0, self.n_memory, (sample_size,), device=self.device)
            
            sampled_heads = heads[:, sample_indices].reshape(-1)
            sampled_relations = relations[:, sample_indices].reshape(-1)
            sampled_tails = tails[:, sample_indices].reshape(-1)
            
            reg_embeddings.append(self.entity_embedding(sampled_heads))
            reg_embeddings.append(self.relation_embedding(sampled_relations))
            reg_embeddings.append(self.entity_embedding(sampled_tails))
        
        reg_loss = self.reg_weight * self.reg_loss(*reg_embeddings)
        
        total_loss = rec_loss + reg_loss
        
        return total_loss
    
    def predict(self, interaction):
        """Predict scores for user-item pairs.
        
        Args:
            interaction: Interaction object
        
        Returns:
            scores: Predicted scores
        """
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        return self.forward(users, items)
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users.
        
        Args:
            interaction: Interaction object containing users
        
        Returns:
            scores: [batch_size * n_items]
        """
        users = interaction[self.USER_ID]
        batch_size = users.size(0)
        
        # Get all item embeddings
        all_item_emb = self.entity_embedding.weight[:self.n_items]  # [n_items, embedding_size]
        
        # Get ripple sets for users
        ripple_set_batch = self._get_ripple_set_batch(users)
        
        # We need to compute scores for each user-item pair
        # To avoid memory issues, we'll compute in batches
        all_scores = []
        
        for item_idx in range(self.n_items):
            # Create batch of same item for all users
            items = torch.full((batch_size,), item_idx, dtype=torch.long, device=self.device)
            
            # Get item embeddings
            item_emb = self.entity_embedding(items)
            
            # Initialize user representation
            user_repr = item_emb
            
            # Propagate through K hops
            for hop in range(self.n_hop):
                aggregated_emb = self._key_addressing(user_repr, ripple_set_batch[hop])
                user_repr = self.transform_matrix(user_repr + aggregated_emb)
            
            # Compute scores
            scores = torch.sum(user_repr * item_emb, dim=1)
            all_scores.append(scores)
        
        # Stack and reshape
        all_scores = torch.stack(all_scores, dim=1)  # [batch_size, n_items]
        
        return all_scores.view(-1)