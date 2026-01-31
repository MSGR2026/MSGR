# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : SimpleX Implementation
# @Email  : simplex@example.com

r"""
SimpleX
################################################
Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

Reference code:
    https://github.com/xue-pai/Open-CF-Benchmark
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class SimpleX(GeneralRecommender):
    r"""SimpleX is a simple and strong baseline for collaborative filtering.
    
    SimpleX combines user embeddings with aggregated behavior history using three
    aggregation strategies: average pooling, self-attention, and user-attention.
    It uses Cosine Contrastive Loss (CCL) for training.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]  # m in CCL loss
        self.negative_weight = config["negative_weight"]  # w in CCL loss
        self.gamma = config["gamma"]  # g in fusion equation
        self.aggregator = config["aggregator"]  # 'mean', 'self_attention', 'user_attention'
        self.reg_weight = config["reg_weight"]  # regularization weight
        self.history_len = config["history_len"]  # K: max history length

        # define layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # fusion transform V
        self.behavior_transform = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        
        # attention parameters
        if self.aggregator == "self_attention":
            # global query vector q
            self.attention_query = nn.Parameter(torch.zeros(self.embedding_size))
            self.attention_W = nn.Linear(self.embedding_size, self.embedding_size)
        elif self.aggregator == "user_attention":
            self.attention_W = nn.Linear(self.embedding_size, self.embedding_size)
        
        # regularization loss
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # build history interaction data
        self._build_history_items(dataset)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        if self.aggregator == "self_attention":
            nn.init.normal_(self.attention_query, mean=0, std=0.1)
        
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def _build_history_items(self, dataset):
        r"""Build history item sequences for all users.
        
        For each user, we store their interacted items (up to history_len).
        """
        # Get interaction matrix
        inter_matrix = dataset.inter_matrix(form="csr")
        
        history_item_id = []
        history_item_len = []
        
        for user_id in range(self.n_users):
            # Get items interacted by this user
            items = inter_matrix[user_id].indices
            
            if len(items) > self.history_len:
                # Sample or truncate to history_len
                items = items[:self.history_len]
            
            history_item_len.append(len(items))
            
            # Pad to history_len
            padded_items = np.zeros(self.history_len, dtype=np.int64)
            if len(items) > 0:
                padded_items[:len(items)] = items
            history_item_id.append(padded_items)
        
        self.history_item_id = torch.LongTensor(np.array(history_item_id)).to(self.device)
        self.history_item_len = torch.LongTensor(history_item_len).to(self.device)

    def aggregate_behavior(self, user):
        r"""Aggregate user behavior history.
        
        Args:
            user: user indices [batch_size]
            
        Returns:
            aggregated behavior vector [batch_size, embedding_size]
        """
        # Get history items for users [batch_size, history_len]
        history_items = self.history_item_id[user]
        history_len = self.history_item_len[user]  # [batch_size]
        
        # Get item embeddings [batch_size, history_len, embedding_size]
        history_embeddings = self.item_embedding(history_items)
        
        # Create mask [batch_size, history_len]
        # mask = 1 for valid items, 0 for padding
        mask = torch.arange(self.history_len, device=self.device).unsqueeze(0) < history_len.unsqueeze(1)
        mask = mask.float()
        
        if self.aggregator == "mean":
            # Average pooling
            # [batch_size, history_len, 1]
            mask_expanded = mask.unsqueeze(-1)
            # Sum of embeddings
            sum_embeddings = (history_embeddings * mask_expanded).sum(dim=1)
            # Normalize by actual length
            history_len_safe = history_len.float().clamp(min=1).unsqueeze(-1)
            aggregated = sum_embeddings / history_len_safe
            
        elif self.aggregator == "self_attention":
            # Self-attention with global query
            # beta_k = q^T tanh(W_1 * e_k + b_1)
            transformed = torch.tanh(self.attention_W(history_embeddings))  # [batch_size, history_len, embedding_size]
            beta = torch.matmul(transformed, self.attention_query)  # [batch_size, history_len]
            
            # Apply mask (set padding to -inf for softmax)
            beta = beta.masked_fill(~mask.bool(), float('-inf'))
            
            # Softmax to get attention weights
            alpha = F.softmax(beta, dim=1)  # [batch_size, history_len]
            
            # Handle all-padding case
            alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
            
            # Weighted sum
            aggregated = (history_embeddings * alpha.unsqueeze(-1)).sum(dim=1)
            
        elif self.aggregator == "user_attention":
            # User-attention with user-specific query
            user_emb = self.user_embedding(user)  # [batch_size, embedding_size]
            
            # beta_k = e_u^T tanh(W_2 * e_k + b_2)
            transformed = torch.tanh(self.attention_W(history_embeddings))  # [batch_size, history_len, embedding_size]
            beta = torch.bmm(transformed, user_emb.unsqueeze(-1)).squeeze(-1)  # [batch_size, history_len]
            
            # Apply mask
            beta = beta.masked_fill(~mask.bool(), float('-inf'))
            
            # Softmax
            alpha = F.softmax(beta, dim=1)
            alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
            
            # Weighted sum
            aggregated = (history_embeddings * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        return aggregated

    def get_user_embedding(self, user):
        r"""Get final user representation by fusing user embedding and behavior aggregation.
        
        h_u = g * e_u + (1 - g) * V * p_u
        
        Args:
            user: user indices [batch_size]
            
        Returns:
            user representation [batch_size, embedding_size]
        """
        # Get user embedding
        user_emb = self.user_embedding(user)  # [batch_size, embedding_size]
        
        # Get aggregated behavior
        behavior_emb = self.aggregate_behavior(user)  # [batch_size, embedding_size]
        
        # Transform behavior embedding
        transformed_behavior = self.behavior_transform(behavior_emb)
        
        # Fuse
        user_repr = self.gamma * user_emb + (1 - self.gamma) * transformed_behavior
        
        return user_repr

    def forward(self, user):
        r"""Forward pass to get user and item embeddings.
        
        Args:
            user: user indices [batch_size]
            
        Returns:
            user_repr: user representation [batch_size, embedding_size]
        """
        return self.get_user_embedding(user)

    def cosine_similarity(self, user_emb, item_emb):
        r"""Calculate cosine similarity between user and item embeddings.
        
        Args:
            user_emb: [batch_size, embedding_size] or [batch_size, n_neg, embedding_size]
            item_emb: [batch_size, embedding_size] or [batch_size, n_neg, embedding_size]
            
        Returns:
            similarity scores
        """
        user_norm = F.normalize(user_emb, p=2, dim=-1)
        item_norm = F.normalize(item_emb, p=2, dim=-1)
        
        return (user_norm * item_norm).sum(dim=-1)

    def calculate_ccl_loss(self, user, pos_item, neg_item):
        r"""Calculate Cosine Contrastive Loss.
        
        L_CCL(u, i) = (1 - y_ui) + (w / |N|) * sum_j max(0, y_uj - m)
        
        Args:
            user: user indices [batch_size]
            pos_item: positive item indices [batch_size]
            neg_item: negative item indices [batch_size] or [batch_size, n_neg]
            
        Returns:
            CCL loss
        """
        # Get user representation
        user_repr = self.get_user_embedding(user)  # [batch_size, embedding_size]
        
        # Get item embeddings
        pos_item_emb = self.item_embedding(pos_item)  # [batch_size, embedding_size]
        
        # Positive similarity
        pos_sim = self.cosine_similarity(user_repr, pos_item_emb)  # [batch_size]
        
        # Positive loss: (1 - y_ui)
        pos_loss = (1 - pos_sim).mean()
        
        # Handle negative items
        if neg_item.dim() == 1:
            neg_item = neg_item.unsqueeze(1)  # [batch_size, 1]
        
        n_neg = neg_item.size(1)
        neg_item_emb = self.item_embedding(neg_item)  # [batch_size, n_neg, embedding_size]
        
        # Expand user representation for negative samples
        user_repr_expanded = user_repr.unsqueeze(1).expand(-1, n_neg, -1)  # [batch_size, n_neg, embedding_size]
        
        # Negative similarities
        neg_sim = self.cosine_similarity(user_repr_expanded, neg_item_emb)  # [batch_size, n_neg]
        
        # Negative loss with margin: max(0, y_uj - m)
        neg_loss_per_sample = F.relu(neg_sim - self.margin)  # [batch_size, n_neg]
        neg_loss = neg_loss_per_sample.mean()  # average over batch and negatives
        
        # Total CCL loss
        ccl_loss = pos_loss + self.negative_weight * neg_loss
        
        return ccl_loss

    def calculate_loss(self, interaction):
        # Clear storage variables when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # CCL loss
        ccl_loss = self.calculate_ccl_loss(user, pos_item, neg_item)

        # Regularization loss on ego embeddings
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
        )

        loss = ccl_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_repr = self.get_user_embedding(user)
        item_emb = self.item_embedding(item)
        
        scores = self.cosine_similarity(user_repr, item_emb)
        
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        if self.restore_user_e is None or self.restore_item_e is None:
            # Compute all user representations
            all_users = torch.arange(self.n_users, device=self.device)
            self.restore_user_e = self.get_user_embedding(all_users)
            self.restore_item_e = self.item_embedding.weight
        
        # Get user embeddings
        u_embeddings = self.restore_user_e[user]
        
        # Normalize for cosine similarity
        u_embeddings_norm = F.normalize(u_embeddings, p=2, dim=-1)
        i_embeddings_norm = F.normalize(self.restore_item_e, p=2, dim=-1)
        
        # Compute cosine similarity with all items
        scores = torch.matmul(u_embeddings_norm, i_embeddings_norm.transpose(0, 1))

        return scores.view(-1)