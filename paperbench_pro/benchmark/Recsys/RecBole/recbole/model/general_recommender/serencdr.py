# -*- coding: utf-8 -*-
# @Time   : 2025/1/6
# @Author : Integration Team

r"""
SerenCDR
################################################
Reference:
    Fu, Z., Niu, X., Wu, X. and Rahman, R., 2025. 
    "A deep learning model for cross-domain serendipity recommendations." 
    ACM Transactions on Recommender Systems, 3(3), pp.1-21.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class DomainSpecific(nn.Module):
    """Domain-specific knowledge learning module"""
    
    def __init__(self, input_dim, hidden_dim):
        super(DomainSpecific, self).__init__()
        self.user_layer = nn.Linear(input_dim, hidden_dim)
        self.item_layer = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, user_emb, item_emb):
        user_sp = F.relu(self.user_layer(user_emb))
        item_sp = F.relu(self.item_layer(item_emb))
        return user_sp, item_sp


class DomainSharing(nn.Module):
    """Domain-sharing knowledge learning module"""
    
    def __init__(self, input_dim, num_groups, feature_dim, l2_reg=0.01):
        super(DomainSharing, self).__init__()
        # Group projection layers
        self.source_proj = nn.Linear(input_dim, num_groups)
        self.target_proj = nn.Linear(input_dim, num_groups)
        
        # Shared knowledge matrix
        self.shared_knowledge = nn.Parameter(torch.randn(num_groups, feature_dim))
        
        # Linear transformation for target domain
        self.linear_transform = nn.Parameter(torch.randn(feature_dim, feature_dim))
        
        self.l2_reg = l2_reg
        
    def forward(self, source_emb, target_emb):
        # Project to groups
        p_s = F.relu(self.source_proj(source_emb))
        p_t = F.relu(self.target_proj(target_emb))
        
        # Source domain sharing
        source_sh = torch.matmul(p_s, self.shared_knowledge)
        
        # Target domain sharing with transformation
        transformed_knowledge = torch.matmul(self.shared_knowledge, self.linear_transform)
        target_sh = torch.matmul(p_t, transformed_knowledge)
        
        return source_sh, target_sh
    
    def get_regularization_loss(self):
        """Get L2 regularization loss for shared parameters"""
        reg_loss = self.l2_reg * (
            torch.norm(self.shared_knowledge) + 
            torch.norm(self.linear_transform)
        )
        return reg_loss


class SerenCDR(GeneralRecommender):
    r"""SerenCDR is a deep learning model for cross-domain serendipity recommendations.
    
    It uses domain-specific and domain-sharing knowledge learning modules to capture
    serendipity-oriented user preferences across domains.
    """
    
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SerenCDR, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.num_user_groups = config["num_user_groups"]
        self.num_item_groups = config["num_item_groups"]
        self.feature_dim = config["feature_dim"]
        self.l2_reg = config["l2_reg"]
        self.aux_loss_weight = config["aux_loss_weight"]
        self.use_cross_domain = config.get("use_cross_domain", False)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # Domain-specific modules
        self.source_specific = DomainSpecific(self.embedding_size, self.hidden_size)
        self.target_specific = DomainSpecific(self.embedding_size, self.feature_dim)
        
        # Domain-sharing modules
        self.user_sharing = DomainSharing(
            self.embedding_size, self.num_user_groups, self.feature_dim, self.l2_reg
        )
        self.item_sharing = DomainSharing(
            self.embedding_size, self.num_item_groups, self.feature_dim, self.l2_reg
        )
        
        # Loss function
        self.bpr_loss = BPRLoss()
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        """Get user embedding"""
        return self.user_embedding(user)
    
    def get_item_embedding(self, item):
        """Get item embedding"""
        return self.item_embedding(item)

    def forward(self, user, item, domain='target'):
        """
        Forward propagation
        
        Args:
            user: user id tensor
            item: item id tensor
            domain: 'source' or 'target' domain
            
        Returns:
            Fused user and item representations
        """
        user_emb = self.get_user_embedding(user)
        item_emb = self.get_item_embedding(item)
        
        if domain == 'source':
            # Source domain processing
            user_sp, item_sp = self.source_specific(user_emb, item_emb)
        else:
            # Target domain processing (default)
            user_sp, item_sp = self.target_specific(user_emb, item_emb)
        
        # Domain-sharing knowledge
        user_sh, _ = self.user_sharing(user_emb, user_emb)
        item_sh, _ = self.item_sharing(item_emb, item_emb)
        
        # Fusion: concatenate domain-specific and domain-sharing
        # Shape: [batch_size, 1, feature_dim] for broadcasting
        user_sh = user_sh.unsqueeze(1)
        user_sp = user_sp.unsqueeze(1)
        item_sh = item_sh.unsqueeze(1)
        item_sp = item_sp.unsqueeze(1)
        
        # Concatenate along feature dimension
        user_fused = torch.cat([user_sh, user_sp], dim=1)  # [batch, 2, feature_dim]
        item_fused = torch.cat([item_sh, item_sp], dim=1)  # [batch, 2, feature_dim]
        
        return user_fused, item_fused
    
    def calculate_serendipity_score(self, user_fused, item_fused):
        """
        Calculate serendipity score using matrix multiplication
        
        Args:
            user_fused: fused user representation [batch, 2, feature_dim]
            item_fused: fused item representation [batch, 2, feature_dim]
            
        Returns:
            Serendipity scores [batch]
        """
        # Matrix multiplication and sigmoid activation
        # [batch, 2, feature_dim] x [batch, feature_dim, 2] -> [batch, 2, 2]
        scores = torch.matmul(user_fused, item_fused.transpose(1, 2))
        scores = scores.reshape(scores.size(0), -1)  # [batch, 4]
        scores = torch.sigmoid(scores)
        
        # Average pooling across all combinations
        final_score = scores.mean(dim=1)  # [batch]
        
        return final_score

    def calculate_loss(self, interaction):
        """
        Calculate training loss including main loss and regularization
        
        Args:
            interaction: interaction batch data
            
        Returns:
            Total loss
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        # Forward pass for positive and negative items
        user_pos, pos_item_e = self.forward(user, pos_item)
        _, neg_item_e = self.forward(user, neg_item)
        
        # Calculate serendipity scores
        pos_score = self.calculate_serendipity_score(user_pos, pos_item_e)
        neg_score = self.calculate_serendipity_score(user_pos, neg_item_e)
        
        # BPR loss
        bpr_loss = self.bpr_loss(pos_score, neg_score)
        
        # Regularization loss from sharing modules
        reg_loss = (
            self.user_sharing.get_regularization_loss() +
            self.item_sharing.get_regularization_loss()
        )
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss

    def predict(self, interaction):
        """
        Predict scores for given user-item pairs
        
        Args:
            interaction: interaction batch data
            
        Returns:
            Predicted scores [batch]
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_e, item_e = self.forward(user, item)
        scores = self.calculate_serendipity_score(user_e, item_e)
        
        return scores

    def full_sort_predict(self, interaction):
        """
        Full ranking prediction for all items
        
        Args:
            interaction: interaction batch data
            
        Returns:
            Scores for all items [batch * n_items]
        """
        user = interaction[self.USER_ID]
        user_e, _ = self.forward(user, torch.zeros(1, dtype=torch.long, device=user.device))
        
        # Get all item embeddings
        all_item_ids = torch.arange(
            self.n_items, dtype=torch.long, device=user.device
        )
        
        # Batch process to avoid OOM
        batch_size = 1024
        all_scores = []
        
        for i in range(0, self.n_items, batch_size):
            end_idx = min(i + batch_size, self.n_items)
            batch_items = all_item_ids[i:end_idx]
            
            # Expand user embeddings to match batch items
            batch_user_e = user_e.repeat(len(batch_items), 1, 1)
            _, batch_item_e = self.forward(
                user.repeat(len(batch_items)), batch_items
            )
            
            batch_scores = self.calculate_serendipity_score(batch_user_e, batch_item_e)
            all_scores.append(batch_scores)
        
        scores = torch.cat(all_scores, dim=0)
        return scores.view(-1)

