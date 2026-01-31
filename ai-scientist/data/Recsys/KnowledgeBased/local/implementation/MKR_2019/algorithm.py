# -*- coding: utf-8 -*-
# @Time   : 2024/01/01
# @Author : MKR Implementation
# @Email  : mkr@example.com

r"""
MKR
##################################################
Reference:
    Hongwei Wang et al. "Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation." 
    in WWW 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class CrossCompressUnit(nn.Module):
    r"""Cross&Compress Unit for feature interaction between items and entities.
    
    The unit performs:
    1. Cross operation: C_l = v_l * e_l^T (outer product creating d×d interaction matrix)
    2. Compress operation: project C_l back to d-dimensional space using 4 weight vectors
    """
    
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        
        # Four weight vectors for compression: VV, EV, VE, EE
        self.weight_vv = nn.Parameter(torch.Tensor(dim))
        self.weight_ev = nn.Parameter(torch.Tensor(dim))
        self.weight_ve = nn.Parameter(torch.Tensor(dim))
        self.weight_ee = nn.Parameter(torch.Tensor(dim))
        
        # Bias vectors
        self.bias_v = nn.Parameter(torch.Tensor(dim))
        self.bias_e = nn.Parameter(torch.Tensor(dim))
        
        # Initialize parameters
        nn.init.xavier_normal_(self.weight_vv.unsqueeze(0))
        nn.init.xavier_normal_(self.weight_ev.unsqueeze(0))
        nn.init.xavier_normal_(self.weight_ve.unsqueeze(0))
        nn.init.xavier_normal_(self.weight_ee.unsqueeze(0))
        nn.init.zeros_(self.bias_v)
        nn.init.zeros_(self.bias_e)
    
    def forward(self, v, e):
        r"""Forward pass of cross&compress unit.
        
        Args:
            v: item feature vector [batch_size, dim]
            e: entity feature vector [batch_size, dim]
            
        Returns:
            v_next: next layer item feature [batch_size, dim]
            e_next: next layer entity feature [batch_size, dim]
        """
        # Cross operation: C = v * e^T, shape: [batch_size, dim, dim]
        # We compute this implicitly through the compress operations
        
        # Compress for item: v_{l+1} = C_l * w_vv + C_l^T * w_ev + b_v
        #                            = (v * e^T) * w_vv + (e * v^T) * w_ev + b_v
        #                            = v * (e^T * w_vv) + e * (v^T * w_ev) + b_v
        v_next = v * torch.matmul(e, self.weight_vv) + e * torch.matmul(v, self.weight_ev) + self.bias_v
        
        # Compress for entity: e_{l+1} = C_l * w_ve + C_l^T * w_ee + b_e
        #                              = (v * e^T) * w_ve + (e * v^T) * w_ee + b_e
        #                              = v * (e^T * w_ve) + e * (v^T * w_ee) + b_e
        e_next = v * torch.matmul(e, self.weight_ve) + e * torch.matmul(v, self.weight_ee) + self.bias_e
        
        return v_next, e_next


class MKR(KnowledgeRecommender):
    r"""MKR is a multi-task learning framework that combines recommendation and knowledge graph embedding.
    
    It uses cross&compress units to bridge the two tasks and enable knowledge transfer between items
    and entities. The model consists of:
    1. Recommendation module: processes user-item interactions
    2. KGE module: processes knowledge graph triples
    3. Cross&compress units: enable feature interaction and knowledge transfer
    """
    
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(MKR, self).__init__(config, dataset)
        
        # Load parameters
        self.embedding_size = config['embedding_size']
        self.L = config['low_layers_num']  # Number of cross&compress layers
        self.H = config['high_layers_num']  # Number of high layers in RS
        self.K = config['high_layers_num']  # Number of high layers in KGE (use same config)
        self.use_inner_product = config['use_inner_product']  # Whether to use inner product for prediction
        self.kg_weight = config['kg_weight']  # λ_1 in loss function
        self.reg_weight = config['reg_weight']  # λ_2 in loss function
        
        # User embedding and MLP layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_mlp = nn.ModuleList()
        input_dim = self.embedding_size
        for _ in range(self.L):
            self.user_mlp.append(nn.Linear(input_dim, self.embedding_size))
            input_dim = self.embedding_size
        
        # Item and entity embeddings
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        
        # Relation embedding and MLP layers
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.relation_mlp = nn.ModuleList()
        input_dim = self.embedding_size
        for _ in range(self.L):
            self.relation_mlp.append(nn.Linear(input_dim, self.embedding_size))
            input_dim = self.embedding_size
        
        # Cross&compress units (L layers)
        self.cc_units = nn.ModuleList()
        for _ in range(self.L):
            self.cc_units.append(CrossCompressUnit(self.embedding_size))
        
        # Prediction MLP for recommendation (H layers)
        if not self.use_inner_product:
            self.predict_mlp = nn.ModuleList()
            input_dim = self.embedding_size * 2
            for i in range(self.H):
                if i == self.H - 1:
                    self.predict_mlp.append(nn.Linear(input_dim, 1))
                else:
                    self.predict_mlp.append(nn.Linear(input_dim, self.embedding_size))
                    input_dim = self.embedding_size
        
        # Tail prediction MLP for KGE (K layers)
        self.tail_mlp = nn.ModuleList()
        input_dim = self.embedding_size * 2
        for i in range(self.K):
            if i == self.K - 1:
                self.tail_mlp.append(nn.Linear(input_dim, self.embedding_size))
            else:
                self.tail_mlp.append(nn.Linear(input_dim, self.embedding_size))
                input_dim = self.embedding_size
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()
        
        # Apply initialization
        self.apply(xavier_normal_initialization)
    
    def _extract_user_feature(self, user):
        r"""Extract user latent feature through L-layer MLP.
        
        Args:
            user: user id tensor [batch_size]
            
        Returns:
            user_emb: user latent feature [batch_size, embedding_size]
        """
        user_emb = self.user_embedding(user)
        for i in range(self.L):
            user_emb = self.user_mlp[i](user_emb)
            if i < self.L - 1:
                user_emb = F.relu(user_emb)
            else:
                user_emb = torch.tanh(user_emb)
        return user_emb
    
    def _extract_item_feature(self, item):
        r"""Extract item latent feature through L cross&compress units.
        
        Args:
            item: item id tensor [batch_size]
            
        Returns:
            item_emb: item latent feature [batch_size, embedding_size]
        """
        item_emb = self.item_embedding(item)
        # Items are entities, so we use entity embedding as associated entity
        entity_emb = self.entity_embedding(item)
        
        for i in range(self.L):
            item_emb, entity_emb = self.cc_units[i](item_emb, entity_emb)
            if i < self.L - 1:
                item_emb = F.relu(item_emb)
                entity_emb = F.relu(entity_emb)
            else:
                item_emb = torch.tanh(item_emb)
                entity_emb = torch.tanh(entity_emb)
        
        return item_emb
    
    def _extract_head_relation_feature(self, head, relation):
        r"""Extract head and relation latent features for KGE.
        
        Args:
            head: head entity id tensor [batch_size]
            relation: relation id tensor [batch_size]
            
        Returns:
            head_emb: head latent feature [batch_size, embedding_size]
            relation_emb: relation latent feature [batch_size, embedding_size]
        """
        # Extract head feature through cross&compress units
        # Head entities may correspond to items
        head_emb = self.entity_embedding(head)
        # For simplicity, we use item embedding if head < n_items, else use entity embedding
        item_emb = torch.where(
            (head < self.n_items).unsqueeze(1),
            self.item_embedding(torch.clamp(head, max=self.n_items-1)),
            head_emb
        )
        
        for i in range(self.L):
            item_emb, head_emb = self.cc_units[i](item_emb, head_emb)
            if i < self.L - 1:
                item_emb = F.relu(item_emb)
                head_emb = F.relu(head_emb)
            else:
                item_emb = torch.tanh(item_emb)
                head_emb = torch.tanh(head_emb)
        
        # Extract relation feature through MLP
        relation_emb = self.relation_embedding(relation)
        for i in range(self.L):
            relation_emb = self.relation_mlp[i](relation_emb)
            if i < self.L - 1:
                relation_emb = F.relu(relation_emb)
            else:
                relation_emb = torch.tanh(relation_emb)
        
        return head_emb, relation_emb
    
    def _predict_tail(self, head_emb, relation_emb):
        r"""Predict tail entity from head and relation features.
        
        Args:
            head_emb: head latent feature [batch_size, embedding_size]
            relation_emb: relation latent feature [batch_size, embedding_size]
            
        Returns:
            tail_pred: predicted tail feature [batch_size, embedding_size]
        """
        # Concatenate head and relation
        concat_emb = torch.cat([head_emb, relation_emb], dim=1)
        
        # K-layer MLP
        for i in range(self.K):
            concat_emb = self.tail_mlp[i](concat_emb)
            if i < self.K - 1:
                concat_emb = F.relu(concat_emb)
            else:
                concat_emb = torch.tanh(concat_emb)
        
        return concat_emb
    
    def forward_rs(self, user, item):
        r"""Forward pass for recommendation task.
        
        Args:
            user: user id tensor [batch_size]
            item: item id tensor [batch_size]
            
        Returns:
            score: predicted score [batch_size]
        """
        user_emb = self._extract_user_feature(user)
        item_emb = self._extract_item_feature(item)
        
        if self.use_inner_product:
            # Inner product
            score = torch.sum(user_emb * item_emb, dim=1)
        else:
            # H-layer MLP
            concat_emb = torch.cat([user_emb, item_emb], dim=1)
            for i in range(self.H):
                concat_emb = self.predict_mlp[i](concat_emb)
                if i < self.H - 1:
                    concat_emb = F.relu(concat_emb)
            score = concat_emb.squeeze(1)
        
        return score
    
    def forward_kg(self, head, relation, tail):
        r"""Forward pass for KGE task.
        
        Args:
            head: head entity id tensor [batch_size]
            relation: relation id tensor [batch_size]
            tail: tail entity id tensor [batch_size]
            
        Returns:
            score: similarity score between predicted and actual tail [batch_size]
        """
        head_emb, relation_emb = self._extract_head_relation_feature(head, relation)
        tail_pred = self._predict_tail(head_emb, relation_emb)
        tail_emb = self.entity_embedding(tail)
        
        # Normalized inner product
        tail_pred = F.normalize(tail_pred, p=2, dim=1)
        tail_emb = F.normalize(tail_emb, p=2, dim=1)
        score = torch.sum(tail_pred * tail_emb, dim=1)
        
        return score
    
    def calculate_rs_loss(self, interaction):
        r"""Calculate recommendation loss.
        
        Args:
            interaction: interaction data
            
        Returns:
            rs_loss: recommendation loss
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        
        score = self.forward_rs(user, item)
        rs_loss = self.bce_loss(score, label.float())
        
        return rs_loss
    
    def calculate_kg_loss(self, interaction):
        r"""Calculate KGE loss.
        
        Args:
            interaction: interaction data containing KG triples
            
        Returns:
            kg_loss: knowledge graph embedding loss
        """
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]
        
        pos_score = self.forward_kg(head, relation, pos_tail)
        neg_score = self.forward_kg(head, relation, neg_tail)
        
        # Maximize score difference: score(h,r,t) - score(h',r,t')
        kg_loss = -torch.mean(pos_score - neg_score)
        
        return kg_loss
    
    def calculate_loss(self, interaction):
        r"""Calculate total loss combining RS and KG losses.
        
        Args:
            interaction: interaction data
            
        Returns:
            total_loss: combined loss
        """
        # Check if this is RS or KG interaction
        if self.LABEL in interaction:
            # Recommendation task
            rs_loss = self.calculate_rs_loss(interaction)
            
            # Regularization
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
            user_emb = self.user_embedding(user)
            item_emb = self.item_embedding(item)
            entity_emb = self.entity_embedding(item)
            
            reg_loss = self.reg_loss(user_emb, item_emb, entity_emb)
            
            total_loss = rs_loss + self.reg_weight * reg_loss
            
        else:
            # KGE task
            kg_loss = self.calculate_kg_loss(interaction)
            
            # Regularization
            head = interaction[self.HEAD_ENTITY_ID]
            relation = interaction[self.RELATION_ID]
            pos_tail = interaction[self.TAIL_ENTITY_ID]
            
            head_emb = self.entity_embedding(head)
            relation_emb = self.relation_embedding(relation)
            tail_emb = self.entity_embedding(pos_tail)
            
            reg_loss = self.reg_loss(head_emb, relation_emb, tail_emb)
            
            total_loss = self.kg_weight * kg_loss + self.reg_weight * reg_loss
        
        return total_loss
    
    def predict(self, interaction):
        r"""Predict for given user-item pairs.
        
        Args:
            interaction: interaction data
            
        Returns:
            score: predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward_rs(user, item)
        return torch.sigmoid(score)
    
    def full_sort_predict(self, interaction):
        r"""Predict for all items for given users.
        
        Args:
            interaction: interaction data
            
        Returns:
            score: predicted scores for all items [batch_size * n_items]
        """
        user = interaction[self.USER_ID]
        user_emb = self._extract_user_feature(user)
        
        # Extract all item embeddings
        all_item_ids = torch.arange(self.n_items, device=self.device)
        all_item_emb = self._extract_item_feature(all_item_ids)
        
        if self.use_inner_product:
            # [batch_size, embedding_size] @ [n_items, embedding_size]^T
            score = torch.matmul(user_emb, all_item_emb.t())
        else:
            # Expand user embedding to match all items
            batch_size = user_emb.size(0)
            user_emb_expanded = user_emb.unsqueeze(1).expand(batch_size, self.n_items, self.embedding_size)
            all_item_emb_expanded = all_item_emb.unsqueeze(0).expand(batch_size, self.n_items, self.embedding_size)
            
            # Concatenate and pass through MLP
            concat_emb = torch.cat([user_emb_expanded, all_item_emb_expanded], dim=2)
            concat_emb = concat_emb.view(-1, self.embedding_size * 2)
            
            for i in range(self.H):
                concat_emb = self.predict_mlp[i](concat_emb)
                if i < self.H - 1:
                    concat_emb = F.relu(concat_emb)
            
            score = concat_emb.view(batch_size, self.n_items)
        
        score = torch.sigmoid(score)
        return score.view(-1)