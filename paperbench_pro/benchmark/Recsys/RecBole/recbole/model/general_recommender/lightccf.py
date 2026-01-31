# -*- coding: utf-8 -*-
# @Time   : 2025/01/12
# @Author : AI-Scientist
# @Paper  : Unveiling Contrastive Learning's Capability of Neighborhood 
#           Aggregation for Collaborative Filtering (SIGIR'25)

r"""
recbole.model.general_recommender.lightccf
############################################

Reference:
    Yu Zhang et al. "Unveiling Contrastive Learning's Capability of 
    Neighborhood Aggregation for Collaborative Filtering." SIGIR 2025.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NeighborAggregateLoss(nn.Module):
    r"""Neighbor Aggregate Loss for LightCCF"""
    def __init__(self, tau=0.28):
        super(NeighborAggregateLoss, self).__init__()
        self.tau = tau
    
    def forward(self, user_embed, pos_item_embed):
        # Normalize embeddings (必须进行归一化)
        user_embed = F.normalize(user_embed, dim=1)
        pos_item_embed = F.normalize(pos_item_embed, dim=1)
        
        # 1. 计算分子：sim(u, i) / tau
        # (Batch Size,)
        pos_score = (user_embed * pos_item_embed).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.tau)
        
        # 2. 计算分母
        # 原始实现逻辑：total_score = exp(u*i^T) + exp(u*u^T)
        # 注意：这里计算的是当前 Batch 内所有用户和所有正样本物品的相似度
        user_item_scores = torch.matmul(user_embed, pos_item_embed.transpose(0, 1))
        user_user_scores = torch.matmul(user_embed, user_embed.transpose(0, 1))
        
        # 对应原始代码 losses.py 中的 get_neighbor_aggregate_loss
        total_score = user_item_scores + user_user_scores
        total_score = torch.exp(total_score / self.tau).sum(dim=1)
        
        # 3. 计算对比损失
        # 添加 1e-8 保持数值稳定
        na_loss = -torch.log(pos_score / (total_score + 1e-8))
        
        return torch.mean(na_loss)


class LightCCF(GeneralRecommender):
    r"""LightCCF: Light Contrastive Collaborative Filtering
    
    Code implemented based on: https://github.com/zzyuuuu/LightCCF
    """
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(LightCCF, self).__init__(config, dataset)
        
        # Load parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config.get('n_layers', 3)
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.ssl_weight = config.get('ssl_weight', 5.0) # 原始代码默认值为 5.0
        self.tau = config.get('tau', 0.28)              # 原始代码默认值为 0.28
        self.encoder = config.get('encoder', 'MF')
        self.require_pow = config.get('require_pow', False)
        
        # Define Embeddings
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users,
            embedding_dim=self.embedding_size
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=self.embedding_size
        )
        
        # Parameters initialization
        self.apply(xavier_uniform_initialization)
        
        # Get interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        
        # Build normalized adjacency matrix for GCN (if needed)
        if self.encoder == 'LightGCN':
            self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        # Loss functions
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.na_loss = NeighborAggregateLoss(tau=self.tau)
        
        # Storage for evaluation
        self.restore_user_e = None
        self.restore_item_e = None

    def get_norm_adj_mat(self):
        r"""Construct normalized adjacency matrix"""
        inter_M = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items
        
        row_user = inter_M.row
        col_item = inter_M.col + n_users
        data_user_item = np.ones(inter_M.nnz, dtype=np.float32)
        
        row_item = inter_M.col + n_users
        col_user = inter_M.row
        data_item_user = np.ones(inter_M.nnz, dtype=np.float32)
        
        row = np.concatenate([row_user, row_item])
        col = np.concatenate([col_item, col_user])
        data = np.concatenate([data_user_item, data_item_user])
        
        A = sp.coo_matrix(
            (data, (row, col)),
            shape=(n_users + n_items, n_users + n_items),
            dtype=np.float32
        )
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        
        L = D @ A @ D
        L = sp.coo_matrix(L)
        
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        
        return SparseL
    
    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self):
        if self.encoder == 'MF':
            user_embeddings = self.user_embedding.weight
            item_embeddings = self.item_embedding.weight
        else:
            all_embeddings = self.get_ego_embeddings()
            embeddings_list = [all_embeddings]
            
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
                embeddings_list.append(all_embeddings)
            
            lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
            lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
            
            user_embeddings, item_embeddings = torch.split(
                lightgcn_all_embeddings, [self.n_users, self.n_items]
            )
        
        return user_embeddings, item_embeddings
    
    def calculate_loss(self, interaction):
        # Clear storage
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        
        # 1. BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.bpr_loss(pos_scores, neg_scores)
        
        # 2. Regularization Loss
        # 原始代码 losses.py: reg_loss += ... / float(embedding.shape[0])
        user_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        
        reg_loss = self.reg_loss(
            user_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )
        
        # !!! CRITICAL FIX !!!
        # RecBole 的 EmbLoss 默认返回 sum，必须除以 batch size 才能与原始代码的 scale 匹配
        batch_size = user.shape[0]
        reg_loss = reg_loss / batch_size
        
        # 3. Neighbor Aggregate Loss (Contrastive Loss)
        na_loss = self.na_loss(u_embeddings, pos_embeddings)
        
        # Total loss
        total_loss = mf_loss + self.reg_weight * reg_loss + self.ssl_weight * na_loss
        
        return total_loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        
        return scores.view(-1)