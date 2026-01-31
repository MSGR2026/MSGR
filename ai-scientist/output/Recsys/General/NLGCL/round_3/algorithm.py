# -*- coding: utf-8 -*-
# @Time   : 2025/4/5
# @Author : NLGCL Implementation
# @Email  : nlgcl@example.com

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NLGCL(GeneralRecommender):
    r"""NLGCL is a graph contrastive learning model that leverages neighbor layer embeddings for contrastive learning.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.ssl_temp = config["ssl_temp"]
        self.G = config["G"]  # number of contrastive view groups
        self.scope = config["scope"]  # 'heterogeneous' or 'entire'

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # build graph
        self.norm_adj = self._get_norm_adj_mat()
        self.user_neighbors = self._build_user_neighbors()
        self.item_neighbors = self._build_item_neighbors()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the adjacency matrix as:
            - A = [[0, R],
                   [R^T, 0]]
        And then, normalize it using Laplacian normalization.

        Returns:
            Sparse tensor of the normalized adjacency matrix.
        """
        # Build adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        interaction_tensor = self.interaction_matrix
        # User-item interactions
        A[self.interaction_matrix.row, self.interaction_matrix.col + self.n_users] = 1.0
        A[self.interaction_matrix.col + self.n_users, self.interaction_matrix.row] = 1.0

        # Degree matrix normalization
        sumArr = np.array(A.sum(axis=1))
        # For zero-rows, set D^{-1} to zero vector
        D = np.diag(1.0 / (sumArr.flatten() + 1e-7))
        D = sp.diags(D.flatten(), [0])
        A = D @ A

        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.array(A.nonzero()))
        values = torch.FloatTensor(A.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(A.shape)).to(self.device)

    def _build_user_neighbors(self):
        r"""Build neighbor list for each user."""
        user_neighbors = [[] for _ in range(self.n_users)]
        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            # Convert item id to global id (user_count + item_id)
            global_item_id = i + self.n_users
            user_neighbors[u].append(global_item_id)
        return user_neighbors

    def _build_item_neighbors(self):
        r"""Build neighbor list for each item."""
        item_neighbors = [[] for _ in range(self.n_items)]
        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            # Convert user id to global id (user_id)
            item_neighbors[i].append(u)
        return item_neighbors

    def forward(self):
        r"""Propagate embeddings through graph convolution layers."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Split user/item embeddings
        user_embeddings = torch.stack([e[:self.n_users] for e in embeddings_list], dim=1)
        item_embeddings = torch.stack([e[self.n_users:] for e in embeddings_list], dim=1)
        return user_embeddings, item_embeddings

    def _compute_ssl_loss(self, user_all_embeddings, item_all_embeddings):
        r"""Compute the neighbor layer contrastive learning loss."""
        user_layer_embeddings, item_layer_embeddings = user_all_embeddings, item_all_embeddings
        total_loss = 0.0

        for g in range(self.G):
            # Get embeddings from layer g and g+1
            u_g = user_layer_embeddings[g]
            u_g_plus = user_layer_embeddings[g+1]
            i_g = item_layer_embeddings[g]
            i_g_plus = item_layer_embeddings[g+1]

            # User-side loss
            user_loss = 0.0
            for u_idx in range(self.n_users):
                user_neighbors = torch.tensor(self.user_neighbors[u_idx], device=self.device)
                if len(user_neighbors) == 0:
                    continue
                # Get positive samples (neighbors)
                pos_i = i_g_plus[user_neighbors]
                # Compute similarity
                pos_sim = torch.sum(u_g[u_idx].unsqueeze(0) * pos_i, dim=1) / self.ssl_temp
                pos_logsum = torch.logsumexp(pos_sim, dim=0)
                
                # Negative samples (all items in layer g+1)
                if self.scope == 'heterogeneous':
                    neg_i = i_g_plus
                else:  # 'entire'
                    neg_i = torch.cat([i_g_plus, u_g_plus], dim=0)
                neg_sim = torch.sum(u_g[u_idx].unsqueeze(0) * neg_i, dim=1) / self.ssl_temp
                neg_logsum = torch.logsumexp(neg_sim, dim=0)
                
                user_loss += -(pos_logsum - neg_logsum)
            user_loss /= self.n_users

            # Item-side loss
            item_loss = 0.0
            for i_idx in range(self.n_items):
                item_neighbors = torch.tensor(self.item_neighbors[i_idx], device=self.device)
                if len(item_neighbors) == 0:
                    continue
                # Get positive samples (neighbors)
                pos_u = u_g_plus[item_neighbors]
                # Compute similarity
                pos_sim = torch.sum(i_g[i_idx].unsqueeze(0) * pos_u, dim=1) / self.ssl_temp
                pos_logsum = torch.logsumexp(pos_sim, dim=0)
                
                # Negative samples
                if self.scope == 'heterogeneous':
                    neg_u = u_g_plus
                else:  # 'entire'
                    neg_u = torch.cat([u_g_plus, i_g_plus], dim=0)
                neg_sim = torch.sum(i_g[i_idx].unsqueeze(0) * neg_u, dim=1) / self.ssl_temp
                neg_logsum = torch.logsumexp(neg_sim, dim=0)
                
                item_loss += -(pos_logsum - neg_logsum)
            item_loss /= self.n_items

            total_loss += (user_loss + item_loss) / self.G

        return total_loss

    def calculate_loss(self, interaction):
        # Get embeddings
        user_all_embeddings, item_all_embeddings = self.forward()
        
        # BPR Loss
        user_idx = interaction[self.USER_ID]
        pos_item_idx = interaction[self.ITEM_ID]
        neg_item_idx = interaction[self.NEG_ITEM_ID]
        
        u_e = user_all_embeddings[0]
        pos_i_e = item_all_embeddings[0]
        neg_i_e = item_all_embeddings[0]
        
        pos_scores = torch.sum(u_e[user_idx] * pos_item_idx, dim=1)
        neg_scores = torch.sum(u_e[user_idx] * neg_i_e[neg_item_idx], dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        # Regularization loss
        reg_loss = self.reg_loss(
            self.user_embedding(user_idx),
            self.item_embedding(pos_item_idx),
            self.item_embedding(neg_item_idx)
        )

        # SSL loss
        ssl_loss = self._compute_ssl_loss(user_all_embeddings, item_all_embeddings)

        # Total loss
        loss = bpr_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item)
        scores = torch.sum(u_e * i_e, dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].long()
        u_g_embeddings, i_g_embeddings = self.forward()
        u_e = u_g_embeddings[0]
        scores = torch.matmul(u_e[user], i_g_embeddings[0].t())
        return scores.view(-1)