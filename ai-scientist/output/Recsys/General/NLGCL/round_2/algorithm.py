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
        self.ssl_weight = config["ssl_weight"]  # λ1 in paper
        self.ssl_temp = config["ssl_temp"]  # τ in paper
        self.G = config["G"]  # number of contrastive view groups
        self.scope = config["scope"]  # 'heterogeneous' or 'entire'
        self.require_pow = config.get("require_pow", False)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.user_neighbors = self._build_user_neighbors()
        self.item_neighbors = self._build_item_neighbors()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # Build adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M.row + self.n_users, inter_M.col), [1] * inter_M.nnz)))
        A._update(data_dict)

        # Symmetric normalization
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D

        # Convert to PyTorch sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def _build_user_neighbors(self):
        r"""Build neighbor list for each user."""
        user_neighbors = [[] for _ in range(self.n_users)]
        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            user_neighbors[u].append(i + self.n_users)  # items are stored after users
        return user_neighbors

    def _build_item_neighbors(self):
        r"""Build neighbor list for each item."""
        item_neighbors = [[] for _ in range(self.n_items)]
        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            item_neighbors[i].append(u)  # users are stored before items
        return item_neighbors

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        r"""Forward propagation through graph convolution layers."""
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        user_all_embeddings, item_all_embeddings = torch.split(
            torch.stack(embeddings_list, dim=1), [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def _get_layer_neighbors(self, layer_embeddings, neighbors):
        r"""Get neighbor embeddings for each node in the current layer."""
        neighbor_embeddings = []
        for node_idx, neighbor_list in enumerate(neighbors):
            if not neighbor_list:
                neighbor_embeddings.append(torch.zeros_like(layer_embeddings[0]))
            else:
                neighbor_embeddings.append(torch.mean(layer_embeddings[neighbor_list], dim=0))
        return torch.stack(neighbor_embeddings)

    def _compute_ssl_loss(self, user_embeddings, item_embeddings):
        r"""Compute the neighbor layer contrastive learning loss."""
        total_loss = 0.0
        user_layer_embeddings, item_layer_embeddings = self.forward()
        
        for g in range(self.G):
            u_g = user_layer_embeddings[:, g]
            i_g = item_layer_embeddings[:, g]
            u_g_plus = user_layer_embeddings[:, g+1]
            i_g_plus = item_layer_embeddings[:, g+1]

            # User-side loss
            u_loss = 0.0
            for u_idx in range(self.n_users):
                if not self.user_neighbors[u_idx]:
                    continue
                u_g_u = u_g[u_idx]
                pos_i = i_g_plus[self.user_neighbors[u_idx]]
                pos_sim = torch.sum(u_g_u.unsqueeze(0) * pos_i, dim=1) / self.ssl_temp
                if self.scope == 'heterogeneous':
                    neg_i = i_g_plus
                else:  # 'entire'
                    neg_i = torch.cat([i_g_plus, u_g_plus], dim=0)
                neg_sim = torch.sum(u_g_u.unsqueeze(0) * neg_i, dim=1) / self.ssl_temp
                pos_logsum = torch.logsumexp(pos_sim, dim=0)
                neg_logsum = torch.logsumexp(neg_sim, dim=0)
                u_loss += - (pos_logsum - neg_logsum)
            u_loss /= self.n_users
            u_loss /= len(self.user_neighbors[0]) if self.user_neighbors[0] else 1

            # Item-side loss
            i_loss = 0.0
            for i_idx in range(self.n_items):
                if not self.item_neighbors[i_idx]:
                    continue
                i_g_i = i_g[i_idx]
                pos_u = u_g_plus[self.item_neighbors[i_idx]]
                pos_sim = torch.sum(i_g_i.unsqueeze(0) * pos_u, dim=1) / self.ssl_temp
                if self.scope == 'heterogeneous':
                    neg_u = u_g_plus
                else:  # 'entire'
                    neg_u = torch.cat([u_g_plus, i_g_plus], dim=0)
                neg_sim = torch.sum(i_g_i.unsqueeze(0) * neg_u, dim=1) / self.ssl_temp
                pos_logsum = torch.logsumexp(pos_sim, dim=0)
                neg_logsum = torch.logsumexp(neg_sim, dim=0)
                i_loss += - (pos_logsum - neg_logsum)
            i_loss /= self.n_items
            i_loss /= len(self.item_neighbors[0]) if self.item_neighbors[0] else 1

            total_loss += (u_loss + i_loss) / self.G

        return total_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        # Self-supervised loss
        ssl_loss = self._compute_ssl_loss(user_all_embeddings, item_all_embeddings)

        # Total loss
        loss = mf_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss

        return loss

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
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)