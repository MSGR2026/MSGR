# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class SGL(GeneralRecommender):
    r"""SGL is a self-supervised graph learning framework for recommendation.
    
    SGL augments the user-item interaction graph with node dropout, edge dropout, 
    or random walk, and applies contrastive learning to improve representation learning.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.ssl_tau = config["ssl_tau"]
        self.ssl_weight = config["ssl_weight"]
        self.drop_ratio = config["drop_ratio"]
        self.aug_type = config["aug_type"]  # 'ND', 'ED', 'RW'

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
        
        # augmented graph views (will be generated per epoch)
        self.sub_graph1 = None
        self.sub_graph2 = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def _sparse_dropout(self, adj_matrix, drop_ratio):
        """Apply dropout on sparse adjacency matrix."""
        if drop_ratio == 0.0:
            return adj_matrix
        
        adj_matrix = adj_matrix.coalesce()
        indices = adj_matrix.indices()
        values = adj_matrix.values()
        
        # generate mask
        mask = torch.rand(values.size(0), device=self.device) > drop_ratio
        new_indices = indices[:, mask]
        new_values = values[mask] / (1.0 - drop_ratio)  # scale to maintain expectation
        
        return torch.sparse_coo_tensor(
            new_indices, new_values, adj_matrix.size(), device=self.device
        )

    def _node_dropout(self, adj_matrix, drop_ratio):
        """Apply node dropout on sparse adjacency matrix."""
        if drop_ratio == 0.0:
            return adj_matrix
        
        n_nodes = self.n_users + self.n_items
        # generate node mask
        node_mask = torch.rand(n_nodes, device=self.device) > drop_ratio
        
        adj_matrix = adj_matrix.coalesce()
        indices = adj_matrix.indices()
        values = adj_matrix.values()
        
        # keep edges where both endpoints are kept
        row_mask = node_mask[indices[0]]
        col_mask = node_mask[indices[1]]
        edge_mask = row_mask & col_mask
        
        new_indices = indices[:, edge_mask]
        new_values = values[edge_mask]
        
        # re-normalize the adjacency matrix
        if new_values.size(0) == 0:
            return adj_matrix
        
        # create new sparse matrix and normalize
        new_adj = torch.sparse_coo_tensor(
            new_indices, new_values, adj_matrix.size(), device=self.device
        ).coalesce()
        
        return self._normalize_sparse_adj(new_adj)

    def _normalize_sparse_adj(self, adj):
        """Normalize sparse adjacency matrix with D^{-0.5} A D^{-0.5}."""
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        
        if values.size(0) == 0:
            return adj
        
        # compute degree
        n_nodes = adj.size(0)
        degree = torch.zeros(n_nodes, device=self.device)
        degree.scatter_add_(0, indices[0], values)
        
        # D^{-0.5}
        degree_inv_sqrt = torch.pow(degree + 1e-7, -0.5)
        
        # normalize values
        new_values = values * degree_inv_sqrt[indices[0]] * degree_inv_sqrt[indices[1]]
        
        return torch.sparse_coo_tensor(indices, new_values, adj.size(), device=self.device)

    def graph_construction(self):
        """Construct augmented graph views based on augmentation type."""
        if self.aug_type == 'ND':
            # Node Dropout
            self.sub_graph1 = self._node_dropout(self.norm_adj_matrix, self.drop_ratio)
            self.sub_graph2 = self._node_dropout(self.norm_adj_matrix, self.drop_ratio)
        elif self.aug_type == 'ED':
            # Edge Dropout
            self.sub_graph1 = self._sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
            self.sub_graph2 = self._sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
            # re-normalize after edge dropout
            self.sub_graph1 = self._normalize_sparse_adj(self.sub_graph1)
            self.sub_graph2 = self._normalize_sparse_adj(self.sub_graph2)
        elif self.aug_type == 'RW':
            # Random Walk - generate per-layer subgraphs
            self.sub_graph1 = []
            self.sub_graph2 = []
            for _ in range(self.n_layers):
                g1 = self._sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
                g2 = self._sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
                self.sub_graph1.append(self._normalize_sparse_adj(g1))
                self.sub_graph2.append(self._normalize_sparse_adj(g2))
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, graph=None):
        """Forward propagation on the given graph (or main graph if None)."""
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        if graph is None:
            graph = self.norm_adj_matrix

        if isinstance(graph, list):
            # RW mode: different graph per layer
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph[layer_idx], all_embeddings)
                embeddings_list.append(all_embeddings)
        else:
            # ND/ED mode: same graph for all layers
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph, all_embeddings)
                embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calc_ssl_loss(self, user_embeddings1, item_embeddings1, 
                      user_embeddings2, item_embeddings2, users, items):
        """Calculate the self-supervised contrastive loss (InfoNCE)."""
        # normalize embeddings
        user_emb1 = F.normalize(user_embeddings1, dim=1)
        user_emb2 = F.normalize(user_embeddings2, dim=1)
        item_emb1 = F.normalize(item_embeddings1, dim=1)
        item_emb2 = F.normalize(item_embeddings2, dim=1)

        # user SSL loss
        # get embeddings for users in batch
        user_emb1_batch = user_emb1[users]  # [B, d]
        user_emb2_batch = user_emb2[users]  # [B, d]
        
        # positive scores
        pos_score_user = torch.sum(user_emb1_batch * user_emb2_batch, dim=1)  # [B]
        pos_score_user = torch.exp(pos_score_user / self.ssl_tau)
        
        # negative scores: all users as negatives
        ttl_score_user = torch.matmul(user_emb1_batch, user_emb2.T)  # [B, n_users]
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_tau).sum(dim=1)  # [B]
        
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user + 1e-8).mean()

        # item SSL loss
        item_emb1_batch = item_emb1[items]  # [B, d]
        item_emb2_batch = item_emb2[items]  # [B, d]
        
        pos_score_item = torch.sum(item_emb1_batch * item_emb2_batch, dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_tau)
        
        ttl_score_item = torch.matmul(item_emb1_batch, item_emb2.T)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_tau).sum(dim=1)
        
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item + 1e-8).mean()

        return ssl_loss_user + ssl_loss_item

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # forward on main graph
        user_all_embeddings, item_all_embeddings = self.forward()
        
        # forward on augmented graphs
        user_emb1, item_emb1 = self.forward(self.sub_graph1)
        user_emb2, item_emb2 = self.forward(self.sub_graph2)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # calculate SSL Loss
        ssl_loss = self.calc_ssl_loss(
            user_emb1, item_emb1, user_emb2, item_emb2, user, pos_item
        )

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

    def train(self, mode=True):
        """Override train to construct augmented graphs at the beginning of each epoch."""
        super().train(mode)
        if mode:
            # construct augmented graph views for SSL
            self.graph_construction()
        return self