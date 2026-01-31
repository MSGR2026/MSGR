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
    
    SGL augments the user-item bipartite graph with three operators (Node Dropout, 
    Edge Dropout, Random Walk) to generate contrastive views, and uses InfoNCE loss
    to maximize agreement between different views of the same node.
    
    We implement the model following the original paper with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type: the embedding size
        self.n_layers = config["n_layers"]  # int type: the layer num of LightGCN
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        
        # SGL specific parameters
        self.ssl_tau = config["ssl_tau"]  # float32 type: temperature for InfoNCE
        self.ssl_weight = config["ssl_weight"]  # float32 type: weight for SSL loss
        self.drop_ratio = config["drop_ratio"]  # float32 type: dropout ratio for augmentation
        self.aug_type = config["aug_type"]  # str type: augmentation type (ND, ED, RW)

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
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, adj, drop_ratio):
        r"""Perform dropout on sparse adjacency matrix.
        
        Args:
            adj: Sparse tensor of adjacency matrix
            drop_ratio: Dropout ratio
            
        Returns:
            Dropped sparse tensor
        """
        if drop_ratio == 0.0:
            return adj
            
        # Get indices and values
        indices = adj._indices()
        values = adj._values()
        
        # Generate dropout mask
        mask = torch.rand(values.size(0), device=self.device) > drop_ratio
        
        # Apply mask
        new_indices = indices[:, mask]
        new_values = values[mask] / (1.0 - drop_ratio)  # Scale to maintain expectation
        
        # Create new sparse tensor
        return torch.sparse.FloatTensor(new_indices, new_values, adj.size()).coalesce()

    def node_dropout(self, adj, drop_ratio):
        r"""Perform node dropout on adjacency matrix.
        
        Args:
            adj: Sparse tensor of adjacency matrix
            drop_ratio: Dropout ratio for nodes
            
        Returns:
            Dropped sparse tensor
        """
        if drop_ratio == 0.0:
            return adj
            
        # Generate node mask
        n_nodes = self.n_users + self.n_items
        node_mask = torch.rand(n_nodes, device=self.device) > drop_ratio
        
        # Get indices and values
        indices = adj._indices()
        values = adj._values()
        
        # Keep edges where both endpoints are not dropped
        row_mask = node_mask[indices[0]]
        col_mask = node_mask[indices[1]]
        edge_mask = row_mask & col_mask
        
        # Apply mask
        new_indices = indices[:, edge_mask]
        new_values = values[edge_mask]
        
        # Create new sparse tensor and renormalize
        dropped_adj = torch.sparse.FloatTensor(new_indices, new_values, adj.size()).coalesce()
        
        return self._normalize_sparse_adj(dropped_adj)

    def _normalize_sparse_adj(self, adj):
        r"""Normalize sparse adjacency matrix using D^{-0.5} A D^{-0.5}.
        
        Args:
            adj: Sparse tensor of adjacency matrix
            
        Returns:
            Normalized sparse tensor
        """
        # Convert to dense for degree computation, then back to sparse
        indices = adj._indices()
        values = adj._values()
        size = adj.size()
        
        # Compute degree
        row = indices[0]
        degree = torch.zeros(size[0], device=self.device)
        degree.scatter_add_(0, row, values)
        
        # Add epsilon and compute D^{-0.5}
        degree = degree + 1e-7
        degree_inv_sqrt = torch.pow(degree, -0.5)
        
        # Normalize values
        new_values = values * degree_inv_sqrt[indices[0]] * degree_inv_sqrt[indices[1]]
        
        return torch.sparse.FloatTensor(indices, new_values, size).coalesce()

    def graph_construction(self):
        r"""Construct augmented graph views based on augmentation type.
        
        This method should be called at the beginning of each training epoch
        to generate fresh augmented views.
        """
        if self.aug_type == "ND":
            # Node Dropout
            self.sub_graph1 = self.node_dropout(self.norm_adj_matrix, self.drop_ratio)
            self.sub_graph2 = self.node_dropout(self.norm_adj_matrix, self.drop_ratio)
        elif self.aug_type == "ED":
            # Edge Dropout
            self.sub_graph1 = self.sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
            self.sub_graph2 = self.sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
            # Renormalize after edge dropout
            self.sub_graph1 = self._normalize_sparse_adj(self.sub_graph1)
            self.sub_graph2 = self._normalize_sparse_adj(self.sub_graph2)
        elif self.aug_type == "RW":
            # Random Walk - layer-wise edge dropout
            self.sub_graph1 = []
            self.sub_graph2 = []
            for _ in range(self.n_layers):
                g1 = self.sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
                g2 = self.sparse_dropout(self.norm_adj_matrix, self.drop_ratio)
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
        r"""Forward propagation on the given graph.
        
        Args:
            graph: Sparse adjacency matrix or list of matrices (for RW).
                   If None, use the main normalized adjacency matrix.
                   
        Returns:
            user_all_embeddings: User embeddings after aggregation
            item_all_embeddings: Item embeddings after aggregation
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        if graph is None:
            graph = self.norm_adj_matrix

        for layer_idx in range(self.n_layers):
            if isinstance(graph, list):
                # RW mode: different graph per layer
                all_embeddings = torch.sparse.mm(graph[layer_idx], all_embeddings)
            else:
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
        r"""Calculate the self-supervised contrastive loss (InfoNCE).
        
        Args:
            user_embeddings1: User embeddings from view 1
            item_embeddings1: Item embeddings from view 1
            user_embeddings2: User embeddings from view 2
            item_embeddings2: Item embeddings from view 2
            users: User indices in the batch
            items: Item indices in the batch
            
        Returns:
            ssl_loss: The contrastive loss
        """
        # Get unique users and items in batch
        users = torch.unique(users)
        items = torch.unique(items)
        
        # User contrastive loss
        user_emb1 = user_embeddings1[users]
        user_emb2 = user_embeddings2[users]
        
        # Normalize embeddings
        user_emb1 = F.normalize(user_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        
        # All user embeddings as negatives (normalized)
        all_user_emb2 = F.normalize(user_embeddings2, dim=1)
        
        # Positive scores
        pos_score_user = torch.sum(user_emb1 * user_emb2, dim=1)  # [batch_users]
        pos_score_user = torch.exp(pos_score_user / self.ssl_tau)
        
        # All scores (including negatives)
        all_score_user = torch.matmul(user_emb1, all_user_emb2.T)  # [batch_users, n_users]
        all_score_user = torch.exp(all_score_user / self.ssl_tau)
        all_score_user = torch.sum(all_score_user, dim=1)  # [batch_users]
        
        # User InfoNCE loss
        ssl_loss_user = -torch.log(pos_score_user / all_score_user + 1e-10).sum()
        
        # Item contrastive loss
        item_emb1 = item_embeddings1[items]
        item_emb2 = item_embeddings2[items]
        
        # Normalize embeddings
        item_emb1 = F.normalize(item_emb1, dim=1)
        item_emb2 = F.normalize(item_emb2, dim=1)
        
        # All item embeddings as negatives (normalized)
        all_item_emb2 = F.normalize(item_embeddings2, dim=1)
        
        # Positive scores
        pos_score_item = torch.sum(item_emb1 * item_emb2, dim=1)  # [batch_items]
        pos_score_item = torch.exp(pos_score_item / self.ssl_tau)
        
        # All scores (including negatives)
        all_score_item = torch.matmul(item_emb1, all_item_emb2.T)  # [batch_items, n_items]
        all_score_item = torch.exp(all_score_item / self.ssl_tau)
        all_score_item = torch.sum(all_score_item, dim=1)  # [batch_items]
        
        # Item InfoNCE loss
        ssl_loss_item = -torch.log(pos_score_item / all_score_item + 1e-10).sum()
        
        ssl_loss = ssl_loss_user + ssl_loss_item
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Main task: forward on original graph
        user_all_embeddings, item_all_embeddings = self.forward()
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

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
        )

        # SSL task: forward on augmented graphs
        if self.sub_graph1 is not None and self.sub_graph2 is not None:
            user_embeddings1, item_embeddings1 = self.forward(self.sub_graph1)
            user_embeddings2, item_embeddings2 = self.forward(self.sub_graph2)
            
            ssl_loss = self.calc_ssl_loss(
                user_embeddings1, item_embeddings1,
                user_embeddings2, item_embeddings2,
                user, pos_item
            )
        else:
            ssl_loss = torch.tensor(0.0, device=self.device)

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
        r"""Override train method to construct augmented graphs at the beginning of each epoch.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            self
        """
        if mode:
            # Construct new augmented graph views for this epoch
            self.graph_construction()
        return super().train(mode)