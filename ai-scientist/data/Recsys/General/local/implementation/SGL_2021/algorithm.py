# -*- coding: utf-8 -*-
# @Time   : 2021/3/10
# @Author : SGL Implementation
# @Email  : sgl@example.com

r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL-Torch
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


class SGL(GeneralRecommender):
    r"""SGL is a GCN-based recommender model with self-supervised graph learning.

    SGL enhances LightGCN with contrastive learning on augmented graph views.
    It uses three augmentation operators: Node Dropout (ND), Edge Dropout (ED), and Random Walk (RW).
    The self-supervised task maximizes agreement between different views of the same node.
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
        self.ssl_weight = config["ssl_weight"]  # λ1 in paper
        self.ssl_temp = config["ssl_temp"]  # τ in paper
        self.ssl_mode = config["ssl_mode"]  # 'ED', 'ND', or 'RW'
        self.ssl_ratio = config["ssl_ratio"]  # ρ in paper
        self.require_pow = config.get("require_pow", False)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate main graph
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        # augmented graphs (initialized, will be refreshed per epoch)
        self.sub_graph1 = None
        self.sub_graph2 = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
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
        inter_M = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items

        # Build adjacency matrix
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

    def _create_augmented_graph(self, adj_matrix, mode, ratio):
        r"""Create augmented graph by applying dropout.

        Args:
            adj_matrix: Original adjacency matrix in scipy sparse format
            mode: 'ND' for node dropout, 'ED' for edge dropout
            ratio: Dropout ratio

        Returns:
            Augmented adjacency matrix as PyTorch sparse tensor
        """
        if mode == 'ND':
            # Node Dropout: randomly drop nodes
            n_nodes = self.n_users + self.n_items
            keep_idx = np.random.choice(
                n_nodes,
                size=int(n_nodes * (1 - ratio)),
                replace=False
            )
            keep_mask = np.zeros(n_nodes, dtype=bool)
            keep_mask[keep_idx] = True

            # Filter edges connected to kept nodes
            row_mask = keep_mask[adj_matrix.row]
            col_mask = keep_mask[adj_matrix.col]
            edge_mask = row_mask & col_mask

            row = adj_matrix.row[edge_mask]
            col = adj_matrix.col[edge_mask]
            data = adj_matrix.data[edge_mask]

            aug_adj = sp.coo_matrix(
                (data, (row, col)),
                shape=adj_matrix.shape,
                dtype=np.float32
            )

        elif mode == 'ED':
            # Edge Dropout: randomly drop edges
            n_edges = adj_matrix.nnz
            keep_idx = np.random.choice(
                n_edges,
                size=int(n_edges * (1 - ratio)),
                replace=False
            )

            row = adj_matrix.row[keep_idx]
            col = adj_matrix.col[keep_idx]
            data = adj_matrix.data[keep_idx]

            aug_adj = sp.coo_matrix(
                (data, (row, col)),
                shape=adj_matrix.shape,
                dtype=np.float32
            )

        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

        # Renormalize
        sumArr = (aug_adj > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ aug_adj @ D

        # Convert to PyTorch sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def _create_rw_augmented_graph(self, adj_matrix, ratio):
        r"""Create augmented graph for Random Walk (layer-specific edge dropout).

        Args:
            adj_matrix: Original adjacency matrix in scipy sparse format
            ratio: Dropout ratio

        Returns:
            List of augmented adjacency matrices (one per layer) as PyTorch sparse tensors
        """
        aug_graphs = []
        for _ in range(self.n_layers):
            # Apply edge dropout with different random seed per layer
            n_edges = adj_matrix.nnz
            keep_idx = np.random.choice(
                n_edges,
                size=int(n_edges * (1 - ratio)),
                replace=False
            )

            row = adj_matrix.row[keep_idx]
            col = adj_matrix.col[keep_idx]
            data = adj_matrix.data[keep_idx]

            aug_adj = sp.coo_matrix(
                (data, (row, col)),
                shape=adj_matrix.shape,
                dtype=np.float32
            )

            # Renormalize
            sumArr = (aug_adj > 0).sum(axis=1)
            diag = np.array(sumArr.flatten())[0] + 1e-7
            diag = np.power(diag, -0.5)
            D = sp.diags(diag)
            L = D @ aug_adj @ D

            # Convert to PyTorch sparse tensor
            L = sp.coo_matrix(L)
            row = L.row
            col = L.col
            i = torch.LongTensor(np.array([row, col]))
            data = torch.FloatTensor(L.data)
            SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

            aug_graphs.append(SparseL.to(self.device))

        return aug_graphs

    def graph_construction(self):
        r"""Construct augmented graphs for self-supervised learning.
        
        This method is called at the beginning of each training epoch to generate
        two different views of the graph.
        """
        # Convert normalized adjacency matrix back to scipy format for augmentation
        indices = self.norm_adj_matrix.coalesce().indices().cpu().numpy()
        values = self.norm_adj_matrix.coalesce().values().cpu().numpy()
        shape = self.norm_adj_matrix.shape
        
        # Reconstruct original adjacency (before normalization) for augmentation
        # We use the interaction matrix to rebuild
        inter_M = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items

        row_user = inter_M.row
        col_item = inter_M.col + n_users
        row_item = inter_M.col + n_users
        col_user = inter_M.row

        row = np.concatenate([row_user, row_item])
        col = np.concatenate([col_item, col_user])
        data = np.ones(len(row), dtype=np.float32)

        adj_matrix = sp.coo_matrix(
            (data, (row, col)),
            shape=(n_users + n_items, n_users + n_items),
            dtype=np.float32
        )

        if self.ssl_mode in ['ND', 'ED']:
            # Create two independent augmented views
            self.sub_graph1 = self._create_augmented_graph(
                adj_matrix, self.ssl_mode, self.ssl_ratio
            ).to(self.device)
            self.sub_graph2 = self._create_augmented_graph(
                adj_matrix, self.ssl_mode, self.ssl_ratio
            ).to(self.device)
        elif self.ssl_mode == 'RW':
            # Create layer-specific augmented graphs
            self.sub_graph1 = self._create_rw_augmented_graph(adj_matrix, self.ssl_ratio)
            self.sub_graph2 = self._create_rw_augmented_graph(adj_matrix, self.ssl_ratio)
        else:
            raise ValueError(f"Unknown ssl_mode: {self.ssl_mode}")

    def train(self, mode=True):
        r"""Override train method to refresh augmented graphs at the beginning of training."""
        super().train(mode)
        if mode:
            self.graph_construction()
        return self

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, graph=None):
        r"""Forward propagation through graph convolution layers.

        Args:
            graph: Adjacency matrix to use. If None, use main graph.

        Returns:
            user_all_embeddings: User embeddings after aggregation
            item_all_embeddings: Item embeddings after aggregation
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        if graph is None:
            graph = self.norm_adj_matrix

        if self.ssl_mode == 'RW' and isinstance(graph, list):
            # Random Walk: different graph per layer
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph[layer_idx], all_embeddings)
                embeddings_list.append(all_embeddings)
        else:
            # ND or ED: same graph across all layers
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph, all_embeddings)
                embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calc_ssl_loss(self, user_emb1, user_emb2, item_emb1, item_emb2):
        r"""Calculate self-supervised contrastive loss.

        Args:
            user_emb1: User embeddings from view 1
            user_emb2: User embeddings from view 2
            item_emb1: Item embeddings from view 1
            item_emb2: Item embeddings from view 2

        Returns:
            ssl_loss: Self-supervised contrastive loss
        """
        # Normalize embeddings for cosine similarity
        user_emb1 = F.normalize(user_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        item_emb1 = F.normalize(item_emb1, dim=1)
        item_emb2 = F.normalize(item_emb2, dim=1)

        # User contrastive loss
        # Positive scores: same user across views
        pos_user_scores = torch.sum(user_emb1 * user_emb2, dim=1) / self.ssl_temp  # [n_users]
        
        # Negative scores: all users in view 2
        # For efficiency, we use in-batch negatives instead of all users
        # Full implementation: neg_user_scores = torch.mm(user_emb1, user_emb2.T) / self.ssl_temp
        # We'll use all users for correctness as per paper
        ttl_user_scores = torch.mm(user_emb1, user_emb2.T) / self.ssl_temp  # [n_users, n_users]
        
        # InfoNCE loss for users
        pos_user_scores = torch.exp(pos_user_scores)  # [n_users]
        ttl_user_scores = torch.exp(ttl_user_scores).sum(dim=1)  # [n_users]
        ssl_loss_user = -torch.log(pos_user_scores / ttl_user_scores).sum()

        # Item contrastive loss (same logic)
        pos_item_scores = torch.sum(item_emb1 * item_emb2, dim=1) / self.ssl_temp
        ttl_item_scores = torch.mm(item_emb1, item_emb2.T) / self.ssl_temp
        
        pos_item_scores = torch.exp(pos_item_scores)
        ttl_item_scores = torch.exp(ttl_item_scores).sum(dim=1)
        ssl_loss_item = -torch.log(pos_item_scores / ttl_item_scores).sum()

        ssl_loss = ssl_loss_user + ssl_loss_item
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Main task: forward with main graph
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

        # Self-supervised task: forward with augmented graphs
        if self.training and self.sub_graph1 is not None and self.sub_graph2 is not None:
            user_emb1, item_emb1 = self.forward(self.sub_graph1)
            user_emb2, item_emb2 = self.forward(self.sub_graph2)
            ssl_loss = self.calc_ssl_loss(user_emb1, user_emb2, item_emb1, item_emb2)
        else:
            ssl_loss = torch.tensor(0.0).to(self.device)

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