# -*- coding: utf-8 -*-
# @Time   : 2021
# @Author : Based on SGL paper implementation
# @Email  : 

r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

Self-supervised Graph Learning (SGL) enhances recommendation with contrastive learning
through graph augmentation (Node Dropout, Edge Dropout, Random Walk).
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
    r"""SGL is a self-supervised graph learning framework for recommendation.
    
    It uses graph augmentation (Node Dropout, Edge Dropout, or Random Walk) to create
    multiple views of the user-item graph and applies contrastive learning to improve
    the quality of learned representations.
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
        self.ssl_weight = config["ssl_weight"]
        self.ssl_tau = config["ssl_tau"]
        self.drop_ratio = config["drop_ratio"]
        
        # Augmentation type: ND (Node Dropout), ED (Edge Dropout), RW (Random Walk)
        self.aug_type = config["aug_type"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data - main graph
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        # Augmented graph views (initialized as None, will be constructed during training)
        self.sub_graph1 = None
        self.sub_graph2 = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix: A_hat = D^{-0.5} * A * D^{-0.5}

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

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def graph_construction(self):
        r"""Construct augmented graphs for self-supervised learning.
        
        Supports three augmentation types:
        - ND: Node Dropout
        - ED: Edge Dropout  
        - RW: Random Walk (layer-wise edge dropout)
        """
        if self.aug_type == "ND":
            self.sub_graph1 = self._node_dropout_graph()
            self.sub_graph2 = self._node_dropout_graph()
        elif self.aug_type == "ED":
            self.sub_graph1 = self._edge_dropout_graph()
            self.sub_graph2 = self._edge_dropout_graph()
        elif self.aug_type == "RW":
            # For RW, we create layer-wise graphs
            self.sub_graph1 = [self._edge_dropout_graph() for _ in range(self.n_layers)]
            self.sub_graph2 = [self._edge_dropout_graph() for _ in range(self.n_layers)]
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")

    def _node_dropout_graph(self):
        r"""Create a graph with node dropout.
        
        Randomly drops nodes with probability drop_ratio, along with their edges.
        
        Returns:
            Sparse tensor of the augmented normalized adjacency matrix.
        """
        # Create node mask
        keep_prob = 1 - self.drop_ratio
        node_mask = torch.rand(self.n_users + self.n_items) > self.drop_ratio
        
        # Get edges from interaction matrix
        inter_M = self.interaction_matrix
        
        # Filter edges based on node mask
        user_mask = node_mask[:self.n_users].numpy()
        item_mask = node_mask[self.n_users:].numpy()
        
        # Keep edges where both endpoints are not dropped
        edge_mask = user_mask[inter_M.row] & item_mask[inter_M.col]
        
        # Create new sparse matrix with remaining edges
        new_row = inter_M.row[edge_mask]
        new_col = inter_M.col[edge_mask]
        new_data = np.ones(len(new_row), dtype=np.float32)
        
        # Build augmented adjacency matrix
        aug_inter_M = sp.coo_matrix(
            (new_data, (new_row, new_col)), 
            shape=(self.n_users, self.n_items)
        )
        
        return self._create_norm_adj_from_inter(aug_inter_M)

    def _edge_dropout_graph(self):
        r"""Create a graph with edge dropout.
        
        Randomly drops edges with probability drop_ratio.
        
        Returns:
            Sparse tensor of the augmented normalized adjacency matrix.
        """
        inter_M = self.interaction_matrix
        
        # Create edge mask
        edge_mask = np.random.rand(inter_M.nnz) > self.drop_ratio
        
        # Create new sparse matrix with remaining edges
        new_row = inter_M.row[edge_mask]
        new_col = inter_M.col[edge_mask]
        new_data = np.ones(len(new_row), dtype=np.float32)
        
        # Build augmented adjacency matrix
        aug_inter_M = sp.coo_matrix(
            (new_data, (new_row, new_col)), 
            shape=(self.n_users, self.n_items)
        )
        
        return self._create_norm_adj_from_inter(aug_inter_M)

    def _create_norm_adj_from_inter(self, inter_M):
        r"""Create normalized adjacency matrix from interaction matrix.
        
        Args:
            inter_M: Sparse interaction matrix (users x items)
            
        Returns:
            Sparse tensor of the normalized adjacency matrix.
        """
        inter_M = sp.coo_matrix(inter_M)
        inter_M_t = inter_M.transpose()
        
        # Build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
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
        
        # Normalize
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        
        # Convert to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape)).to(self.device)
        return SparseL

    def forward(self, graph=None):
        r"""Forward propagation through the graph.
        
        Args:
            graph: Adjacency matrix to use. If None, uses main graph.
                   Can be a single matrix or list of matrices (for RW).
        
        Returns:
            user_all_embeddings: Final user embeddings
            item_all_embeddings: Final item embeddings
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        if graph is None:
            graph = self.norm_adj_matrix

        for layer_idx in range(self.n_layers):
            if isinstance(graph, list):
                # Random Walk: use different graph for each layer
                layer_graph = graph[layer_idx]
            else:
                layer_graph = graph
            all_embeddings = torch.sparse.mm(layer_graph, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calc_ssl_loss(self, user_idx, item_idx, user_emb1, user_emb2, item_emb1, item_emb2):
        r"""Calculate the self-supervised contrastive loss (InfoNCE).
        
        Args:
            user_idx: User indices in the batch
            item_idx: Item indices in the batch
            user_emb1: User embeddings from view 1
            user_emb2: User embeddings from view 2
            item_emb1: Item embeddings from view 1
            item_emb2: Item embeddings from view 2
            
        Returns:
            ssl_loss: The contrastive loss
        """
        # Get embeddings for batch users/items
        user_emb1_batch = user_emb1[user_idx]
        user_emb2_batch = user_emb2[user_idx]
        item_emb1_batch = item_emb1[item_idx]
        item_emb2_batch = item_emb2[item_idx]
        
        # Normalize embeddings
        user_emb1_batch = F.normalize(user_emb1_batch, dim=1)
        user_emb2_batch = F.normalize(user_emb2_batch, dim=1)
        item_emb1_batch = F.normalize(item_emb1_batch, dim=1)
        item_emb2_batch = F.normalize(item_emb2_batch, dim=1)
        
        # Normalize all embeddings for negative samples
        user_emb2_all = F.normalize(user_emb2, dim=1)
        item_emb2_all = F.normalize(item_emb2, dim=1)
        
        # User SSL loss
        pos_score_user = torch.sum(user_emb1_batch * user_emb2_batch, dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_tau)
        
        ttl_score_user = torch.matmul(user_emb1_batch, user_emb2_all.T)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_tau).sum(dim=1)
        
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user + 1e-10).sum()
        
        # Item SSL loss
        pos_score_item = torch.sum(item_emb1_batch * item_emb2_batch, dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_tau)
        
        ttl_score_item = torch.matmul(item_emb1_batch, item_emb2_all.T)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_tau).sum(dim=1)
        
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item + 1e-10).sum()
        
        ssl_loss = ssl_loss_user + ssl_loss_item
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Forward pass on main graph
        user_all_embeddings, item_all_embeddings = self.forward()
        
        # Forward pass on augmented graphs
        user_emb1, item_emb1 = self.forward(self.sub_graph1)
        user_emb2, item_emb2 = self.forward(self.sub_graph2)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # Calculate SSL Loss
        ssl_loss = self.calc_ssl_loss(user, pos_item, user_emb1, user_emb2, item_emb1, item_emb2)

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

    def train(self, mode: bool = True):
        r"""Override train method to construct augmented graphs at the beginning of each epoch.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T