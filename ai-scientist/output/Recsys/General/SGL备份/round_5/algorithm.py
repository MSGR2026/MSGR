# -*- coding: utf-8 -*-
r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

Self-supervised Graph Learning (SGL) uses graph augmentation (Node Dropout, Edge Dropout, 
Random Walk) to create multiple views and applies contrastive learning for better representations.
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
    
    It augments the user-item interaction graph using Node Dropout (ND), Edge Dropout (ED),
    or Random Walk (RW), and applies contrastive learning to improve representations.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # Load parameters from config
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.ssl_tau = config["ssl_tau"]
        self.drop_ratio = config["drop_ratio"]
        
        # Augmentation type: ND (Node Dropout), ED (Edge Dropout), RW (Random Walk)
        # Note: config uses 'type' as the key name
        self.aug_type = config.get("aug_type", config.get("type", "ED"))

        # Define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Generate normalized adjacency matrix for main graph
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        # Augmented graph views (initialized as None, will be constructed during training)
        self.sub_graph1 = None
        self.sub_graph2 = None

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix: A_hat = D^{-0.5} * A * D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # Build adjacency matrix
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
        
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        
        # Convert to sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_to_coo(self, sparse_tensor):
        """Convert sparse tensor to COO format data."""
        indices = sparse_tensor.coalesce().indices()
        values = sparse_tensor.coalesce().values()
        shape = sparse_tensor.shape
        return indices, values, shape

    def graph_construction(self):
        """Construct augmented graph views based on augmentation type."""
        if self.aug_type == "ND":
            self.sub_graph1 = self.node_dropout_aug()
            self.sub_graph2 = self.node_dropout_aug()
        elif self.aug_type == "ED":
            self.sub_graph1 = self.edge_dropout_aug()
            self.sub_graph2 = self.edge_dropout_aug()
        elif self.aug_type == "RW":
            # For RW, we generate per-layer subgraphs
            self.sub_graph1 = [self.edge_dropout_aug() for _ in range(self.n_layers)]
            self.sub_graph2 = [self.edge_dropout_aug() for _ in range(self.n_layers)]
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")

    def node_dropout_aug(self):
        """Node dropout augmentation: drop nodes with probability drop_ratio."""
        # Create node mask
        keep_prob = 1.0 - self.drop_ratio
        node_mask = torch.rand(self.n_users + self.n_items) > self.drop_ratio
        
        # Get original adjacency matrix indices and values
        indices = self.norm_adj_matrix.coalesce().indices()
        values = self.norm_adj_matrix.coalesce().values()
        
        # Filter edges based on node mask
        row_mask = node_mask[indices[0]]
        col_mask = node_mask[indices[1]]
        edge_mask = row_mask & col_mask
        
        new_indices = indices[:, edge_mask]
        new_values = values[edge_mask]
        
        # Rebuild adjacency matrix and renormalize
        if new_indices.shape[1] == 0:
            return self.norm_adj_matrix.to(self.device)
        
        # Create sparse matrix for renormalization
        size = self.n_users + self.n_items
        adj = sp.coo_matrix(
            (new_values.cpu().numpy(), (new_indices[0].cpu().numpy(), new_indices[1].cpu().numpy())),
            shape=(size, size)
        )
        
        # Renormalize
        sumArr = np.array(adj.sum(axis=1)).flatten() + 1e-7
        diag = np.power(sumArr, -0.5)
        D = sp.diags(diag)
        L = D @ adj @ D
        L = sp.coo_matrix(L)
        
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size(L.shape)).to(self.device)

    def edge_dropout_aug(self):
        """Edge dropout augmentation: drop edges with probability drop_ratio."""
        indices = self.norm_adj_matrix.coalesce().indices()
        values = self.norm_adj_matrix.coalesce().values()
        
        # Create edge mask
        edge_mask = torch.rand(values.shape[0]) > self.drop_ratio
        
        new_indices = indices[:, edge_mask]
        new_values = values[edge_mask]
        
        if new_indices.shape[1] == 0:
            return self.norm_adj_matrix.to(self.device)
        
        # Create sparse matrix for renormalization
        size = self.n_users + self.n_items
        adj = sp.coo_matrix(
            (new_values.cpu().numpy(), (new_indices[0].cpu().numpy(), new_indices[1].cpu().numpy())),
            shape=(size, size)
        )
        
        # Renormalize
        sumArr = np.array(adj.sum(axis=1)).flatten() + 1e-7
        diag = np.power(sumArr, -0.5)
        D = sp.diags(diag)
        L = D @ adj @ D
        L = sp.coo_matrix(L)
        
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size(L.shape)).to(self.device)

    def get_ego_embeddings(self):
        """Get the embedding of users and items and combine to an embedding matrix."""
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
            # Random Walk: different graph for each layer
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph[layer_idx], all_embeddings)
                embeddings_list.append(all_embeddings)
        else:
            # ND or ED: same graph for all layers
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(graph, all_embeddings)
                embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calc_ssl_loss(self, user_idx, item_idx, user_emb1, user_emb2, item_emb1, item_emb2):
        """Calculate the self-supervised contrastive loss (InfoNCE)."""
        # Normalize embeddings
        user_emb1 = F.normalize(user_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        item_emb1 = F.normalize(item_emb1, dim=1)
        item_emb2 = F.normalize(item_emb2, dim=1)

        # Get batch embeddings
        user_emb1_batch = user_emb1[user_idx]
        user_emb2_batch = user_emb2[user_idx]
        item_emb1_batch = item_emb1[item_idx]
        item_emb2_batch = item_emb2[item_idx]

        # User-side contrastive loss
        pos_score_user = torch.sum(user_emb1_batch * user_emb2_batch, dim=1) / self.ssl_tau
        all_score_user = torch.matmul(user_emb1_batch, user_emb2.T) / self.ssl_tau
        ssl_loss_user = -torch.mean(pos_score_user - torch.logsumexp(all_score_user, dim=1))

        # Item-side contrastive loss
        pos_score_item = torch.sum(item_emb1_batch * item_emb2_batch, dim=1) / self.ssl_tau
        all_score_item = torch.matmul(item_emb1_batch, item_emb2.T) / self.ssl_tau
        ssl_loss_item = -torch.mean(pos_score_item - torch.logsumexp(all_score_item, dim=1))

        ssl_loss = ssl_loss_user + ssl_loss_item
        return ssl_loss

    def calculate_loss(self, interaction):
        """Calculate the total loss including BPR loss, SSL loss, and regularization."""
        # Clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Main task forward
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
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # SSL Loss
        if self.sub_graph1 is not None and self.sub_graph2 is not None:
            user_emb1, item_emb1 = self.forward(self.sub_graph1)
            user_emb2, item_emb2 = self.forward(self.sub_graph2)
            ssl_loss = self.calc_ssl_loss(user, pos_item, user_emb1, user_emb2, item_emb1, item_emb2)
        else:
            ssl_loss = torch.tensor(0.0).to(self.device)

        loss = mf_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss
        return loss

    def predict(self, interaction):
        """Predict scores for given user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users."""
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

    def train(self, mode: bool = True):
        """Override train to construct augmented graphs at the start of each epoch."""
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T