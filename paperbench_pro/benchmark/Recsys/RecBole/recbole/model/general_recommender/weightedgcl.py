# -*- coding: utf-8 -*-
# @Time   : 2025/1/11
# @Author : WeightedGCL Integration Team
# @Email  : 

r"""
WeightedGCL
################################################

Reference:
    Chen, Zheyu et al. "Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering." 
    in SIGIR 2025.

Reference code:
    https://github.com/example/WeightedGCL (original implementation with RecBole-GNN)
    
This implementation is adapted to work directly with RecBole without RecBole-GNN dependency.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import numpy as np
import scipy.sparse as sp


class WeightedGCL(GeneralRecommender):
    r"""WeightedGCL is a graph contrastive learning model with weighted feature aggregation.
    
    It extends LightGCN by introducing:
    1. Squeeze-and-Excitation network for adaptive feature weighting
    2. Contrastive learning between perturbed graph views
    3. InfoNCE loss for self-supervised learning
    
    The model learns to weight different embedding dimensions based on their importance,
    which helps capture more discriminative user/item representations.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(WeightedGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # embedding dimension
        self.n_layers = config['n_layers']  # number of GCN layers
        self.reg_weight = config['reg_weight']  # L2 regularization weight
        self.require_pow = config.get('require_pow', False)  # whether to use pow in regularization
        
        # WeightedGCL specific parameters
        self.cl_rate = config['cl_rate']  # contrastive learning loss weight (lambda in paper)
        self.eps = config['eps']  # perturbation magnitude
        self.temperature = config['temperature']  # temperature for InfoNCE loss
        self.k = config.get('k', 64)  # dimension parameter (not used in current implementation)

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        
        # Squeeze-and-Excitation network: 1 -> latent_dim/16 -> latent_dim/4 -> latent_dim
        # This network learns to adaptively weight each dimension of the embeddings
        self.excitation = nn.Sequential(
            nn.Linear(1, self.latent_dim // 16),
            nn.Linear(self.latent_dim // 16, self.latent_dim // 4),
            nn.Linear(self.latent_dim // 4, self.latent_dim),
            nn.Sigmoid()
        )
        
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate normalized adjacency matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # Extract user-item interaction coo matrix
        inter_M = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items

        # Build user->item edges: (user_id, item_id + n_users)
        row_user = inter_M.row
        col_item = inter_M.col + n_users
        data_user_item = np.ones(inter_M.nnz, dtype=np.float32)

        # Build item->user edges: (item_id + n_users, user_id)
        row_item = inter_M.col + n_users
        col_user = inter_M.row
        data_item_user = np.ones(inter_M.nnz, dtype=np.float32)

        # Merge into symmetric adjacency matrix
        row = np.concatenate([row_user, row_item])
        col = np.concatenate([col_item, col_user])
        data = np.concatenate([data_user_item, data_item_user])

        # Build coo format adjacency matrix
        A = sp.coo_matrix(
            (data, (row, col)),
            shape=(n_users + n_items, n_users + n_items),
            dtype=np.float32
        )

        # Calculate degree matrix and perform symmetric normalization
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # add epsilon to avoid division by zero
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)

        # Normalized adjacency matrix: L = D^(-0.5) * A * D^(-0.5)
        L = D @ A @ D

        # Convert to PyTorch sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, perturbed=False):
        r"""Forward propagation with optional perturbation.
        
        Args:
            perturbed (bool): Whether to add perturbation to the final layer embeddings.
                             Used for contrastive learning.
        
        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        all_embs = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            # Graph convolution using sparse matrix multiplication
            all_embs = torch.sparse.mm(self.norm_adj_matrix, all_embs)
            
            # Add perturbation to the last layer if requested
            if perturbed and layer_idx >= self.n_layers - 1:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            
            embeddings_list.append(all_embs)
        
        # Average embeddings from all layers
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def weight_features(self, view1, view2):
        r"""Apply Squeeze-and-Excitation weighting to two views.
        
        Args:
            view1 (Tensor): First view embeddings [batch_size, embedding_dim]
            view2 (Tensor): Second view embeddings [batch_size, embedding_dim]
        
        Returns:
            tuple: Weighted (view1, view2)
        """
        # Squeeze: global average pooling
        mid_view1 = torch.mean(view1, dim=1, keepdim=True)  # [batch_size, 1]
        mid_view2 = torch.mean(view2, dim=1, keepdim=True)  # [batch_size, 1]
        
        # Excitation: learn channel-wise weights
        result1 = self.excitation(mid_view1)  # [batch_size, embedding_dim]
        result2 = self.excitation(mid_view2)  # [batch_size, embedding_dim]
        
        # Apply weights element-wise
        return result1 * view1, result2 * view2

    def calculate_cl_loss(self, x1, x2):
        r"""Calculate contrastive learning loss using InfoNCE.
        
        Args:
            x1 (Tensor): First view embeddings [batch_size, embedding_dim]
            x2 (Tensor): Second view embeddings [batch_size, embedding_dim]
        
        Returns:
            Tensor: Contrastive loss scalar
        """
        # Normalize embeddings
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        
        # Positive score: similarity between corresponding pairs
        pos_score = (x1 * x2).sum(dim=-1)  # [batch_size]
        pos_score = torch.exp(pos_score / self.temperature)
        
        # Total score: similarity to all items in the batch
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))  # [batch_size, batch_size]
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)  # [batch_size]
        
        # InfoNCE loss
        cl_loss = -torch.log(pos_score / ttl_score).sum()
        return cl_loss

    def calculate_loss(self, interaction):
        r"""Calculate the training loss.
        
        Combines three losses:
        1. BPR loss for recommendation
        2. Regularization loss
        3. Contrastive learning loss (user + item)
        
        Returns:
            Tensor: Total loss
        """
        # Clear storage variables when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Forward pass for recommendation
        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)
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

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        # Base loss (BPR + regularization)
        base_loss = mf_loss + self.reg_weight * reg_loss

        # Contrastive learning loss
        # Get unique users and items in this batch
        unique_user = torch.unique(user)
        unique_pos_item = torch.unique(pos_item)

        # Generate two perturbed views
        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        # Apply Squeeze-and-Excitation weighting
        user_view1, user_view2 = self.weight_features(
            perturbed_user_embs_1[unique_user], 
            perturbed_user_embs_2[unique_user]
        )
        item_view1, item_view2 = self.weight_features(
            perturbed_item_embs_1[unique_pos_item], 
            perturbed_item_embs_2[unique_pos_item]
        )

        # Calculate contrastive losses
        user_cl_loss = self.calculate_cl_loss(user_view1, user_view2)
        item_cl_loss = self.calculate_cl_loss(item_view1, item_view2)

        # Total loss
        total_loss = base_loss + self.cl_rate * (user_cl_loss + item_cl_loss)

        return total_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(perturbed=False)
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
