# -*- coding: utf-8 -*-
# @Time   : 2025
# @Author : NLGCL Implementation
# @Email  : nlgcl@example.com

r"""
NLGCL
################################################
Reference:
    Neighbor Layers Graph Contrastive Learning for Recommendation (2025)

This implementation follows the paper's methodology:
- Uses LightGCN as the graph convolution backbone
- Introduces neighbor layers contrastive learning with two scopes: heterogeneous and entire
- Positive pairs: user/item embeddings at layer (l-1) with their neighbors' embeddings at layer (l)
- Negative pairs: all other nodes' embeddings at layer (l)
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


class NLGCL(GeneralRecommender):
    r"""NLGCL: Neighbor Layers Graph Contrastive Learning for Recommendation.

    NLGCL leverages naturally existing contrastive views between neighbor layers in GCNs.
    It constructs positive pairs from node embeddings and their neighbors' embeddings
    across adjacent layers, eliminating the need for explicit graph augmentation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config["embedding_size"]  # embedding dimension
        self.n_layers = config["n_layers"]  # number of GCN layers (L)
        self.reg_weight = config["reg_weight"]  # λ2: L2 regularization weight
        self.cl_weight = config["cl_weight"]  # λ1: contrastive learning weight
        self.temperature = config["temperature"]  # τ: temperature for contrastive loss
        self.n_groups = config["n_groups"]  # G: number of contrastive view groups
        self.cl_scope = config["cl_scope"]  # 'heterogeneous' or 'entire'
        self.require_pow = config.get("require_pow", False)

        # Ensure n_groups <= n_layers
        if self.n_groups > self.n_layers:
            self.n_groups = self.n_layers

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
        
        # Build neighbor indices for contrastive learning
        self.user_neighbors, self.item_neighbors = self._build_neighbor_indices()

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

    def _build_neighbor_indices(self):
        r"""Build neighbor indices for users and items.
        
        For each user u, N_u contains all items that u has interacted with.
        For each item i, N_i contains all users that have interacted with i.
        
        Returns:
            user_neighbors: List of tensors, user_neighbors[u] contains item indices
            item_neighbors: List of tensors, item_neighbors[i] contains user indices
        """
        inter_M = self.interaction_matrix
        
        # Build user neighbors (items that user has interacted with)
        user_neighbors = [[] for _ in range(self.n_users)]
        for u, i in zip(inter_M.row, inter_M.col):
            user_neighbors[u].append(i)
        
        # Build item neighbors (users that have interacted with item)
        item_neighbors = [[] for _ in range(self.n_items)]
        for u, i in zip(inter_M.row, inter_M.col):
            item_neighbors[i].append(u)
        
        # Convert to tensors
        user_neighbors = [torch.LongTensor(neighbors) if len(neighbors) > 0 
                         else torch.LongTensor([0]) for neighbors in user_neighbors]
        item_neighbors = [torch.LongTensor(neighbors) if len(neighbors) > 0 
                         else torch.LongTensor([0]) for neighbors in item_neighbors]
        
        return user_neighbors, item_neighbors

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        r"""Forward propagation through LightGCN layers.
        
        Returns:
            user_all_embeddings: Aggregated user embeddings
            item_all_embeddings: Aggregated item embeddings
            all_layer_embeddings: List of embeddings at each layer for CL
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Store all layer embeddings for contrastive learning
        all_layer_embeddings = embeddings_list

        # Aggregate embeddings across layers (mean pooling)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        
        return user_all_embeddings, item_all_embeddings, all_layer_embeddings

    def calc_nl_loss_heterogeneous(self, all_layer_embeddings, users, items):
        r"""Calculate neighbor layer contrastive loss for heterogeneous scope.
        
        For heterogeneous scope:
        - User u at layer (g): positive pairs with neighbors' (items) embeddings at layer (g+1)
        - Negative pairs: all items at layer (g+1)
        
        Args:
            all_layer_embeddings: List of embeddings at each layer
            users: User indices in the batch
            items: Item indices in the batch (positive items)
            
        Returns:
            nl_loss: Neighbor layer contrastive loss
        """
        nl_loss_user = 0.0
        nl_loss_item = 0.0
        
        unique_users = torch.unique(users)
        unique_items = torch.unique(items)
        
        for g in range(self.n_groups):
            # Split embeddings at layer g and g+1
            user_emb_g, item_emb_g = torch.split(
                all_layer_embeddings[g], [self.n_users, self.n_items]
            )
            user_emb_g1, item_emb_g1 = torch.split(
                all_layer_embeddings[g + 1], [self.n_users, self.n_items]
            )
            
            # Normalize embeddings
            user_emb_g = F.normalize(user_emb_g, dim=1)
            item_emb_g = F.normalize(item_emb_g, dim=1)
            user_emb_g1 = F.normalize(user_emb_g1, dim=1)
            item_emb_g1 = F.normalize(item_emb_g1, dim=1)
            
            # User-side loss: for each user u, positive pairs are neighbors (items)
            for u in unique_users:
                u = u.item()
                neighbors = self.user_neighbors[u].to(self.device)
                if len(neighbors) == 0:
                    continue
                
                # User embedding at layer g
                u_emb = user_emb_g[u]  # [dim]
                
                # Positive: neighbors' (items) embeddings at layer g+1
                pos_emb = item_emb_g1[neighbors]  # [num_neighbors, dim]
                
                # Compute positive scores
                pos_scores = torch.sum(u_emb.unsqueeze(0) * pos_emb, dim=1) / self.temperature  # [num_neighbors]
                
                # Compute denominator: all items at layer g+1
                all_item_scores = torch.matmul(u_emb.unsqueeze(0), item_emb_g1.T) / self.temperature  # [1, n_items]
                
                # InfoNCE loss with multiple positives
                # log(prod(exp(pos))) - |N_u| * log(sum(exp(all)))
                # = sum(pos) - |N_u| * log(sum(exp(all)))
                pos_sum = pos_scores.sum()
                neg_logsumexp = torch.logsumexp(all_item_scores, dim=1) * len(neighbors)
                
                nl_loss_user += (-pos_sum + neg_logsumexp) / len(neighbors)
            
            # Item-side loss: for each item i, positive pairs are neighbors (users)
            for i in unique_items:
                i = i.item()
                neighbors = self.item_neighbors[i].to(self.device)
                if len(neighbors) == 0:
                    continue
                
                # Item embedding at layer g
                i_emb = item_emb_g[i]  # [dim]
                
                # Positive: neighbors' (users) embeddings at layer g+1
                pos_emb = user_emb_g1[neighbors]  # [num_neighbors, dim]
                
                # Compute positive scores
                pos_scores = torch.sum(i_emb.unsqueeze(0) * pos_emb, dim=1) / self.temperature
                
                # Compute denominator: all users at layer g+1
                all_user_scores = torch.matmul(i_emb.unsqueeze(0), user_emb_g1.T) / self.temperature
                
                pos_sum = pos_scores.sum()
                neg_logsumexp = torch.logsumexp(all_user_scores, dim=1) * len(neighbors)
                
                nl_loss_item += (-pos_sum + neg_logsumexp) / len(neighbors)
        
        # Average over groups and batch
        nl_loss_user = nl_loss_user / (self.n_groups * len(unique_users) + 1e-8)
        nl_loss_item = nl_loss_item / (self.n_groups * len(unique_items) + 1e-8)
        
        return nl_loss_user + nl_loss_item

    def calc_nl_loss_entire(self, all_layer_embeddings, users, items):
        r"""Calculate neighbor layer contrastive loss for entire scope.
        
        For entire scope:
        - User u at layer (g): positive pairs with neighbors' (items) embeddings at layer (g+1)
        - Negative pairs: all nodes (users + items) at layer (g+1)
        
        Args:
            all_layer_embeddings: List of embeddings at each layer
            users: User indices in the batch
            items: Item indices in the batch (positive items)
            
        Returns:
            nl_loss: Neighbor layer contrastive loss
        """
        nl_loss_user = 0.0
        nl_loss_item = 0.0
        
        unique_users = torch.unique(users)
        unique_items = torch.unique(items)
        
        for g in range(self.n_groups):
            # Split embeddings at layer g and g+1
            user_emb_g, item_emb_g = torch.split(
                all_layer_embeddings[g], [self.n_users, self.n_items]
            )
            all_emb_g1 = all_layer_embeddings[g + 1]  # All nodes at layer g+1
            
            # Normalize embeddings
            user_emb_g = F.normalize(user_emb_g, dim=1)
            item_emb_g = F.normalize(item_emb_g, dim=1)
            all_emb_g1 = F.normalize(all_emb_g1, dim=1)
            
            # User-side loss
            for u in unique_users:
                u = u.item()
                neighbors = self.user_neighbors[u].to(self.device)
                if len(neighbors) == 0:
                    continue
                
                u_emb = user_emb_g[u]
                
                # Positive: neighbors' (items) embeddings at layer g+1
                # Note: items are offset by n_users in all_emb_g1
                pos_indices = neighbors + self.n_users
                pos_emb = all_emb_g1[pos_indices]
                
                pos_scores = torch.sum(u_emb.unsqueeze(0) * pos_emb, dim=1) / self.temperature
                
                # Denominator: all nodes at layer g+1
                all_scores = torch.matmul(u_emb.unsqueeze(0), all_emb_g1.T) / self.temperature
                
                pos_sum = pos_scores.sum()
                neg_logsumexp = torch.logsumexp(all_scores, dim=1) * len(neighbors)
                
                nl_loss_user += (-pos_sum + neg_logsumexp) / len(neighbors)
            
            # Item-side loss
            for i in unique_items:
                i = i.item()
                neighbors = self.item_neighbors[i].to(self.device)
                if len(neighbors) == 0:
                    continue
                
                i_emb = item_emb_g[i]
                
                # Positive: neighbors' (users) embeddings at layer g+1
                pos_emb = all_emb_g1[neighbors]
                
                pos_scores = torch.sum(i_emb.unsqueeze(0) * pos_emb, dim=1) / self.temperature
                
                # Denominator: all nodes at layer g+1
                all_scores = torch.matmul(i_emb.unsqueeze(0), all_emb_g1.T) / self.temperature
                
                pos_sum = pos_scores.sum()
                neg_logsumexp = torch.logsumexp(all_scores, dim=1) * len(neighbors)
                
                nl_loss_item += (-pos_sum + neg_logsumexp) / len(neighbors)
        
        nl_loss_user = nl_loss_user / (self.n_groups * len(unique_users) + 1e-8)
        nl_loss_item = nl_loss_item / (self.n_groups * len(unique_items) + 1e-8)
        
        return nl_loss_user + nl_loss_item

    def calc_nl_loss_efficient(self, all_layer_embeddings, users, items):
        r"""Efficient batch-wise neighbor layer contrastive loss.
        
        This is a more efficient implementation that processes the batch together
        instead of iterating over individual users/items.
        
        Args:
            all_layer_embeddings: List of embeddings at each layer
            users: User indices in the batch
            items: Item indices in the batch (positive items)
            
        Returns:
            nl_loss: Neighbor layer contrastive loss
        """
        nl_loss = 0.0
        batch_size = users.shape[0]
        
        for g in range(self.n_groups):
            # Split embeddings at layer g and g+1
            user_emb_g, item_emb_g = torch.split(
                all_layer_embeddings[g], [self.n_users, self.n_items]
            )
            user_emb_g1, item_emb_g1 = torch.split(
                all_layer_embeddings[g + 1], [self.n_users, self.n_items]
            )
            
            # Normalize embeddings
            user_emb_g = F.normalize(user_emb_g, dim=1)
            item_emb_g = F.normalize(item_emb_g, dim=1)
            user_emb_g1 = F.normalize(user_emb_g1, dim=1)
            item_emb_g1 = F.normalize(item_emb_g1, dim=1)
            
            # User-side: use batch users and their positive items as proxy for neighbors
            # This is an approximation that uses in-batch positives
            batch_user_emb_g = user_emb_g[users]  # [batch, dim]
            batch_item_emb_g1 = item_emb_g1[items]  # [batch, dim]
            
            # Positive scores: user at layer g with their positive item at layer g+1
            pos_user_scores = torch.sum(batch_user_emb_g * batch_item_emb_g1, dim=1) / self.temperature
            
            if self.cl_scope == 'heterogeneous':
                # Negative: all items at layer g+1
                all_item_scores = torch.matmul(batch_user_emb_g, item_emb_g1.T) / self.temperature
            else:  # entire scope
                # Negative: all nodes at layer g+1
                all_emb_g1 = torch.cat([user_emb_g1, item_emb_g1], dim=0)
                all_item_scores = torch.matmul(batch_user_emb_g, all_emb_g1.T) / self.temperature
            
            # InfoNCE loss for users
            pos_user_exp = torch.exp(pos_user_scores)
            neg_user_exp = torch.exp(all_item_scores).sum(dim=1)
            nl_loss_user = -torch.log(pos_user_exp / neg_user_exp + 1e-8).mean()
            
            # Item-side: use batch items and their users as proxy for neighbors
            batch_item_emb_g = item_emb_g[items]  # [batch, dim]
            batch_user_emb_g1 = user_emb_g1[users]  # [batch, dim]
            
            # Positive scores: item at layer g with their user at layer g+1
            pos_item_scores = torch.sum(batch_item_emb_g * batch_user_emb_g1, dim=1) / self.temperature
            
            if self.cl_scope == 'heterogeneous':
                # Negative: all users at layer g+1
                all_user_scores = torch.matmul(batch_item_emb_g, user_emb_g1.T) / self.temperature
            else:  # entire scope
                # Negative: all nodes at layer g+1
                all_emb_g1 = torch.cat([user_emb_g1, item_emb_g1], dim=0)
                all_user_scores = torch.matmul(batch_item_emb_g, all_emb_g1.T) / self.temperature
            
            # InfoNCE loss for items
            pos_item_exp = torch.exp(pos_item_scores)
            neg_item_exp = torch.exp(all_user_scores).sum(dim=1)
            nl_loss_item = -torch.log(pos_item_exp / neg_item_exp + 1e-8).mean()
            
            nl_loss += (nl_loss_user + nl_loss_item)
        
        return nl_loss / self.n_groups

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, all_layer_embeddings = self.forward()
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
            require_pow=self.require_pow,
        )

        # calculate neighbor layer contrastive loss
        nl_loss = self.calc_nl_loss_efficient(all_layer_embeddings, user, pos_item)

        # Total loss: L = L_bpr + λ1 * L_nl + λ2 * ||Θ||^2
        loss = mf_loss + self.cl_weight * nl_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)