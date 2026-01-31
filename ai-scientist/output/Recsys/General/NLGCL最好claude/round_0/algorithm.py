# -*- coding: utf-8 -*-
# @Time   : 2025
# @Author : NLGCL Implementation
# @Email  : nlgcl@example.com

r"""
NLGCL
################################################
Reference:
    Neighbor Layers Graph Contrastive Learning for Recommendation (2025)

This implementation follows the NLGCL paper which introduces a novel contrastive learning
paradigm that leverages naturally existing contrastive views between neighbor layers in GCNs.
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

    NLGCL introduces a novel contrastive learning paradigm that leverages the naturally
    existing contrastive views between neighbor layers in GCNs. Unlike traditional CL methods
    that require expensive graph augmentation, NLGCL uses the inherent structure of GNN
    propagation to define positive and negative pairs.

    Key ideas:
    - Positive pairs: For user u at layer (l-1), its neighbors' embeddings at layer l
    - Negative pairs: All other nodes' embeddings at layer l (heterogeneous or entire scope)
    - No need for graph augmentation, leveraging natural layer-wise contrastive views
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config["embedding_size"]  # embedding dimension
        self.n_layers = config["n_layers"]  # number of GCN layers (L)
        self.reg_weight = config["reg_weight"]  # lambda_2: L2 regularization weight
        self.ssl_weight = config["ssl_weight"]  # lambda_1: contrastive learning weight
        self.ssl_temp = config["ssl_temp"]  # tau: temperature for contrastive loss
        self.n_groups = config["n_groups"]  # G: number of contrastive view groups
        self.ssl_scope = config.get("ssl_scope", "heterogeneous")  # 'heterogeneous' or 'entire'
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
        # user_neighbors[u] contains item indices that user u has interacted with
        # item_neighbors[i] contains user indices that have interacted with item i
        self._build_neighbor_indices()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def _build_neighbor_indices(self):
        """Build neighbor indices from interaction matrix for contrastive learning."""
        inter_M = self.interaction_matrix

        # For users: neighbors are items
        self.user_neighbor_indices = {}
        for u in range(self.n_users):
            items = inter_M.row == u
            self.user_neighbor_indices[u] = inter_M.col[items].tolist()

        # For items: neighbors are users
        self.item_neighbor_indices = {}
        for i in range(self.n_items):
            users = inter_M.col == i
            self.item_neighbor_indices[i] = inter_M.row[users].tolist()

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
        """Forward propagation through LightGCN layers.

        Returns:
            user_all_embeddings: Final user embeddings after layer aggregation
            item_all_embeddings: Final item embeddings after layer aggregation
            all_layer_embeddings: List of embeddings at each layer [E^(0), E^(1), ..., E^(L)]
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Stack and mean for final embeddings (LightGCN aggregation)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings, embeddings_list

    def calc_nl_loss_heterogeneous(self, all_layer_embeddings, users, pos_items):
        """Calculate neighbor layer contrastive loss for heterogeneous scope.

        In heterogeneous scope:
        - For user u at layer (g): positives are neighbor items at layer (g+1)
        - Negatives are all items at layer (g+1)

        Args:
            all_layer_embeddings: List of embeddings at each layer
            users: User indices in the batch
            pos_items: Positive item indices in the batch (these are neighbors of users)

        Returns:
            nl_loss: Neighbor layer contrastive loss
        """
        nl_loss_user = 0.0
        nl_loss_item = 0.0

        for g in range(self.n_groups):
            # Get embeddings at layer g and g+1
            emb_g = all_layer_embeddings[g]
            emb_g1 = all_layer_embeddings[g + 1]

            # Split into user and item embeddings
            user_emb_g = emb_g[:self.n_users]  # [n_users, dim]
            item_emb_g = emb_g[self.n_users:]  # [n_items, dim]
            user_emb_g1 = emb_g1[:self.n_users]  # [n_users, dim]
            item_emb_g1 = emb_g1[self.n_users:]  # [n_items, dim]

            # === User-side loss ===
            # For each user in batch, positive is the pos_item (which is a neighbor)
            # This is an efficient approximation: instead of using all neighbors,
            # we use the sampled positive item as the representative neighbor
            batch_user_emb_g = user_emb_g[users]  # [batch, dim]
            batch_pos_item_emb_g1 = item_emb_g1[pos_items]  # [batch, dim]

            # Positive scores: user at layer g with its neighbor item at layer g+1
            pos_scores_user = torch.sum(batch_user_emb_g * batch_pos_item_emb_g1, dim=1) / self.ssl_temp  # [batch]

            # Negative scores: user at layer g with all items at layer g+1
            # Using in-batch negatives for efficiency
            neg_scores_user = torch.mm(batch_user_emb_g, item_emb_g1.T) / self.ssl_temp  # [batch, n_items]

            # InfoNCE loss for users
            # log(exp(pos) / sum(exp(neg)))
            # = pos - logsumexp(neg)
            nl_loss_user += (-pos_scores_user + torch.logsumexp(neg_scores_user, dim=1)).mean()

            # === Item-side loss ===
            batch_item_emb_g = item_emb_g[pos_items]  # [batch, dim]
            batch_pos_user_emb_g1 = user_emb_g1[users]  # [batch, dim]

            # Positive scores: item at layer g with its neighbor user at layer g+1
            pos_scores_item = torch.sum(batch_item_emb_g * batch_pos_user_emb_g1, dim=1) / self.ssl_temp  # [batch]

            # Negative scores: item at layer g with all users at layer g+1
            neg_scores_item = torch.mm(batch_item_emb_g, user_emb_g1.T) / self.ssl_temp  # [batch, n_users]

            # InfoNCE loss for items
            nl_loss_item += (-pos_scores_item + torch.logsumexp(neg_scores_item, dim=1)).mean()

        # Average over groups
        nl_loss = (nl_loss_user + nl_loss_item) / self.n_groups

        return nl_loss

    def calc_nl_loss_entire(self, all_layer_embeddings, users, pos_items):
        """Calculate neighbor layer contrastive loss for entire scope.

        In entire scope:
        - For user u at layer (g): positives are neighbor items at layer (g+1)
        - Negatives are ALL nodes (users + items) at layer (g+1)

        Args:
            all_layer_embeddings: List of embeddings at each layer
            users: User indices in the batch
            pos_items: Positive item indices in the batch (these are neighbors of users)

        Returns:
            nl_loss: Neighbor layer contrastive loss
        """
        nl_loss_user = 0.0
        nl_loss_item = 0.0

        for g in range(self.n_groups):
            # Get embeddings at layer g and g+1
            emb_g = all_layer_embeddings[g]
            emb_g1 = all_layer_embeddings[g + 1]

            # Split into user and item embeddings
            user_emb_g = emb_g[:self.n_users]  # [n_users, dim]
            item_emb_g = emb_g[self.n_users:]  # [n_items, dim]

            # === User-side loss ===
            batch_user_emb_g = user_emb_g[users]  # [batch, dim]
            batch_pos_item_emb_g1 = emb_g1[self.n_users + pos_items]  # [batch, dim]

            # Positive scores
            pos_scores_user = torch.sum(batch_user_emb_g * batch_pos_item_emb_g1, dim=1) / self.ssl_temp

            # Negative scores: against ALL nodes at layer g+1
            neg_scores_user = torch.mm(batch_user_emb_g, emb_g1.T) / self.ssl_temp  # [batch, n_users + n_items]

            nl_loss_user += (-pos_scores_user + torch.logsumexp(neg_scores_user, dim=1)).mean()

            # === Item-side loss ===
            batch_item_emb_g = item_emb_g[pos_items]  # [batch, dim]
            batch_pos_user_emb_g1 = emb_g1[users]  # [batch, dim]

            # Positive scores
            pos_scores_item = torch.sum(batch_item_emb_g * batch_pos_user_emb_g1, dim=1) / self.ssl_temp

            # Negative scores: against ALL nodes at layer g+1
            neg_scores_item = torch.mm(batch_item_emb_g, emb_g1.T) / self.ssl_temp  # [batch, n_users + n_items]

            nl_loss_item += (-pos_scores_item + torch.logsumexp(neg_scores_item, dim=1)).mean()

        # Average over groups
        nl_loss = (nl_loss_user + nl_loss_item) / self.n_groups

        return nl_loss

    def calculate_loss(self, interaction):
        """Calculate the total loss.

        Total loss = BPR loss + lambda_1 * NL loss + lambda_2 * L2 regularization

        Args:
            interaction: Interaction batch containing user, pos_item, neg_item

        Returns:
            loss: Total loss value
        """
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Forward pass to get embeddings at all layers
        user_all_embeddings, item_all_embeddings, all_layer_embeddings = self.forward()

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

        # Calculate Neighbor Layer Contrastive Loss
        if self.ssl_scope == "heterogeneous":
            nl_loss = self.calc_nl_loss_heterogeneous(all_layer_embeddings, user, pos_item)
        else:  # entire scope
            nl_loss = self.calc_nl_loss_entire(all_layer_embeddings, user, pos_item)

        # Total loss: L = L_bpr + lambda_1 * L_nl + lambda_2 * ||Theta||^2
        loss = mf_loss + self.ssl_weight * nl_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        """Predict scores for given user-item pairs.

        Args:
            interaction: Interaction batch containing user and item

        Returns:
            scores: Predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users.

        Args:
            interaction: Interaction batch containing users

        Returns:
            scores: Predicted scores for all items
        """
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)