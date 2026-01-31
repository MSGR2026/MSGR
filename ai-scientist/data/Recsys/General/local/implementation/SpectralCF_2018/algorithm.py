# -*- coding: utf-8 -*-
# @Time   : 2025/01/15
# @Author : SpectralCF Implementation
# @Email  : spectralcf@example.com

r"""
SpectralCF
################################################
Reference:
    Zhiwei Liu et al. "Spectral Collaborative Filtering." in RecSys 2018.

"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class SpectralCF(GeneralRecommender):
    r"""SpectralCF is a model that performs spectral collaborative filtering
    by conducting graph convolution in the spectral domain of user-item bipartite graph.
    
    The model performs eigendecomposition on the graph Laplacian matrix and uses
    polynomial approximation for efficient spectral convolution operations.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SpectralCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = config["embedding_size"]  # C in paper
        self.n_layers = config["n_layers"]  # K in paper
        self.n_filters = config["n_filters"]  # F in paper
        self.reg_weight = config["reg_weight"]  # lambda_reg in paper

        # total number of nodes in bipartite graph
        self.num_nodes = self.n_users + self.n_items

        # define layers and loss
        # Initialize user and item embeddings from Gaussian N(0.01, 0.02)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.normal_(self.user_embedding.weight, mean=0.01, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.01, std=0.02)

        # Spectral convolution filter parameters for each layer
        # Layer 0: C -> F, Layers 1 to K-1: F -> F
        self.filter_params = nn.ParameterList()
        for k in range(self.n_layers):
            if k == 0:
                self.filter_params.append(
                    nn.Parameter(torch.FloatTensor(self.embedding_size, self.n_filters))
                )
            else:
                self.filter_params.append(
                    nn.Parameter(torch.FloatTensor(self.n_filters, self.n_filters))
                )
            # Xavier initialization for filter parameters
            nn.init.xavier_normal_(self.filter_params[k])

        # Loss functions
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Precompute graph Laplacian and its eigendecomposition
        self.laplacian_matrix = self.get_laplacian_matrix()
        self.eigenvectors, self.eigenvalues = self.eigendecompose_laplacian()
        
        # Precompute spectral convolution operator (UU^T + U*Lambda*U^T)
        self.spectral_operator = self.compute_spectral_operator()

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_laplacian_matrix(self):
        r"""Construct the normalized graph Laplacian matrix L = I - D^{-0.5}AD^{-0.5}.
        
        Returns:
            torch.sparse.FloatTensor: The normalized Laplacian matrix.
        """
        # Build adjacency matrix for bipartite graph
        A = sp.dok_matrix(
            (self.num_nodes, self.num_nodes), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        # Fill in edges: user->item and item->user
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
        
        # Compute degree matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # add epsilon to avoid divide by zero
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        
        # Normalized adjacency: D^{-0.5}AD^{-0.5}
        norm_adj = D @ A @ D
        
        # Laplacian: L = I - D^{-0.5}AD^{-0.5}
        identity = sp.eye(self.num_nodes)
        laplacian = identity - norm_adj
        
        # Convert to COO format for PyTorch
        laplacian = sp.coo_matrix(laplacian)
        row = laplacian.row
        col = laplacian.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(laplacian.data)
        sparse_laplacian = torch.sparse.FloatTensor(i, data, torch.Size(laplacian.shape))
        
        return sparse_laplacian.to(self.device)

    def eigendecompose_laplacian(self):
        r"""Perform eigendecomposition on the Laplacian matrix to obtain U and Lambda.
        
        Returns:
            tuple: (eigenvectors U, eigenvalues Lambda) as dense tensors on device.
        """
        # Convert sparse Laplacian to dense for eigendecomposition
        laplacian_dense = self.laplacian_matrix.to_dense()
        
        # Perform eigendecomposition: L = U * Lambda * U^T
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_dense)
        
        return eigenvectors.to(self.device), eigenvalues.to(self.device)

    def compute_spectral_operator(self):
        r"""Precompute the spectral convolution operator (UU^T + U*Lambda*U^T).
        
        This operator is used in Eq. (10) for spectral convolution.
        
        Returns:
            torch.Tensor: The spectral operator matrix.
        """
        U = self.eigenvectors
        Lambda = torch.diag(self.eigenvalues)
        
        # Compute UU^T + U*Lambda*U^T
        UUT = torch.mm(U, U.t())
        ULambdaUT = torch.mm(torch.mm(U, Lambda), U.t())
        spectral_op = UUT + ULambdaUT
        
        return spectral_op

    def spectral_convolution(self, x, filter_param):
        r"""Perform spectral convolution operation as defined in Eq. (10).
        
        Args:
            x (torch.Tensor): Input graph signal, shape [num_nodes, input_dim]
            filter_param (torch.Tensor): Filter parameter matrix, shape [input_dim, output_dim]
        
        Returns:
            torch.Tensor: Output graph signal after convolution, shape [num_nodes, output_dim]
        """
        # Spectral convolution: sigma((UU^T + U*Lambda*U^T) * X * Theta')
        # x shape: [num_nodes, input_dim]
        # filter_param shape: [input_dim, output_dim]
        
        # Step 1: X * Theta'
        x_transformed = torch.mm(x, filter_param)  # [num_nodes, output_dim]
        
        # Step 2: (UU^T + U*Lambda*U^T) * X * Theta'
        convolved = torch.mm(self.spectral_operator, x_transformed)  # [num_nodes, output_dim]
        
        # Step 3: Apply sigmoid activation
        output = torch.sigmoid(convolved)
        
        return output

    def get_ego_embeddings(self):
        r"""Get the initial embedding of users and items and combine to an embedding matrix.
        
        Returns:
            torch.Tensor: The embedding matrix. Shape of [n_users+n_items, embedding_size]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        r"""Forward propagation through K spectral convolution layers.
        
        Returns:
            tuple: (user_embeddings, item_embeddings) after multi-layer spectral convolution
        """
        # Get initial embeddings X_0
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        
        # Propagate through K layers
        for k in range(self.n_layers):
            all_embeddings = self.spectral_convolution(all_embeddings, self.filter_params[k])
            embeddings_list.append(all_embeddings)
        
        # Concatenate embeddings from all layers (including initial layer)
        # [X_0, X_1, ..., X_K] as in Eq. (12)
        spectralcf_all_embeddings = torch.cat(embeddings_list, dim=1)
        
        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            spectralcf_all_embeddings, [self.n_users, self.n_items]
        )
        
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        r"""Calculate the BPR loss with L2 regularization as in Eq. (13).
        
        Args:
            interaction (Interaction): The interaction data.
        
        Returns:
            torch.Tensor: The total loss.
        """
        # Clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR loss: -ln(sigma(v_r^{u⊤}v_j^i - v_r^{u⊤}v_{j'}^i))
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # L2 regularization on embeddings
        # Apply regularization on the original embeddings (ego embeddings)
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        
        reg_loss = self.reg_loss(
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings
        )

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        r"""Predict the preference score for user-item pairs.
        
        Args:
            interaction (Interaction): The interaction data.
        
        Returns:
            torch.Tensor: The predicted scores.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users (for ranking evaluation).
        
        Args:
            interaction (Interaction): The interaction data containing users.
        
        Returns:
            torch.Tensor: The predicted scores for all items.
        """
        user = interaction[self.USER_ID]
        
        # Cache embeddings for acceleration
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        # Get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # Dot with all item embeddings to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)