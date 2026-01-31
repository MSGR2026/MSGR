# -*- coding: utf-8 -*-
# @Time   : 2024/01/09
# @Author : RecBole Team
# @Email  : recbole@gmail.com

r"""
EASE
################################################
Reference:
    Harald Steck. "Embarrassingly Shallow Autoencoders for Sparse Data." in WWW 2019.

"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class EASE(GeneralRecommender):
    r"""EASE is a simple linear model for collaborative filtering that solves a constrained 
    optimization problem in closed-form. The model learns an item-item weight matrix B with 
    zero diagonal constraint to prevent trivial self-similarity solutions.
    
    The prediction is computed as: S = X @ B, where X is the user-item interaction matrix.
    The weight matrix B is learned by solving: min ||X - XB||_F^2 + λ||B||_F^2, s.t. diag(B) = 0
    
    The closed-form solution is: B = I - P @ diagMat(1 / diag(P)), where P = (X^T X + λI)^(-1)
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(EASE, self).__init__(config, dataset)

        # Get regularization parameter
        self.reg_weight = config["reg_weight"]
        
        # Get the user-item interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        
        # Train the model immediately (closed-form solution)
        self.item_similarity = self._train_ease()
        
        # Dummy parameter to satisfy RecBole's requirement
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _train_ease(self):
        r"""Train EASE model using closed-form solution.
        
        Algorithm:
        1. Compute Gram matrix G = X^T X
        2. Add regularization to diagonal: G_reg = G + λI
        3. Compute inverse: P = (G_reg)^(-1)
        4. Compute B: B_ij = -P_ij / P_jj for i ≠ j, B_ii = 0
        
        Returns:
            np.ndarray: Item-item similarity matrix B of shape (n_items, n_items)
        """
        X = self.interaction_matrix
        
        # Compute Gram matrix G = X^T @ X (item-item co-occurrence matrix)
        G = (X.T @ X).toarray().astype(np.float32)
        
        # Add L2 regularization to diagonal
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.reg_weight
        
        # Compute the inverse: P = (G + λI)^(-1)
        P = np.linalg.inv(G)
        
        # Compute B according to Eq. 8:
        # B_ij = 0 if i == j
        # B_ij = -P_ij / P_jj otherwise
        B = np.identity(self.n_items) - P / np.diag(P)
        
        # Ensure diagonal is exactly zero (numerical stability)
        B[diag_indices] = 0.0
        
        return B.astype(np.float32)

    def forward(self):
        r"""This method is not used in EASE as it's a traditional model with closed-form solution."""
        pass

    def calculate_loss(self, interaction):
        r"""Calculate loss. For EASE, training is done in closed-form, so this returns a dummy loss.
        
        Args:
            interaction (Interaction): Interaction object containing user-item pairs
            
        Returns:
            torch.Tensor: Dummy loss (always 0)
        """
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        r"""Predict scores for user-item pairs.
        
        Args:
            interaction (Interaction): Interaction object containing user and item indices
            
        Returns:
            torch.Tensor: Predicted scores for the user-item pairs
        """
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        # Get user interaction history
        user_interactions = self.interaction_matrix[user, :].toarray()

        # Compute scores: S_u,j = X_u,· @ B_·,j
        # For specific items, we only need the corresponding columns of B
        scores = (user_interactions @ self.item_similarity[:, item]).diagonal()

        # Add small noise to break ties for ranking
        scores_tensor = torch.from_numpy(scores).float()
        scores_tensor = scores_tensor + 1e-7 * torch.rand(scores_tensor.shape)
        
        return scores_tensor.to(self.device)

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.
        
        Args:
            interaction (Interaction): Interaction object containing user indices
            
        Returns:
            torch.Tensor: Predicted scores for all items, shape: (batch_size * n_items,)
        """
        user = interaction[self.USER_ID].cpu().numpy()

        # Get user interaction history
        user_interactions = self.interaction_matrix[user, :].toarray()

        # Compute scores for all items: S = X @ B
        scores = user_interactions @ self.item_similarity

        # Flatten to (batch_size * n_items,)
        scores = scores.flatten()

        # Add small noise to break ties for ranking
        scores_tensor = torch.from_numpy(scores).float()
        scores_tensor = scores_tensor + 1e-7 * torch.rand(scores_tensor.shape)
        
        return scores_tensor.to(self.device)