# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
NCEPLRec
################################################
Reference:
    Ga Wu et al. "Noise Contrastive Estimation for Scalable Learning of Discrete MRFs." in RecSys 2021.

NCE-PLRec uses Noise Contrastive Estimation to derive improved item embeddings
by constructing a popularity-debiased matrix D, then applying truncated SVD
followed by ridge regression for prediction.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class NCEPLRec(GeneralRecommender):
    r"""NCE-PLRec is a scalable one-class collaborative filtering method that uses
    Noise Contrastive Estimation to derive depopularized item embeddings, followed
    by projected linear regression for recommendation.
    
    The algorithm:
    1. Construct matrix D with NCE-based reweighting: d_{i,j} = max(log(sum_popularity) - beta * log(item_popularity), 0)
    2. Apply truncated SVD to D to get item embeddings V
    3. Project user interactions onto item embeddings: Q = R @ V
    4. Solve ridge regression: W = (Q^T Q + lambda I)^{-1} Q^T R
    5. Predict: scores = Q @ W^T
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(NCEPLRec, self).__init__(config, dataset)

        # Load hyperparameters
        self.beta = config['beta']  # NCE popularity penalty hyperparameter
        self.rank = config['rank']  # Truncated SVD rank (embedding dimension)
        self.reg_weight = config['reg_weight']  # Ridge regression regularization (lambda)
        self.seed = config.get('seed', 42)  # Random seed for reproducibility

        # Get the implicit feedback matrix R (users x items)
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        
        # Compute user and item embeddings using NCE-PLRec algorithm
        self.user_embeddings, self.item_embeddings = self._train_nceplrec()
        
        # Convert to torch tensors and move to device
        self.user_embeddings = torch.from_numpy(self.user_embeddings).float().to(self.device)
        self.item_embeddings = torch.from_numpy(self.item_embeddings).float().to(self.device)
        
        # Dummy parameter for compatibility (no actual training needed)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _construct_nce_matrix(self, R):
        """
        Construct the NCE-reweighted matrix D from implicit feedback matrix R.
        
        For observed interactions (r_{i,j} = 1):
            d_{i,j} = max(log(sum_j' |r_{.,j'}|) - beta * log(|r_{.,j}|), 0)
        For unobserved interactions:
            d_{i,j} = 0
            
        Args:
            R: scipy.sparse.csr_matrix, shape (n_users, n_items)
            
        Returns:
            D: scipy.sparse.csr_matrix, shape (n_users, n_items)
        """
        # Compute item popularity: |r_{.,j}| for each item j
        item_popularity = np.array(R.sum(axis=0)).flatten()  # shape: (n_items,)
        
        # Total popularity sum: sum_j' |r_{.,j'}|
        total_popularity = item_popularity.sum()
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        item_popularity = np.maximum(item_popularity, epsilon)
        
        # Compute log values
        log_total = np.log(total_popularity + epsilon)
        log_item_pop = np.log(item_popularity)
        
        # Create D matrix with same sparsity pattern as R
        D = R.copy()
        D = D.tocoo()
        
        # For each non-zero entry, compute the NCE weight
        # d_{i,j} = max(log(total_pop) - beta * log(item_pop[j]), 0)
        new_data = np.maximum(log_total - self.beta * log_item_pop[D.col], 0)
        
        D_csr = csr_matrix((new_data, (D.row, D.col)), shape=R.shape)
        
        return D_csr

    def _train_nceplrec(self):
        """
        Train NCE-PLRec model using closed-form solution.
        
        Algorithm:
        1. D* <- NCE(R, beta)  # Construct D matrix
        2. U_D, Sigma_D, V_D^T <- Truncated SVD(D*)
        3. V* = V_D @ Sigma_D^{1/2}  # Item embeddings
        4. Q <- R @ V*  # Project implicit matrix to get user representations
        5. W <- (Q^T Q + lambda I)^{-1} Q^T R  # Ridge regression
        6. Return Q, W for prediction: Q @ W^T
        
        Returns:
            user_embeddings: np.ndarray, shape (n_users, rank)
            item_embeddings: np.ndarray, shape (n_items, rank)
        """
        R = self.interaction_matrix
        
        # Step 1: Construct NCE-reweighted matrix D
        D = self._construct_nce_matrix(R)
        
        # Step 2: Truncated SVD on D
        # D ≈ U_D @ diag(Sigma_D) @ V_D^T
        U_D, Sigma_D, V_D_T = randomized_svd(
            D, 
            n_components=self.rank,
            random_state=self.seed
        )
        # U_D: (n_users, rank)
        # Sigma_D: (rank,)
        # V_D_T: (rank, n_items)
        
        V_D = V_D_T.T  # (n_items, rank)
        
        # Step 3: Compute item embeddings V* = V_D @ Sigma_D^{1/2}
        Sigma_sqrt = np.sqrt(Sigma_D)  # (rank,)
        V_star = V_D * Sigma_sqrt  # (n_items, rank)
        
        # Step 4: Project implicit matrix R onto item embeddings
        # Q = R @ V* (user representation as sum of item embeddings)
        Q = R.dot(V_star)  # (n_users, rank)
        
        # Step 5: Solve ridge regression W = (Q^T Q + lambda I)^{-1} Q^T R
        # For alpha = 0 (no user-item weighting), this is a global closed-form solution
        
        # Q^T Q: (rank, rank)
        QtQ = Q.T.dot(Q)
        
        # Add regularization: Q^T Q + lambda I
        QtQ_reg = QtQ + self.reg_weight * np.eye(self.rank)
        
        # Q^T R: (rank, n_items)
        QtR = Q.T.dot(R.toarray())
        
        # Solve for W: W^T = (Q^T Q + lambda I)^{-1} Q^T R
        # W: (n_items, rank)
        W_T = np.linalg.solve(QtQ_reg, QtR)  # (rank, n_items)
        W = W_T.T  # (n_items, rank)
        
        # User embeddings are Q, item embeddings are W
        # Prediction: Q @ W^T
        return Q, W

    def forward(self, user):
        """
        Get user embeddings for given user indices.
        
        Args:
            user: torch.LongTensor, shape (batch_size,)
            
        Returns:
            user_e: torch.FloatTensor, shape (batch_size, rank)
        """
        return self.user_embeddings[user]

    def calculate_loss(self, interaction):
        """
        NCE-PLRec is a closed-form method with no gradient-based training.
        Return a dummy zero loss for compatibility with RecBole framework.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def predict(self, interaction):
        """
        Predict scores for given user-item pairs.
        
        Args:
            interaction: Interaction object containing USER_ID and ITEM_ID
            
        Returns:
            scores: torch.FloatTensor, shape (batch_size,)
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_e = self.user_embeddings[user]  # (batch_size, rank)
        item_e = self.item_embeddings[item]  # (batch_size, rank)
        
        # Score is dot product: user_e @ item_e^T (element-wise then sum)
        scores = torch.mul(user_e, item_e).sum(dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        
        Args:
            interaction: Interaction object containing USER_ID
            
        Returns:
            scores: torch.FloatTensor, shape (batch_size * n_items,)
        """
        user = interaction[self.USER_ID]
        
        user_e = self.user_embeddings[user]  # (batch_size, rank)
        
        # Compute scores for all items: user_e @ item_embeddings^T
        # scores: (batch_size, n_items)
        scores = torch.matmul(user_e, self.item_embeddings.T)
        
        return scores.view(-1)