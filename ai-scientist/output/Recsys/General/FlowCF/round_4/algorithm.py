# -*- coding: utf-8 -*-
# @Time   : 2025-08-03
# @Author : James Caverlee
# @Email  : caverlee@cse.tamu.edu

r"""
FlowCF
################################################
Reference:
    James Caverlee et al. "FlowCF: Flow-Based Recommender" in KDD 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender, AutoEncoderMixin
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers


class FlowCF(GeneralRecommender, AutoEncoderMixin):
    r"""FlowCF is a flow-based recommender system that models the evolution of user-item interactions
    from a behavior-guided prior distribution to a target preference distribution using flow matching.
    """
    
    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)
        self.build_histroy_items(dataset)

        # Flow parameters
        self.num_steps = config["num_steps"]
        self.item_freq = self._compute_item_frequency(dataset)
        self.prior_distribution = self._create_behavior_guided_prior()
        
        # Time embedding parameters
        self.time_emb_dim = config["time_emb_dim"]
        self.max_period = config["max_period"]
        
        # MLP architecture
        input_dim = self.n_items + self.time_emb_dim
        hidden_dims = config["hidden_dims"]
        output_dim = self.n_items
        
        # Flow model
        self.mlp = MLPLayers(
            layers=[input_dim] + hidden_dims + [output_dim],
            dropout=config["dropout"],
            activation=config["activation"],
            last_activation=False
        )
        
        # Initialize parameters
        self.apply(xavier_normal_initialization)

    def predict(self, interaction):
        r"""Predict scores for a single item"""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # Get full prediction scores for all items
        scores = self.full_sort_predict(interaction)
        
        # Extract the predicted score for the specific item
        batch_size = user.shape[0]
        idx = torch.arange(batch_size, device=self.device) * self.n_items + item
        return scores[idx]

    def _compute_item_frequency(self, dataset):
        r"""Compute global item frequency vector f_i = (number of interactions for item i) / (total users)"""
        inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        item_freq = np.zeros(self.n_items)
        
        for item_id in range(self.n_items):
            item_freq[item_id] = np.sum(inter_matrix.col == item_id)
        
        item_freq = item_freq / self.n_users  # normalize by total users
        return torch.tensor(item_freq, device=self.device)

    def _create_behavior_guided_prior(self):
        r"""Create Bernoulli(p=expanded_item_freq) distribution for prior sampling"""
        # Expand item frequency to user x item shape
        expanded_freq = self.item_freq.expand(self.n_users, -1)
        return expanded_freq

    def _timestep_embedding(self, t, dim):
        r"""Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * 
            torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def _discretized_linear_interpolation(self, X0, X1, t):
        r"""Create binary mask M_t ~ Bernoulli(t) and compute X_t = M_t ⊙ X1 + (1-M_t) ⊙ X0"""
        batch_size = X0.shape[0]
        t_expanded = t.unsqueeze(1).expand(batch_size, self.n_items).to(self.device).float()
        M_t = torch.bernoulli(t_expanded)
        X_t = M_t * X1.float() + (1 - M_t) * X0.float()
        return X_t.float()

    def forward(self, X_t, t):
        r"""Predict target interactions X1 from noisy X_t at time t"""
        # Create time embeddings
        t_emb = self._timestep_embedding(t, self.time_emb_dim)
        
        # Concatenate X_t and time embeddings
        input_tensor = torch.cat([X_t, t_emb], dim=-1)
        
        # Pass through MLP
        output = self.mlp(input_tensor)
        return output

    def calculate_loss(self, interaction):
        r"""Calculate loss for training"""
        # Get batch of users and their interactions
        user = interaction[self.USER_ID]
        X1 = self.get_rating_matrix(user)  # Ground truth interactions
        
        # Sample time steps uniformly
        t = torch.randint(0, self.num_steps, (X1.shape[0],), device=self.device).float() / self.num_steps
        
        # Sample from behavior-guided prior
        X0 = torch.bernoulli(self.prior_distribution[:X1.shape[0]])
        
        # Compute noisy interpolation
        X_t = self._discretized_linear_interpolation(X0, X1, t)
        
        # Predict X1
        X1_pred = self.forward(X_t, t)
        
        # Compute loss
        loss = F.mse_loss(X1_pred, X1)
        return loss

    def full_sort_predict(self, interaction):
        r"""Make full sort prediction for all items"""
        user = interaction[self.USER_ID]
        X = self.get_rating_matrix(user)  # Observed interactions
        
        # Start from second-to-last step (s = N-2)
        batch_size = X.shape[0]
        t = torch.full((batch_size,), self.num_steps - 2, device=self.device).float() / self.num_steps
        
        # Iterative refinement (only 2 steps as per paper)
        for _ in range(2):
            # Predict X1
            X1_pred = self.forward(X, t)
            
            # Compute vector field
            v_t = (X1_pred - X) / (1 - t.unsqueeze(-1))
            
            # Update X using Euler method with binary thresholding
            X = ((X + v_t / self.num_steps) >= 0.5).float()
            
            # Preserve observed interactions (X remains 1 if originally 1)
            X = X * X + X * (1 - X)  # X = X ∨ X (identity)
            
            # Move to next time step
            t = t + 1 / self.num_steps
        
        return X.view(-1)