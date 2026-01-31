# -*- coding: utf-8 -*-
# @Time   : 2025
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
FlowCF
################################################
Reference:
    "FlowCF: A Flow-Based Recommender for Collaborative Filtering" in KDD 2025.

FlowCF is a flow-based recommender system that models the evolution of user-item 
interactions from a noisy source distribution to a target preference distribution.
Key innovations include:
1. Behavior-Guided Prior: Uses item popularity-based Bernoulli distribution instead of Gaussian
2. Discrete Flow Framework: Preserves binary nature of implicit feedback
3. Efficient Inference: Achieves good performance with just 2 sampling steps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils import InputType


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element. (N,)
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FlowModel(nn.Module):
    """
    Flow model implemented as MLP with timestep conditioning.
    Learns to predict the target interactions X_1 from noisy interpolation X_t.
    """
    
    def __init__(self, n_items, hidden_dims, emb_size, dropout=0.5, activation='tanh', norm=False):
        super(FlowModel, self).__init__()
        self.n_items = n_items
        self.emb_size = emb_size
        self.norm = norm
        
        # Timestep embedding layer
        self.time_emb_layer = nn.Linear(emb_size, emb_size)
        
        # Input dimension includes timestep embedding (concatenation)
        input_dim = n_items + emb_size
        
        # Build MLP layers
        dims = [input_dim] + list(hidden_dims) + [n_items]
        self.mlp = MLPLayers(
            layers=dims,
            dropout=dropout,
            activation=activation,
            last_activation=False
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_t, t):
        """
        Forward pass of the flow model.
        
        Args:
            x_t: Noisy interaction matrix at timestep t [batch_size, n_items]
            t: Timestep values [batch_size]
        Returns:
            x_1_pred: Predicted target interactions [batch_size, n_items]
        """
        # Create timestep embedding
        time_emb = timestep_embedding(t, self.emb_size)
        time_emb = self.time_emb_layer(time_emb)
        
        # Normalize input if specified
        if self.norm:
            x_t = F.normalize(x_t, p=2, dim=1)
        
        # Apply dropout to input
        x_t = self.dropout(x_t)
        
        # Concatenate input with timestep embedding
        h = torch.cat([x_t, time_emb], dim=-1)
        
        # Pass through MLP
        x_1_pred = self.mlp(h)
        
        return x_1_pred


class FlowCF(GeneralRecommender, AutoEncoderMixin):
    """
    FlowCF: A Flow-Based Recommender for Collaborative Filtering.
    
    This model uses flow matching to learn the transformation from a behavior-guided
    prior distribution to the target user preference distribution.
    
    Key components:
    1. Behavior-Guided Prior: Bernoulli distribution based on item popularity
    2. Discrete Flow Framework: Maintains binary structure of interactions
    3. Discretized Linear Interpolation: Uses binary masks for interpolation
    4. Efficient Inference: Can start from later timesteps
    """
    
    input_type = InputType.LISTWISE
    
    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)
        
        # Model hyperparameters
        self.hidden_dims = config['hidden_dims']
        self.emb_size = config['embedding_size']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.norm = config['norm']
        self.steps = config['steps']  # Number of discretization steps N
        self.sampling_steps = config['sampling_steps']  # Number of inference steps
        self.start_step = config['start_step']  # Starting step for inference
        
        # Build history items for AutoEncoderMixin
        self.build_histroy_items(dataset)
        
        # Compute behavior-guided prior (item frequency vector)
        self.register_buffer('item_freq', self._compute_item_frequency(dataset))
        
        # Build flow model
        self.flow_model = FlowModel(
            n_items=self.n_items,
            hidden_dims=self.hidden_dims,
            emb_size=self.emb_size,
            dropout=self.dropout,
            activation=self.activation,
            norm=self.norm
        )
        
        # Initialize parameters
        self.apply(xavier_normal_initialization)
    
    def _compute_item_frequency(self, dataset):
        """
        Compute global item frequency vector f for behavior-guided prior.
        f_i = number of interactions for item i / total number of users
        
        Args:
            dataset: RecBole dataset object
        Returns:
            item_freq: Item frequency vector [n_items]
        """
        # Get interaction matrix
        inter_matrix = dataset.inter_matrix(form='coo').astype('float32')
        
        # Sum interactions per item
        item_counts = torch.zeros(self.n_items, dtype=torch.float32)
        for item_id in inter_matrix.col:
            if item_id < self.n_items:
                item_counts[item_id] += 1
        
        # Normalize by number of users
        n_users = dataset.user_num
        item_freq = item_counts / n_users
        
        # Clamp to avoid extreme values (0 or 1)
        item_freq = torch.clamp(item_freq, min=1e-6, max=1 - 1e-6)
        
        return item_freq
    
    def _sample_from_prior(self, batch_size):
        """
        Sample X_0 from the behavior-guided prior distribution.
        X_0 ~ Bernoulli(f) where f is the item frequency vector.
        
        Args:
            batch_size: Number of samples to generate
        Returns:
            X_0: Sampled binary interaction matrix [batch_size, n_items]
        """
        # Expand item frequency to batch size
        probs = self.item_freq.unsqueeze(0).expand(batch_size, -1)
        
        # Sample from Bernoulli distribution
        X_0 = torch.bernoulli(probs)
        
        return X_0
    
    def _discrete_interpolation(self, X_0, X_1, t):
        """
        Compute discretized linear interpolation X_t.
        X_t = M_t ⊙ X_1 + (1 - M_t) ⊙ X_0
        where M_t is a binary mask with each element sampled from Bernoulli(t).
        
        Args:
            X_0: Source sample from prior [batch_size, n_items]
            X_1: Target interactions [batch_size, n_items]
            t: Timestep values [batch_size]
        Returns:
            X_t: Interpolated interactions [batch_size, n_items]
        """
        batch_size = X_0.shape[0]
        
        # Expand t to match shape [batch_size, n_items]
        t_expanded = t.unsqueeze(1).expand(-1, self.n_items)
        
        # Sample binary mask M_t from Bernoulli(t)
        M_t = torch.bernoulli(t_expanded)
        
        # Compute discretized interpolation
        X_t = M_t * X_1 + (1 - M_t) * X_0
        
        return X_t
    
    def forward(self, X_t, t):
        """
        Forward pass: predict target interactions from noisy interpolation.
        
        Args:
            X_t: Noisy interaction matrix [batch_size, n_items]
            t: Timestep values [batch_size]
        Returns:
            X_1_pred: Predicted target interactions [batch_size, n_items]
        """
        return self.flow_model(X_t, t)
    
    def calculate_loss(self, interaction):
        """
        Calculate training loss for FlowCF.
        Loss = E[||X_1_pred - X_1||^2]
        
        Args:
            interaction: Interaction dict containing USER_ID
        Returns:
            loss: Scalar loss value
        """
        user = interaction[self.USER_ID]
        
        # Get target interactions X_1
        X_1 = self.get_rating_matrix(user)
        batch_size = X_1.shape[0]
        
        # Sample X_0 from behavior-guided prior
        X_0 = self._sample_from_prior(batch_size)
        
        # Sample random timesteps t from discrete steps T
        # t_i = i/N for i = 0, 1, ..., N
        step_indices = torch.randint(0, self.steps, (batch_size,), device=self.device)
        t = step_indices.float() / self.steps
        
        # Compute discretized interpolation X_t
        X_t = self._discrete_interpolation(X_0, X_1, t)
        
        # Predict target interactions
        X_1_pred = self.forward(X_t, t)
        
        # Compute MSE loss
        loss = F.mse_loss(X_1_pred, X_1)
        
        return loss
    
    def _inference_step(self, X_t, t, step_size):
        """
        Perform one inference step with discrete update rule.
        
        Args:
            X_t: Current interaction state [batch_size, n_items]
            t: Current timestep [batch_size]
            step_size: Step size (1/N)
        Returns:
            X_t_next: Updated interaction state [batch_size, n_items]
        """
        # Predict target interactions
        X_1_pred = self.forward(X_t, t)
        
        # Compute predicted vector field v_t = (X_1_pred - X_t) / (1 - t)
        t_expanded = t.unsqueeze(1).expand(-1, self.n_items)
        v_t = (X_1_pred - X_t) / (1 - t_expanded + 1e-8)
        
        # Discrete update: X_{t+1/N}^i = 1 if X_t^i + v_t^i/N >= 0.5 else 0
        X_t_next = (X_t + v_t * step_size >= 0.5).float()
        
        return X_t_next, X_1_pred
    
    def p_sample(self, X):
        """
        Generate predicted interactions through iterative flow inference.
        
        Args:
            X: Observed user interactions [batch_size, n_items]
        Returns:
            X_pred: Predicted interactions [batch_size, n_items]
        """
        batch_size = X.shape[0]
        step_size = 1.0 / self.steps
        
        # Initialize X_t with observed interactions
        X_t = X.clone()
        
        # Determine starting step (can start from later step since X contains meaningful info)
        start_idx = self.start_step
        
        # Iterative sampling
        for i in range(start_idx, self.steps - 1):
            t = torch.full((batch_size,), i / self.steps, device=self.device)
            
            # Perform inference step
            X_t, _ = self._inference_step(X_t, t, step_size)
            
            # Preserve observed interactions: X_t = X_t ∨ X
            X_t = torch.max(X_t, X)
        
        # Final prediction at last step
        t = torch.full((batch_size,), (self.steps - 1) / self.steps, device=self.device)
        X_pred = self.forward(X_t, t)
        
        return X_pred
    
    def predict(self, interaction):
        """
        Predict scores for specific user-item pairs.
        
        Args:
            interaction: Interaction dict with USER_ID and ITEM_ID
        Returns:
            scores: Predicted scores [batch_size]
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # Get user interactions
        X = self.get_rating_matrix(user)
        
        # Generate predictions
        scores = self.p_sample(X)
        
        # Extract scores for specific items
        return scores[torch.arange(len(item)).to(self.device), item]
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        
        Args:
            interaction: Interaction dict with USER_ID
        Returns:
            scores: Predicted scores [batch_size, n_items] flattened
        """
        user = interaction[self.USER_ID]
        
        # Get user interactions
        X = self.get_rating_matrix(user)
        
        # Generate predictions
        scores = self.p_sample(X)
        
        return scores.view(-1)