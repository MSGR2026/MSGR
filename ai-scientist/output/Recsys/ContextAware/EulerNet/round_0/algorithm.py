# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com
# @File   : eulernet.py

r"""
EulerNet
################################################
Reference:
    Zhen Tian et al. "EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction"
    in SIGIR 2023.

Note: The paper provided in the task description appears to be HRM (Hierarchical Representation Model) for 
next basket recommendation, but the task name is EulerNet_2021 for ContextAware recommendation.
Based on the reference implementations and domain knowledge provided, this implementation follows
the EulerNet architecture for CTR prediction using complex-space feature interactions via Euler's formula.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.loss import RegLoss


class EulerInteractionLayer(nn.Module):
    """Euler Interaction Layer that models feature interactions in complex space.
    
    Uses Euler's formula: e^(ix) = cos(x) + i*sin(x) to represent features in polar form.
    The interaction is modeled through amplitude (r) and phase (p) components.
    """
    
    def __init__(self, num_fields, embedding_size, inter_orders, drop_ex=0.0, drop_im=0.0):
        super(EulerInteractionLayer, self).__init__()
        
        self.num_fields = num_fields
        self.embedding_size = embedding_size
        self.inter_orders = inter_orders  # Number of interaction orders
        
        # Shared linear transformation for imaginary part
        self.im_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        
        # Bias terms for amplitude (lambda) and phase (theta)
        self.bias_lam = nn.Parameter(torch.zeros(1, num_fields, embedding_size))
        self.bias_theta = nn.Parameter(torch.zeros(1, num_fields, embedding_size))
        
        # Order weights (initialized with softmax-like distribution)
        self.order_weights = nn.Parameter(torch.ones(inter_orders) / inter_orders)
        
        # Dropout layers
        self.drop_ex = nn.Dropout(drop_ex)
        self.drop_im = nn.Dropout(drop_im)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_size)
        
        self._init_weights()
    
    def _init_weights(self):
        xavier_normal_(self.im_linear.weight)
        nn.init.zeros_(self.bias_lam)
        nn.init.zeros_(self.bias_theta)
    
    def forward(self, x, mu):
        """
        Args:
            x: Input embeddings [batch_size, num_fields, embedding_size]
            mu: Amplitude scaling factor [batch_size, num_fields, embedding_size]
        
        Returns:
            r: Real part of complex representation
            p: Imaginary part of complex representation
        """
        batch_size = x.shape[0]
        
        # Apply dropout to input
        x_ex = self.drop_ex(x)
        x_im = self.drop_im(x)
        
        # Transform to polar coordinates using Euler's formula
        # r = mu * cos(x), p = mu * sin(x)
        r = mu * torch.cos(x_ex)  # Real part (amplitude * cos)
        p = mu * torch.sin(x_im)  # Imaginary part (amplitude * sin)
        
        # Apply shared linear transformation to imaginary part
        p = self.im_linear(p)
        
        # Multi-order interactions
        r_orders = [r]
        p_orders = [p]
        
        for order in range(1, self.inter_orders):
            # Higher order interactions through element-wise operations
            # Using log-exp trick for numerical stability in multiplication
            r_log = torch.log(torch.abs(r_orders[-1]) + 1e-8)
            p_log = torch.log(torch.abs(p_orders[-1]) + 1e-8)
            
            # Clamp for numerical stability
            r_log = torch.clamp(r_log, min=-10, max=10)
            p_log = torch.clamp(p_log, min=-10, max=10)
            
            # Interaction across fields (mean pooling followed by broadcast)
            r_agg = r_log.mean(dim=1, keepdim=True)
            p_agg = p_log.mean(dim=1, keepdim=True)
            
            # Combine with previous order
            r_new = torch.exp(r_log + r_agg)
            p_new = torch.exp(p_log + p_agg)
            
            # Clamp output
            r_new = torch.clamp(r_new, min=-1e6, max=1e6)
            p_new = torch.clamp(p_new, min=-1e6, max=1e6)
            
            r_orders.append(r_new)
            p_orders.append(p_new)
        
        # Weighted sum of different orders
        order_weights = torch.softmax(self.order_weights, dim=0)
        
        r_out = sum(w * r_o for w, r_o in zip(order_weights, r_orders))
        p_out = sum(w * p_o for w, p_o in zip(order_weights, p_orders))
        
        # Add bias terms
        r_out = r_out + self.bias_lam
        p_out = p_out + self.bias_theta
        
        # Apply layer normalization
        r_out = self.layer_norm(r_out)
        p_out = self.layer_norm(p_out)
        
        return r_out, p_out


class EulerNet(ContextRecommender):
    """EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction.
    
    EulerNet models feature interactions in complex space using Euler's formula.
    It represents each feature as a complex number with amplitude and phase,
    enabling adaptive and expressive feature interaction learning.
    """
    
    def __init__(self, config, dataset):
        super(EulerNet, self).__init__(config, dataset)
        
        # Load parameters from config
        self.n_layers = config.get("n_layers", 2)
        self.inter_orders = config.get("inter_orders", 3)
        self.drop_ex = config.get("drop_ex", 0.2)
        self.drop_im = config.get("drop_im", 0.2)
        self.reg_weight = config.get("reg_weight", 0.0)
        
        # Amplitude parameter mu (learnable per field and dimension)
        self.mu = nn.Parameter(torch.ones(1, self.num_feature_field, self.embedding_size))
        
        # Stack of Euler interaction layers
        self.euler_interaction_layers = nn.ModuleList([
            EulerInteractionLayer(
                num_fields=self.num_feature_field,
                embedding_size=self.embedding_size,
                inter_orders=self.inter_orders,
                drop_ex=self.drop_ex,
                drop_im=self.drop_im
            )
            for _ in range(self.n_layers)
        ])
        
        # Output projection: combine real and imaginary parts to single logit
        # First aggregate across fields, then project to scalar
        self.output_layer = nn.Sequential(
            nn.Linear(self.num_feature_field * self.embedding_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.drop_ex),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Alternative simpler output: direct projection
        self.reg_layer = nn.Linear(self.num_feature_field * self.embedding_size * 2, 1)
        
        # Loss function
        self.loss = nn.BCEWithLogitsLoss()
        self.reg_loss = RegLoss()
        
        # Sigmoid for prediction
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
    
    def euler_forward(self, x):
        """Forward pass through Euler interaction layers.
        
        Args:
            x: Input embeddings [batch_size, num_fields, embedding_size]
        
        Returns:
            Combined real and imaginary representations
        """
        batch_size = x.shape[0]
        
        # Expand mu for batch
        mu = self.mu.expand(batch_size, -1, -1)
        
        # Initial transformation to polar form
        r = mu * torch.cos(x)
        p = mu * torch.sin(x)
        
        # Pass through Euler interaction layers
        for layer in self.euler_interaction_layers:
            r, p = layer(x, mu)
            # Residual connection with input
            x = x + r  # Update for next layer
        
        # Flatten and concatenate real and imaginary parts
        r_flat = r.view(batch_size, -1)
        p_flat = p.view(batch_size, -1)
        
        combined = torch.cat([r_flat, p_flat], dim=-1)
        
        return combined
    
    def forward(self, interaction):
        """Forward pass for CTR prediction.
        
        Args:
            interaction: Input interaction data
        
        Returns:
            Predicted logits [batch_size]
        """
        # Get embeddings from all feature fields
        euler_all_embeddings = self.concat_embed_input_fields(interaction)
        # Shape: [batch_size, num_feature_field, embedding_size]
        
        # First order linear term
        first_order = self.first_order_linear(interaction)
        
        # Euler interaction
        euler_output = self.euler_forward(euler_all_embeddings)
        
        # Project to logit
        logit = self.reg_layer(euler_output)
        
        # Combine with first order
        output = first_order + logit
        
        return output.squeeze(-1)
    
    def calculate_loss(self, interaction):
        """Calculate training loss.
        
        Args:
            interaction: Input interaction data with labels
        
        Returns:
            Total loss (BCE + regularization)
        """
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        
        # BCE loss
        bce_loss = self.loss(output, label)
        
        # Regularization loss on Euler layers and mu
        if self.reg_weight > 0:
            reg_params = [self.mu]
            for layer in self.euler_interaction_layers:
                reg_params.extend([layer.bias_lam, layer.bias_theta, layer.order_weights])
            
            reg_loss = self.reg_weight * sum(
                torch.norm(p, p=2) for p in reg_params if p.requires_grad
            )
            return bce_loss + reg_loss
        
        return bce_loss
    
    def predict(self, interaction):
        """Predict click probability.
        
        Args:
            interaction: Input interaction data
        
        Returns:
            Click probabilities [batch_size]
        """
        return self.sigmoid(self.forward(interaction))