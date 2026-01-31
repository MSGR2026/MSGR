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

Note: This implementation is based on the paper's description of using Euler's formula
to model feature interactions in complex space with explicit order control.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.loss import RegLoss


class EulerInteractionLayer(nn.Module):
    """Euler Interaction Layer that models feature interactions in complex space.
    
    Uses Euler's formula: e^(ix) = cos(x) + i*sin(x) to represent features in polar form.
    The interaction is modeled as multiplication in complex space.
    """
    
    def __init__(self, num_fields, embedding_size, inter_orders, drop_ex=0.0, drop_im=0.0):
        """
        Args:
            num_fields: Number of feature fields
            embedding_size: Dimension of embeddings
            inter_orders: List of interaction orders to model
            drop_ex: Dropout rate for explicit (real) part
            drop_im: Dropout rate for implicit (imaginary) part
        """
        super(EulerInteractionLayer, self).__init__()
        
        self.num_fields = num_fields
        self.embedding_size = embedding_size
        self.inter_orders = inter_orders
        self.num_orders = len(inter_orders)
        
        # Order weights (softmax initialized for weighted combination)
        self.order_weights = nn.Parameter(torch.zeros(self.num_orders))
        
        # Bias terms for amplitude (lambda) and phase (theta)
        self.bias_lam = nn.Parameter(torch.zeros(1, num_fields, embedding_size))
        self.bias_theta = nn.Parameter(torch.zeros(1, num_fields, embedding_size))
        
        # Linear transformation for interaction
        self.linear_re = nn.Linear(num_fields * embedding_size, num_fields * embedding_size, bias=True)
        self.linear_im = nn.Linear(num_fields * embedding_size, num_fields * embedding_size, bias=True)
        
        # Dropout layers
        self.drop_ex = nn.Dropout(drop_ex)
        self.drop_im = nn.Dropout(drop_im)
        
        # Layer normalization for stability
        self.layer_norm_re = nn.LayerNorm(num_fields * embedding_size)
        self.layer_norm_im = nn.LayerNorm(num_fields * embedding_size)
        
        self._init_weights()
    
    def _init_weights(self):
        # Use smaller initialization for stability
        nn.init.xavier_uniform_(self.linear_re.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_im.weight, gain=0.1)
        if self.linear_re.bias is not None:
            constant_(self.linear_re.bias, 0.0)
        if self.linear_im.bias is not None:
            constant_(self.linear_im.bias, 0.0)
    
    def forward(self, r, p):
        """
        Forward pass using Euler's formula representation.
        
        Args:
            r: Real part (amplitude * cos(phase)), shape [batch_size, num_fields, embedding_size]
            p: Imaginary part (amplitude * sin(phase)), shape [batch_size, num_fields, embedding_size]
            
        Returns:
            r_out: Updated real part
            p_out: Updated imaginary part
        """
        batch_size = r.size(0)
        
        # Apply dropout
        r_drop = self.drop_ex(r)
        p_drop = self.drop_im(p)
        
        # Compute amplitude (lambda) and phase (theta) from r and p
        # r = lambda * cos(theta), p = lambda * sin(theta)
        # lambda = sqrt(r^2 + p^2), theta = atan2(p, r)
        
        # Add small epsilon for numerical stability
        eps = 1e-7
        
        # Compute amplitude with clamping for stability
        lam_sq = r_drop ** 2 + p_drop ** 2 + eps
        lam = torch.sqrt(lam_sq)
        
        # Compute phase
        theta = torch.atan2(p_drop + eps, r_drop + eps)
        
        # Add bias terms (small scale)
        lam = lam + 0.1 * self.bias_lam
        theta = theta + 0.1 * self.bias_theta
        
        # Weighted combination of different interaction orders
        order_weights = torch.softmax(self.order_weights, dim=0)
        
        # Initialize output accumulator
        r_out = torch.zeros_like(r)
        p_out = torch.zeros_like(p)
        
        for i, order in enumerate(self.inter_orders):
            # Scale amplitude and phase by order
            # Use soft power to avoid extreme values: lam^order approximated for stability
            lam_order = torch.pow(torch.clamp(lam, min=eps, max=5.0), order)
            theta_order = theta * order
            
            # Convert back to Cartesian coordinates
            r_order = lam_order * torch.cos(theta_order)
            p_order = lam_order * torch.sin(theta_order)
            
            # Accumulate weighted contribution
            r_out = r_out + order_weights[i] * r_order
            p_out = p_out + order_weights[i] * p_order
        
        # Apply linear transformation
        r_flat = r_out.view(batch_size, -1)
        p_flat = p_out.view(batch_size, -1)
        
        # Layer norm before linear for stability
        r_flat = self.layer_norm_re(r_flat)
        p_flat = self.layer_norm_im(p_flat)
        
        r_out = self.linear_re(r_flat).view(batch_size, self.num_fields, self.embedding_size)
        p_out = self.linear_im(p_flat).view(batch_size, self.num_fields, self.embedding_size)
        
        return r_out, p_out


class EulerNet(ContextRecommender):
    """EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction.
    
    EulerNet represents feature embeddings in complex space using Euler's formula,
    where amplitude captures feature importance and phase captures feature semantics.
    Feature interactions are modeled as multiplication in complex space with explicit
    order control.
    """
    
    input_type = None  # Use default POINTWISE
    
    def __init__(self, config, dataset):
        super(EulerNet, self).__init__(config, dataset)
        
        # Load parameters from config
        self.order_list = config.get("order_list", [1, 2, 3])
        self.num_layers = config.get("num_layers", 2)
        self.drop_ex = config.get("drop_ex", 0.2)
        self.drop_im = config.get("drop_im", 0.2)
        self.reg_weight = config.get("reg_weight", 1e-5)
        
        # Amplitude scaling parameter (mu in the paper) - initialized to small values
        self.mu = nn.Parameter(torch.ones(1, self.num_feature_field, self.embedding_size) * 0.5)
        
        # Build Euler interaction layers
        self.euler_layers = nn.ModuleList([
            EulerInteractionLayer(
                num_fields=self.num_feature_field,
                embedding_size=self.embedding_size,
                inter_orders=self.order_list,
                drop_ex=self.drop_ex,
                drop_im=self.drop_im
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projection layer
        # Combine real and imaginary parts for final prediction
        self.output_layer = nn.Linear(
            self.num_feature_field * self.embedding_size * 2,
            1,
            bias=True
        )
        
        # Regularization loss
        self.reg_loss = RegLoss()
        
        # Loss function
        self.loss = nn.BCEWithLogitsLoss()
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
    
    def forward(self, interaction):
        """
        Forward pass of EulerNet.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            logits: Prediction logits of shape [batch_size]
        """
        # Get embeddings: [batch_size, num_field, embed_dim]
        embed = self.concat_embed_input_fields(interaction)
        batch_size = embed.size(0)
        
        # Initialize complex representation using Euler's formula
        # x -> mu * e^(ix) = mu * cos(x) + i * mu * sin(x)
        # Here we use the embedding values as the phase
        
        # Scale embeddings to a reasonable range for phase (normalize to [-pi, pi])
        # Use tanh to bound the phase
        phase = torch.tanh(embed) * 3.14159  # Scale to approximately [-pi, pi]
        
        # Initialize amplitude with learnable mu (clamped for stability)
        amplitude = torch.clamp(self.mu.expand(batch_size, -1, -1), min=0.1, max=2.0)
        
        # Convert to Cartesian form: r = amplitude * cos(phase), p = amplitude * sin(phase)
        r = amplitude * torch.cos(phase)
        p = amplitude * torch.sin(phase)
        
        # Apply Euler interaction layers
        for euler_layer in self.euler_layers:
            r_new, p_new = euler_layer(r, p)
            # Residual connection with scaling for stability
            r = r + 0.1 * r_new
            p = p + 0.1 * p_new
        
        # Flatten and concatenate real and imaginary parts
        r_flat = r.view(batch_size, -1)
        p_flat = p.view(batch_size, -1)
        combined = torch.cat([r_flat, p_flat], dim=-1)
        
        # Add first-order linear term
        first_order = self.first_order_linear(interaction)
        
        # Final prediction
        logits = self.output_layer(combined) + first_order
        
        return logits.squeeze(-1)
    
    def calculate_loss(self, interaction):
        """
        Calculate the training loss.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            loss: Total loss value
        """
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        
        # BCE loss
        bce_loss = self.loss(output, label)
        
        # Regularization loss on Euler layers and mu
        reg_params = []
        for euler_layer in self.euler_layers:
            reg_params.extend([
                euler_layer.linear_re.weight,
                euler_layer.linear_im.weight,
                euler_layer.bias_lam,
                euler_layer.bias_theta
            ])
        reg_params.append(self.mu)
        
        reg_loss = self.reg_loss(reg_params)
        
        return bce_loss + self.reg_weight * reg_loss
    
    def predict(self, interaction):
        """
        Predict click probability.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            prob: Click probability of shape [batch_size]
        """
        return self.sigmoid(self.forward(interaction))