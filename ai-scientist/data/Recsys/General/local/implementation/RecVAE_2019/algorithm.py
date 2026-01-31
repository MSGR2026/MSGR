# -*- coding: utf-8 -*-
# @Time   : 2025-01-15
# @Author : RecBole Team (Generated)
# @Email  : recbole@example.com

r"""
RecVAE
################################################
Reference:
    Ilya Shenbin et al. "RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback" in WSDM 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class DenseEncoder(nn.Module):
    """Dense encoder with layer normalization and swish activation"""
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_prob=0.5):
        super(DenseEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        
        # Build dense layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim += hidden_dim  # Dense connection: concatenate all previous outputs
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        """
        Args:
            x: input tensor [batch_size, input_dim]
        Returns:
            mu: mean of latent distribution [batch_size, latent_dim]
            logvar: log variance of latent distribution [batch_size, latent_dim]
        """
        h = x
        outputs = [h]
        
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            h = layer(h)
            h = layer_norm(h)
            h = self.swish(h)
            h = self.dropout(h)
            outputs.append(h)
            h = torch.cat(outputs, dim=1)  # Dense connection
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class RecVAE(GeneralRecommender, AutoEncoderMixin):
    r"""RecVAE is a variational autoencoder for top-N recommendation with implicit feedback.
    
    It introduces several novel techniques:
    1. Composite prior that mixes standard Gaussian with previous epoch's posterior
    2. User-specific KL weight scaling proportional to feedback amount
    3. Alternating training between encoder and decoder
    4. Dense encoder architecture with layer normalization and swish activation
    5. Denoising applied only during encoder training
    """
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(RecVAE, self).__init__(config, dataset)
        
        # Load parameters
        self.latent_dim = config["latent_dimension"]
        self.hidden_dims = config["hidden_dimension"]
        self.dropout_prob = config["dropout_prob"]
        self.gamma = config["gamma"]  # KL weight coefficient
        self.alpha = config["alpha"]  # Composite prior mixture weight
        self.noise_prob = config["noise_prob"]  # Bernoulli noise probability for denoising
        self.n_enc_epochs = config["n_enc_epochs"]  # Number of encoder update steps per epoch
        self.n_dec_epochs = config["n_dec_epochs"]  # Number of decoder update steps per epoch
        
        self.build_histroy_items(dataset)
        
        # Build encoder (inference network)
        self.encoder = DenseEncoder(
            input_dim=self.n_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout_prob=self.dropout_prob
        )
        
        # Build decoder (generative network) - single linear layer
        self.decoder = nn.Linear(self.latent_dim, self.n_items)
        
        # Store previous epoch's encoder parameters for composite prior
        self.encoder_old_state = None
        
        # Training phase flag
        self.training_encoder = True
        
        # Cache for evaluation
        self.restore_user_e = None
        self.restore_item_e = None
        
        # Initialize parameters
        self.apply(xavier_normal_initialization)
        
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x, use_noise=True):
        """
        Encode input to latent distribution
        Args:
            x: input rating matrix [batch_size, n_items]
            use_noise: whether to apply input noise (for denoising)
        Returns:
            mu: mean [batch_size, latent_dim]
            logvar: log variance [batch_size, latent_dim]
        """
        # Apply Bernoulli noise for denoising (only during encoder training)
        if use_noise and self.training and self.training_encoder:
            x = F.dropout(x, p=self.noise_prob, training=True)
        
        # Normalize input
        x = F.normalize(x, p=2, dim=1)
        
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        """
        Decode latent code to item logits
        Args:
            z: latent code [batch_size, latent_dim]
        Returns:
            logits: item logits [batch_size, n_items]
        """
        return self.decoder(z)
    
    def forward(self, rating_matrix, use_noise=True):
        """
        Forward pass through encoder and decoder
        Args:
            rating_matrix: user-item interaction matrix [batch_size, n_items]
            use_noise: whether to apply input noise
        Returns:
            recon_logits: reconstructed item logits [batch_size, n_items]
            mu: latent mean [batch_size, latent_dim]
            logvar: latent log variance [batch_size, latent_dim]
        """
        mu, logvar = self.encode(rating_matrix, use_noise=use_noise)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar
    
    def compute_kl_divergence(self, mu, logvar, rating_matrix):
        """
        Compute KL divergence between approximate posterior and composite prior
        Args:
            mu: posterior mean [batch_size, latent_dim]
            logvar: posterior log variance [batch_size, latent_dim]
            rating_matrix: user interaction matrix for computing prior [batch_size, n_items]
        Returns:
            kl: KL divergence [batch_size]
        """
        # Sample z from current posterior using reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Compute log probability under current posterior q_phi(z|x)
        log_q = -0.5 * torch.sum(
            logvar + ((z - mu) ** 2) / torch.exp(logvar), dim=1
        )
        
        # Compute log probability under composite prior p(z|phi_old, x)
        # p(z) = alpha * N(0, I) + (1-alpha) * q_phi_old(z|x)
        
        # Standard Gaussian component: log N(z|0, I)
        log_p_standard = -0.5 * torch.sum(z ** 2, dim=1)
        
        # Previous posterior component: log q_phi_old(z|x)
        if self.encoder_old_state is not None and self.alpha < 1.0:
            # Use old encoder parameters to compute old posterior
            with torch.no_grad():
                # Temporarily load old parameters
                current_state = self.encoder.state_dict()
                self.encoder.load_state_dict(self.encoder_old_state)
                
                # Compute old posterior (no noise for prior computation)
                mu_old, logvar_old = self.encode(rating_matrix, use_noise=False)
                
                # Restore current parameters
                self.encoder.load_state_dict(current_state)
            
            log_q_old = -0.5 * torch.sum(
                logvar_old + ((z - mu_old) ** 2) / torch.exp(logvar_old), dim=1
            )
            
            # Mixture: log(alpha * exp(log_p_standard) + (1-alpha) * exp(log_q_old))
            max_log = torch.max(log_p_standard, log_q_old)
            log_p_composite = max_log + torch.log(
                self.alpha * torch.exp(log_p_standard - max_log) +
                (1 - self.alpha) * torch.exp(log_q_old - max_log)
            )
        else:
            # First epoch or alpha=1: use only standard Gaussian
            log_p_composite = log_p_standard
        
        # KL divergence: log q(z|x) - log p(z)
        kl = log_q - log_p_composite
        
        return kl
    
    def calculate_loss(self, interaction):
        """
        Calculate loss for training
        Args:
            interaction: interaction dict with USER_ID
        Returns:
            loss: scalar loss value
        """
        # Clear cache at the beginning of each batch
        self.restore_user_e = None
        self.restore_item_e = None
        
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)
        
        if self.training_encoder:
            # Encoder training: use denoising and KL divergence
            recon_logits, mu, logvar = self.forward(rating_matrix, use_noise=True)
            
            # Reconstruction loss: multinomial log-likelihood (cross-entropy)
            # log p(x|z) = sum_i x_i * log(softmax(logits)_i)
            log_softmax_logits = F.log_softmax(recon_logits, dim=1)
            recon_loss = -(log_softmax_logits * rating_matrix).sum(1)
            
            # KL divergence with user-specific weight
            kl_divergence = self.compute_kl_divergence(mu, logvar, rating_matrix)
            
            # User-specific KL weight: beta'(x) = gamma * |X_u^o|
            user_feedback_count = rating_matrix.sum(1)
            kl_weight = self.gamma * user_feedback_count
            
            # Total loss
            loss = (recon_loss + kl_weight * kl_divergence).mean()
        else:
            # Decoder training: no denoising, no KL divergence
            recon_logits, mu, logvar = self.forward(rating_matrix, use_noise=False)
            
            # Only reconstruction loss
            log_softmax_logits = F.log_softmax(recon_logits, dim=1)
            recon_loss = -(log_softmax_logits * rating_matrix).sum(1)
            
            loss = recon_loss.mean()
        
        return loss
    
    def predict(self, interaction):
        """
        Predict scores for user-item pairs
        Args:
            interaction: interaction dict with USER_ID and ITEM_ID
        Returns:
            scores: predicted scores [batch_size]
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        rating_matrix = self.get_rating_matrix(user)
        
        # Encode (no noise during inference)
        mu, _ = self.encode(rating_matrix, use_noise=False)
        
        # Decode
        logits = self.decode(mu)
        scores = F.softmax(logits, dim=1)
        
        return scores[torch.arange(len(item)).to(self.device), item]
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users
        Args:
            interaction: interaction dict with USER_ID
        Returns:
            scores: predicted scores [batch_size * n_items]
        """
        user = interaction[self.USER_ID]
        
        # Use cache if available
        if self.restore_user_e is None or self.restore_item_e is None:
            rating_matrix = self.get_rating_matrix(user)
            
            # Encode (no noise during inference)
            mu, _ = self.encode(rating_matrix, use_noise=False)
            
            # Decode
            logits = self.decode(mu)
            scores = F.softmax(logits, dim=1)
            
            # Cache results
            self.restore_user_e = mu
            self.restore_item_e = scores
        else:
            scores = self.restore_item_e
        
        return scores.view(-1)
    
    def update_encoder_old_state(self):
        """Store current encoder state as old state for composite prior"""
        self.encoder_old_state = {
            key: value.clone().detach()
            for key, value in self.encoder.state_dict().items()
        }
    
    def set_training_mode(self, mode):
        """Set training mode for encoder or decoder"""
        self.training_encoder = mode