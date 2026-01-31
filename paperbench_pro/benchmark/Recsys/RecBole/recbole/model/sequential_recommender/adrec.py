# -*- coding: utf-8 -*-
# @Time    : 2026/1/8
# @Author  : AI Assistant
# @Email   : 

"""
ADRec
################################################

Reference:
    Jialei Chen et al. "Unlocking the Power of Diffusion Models in Sequential Recommendation: 
    A Simple and Effective Approach" in KDD 2025.

Reference:
    https://github.com/Nemo-1024/ADRec

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class SiLU(nn.Module):
    """SiLU activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.unsqueeze(-1).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DenoisedModel(nn.Module):
    """Denoising model for diffusion process"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.n_layers = config.get('dif_decoder_layers', 2)
        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        
        # Diffusion decoder (Transformer or MLP)
        dif_decoder = config.get('dif_decoder', 'att')
        if dif_decoder == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                SiLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.decoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
            )
        
        self.time_embed = TimestepEmbedder(self.hidden_size)
        self.lambda_uncertainty = config.get('lambda_uncertainty', 0.001)

    def forward(self, rep_item, x_t, t, attention_mask):
        """
        Args:
            rep_item: [B, L, H] - conditional representation from causal attention
            x_t: [B, L, H] - noised target sequence
            t: [B, L] - diffusion timesteps
            attention_mask: [B, L] - attention mask
        """
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(t)
        
        # Feature aggregation
        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)
        
        # Apply decoder
        if isinstance(self.decoder, nn.Sequential):
            # MLP decoder
            rep_diffu = self.decoder(rep_diffu)
        else:
            # Transformer decoder
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            rep_diffu = self.decoder(rep_diffu, extended_attention_mask, output_all_encoded_layers=False)
            rep_diffu = rep_diffu[-1] if isinstance(rep_diffu, list) else rep_diffu
        
        return rep_diffu


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_a=0.3, beta_b=10):
    """Get beta schedule for diffusion process"""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        betas[betas < beta_a] = beta_a
        betas[betas > beta_b] = beta_b
        return betas
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # Reshape res to match broadcast_shape
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    # Only expand dimensions that are singleton (size 1)
    # This handles both batch-level and token-level diffusion
    return res


class ADRec(SequentialRecommender):
    r"""
    ADRec is an auto-regressive diffusion recommendation model that applies 
    independent noise processes to each token and performs diffusion across 
    the entire target sequence during training.

    """

    def __init__(self, config, dataset):
        super(ADRec, self).__init__(config, dataset)

        # Load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        
        # Diffusion parameters
        self.diffusion_steps = config.get('diffusion_steps', 32)
        self.noise_schedule = config.get('noise_schedule', 'trunc_lin')
        self.beta_a = config.get('beta_a', 0.3)
        self.beta_b = config.get('beta_b', 10)
        self.independent_diffusion = config.get('independent', True)
        self.lambda_uncertainty = config.get('lambda_uncertainty', 0.001)
        self.rescale_timesteps = config.get('rescale_timesteps', True)
        self.loss_type = config.get('loss_type', 'CE')
        self.use_mse_loss = config.get('use_mse_loss', True)  # Whether to use MSE loss
        self.loss_scale = config.get('loss_scale', 1.0)  # Scale for MSE loss
        
        # Embedding dropout (applied before LayerNorm in original code)
        self.emb_dropout = config.get('emb_dropout', 0.3)
        
        # Define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        
        # Embedding dropout (applied BEFORE LayerNorm as in original code)
        self.embed_dropout = nn.Dropout(self.emb_dropout)
        
        # No positional encoding as mentioned in the paper (Appendix D)
        
        # Causal Attention Module (CAM)
        self.causal_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        
        # Auto-regressive Diffusion Module (ADM)
        self.denoised_model = DenoisedModel(config)
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Loss functions
        self.ce_loss_fct = nn.CrossEntropyLoss()
        self.mse_loss_fct = nn.MSELoss(reduction='none')
        
        # Diffusion setup
        self.setup_diffusion()
        
        # Parameters initialization
        self.apply(self._init_weights)

    def setup_diffusion(self):
        """Setup diffusion process parameters"""
        betas = get_named_beta_schedule(
            self.noise_schedule, 
            self.diffusion_steps,
            self.beta_a,
            self.beta_b
        )
        
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.num_timesteps = int(self.betas.shape[0])

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _scale_timesteps(self, t):
        """Scale timesteps for better numerical stability"""
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        Sample from q(x_t | x_0).
        
        Args:
            x_start: [B*L, H] or [B, L, H] - original data
            t: [B*L] or [B, L] - timesteps
            noise: optional noise tensor with same shape as x_start
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Handle both flattened and non-flattened inputs
        original_shape = x_start.shape
        if len(x_start.shape) == 3:  # [B, L, H]
            x_start_flat = x_start.reshape(-1, x_start.shape[-1])
            noise_flat = noise.reshape(-1, noise.shape[-1])
            t_flat = t.reshape(-1)
        else:  # Already flattened [B*L, H]
            x_start_flat = x_start
            noise_flat = noise
            t_flat = t
        
        # Extract coefficients
        sqrt_alpha_cumprod = torch.from_numpy(self.sqrt_alphas_cumprod).to(device=t_flat.device)[t_flat].float()
        sqrt_one_minus_alpha_cumprod = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod).to(device=t_flat.device)[t_flat].float()
        
        # Add dimension for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)  # [B*L, 1]
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)  # [B*L, 1]
        
        # Apply noise
        x_t = sqrt_alpha_cumprod * x_start_flat + sqrt_one_minus_alpha_cumprod * noise_flat
        
        # Reshape back if needed
        if len(original_shape) == 3:
            x_t = x_t.reshape(original_shape)
        
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute the mean of the diffusion posterior q(x_{t-1} | x_t, x_0)"""
        # Handle both flattened and non-flattened inputs
        original_shape = x_t.shape
        if len(x_t.shape) == 3:  # [B, L, H]
            x_start_flat = x_start.reshape(-1, x_start.shape[-1])
            x_t_flat = x_t.reshape(-1, x_t.shape[-1])
            t_flat = t.reshape(-1)
        else:  # Already flattened
            x_start_flat = x_start
            x_t_flat = x_t
            t_flat = t
        
        # Extract coefficients
        coef1 = torch.from_numpy(self.posterior_mean_coef1).to(device=t_flat.device)[t_flat].float()
        coef2 = torch.from_numpy(self.posterior_mean_coef2).to(device=t_flat.device)[t_flat].float()
        
        # Add dimension for broadcasting
        coef1 = coef1.unsqueeze(-1)  # [B*L, 1]
        coef2 = coef2.unsqueeze(-1)  # [B*L, 1]
        
        # Compute posterior mean
        posterior_mean = coef1 * x_start_flat + coef2 * x_t_flat
        
        # Reshape back if needed
        if len(original_shape) == 3:
            posterior_mean = posterior_mean.reshape(original_shape)
        
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, attention_mask):
        """Predict mean and variance for reverse process"""
        x_0 = self.denoised_model(rep_item, x_t, self._scale_timesteps(t), attention_mask)
        
        # Compute log variance
        log_variance_arr = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        
        # Handle both flattened and non-flattened inputs
        if len(x_t.shape) == 3:  # [B, L, H]
            t_flat = t.reshape(-1)
            model_log_variance = torch.from_numpy(log_variance_arr).to(device=t_flat.device)[t_flat].float()
            # Reshape to [B, L, 1] for broadcasting
            model_log_variance = model_log_variance.view(x_t.shape[0], x_t.shape[1], 1)
        else:  # Already flattened
            model_log_variance = torch.from_numpy(log_variance_arr).to(device=t.device)[t].float()
            model_log_variance = model_log_variance.unsqueeze(-1)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance, x_0

    def p_sample(self, rep_item, noise_x_t, t, attention_mask):
        """Sample x_{t-1} from the model"""
        model_mean, model_log_variance, _ = self.p_mean_variance(rep_item, noise_x_t, t, attention_mask)
        noise = torch.randn_like(noise_x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float()
        while len(nonzero_mask.shape) < len(noise_x_t.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample_xt

    def forward(self, item_seq, item_seq_len):
        """Forward pass during training"""
        # Get item embeddings
        item_emb = self.item_embedding(item_seq)  # [B, L, H]
        # Apply embedding dropout BEFORE LayerNorm (as in original code)
        item_emb = self.embed_dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        
        # Create attention mask
        attention_mask = (item_seq != 0).long()
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        # Causal Attention Module (CAM)
        rep_item = self.causal_attention(
            item_emb, extended_attention_mask, output_all_encoded_layers=False
        )
        rep_item = rep_item[-1] if isinstance(rep_item, list) else rep_item
        
        return rep_item, item_emb, attention_mask

    def calculate_loss(self, interaction):
        """Calculate loss for training"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        rep_item, item_emb, attention_mask = self.forward(item_seq, item_seq_len)
        
        # Get target items
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Prepare target sequence embeddings
        # For per-step prediction: shift sequence by one position
        batch_size = item_seq.size(0)
        seq_len = item_seq.size(1)
        
        # Create target sequence: [i^1, i^2, ..., i^L]
        target_seq = torch.cat([item_seq[:, 1:], pos_items.unsqueeze(1)], dim=1)
        target_emb_full = self.item_embedding(target_seq)  # [B, L, H]
        
        # Create mask for valid positions
        target_mask = (target_seq != 0).float()
        
        # Sample timesteps (independent for each token if enabled)
        if self.independent_diffusion:
            # Independent diffusion: each token has independent timestep
            t = torch.randint(
                0, self.num_timesteps, 
                (batch_size, seq_len), 
                device=item_seq.device
            ).long()
            t = t * target_mask.long()  # Zero out padding positions
        else:
            # Sequence-level diffusion: same timestep for all tokens
            t = torch.randint(
                0, self.num_timesteps, 
                (batch_size,), 
                device=item_seq.device
            ).long()
            t = t.unsqueeze(1).expand(-1, seq_len)
        
        # Add noise to target embeddings
        x_t = self.q_sample(target_emb_full, t)
        
        # Denoise
        denoised = self.denoised_model(rep_item, x_t, t, attention_mask)
        
        # Calculate CE loss on VALID positions only (memory efficient)
        # Flatten and filter by mask
        valid_mask = target_mask.bool()
        denoised_valid = denoised[valid_mask]  # [num_valid, H]
        target_valid = target_seq[valid_mask]  # [num_valid]
        
        # Compute scores only for valid positions
        scores = torch.matmul(denoised_valid, self.item_embedding.weight.t())  # [num_valid, num_items]
        ce_loss = self.ce_loss_fct(scores, target_valid)
        
        # Calculate MSE loss if enabled
        if self.use_mse_loss:
            # MSE loss with mask normalization (as in original code)
            mse_loss = self.mse_loss_fct(denoised, target_emb_full)
            # Normalize by mask
            mse_loss = (mse_loss * (target_mask / target_mask.sum(1, keepdim=True)).unsqueeze(-1)).sum(1).mean()
            
            # Combined loss
            total_loss = ce_loss + self.loss_scale * mse_loss
        else:
            total_loss = ce_loss
        
        return total_loss

    def predict(self, interaction):
        """Predict for evaluation"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.inference(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Full sort prediction for evaluation"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.inference(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def inference(self, item_seq, item_seq_len):
        """
        Inference with diffusion denoising.
        Only denoise the LAST position, keeping history clean (as in original code).
        """
        # Get item embeddings
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        
        # Create attention mask
        attention_mask = (item_seq != 0).long()
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        # Causal Attention Module (extract features from history)
        rep_item = self.causal_attention(
            item_emb, extended_attention_mask, output_all_encoded_layers=False
        )
        rep_item = rep_item[-1] if isinstance(rep_item, list) else rep_item
        
        batch_size = item_seq.size(0)
        seq_len = item_seq.size(1)
        
        # Start with pure noise for the LAST position only
        # For other positions, use the clean embeddings from history
        noise_x_t = torch.randn(batch_size, seq_len, self.hidden_size, device=item_seq.device)
        
        # Iterative denoising from T to 0 (only on last position)
        for i in reversed(range(self.num_timesteps)):
            # Create timestep tensor: [0, 0, ..., 0, i] for each sequence
            # This ensures only the last position is denoised
            t = torch.zeros(batch_size, seq_len, device=item_seq.device, dtype=torch.long)
            t[:, -1] = i
            
            # Concatenate clean history with noisy last position
            # This is key: history stays clean, only last position is noised
            x_t_full = torch.cat([rep_item[:, :-1, :], noise_x_t[:, -1:, :]], dim=1)
            
            # Denoise
            noise_x_t = self.p_sample(rep_item, x_t_full, t, attention_mask)
        
        # Extract the final denoised embedding (last position)
        seq_output = noise_x_t[:, -1, :]
        
        return seq_output
