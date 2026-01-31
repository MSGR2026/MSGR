# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Implementation based on FEARec paper
# @Email   : 

r"""
FEARec
################################################

Reference:
    Xinyu Du et al. "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation"
    In SIGIR 2023.

"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class HybridAttention(nn.Module):
    """
    Hybrid Attention Layer combining Time Domain Self-Attention and Frequency Domain Auto-Correlation.
    """
    
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, 
                 layer_norm_eps, max_seq_length, global_ratio, topk_factor, gamma):
        super(HybridAttention, self).__init__()
        
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.max_seq_length = max_seq_length
        self.global_ratio = global_ratio  # alpha in the paper
        self.topk_factor = topk_factor  # m in the paper
        self.gamma = gamma  # weight for combining time and frequency attention
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def get_frequency_indices(self, layer_idx, n_layers, seq_length):
        """
        Get frequency sampling indices based on layer position.
        Implements the frequency ramp structure from Section 3.2.1.
        """
        M = seq_length // 2 + 1  # Half spectrum due to conjugate symmetry
        alpha = self.global_ratio
        
        if alpha <= 1.0 / n_layers:
            # Mode 1: Non-overlapping sampling (Eq. 8, 9)
            p = int(M * (1 - (layer_idx + 1) / n_layers))
            q = int(p + M / n_layers)
        else:
            # Mode 2: Overlapping sampling (Eq. 6, 7)
            if n_layers > 1:
                p = int(M * (1 - alpha) * (1 - layer_idx / (n_layers - 1)))
            else:
                p = 0
            q = int(p + alpha * M)
        
        p = max(0, min(p, M - 1))
        q = max(p + 1, min(q, M))
        
        return p, q, M
    
    def frequency_sample_and_pad(self, x_freq, p, q, M, original_len):
        """
        Sample frequency components and zero-pad back to original size.
        """
        # Sample frequency range [p:q]
        sampled = x_freq[:, :, p:q, :]
        
        # Zero-pad to original M size
        batch_size, n_heads, _, head_dim = sampled.shape
        padded = torch.zeros(batch_size, n_heads, M, head_dim, 
                            dtype=sampled.dtype, device=sampled.device)
        padded[:, :, p:q, :] = sampled
        
        return padded
    
    def time_domain_attention(self, Q, K, V, attention_mask, layer_idx, n_layers):
        """
        Time Domain Self-Attention with frequency filtering (Section 3.2.2).
        """
        batch_size, seq_length, _ = Q.shape
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Get frequency indices
        p, q, M = self.get_frequency_indices(layer_idx, n_layers, seq_length)
        
        # Transform to frequency domain using FFT
        Q_freq = torch.fft.rfft(Q, dim=2)
        K_freq = torch.fft.rfft(K, dim=2)
        V_freq = torch.fft.rfft(V, dim=2)
        
        # Sample and pad frequency components
        Q_freq_padded = self.frequency_sample_and_pad(Q_freq, p, q, M, seq_length)
        K_freq_padded = self.frequency_sample_and_pad(K_freq, p, q, M, seq_length)
        V_freq_padded = self.frequency_sample_and_pad(V_freq, p, q, M, seq_length)
        
        # Transform back to time domain using inverse FFT
        Q_filtered = torch.fft.irfft(Q_freq_padded, n=seq_length, dim=2)
        K_filtered = torch.fft.irfft(K_freq_padded, n=seq_length, dim=2)
        V_filtered = torch.fft.irfft(V_freq_padded, n=seq_length, dim=2)
        
        # Standard scaled dot-product attention (Eq. 11)
        attention_scores = torch.matmul(Q_filtered, K_filtered.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V_filtered)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return context
    
    def frequency_domain_attention(self, Q, K, V, layer_idx, n_layers):
        """
        Frequency Domain Auto-Correlation Attention (Section 3.2.3).
        Vectorized implementation for efficiency.
        """
        batch_size, seq_length, _ = Q.shape
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Get frequency indices
        p, q, M = self.get_frequency_indices(layer_idx, n_layers, seq_length)
        
        # Transform Q and K to frequency domain
        Q_freq = torch.fft.rfft(Q, dim=2)
        K_freq = torch.fft.rfft(K, dim=2)
        
        # Sample frequency components (Eq. 14)
        Q_freq_sampled = Q_freq[:, :, p:q, :]
        K_freq_sampled = K_freq[:, :, p:q, :]
        
        # Compute auto-correlation in frequency domain: Q * conj(K) (Eq. 14)
        # Using Wiener-Khinchin theorem
        auto_corr_freq = Q_freq_sampled * torch.conj(K_freq_sampled)
        
        # Pad back to full spectrum
        auto_corr_padded = torch.zeros(batch_size, self.n_heads, M, self.head_dim,
                                       dtype=auto_corr_freq.dtype, device=auto_corr_freq.device)
        auto_corr_padded[:, :, p:q, :] = auto_corr_freq
        
        # Transform back to time domain to get auto-correlation
        auto_corr = torch.fft.irfft(auto_corr_padded, n=seq_length, dim=2)
        
        # Take mean over head dimension to get correlation scores
        auto_corr_scores = auto_corr.mean(dim=-1)  # [B, H, N]
        
        # Select top-k time delays (Eq. 15)
        k = max(1, int(self.topk_factor * math.log(seq_length + 1)))
        k = min(k, seq_length)
        
        # Get top-k correlation values and indices
        topk_values, topk_indices = torch.topk(auto_corr_scores, k, dim=-1)  # [B, H, k]
        
        # Apply softmax to get attention weights (Eq. 16)
        topk_weights = F.softmax(topk_values, dim=-1)  # [B, H, k]
        
        # Time delay aggregation (Eq. 17) - Vectorized implementation
        # Create all rolled versions of V at once using index gathering
        # V: [B, H, N, D]
        
        # Create index tensor for gathering rolled values
        # For each delay tau, we need indices: (tau, tau+1, ..., N-1, 0, 1, ..., tau-1)
        init_index = torch.arange(seq_length, device=V.device).unsqueeze(0)  # [1, N]
        
        # Expand topk_indices for gathering: [B, H, k] -> [B, H, k, N]
        # For each delay in topk_indices, create the rolled index pattern
        delays = topk_indices.unsqueeze(-1)  # [B, H, k, 1]
        
        # Create rolled indices: (init_index + delay) % seq_length
        rolled_indices = (init_index.unsqueeze(0).unsqueeze(0) + delays) % seq_length  # [B, H, k, N]
        
        # Expand rolled_indices for gathering from V: [B, H, k, N, D]
        rolled_indices = rolled_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        
        # Expand V for gathering: [B, H, 1, N, D]
        V_expanded = V.unsqueeze(2).expand(-1, -1, k, -1, -1)
        
        # Gather rolled values: [B, H, k, N, D]
        rolled_V = torch.gather(V_expanded, 3, rolled_indices)
        
        # Weight and sum: topk_weights [B, H, k] -> [B, H, k, 1, 1]
        topk_weights = topk_weights.unsqueeze(-1).unsqueeze(-1)  # [B, H, k, 1, 1]
        
        # Weighted sum over k dimension: [B, H, N, D]
        output = (rolled_V * topk_weights).sum(dim=2)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return output
    
    def forward(self, hidden_states, attention_mask, layer_idx, n_layers):
        """
        Forward pass combining time and frequency domain attention (Eq. 18).
        """
        # Linear projections
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Time domain attention
        time_output = self.time_domain_attention(Q, K, V, attention_mask, layer_idx, n_layers)
        
        # Frequency domain attention
        freq_output = self.frequency_domain_attention(Q, K, V, layer_idx, n_layers)
        
        # Combine outputs (Eq. 18)
        combined_output = self.gamma * time_output + (1 - self.gamma) * freq_output
        
        # Output projection
        output = self.out_proj(combined_output)
        
        return output


class FEABlock(nn.Module):
    """
    Frequency Enhanced Attention Block.
    Consists of HybridAttention + FFN with residual connections.
    """
    
    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob, 
                 attn_dropout_prob, hidden_act, layer_norm_eps, max_seq_length,
                 global_ratio, topk_factor, gamma):
        super(FEABlock, self).__init__()
        
        self.hybrid_attention = HybridAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_seq_length=max_seq_length,
            global_ratio=global_ratio,
            topk_factor=topk_factor,
            gamma=gamma
        )
        
        # Point-wise Feed Forward Network (Eq. 19)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask, layer_idx, n_layers):
        # Hybrid attention with residual (Eq. 20)
        attn_output = self.hybrid_attention(hidden_states, attention_mask, layer_idx, n_layers)
        
        # FFN with residual
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout(ffn_output)
        
        # Residual connection and layer norm (Eq. 20)
        output = self.layer_norm2(hidden_states + attn_output + ffn_output)
        
        return output


class FEAEncoder(nn.Module):
    """
    Frequency Enhanced Attention Encoder.
    Stack of FEA blocks.
    """
    
    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob,
                 attn_dropout_prob, hidden_act, layer_norm_eps, max_seq_length,
                 global_ratio, topk_factor, gamma):
        super(FEAEncoder, self).__init__()
        
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            FEABlock(
                n_heads=n_heads,
                hidden_size=hidden_size,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                max_seq_length=max_seq_length,
                global_ratio=global_ratio,
                topk_factor=topk_factor,
                gamma=gamma
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, layer_idx, self.n_layers)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers


class FEARec(SequentialRecommender):
    r"""
    FEARec: Frequency Enhanced Hybrid Attention Network for Sequential Recommendation.
    
    This model combines time domain self-attention with frequency domain auto-correlation
    to capture both item-level correlations and sequence-level periodic patterns.
    """
    
    def __init__(self, config, dataset):
        super(FEARec, self).__init__(config, dataset)
        
        # Load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # FEARec specific parameters
        self.global_ratio = config["global_ratio"]  # alpha: frequency sampling ratio
        self.topk_factor = config["topk_factor"]  # m: factor for top-k selection
        self.gamma = config["gamma"]  # weight for combining time and frequency attention
        
        # Contrastive learning parameters
        self.lmd = config["lmd"]  # lambda_1: weight for contrastive loss
        self.lmd_sem = config["lmd_sem"]  # weight for supervised contrastive
        self.tau = config["tau"]  # temperature for contrastive learning
        
        # Frequency domain regularization
        self.fredom = config["fredom"]  # whether to use frequency domain regularization
        self.fredom_weight = config["fredom_weight"]  # lambda_2: weight for frequency regularization
        
        # Define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        
        # FEA Encoder
        self.fea_encoder = FEAEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_seq_length=self.max_seq_length,
            global_ratio=self.global_ratio,
            topk_factor=self.topk_factor,
            gamma=self.gamma
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Loss function
        if self.loss_type == "BPR":
            # BPR loss is computed manually in calculate_loss
            self.loss_fct = None
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Parameters initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate attention mask for sequential recommendation."""
        attention_mask = (item_seq != 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        if not bidirectional:
            # Causal mask
            max_len = item_seq.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(
                torch.ones(attn_shape, device=item_seq.device), diagonal=1
            )
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
            extended_attention_mask = extended_attention_mask * subsequent_mask
        
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, item_seq, item_seq_len):
        """
        Forward pass through the FEARec model.
        
        Args:
            item_seq: [B, L] tensor of item IDs
            item_seq_len: [B] tensor of sequence lengths
            
        Returns:
            seq_output: [B, H] tensor of sequence representations
        """
        # Embedding layer (Eq. 1, 2)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Get attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        # FEA Encoder
        encoder_output = self.fea_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = encoder_output[-1]
        
        # Gather output at last position
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        
        return seq_output  # [B, H]
    
    def forward_with_all_layers(self, item_seq, item_seq_len):
        """
        Forward pass returning all layer outputs (for contrastive learning).
        """
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        encoder_output = self.fea_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = encoder_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        
        return seq_output, encoder_output
    
    def contrastive_loss(self, seq_output1, seq_output2, batch_size):
        """
        Compute contrastive loss between two augmented views (Eq. 23, 24).
        """
        # Normalize
        seq_output1 = F.normalize(seq_output1, dim=-1)
        seq_output2 = F.normalize(seq_output2, dim=-1)
        
        # Compute similarity scores
        sim_matrix = torch.matmul(seq_output1, seq_output2.T) / self.tau  # [B, B]
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=seq_output1.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def frequency_regularization(self, h1, h2):
        """
        Frequency domain regularization (Eq. 26).
        """
        # Transform to frequency domain
        h1_freq = torch.fft.fft(h1, dim=-1)
        h2_freq = torch.fft.fft(h2, dim=-1)
        
        # L1 loss in frequency domain
        loss = torch.abs(h1_freq - h2_freq).mean()
        
        return loss
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Forward pass (first view)
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Main recommendation loss (Eq. 22)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        # Contrastive learning loss (Eq. 23, 24)
        if self.training and self.lmd > 0:
            # Second forward pass with dropout for augmentation
            seq_output2 = self.forward(item_seq, item_seq_len)
            cl_loss = self.contrastive_loss(seq_output, seq_output2, seq_output.size(0))
            loss = loss + self.lmd * cl_loss
        
        # Frequency domain regularization (Eq. 26)
        if self.training and self.fredom and self.fredom_weight > 0:
            if self.lmd <= 0:  # Only compute if not already computed above
                seq_output2 = self.forward(item_seq, item_seq_len)
            freq_loss = self.frequency_regularization(seq_output, seq_output2)
            loss = loss + self.fredom_weight * freq_loss
        
        return loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores