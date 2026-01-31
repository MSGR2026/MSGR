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

This implementation follows the paper's methodology:
- Frequency Enhanced Hybrid Attention (FEA) blocks with frequency ramp structure
- Time domain self-attention with FFT-based frequency sampling
- Frequency domain attention using auto-correlation
- Contrastive learning and frequency domain regularization
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class FrequencyRampSampler:
    """
    Frequency Ramp Structure for sampling specific frequency components at each layer.
    
    According to the paper:
    - When alpha > 1/L: overlapping frequencies (mode 2)
    - When alpha <= 1/L: average sampling (mode 1)
    """
    
    def __init__(self, max_seq_length, n_layers, alpha):
        self.M = math.ceil(max_seq_length / 2) + 1  # Half spectrum due to conjugate symmetry
        self.L = n_layers
        self.alpha = alpha
        
    def get_sample_indices(self, layer_idx):
        """
        Get the start and end indices for frequency sampling at a given layer.
        
        Args:
            layer_idx: 0-indexed layer number (l-1 in paper notation)
            
        Returns:
            (p, q): start and end indices for sampling
        """
        l = layer_idx + 1  # Convert to 1-indexed
        
        if self.alpha > 1.0 / self.L:
            # Mode 2: Overlapping frequencies (Eq. 6, 7)
            p = int(self.M * (1 - self.alpha) * (1 - (l - 1) / (self.L - 1))) if self.L > 1 else 0
            q = int(p + self.alpha * self.M)
        else:
            # Mode 1: Average sampling (Eq. 8, 9)
            p = int(self.M * (1 - l / self.L))
            q = int(p + self.M / self.L)
        
        # Ensure valid indices
        p = max(0, min(p, self.M - 1))
        q = max(p + 1, min(q, self.M))
        
        return p, q


class HybridAttention(nn.Module):
    """
    Hybrid Attention Layer combining Time Domain Self-Attention and Frequency Domain Attention.
    
    According to the paper Section 3.2.2 and 3.2.3:
    - Time domain: FFT -> Sample -> Pad -> IFFT -> Standard attention
    - Frequency domain: FFT -> Sample -> Auto-correlation -> Time delay aggregation
    """
    
    def __init__(self, hidden_size, n_heads, attn_dropout_prob, hidden_dropout_prob, 
                 max_seq_length, n_layers, alpha, topk_factor, gamma):
        super(HybridAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.gamma = gamma
        self.topk_factor = topk_factor
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Frequency ramp sampler
        self.sampler = FrequencyRampSampler(max_seq_length, n_layers, alpha)
        self.max_seq_length = max_seq_length
        self.M = self.sampler.M
        
    def forward(self, hidden_states, attention_mask, layer_idx):
        """
        Args:
            hidden_states: [B, N, D]
            attention_mask: [B, 1, N, N] causal mask
            layer_idx: current layer index (0-indexed)
            
        Returns:
            output: [B, N, D]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Linear projections
        Q = self.query(hidden_states)  # [B, N, D]
        K = self.key(hidden_states)    # [B, N, D]
        V = self.value(hidden_states)  # [B, N, D]
        
        # Get sampling indices for current layer
        p, q = self.sampler.get_sample_indices(layer_idx)
        
        # Time domain attention
        time_output = self.time_domain_attention(Q, K, V, attention_mask, p, q, seq_len)
        
        # Frequency domain attention (auto-correlation)
        freq_output = self.frequency_domain_attention(Q, K, V, p, q, seq_len)
        
        # Combine outputs (Eq. 18)
        output = self.gamma * time_output + (1 - self.gamma) * freq_output
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
    
    def time_domain_attention(self, Q, K, V, attention_mask, p, q, seq_len):
        """
        Time Domain Self-Attention with frequency sampling (Section 3.2.2, Eq. 10-11)
        """
        batch_size = Q.size(0)
        
        # FFT along sequence dimension
        Q_fft = torch.fft.rfft(Q, dim=1)  # [B, M, D]
        K_fft = torch.fft.rfft(K, dim=1)
        V_fft = torch.fft.rfft(V, dim=1)
        
        # Sample frequency components
        Q_sampled = Q_fft[:, p:q, :]
        K_sampled = K_fft[:, p:q, :]
        V_sampled = V_fft[:, p:q, :]
        
        # Zero-padding back to original size
        Q_padded = torch.zeros_like(Q_fft)
        K_padded = torch.zeros_like(K_fft)
        V_padded = torch.zeros_like(V_fft)
        
        Q_padded[:, p:q, :] = Q_sampled
        K_padded[:, p:q, :] = K_sampled
        V_padded[:, p:q, :] = V_sampled
        
        # IFFT back to time domain
        Q_t = torch.fft.irfft(Q_padded, n=seq_len, dim=1)  # [B, N, D]
        K_t = torch.fft.irfft(K_padded, n=seq_len, dim=1)
        V_t = torch.fft.irfft(V_padded, n=seq_len, dim=1)
        
        # Reshape for multi-head attention
        Q_t = Q_t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_t = K_t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_t = V_t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Standard scaled dot-product attention (Eq. 11)
        attn_scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Weighted sum
        context = torch.matmul(attn_probs, V_t)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return context
    
    def frequency_domain_attention(self, Q, K, V, p, q, seq_len):
        """
        Frequency Domain Attention using Auto-correlation (Section 3.2.3, Eq. 14-17)
        """
        batch_size = Q.size(0)
        
        # FFT along sequence dimension
        Q_fft = torch.fft.rfft(Q, dim=1)  # [B, M, D]
        K_fft = torch.fft.rfft(K, dim=1)
        
        # Sample frequency components
        Q_sampled = Q_fft[:, p:q, :]  # [B, F, D]
        K_sampled = K_fft[:, p:q, :]
        
        # Auto-correlation via Wiener-Khinchin theorem (Eq. 14)
        # R = IFFT(Q * conj(K))
        correlation_fft = Q_sampled * torch.conj(K_sampled)
        
        # Zero-padding before IFFT
        correlation_padded = torch.zeros(batch_size, Q_fft.size(1), self.hidden_size, 
                                         dtype=torch.complex64, device=Q.device)
        correlation_padded[:, p:q, :] = correlation_fft
        
        # IFFT to get auto-correlation in time domain
        R = torch.fft.irfft(correlation_padded, n=seq_len, dim=1)  # [B, N, D]
        
        # Mean across feature dimension to get correlation scores (Eq. 15)
        R_mean = R.mean(dim=-1)  # [B, N]
        
        # Select top-k time delays (Eq. 15)
        k = max(1, int(self.topk_factor * math.log(seq_len + 1)))
        k = min(k, seq_len)
        
        # Get top-k indices
        _, top_indices = torch.topk(R_mean, k, dim=-1)  # [B, k]
        
        # Get attention weights via softmax (Eq. 16)
        top_R = torch.gather(R_mean, 1, top_indices)  # [B, k]
        attn_weights = F.softmax(top_R, dim=-1)  # [B, k]
        
        # Time delay aggregation (Eq. 17)
        # Roll V by each selected time delay and aggregate
        output = torch.zeros_like(V)  # [B, N, D]
        
        for i in range(k):
            tau = top_indices[:, i]  # [B]
            weight = attn_weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            
            # Roll V by tau for each sample in batch
            V_rolled = self._batch_roll(V, tau)  # [B, N, D]
            output = output + weight * V_rolled
        
        return output
    
    def _batch_roll(self, x, shifts):
        """
        Roll tensor x along dim=1 by different shifts for each batch element.
        
        Args:
            x: [B, N, D]
            shifts: [B] tensor of shift amounts
            
        Returns:
            rolled: [B, N, D]
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Create indices for gathering
        indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        shifts = shifts.unsqueeze(1)
        rolled_indices = (indices - shifts) % seq_len
        
        # Expand for hidden dimension
        rolled_indices = rolled_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        
        return torch.gather(x, 1, rolled_indices)


class FEABlock(nn.Module):
    """
    Frequency Enhanced Attention Block.
    
    Consists of:
    1. Hybrid Attention Layer
    2. Point-wise Feed Forward Network
    3. Residual connections and Layer Normalization
    """
    
    def __init__(self, hidden_size, n_heads, inner_size, hidden_dropout_prob, 
                 attn_dropout_prob, hidden_act, layer_norm_eps, max_seq_length, 
                 n_layers, alpha, topk_factor, gamma):
        super(FEABlock, self).__init__()
        
        self.hybrid_attention = HybridAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout_prob=attn_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            max_seq_length=max_seq_length,
            n_layers=n_layers,
            alpha=alpha,
            topk_factor=topk_factor,
            gamma=gamma
        )
        
        self.attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Point-wise FFN (Eq. 19)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU() if hidden_act == 'gelu' else nn.ReLU(),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )
        
        self.ffn_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask, layer_idx):
        """
        Args:
            hidden_states: [B, N, D]
            attention_mask: [B, 1, N, N]
            layer_idx: current layer index
            
        Returns:
            output: [B, N, D]
        """
        # Hybrid attention with residual (Eq. 20)
        attn_output = self.hybrid_attention(hidden_states, attention_mask, layer_idx)
        hidden_states = self.attention_layernorm(hidden_states + attn_output)
        
        # FFN with residual (Eq. 20)
        ffn_output = self.ffn(hidden_states)
        output = self.ffn_layernorm(hidden_states + ffn_output)
        
        return output


class FEAEncoder(nn.Module):
    """
    Frequency Enhanced Attention Encoder.
    
    Stacks L FEA blocks with frequency ramp structure.
    """
    
    def __init__(self, n_layers, hidden_size, n_heads, inner_size, hidden_dropout_prob,
                 attn_dropout_prob, hidden_act, layer_norm_eps, max_seq_length, 
                 alpha, topk_factor, gamma):
        super(FEAEncoder, self).__init__()
        
        self.n_layers = n_layers
        
        self.blocks = nn.ModuleList([
            FEABlock(
                hidden_size=hidden_size,
                n_heads=n_heads,
                inner_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                max_seq_length=max_seq_length,
                n_layers=n_layers,
                alpha=alpha,
                topk_factor=topk_factor,
                gamma=gamma
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        """
        Args:
            hidden_states: [B, N, D]
            attention_mask: [B, 1, N, N]
            output_all_encoded_layers: whether to return all layer outputs
            
        Returns:
            all_encoder_layers: list of [B, N, D] tensors
        """
        all_encoder_layers = []
        
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, attention_mask, i)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers


class FEARec(SequentialRecommender):
    """
    FEARec: Frequency Enhanced Hybrid Attention Network for Sequential Recommendation.
    
    This model combines:
    1. Frequency Enhanced Hybrid Attention (FEA) blocks
    2. Frequency ramp structure for layer-wise frequency sampling
    3. Contrastive learning with dropout-based augmentation
    4. Frequency domain regularization
    """
    
    def __init__(self, config, dataset):
        super(FEARec, self).__init__(config, dataset)
        
        # Load model parameters
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        
        # FEARec specific parameters
        self.alpha = config["alpha"]  # Frequency sampling ratio
        self.gamma = config["gamma"]  # Weight for time vs frequency attention
        self.topk_factor = config["topk_factor"]  # Factor for top-k selection in auto-correlation
        
        # Contrastive learning parameters
        self.lmd = config["lmd"]  # Lambda for contrastive loss
        self.lmd_sem = config["lmd_sem"]  # Lambda for supervised contrastive loss
        self.tau = config["tau"]  # Temperature for contrastive learning
        
        # Frequency domain regularization
        self.fredom = config["fredom"]  # Enable frequency domain regularization
        self.fredom_type = config["fredom_type"]  # Type of frequency regularization
        
        self.loss_type = config["loss_type"]
        
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
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_seq_length=self.max_seq_length,
            alpha=self.alpha,
            topk_factor=self.topk_factor,
            gamma=self.gamma
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Loss functions
        if self.loss_type == "BPR":
            self.loss_fct = self.bpr_loss
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Initialize weights
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
    
    def bpr_loss(self, pos_score, neg_score):
        """BPR loss function"""
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
    
    def forward(self, item_seq, item_seq_len):
        """
        Forward pass through the FEARec model.
        
        Args:
            item_seq: [B, N] item sequence
            item_seq_len: [B] sequence lengths
            
        Returns:
            seq_output: [B, D] sequence representation
        """
        # Position embedding (Eq. 1-2)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        # Item embedding
        item_emb = self.item_embedding(item_seq)
        
        # Combine embeddings (Eq. 2)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Create causal attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        # FEA Encoder
        encoder_output = self.fea_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = encoder_output[-1]
        
        # Gather last position output
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        
        return seq_output
    
    def forward_with_aug(self, item_seq, item_seq_len):
        """
        Forward pass with augmentation for contrastive learning.
        Returns both views from different dropout masks.
        """
        # First view
        seq_output1 = self.forward(item_seq, item_seq_len)
        
        # Second view (different dropout)
        seq_output2 = self.forward(item_seq, item_seq_len)
        
        return seq_output1, seq_output2
    
    def contrastive_loss(self, seq_output1, seq_output2, same_target_index=None):
        """
        Compute contrastive loss (Section 3.4.1, Eq. 23-24).
        
        Args:
            seq_output1: [B, D] first view
            seq_output2: [B, D] second view
            same_target_index: [B] indices of sequences with same target (for supervised CL)
            
        Returns:
            cl_loss: contrastive loss
        """
        batch_size = seq_output1.size(0)
        
        # Normalize
        seq_output1 = F.normalize(seq_output1, dim=-1)
        seq_output2 = F.normalize(seq_output2, dim=-1)
        
        # Unsupervised contrastive loss
        # Positive pairs: (seq_output1[i], seq_output2[i])
        # Negative pairs: all other combinations
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(seq_output1, seq_output2.T) / self.tau  # [B, B]
        
        # Positive pairs are on diagonal
        pos_sim = torch.diag(sim_matrix)  # [B]
        
        # InfoNCE loss
        cl_loss = -torch.log(
            torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=-1)
        ).mean()
        
        # Supervised contrastive loss if same_target_index provided
        if same_target_index is not None and self.lmd_sem > 0:
            # Get supervised positive pairs
            sup_seq_output = seq_output2[same_target_index]  # [B, D]
            
            # Compute similarity
            sup_sim = (seq_output1 * sup_seq_output).sum(dim=-1) / self.tau  # [B]
            
            # All negatives
            all_sim = torch.matmul(seq_output1, seq_output2.T) / self.tau  # [B, B]
            
            sup_cl_loss = -torch.log(
                torch.exp(sup_sim) / torch.exp(all_sim).sum(dim=-1)
            ).mean()
            
            cl_loss = cl_loss + self.lmd_sem * sup_cl_loss
        
        return cl_loss
    
    def frequency_regularization(self, seq_output1, seq_output2):
        """
        Frequency domain regularization (Section 3.4.2, Eq. 26).
        
        Args:
            seq_output1: [B, D] first view
            seq_output2: [B, D] second view
            
        Returns:
            freq_loss: L1 loss in frequency domain
        """
        # FFT of both views
        fft1 = torch.fft.fft(seq_output1, dim=-1)
        fft2 = torch.fft.fft(seq_output2, dim=-1)
        
        # L1 loss between frequency representations
        freq_loss = torch.abs(fft1 - fft2).mean()
        
        return freq_loss
    
    def calculate_loss(self, interaction):
        """
        Calculate the total loss (Eq. 27).
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Forward pass with augmentation
        seq_output1, seq_output2 = self.forward_with_aug(item_seq, item_seq_len)
        
        # Main recommendation loss (Eq. 22)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output1 * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output1 * neg_items_emb, dim=-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output1, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)
        
        # Contrastive loss (Eq. 23-24)
        # For supervised CL, we need sequences with same target
        # In practice, we use the batch itself as approximation
        same_target_index = self._get_same_target_index(pos_items)
        cl_loss = self.contrastive_loss(seq_output1, seq_output2, same_target_index)
        
        # Total loss (Eq. 27)
        total_loss = rec_loss + self.lmd * cl_loss
        
        # Frequency domain regularization (Eq. 26)
        if self.fredom:
            freq_loss = self.frequency_regularization(seq_output1, seq_output2)
            total_loss = total_loss + self.fredom * freq_loss
        
        return total_loss
    
    def _get_same_target_index(self, pos_items):
        """
        Find indices of sequences with the same target item.
        Used for supervised contrastive learning.
        """
        batch_size = pos_items.size(0)
        
        # For each item, find another sequence with same target
        same_target_index = torch.arange(batch_size, device=pos_items.device)
        
        for i in range(batch_size):
            # Find sequences with same target
            same_mask = (pos_items == pos_items[i])
            same_indices = torch.where(same_mask)[0]
            
            if len(same_indices) > 1:
                # Choose a different index with same target
                for idx in same_indices:
                    if idx != i:
                        same_target_index[i] = idx
                        break
        
        return same_target_index
    
    def predict(self, interaction):
        """
        Predict scores for test items.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (Eq. 21).
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores