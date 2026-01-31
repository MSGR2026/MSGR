# -*- coding: utf-8 -*-
"""
DIFF
################################################

Reference:
    "Dual Side-Information Filtering and Fusion for Sequential Recommendation"

DIFF effectively removes noisy signals using frequency-based filtering and fully leverages
the correlation across item IDs and attributes through dual fusion (early + intermediate).
"""

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer


class FrequencyFilter(nn.Module):
    """
    Frequency-based noise filtering module using Discrete Fourier Transform.
    Decomposes embeddings into low-frequency and high-frequency components,
    then recombines them with learnable scaling factors.
    """
    
    def __init__(self, cutoff_ratio=0.5):
        super(FrequencyFilter, self).__init__()
        self.cutoff_ratio = cutoff_ratio
    
    def forward(self, x, beta):
        """
        Args:
            x: Input embeddings [B, L, D]
            beta: Learnable scalar for high-frequency component
        Returns:
            Filtered embeddings [B, L, D]
        """
        # Apply FFT along sequence dimension
        # x: [B, L, D]
        x_freq = torch.fft.fft(x, dim=1)  # [B, L, D] complex
        
        seq_len = x.size(1)
        cutoff = max(1, int(seq_len * self.cutoff_ratio))
        
        # Split into low and high frequency components
        x_freq_low = x_freq.clone()
        x_freq_high = x_freq.clone()
        
        # Zero out high frequencies for low-frequency component
        x_freq_low[:, cutoff:, :] = 0
        
        # Zero out low frequencies for high-frequency component
        x_freq_high[:, :cutoff, :] = 0
        
        # Apply inverse FFT to get time-domain signals
        x_lfc = torch.fft.ifft(x_freq_low, dim=1).real  # [B, L, D]
        x_hfc = torch.fft.ifft(x_freq_high, dim=1).real  # [B, L, D]
        
        # Combine with learnable beta
        x_filtered = x_lfc + beta * x_hfc
        
        return x_filtered


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with separate Q, K, V projections.
    """
    
    def __init__(self, hidden_size, n_heads, attn_dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query, key, value: [B, L, D]
            attention_mask: [B, 1, L, L] or [B, 1, 1, L]
        Returns:
            output: [B, L, D]
            attention_scores: [B, H, L, L]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project and reshape for multi-head attention
        Q = self.W_Q(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, H, L, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attention_scores: [B, H, L, L]
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)  # [B, H, L, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.W_O(context)
        
        return output, attention_scores


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act='gelu'):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        if hidden_act == 'gelu':
            self.activation = nn.GELU()
        elif hidden_act == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DualFusionLayer(nn.Module):
    """
    Dual fusion layer combining ID-centric and attribute-enriched fusion.
    """
    
    def __init__(self, hidden_size, n_heads, inner_size, hidden_dropout_prob, 
                 attn_dropout_prob, layer_norm_eps, hidden_act, num_attributes, fusion_type='sum'):
        super(DualFusionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.num_attributes = num_attributes
        self.fusion_type = fusion_type
        
        # ID-centric attention components
        self.id_attention = MultiHeadAttention(hidden_size, n_heads, attn_dropout_prob)
        
        # Attribute attention components (one for each attribute)
        self.attr_attentions = nn.ModuleList([
            MultiHeadAttention(hidden_size, n_heads, attn_dropout_prob)
            for _ in range(num_attributes)
        ])
        
        # Attribute-enriched attention (for fused embeddings)
        self.fused_attention = MultiHeadAttention(hidden_size, n_heads, attn_dropout_prob)
        
        # Layer norms
        self.id_layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.id_layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fused_layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fused_layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # FFN
        self.id_ffn = FeedForwardNetwork(hidden_size, inner_size, hidden_dropout_prob, hidden_act)
        self.fused_ffn = FeedForwardNetwork(hidden_size, inner_size, hidden_dropout_prob, hidden_act)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Fusion weight for attention scores (if using gating)
        if fusion_type == 'gate':
            self.fusion_gate = nn.Linear(hidden_size * (1 + num_attributes), hidden_size)
    
    def forward(self, E_v, E_attrs, E_va, attention_mask):
        """
        Args:
            E_v: Item ID embeddings [B, L, D]
            E_attrs: List of attribute embeddings, each [B, L, D]
            E_va: Fused embeddings [B, L, D]
            attention_mask: [B, 1, L, L]
        Returns:
            R_v: ID-centric representation [B, L, D]
            R_va: Attribute-enriched representation [B, L, D]
        """
        batch_size, seq_len, hidden_size = E_v.size()
        
        # === ID-centric Fusion (Intermediate Fusion) ===
        # Get attention scores from ID and attributes
        _, A_v = self.id_attention(E_v, E_v, E_v, attention_mask)  # [B, H, L, L]
        
        A_attrs = []
        for i, attr_attn in enumerate(self.attr_attentions):
            if i < len(E_attrs):
                _, A_attr = attr_attn(E_attrs[i], E_attrs[i], E_attrs[i], attention_mask)
                A_attrs.append(A_attr)
        
        # Fuse attention scores
        if len(A_attrs) > 0:
            if self.fusion_type == 'sum':
                A_fused = A_v + sum(A_attrs)
            elif self.fusion_type == 'mean':
                A_fused = (A_v + sum(A_attrs)) / (1 + len(A_attrs))
            else:  # default to sum
                A_fused = A_v + sum(A_attrs)
        else:
            A_fused = A_v
        
        # Apply attention mask and softmax
        if attention_mask is not None:
            A_fused = A_fused + attention_mask
        
        attention_probs = F.softmax(A_fused / (self.hidden_size // self.n_heads) ** 0.5, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute value from ID embeddings
        head_dim = hidden_size // self.n_heads
        V_v = E_v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        
        # Apply attention to values
        context_v = torch.matmul(attention_probs, V_v)  # [B, H, L, head_dim]
        context_v = context_v.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Residual connection and layer norm
        R_v = self.id_layer_norm1(E_v + self.dropout(context_v))
        R_v = self.id_layer_norm2(R_v + self.id_ffn(R_v))
        
        # === Attribute-enriched Fusion (Early Fusion) ===
        fused_output, _ = self.fused_attention(E_va, E_va, E_va, attention_mask)
        R_va = self.fused_layer_norm1(E_va + self.dropout(fused_output))
        R_va = self.fused_layer_norm2(R_va + self.fused_ffn(R_va))
        
        return R_v, R_va


class DIFF(SequentialRecommender):
    """
    DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation.
    
    Key components:
    1. Frequency-based Noise Filtering: Uses DFT to decompose embeddings into low/high frequency
       components and recombines with learnable scaling factors.
    2. Dual Multi-sequence Fusion: Combines ID-centric (intermediate) and attribute-enriched (early)
       fusion strategies.
    3. Representation Alignment: Contrastive loss to align ID and attribute embedding spaces.
    """
    
    def __init__(self, config, dataset):
        super(DIFF, self).__init__(config, dataset)
        
        # Load parameters
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        
        # DIFF-specific parameters
        self.alpha = config.get("alpha", 0.5)  # Balance between ID-centric and attribute-enriched
        self.lambda_align = config.get("lambda_align", 0.1)  # Alignment loss weight
        self.fusion_type = config.get("fusion_type", "sum")  # sum, mean, or gate
        self.cutoff_ratio = config.get("cutoff_ratio", 0.5)  # Frequency cutoff ratio
        
        # Feature configuration
        self.selected_features = config.get("selected_features", [])
        self.pooling_mode = config.get("pooling_mode", "mean")
        
        # Validate selected features exist in dataset
        if self.selected_features:
            valid_features = []
            for feat in self.selected_features:
                if hasattr(dataset, 'field2type') and feat in dataset.field2type:
                    valid_features.append(feat)
            self.selected_features = valid_features
        
        self.num_attributes = len(self.selected_features) if self.selected_features else 0
        
        # Define embedding layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # Feature embedding layer (if attributes are available)
        if self.num_attributes > 0:
            self.feature_embed_layer = FeatureSeqEmbLayer(
                dataset,
                self.hidden_size,
                self.selected_features,
                self.pooling_mode,
                self.device,
            )
        else:
            self.feature_embed_layer = None
        
        # Frequency-based filtering
        self.freq_filter = FrequencyFilter(cutoff_ratio=self.cutoff_ratio)
        
        # Learnable beta parameters for frequency filtering
        # beta_0 for item ID, beta_1 to beta_m for attributes, beta_{m+1} for fused
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_attributes + 2)
        ])
        
        # Dual fusion layers
        self.dual_fusion_layers = nn.ModuleList([
            DualFusionLayer(
                self.hidden_size, self.n_heads, self.inner_size,
                self.hidden_dropout_prob, self.attn_dropout_prob,
                self.layer_norm_eps, self.hidden_act,
                self.num_attributes, self.fusion_type
            )
            for _ in range(self.n_layers)
        ])
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Learnable temperature for alignment
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        if self.num_attributes > 0 and self.feature_embed_layer is not None:
            self.other_parameter_name = ["feature_embed_layer"]
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_attribute_embeddings(self, item_seq):
        """
        Get attribute embeddings for the item sequence.
        
        Args:
            item_seq: [B, L]
        Returns:
            List of attribute embeddings, each [B, L, D]
        """
        if self.num_attributes == 0 or self.feature_embed_layer is None:
            return []
        
        try:
            sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding.get("item", None) if sparse_embedding else None
            dense_embedding = dense_embedding.get("item", None) if dense_embedding else None
            
            attr_embeddings = []
            
            if sparse_embedding is not None and sparse_embedding.dim() >= 3:
                # sparse_embedding: [B, L, num_sparse_features, D]
                if sparse_embedding.dim() == 4:
                    for i in range(sparse_embedding.size(2)):
                        attr_embeddings.append(sparse_embedding[:, :, i, :])
                else:
                    # [B, L, D]
                    attr_embeddings.append(sparse_embedding)
            
            if dense_embedding is not None and dense_embedding.dim() >= 3:
                # dense_embedding: [B, L, num_dense_features, D]
                if dense_embedding.dim() == 4:
                    for i in range(dense_embedding.size(2)):
                        attr_embeddings.append(dense_embedding[:, :, i, :])
                else:
                    # [B, L, D]
                    attr_embeddings.append(dense_embedding)
            
            return attr_embeddings
        except Exception:
            return []
    
    def fuse_embeddings(self, E_v, E_attrs):
        """
        Early fusion of item ID and attribute embeddings.
        
        Args:
            E_v: Item ID embeddings [B, L, D]
            E_attrs: List of attribute embeddings, each [B, L, D]
        Returns:
            E_va: Fused embeddings [B, L, D]
        """
        if len(E_attrs) == 0:
            return E_v
        
        if self.fusion_type == 'sum':
            E_va = E_v + sum(E_attrs)
        elif self.fusion_type == 'mean':
            E_va = (E_v + sum(E_attrs)) / (1 + len(E_attrs))
        else:  # default to sum
            E_va = E_v + sum(E_attrs)
        
        return E_va
    
    def apply_frequency_filtering(self, E_v, E_attrs, E_va):
        """
        Apply frequency-based noise filtering to all embeddings.
        
        Args:
            E_v: Item ID embeddings [B, L, D]
            E_attrs: List of attribute embeddings, each [B, L, D]
            E_va: Fused embeddings [B, L, D]
        Returns:
            Filtered embeddings
        """
        # Filter item ID embeddings
        E_v_filtered = self.freq_filter(E_v, self.beta[0])
        
        # Filter attribute embeddings
        E_attrs_filtered = []
        for i, E_attr in enumerate(E_attrs):
            E_attr_filtered = self.freq_filter(E_attr, self.beta[i + 1])
            E_attrs_filtered.append(E_attr_filtered)
        
        # Filter fused embeddings
        E_va_filtered = self.freq_filter(E_va, self.beta[-1])
        
        return E_v_filtered, E_attrs_filtered, E_va_filtered
    
    def compute_alignment_loss(self, E_v, E_attrs, item_seq):
        """
        Compute representation alignment loss between item IDs and fused attributes.
        
        Args:
            E_v: Item ID embeddings [B, L, D]
            E_attrs: List of attribute embeddings, each [B, L, D]
            item_seq: [B, L]
        Returns:
            Alignment loss
        """
        if len(E_attrs) == 0:
            return torch.tensor(0.0, device=E_v.device)
        
        # Fuse attributes using summation
        E_a = sum(E_attrs)  # [B, L, D]
        
        # Normalize embeddings
        E_v_norm = F.normalize(E_v, dim=-1)
        E_a_norm = F.normalize(E_a, dim=-1)
        
        # Compute similarity matrices
        # [B, L, L]
        sim_v_a = torch.bmm(E_v_norm, E_a_norm.transpose(1, 2)) / self.temperature.abs().clamp(min=0.01)
        sim_a_v = torch.bmm(E_a_norm, E_v_norm.transpose(1, 2)) / self.temperature.abs().clamp(min=0.01)
        
        # Create ground truth: items with same attributes should be aligned
        # For simplicity, use diagonal as positive (same position = same item = same attributes)
        batch_size, seq_len = item_seq.size()
        
        # Create mask for valid positions (non-padding)
        valid_mask = (item_seq != 0).float()  # [B, L]
        
        # Ground truth: diagonal is positive
        labels = torch.arange(seq_len, device=E_v.device).unsqueeze(0).expand(batch_size, -1)
        
        # Apply valid mask to loss computation
        loss_v_a = F.cross_entropy(
            sim_v_a.reshape(-1, seq_len), 
            labels.reshape(-1), 
            reduction='none'
        )
        loss_a_v = F.cross_entropy(
            sim_a_v.reshape(-1, seq_len), 
            labels.reshape(-1), 
            reduction='none'
        )
        
        # Mask out padding positions
        valid_mask_flat = valid_mask.reshape(-1)
        loss_v_a = (loss_v_a * valid_mask_flat).sum() / valid_mask_flat.sum().clamp(min=1)
        loss_a_v = (loss_a_v * valid_mask_flat).sum() / valid_mask_flat.sum().clamp(min=1)
        
        return (loss_v_a + loss_a_v) / 2
    
    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of DIFF.
        
        Args:
            item_seq: [B, L]
            item_seq_len: [B]
        Returns:
            seq_output: [B, D]
        """
        # Get position embeddings
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        # Get item ID embeddings
        E_v = self.item_embedding(item_seq)  # [B, L, D]
        E_v = E_v + position_embedding
        E_v = self.LayerNorm(E_v)
        E_v = self.dropout(E_v)
        
        # Get attribute embeddings
        E_attrs = self.get_attribute_embeddings(item_seq)
        
        # Add position embeddings to attributes
        E_attrs = [self.dropout(self.LayerNorm(e + position_embedding)) for e in E_attrs]
        
        # Early fusion
        E_va = self.fuse_embeddings(E_v, E_attrs)
        
        # Apply frequency-based filtering
        E_v, E_attrs, E_va = self.apply_frequency_filtering(E_v, E_attrs, E_va)
        
        # Get attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)
        
        # Apply dual fusion layers
        R_v, R_va = E_v, E_va
        for layer in self.dual_fusion_layers:
            R_v, R_va = layer(R_v, E_attrs, R_va, extended_attention_mask)
        
        # Combine ID-centric and attribute-enriched representations
        R_u = self.alpha * R_v + (1 - self.alpha) * R_va
        
        # Get final representation (last valid position)
        seq_output = self.gather_indexes(R_u, item_seq_len - 1)
        
        return seq_output
    
    def calculate_loss(self, interaction):
        """
        Calculate the training loss.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Forward pass
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Recommendation loss
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss_rec = self.loss_fct(logits, pos_items)
        
        # Alignment loss
        if self.lambda_align > 0 and self.num_attributes > 0:
            # Get embeddings for alignment loss
            E_v = self.item_embedding(item_seq)
            E_attrs = self.get_attribute_embeddings(item_seq)
            loss_align = self.compute_alignment_loss(E_v, E_attrs, item_seq)
            loss = loss_rec + self.lambda_align * loss_align
        else:
            loss = loss_rec
        
        return loss
    
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
        Predict scores for all items.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores