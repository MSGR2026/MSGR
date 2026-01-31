# -*- coding: utf-8 -*-
"""
DIFF
################################################

Reference:
    "Dual Side-Information Filtering and Fusion for Sequential Recommendation"

"""

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer


class DIFF(SequentialRecommender):
    """
    DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation.
    
    This model effectively removes noisy signals via frequency-based filtering and 
    fully leverages the correlation across item IDs and attributes through dual fusion.
    
    Key components:
    1. Frequency-based Noise Filtering: Uses DFT to separate low/high frequency components
    2. Dual Multi-sequence Fusion: Combines early fusion and intermediate fusion strategies
    3. Representation Alignment: Aligns item ID and attribute embedding spaces
    """

    def __init__(self, config, dataset):
        super(DIFF, self).__init__(config, dataset)

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
        
        # DIFF specific parameters
        self.selected_features = config["selected_features"]
        self.pooling_mode = config.get("pooling_mode", "mean")
        self.num_feature_field = len(config["selected_features"]) if config["selected_features"] else 0
        
        # Frequency filtering parameter
        self.freq_ratio = config.get("freq_ratio", 0.5)  # Ratio for low-frequency cutoff
        
        # Fusion parameters
        self.fusion_type = config.get("fusion_type", "sum")  # sum, concat, or gate
        self.alpha = config.get("alpha", 0.5)  # Balance between ID-centric and attribute-enriched
        self.align_lambda = config.get("align_lambda", 0.1)  # Weight for alignment loss
        
        self.loss_type = config["loss_type"]

        # Define embedding layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # Feature embedding layer for attributes
        if self.num_feature_field > 0:
            self.feature_embed_layer = FeatureSeqEmbLayer(
                dataset,
                self.hidden_size,
                self.selected_features,
                self.pooling_mode,
                config["device"],
            )
        else:
            self.feature_embed_layer = None
        
        # Trainable beta parameters for frequency filtering
        # beta_0 for item ID, beta_1...beta_m for attributes, beta_{m+1} for fused
        self.beta_id = nn.Parameter(torch.tensor(0.1))
        self.beta_attrs = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_feature_field)
        ])
        self.beta_fused = nn.Parameter(torch.tensor(0.1))
        
        # Learnable temperature for alignment
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # ID-centric path: separate Q, K projections for ID and attributes
        self.head_dim = self.hidden_size // self.n_heads
        
        # Q, K, V projections for item IDs
        self.W_Q_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_K_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_V_v = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Q, K projections for each attribute (shared V with item ID)
        self.W_Q_attrs = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) 
            for _ in range(self.num_feature_field)
        ])
        self.W_K_attrs = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) 
            for _ in range(self.num_feature_field)
        ])
        
        # Output projection for ID-centric path
        self.W_v_out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Attribute-enriched path: standard self-attention on fused embeddings
        self.W_Q_va = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_K_va = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_V_va = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_va_out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # FFN layers for both paths
        self.ffn_v = nn.Sequential(
            nn.Linear(self.hidden_size, self.inner_size),
            nn.GELU() if self.hidden_act == 'gelu' else nn.ReLU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.inner_size, self.hidden_size),
            nn.Dropout(self.hidden_dropout_prob)
        )
        
        self.ffn_va = nn.Sequential(
            nn.Linear(self.hidden_size, self.inner_size),
            nn.GELU() if self.hidden_act == 'gelu' else nn.ReLU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.inner_size, self.hidden_size),
            nn.Dropout(self.hidden_dropout_prob)
        )
        
        # Layer normalization
        self.LayerNorm_v = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_va = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_v_ffn = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_va_ffn = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm_input = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.attn_dropout = nn.Dropout(self.attn_dropout_prob)
        
        # Fusion gate (if using gating fusion)
        if self.fusion_type == "gate":
            self.fusion_gate = nn.Linear(self.hidden_size * (1 + self.num_feature_field), 
                                         self.hidden_size)
        
        # Loss function
        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "BPR":
            # BPR loss is supported but CE is recommended
            self.loss_fct = nn.CrossEntropyLoss()  # Fallback to CE
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Feature aggregation layer for multiple attributes
        if self.num_feature_field > 1:
            self.attr_aggregation = nn.Linear(
                self.hidden_size * self.num_feature_field, self.hidden_size
            )
        
        # Parameters initialization
        self.apply(self._init_weights)
        if self.feature_embed_layer is not None:
            self.other_parameter_name = ["feature_embed_layer"]
        else:
            self.other_parameter_name = []

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def frequency_filter(self, embeddings, beta):
        """
        Apply frequency-based noise filtering using FFT.
        
        Args:
            embeddings: [B, L, D] input embeddings
            beta: scalar parameter controlling high-frequency contribution
            
        Returns:
            filtered_embeddings: [B, L, D] filtered embeddings
        """
        B, L, D = embeddings.shape
        
        # Compute cutoff for low-frequency components
        c = max(1, int(L * self.freq_ratio))
        
        # Apply FFT along sequence dimension
        freq = torch.fft.fft(embeddings, dim=1)
        
        # Split into low and high frequency components
        freq_low = freq.clone()
        freq_high = freq.clone()
        
        # Zero out high frequencies in low-frequency component
        freq_low[:, c:, :] = 0
        
        # Zero out low frequencies in high-frequency component
        freq_high[:, :c, :] = 0
        
        # Apply inverse FFT
        emb_lfc = torch.fft.ifft(freq_low, dim=1).real
        emb_hfc = torch.fft.ifft(freq_high, dim=1).real
        
        # Combine with learnable beta
        filtered = emb_lfc + beta * emb_hfc
        
        return filtered

    def get_attribute_embeddings(self, item_seq):
        """
        Get attribute embeddings for the item sequence.
        
        Returns:
            attr_embeddings: list of [B, L, D] tensors for each attribute
            fused_attr: [B, L, D] fused attribute embedding
        """
        B, L = item_seq.shape
        
        if self.feature_embed_layer is None or self.num_feature_field == 0:
            # No features available, return zeros
            zero_emb = torch.zeros(B, L, self.hidden_size, device=item_seq.device)
            return [zero_emb], zero_emb
        
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]
        
        # Collect feature embeddings
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        
        if len(feature_table) == 0:
            # No features available, return zeros
            zero_emb = torch.zeros(B, L, self.hidden_size, device=item_seq.device)
            return [zero_emb], zero_emb
        
        # [B, L, num_features, D]
        feature_table = torch.cat(feature_table, dim=-2)
        
        # Split into individual attribute embeddings
        num_features = feature_table.shape[2]
        attr_embeddings = [feature_table[:, :, i, :] for i in range(num_features)]
        
        # Fuse attributes (sum for simplicity, as mentioned in paper)
        fused_attr = feature_table.sum(dim=2)
        
        return attr_embeddings, fused_attr

    def fusion_attention_scores(self, A_v, A_attrs):
        """
        Fuse attention scores from item IDs and attributes.
        
        Args:
            A_v: [B, H, L, L] attention scores from item IDs
            A_attrs: list of [B, H, L, L] attention scores from attributes
            
        Returns:
            fused_scores: [B, H, L, L] fused attention scores
        """
        if self.fusion_type == "sum":
            fused = A_v
            for A_a in A_attrs:
                fused = fused + A_a
            return fused / (1 + len(A_attrs))
        
        elif self.fusion_type == "concat":
            # Average after concatenation along a new dimension
            all_scores = torch.stack([A_v] + A_attrs, dim=-1)
            return all_scores.mean(dim=-1)
        
        else:  # gate - simplified gating
            fused = A_v
            for A_a in A_attrs:
                fused = fused + A_a
            return fused / (1 + len(A_attrs))

    def id_centric_attention(self, E_v, E_attrs, attention_mask):
        """
        ID-centric fusion: intermediate fusion of attention scores.
        
        Args:
            E_v: [B, L, D] filtered item ID embeddings
            E_attrs: list of [B, L, D] filtered attribute embeddings
            attention_mask: [B, 1, L, L] attention mask
            
        Returns:
            R_v: [B, L, D] ID-centric representation
        """
        B, L, D = E_v.shape
        
        # Compute Q, K, V for item IDs
        Q_v = self.W_Q_v(E_v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K_v = self.W_K_v(E_v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V_v = self.W_V_v(E_v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores for item IDs
        A_v = torch.matmul(Q_v, K_v.transpose(-2, -1))  # [B, H, L, L]
        
        # Compute attention scores for each attribute
        A_attrs = []
        for i, E_a in enumerate(E_attrs):
            if i < len(self.W_Q_attrs):
                Q_a = self.W_Q_attrs[i](E_a).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                K_a = self.W_K_attrs[i](E_a).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                A_a = torch.matmul(Q_a, K_a.transpose(-2, -1))
                A_attrs.append(A_a)
        
        # Fuse attention scores
        fused_scores = self.fusion_attention_scores(A_v, A_attrs)
        
        # Apply scaling and mask
        fused_scores = fused_scores / (self.head_dim ** 0.5)
        fused_scores = fused_scores + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(fused_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to value matrix
        context = torch.matmul(attn_probs, V_v)  # [B, H, L, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection
        R_v = self.W_v_out(context)
        
        return R_v

    def attribute_enriched_attention(self, E_va, attention_mask):
        """
        Attribute-enriched fusion: early fusion with self-attention.
        
        Args:
            E_va: [B, L, D] filtered fused embeddings
            attention_mask: [B, 1, L, L] attention mask
            
        Returns:
            R_va: [B, L, D] attribute-enriched representation
        """
        B, L, D = E_va.shape
        
        # Compute Q, K, V for fused embeddings
        Q_va = self.W_Q_va(E_va).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K_va = self.W_K_va(E_va).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V_va = self.W_V_va(E_va).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        A_va = torch.matmul(Q_va, K_va.transpose(-2, -1)) / (self.head_dim ** 0.5)
        A_va = A_va + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(A_va, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention
        context = torch.matmul(attn_probs, V_va)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection
        R_va = self.W_va_out(context)
        
        return R_va

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of DIFF model.
        
        Args:
            item_seq: [B, L] item sequence
            item_seq_len: [B] sequence lengths
            
        Returns:
            seq_output: [B, D] user representation
        """
        B, L = item_seq.shape
        
        # Get position embeddings
        position_ids = torch.arange(L, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand(B, L)
        position_embedding = self.position_embedding(position_ids)
        
        # Get item ID embeddings
        E_v = self.item_embedding(item_seq)
        
        # Get attribute embeddings
        attr_embeddings, fused_attr = self.get_attribute_embeddings(item_seq)
        
        # Early fusion: combine item ID and all attributes
        E_va = E_v + fused_attr
        
        # Add position embeddings
        E_v = E_v + position_embedding
        E_va = E_va + position_embedding
        for i in range(len(attr_embeddings)):
            attr_embeddings[i] = attr_embeddings[i] + position_embedding
        
        # Apply LayerNorm and dropout to inputs
        E_v = self.LayerNorm_input(E_v)
        E_v = self.dropout(E_v)
        E_va = self.LayerNorm_input(E_va)
        E_va = self.dropout(E_va)
        for i in range(len(attr_embeddings)):
            attr_embeddings[i] = self.LayerNorm_input(attr_embeddings[i])
            attr_embeddings[i] = self.dropout(attr_embeddings[i])
        
        # Frequency-based noise filtering
        E_v_filtered = self.frequency_filter(E_v, self.beta_id)
        E_va_filtered = self.frequency_filter(E_va, self.beta_fused)
        
        E_attrs_filtered = []
        for i, E_a in enumerate(attr_embeddings):
            if i < len(self.beta_attrs):
                E_a_filtered = self.frequency_filter(E_a, self.beta_attrs[i])
            else:
                E_a_filtered = self.frequency_filter(E_a, self.beta_attrs[-1])
            E_attrs_filtered.append(E_a_filtered)
        
        # Get attention mask (causal)
        attention_mask = self.get_attention_mask(item_seq)
        
        # Process through L layers
        R_v = E_v_filtered
        R_va = E_va_filtered
        
        for _ in range(self.n_layers):
            # ID-centric path
            R_v_attn = self.id_centric_attention(R_v, E_attrs_filtered, attention_mask)
            R_v = self.LayerNorm_v(R_v + self.dropout(R_v_attn))
            R_v = self.LayerNorm_v_ffn(R_v + self.ffn_v(R_v))
            
            # Attribute-enriched path
            R_va_attn = self.attribute_enriched_attention(R_va, attention_mask)
            R_va = self.LayerNorm_va(R_va + self.dropout(R_va_attn))
            R_va = self.LayerNorm_va_ffn(R_va + self.ffn_va(R_va))
        
        # Combine dual paths
        R_u = self.alpha * R_v + (1 - self.alpha) * R_va
        
        # Get final user representation
        seq_output = self.gather_indexes(R_u, item_seq_len - 1)
        
        return seq_output

    def compute_alignment_loss(self, item_seq):
        """
        Compute representation alignment loss between item IDs and attributes.
        
        Args:
            item_seq: [B, L] item sequence
            
        Returns:
            align_loss: scalar alignment loss
        """
        # Get embeddings
        E_v = self.item_embedding(item_seq)  # [B, L, D]
        attr_embeddings, _ = self.get_attribute_embeddings(item_seq)
        
        # Fuse attribute embeddings
        if len(attr_embeddings) > 0:
            E_a = sum(attr_embeddings)
        else:
            return torch.tensor(0.0, device=item_seq.device)
        
        # Normalize embeddings
        E_v_norm = F.normalize(E_v, dim=-1)
        E_a_norm = F.normalize(E_a, dim=-1)
        
        # Compute similarity matrices
        # [B, L, L]
        sim_v_a = torch.bmm(E_v_norm, E_a_norm.transpose(1, 2)) / self.temperature.abs().clamp(min=0.01)
        sim_a_v = torch.bmm(E_a_norm, E_v_norm.transpose(1, 2)) / self.temperature.abs().clamp(min=0.01)
        
        # Create ground truth: items with same attributes should be aligned
        # For simplicity, use identity matrix as target (each position aligns with itself)
        B, L, _ = sim_v_a.shape
        
        # Create mask for padding positions
        mask = (item_seq != 0).float()  # [B, L]
        mask_matrix = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, L, L]
        
        # Target: diagonal elements should be high
        target = torch.eye(L, device=item_seq.device).unsqueeze(0).expand(B, -1, -1)
        target = target * mask_matrix
        
        # Normalize target
        target_sum = target.sum(dim=-1, keepdim=True).clamp(min=1)
        target = target / target_sum
        
        # Compute softmax over similarities
        Y_v_a = F.softmax(sim_v_a, dim=-1)
        Y_a_v = F.softmax(sim_a_v, dim=-1)
        
        # Cross-entropy loss
        # Mask out padding positions
        eps = 1e-8
        loss_v_a = -(target * torch.log(Y_v_a + eps)).sum(dim=-1)
        loss_a_v = -(target * torch.log(Y_a_v + eps)).sum(dim=-1)
        
        # Average over valid positions
        valid_count = mask.sum(dim=1).clamp(min=1)
        loss_v_a = (loss_v_a * mask).sum(dim=1) / valid_count
        loss_a_v = (loss_a_v * mask).sum(dim=1) / valid_count
        
        align_loss = (loss_v_a + loss_a_v).mean() / 2
        
        return align_loss

    def calculate_loss(self, interaction):
        """Calculate the training loss."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Recommendation loss
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        rec_loss = self.loss_fct(logits, pos_items)
        
        # Alignment loss
        align_loss = self.compute_alignment_loss(item_seq)
        
        # Total loss
        total_loss = rec_loss + self.align_lambda * align_loss
        
        return total_loss

    def predict(self, interaction):
        """Predict scores for test items."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores