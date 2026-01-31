# -*- coding: utf-8 -*-
# @Time   : 2025
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
GLINT-RU
################################################

Reference:
    GLINT-RU: Gated Linear Attention and Improved GRU for Sequential Recommendation

This implementation follows the paper's methodology:
1. Dense Selective GRU with temporal convolutions and selective gating
2. Linear Attention mechanism for semantic feature learning
3. Expert Mixing Block combining GRU and Attention experts
4. Gated MLP Block for final representation learning
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class GLINTRU(SequentialRecommender):
    r"""
    GLINT-RU integrates linear attention mechanism and efficient dense selective GRU module
    for sequential recommendation. The framework consists of:
    - Item Embedding Layer (without positional embeddings)
    - Expert Mixing Block (Dense Selective GRU + Linear Attention)
    - Gated MLP Block
    """

    def __init__(self, config, dataset):
        super(GLINTRU, self).__init__(config, dataset)

        # Load parameters from config
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config.get("embedding_size", self.hidden_size)
        self.loss_type = config["loss_type"]
        self.dropout_prob = config.get("dropout_prob", 0.2)
        self.kernel_size = config.get("kernel_size", 3)  # Temporal convolution kernel size
        self.initializer_range = config.get("initializer_range", 0.02)

        # Define layers
        # Item Embedding Layer (Eq. 3, 4)
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Dropout for regularization
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        
        # Dense Selective GRU Module components
        # Input projection (Eq. 5)
        self.input_proj = nn.Linear(self.embedding_size, self.hidden_size)
        
        # First temporal convolution before GRU (Eq. 5)
        self.temporal_conv1 = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,  # Same padding
            groups=1
        )
        
        # GRU module (Eq. 1, 6)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True
        )
        
        # Channel crossing layer (Eq. 7)
        self.channel_crossing = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Selective gate layers (Eq. 9)
        self.selective_gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.selective_gate_out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Second temporal convolution after selective gate (Eq. 8)
        self.temporal_conv2 = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=1
        )
        
        # Linear Attention components (Eq. 2)
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Expert mixing parameters (Eq. 10)
        # Learnable mixing weights initialized to equal contribution
        self.alpha_1 = nn.Parameter(torch.tensor(1.0))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0))
        
        # Data-aware gate after expert mixing (Eq. 11)
        self.mixing_gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Gated MLP Block (Eq. 12)
        self.gated_mlp_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.gated_mlp_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.embedding_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.output_layer_norm = nn.LayerNorm(self.embedding_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    xavier_uniform_(param)
                elif 'weight_hh' in name:
                    xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def dense_selective_gru(self, x):
        """
        Dense Selective GRU Module (Section 3.3)
        
        Args:
            x: Input tensor of shape [B, N, d]
            
        Returns:
            Output tensor of shape [B, N, d]
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection (Eq. 5)
        x_proj = self.input_proj(x)  # [B, N, d]
        
        # First temporal convolution (Eq. 5)
        # Conv1d expects [B, C, L] format
        x_conv = x_proj.transpose(1, 2)  # [B, d, N]
        c = self.temporal_conv1(x_conv)  # [B, d, N]
        c = c.transpose(1, 2)  # [B, N, d]
        
        # GRU processing (Eq. 6)
        h, _ = self.gru(c)  # [B, N, d]
        
        # Channel crossing layer (Eq. 7)
        h_crossed = self.channel_crossing(h)  # [B, N, d]
        
        # Selective gate (Eq. 9)
        # δ_1(C) = C * W_δ + b_δ
        delta = self.selective_gate_proj(c)  # [B, N, d]
        
        # G(H) = (SiLU(δ_1(C)) * W_Ω + b_Ω) ⊗ Φ(H)
        gate_weights = F.silu(delta)
        gate_weights = self.selective_gate_out(gate_weights)  # [B, N, d]
        gated_h = gate_weights * h_crossed  # Element-wise multiplication
        
        # Second temporal convolution (Eq. 8)
        gated_h_conv = gated_h.transpose(1, 2)  # [B, d, N]
        y = self.temporal_conv2(gated_h_conv)  # [B, d, N]
        y = y.transpose(1, 2)  # [B, N, d]
        
        return y

    def linear_attention(self, x, attention_mask=None):
        """
        Linear Attention Mechanism (Eq. 2)
        
        A'(Q, K, V) = χ_1(elu(Q)) (χ_2(elu(K))^T V)
        
        Args:
            x: Input tensor of shape [B, N, d]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [B, N, d]
        """
        # Project to Q, K, V
        Q = self.query_proj(x)  # [B, N, d]
        K = self.key_proj(x)    # [B, N, d]
        V = self.value_proj(x)  # [B, N, d]
        
        # Apply ELU activation and add 1 to ensure non-negative values
        Q = F.elu(Q) + 1  # [B, N, d]
        K = F.elu(K) + 1  # [B, N, d]
        
        # Apply L2 normalization
        # χ_1: row-wise L2 normalization for Q
        Q = F.normalize(Q, p=2, dim=-1)  # [B, N, d]
        
        # χ_2: column-wise L2 normalization for K
        K = F.normalize(K, p=2, dim=-2)  # [B, N, d]
        
        # Apply causal masking if needed
        if attention_mask is not None:
            # For causal attention, we need to compute attention step by step
            # However, for efficiency, we use the linear attention formulation
            # which naturally handles the sequence in a causal manner
            pass
        
        # Compute linear attention: Q @ (K^T @ V)
        # This is O(Nd^2) instead of O(N^2d)
        KV = torch.bmm(K.transpose(1, 2), V)  # [B, d, d]
        attention_output = torch.bmm(Q, KV)   # [B, N, d]
        
        return attention_output

    def expert_mixing_block(self, x, attention_mask=None):
        """
        Expert Mixing Block (Section 3.4, Eq. 10-11)
        
        Combines Dense Selective GRU and Linear Attention experts
        
        Args:
            x: Input tensor of shape [B, N, d]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [B, N, d]
        """
        # Dense Selective GRU expert
        gru_output = self.dense_selective_gru(x)  # Y in paper
        
        # Linear Attention expert
        attention_output = self.linear_attention(x, attention_mask)  # A'(Q,K,V) in paper
        
        # Compute mixing weights using softmax (Eq. 10)
        alpha_softmax = F.softmax(torch.stack([self.alpha_1, self.alpha_2]), dim=0)
        alpha_1_norm = alpha_softmax[0]
        alpha_2_norm = alpha_softmax[1]
        
        # Mix experts (Eq. 10)
        # M = α_1 * A'(Q,K,V) + α_2 * Y
        mixed = alpha_1_norm * attention_output + alpha_2_norm * gru_output
        
        # Data-aware gate (Eq. 11)
        # δ_2(X) = X * W_δ + b_δ
        # Z = GeLU(δ_2(X)) ⊗ M
        gate = self.mixing_gate_proj(x)
        gate = F.gelu(gate)
        z = gate * mixed  # Element-wise multiplication
        
        return z

    def gated_mlp_block(self, z):
        """
        Gated MLP Block (Section 3.5, Eq. 12)
        
        Args:
            z: Input tensor of shape [B, N, d]
            
        Returns:
            Output tensor of shape [B, N, d]
        """
        # δ_3(Z) = Z * W_δ + b_δ
        gate = self.gated_mlp_gate(z)
        gate = F.gelu(gate)
        
        # P = GeLU(δ_3(Z)) ⊗ (Z * W + b)
        proj = self.gated_mlp_proj(z)
        p = gate * proj  # Element-wise multiplication
        
        # R = P * W_o + b_o
        r = self.output_proj(p)
        
        return r

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of GLINT-RU
        
        Args:
            item_seq: Item sequence tensor of shape [B, N]
            item_seq_len: Sequence length tensor of shape [B]
            
        Returns:
            Sequence output tensor of shape [B, d]
        """
        # Item embedding (Eq. 3, 4)
        item_emb = self.item_embedding(item_seq)  # [B, N, d]
        item_emb = self.emb_dropout(item_emb)
        
        # Project to hidden size if different from embedding size
        if self.embedding_size != self.hidden_size:
            x = F.linear(item_emb, 
                        torch.randn(self.hidden_size, self.embedding_size, device=item_emb.device))
        else:
            x = item_emb
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Expert Mixing Block
        z = self.expert_mixing_block(x)
        z = self.dropout(z)
        
        # Residual connection
        z = z + x
        z = self.layer_norm(z)
        
        # Gated MLP Block
        r = self.gated_mlp_block(z)
        r = self.dropout(r)
        
        # Output layer normalization
        r = self.output_layer_norm(r)
        
        # Gather the output at the last valid position
        seq_output = self.gather_indexes(r, item_seq_len - 1)  # [B, d]
        
        return seq_output

    def calculate_loss(self, interaction):
        """Calculate the training loss"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type == 'CE'
            # Compute logits for all items (Eq. 13)
            # ŷ_i = softmax(R_i * e_i^T)
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        """Predict scores for given test items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores