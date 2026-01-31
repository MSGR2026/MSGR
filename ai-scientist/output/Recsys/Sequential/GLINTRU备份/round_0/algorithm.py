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
- Dense Selective GRU module with temporal convolutions
- Linear Attention mechanism with ELU and L2 normalization
- Expert Mixing Block combining GRU and Attention
- Gated MLP Block for final prediction
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class GLINTRU(SequentialRecommender):
    r"""GLINT-RU integrates linear attention mechanism and efficient dense selective GRU module
    for sequential recommendation.
    
    The framework consists of:
    - Item Embedding Layer (without positional embeddings)
    - Expert Mixing Block (Dense Selective GRU + Linear Attention)
    - Gated MLP Block
    - Prediction Layer
    """

    def __init__(self, config, dataset):
        super(GLINTRU, self).__init__(config, dataset)

        # Load parameters info
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.dropout_prob = config["dropout_prob"]
        self.kernel_size = config["kernel_size"]  # Temporal convolution kernel size k
        self.inner_size = config["inner_size"]  # FFN inner size
        
        # Define layers
        # Item Embedding Layer (Eq. 3-4)
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        
        # Dense Selective GRU Module components
        # Pre-GRU linear projection (Eq. 5: X * W_0 + b_0)
        self.pre_gru_linear = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Pre-GRU temporal convolution (Eq. 5)
        self.pre_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,  # Same padding
            groups=1
        )
        
        # GRU module (Eq. 1, 6)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        
        # Channel crossing layer (Eq. 7: Phi(H) = H * W_H + b_H)
        self.channel_crossing = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Selective Gate components (Eq. 9)
        self.gate_delta1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate_omega1 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Post-GRU temporal convolution (Eq. 8)
        self.post_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            groups=1
        )
        
        # Linear Attention components (Eq. 2)
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Expert Mixing parameters (Eq. 10)
        # Initialize alpha parameters for mixing
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
        # Data-aware gate after expert mixing (Eq. 11)
        self.gate_delta2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Gated MLP Block components (Eq. 12)
        self.gate_delta3 = nn.Linear(self.hidden_size, self.inner_size)
        self.mlp_linear = nn.Linear(self.hidden_size, self.inner_size)
        self.output_linear = nn.Linear(self.inner_size, self.hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
        elif isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def dense_selective_gru(self, x):
        """
        Dense Selective GRU Module (Section 3.3)
        
        Args:
            x: Input tensor of shape [B, N, d]
            
        Returns:
            Y: Output tensor of shape [B, N, d]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Eq. 5: Pre-GRU linear projection and temporal convolution
        # C = TemporalConv1d(X * W_0 + b_0)
        pre_linear = self.pre_gru_linear(x)  # [B, N, d]
        
        # Conv1d expects [B, C, L] format
        pre_linear_transposed = pre_linear.transpose(1, 2)  # [B, d, N]
        C = self.pre_conv(pre_linear_transposed)  # [B, d, N]
        C = C.transpose(1, 2)  # [B, N, d]
        
        # Eq. 6: GRU processing
        H, _ = self.gru(C)  # [B, N, d]
        
        # Eq. 7: Channel crossing layer
        # Phi(H) = H * W_H + b_H
        phi_H = self.channel_crossing(H)  # [B, N, d]
        
        # Eq. 9: Selective Gate
        # delta_1(C) = C * W_delta^(1) + b_delta^(1)
        delta1 = self.gate_delta1(C)  # [B, N, d]
        
        # G(H) = (SiLU(delta_1(C)) * W_Omega^(1) + b_Omega^(1)) ⊗ Phi(H)
        gate_weights = F.silu(delta1)  # [B, N, d]
        gate_weights = self.gate_omega1(gate_weights)  # [B, N, d]
        G = gate_weights * phi_H  # Element-wise multiplication [B, N, d]
        
        # Eq. 8: Post-GRU temporal convolution
        # Y = TemporalConv1d(G)
        G_transposed = G.transpose(1, 2)  # [B, d, N]
        Y = self.post_conv(G_transposed)  # [B, d, N]
        Y = Y.transpose(1, 2)  # [B, N, d]
        
        return Y

    def linear_attention(self, x):
        """
        Linear Attention Mechanism (Section 2.3, Eq. 2)
        
        A'(Q, K, V) = χ_1(elu(Q)) (χ_2(elu(K))^T V)
        
        where χ_1 and χ_2 are row-wise and column-wise L2 normalization
        
        Args:
            x: Input tensor of shape [B, N, d]
            
        Returns:
            attention_output: Output tensor of shape [B, N, d]
        """
        # Project to Q, K, V
        Q = self.query_proj(x)  # [B, N, d]
        K = self.key_proj(x)    # [B, N, d]
        V = self.value_proj(x)  # [B, N, d]
        
        # Apply ELU activation and add 1 to ensure positive values
        # ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        Q_elu = F.elu(Q) + 1  # [B, N, d]
        K_elu = F.elu(K) + 1  # [B, N, d]
        
        # Row-wise L2 normalization for Q (χ_1)
        Q_norm = F.normalize(Q_elu, p=2, dim=-1)  # [B, N, d]
        
        # Column-wise L2 normalization for K (χ_2)
        K_norm = F.normalize(K_elu, p=2, dim=-2)  # [B, N, d]
        
        # Linear attention: Q_norm @ (K_norm^T @ V)
        # This changes the order of computation for O(Nd^2) complexity
        # First compute K^T @ V: [B, d, N] @ [B, N, d] = [B, d, d]
        KV = torch.bmm(K_norm.transpose(1, 2), V)  # [B, d, d]
        
        # Then compute Q @ KV: [B, N, d] @ [B, d, d] = [B, N, d]
        attention_output = torch.bmm(Q_norm, KV)  # [B, N, d]
        
        return attention_output

    def expert_mixing_block(self, x):
        """
        Expert Mixing Block (Section 3.4, Eq. 10-11)
        
        Combines Dense Selective GRU and Linear Attention outputs
        with learnable mixing weights.
        
        Args:
            x: Input tensor of shape [B, N, d]
            
        Returns:
            Z: Output tensor of shape [B, N, d]
        """
        # Get outputs from both experts
        Y = self.dense_selective_gru(x)  # [B, N, d]
        A_prime = self.linear_attention(x)  # [B, N, d]
        
        # Eq. 10: Expert mixing with softmax normalized weights
        # alpha_i^(t) = softmax(alpha_i^(t-1))
        alpha_softmax = F.softmax(torch.stack([self.alpha1, self.alpha2]), dim=0)
        alpha1_norm = alpha_softmax[0]
        alpha2_norm = alpha_softmax[1]
        
        # M = alpha_1 * A'(Q, K, V) + alpha_2 * Y
        M = alpha1_norm * A_prime + alpha2_norm * Y  # [B, N, d]
        
        # Eq. 11: Data-aware gate
        # delta_2(X) = X * W_delta^(2) + b_delta^(2)
        delta2 = self.gate_delta2(x)  # [B, N, d]
        
        # Z = GeLU(delta_2(X)) ⊗ M
        Z = F.gelu(delta2) * M  # [B, N, d]
        
        return Z

    def gated_mlp_block(self, z):
        """
        Gated MLP Block (Section 3.5, Eq. 12)
        
        Args:
            z: Input tensor of shape [B, N, d]
            
        Returns:
            R: Item representation tensor of shape [B, N, d]
        """
        # Eq. 12: Gated MLP
        # delta_3(Z) = Z * W_delta^(3) + b_delta^(3)
        delta3 = self.gate_delta3(z)  # [B, N, inner_size]
        
        # P = GeLU(delta_3(Z)) ⊗ (Z * W + b)
        mlp_output = self.mlp_linear(z)  # [B, N, inner_size]
        P = F.gelu(delta3) * mlp_output  # [B, N, inner_size]
        
        # R = P * W_o + b_o
        R = self.output_linear(P)  # [B, N, d]
        
        return R

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of GLINT-RU
        
        Args:
            item_seq: Item sequence tensor of shape [B, N]
            item_seq_len: Sequence length tensor of shape [B]
            
        Returns:
            seq_output: Sequence representation of shape [B, d]
        """
        # Eq. 3-4: Item Embedding Layer (no positional embeddings)
        item_emb = self.item_embedding(item_seq)  # [B, N, d]
        item_emb = self.emb_dropout(item_emb)
        
        # Layer normalization for stability
        x = self.layer_norm(item_emb)  # [B, N, d]
        
        # Expert Mixing Block
        z = self.expert_mixing_block(x)  # [B, N, d]
        
        # Residual connection
        z = z + x
        z = self.dropout(z)
        
        # Gated MLP Block
        R = self.gated_mlp_block(z)  # [B, N, d]
        
        # Residual connection
        R = R + z
        R = self.dropout(R)
        
        # Extract the representation at the last valid position
        # This is the item representation for prediction
        seq_output = self.gather_indexes(R, item_seq_len - 1)  # [B, d]
        
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
        else:  # self.loss_type = 'CE'
            # Eq. 13: Prediction scores via softmax
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        """Predict scores for test items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items (full ranking)"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores