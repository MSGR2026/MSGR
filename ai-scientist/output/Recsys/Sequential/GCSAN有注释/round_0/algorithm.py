# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
GCSAN
################################################

Reference:
    Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss, EmbLoss


class GNN(nn.Module):
    """Graph Neural Network module for session graph.
    
    This module implements the gated GNN described in the paper to update node vectors
    by propagating information through the session graph structure.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        
        # GRU-style gated parameters
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        # Linear transformations for incoming and outgoing edges
        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

    def GNNCell(self, A, hidden):
        """Single GNN propagation step.
        
        Implements Equation (1) and (2) from the paper:
        - Information aggregation from neighbors via adjacency matrix
        - GRU-style gated update for node representations
        
        Args:
            A: Connection matrix [batch_size, max_session_len, 2 * max_session_len]
               First half is incoming edges, second half is outgoing edges
            hidden: Node embeddings [batch_size, max_session_len, embedding_size]
            
        Returns:
            Updated node embeddings [batch_size, max_session_len, embedding_size]
        """
        # Equation (1): Aggregate neighbor information
        # a_t = Concat(M_t^I * ([s_1,...,s_n] * W_a^I + b^I), M_t^O * ([s_1,...,s_n] * W_a^O + b^O))
        input_in = (
            torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # Concatenate incoming and outgoing aggregations [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # Equation (2): GRU-style gated update
        # gi and gh for computing gates
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        # Split into reset, update, and new gates
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        # z_t = sigmoid(W_z * a_t + P_z * s_{t-1})
        reset_gate = torch.sigmoid(i_r + h_r)
        # r_t = sigmoid(W_r * a_t + P_r * s_{t-1})
        input_gate = torch.sigmoid(i_i + h_i)
        # h_tilde = tanh(W_h * a_t + P_h * (r_t ⊙ s_{t-1}))
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        # h_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ h_tilde
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        """Multi-step GNN propagation.
        
        Args:
            A: Connection matrix
            hidden: Initial node embeddings
            
        Returns:
            Updated node embeddings after `step` propagation steps
        """
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SelfAttentionLayer(nn.Module):
    """Self-Attention Layer as described in Section 3.3.
    
    Implements scaled dot-product attention followed by a point-wise
    feed-forward network with residual connection.
    """
    
    def __init__(self, hidden_size, inner_size, n_heads, attn_dropout_prob, hidden_dropout_prob):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        
        # Projection matrices for Q, K, V (Equation 3)
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Point-wise feed-forward network (Equation 4)
        self.W_1 = nn.Linear(hidden_size, inner_size)
        self.W_2 = nn.Linear(inner_size, hidden_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, H, attention_mask=None):
        """
        Implements Equation (3), (4), and (5) from the paper.
        
        Args:
            H: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Output after self-attention and FFN [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = H.size()
        
        # Equation (3): Self-attention
        # F = softmax((H * W^Q)(H * W^K)^T / sqrt(d)) * (H * W^V)
        Q = self.W_Q(H)  # [batch_size, seq_len, hidden_size]
        K = self.W_K(H)
        V = self.W_V(H)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Residual connection and layer norm for attention
        F_out = self.layer_norm1(H + self.hidden_dropout(context))
        
        # Equation (4): Point-wise feed-forward network with residual
        # E = ReLU(F * W_1 + b_1) * W_2 + b_2 + F
        ffn_output = self.W_2(F.relu(self.W_1(F_out)))
        E = self.layer_norm2(F_out + self.hidden_dropout(ffn_output))
        
        return E


class GCSAN(SequentialRecommender):
    """Graph Contextualized Self-Attention Network for Session-based Recommendation.
    
    This model combines:
    1. Graph Neural Network to capture local item transitions in session graph
    2. Multi-layer Self-Attention to capture long-range dependencies
    3. Weighted combination of global and local session representations
    
    Reference:
        Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for 
        Session-based Recommendation." in IJCAI 2019.
    """

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__(config, dataset)

        # Load parameters from config
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.step = config["step"]  # Number of GNN propagation steps
        self.n_layers = config["n_layers"]  # Number of self-attention layers
        self.n_heads = config["n_heads"]  # Number of attention heads
        self.inner_size = config["inner_size"]  # FFN inner dimension
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.loss_type = config["loss_type"]
        self.weight = config["weight"]  # Fusion weight ω in Equation (7)
        self.reg_weight = config["reg_weight"]  # Regularization weight λ in Equation (9)
        self.device = config["device"]

        # Item embedding layer
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Graph Neural Network layer
        self.gnn = GNN(self.embedding_size, self.step)
        
        # Multi-layer self-attention network (Equation 5, 6)
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                self.hidden_size, 
                self.inner_size, 
                self.n_heads,
                self.attn_dropout_prob,
                self.hidden_dropout_prob
            )
            for _ in range(self.n_layers)
        ])
        
        # Projection layer if embedding_size != hidden_size
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None
        
        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Embedding regularization loss
        self.reg_loss = EmbLoss()

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters."""
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        """Construct session graph from item sequence.
        
        As described in Section 3.2, this method:
        1. Builds a directed graph where (s_{i-1}, s_i) forms an edge
        2. Computes normalized adjacency matrices M^I (incoming) and M^O (outgoing)
        3. Creates alias_inputs to map sequence positions to unique node indices
        
        Args:
            item_seq: Item sequence tensor [batch_size, max_seq_len]
            
        Returns:
            alias_inputs: Mapping from sequence positions to node indices
            A: Concatenated adjacency matrix [batch_size, max_seq_len, 2 * max_seq_len]
            items: Unique item indices per session [batch_size, max_seq_len]
            mask: Valid position mask [batch_size, max_seq_len]
        """
        # Create mask for valid (non-padding) positions
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq_np = item_seq.cpu().numpy()
        
        for u_input in item_seq_np:
            # Get unique items in this session
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            
            # Build adjacency matrix
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                # Find indices of consecutive items in unique node list
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            # Normalize adjacency matrices (as described in Section 3.2)
            # M^I: incoming edges, normalized by column sum (in-degree)
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            
            # M^O: outgoing edges, normalized by row sum (out-degree)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            
            # Concatenate M^I and M^O
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            # Create alias mapping from sequence position to unique node index
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        
        # Convert to tensors and move to device
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def get_attention_mask(self, item_seq):
        """Generate attention mask for self-attention layers.
        
        Creates a causal mask that prevents attending to padding positions
        and future positions (for autoregressive modeling).
        
        Args:
            item_seq: Item sequence [batch_size, seq_len]
            
        Returns:
            Attention mask [batch_size, 1, 1, seq_len] with -inf for masked positions
        """
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        """Forward pass of GCSAN.
        
        Implements the full model pipeline:
        1. Build session graph and get adjacency matrices
        2. Apply GNN to get node representations H
        3. Apply multi-layer self-attention to get E^(k)
        4. Combine global (E_n^(k)) and local (h_n) representations
        
        Args:
            item_seq: Item sequence [batch_size, max_seq_len]
            item_seq_len: Length of each sequence [batch_size]
            
        Returns:
            Session representation [batch_size, hidden_size]
        """
        # Step 1: Construct session graph (Section 3.2)
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        
        # Step 2: Get item embeddings and apply GNN
        hidden = self.item_embedding(items)  # [batch_size, max_seq_len, embedding_size]
        hidden = self.gnn(A, hidden)  # GNN output H = [h_1, h_2, ..., h_n]
        
        # Map GNN output back to sequence positions using alias_inputs
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        
        # Get the last valid hidden state h_n for local embedding
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)  # [batch_size, embedding_size]
        
        # Step 3: Apply multi-layer self-attention (Section 3.3)
        # Project to hidden_size if needed
        if self.embedding_proj is not None:
            H = self.embedding_proj(seq_hidden)
        else:
            H = seq_hidden
            
        # Generate attention mask
        attention_mask = self.get_attention_mask(item_seq)
        
        # Multi-layer self-attention (Equation 5, 6)
        E = H
        for layer in self.self_attention_layers:
            E = layer(E, attention_mask)
        
        # Get global embedding E_n^(k) - last position of self-attention output
        at = self.gather_indexes(E, item_seq_len - 1)  # [batch_size, hidden_size]
        
        # Project ht to hidden_size if needed for fusion
        if self.embedding_proj is not None:
            ht_proj = self.embedding_proj(ht)
        else:
            ht_proj = ht
        
        # Step 4: Combine global and local representations (Equation 7)
        # S_f = ω * E_n^(k) + (1 - ω) * h_n
        seq_output = self.weight * at + (1 - self.weight) * ht_proj
        
        return seq_output

    def calculate_loss(self, interaction):
        """Calculate training loss.
        
        Implements Equation (9): cross-entropy loss with L2 regularization.
        
        Args:
            interaction: Batch data containing item sequences and targets
            
        Returns:
            Total loss value
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            
            # Project embeddings if needed
            if self.embedding_proj is not None:
                pos_items_emb = self.embedding_proj(pos_items_emb)
                neg_items_emb = self.embedding_proj(neg_items_emb)
            
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE loss
            # Get all item embeddings for scoring
            test_item_emb = self.item_embedding.weight
            if self.embedding_proj is not None:
                test_item_emb = self.embedding_proj(test_item_emb)
            
            # Equation (8): y_i = softmax(S_f^T * v_i)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        # Add L2 regularization on embeddings (λ||θ||^2 in Equation 9)
        if self.reg_weight > 0:
            reg_loss = self.reg_loss(self.item_embedding.weight)
            loss = loss + self.reg_weight * reg_loss
            
        return loss

    def predict(self, interaction):
        """Predict scores for specific test items.
        
        Args:
            interaction: Batch data containing item sequences and test items
            
        Returns:
            Prediction scores [batch_size]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        
        test_item_emb = self.item_embedding(test_item)
        if self.embedding_proj is not None:
            test_item_emb = self.embedding_proj(test_item_emb)
            
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items (full ranking).
        
        Args:
            interaction: Batch data containing item sequences
            
        Returns:
            Prediction scores for all items [batch_size, n_items]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        
        test_items_emb = self.item_embedding.weight
        if self.embedding_proj is not None:
            test_items_emb = self.embedding_proj(test_items_emb)
            
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores