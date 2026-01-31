# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
GCSAN
################################################

Reference:
    Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

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
    """Graph Neural Network module for session-based recommendation.
    
    This module uses a GRU-style gating mechanism to update node representations
    based on the session graph structure.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        
        # GRU-style parameters for gating mechanism
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
        """Perform one step of GNN message passing.
        
        Args:
            A (torch.FloatTensor): Connection matrix of shape [batch_size, max_session_len, 2 * max_session_len]
            hidden (torch.FloatTensor): Node embeddings of shape [batch_size, max_session_len, embedding_size]
            
        Returns:
            torch.FloatTensor: Updated node embeddings of shape [batch_size, max_session_len, embedding_size]
        """
        # Aggregate information from incoming edges (Equation 1 - first part)
        input_in = (
            torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        # Aggregate information from outgoing edges (Equation 1 - second part)
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # Concatenate incoming and outgoing aggregations: a_t in Equation 1
        inputs = torch.cat([input_in, input_out], 2)  # [batch_size, max_session_len, embedding_size * 2]

        # GRU-style gating (Equation 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        # Split into reset, update, and new gates
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        # z_t: update gate, r_t: reset gate
        reset_gate = torch.sigmoid(i_r + h_r)  # r_t
        update_gate = torch.sigmoid(i_i + h_i)  # z_t
        new_gate = torch.tanh(i_n + reset_gate * h_n)  # h_tilde
        
        # h_t = (1 - z_t) * s_{t-1} + z_t * h_tilde
        hy = (1 - update_gate) * hidden + update_gate * new_gate
        return hy

    def forward(self, A, hidden):
        """Forward pass through multiple GNN steps.
        
        Args:
            A (torch.FloatTensor): Connection matrix
            hidden (torch.FloatTensor): Initial node embeddings
            
        Returns:
            torch.FloatTensor: Final node embeddings after all GNN steps
        """
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SelfAttentionLayer(nn.Module):
    """Self-Attention Layer as described in Section 3.3.
    
    Implements scaled dot-product attention followed by a feed-forward network
    with residual connection.
    """

    def __init__(self, hidden_size, n_heads, inner_size, attn_dropout_prob, hidden_dropout_prob, layer_norm_eps):
        super(SelfAttentionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        
        # Query, Key, Value projections (Equation 3)
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Feed-forward network (Equation 4)
        self.W_1 = nn.Linear(hidden_size, inner_size)
        self.W_2 = nn.Linear(inner_size, hidden_size)
        
        # Layer normalization and dropout
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        
        self.scale = math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        """Forward pass through self-attention layer.
        
        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.FloatTensor, optional): Attention mask
            
        Returns:
            torch.FloatTensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Multi-head self-attention (Equation 3)
        Q = self.W_Q(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Residual connection and layer norm for attention
        F_output = self.attn_layer_norm(hidden_states + self.hidden_dropout(context))
        
        # Feed-forward network with residual connection (Equation 4)
        ffn_output = self.W_2(F.relu(self.W_1(F_output)))
        E_output = self.ffn_layer_norm(F_output + self.hidden_dropout(ffn_output))
        
        return E_output


class GCSAN(SequentialRecommender):
    """Graph Contextualized Self-Attention Network for Session-based Recommendation.
    
    This model combines Graph Neural Networks for capturing local item transitions
    with Self-Attention Networks for modeling long-range dependencies.
    
    The architecture consists of:
    1. Session graph construction from item sequences
    2. GNN for learning node representations with local context
    3. Multi-layer self-attention for capturing global dependencies
    4. Prediction layer combining global and local preferences
    """

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__(config, dataset)

        # Load parameters from config
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.step = config["step"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.weight = config["weight"]  # omega in Equation 7
        self.reg_weight = config["reg_weight"]
        self.loss_type = config["loss_type"]
        self.device = config["device"]

        # Item embedding layer
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Graph Neural Network
        self.gnn = GNN(self.embedding_size, self.step)
        
        # Multi-layer Self-Attention Network (Section 3.3)
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                self.hidden_size,
                self.n_heads,
                self.inner_size,
                self.attn_dropout_prob,
                self.hidden_dropout_prob,
                self.layer_norm_eps
            )
            for _ in range(self.n_layers)
        ])
        
        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Loss functions
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Embedding regularization loss
        self.reg_loss = EmbLoss()

        # Parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters."""
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        """Construct session graph from item sequences.
        
        For each session, builds a directed graph where edges represent
        item transitions. Returns alias_inputs (mapping from sequence positions
        to unique node indices), adjacency matrix A, and unique items.
        
        Args:
            item_seq (torch.LongTensor): Item sequences of shape [batch_size, max_seq_len]
            
        Returns:
            tuple: (alias_inputs, A, items, mask)
                - alias_inputs: Mapping from positions to node indices [batch_size, max_seq_len]
                - A: Adjacency matrix [batch_size, max_seq_len, 2 * max_seq_len]
                - items: Unique items per session [batch_size, max_seq_len]
                - mask: Valid position mask [batch_size, max_seq_len]
        """
        # Create mask for valid positions (non-padding)
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
            # Incoming edges: M^I - normalized by in-degree
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            
            # Outgoing edges: M^O - normalized by out-degree
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            
            # Concatenate incoming and outgoing adjacency matrices
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            
            # Create alias inputs: mapping from sequence position to node index
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        
        # Convert to tensors and move to device
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)
        
        return alias_inputs, A, items, mask

    def _get_attention_mask(self, item_seq):
        """Generate attention mask for self-attention layers.
        
        Creates a causal mask that prevents attending to future positions
        and padding positions.
        
        Args:
            item_seq (torch.LongTensor): Item sequences of shape [batch_size, seq_len]
            
        Returns:
            torch.FloatTensor: Attention mask of shape [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = item_seq.size()
        
        # Padding mask: True for valid positions
        padding_mask = item_seq.gt(0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        # Causal mask: lower triangular
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, L, L]
        
        # Combine masks
        attention_mask = padding_mask & causal_mask
        
        # Convert to additive mask (0 for valid, -inf for invalid)
        attention_mask = attention_mask.float()
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        return attention_mask

    def forward(self, item_seq, item_seq_len):
        """Forward pass of GCSAN.
        
        Args:
            item_seq (torch.LongTensor): Item sequences [batch_size, max_seq_len]
            item_seq_len (torch.LongTensor): Sequence lengths [batch_size]
            
        Returns:
            torch.FloatTensor: Session representations [batch_size, hidden_size]
        """
        # Step 1: Construct session graph and get components
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        
        # Step 2: Get initial node embeddings
        hidden = self.item_embedding(items)  # [B, max_n_node, embedding_size]
        
        # Step 3: Apply GNN to update node representations (Section 3.2)
        hidden = self.gnn(A, hidden)  # H = [h_1, h_2, ..., h_n]
        
        # Step 4: Map node representations back to sequence positions using alias_inputs
        alias_inputs_expanded = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs_expanded)
        
        # Get the last item's GNN representation (h_n for local embedding)
        h_n = self.gather_indexes(seq_hidden, item_seq_len - 1)  # [B, hidden_size]
        
        # Step 5: Apply multi-layer self-attention (Section 3.3)
        attention_mask = self._get_attention_mask(item_seq)
        
        # Apply dropout to input
        E = self.emb_dropout(seq_hidden)
        
        # Stack self-attention layers (Equation 5, 6)
        for layer in self.self_attention_layers:
            E = layer(E, attention_mask)
        
        # E^(k) is the output after k self-attention layers
        # Get the last position as global embedding (E_n^(k))
        E_n = self.gather_indexes(E, item_seq_len - 1)  # [B, hidden_size]
        
        # Step 6: Combine global and local embeddings (Equation 7)
        # S_f = omega * E_n^(k) + (1 - omega) * h_n
        seq_output = self.weight * E_n + (1 - self.weight) * h_n
        
        return seq_output

    def calculate_loss(self, interaction):
        """Calculate training loss.
        
        Args:
            interaction: Batch data containing item sequences and targets
            
        Returns:
            torch.Tensor: Loss value
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        # Add embedding regularization loss (Equation 9)
        if self.reg_weight > 0:
            reg_loss = self.reg_loss(self.item_embedding.weight)
            loss = loss + self.reg_weight * reg_loss
        
        return loss

    def predict(self, interaction):
        """Predict scores for given test items.
        
        Args:
            interaction: Batch data containing item sequences and test items
            
        Returns:
            torch.Tensor: Prediction scores [batch_size]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items.
        
        Args:
            interaction: Batch data containing item sequences
            
        Returns:
            torch.Tensor: Prediction scores [batch_size, n_items]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores