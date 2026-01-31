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
    """Graph Neural Network module for GCSAN.
    
    This module implements the gated graph neural network to capture
    local session information through item transitions.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        
        # GRU-style gating parameters
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
        """Single step of GNN computation.
        
        Args:
            A: Connection matrix [batch_size, max_session_len, 2 * max_session_len]
            hidden: Item node embedding matrix [batch_size, max_session_len, embedding_size]
            
        Returns:
            Updated hidden states [batch_size, max_session_len, embedding_size]
        """
        # Aggregate information from incoming edges
        input_in = (
            torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        # Aggregate information from outgoing edges
        input_out = (
            torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden))
            + self.b_ioh
        )
        # Concatenate incoming and outgoing information
        inputs = torch.cat([input_in, input_out], 2)

        # GRU-style gating mechanism
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - update_gate) * hidden + update_gate * new_gate
        return hy

    def forward(self, A, hidden):
        """Forward pass through multiple GNN steps.
        
        Args:
            A: Connection matrix
            hidden: Initial node embeddings
            
        Returns:
            Updated node embeddings after self.step iterations
        """
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SelfAttentionLayer(nn.Module):
    """Self-Attention Layer for GCSAN.
    
    Implements scaled dot-product attention with feed-forward network
    and residual connections.
    """

    def __init__(self, hidden_size, inner_size, dropout_prob=0.2):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        
        # Attention projections
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Feed-forward network
        self.W_1 = nn.Linear(hidden_size, inner_size)
        self.W_2 = nn.Linear(inner_size, hidden_size)
        
        # Dropout and layer normalization
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.scale = math.sqrt(hidden_size)

    def forward(self, hidden, attention_mask=None):
        """Forward pass of self-attention layer.
        
        Args:
            hidden: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Self-attention
        Q = self.W_Q(hidden)
        K = self.W_K(hidden)
        V = self.W_V(hidden)
        
        # Scaled dot-product attention
        # attention_scores: [batch_size, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        
        if attention_mask is not None:
            # attention_mask is [batch_size, 1, 1, seq_len], broadcast to [batch_size, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask.squeeze(1)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        
        # First residual connection (after attention)
        hidden = self.layer_norm1(hidden + self.dropout(context))
        
        # Feed-forward network with residual connection
        ff_output = self.W_2(F.relu(self.W_1(hidden)))
        output = self.layer_norm2(hidden + self.dropout(ff_output))
        
        return output


class GCSAN(SequentialRecommender):
    """Graph Contextualized Self-Attention Network for Session-based Recommendation.
    
    GCSAN combines graph neural networks with self-attention mechanism to capture
    both local session structure and global dependencies between items.
    
    The model first constructs a session graph and uses GNN to obtain node embeddings,
    then applies multi-layer self-attention to capture long-range dependencies.
    Finally, it combines global preference (from self-attention) and local interest
    (from GNN) for prediction.
    """

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__(config, dataset)

        # Load parameters from config
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config.get("hidden_size", self.embedding_size)
        self.step = config.get("step", 1)
        self.n_layers = config.get("n_layers", 1)
        self.inner_size = config.get("inner_size", self.hidden_size * 4)
        self.dropout_prob = config.get("dropout_prob", 0.2)
        self.weight = config.get("weight", 0.5)  # Fusion weight omega
        self.reg_weight = config.get("reg_weight", 0.0)
        self.loss_type = config.get("loss_type", "BPR")
        self.device = config["device"]

        # Item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Graph Neural Network
        self.gnn = GNN(self.embedding_size, self.step)
        
        # Multi-layer Self-Attention Network
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(self.embedding_size, self.inner_size, self.dropout_prob)
            for _ in range(self.n_layers)
        ])
        
        # Loss functions
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        if self.reg_weight > 0:
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
        
        Args:
            item_seq: Item sequence tensor [batch_size, max_seq_len]
            
        Returns:
            alias_inputs: Mapping from sequence position to unique node index
            A: Adjacency matrix with incoming and outgoing edges
            items: Unique items in each session
            mask: Padding mask
        """
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
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            # Normalize adjacency matrix
            # Incoming edges (column normalization)
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            
            # Outgoing edges (row normalization)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            
            # Concatenate incoming and outgoing adjacency
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            # Create alias mapping from sequence position to node index
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        # Convert to tensors
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def _create_attention_mask(self, item_seq):
        """Create attention mask for padding positions.
        
        Args:
            item_seq: Item sequence [batch_size, seq_len]
            
        Returns:
            Attention mask [batch_size, 1, 1, seq_len] for broadcasting
        """
        # Create mask where padding positions (item_seq == 0) are True
        padding_mask = (item_seq == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
        # Convert to attention mask (0 for valid, -10000 for padding)
        attention_mask = padding_mask.float() * -10000.0
        return attention_mask

    def forward(self, item_seq, item_seq_len):
        """Forward pass of GCSAN.
        
        Args:
            item_seq: Item sequence [batch_size, max_seq_len]
            item_seq_len: Sequence lengths [batch_size]
            
        Returns:
            Session representation [batch_size, embedding_size]
        """
        # Construct session graph
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        
        # Get item embeddings for unique items in graph
        hidden = self.item_embedding(items)
        
        # Apply GNN to get node representations
        hidden = self.gnn(A, hidden)
        
        # Map node representations back to sequence positions
        alias_inputs_expanded = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs_expanded)
        
        # Get the last valid position's GNN output (local interest h_n)
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        
        # Apply multi-layer self-attention
        # Create attention mask for padding (simple 2D mask for our custom attention)
        attention_mask = self._create_attention_mask(item_seq)
        
        sa_output = seq_hidden
        for layer in self.self_attention_layers:
            sa_output = layer(sa_output, attention_mask)
        
        # Get the last position's self-attention output (global preference E_n^k)
        at = self.gather_indexes(sa_output, item_seq_len - 1)
        
        # Combine global preference and local interest
        # S_f = omega * E_n^(k) + (1 - omega) * h_n
        seq_output = self.weight * at + (1 - self.weight) * ht
        
        return seq_output

    def calculate_loss(self, interaction):
        """Calculate training loss.
        
        Args:
            interaction: Interaction data containing item sequences and targets
            
        Returns:
            Loss value
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
        
        # Add regularization loss if specified
        if self.reg_weight > 0:
            reg_loss = self.reg_loss(self.item_embedding.weight)
            loss = loss + self.reg_weight * reg_loss
        
        return loss

    def predict(self, interaction):
        """Predict scores for specific test items.
        
        Args:
            interaction: Interaction data
            
        Returns:
            Prediction scores [batch_size]
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
            interaction: Interaction data
            
        Returns:
            Prediction scores [batch_size, n_items]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores