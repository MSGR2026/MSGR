# -*- coding: utf-8 -*-
# @Time    : 2025
# @Author  : Implementation based on FAME paper
# @Email   : 

"""
FAME
################################################

Reference:
    "FAME: Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation" (2025)

This implementation incorporates:
1. Facet-Aware Multi-Head Prediction Mechanism - each head independently generates recommendations
2. Mixture-of-Experts Self-Attention Layer - MoE network for capturing diverse preferences within each facet
"""

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class FAME(SequentialRecommender):
    """
    FAME: Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation.
    
    The framework incorporates two key components:
    1. Facet-Aware Multi-Head Prediction Mechanism: learns to represent each item with multiple 
       sub-embedding vectors, each capturing a specific facet of the item
    2. Mixture-of-Experts Self-Attention Layer: employs MoE network within each subspace to 
       capture users' specific preferences within each facet
    """

    def __init__(self, config, dataset):
        super(FAME, self).__init__(config, dataset)

        # Load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # d in the paper
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # FAME specific parameters
        self.n_experts = config.get("n_experts", 4)  # N in the paper - number of experts per head
        
        # Derived dimensions
        self.head_dim = self.hidden_size // self.n_heads  # d' in the paper

        # Define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # Base Transformer encoder (for layers before the final layer)
        if self.n_layers > 1:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers - 1,  # All layers except the final one
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
            )
        else:
            self.trm_encoder = None

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # ============ FAME Components ============
        
        # MoE Self-Attention Layer components (Section 4.3)
        # Query projection matrices for each expert in each head: W_Q(n)^(h)
        # Shape: [n_heads, n_experts, hidden_size, head_dim]
        self.expert_query_weights = nn.Parameter(
            torch.empty(self.n_heads, self.n_experts, self.hidden_size, self.head_dim)
        )
        
        # Key and Value projection matrices (shared across experts within each head)
        # W_K^(h) and W_V^(h)
        self.key_weights = nn.Parameter(
            torch.empty(self.n_heads, self.hidden_size, self.head_dim)
        )
        self.value_weights = nn.Parameter(
            torch.empty(self.n_heads, self.hidden_size, self.head_dim)
        )
        
        # Router network for expert importance (Equation 16): W_exp^(h)
        # Shape: [n_heads, n_experts * head_dim, n_experts]
        self.expert_router = nn.Parameter(
            torch.empty(self.n_heads, self.n_experts * self.head_dim, self.n_experts)
        )
        
        # Attention layer norm and dropout
        self.attn_layer_norm = nn.LayerNorm(self.head_dim, eps=self.layer_norm_eps)
        self.attn_dropout = nn.Dropout(self.attn_dropout_prob)
        
        # Facet-Aware Multi-Head Prediction components (Section 4.2)
        # Shared FFN' for all heads (Equation 7) - operates on head_dim
        self.ffn_linear1 = nn.Linear(self.head_dim, self.head_dim)
        self.ffn_linear2 = nn.Linear(self.head_dim, self.head_dim)
        self.ffn_layer_norm = nn.LayerNorm(self.head_dim, eps=self.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Head-specific item projection matrices (Equation 9): W_f^(h)
        # Shape: [n_heads, hidden_size, head_dim]
        self.item_head_projections = nn.Parameter(
            torch.empty(self.n_heads, self.hidden_size, self.head_dim)
        )
        
        # Gate mechanism (Equation 10): W_g and b_g
        self.gate_weight = nn.Linear(self.hidden_size, self.n_heads)
        
        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = nn.functional.softplus
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Parameters initialization
        self.apply(self._init_weights)
        self._init_fame_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_fame_weights(self):
        """Initialize FAME-specific weights"""
        nn.init.normal_(self.expert_query_weights, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.key_weights, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.value_weights, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.expert_router, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.item_head_projections, mean=0.0, std=self.initializer_range)

    def moe_self_attention(self, hidden_states, attention_mask):
        """
        Mixture-of-Experts Self-Attention Layer (Section 4.3)
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len]
            
        Returns:
            head_outputs: [batch_size, seq_len, n_heads, head_dim] - output for each head
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute keys and values for all heads
        # keys: [batch_size, seq_len, n_heads, head_dim]
        # values: [batch_size, seq_len, n_heads, head_dim]
        keys = torch.einsum('bsd,hdk->bshk', hidden_states, self.key_weights)
        values = torch.einsum('bsd,hdk->bshk', hidden_states, self.value_weights)
        
        # Compute queries for all experts in all heads
        # queries: [batch_size, seq_len, n_heads, n_experts, head_dim]
        queries = torch.einsum('bsd,hndk->bshnk', hidden_states, self.expert_query_weights)
        
        # Compute attention scores for each expert (Equation 14)
        # attention_scores: [batch_size, n_heads, n_experts, seq_len, seq_len]
        # queries: [batch_size, seq_len, n_heads, n_experts, head_dim] -> [batch_size, n_heads, n_experts, seq_len, head_dim]
        queries = queries.permute(0, 2, 3, 1, 4)
        # keys: [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        keys_t = keys.permute(0, 2, 1, 3)
        
        # [batch_size, n_heads, n_experts, seq_len, head_dim] @ [batch_size, n_heads, head_dim, seq_len]
        # -> [batch_size, n_heads, n_experts, seq_len, seq_len]
        attention_scores = torch.einsum('bhnsk,bhkj->bhnsj', queries, keys_t.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        
        # Apply attention mask (causal + padding)
        # attention_mask: [batch_size, 1, seq_len, seq_len] -> [batch_size, 1, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(2)
        attention_scores = attention_scores + attention_mask
        
        # Softmax over the last dimension (keys dimension)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Compute expert outputs (Equation 15)
        # values: [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        values_t = values.permute(0, 2, 1, 3)
        # [batch_size, n_heads, n_experts, seq_len, seq_len] @ [batch_size, n_heads, seq_len, head_dim]
        # -> [batch_size, n_heads, n_experts, seq_len, head_dim]
        expert_outputs = torch.einsum('bhnsj,bhjk->bhnsk', attention_probs, values_t)
        
        # Compute router weights (Equation 16)
        # Concatenate expert outputs: [batch_size, n_heads, seq_len, n_experts * head_dim]
        expert_concat = expert_outputs.permute(0, 1, 3, 2, 4).reshape(
            batch_size, self.n_heads, seq_len, self.n_experts * self.head_dim
        )
        
        # Router scores: [batch_size, n_heads, seq_len, n_experts]
        router_logits = torch.einsum('bhsd,hdn->bhsn', expert_concat, self.expert_router)
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Compute integrated item representation (Equation 17)
        # expert_outputs: [batch_size, n_heads, n_experts, seq_len, head_dim]
        # router_weights: [batch_size, n_heads, seq_len, n_experts]
        # -> [batch_size, n_heads, seq_len, head_dim]
        expert_outputs_t = expert_outputs.permute(0, 1, 3, 2, 4)  # [batch_size, n_heads, seq_len, n_experts, head_dim]
        integrated_output = torch.einsum('bhsnd,bhsn->bhsd', expert_outputs_t, router_weights)
        
        # Apply layer norm (residual connection with input projected to head_dim)
        # Project input to head dimension for residual
        input_projected = torch.einsum('bsd,hdk->bshk', hidden_states, self.key_weights)
        input_projected = input_projected.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, head_dim]
        
        integrated_output = self.attn_layer_norm(integrated_output + input_projected)
        
        # Output: [batch_size, seq_len, n_heads, head_dim]
        return integrated_output.permute(0, 2, 1, 3)

    def head_ffn(self, head_output):
        """
        Shared FFN' for all heads (Equation 7)
        
        Args:
            head_output: [batch_size, seq_len, n_heads, head_dim]
            
        Returns:
            output: [batch_size, seq_len, n_heads, head_dim]
        """
        # Apply FFN to each head independently
        # FFN'(x) = ReLU(x @ W1 + b1) @ W2 + b2
        ffn_output = self.ffn_linear1(head_output)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.ffn_linear2(ffn_output)
        
        # Layer norm with residual connection and dropout
        output = self.ffn_layer_norm(head_output + self.ffn_dropout(ffn_output))
        
        return output

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of FAME model
        
        Args:
            item_seq: [batch_size, seq_len]
            item_seq_len: [batch_size]
            
        Returns:
            seq_output: [batch_size, hidden_size] - sequence representation for prediction
        """
        # Position embeddings
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Item embeddings
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Generate attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)

        # Pass through base transformer layers (if any)
        if self.trm_encoder is not None:
            trm_output = self.trm_encoder(
                input_emb, extended_attention_mask, output_all_encoded_layers=True
            )
            hidden_states = trm_output[-1]
        else:
            hidden_states = input_emb

        # Apply MoE Self-Attention Layer (final layer)
        # Output: [batch_size, seq_len, n_heads, head_dim]
        moe_output = self.moe_self_attention(hidden_states, extended_attention_mask)
        
        # Apply shared FFN' to each head (Equation 7)
        # Output: [batch_size, seq_len, n_heads, head_dim]
        head_outputs = self.head_ffn(moe_output)
        
        # Gather the last valid position for each sequence
        # [batch_size, n_heads, head_dim]
        gather_index = (item_seq_len - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_heads, self.head_dim)
        final_head_outputs = head_outputs.gather(1, gather_index).squeeze(1)
        
        return final_head_outputs  # [batch_size, n_heads, head_dim]

    def compute_prediction_scores(self, head_outputs, item_embeddings):
        """
        Compute prediction scores using Facet-Aware Multi-Head Prediction (Section 4.2)
        
        Args:
            head_outputs: [batch_size, n_heads, head_dim] - F_t^(h) for each head
            item_embeddings: [n_items, hidden_size] or [batch_size, hidden_size]
            
        Returns:
            scores: [batch_size, n_items] or [batch_size]
        """
        batch_size = head_outputs.shape[0]
        
        # Compute gate weights (Equation 10)
        # Concatenate head outputs: [batch_size, hidden_size]
        concat_heads = head_outputs.reshape(batch_size, -1)
        gate_logits = self.gate_weight(concat_heads)  # [batch_size, n_heads]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch_size, n_heads]
        
        # Project item embeddings to each head's subspace (Equation 9)
        # item_head_projections: [n_heads, hidden_size, head_dim]
        # For full sort: item_embeddings: [n_items, hidden_size]
        # item_sub_embeddings: [n_items, n_heads, head_dim]
        if item_embeddings.dim() == 2 and item_embeddings.shape[0] == self.n_items:
            # Full sort prediction
            item_sub_embeddings = torch.einsum('id,hdk->ihk', item_embeddings, self.item_head_projections)
            
            # Compute scores for each head (Equation 8)
            # head_outputs: [batch_size, n_heads, head_dim]
            # item_sub_embeddings: [n_items, n_heads, head_dim]
            # head_scores: [batch_size, n_heads, n_items]
            head_scores = torch.einsum('bhd,ihd->bhi', head_outputs, item_sub_embeddings)
            
            # Weighted sum of head scores (Equation 11)
            # gate_weights: [batch_size, n_heads]
            # scores: [batch_size, n_items]
            scores = torch.einsum('bh,bhi->bi', gate_weights, head_scores)
        else:
            # Single item prediction (for predict method)
            # item_embeddings: [batch_size, hidden_size]
            item_sub_embeddings = torch.einsum('bd,hdk->bhk', item_embeddings, self.item_head_projections)
            
            # Compute scores for each head
            # head_outputs: [batch_size, n_heads, head_dim]
            # item_sub_embeddings: [batch_size, n_heads, head_dim]
            # head_scores: [batch_size, n_heads]
            head_scores = (head_outputs * item_sub_embeddings).sum(dim=-1)
            
            # Weighted sum of head scores
            scores = (gate_weights * head_scores).sum(dim=-1)
        
        return scores

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Get sequence representation
        head_outputs = self.forward(item_seq, item_seq_len)
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            
            pos_score = self.compute_prediction_scores(head_outputs, pos_items_emb)
            neg_score = self.compute_prediction_scores(head_outputs, neg_items_emb)
            
            loss = self.loss_fct(neg_score - pos_score).mean()
            return loss
        else:  # CE loss
            # Full softmax over all items
            all_item_emb = self.item_embedding.weight
            logits = self.compute_prediction_scores(head_outputs, all_item_emb)
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        head_outputs = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = self.compute_prediction_scores(head_outputs, test_item_emb)
        
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        head_outputs = self.forward(item_seq, item_seq_len)
        all_item_emb = self.item_embedding.weight
        scores = self.compute_prediction_scores(head_outputs, all_item_emb)
        
        return scores