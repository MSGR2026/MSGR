# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
SINE
################################################

Reference:
    Tan et al. "Sparse-Interest Network for Sequential Recommendation" in WSDM 2021.

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class SINE(SequentialRecommender):
    r"""SINE is a sparse-interest network for sequential recommendation.
    
    It adaptively activates a subset of concepts from a large concept pool for each user,
    and generates multiple interest embeddings through intention assignment and attention weighting.
    """

    def __init__(self, config, dataset):
        super(SINE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        
        # SINE specific parameters
        self.prototype_size = config["prototype_size"]  # L: total number of concepts
        self.interest_size = config["interest_size"]  # K: number of activated concepts per user
        self.tau = config["tau"]  # temperature for interest aggregation
        self.reg_weight = config["reg_weight"]  # lambda for covariance regularization
        
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)

        # define layers
        # Item embedding table
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Positional embedding for attention weighting
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        # Conceptual prototype matrix C (L x D)
        self.concept_embedding = nn.Embedding(self.prototype_size, self.embedding_size)
        
        # Parameters for concept activation (Equation 1)
        self.W1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W2 = nn.Linear(self.embedding_size, 1, bias=False)
        
        # Parameters for intention assignment (Equation 3)
        self.W3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        # Parameters for attention weighting (Equation 4)
        # K separate attention layers for each activated concept
        self.W_k_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_k_2 = nn.Linear(self.embedding_size, self.interest_size, bias=False)
        
        # LayerNorm for interest embedding generation (Equation 5)
        self.ln3 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        # Parameters for interest aggregation (Equation 6)
        self.W4 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W5 = nn.Linear(self.embedding_size, 1, bias=False)
        self.ln4 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "NLL":
            # NLL loss with log_softmax (as mentioned in paper for sampled softmax)
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE', 'NLL']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def concept_activation(self, item_seq_emb, mask):
        """
        Concept activation module (Section 3.2.1)
        
        Args:
            item_seq_emb: [B, n, D] item embeddings
            mask: [B, n] padding mask (1 for valid, 0 for padding)
            
        Returns:
            C_u: [B, K, D] activated concept embeddings
            idx: [B, K] indices of activated concepts
            s_u_topk: [B, K] activation scores for top-K concepts
        """
        # Equation 1: Self-attention to get virtual concept vector z_u
        # a = softmax(tanh(X^u * W1) * W2)
        hidden = torch.tanh(self.W1(item_seq_emb))  # [B, n, D]
        attn_scores = self.W2(hidden).squeeze(-1)  # [B, n]
        
        # Apply mask for padding
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        a = torch.softmax(attn_scores, dim=-1)  # [B, n]
        
        # z_u = a^T * X^u
        z_u = torch.bmm(a.unsqueeze(1), item_seq_emb).squeeze(1)  # [B, D]
        
        # Equation 2: Activate top-K concepts
        # s^u = <C, z_u>
        C = self.concept_embedding.weight  # [L, D]
        s_u = torch.matmul(z_u, C.t())  # [B, L]
        
        # idx = rank(s^u, K)
        _, idx = torch.topk(s_u, self.interest_size, dim=-1)  # [B, K]
        
        # C^u = C(idx, :) * Sigmoid(s^u(idx))
        s_u_topk = torch.gather(s_u, 1, idx)  # [B, K]
        C_selected = C[idx]  # [B, K, D]
        
        # Apply sigmoid gating
        gate = torch.sigmoid(s_u_topk).unsqueeze(-1)  # [B, K, 1]
        C_u = C_selected * gate  # [B, K, D]
        
        return C_u, idx, s_u_topk

    def intention_assignment(self, item_seq_emb, C_u):
        """
        Intention assignment module (Section 3.2.2)
        
        Args:
            item_seq_emb: [B, n, D] item embeddings
            C_u: [B, K, D] activated concept embeddings
            
        Returns:
            P_k_t: [B, n, K] intention assignment probabilities
        """
        # Equation 3: P_{k|t} using cosine similarity with LayerNorm
        # LayerNorm1(X_t^u * W3)
        item_proj = self.ln1(self.W3(item_seq_emb))  # [B, n, D]
        
        # LayerNorm2(C_k^u)
        concept_norm = self.ln2(C_u)  # [B, K, D]
        
        # Cosine similarity (dot product after normalization)
        # [B, n, D] x [B, D, K] -> [B, n, K]
        sim = torch.bmm(item_proj, concept_norm.transpose(1, 2))  # [B, n, K]
        
        # Softmax over concepts
        P_k_t = torch.softmax(sim, dim=-1)  # [B, n, K]
        
        return P_k_t

    def attention_weighting(self, item_seq_emb, mask):
        """
        Attention weighting module (Section 3.2.3)
        
        Args:
            item_seq_emb: [B, n, D] item embeddings with positional encoding
            mask: [B, n] padding mask
            
        Returns:
            P_t_k: [B, n, K] attention weights for each position under each concept
        """
        # Equation 4: a^k = softmax(tanh(X^u * W_{k,1}) * W_{k,2})
        # We use a shared layer to compute K attention heads
        hidden = torch.tanh(self.W_k_1(item_seq_emb))  # [B, n, D]
        attn_scores = self.W_k_2(hidden)  # [B, n, K]
        
        # Apply mask for padding
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        P_t_k = torch.softmax(attn_scores, dim=1)  # [B, n, K]
        
        return P_t_k

    def interest_embedding_generation(self, item_seq_emb, P_k_t, P_t_k):
        """
        Interest embedding generation (Section 3.2.4)
        
        Args:
            item_seq_emb: [B, n, D] item embeddings
            P_k_t: [B, n, K] intention assignment probabilities
            P_t_k: [B, n, K] attention weights
            
        Returns:
            interest_embs: [B, K, D] multiple interest embeddings
        """
        # Equation 5: phi_k = LayerNorm(sum_t P_{k|t} * P_{t|k} * X_t^u)
        # Combined weight: [B, n, K]
        combined_weight = P_k_t * P_t_k  # [B, n, K]
        
        # Weighted sum: [B, K, n] x [B, n, D] -> [B, K, D]
        interest_embs = torch.bmm(combined_weight.transpose(1, 2), item_seq_emb)  # [B, K, D]
        
        # Apply LayerNorm
        interest_embs = self.ln3(interest_embs)  # [B, K, D]
        
        return interest_embs

    def interest_aggregation(self, item_seq_emb, P_k_t, C_u, interest_embs, mask):
        """
        Interest aggregation module (Section 3.3)
        
        Args:
            item_seq_emb: [B, n, D] item embeddings
            P_k_t: [B, n, K] intention assignment probabilities
            C_u: [B, K, D] activated concept embeddings
            interest_embs: [B, K, D] multiple interest embeddings
            mask: [B, n] padding mask
            
        Returns:
            v_u: [B, D] final user representation
        """
        # Equation 6: Compute intention sequence X_hat^u = P^u * C^u
        X_hat_u = torch.bmm(P_k_t, C_u)  # [B, n, D]
        
        # Adaptive intention prediction: C_apt^u
        hidden = torch.tanh(self.W4(X_hat_u))  # [B, n, D]
        attn_scores = self.W5(hidden).squeeze(-1)  # [B, n]
        
        # Apply mask for padding
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        a_apt = torch.softmax(attn_scores, dim=-1)  # [B, n]
        
        # C_apt^u = LayerNorm(a_apt^T * X_hat^u)
        C_apt_u = torch.bmm(a_apt.unsqueeze(1), X_hat_u).squeeze(1)  # [B, D]
        C_apt_u = self.ln4(C_apt_u)  # [B, D]
        
        # Equation 7: Aggregation weights
        # e_k^u = softmax(C_apt^u^T * phi_k / tau)
        # [B, D] x [B, D, K] -> [B, K]
        scores = torch.bmm(C_apt_u.unsqueeze(1), interest_embs.transpose(1, 2)).squeeze(1)  # [B, K]
        e_u = torch.softmax(scores / self.tau, dim=-1)  # [B, K]
        
        # Equation 8: Final user representation
        # v^u = sum_k e_k^u * phi_k
        v_u = torch.bmm(e_u.unsqueeze(1), interest_embs).squeeze(1)  # [B, D]
        
        return v_u

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of SINE model.
        
        Args:
            item_seq: [B, n] item sequence
            item_seq_len: [B] sequence lengths
            
        Returns:
            seq_output: [B, D] final sequence representation
        """
        # Create padding mask
        mask = (item_seq != 0).float()  # [B, n]
        
        # Get item embeddings
        item_seq_emb = self.item_embedding(item_seq)  # [B, n, D]
        
        # Add positional embeddings for attention weighting
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_emb = self.position_embedding(position_ids)  # [B, n, D]
        item_seq_emb_pos = item_seq_emb + position_emb  # [B, n, D]
        
        # Step 1: Concept activation
        C_u, idx, s_u_topk = self.concept_activation(item_seq_emb, mask)  # [B, K, D]
        
        # Step 2: Intention assignment
        P_k_t = self.intention_assignment(item_seq_emb, C_u)  # [B, n, K]
        
        # Step 3: Attention weighting (with positional embeddings)
        P_t_k = self.attention_weighting(item_seq_emb_pos, mask)  # [B, n, K]
        
        # Step 4: Interest embedding generation
        interest_embs = self.interest_embedding_generation(item_seq_emb, P_k_t, P_t_k)  # [B, K, D]
        
        # Step 5: Interest aggregation
        seq_output = self.interest_aggregation(item_seq_emb, P_k_t, C_u, interest_embs, mask)  # [B, D]
        
        return seq_output

    def calculate_reg_loss(self):
        """
        Calculate covariance regularization loss (Equation 10)
        
        Returns:
            reg_loss: scalar regularization loss
        """
        # Get concept embeddings
        C = self.concept_embedding.weight  # [L, D]
        
        # Center the embeddings
        C_mean = C.mean(dim=0, keepdim=True)  # [1, D]
        C_centered = C - C_mean  # [L, D]
        
        # Compute covariance matrix M = (1/D) * (C - C_bar) * (C - C_bar)^T
        D = C.size(1)
        M = torch.mm(C_centered, C_centered.t()) / D  # [L, L]
        
        # L_c = 0.5 * (||M||_F^2 - ||diag(M)||_F^2)
        # This penalizes off-diagonal elements (correlation between concepts)
        M_frob_sq = (M ** 2).sum()
        M_diag_frob_sq = (torch.diag(M) ** 2).sum()
        reg_loss = 0.5 * (M_frob_sq - M_diag_frob_sq)
        
        return reg_loss

    def calculate_loss(self, interaction):
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
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        # Add covariance regularization loss
        reg_loss = self.calculate_reg_loss()
        loss = loss + self.reg_weight * reg_loss
        
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores