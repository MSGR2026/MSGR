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
    r"""
    SINE: Sparse-Interest Network for Sequential Recommendation.
    
    This model captures a user's diverse intentions by adaptively activating a subset of 
    conceptual prototypes from a large concept pool. It uses:
    1. Concept activation to select top-K prototypes for each user
    2. Intention assignment to estimate user intention related to each item
    3. Attention weighting to estimate item importance for next-item prediction
    4. Interest aggregation to combine multiple interest embeddings
    """

    def __init__(self, config, dataset):
        super(SINE, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        
        # SINE specific parameters
        self.prototype_size = config["prototype_size"]  # L: total number of concepts
        self.interest_size = config["interest_size"]  # K: number of activated concepts per user
        self.tau = config["tau"]  # temperature parameter for interest aggregation
        self.reg_weight = config["reg_weight"]  # lambda: trade-off parameter for covariance loss
        
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        self.initializer_range = config.get("initializer_range", 0.02)

        # Define layers
        # Item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        
        # Position embedding for attention weighting (Equation 4)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        
        # Conceptual prototype matrix C (L x D)
        self.prototype_embedding = nn.Embedding(self.prototype_size, self.embedding_size)
        
        # Concept activation parameters (Equation 1)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, 1, bias=False)
        
        # Intention assignment parameters (Equation 3)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        # Attention weighting parameters for each interest (Equation 4)
        # Using shared parameters with interest-specific transformations
        self.w_k_1 = nn.Linear(self.embedding_size, self.embedding_size * self.interest_size, bias=False)
        self.w_k_2 = nn.Linear(self.embedding_size, self.interest_size, bias=False)
        
        # Interest embedding generation layer norm (Equation 5)
        self.ln3 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        # Interest aggregation parameters (Equation 6)
        self.w_apt_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w_apt_2 = nn.Linear(self.embedding_size, 1, bias=False)
        self.ln4 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
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
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def concept_activation(self, item_emb, mask):
        """
        Concept activation module (Section 3.2.1).
        
        Args:
            item_emb: [B, n, D] - item embeddings
            mask: [B, n] - sequence mask (1 for valid, 0 for padding)
            
        Returns:
            C_u: [B, K, D] - activated prototype embeddings
            idx: [B, K] - indices of activated prototypes
            s_u_selected: [B, K] - activation scores for selected prototypes
        """
        # Self-attention to get virtual concept vector z_u (Equation 1)
        # a = softmax(tanh(X^u W_1) W_2)
        hidden = torch.tanh(self.w1(item_emb))  # [B, n, D]
        attn_scores = self.w2(hidden).squeeze(-1)  # [B, n]
        
        # Apply mask to attention scores
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, n]
        
        # z_u = (a^T X^u)^T
        z_u = torch.bmm(attn_weights.unsqueeze(1), item_emb).squeeze(1)  # [B, D]
        
        # Compute activation scores s^u = <C, z_u> (Equation 2)
        C = self.prototype_embedding.weight  # [L, D]
        s_u = torch.matmul(z_u, C.t())  # [B, L]
        
        # Top-K selection: idx = rank(s^u, K)
        _, idx = torch.topk(s_u, self.interest_size, dim=-1)  # [B, K]
        
        # Get selected prototype embeddings and scores
        # C(idx, :) - row extraction
        C_selected = self.prototype_embedding(idx)  # [B, K, D]
        
        # s^u(idx, :) - extract values
        s_u_selected = torch.gather(s_u, 1, idx)  # [B, K]
        
        # C^u = C(idx, :) ⊙ Sigmoid(s^u(idx, :))^T
        gate = torch.sigmoid(s_u_selected).unsqueeze(-1)  # [B, K, 1]
        C_u = C_selected * gate  # [B, K, D]
        
        return C_u, idx, s_u_selected

    def intention_assignment(self, item_emb, C_u):
        """
        Intention assignment module (Section 3.2.2).
        
        Args:
            item_emb: [B, n, D] - item embeddings
            C_u: [B, K, D] - activated prototype embeddings
            
        Returns:
            P_k_t: [B, n, K] - intention assignment probabilities
        """
        # LayerNorm_1(X_t^u W_3)
        item_proj = self.ln1(self.w3(item_emb))  # [B, n, D]
        
        # LayerNorm_2(C_k^u)
        proto_norm = self.ln2(C_u)  # [B, K, D]
        
        # Compute cosine similarity (due to normalization)
        # [B, n, D] @ [B, D, K] -> [B, n, K]
        logits = torch.bmm(item_proj, proto_norm.transpose(1, 2))  # [B, n, K]
        
        # Softmax over K prototypes
        P_k_t = torch.softmax(logits, dim=-1)  # [B, n, K]
        
        return P_k_t

    def attention_weighting(self, item_emb, mask):
        """
        Attention weighting module (Section 3.2.3).
        
        Args:
            item_emb: [B, n, D] - item embeddings (with position embeddings added)
            mask: [B, n] - sequence mask
            
        Returns:
            P_t_k: [B, K, n] - attention weights for each interest
        """
        batch_size, seq_len, _ = item_emb.shape
        
        # Add positional embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(position_ids)  # [B, n, D]
        item_emb_pos = item_emb + pos_emb  # [B, n, D]
        
        # Compute attention for each interest k
        # a^k = softmax(tanh(X^u W_{k,1}) W_{k,2})
        hidden = torch.tanh(self.w_k_1(item_emb_pos))  # [B, n, D*K]
        hidden = hidden.view(batch_size, seq_len, self.interest_size, self.embedding_size)  # [B, n, K, D]
        
        attn_scores = self.w_k_2(item_emb_pos)  # [B, n, K]
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        P_t_k = torch.softmax(attn_scores, dim=1)  # [B, n, K]
        P_t_k = P_t_k.transpose(1, 2)  # [B, K, n]
        
        return P_t_k

    def interest_embedding_generation(self, item_emb, P_k_t, P_t_k):
        """
        Interest embedding generation (Section 3.2.4, Equation 5).
        
        Args:
            item_emb: [B, n, D] - item embeddings
            P_k_t: [B, n, K] - intention assignment probabilities
            P_t_k: [B, K, n] - attention weights
            
        Returns:
            phi: [B, K, D] - multiple interest embeddings
        """
        # P_k_t: [B, n, K], P_t_k: [B, K, n]
        # Combined weight: P_{k|t} * P_{t|k}
        # For each k: sum_t P_{k|t} * P_{t|k} * X_t^u
        
        # P_t_k.transpose(1, 2): [B, n, K]
        # Element-wise product: [B, n, K]
        combined_weights = P_k_t * P_t_k.transpose(1, 2)  # [B, n, K]
        
        # Weighted sum: [B, K, n] @ [B, n, D] -> [B, K, D]
        phi = torch.bmm(combined_weights.transpose(1, 2), item_emb)  # [B, K, D]
        
        # Apply LayerNorm_3
        phi = self.ln3(phi)  # [B, K, D]
        
        return phi

    def interest_aggregation(self, phi, P_k_t, C_u):
        """
        Interest aggregation module (Section 3.3).
        
        Args:
            phi: [B, K, D] - multiple interest embeddings
            P_k_t: [B, n, K] - intention assignment probabilities
            C_u: [B, K, D] - activated prototype embeddings
            
        Returns:
            v_u: [B, D] - aggregated user representation
        """
        # Compute intention sequence: X_hat^u = P^u C^u (Equation 6)
        # P^u: [B, n, K], C^u: [B, K, D]
        X_hat = torch.bmm(P_k_t, C_u)  # [B, n, D]
        
        # Predict next intention C_apt^u
        # softmax(tanh(X_hat^u W_3) W_4)^T X_hat^u
        hidden = torch.tanh(self.w_apt_1(X_hat))  # [B, n, D]
        attn_scores = self.w_apt_2(hidden).squeeze(-1)  # [B, n]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, n]
        
        C_apt = torch.bmm(attn_weights.unsqueeze(1), X_hat).squeeze(1)  # [B, D]
        C_apt = self.ln4(C_apt)  # [B, D]
        
        # Compute aggregation weights (Equation 7)
        # e_k^u = softmax(C_apt^T phi_k / tau)
        # [B, 1, D] @ [B, D, K] -> [B, 1, K] -> [B, K]
        logits = torch.bmm(C_apt.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)  # [B, K]
        logits = logits / self.tau
        e_u = torch.softmax(logits, dim=-1)  # [B, K]
        
        # Aggregate interests (Equation 8)
        # v^u = sum_k e_k^u * phi_k
        v_u = torch.bmm(e_u.unsqueeze(1), phi).squeeze(1)  # [B, D]
        
        return v_u

    def calculate_reg_loss(self):
        """
        Calculate covariance regularization loss (Equation 10).
        
        Returns:
            L_c: scalar - covariance regularization loss
        """
        C = self.prototype_embedding.weight  # [L, D]
        
        # Compute mean
        C_mean = C.mean(dim=0, keepdim=True)  # [1, D]
        
        # Centered prototypes
        C_centered = C - C_mean  # [L, D]
        
        # Covariance matrix M = 1/D * (C - C_bar)(C - C_bar)^T
        M = torch.mm(C_centered, C_centered.t()) / self.embedding_size  # [L, L]
        
        # L_c = 0.5 * (||M||_F^2 - ||diag(M)||_F^2)
        # This penalizes off-diagonal elements, encouraging orthogonality
        L_c = 0.5 * (torch.norm(M, p='fro') ** 2 - torch.norm(torch.diag(M)) ** 2)
        
        return L_c

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of SINE model.
        
        Args:
            item_seq: [B, n] - item sequence
            item_seq_len: [B] - sequence lengths
            
        Returns:
            seq_output: [B, D] - user representation
        """
        # Get item embeddings
        item_emb = self.item_embedding(item_seq)  # [B, n, D]
        
        # Create mask (1 for valid positions, 0 for padding)
        mask = (item_seq != 0).float()  # [B, n]
        
        # 1. Concept activation
        C_u, idx, s_u_selected = self.concept_activation(item_emb, mask)  # C_u: [B, K, D]
        
        # 2. Intention assignment
        P_k_t = self.intention_assignment(item_emb, C_u)  # [B, n, K]
        
        # 3. Attention weighting
        P_t_k = self.attention_weighting(item_emb, mask)  # [B, K, n]
        
        # 4. Interest embedding generation
        phi = self.interest_embedding_generation(item_emb, P_k_t, P_t_k)  # [B, K, D]
        
        # 5. Interest aggregation
        v_u = self.interest_aggregation(phi, P_k_t, C_u)  # [B, D]
        
        return v_u

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