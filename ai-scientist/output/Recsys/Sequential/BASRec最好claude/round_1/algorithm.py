# -*- coding: utf-8 -*-
# @Time    : 2025
# @Author  : Implementation based on BASRec paper
# @Email   : 

"""
BASRec
################################################

Reference:
    BASRec: Balanced data Augmentation method for Sequential Recommendation (2025)

This implementation includes:
- Single-sequence Augmentation (M-Reorder and M-Substitute operators)
- Cross-sequence Augmentation (Item-wise and Feature-wise Nonlinear Mixup)
- Adaptive Loss Weighting
- Two-stage Training Strategy
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class BASRec(SequentialRecommender):
    """
    BASRec: Balanced data Augmentation method for Sequential Recommendation.
    
    The model consists of two augmentation modules:
    1. Single-sequence Augmentation: M-Reorder and M-Substitute operators with mixup
    2. Cross-sequence Augmentation: Item-wise and Feature-wise Nonlinear Mixup
    """

    def __init__(self, config, dataset):
        super(BASRec, self).__init__(config, dataset)

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
        self.loss_type = config["loss_type"]

        # BASRec specific parameters
        self.alpha = config.get("alpha", 0.4)  # Beta distribution parameter
        self.rate_min = config.get("rate_min", 0.1)  # Minimum augmentation rate
        self.rate_max = config.get("rate_max", 0.5)  # Maximum augmentation rate
        self.stage1_epochs = config.get("stage1_epochs", 50)  # Warm-up epochs
        self.mixup_strategy = config.get("mixup_strategy", "both")  # item_wise/feature_wise/both
        self.enable_ssa = config.get("enable_ssa", True)  # Enable Single-sequence Augmentation
        self.enable_csa = config.get("enable_csa", True)  # Enable Cross-sequence Augmentation
        self.similarity_top_k = config.get("similarity_top_k", 10)  # Top-k similar items for substitution

        # Define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Training stage tracking
        self.current_epoch = 0
        
        # Cache for similar items (computed lazily)
        self.similar_items_cache = None

        # Parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_similar_items(self):
        """Compute top-k similar items for each item based on cosine similarity."""
        with torch.no_grad():
            item_emb = self.item_embedding.weight  # [n_items, hidden_size]
            # Normalize embeddings
            item_emb_norm = F.normalize(item_emb, p=2, dim=1)
            # Compute cosine similarity matrix
            similarity = torch.mm(item_emb_norm, item_emb_norm.t())  # [n_items, n_items]
            # Set self-similarity to -inf to exclude self
            similarity.fill_diagonal_(-float('inf'))
            # Get top-k similar items for each item
            _, top_k_indices = similarity.topk(self.similarity_top_k, dim=1)
            return top_k_indices  # [n_items, top_k]

    def _sample_rate(self):
        """Sample augmentation rate from uniform distribution U(rate_min, rate_max)."""
        return np.random.uniform(self.rate_min, self.rate_max)

    def _sample_mixup_lambda(self):
        """Sample mixup coefficient from Beta(alpha, alpha) distribution."""
        return np.random.beta(self.alpha, self.alpha)

    def _m_reorder(self, item_seq, item_seq_len):
        """
        M-Reorder operator: randomly shuffle a sub-sequence.
        
        Args:
            item_seq: [B, L] original item sequences
            item_seq_len: [B] sequence lengths
            
        Returns:
            reordered_seq: [B, L] reordered sequences
            rate: augmentation rate used
        """
        batch_size, max_len = item_seq.size()
        reordered_seq = item_seq.clone()
        rate = self._sample_rate()
        
        for b in range(batch_size):
            seq_len = item_seq_len[b].item()
            if seq_len < 2:
                continue
                
            # Calculate sub-sequence length
            c = max(1, int(rate * seq_len))
            
            # Random start position
            if seq_len - c > 0:
                start_idx = np.random.randint(0, seq_len - c + 1)
            else:
                start_idx = 0
            
            # Shuffle the sub-sequence
            sub_seq = reordered_seq[b, start_idx:start_idx + c].clone()
            perm = torch.randperm(c, device=item_seq.device)
            reordered_seq[b, start_idx:start_idx + c] = sub_seq[perm]
        
        return reordered_seq, rate

    def _m_substitute(self, item_seq, item_seq_len):
        """
        M-Substitute operator: replace selected items with similar items.
        
        Args:
            item_seq: [B, L] original item sequences
            item_seq_len: [B] sequence lengths
            
        Returns:
            substituted_seq: [B, L] substituted sequences
            rate: augmentation rate used
        """
        batch_size, max_len = item_seq.size()
        substituted_seq = item_seq.clone()
        rate = self._sample_rate()
        
        # Compute similar items cache if not available
        if self.similar_items_cache is None:
            self.similar_items_cache = self._compute_similar_items()
        
        similar_items = self.similar_items_cache  # [n_items, top_k]
        
        for b in range(batch_size):
            seq_len = item_seq_len[b].item()
            if seq_len < 1:
                continue
            
            # Number of items to substitute
            c = max(1, int(rate * seq_len))
            
            # Randomly select indices to substitute
            indices = np.random.choice(seq_len, size=min(c, seq_len), replace=False)
            
            for idx in indices:
                original_item = item_seq[b, idx].item()
                if original_item > 0 and original_item < self.n_items:
                    # Select a random similar item
                    similar_idx = np.random.randint(0, self.similarity_top_k)
                    substitute_item = similar_items[original_item, similar_idx].item()
                    substituted_seq[b, idx] = substitute_item
        
        return substituted_seq, rate

    def _compute_adaptive_weight(self, rate, mixup_lambda, batch_size):
        """
        Compute adaptive loss weight based on rate and mixup coefficient.
        
        Args:
            rate: augmentation rate
            mixup_lambda: mixup coefficient
            batch_size: batch size for weight tensor
            
        Returns:
            omega: normalized weight in [0, 1]
        """
        # omega^(1) = 1 / (rate * lambda)
        omega_1 = 1.0 / (rate * mixup_lambda + 1e-8)
        
        # Compute min and max for normalization
        omega_min = 1.0 / (self.rate_max * 1.0 + 1e-8)  # when rate=max, lambda=1
        omega_max = 1.0 / (self.rate_min * 1e-8 + 1e-8)  # when rate=min, lambda->0
        
        # Clamp omega_max to reasonable value
        omega_max = min(omega_max, 1.0 / (self.rate_min * 0.01 + 1e-8))
        
        # Normalize to [0, 1]
        omega = (omega_1 - omega_min) / (omega_max - omega_min + 1e-8)
        omega = np.clip(omega, 0.0, 1.0)
        
        return torch.tensor(omega, device=self.device, dtype=torch.float32)

    def _single_sequence_augmentation(self, item_seq, item_seq_len, operator='reorder'):
        """
        Single-sequence Augmentation with M-Reorder or M-Substitute.
        
        Args:
            item_seq: [B, L] original item sequences
            item_seq_len: [B] sequence lengths
            operator: 'reorder' or 'substitute'
            
        Returns:
            E_u_In: [B, L, D] augmented input embeddings
            omega: adaptive loss weight
        """
        # Apply operator to get augmented sequence
        if operator == 'reorder':
            aug_seq, rate = self._m_reorder(item_seq, item_seq_len)
        else:  # substitute
            aug_seq, rate = self._m_substitute(item_seq, item_seq_len)
        
        # Sample mixup coefficient
        mixup_lambda = self._sample_mixup_lambda()
        
        # Get embeddings
        E_u = self.item_embedding(item_seq)  # [B, L, D]
        E_u_prime = self.item_embedding(aug_seq)  # [B, L, D]
        
        # Mixup in embedding space (Eq. 7)
        E_u_In = mixup_lambda * E_u + (1 - mixup_lambda) * E_u_prime
        
        # Compute adaptive weight (Eq. 9)
        omega = self._compute_adaptive_weight(rate, mixup_lambda, item_seq.size(0))
        
        return E_u_In, omega

    def _cross_sequence_augmentation(self, H_u, strategy='both'):
        """
        Cross-sequence Augmentation with Item-wise and/or Feature-wise Nonlinear Mixup.
        
        Args:
            H_u: [B, L, D] sequence representations
            strategy: 'item_wise', 'feature_wise', or 'both'
            
        Returns:
            H_u_Out: [B, L, D] mixed representations
        """
        batch_size, seq_len, hidden_dim = H_u.size()
        
        # Shuffle batch to get H_u_prime (Eq. 10)
        perm = torch.randperm(batch_size, device=H_u.device)
        H_u_prime = H_u[perm]
        
        if strategy == 'item_wise':
            # Item-wise Nonlinear Mixup (Eq. 11)
            # Lambda_I: [B, L] -> broadcast to [B, L, 1]
            Lambda_I = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len))
            ).float().to(H_u.device).unsqueeze(-1)
            H_u_Out = Lambda_I * H_u + (1 - Lambda_I) * H_u_prime
            
        elif strategy == 'feature_wise':
            # Feature-wise Nonlinear Mixup (Eq. 12)
            # Lambda_F: [B, D] -> broadcast to [B, 1, D]
            Lambda_F = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_dim))
            ).float().to(H_u.device).unsqueeze(1)
            H_u_Out = Lambda_F * H_u + (1 - Lambda_F) * H_u_prime
            
        else:  # both
            # Apply both item-wise and feature-wise mixup
            Lambda_I = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len))
            ).float().to(H_u.device).unsqueeze(-1)
            H_u_intermediate = Lambda_I * H_u + (1 - Lambda_I) * H_u_prime
            
            # Shuffle again for second mixup
            perm2 = torch.randperm(batch_size, device=H_u.device)
            H_u_prime2 = H_u_intermediate[perm2]
            
            Lambda_F = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_dim))
            ).float().to(H_u.device).unsqueeze(1)
            H_u_Out = Lambda_F * H_u_intermediate + (1 - Lambda_F) * H_u_prime2
        
        return H_u_Out, perm

    def forward(self, item_seq, item_seq_len, input_emb=None):
        """
        Forward pass through the encoder.
        
        Args:
            item_seq: [B, L] item sequences
            item_seq_len: [B] sequence lengths
            input_emb: [B, L, D] optional pre-computed input embeddings (for augmentation)
            
        Returns:
            output: [B, D] sequence representations
        """
        # Position embeddings
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Item embeddings
        if input_emb is None:
            item_emb = self.item_embedding(item_seq)
        else:
            item_emb = input_emb
        
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)

        # Transformer encoding
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        
        return output  # [B, D]

    def forward_with_hidden(self, item_seq, item_seq_len, input_emb=None):
        """
        Forward pass returning both final output and full hidden states.
        
        Args:
            item_seq: [B, L] item sequences
            item_seq_len: [B] sequence lengths
            input_emb: [B, L, D] optional pre-computed input embeddings
            
        Returns:
            output: [B, D] sequence representations
            hidden_states: [B, L, D] full hidden states
        """
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        if input_emb is None:
            item_emb = self.item_embedding(item_seq)
        else:
            item_emb = input_emb
        
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        hidden_states = trm_output[-1]  # [B, L, D]
        output = self.gather_indexes(hidden_states, item_seq_len - 1)
        
        return output, hidden_states

    def _compute_bce_loss(self, seq_output, pos_items, neg_items=None):
        """
        Compute BCE loss for sequence output.
        
        Args:
            seq_output: [B, D] sequence representations
            pos_items: [B] positive item ids
            neg_items: [B] negative item ids (optional, for BPR)
            
        Returns:
            loss: scalar loss value
        """
        if self.loss_type == "BPR":
            if neg_items is None:
                raise ValueError("BPR loss requires negative items")
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def calculate_loss(self, interaction):
        """
        Calculate training loss with optional augmentation.
        
        Implements two-stage training:
        - Stage 1: Standard training without augmentation
        - Stage 2: Training with Single-sequence and Cross-sequence Augmentation
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Handle negative items based on loss type
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
        else:
            neg_items = None

        # Main loss (Eq. 13)
        seq_output = self.forward(item_seq, item_seq_len)
        loss_main = self._compute_bce_loss(seq_output, pos_items, neg_items)

        # Check if we're in stage 2 (augmentation enabled)
        if self.current_epoch < self.stage1_epochs:
            # Stage 1: Only main loss
            return loss_main

        total_loss = loss_main

        # Single-sequence Augmentation (Eq. 14)
        if self.enable_ssa:
            # M-Reorder augmentation
            E_u_In_reorder, omega_reorder = self._single_sequence_augmentation(
                item_seq, item_seq_len, operator='reorder'
            )
            H_u_In_reorder = self.forward(item_seq, item_seq_len, input_emb=E_u_In_reorder)
            loss_ssa_reorder = self._compute_bce_loss(H_u_In_reorder, pos_items, neg_items)
            loss_ssa_reorder = omega_reorder * loss_ssa_reorder

            # M-Substitute augmentation
            E_u_In_sub, omega_sub = self._single_sequence_augmentation(
                item_seq, item_seq_len, operator='substitute'
            )
            H_u_In_sub = self.forward(item_seq, item_seq_len, input_emb=E_u_In_sub)
            loss_ssa_sub = self._compute_bce_loss(H_u_In_sub, pos_items, neg_items)
            loss_ssa_sub = omega_sub * loss_ssa_sub

            total_loss = total_loss + loss_ssa_reorder + loss_ssa_sub

        # Cross-sequence Augmentation (Eq. 15)
        if self.enable_csa:
            # Get hidden states for cross-sequence mixup
            _, H_u = self.forward_with_hidden(item_seq, item_seq_len)
            
            # Apply cross-sequence mixup
            H_u_Out, perm = self._cross_sequence_augmentation(H_u, strategy=self.mixup_strategy)
            
            # Get final output from mixed hidden states
            seq_output_csa = self.gather_indexes(H_u_Out, item_seq_len - 1)
            
            # Mix positive and negative items accordingly
            pos_items_mixed = pos_items[perm]
            neg_items_mixed = neg_items[perm] if neg_items is not None else None
            
            # For cross-sequence, we use the mixed items as targets
            # Following the paper: same mixup for positive/negative items
            loss_csa = self._compute_bce_loss(seq_output_csa, pos_items_mixed, neg_items_mixed)
            
            total_loss = total_loss + loss_csa

        return total_loss

    def predict(self, interaction):
        """Predict scores for given item."""
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

    def set_epoch(self, epoch):
        """Set current epoch for two-stage training control."""
        self.current_epoch = epoch
        # Invalidate similar items cache periodically to reflect updated embeddings
        if epoch > 0 and epoch % 10 == 0:
            self.similar_items_cache = None