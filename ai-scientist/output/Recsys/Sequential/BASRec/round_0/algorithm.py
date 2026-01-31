# -*- coding: utf-8 -*-
# @Time    : 2025
# @Author  : Algorithm Engineer
# @Email   : algorithm@example.com

"""
BASRec
################################################

Reference:
    BASRec: Balanced data Augmentation method for Sequential Recommendation (2025)

This implementation includes:
- Single-sequence Augmentation (M-Reorder, M-Substitute)
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
    BASRec: Balanced data Augmentation method for Sequential Recommendation
    
    This model implements a comprehensive data augmentation framework with:
    1. Single-sequence Augmentation: M-Reorder and M-Substitute operators
    2. Cross-sequence Augmentation: Item-wise and Feature-wise Nonlinear Mixup
    3. Adaptive Loss Weighting based on augmentation intensity
    4. Two-stage training strategy
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
        self.rate_min = config.get("rate_min", 0.1)  # Minimum augmentation rate (a)
        self.rate_max = config.get("rate_max", 0.5)  # Maximum augmentation rate (b)
        self.stage1_epochs = config.get("stage1_epochs", 50)  # Warm-up epochs
        self.mixup_strategy = config.get("mixup_strategy", "both")  # item_wise/feature_wise/both
        self.substitute_topk = config.get("substitute_topk", 10)  # Top-k similar items for substitution
        
        # Current epoch tracker (will be updated during training)
        self.current_epoch = 0

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

        # Similarity matrix for M-Substitute (computed lazily)
        self.similarity_matrix = None
        self.similar_items = None

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
        """Compute top-k similar items for M-Substitute operation"""
        with torch.no_grad():
            item_emb = self.item_embedding.weight  # [n_items, hidden_size]
            # Normalize embeddings for cosine similarity
            item_emb_norm = F.normalize(item_emb, p=2, dim=-1)
            # Compute cosine similarity matrix
            similarity = torch.mm(item_emb_norm, item_emb_norm.t())  # [n_items, n_items]
            # Set self-similarity to -inf to exclude self
            similarity.fill_diagonal_(-float('inf'))
            # Get top-k similar items for each item
            _, top_indices = torch.topk(similarity, self.substitute_topk, dim=-1)
            self.similar_items = top_indices  # [n_items, topk]

    def _sample_rate(self):
        """Sample augmentation rate from uniform distribution U(a, b)"""
        return np.random.uniform(self.rate_min, self.rate_max)

    def _sample_mixup_lambda(self):
        """Sample mixup coefficient from Beta(alpha, alpha)"""
        return np.random.beta(self.alpha, self.alpha)

    def _m_reorder(self, item_seq, item_seq_len):
        """
        M-Reorder: Randomly shuffle a sub-sequence
        
        Args:
            item_seq: [B, N] item sequence
            item_seq_len: [B] sequence lengths
            
        Returns:
            reordered_seq: [B, N] reordered sequence
            rate: augmentation rate used
        """
        batch_size, max_len = item_seq.size()
        reordered_seq = item_seq.clone()
        rate = self._sample_rate()
        
        for b in range(batch_size):
            seq_len = item_seq_len[b].item()
            if seq_len <= 1:
                continue
                
            # Calculate sub-sequence length
            c = max(1, int(rate * seq_len))
            
            # Random start position
            if seq_len - c > 0:
                start_idx = np.random.randint(0, seq_len - c)
            else:
                start_idx = 0
            
            # Extract and shuffle sub-sequence
            sub_seq = reordered_seq[b, start_idx:start_idx + c].clone()
            perm = torch.randperm(c, device=item_seq.device)
            reordered_seq[b, start_idx:start_idx + c] = sub_seq[perm]
        
        return reordered_seq, rate

    def _m_substitute(self, item_seq, item_seq_len):
        """
        M-Substitute: Replace items with similar items
        
        Args:
            item_seq: [B, N] item sequence
            item_seq_len: [B] sequence lengths
            
        Returns:
            substituted_seq: [B, N] substituted sequence
            rate: augmentation rate used
        """
        # Ensure similar items are computed
        if self.similar_items is None:
            self._compute_similar_items()
        
        batch_size, max_len = item_seq.size()
        substituted_seq = item_seq.clone()
        rate = self._sample_rate()
        
        for b in range(batch_size):
            seq_len = item_seq_len[b].item()
            if seq_len <= 0:
                continue
            
            # Calculate number of items to substitute
            c = max(1, int(rate * seq_len))
            
            # Random select indices to substitute
            if seq_len > 0:
                indices = np.random.choice(seq_len, size=min(c, seq_len), replace=False)
                
                for idx in indices:
                    item_id = item_seq[b, idx].item()
                    if item_id > 0 and item_id < self.n_items:  # Valid item
                        # Randomly select from top-k similar items
                        similar_idx = np.random.randint(0, self.substitute_topk)
                        new_item = self.similar_items[item_id, similar_idx].item()
                        substituted_seq[b, idx] = new_item
        
        return substituted_seq, rate

    def _compute_adaptive_weight(self, rate, mixup_lambda, batch_size):
        """
        Compute adaptive loss weight based on augmentation intensity
        
        Args:
            rate: augmentation rate
            mixup_lambda: mixup coefficient
            batch_size: batch size for creating weight tensor
            
        Returns:
            omega: normalized weight in [0, 1]
        """
        # Compute raw weight: omega^(1) = 1 / (rate * lambda)
        omega_raw = 1.0 / (rate * mixup_lambda + 1e-8)
        
        # Compute min and max for normalization
        # Min: when rate=rate_max and lambda=1 (maximum augmentation)
        # Max: when rate=rate_min and lambda approaches 0 (minimum augmentation)
        omega_min = 1.0 / (self.rate_max * 1.0 + 1e-8)
        omega_max = 1.0 / (self.rate_min * 0.01 + 1e-8)  # Use small lambda for max
        
        # Normalize to [0, 1]
        omega = (omega_raw - omega_min) / (omega_max - omega_min + 1e-8)
        omega = np.clip(omega, 0.0, 1.0)
        
        return torch.tensor(omega, device=self.device).float()

    def _single_sequence_augmentation(self, item_seq, item_seq_len, operator='reorder'):
        """
        Single-sequence Augmentation with M-Reorder or M-Substitute
        
        Args:
            item_seq: [B, N] item sequence
            item_seq_len: [B] sequence lengths
            operator: 'reorder' or 'substitute'
            
        Returns:
            E_u_In: [B, N, D] augmented embedding
            omega: adaptive weight
        """
        # Get original embedding
        E_u = self.item_embedding(item_seq)  # [B, N, D]
        
        # Apply operator to get augmented sequence
        if operator == 'reorder':
            aug_seq, rate = self._m_reorder(item_seq, item_seq_len)
        else:  # substitute
            aug_seq, rate = self._m_substitute(item_seq, item_seq_len)
        
        # Get augmented embedding
        E_u_prime = self.item_embedding(aug_seq)  # [B, N, D]
        
        # Sample mixup coefficient
        mixup_lambda = self._sample_mixup_lambda()
        
        # Mixup in embedding space: E_u^In = lambda * E_u + (1-lambda) * E_u'
        E_u_In = mixup_lambda * E_u + (1 - mixup_lambda) * E_u_prime
        
        # Compute adaptive weight
        omega = self._compute_adaptive_weight(rate, mixup_lambda, item_seq.size(0))
        
        return E_u_In, omega

    def _cross_sequence_augmentation(self, H_u, strategy='item_wise'):
        """
        Cross-sequence Augmentation with Nonlinear Mixup
        
        Args:
            H_u: [B, N, D] sequence representations
            strategy: 'item_wise', 'feature_wise', or 'both'
            
        Returns:
            H_u_Out: [B, N, D] augmented representations
        """
        batch_size, seq_len, hidden_dim = H_u.size()
        
        # Shuffle batch to get H_u'
        perm = torch.randperm(batch_size, device=H_u.device)
        H_u_prime = H_u[perm]
        
        if strategy == 'item_wise':
            # Sample Lambda_I from Beta(alpha, alpha) with shape [B, N]
            Lambda_I = torch.tensor(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len)),
                device=H_u.device, dtype=H_u.dtype
            ).unsqueeze(-1)  # [B, N, 1]
            
            # Item-wise mixup: H_u^Out = Lambda_I * H_u + (1 - Lambda_I) * H_u'
            H_u_Out = Lambda_I * H_u + (1 - Lambda_I) * H_u_prime
            
        elif strategy == 'feature_wise':
            # Sample Lambda_F from Beta(alpha, alpha) with shape [B, D]
            Lambda_F = torch.tensor(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_dim)),
                device=H_u.device, dtype=H_u.dtype
            ).unsqueeze(1)  # [B, 1, D]
            
            # Feature-wise mixup: H_u^Out = Lambda_F * H_u + (1 - Lambda_F) * H_u'
            H_u_Out = Lambda_F * H_u + (1 - Lambda_F) * H_u_prime
            
        else:  # both
            # Apply both item-wise and feature-wise mixup
            Lambda_I = torch.tensor(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len)),
                device=H_u.device, dtype=H_u.dtype
            ).unsqueeze(-1)
            
            Lambda_F = torch.tensor(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_dim)),
                device=H_u.device, dtype=H_u.dtype
            ).unsqueeze(1)
            
            # First item-wise, then feature-wise
            H_u_temp = Lambda_I * H_u + (1 - Lambda_I) * H_u_prime
            
            # Shuffle again for feature-wise
            perm2 = torch.randperm(batch_size, device=H_u.device)
            H_u_prime2 = H_u_temp[perm2]
            
            H_u_Out = Lambda_F * H_u_temp + (1 - Lambda_F) * H_u_prime2
        
        return H_u_Out, perm

    def forward(self, item_seq, item_seq_len, input_emb=None):
        """
        Forward pass through the encoder
        
        Args:
            item_seq: [B, N] item sequence
            item_seq_len: [B] sequence lengths
            input_emb: [B, N, D] optional pre-computed input embedding (for augmentation)
            
        Returns:
            output: [B, D] sequence representation
        """
        # Position embedding
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Item embedding
        if input_emb is None:
            item_emb = self.item_embedding(item_seq)
        else:
            item_emb = input_emb
        
        # Combine embeddings
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)

        # Transformer encoder
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        
        return output  # [B, N, D]

    def _get_seq_output(self, item_seq, item_seq_len, input_emb=None):
        """Get the final sequence representation (last position)"""
        output = self.forward(item_seq, item_seq_len, input_emb)
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B, D]

    def calculate_loss(self, interaction):
        """
        Calculate training loss with two-stage strategy
        
        Stage 1: Standard training without augmentation
        Stage 2: Training with Single-sequence and Cross-sequence Augmentation
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Get original sequence output
        seq_output = self._get_seq_output(item_seq, item_seq_len)
        
        # Main loss (L_main)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss_main = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss_main = self.loss_fct(logits, pos_items)
        
        # Stage 1: Only main loss
        if self.current_epoch < self.stage1_epochs:
            return loss_main
        
        # Stage 2: Add augmentation losses
        loss_ssa = 0.0
        loss_csa = 0.0
        
        # ============ Single-sequence Augmentation (L_ssa) ============
        # M-Reorder augmentation
        E_u_In_reorder, omega_reorder = self._single_sequence_augmentation(
            item_seq, item_seq_len, operator='reorder'
        )
        H_u_In_reorder = self._get_seq_output(item_seq, item_seq_len, E_u_In_reorder)
        
        if self.loss_type == "BPR":
            pos_score_reorder = torch.sum(H_u_In_reorder * pos_items_emb, dim=-1)
            neg_score_reorder = torch.sum(H_u_In_reorder * neg_items_emb, dim=-1)
            loss_reorder = self.loss_fct(pos_score_reorder, neg_score_reorder)
        else:
            logits_reorder = torch.matmul(H_u_In_reorder, test_item_emb.transpose(0, 1))
            loss_reorder = self.loss_fct(logits_reorder, pos_items)
        
        loss_ssa += omega_reorder * loss_reorder
        
        # M-Substitute augmentation
        E_u_In_sub, omega_sub = self._single_sequence_augmentation(
            item_seq, item_seq_len, operator='substitute'
        )
        H_u_In_sub = self._get_seq_output(item_seq, item_seq_len, E_u_In_sub)
        
        if self.loss_type == "BPR":
            pos_score_sub = torch.sum(H_u_In_sub * pos_items_emb, dim=-1)
            neg_score_sub = torch.sum(H_u_In_sub * neg_items_emb, dim=-1)
            loss_sub = self.loss_fct(pos_score_sub, neg_score_sub)
        else:
            logits_sub = torch.matmul(H_u_In_sub, test_item_emb.transpose(0, 1))
            loss_sub = self.loss_fct(logits_sub, pos_items)
        
        loss_ssa += omega_sub * loss_sub
        
        # ============ Cross-sequence Augmentation (L_csa) ============
        # Get full sequence representations for cross-sequence mixup
        H_u = self.forward(item_seq, item_seq_len)  # [B, N, D]
        
        # Apply cross-sequence augmentation
        H_u_Out, perm = self._cross_sequence_augmentation(H_u, strategy=self.mixup_strategy)
        
        # Get the last position representation
        H_u_Out_final = self.gather_indexes(H_u_Out, item_seq_len - 1)  # [B, D]
        
        # Shuffle positive and negative items accordingly
        pos_items_shuffled = pos_items[perm]
        
        if self.loss_type == "BPR":
            neg_items_shuffled = neg_items[perm]
            
            # Mixup positive and negative embeddings
            mixup_lambda_items = self._sample_mixup_lambda()
            pos_items_emb_mixed = mixup_lambda_items * pos_items_emb + (1 - mixup_lambda_items) * self.item_embedding(pos_items_shuffled)
            neg_items_emb_mixed = mixup_lambda_items * neg_items_emb + (1 - mixup_lambda_items) * self.item_embedding(neg_items_shuffled)
            
            pos_score_csa = torch.sum(H_u_Out_final * pos_items_emb_mixed, dim=-1)
            neg_score_csa = torch.sum(H_u_Out_final * neg_items_emb_mixed, dim=-1)
            loss_csa = self.loss_fct(pos_score_csa, neg_score_csa)
        else:
            # For CE loss, compute logits and use soft labels
            logits_csa = torch.matmul(H_u_Out_final, test_item_emb.transpose(0, 1))
            
            # Use the original positive items as targets (simplified)
            loss_csa = self.loss_fct(logits_csa, pos_items)
        
        # Total loss: L = L_main + L_ssa + L_csa
        total_loss = loss_main + loss_ssa + loss_csa
        
        return total_loss

    def predict(self, interaction):
        """Predict score for a single candidate item"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self._get_seq_output(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self._get_seq_output(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores

    def set_epoch(self, epoch):
        """Set current epoch for two-stage training control"""
        self.current_epoch = epoch
        # Update similar items periodically for M-Substitute
        if epoch >= self.stage1_epochs and epoch % 10 == 0:
            self._compute_similar_items()