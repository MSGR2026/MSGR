# -*- coding: utf-8 -*-
# @Time   : 2026/1/8
# @Author : Yizhou Dang
# @Email  : yzdang@stumail.neu.edu.cn

"""
BASRec
################################################

Reference:
    Yizhou Dang et al. "Augmenting Sequential Recommendation with Balanced Relevance and Diversity." in AAAI 2025.

Reference Code:
    https://github.com/KingGugu/BASRec

"""

import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class BASRec(SequentialRecommender):
    r"""
    BASRec is a data augmentation plugin for sequential recommendation that generates
    data balancing relevance and diversity through representation-level fusion.
    
    It consists of two modules:
    - Single-sequence Augmentation: M-Reorder and M-Substitute with adaptive loss weighting
    - Cross-sequence Augmentation: Item-wise and Feature-wise nonlinear mixup
    
    The model uses a two-stage training strategy to ensure stable convergence.
    """

    def __init__(self, config, dataset):
        super(BASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # BASRec specific parameters
        self.alpha = config.get("alpha", 0.4)  # Beta distribution parameter for mixup
        self.rate_min = config.get("rate_min", 0.2)  # Minimum augmentation rate
        self.rate_max = config.get("rate_max", 0.7)  # Maximum augmentation rate
        self.tau = config.get("tau", 1.0)  # Temperature for cross-sequence augmentation
        self.two_stage = config.get("two_stage", True)  # Two-stage training
        self.stage1_epochs = config.get("stage1_epochs", 50)  # First stage epochs
        self.current_epoch = 0
        
        # Mixup strategy: 'item_wise', 'feature_wise', or 'both'
        self.mixup_strategy = config.get("mixup_strategy", "both")

        # define layers and loss
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

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Item similarity for substitute augmentation (computed online)
        self.item_similarity_computed = False

        # parameters initialization
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

    def _compute_item_similarity(self):
        """Compute item similarity matrix using cosine similarity on embeddings"""
        with torch.no_grad():
            item_embs = self.item_embedding.weight[1:]  # Exclude padding item
            # Normalize embeddings
            item_embs_norm = F.normalize(item_embs, p=2, dim=1)
            # Compute similarity matrix
            similarity_matrix = torch.mm(item_embs_norm, item_embs_norm.t())
            self.similarity_matrix = similarity_matrix
            self.item_similarity_computed = True

    def _get_similar_item(self, item_id):
        """Get most similar item for substitution"""
        if not self.item_similarity_computed:
            self._compute_item_similarity()
        
        if item_id == 0:  # Padding item
            return 0
        
        # Get similarity scores (item_id is 1-indexed, similarity_matrix is 0-indexed)
        similarities = self.similarity_matrix[item_id - 1]
        # Get top-k similar items (excluding the item itself)
        top_k = torch.topk(similarities, k=2)[1]
        # Return the most similar item (not itself)
        similar_item = top_k[1].item() + 1 if top_k[0].item() == item_id - 1 else top_k[0].item() + 1
        return similar_item

    def m_reorder(self, item_seq):
        """M-Reorder augmentation: randomly reorder a sub-sequence"""
        batch_size, seq_len = item_seq.shape
        augmented_seqs = []
        rates = []
        
        for i in range(batch_size):
            seq = item_seq[i].clone()
            # Find actual sequence length (excluding padding)
            actual_len = (seq != 0).sum().item()
            
            if actual_len <= 1:
                augmented_seqs.append(seq)
                rates.append(0.0)
                continue
            
            # Sample augmentation rate
            rate = np.random.uniform(self.rate_min, self.rate_max)
            sub_len = max(int(rate * actual_len), 1)
            
            # Randomly select start position
            start_idx = random.randint(0, actual_len - sub_len)
            
            # Extract and shuffle sub-sequence
            sub_seq = seq[start_idx:start_idx + sub_len].clone()
            shuffled_indices = torch.randperm(sub_len)
            seq[start_idx:start_idx + sub_len] = sub_seq[shuffled_indices]
            
            augmented_seqs.append(seq)
            rates.append(rate)
        
        return torch.stack(augmented_seqs), torch.tensor(rates, device=item_seq.device)

    def m_substitute(self, item_seq):
        """M-Substitute augmentation: substitute items with similar ones"""
        batch_size, seq_len = item_seq.shape
        augmented_seqs = []
        rates = []
        
        for i in range(batch_size):
            seq = item_seq[i].clone()
            # Find actual sequence length (excluding padding)
            actual_len = (seq != 0).sum().item()
            
            if actual_len <= 0:
                augmented_seqs.append(seq)
                rates.append(0.0)
                continue
            
            # Sample augmentation rate
            rate = np.random.uniform(self.rate_min, self.rate_max)
            sub_nums = max(int(rate * actual_len), 1)
            
            # Randomly select indices to substitute
            indices = random.sample(range(actual_len), min(sub_nums, actual_len))
            
            # Substitute with similar items
            for idx in indices:
                original_item = seq[idx].item()
                similar_item = self._get_similar_item(original_item)
                seq[idx] = similar_item
            
            augmented_seqs.append(seq)
            rates.append(rate)
        
        return torch.stack(augmented_seqs), torch.tensor(rates, device=item_seq.device)

    def adaptive_loss_weight(self, rate, lambda_mixup):
        """Compute adaptive loss weight based on augmentation rate and mixup coefficient"""
        # Avoid division by zero
        rate = torch.clamp(rate, min=1e-7)
        lambda_mixup = torch.clamp(lambda_mixup, min=1e-7)
        
        # Compute raw weight
        omega_1 = 1.0 / (rate * lambda_mixup)
        
        # Normalize to [0, 1] range
        omega_min = 1.0 / (self.rate_max * 1.0)
        omega_max = 1.0 / (self.rate_min * 1e-7)
        omega_2 = (omega_1 - omega_min) / (omega_max - omega_min + 1e-7)
        
        return torch.clamp(omega_2, min=0.0, max=1.0)

    def single_sequence_augmentation(self, item_seq, item_emb):
        """
        Single-sequence augmentation with M-Reorder and M-Substitute
        Returns augmented embeddings and loss weights
        """
        # Randomly choose augmentation method
        if random.random() < 0.5:
            aug_seq, rate = self.m_reorder(item_seq)
        else:
            aug_seq, rate = self.m_substitute(item_seq)
        
        # Get augmented item embeddings
        aug_item_emb = self.item_embedding(aug_seq)
        
        # Sample mixup coefficient from Beta distribution
        lambda_mixup = torch.from_numpy(
            np.random.beta(self.alpha, self.alpha, size=item_seq.shape[0])
        ).float().to(item_seq.device)
        lambda_mixup = lambda_mixup.view(-1, 1, 1)
        
        # Mix original and augmented embeddings
        mixed_emb = lambda_mixup * item_emb + (1 - lambda_mixup) * aug_item_emb
        
        # Compute adaptive loss weights
        loss_weight = self.adaptive_loss_weight(rate, lambda_mixup.squeeze())
        
        return mixed_emb, loss_weight

    def cross_sequence_augmentation(self, seq_output):
        """
        Cross-sequence augmentation with nonlinear mixup
        Performs item-wise or feature-wise mixup between different sequences
        """
        batch_size, seq_len, hidden_size = seq_output.shape
        
        # Shuffle batch to create pairs
        shuffled_idx = torch.randperm(batch_size, device=seq_output.device)
        seq_output_shuffled = seq_output[shuffled_idx]
        
        if self.mixup_strategy == "item_wise":
            # Item-wise nonlinear mixup
            lambda_matrix = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len))
            ).float().to(seq_output.device).unsqueeze(-1)
            
            mixed_output = lambda_matrix * seq_output + (1 - lambda_matrix) * seq_output_shuffled
            
        elif self.mixup_strategy == "feature_wise":
            # Feature-wise nonlinear mixup
            lambda_matrix = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_size))
            ).float().to(seq_output.device).unsqueeze(1)
            
            mixed_output = lambda_matrix * seq_output + (1 - lambda_matrix) * seq_output_shuffled
            
        else:  # both
            # Randomly choose between item-wise and feature-wise
            if random.random() < 0.5:
                lambda_matrix = torch.from_numpy(
                    np.random.beta(self.alpha, self.alpha, size=(batch_size, seq_len))
                ).float().to(seq_output.device).unsqueeze(-1)
            else:
                lambda_matrix = torch.from_numpy(
                    np.random.beta(self.alpha, self.alpha, size=(batch_size, hidden_size))
                ).float().to(seq_output.device).unsqueeze(1)
            
            mixed_output = lambda_matrix * seq_output + (1 - lambda_matrix) * seq_output_shuffled
        
        return mixed_output

    def forward(self, item_seq, item_seq_len, augment=False):
        """
        Forward pass with optional augmentation
        """
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        
        # Single-sequence augmentation (if in stage 2 and augment=True)
        if augment and (not self.two_stage or self.current_epoch >= self.stage1_epochs):
            mixed_emb, loss_weight = self.single_sequence_augmentation(item_seq, item_emb)
            input_emb = mixed_emb + position_embedding
        else:
            input_emb = item_emb + position_embedding
            loss_weight = None
        
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        
        # Gather output at the last position
        output = self.gather_indexes(output, item_seq_len - 1)
        
        return output, loss_weight

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Stage 1: Only main loss (standard training)
        if self.two_stage and self.current_epoch < self.stage1_epochs:
            seq_output, _ = self.forward(item_seq, item_seq_len, augment=False)
            pos_items = interaction[self.POS_ITEM_ID]
            
            if self.loss_type == "BPR":
                neg_items = interaction[self.NEG_ITEM_ID]
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
        
        # Stage 2: Main loss + Single-sequence augmentation loss + Cross-sequence augmentation loss
        # Main loss (no augmentation)
        seq_output_main, _ = self.forward(item_seq, item_seq_len, augment=False)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            
            pos_score_main = torch.sum(seq_output_main * pos_items_emb, dim=-1)
            neg_score_main = torch.sum(seq_output_main * neg_items_emb, dim=-1)
            loss_main = self.loss_fct(pos_score_main, neg_score_main)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits_main = torch.matmul(seq_output_main, test_item_emb.transpose(0, 1))
            loss_main = self.loss_fct(logits_main, pos_items)
        
        # Single-sequence augmentation loss
        seq_output_ssa, loss_weight = self.forward(item_seq, item_seq_len, augment=True)
        
        if self.loss_type == "BPR":
            pos_score_ssa = torch.sum(seq_output_ssa * pos_items_emb, dim=-1)
            neg_score_ssa = torch.sum(seq_output_ssa * neg_items_emb, dim=-1)
            loss_ssa = self.loss_fct(pos_score_ssa, neg_score_ssa)
        else:
            logits_ssa = torch.matmul(seq_output_ssa, test_item_emb.transpose(0, 1))
            loss_ssa = self.loss_fct(logits_ssa, pos_items)
        
        # Apply adaptive weighting
        if loss_weight is not None:
            loss_ssa = (loss_ssa * loss_weight).mean()
        
        # Cross-sequence augmentation loss
        # Get full sequence output for cross-sequence augmentation
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        seq_output_full = trm_output[-1]
        
        # Apply cross-sequence augmentation
        seq_output_csa_full = self.cross_sequence_augmentation(seq_output_full)
        seq_output_csa = self.gather_indexes(seq_output_csa_full, item_seq_len - 1)
        
        # Also apply to item embeddings
        pos_items_emb_csa = self.cross_sequence_augmentation(pos_items_emb.unsqueeze(1)).squeeze(1)
        
        if self.loss_type == "BPR":
            neg_items_emb_csa = self.cross_sequence_augmentation(neg_items_emb.unsqueeze(1)).squeeze(1)
            pos_score_csa = torch.sum(seq_output_csa * pos_items_emb_csa, dim=-1)
            neg_score_csa = torch.sum(seq_output_csa * neg_items_emb_csa, dim=-1)
            loss_csa = self.loss_fct(pos_score_csa, neg_score_csa)
        else:
            # For CE, we need to mix all item embeddings
            test_item_emb_csa = self.cross_sequence_augmentation(
                test_item_emb.unsqueeze(0).expand(seq_output_csa.shape[0], -1, -1)
            )
            logits_csa = torch.bmm(
                seq_output_csa.unsqueeze(1), 
                test_item_emb_csa.transpose(1, 2)
            ).squeeze(1)
            loss_csa = self.loss_fct(logits_csa, pos_items)
        
        # Total loss
        total_loss = loss_main + loss_ssa + loss_csa
        
        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len, augment=False)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len, augment=False)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
