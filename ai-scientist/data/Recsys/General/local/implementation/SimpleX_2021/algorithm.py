# -*- coding: utf-8 -*-
# @Time   : 2021/10/15
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
SimpleX
################################################
Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

Reference code:
    https://github.com/xue-pai/SimpleX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class SimpleX(GeneralRecommender):
    r"""SimpleX is a simple yet effective collaborative filtering model that combines
    matrix factorization with behavior sequence aggregation.

    It uses cosine contrastive loss (CCL) with margin-based negative sampling and
    supports three aggregation methods: average pooling, self-attention, and user-attention.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.aggregator = config["aggregator"]  # 'mean', 'self_attn', 'user_attn'
        self.gamma = config["gamma"]  # gate weight g in Equation 5
        self.margin = config["margin"]  # margin m in CCL
        self.negative_weight = config["negative_weight"]  # weight w in CCL
        self.reg_weight = config["reg_weight"]
        self.history_len = config["history_len"]  # max sequence length K

        # Define embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Define aggregation parameters
        if self.aggregator == "self_attn":
            self.W1 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
            self.query = nn.Parameter(torch.Tensor(self.embedding_size, 1))
            nn.init.xavier_uniform_(self.query)
        elif self.aggregator == "user_attn":
            self.W2 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        # Fusion transformation matrix V
        self.V = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        # Regularization
        self.reg_loss = EmbLoss()

        # Storage for evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Build user history sequences
        self.history_item_id, self.history_item_len = self._build_history_items(dataset)

        # Apply initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def _build_history_items(self, dataset):
        r"""Build user's historical interacted item sequences.

        Args:
            dataset: The dataset object containing interaction data.

        Returns:
            tuple: (history_item_id, history_item_len)
                - history_item_id: tensor of shape [n_users, history_len]
                - history_item_len: tensor of shape [n_users]
        """
        # Get user-item interactions
        user_ids = dataset.inter_feat[self.USER_ID].numpy()
        item_ids = dataset.inter_feat[self.ITEM_ID].numpy()

        # Build history dictionary
        history_dict = {}
        for uid, iid in zip(user_ids, item_ids):
            if uid not in history_dict:
                history_dict[uid] = []
            history_dict[uid].append(iid)

        # Convert to padded sequences
        history_item_id = np.zeros((self.n_users, self.history_len), dtype=np.int64)
        history_item_len = np.zeros(self.n_users, dtype=np.int64)

        for uid in range(self.n_users):
            if uid in history_dict:
                items = history_dict[uid]
                length = min(len(items), self.history_len)
                history_item_id[uid, :length] = items[:length]
                history_item_len[uid] = length

        return (
            torch.LongTensor(history_item_id).to(self.device),
            torch.LongTensor(history_item_len).to(self.device),
        )

    def aggregate_behavior(self, user_ids):
        r"""Aggregate user's historical item embeddings.

        Args:
            user_ids: tensor of user indices, shape [batch_size]

        Returns:
            tensor: aggregated pooling vector p_u, shape [batch_size, embedding_size]
        """
        # Get user's history items
        history_items = self.history_item_id[user_ids]  # [batch_size, history_len]
        history_lens = self.history_item_len[user_ids]  # [batch_size]

        # Get item embeddings
        item_embs = self.item_embedding(history_items)  # [batch_size, history_len, emb_size]

        # Create mask for valid items (I_k in Equation 2)
        mask = torch.arange(self.history_len, device=self.device).unsqueeze(0) < history_lens.unsqueeze(1)
        mask = mask.float().unsqueeze(-1)  # [batch_size, history_len, 1]

        if self.aggregator == "mean":
            # Average pooling (Equation 3, upper part)
            masked_embs = item_embs * mask
            pooled = masked_embs.sum(dim=1) / (history_lens.unsqueeze(1).float() + 1e-8)

        elif self.aggregator == "self_attn":
            # Self-attention (Equation 4, upper part)
            # beta_k = q^T * tanh(W1 * e_k + b1)
            transformed = self.W1(item_embs)  # [batch_size, history_len, emb_size]
            activated = torch.tanh(transformed)  # [batch_size, history_len, emb_size]
            beta = torch.matmul(activated, self.query).squeeze(-1)  # [batch_size, history_len]

            # Apply mask and softmax
            beta = beta.masked_fill(~mask.squeeze(-1).bool(), float("-inf"))
            alpha = F.softmax(beta, dim=1).unsqueeze(-1)  # [batch_size, history_len, 1]

            # Weighted sum
            pooled = (item_embs * alpha).sum(dim=1)  # [batch_size, emb_size]

        elif self.aggregator == "user_attn":
            # User-attention (Equation 4, lower part)
            # beta_k = e_u^T * tanh(W2 * e_k + b2)
            user_embs = self.user_embedding(user_ids)  # [batch_size, emb_size]
            transformed = self.W2(item_embs)  # [batch_size, history_len, emb_size]
            activated = torch.tanh(transformed)  # [batch_size, history_len, emb_size]

            # Compute attention scores
            beta = torch.bmm(
                activated, user_embs.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, history_len]

            # Apply mask and softmax
            beta = beta.masked_fill(~mask.squeeze(-1).bool(), float("-inf"))
            alpha = F.softmax(beta, dim=1).unsqueeze(-1)  # [batch_size, history_len, 1]

            # Weighted sum
            pooled = (item_embs * alpha).sum(dim=1)  # [batch_size, emb_size]

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return pooled

    def forward(self, user_ids):
        r"""Get final user representations by fusing user embeddings and aggregated behaviors.

        Args:
            user_ids: tensor of user indices

        Returns:
            tensor: final user representation h_u
        """
        # Get user embeddings
        user_embs = self.user_embedding(user_ids)  # [batch_size, emb_size]

        # Aggregate behavior sequences
        pooled = self.aggregate_behavior(user_ids)  # [batch_size, emb_size]

        # Transform pooled vector
        transformed_pooled = self.V(pooled)  # [batch_size, emb_size]

        # Fuse user embedding and pooled vector (Equation 5)
        user_repr = self.gamma * user_embs + (1 - self.gamma) * transformed_pooled

        return user_repr

    def cosine_similarity(self, user_repr, item_embs):
        r"""Compute cosine similarity between user and item representations (Equation 6).

        Args:
            user_repr: user representations, shape [batch_size, emb_size]
            item_embs: item embeddings, shape [batch_size, emb_size] or [n_items, emb_size]

        Returns:
            tensor: cosine similarity scores
        """
        # L2 normalization
        user_repr_norm = F.normalize(user_repr, p=2, dim=1)
        item_embs_norm = F.normalize(item_embs, p=2, dim=-1)

        # Compute cosine similarity
        if user_repr_norm.dim() == 2 and item_embs_norm.dim() == 2:
            if user_repr_norm.size(0) == item_embs_norm.size(0):
                # Element-wise similarity
                similarity = (user_repr_norm * item_embs_norm).sum(dim=1)
            else:
                # Matrix multiplication for full sort
                similarity = torch.matmul(user_repr_norm, item_embs_norm.T)
        else:
            similarity = torch.matmul(user_repr_norm, item_embs_norm.T)

        return similarity

    def calculate_ccl_loss(self, user_ids, pos_items, neg_items):
        r"""Calculate Cosine Contrastive Loss (Equation 1).

        Args:
            user_ids: tensor of user indices
            pos_items: tensor of positive item indices
            neg_items: tensor of negative item indices, shape [batch_size, n_negs]

        Returns:
            tensor: CCL loss value
        """
        batch_size = user_ids.size(0)

        # Get user representations
        user_repr = self.forward(user_ids)  # [batch_size, emb_size]

        # Get item embeddings
        pos_item_embs = self.item_embedding(pos_items)  # [batch_size, emb_size]
        neg_item_embs = self.item_embedding(neg_items)  # [batch_size, emb_size] or [batch_size, n_negs, emb_size]

        # Compute cosine similarity for positive pairs
        pos_scores = self.cosine_similarity(user_repr, pos_item_embs)  # [batch_size]

        # Positive loss: 1 - y_ui
        pos_loss = (1 - pos_scores).sum()

        # Compute cosine similarity for negative pairs
        if neg_item_embs.dim() == 2:
            # Single negative per sample
            neg_scores = self.cosine_similarity(user_repr, neg_item_embs)  # [batch_size]
            # Negative loss with margin: max(0, y_uj - m)
            neg_loss = torch.clamp(neg_scores - self.margin, min=0).sum()
            n_negs = 1
        else:
            # Multiple negatives per sample
            n_negs = neg_item_embs.size(1)
            user_repr_expanded = user_repr.unsqueeze(1).expand(-1, n_negs, -1)  # [batch_size, n_negs, emb_size]
            neg_scores = self.cosine_similarity(
                user_repr_expanded.reshape(-1, self.embedding_size),
                neg_item_embs.reshape(-1, self.embedding_size)
            ).reshape(batch_size, n_negs)  # [batch_size, n_negs]
            # Negative loss with margin
            neg_loss = torch.clamp(neg_scores - self.margin, min=0).sum()

        # Weighted negative loss (Equation 1)
        ccl_loss = pos_loss + (self.negative_weight / n_negs) * neg_loss

        return ccl_loss

    def calculate_loss(self, interaction):
        r"""Calculate the training loss.

        Args:
            interaction: interaction data

        Returns:
            tensor: total loss
        """
        # Clear cached embeddings
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_ids = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # Calculate CCL loss
        ccl_loss = self.calculate_ccl_loss(user_ids, pos_items, neg_items)

        # Calculate regularization loss on ego embeddings
        u_ego_embs = self.user_embedding(user_ids)
        pos_ego_embs = self.item_embedding(pos_items)
        neg_ego_embs = self.item_embedding(neg_items)
        reg_loss = self.reg_loss(u_ego_embs, pos_ego_embs, neg_ego_embs)

        # Total loss
        total_loss = ccl_loss + self.reg_weight * reg_loss

        return total_loss

    def predict(self, interaction):
        r"""Predict scores for given user-item pairs.

        Args:
            interaction: interaction data

        Returns:
            tensor: predicted scores
        """
        user_ids = interaction[self.USER_ID]
        item_ids = interaction[self.ITEM_ID]

        # Get representations
        user_repr = self.forward(user_ids)
        item_embs = self.item_embedding(item_ids)

        # Compute cosine similarity
        scores = self.cosine_similarity(user_repr, item_embs)

        return scores

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.

        Args:
            interaction: interaction data

        Returns:
            tensor: predicted scores for all items
        """
        user_ids = interaction[self.USER_ID]

        # Cache item embeddings for efficiency
        if self.restore_item_e is None:
            self.restore_item_e = self.item_embedding.weight

        # Get user representations
        user_repr = self.forward(user_ids)

        # Compute cosine similarity with all items
        scores = self.cosine_similarity(user_repr, self.restore_item_e)

        return scores.view(-1)