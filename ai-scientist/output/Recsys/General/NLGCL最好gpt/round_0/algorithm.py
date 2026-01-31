# -*- coding: utf-8 -*-
# @Time   : 2026/01/12
# @Author : ChatGPT (implemented per paper description)
r"""
NLGCL (Neighbor Layers Graph Contrastive Learning) - 2025
################################################

Paper: NLGCL_2025 (method core reproduced)
Backbone: LightGCN
Self-supervision: Neighbor-layer graph contrastive learning between layer g and g+1.

Key idea:
- Use naturally existing "views" between adjacent GCN layers (g) and (g+1).
- Positives for a user u at layer g: embeddings of its neighbor items at layer (g+1).
  Similarly for item i: neighbor users.
- Negatives:
  - heterogeneous scope: user anchors contrast against all items; item anchors against all users.
  - entire scope: user anchors contrast against all (users+items); item anchors contrast against all (users+items).
- Loss per paper Eq.(6)-(9): multi-positive InfoNCE-like objective with product over positives.
  We implement it in log-domain: sum(pos_logits) - |N_u| * logsumexp(neg_logits), then normalize by |N_u|.

Engineering notes:
- Exact paper uses "all users/items" in denominator -> O(|U||I|) per layer if full-batch.
  We implement a mini-batch approximation:
    * anchors: batch users (from interaction) and batch pos-items (as item anchors)
    * positives: the sampled (u, pos_item) pairs from BPR batch (implicit neighbor)
    * negatives: all items/users (or all nodes) from layer (g+1) embeddings (global) via matmul.
  This matches the formula's negative pool while keeping anchors mini-batched.
- For multi-positive average 1/|N_u|, we approximate by using sampled neighbors in batch:
  if a user appears multiple times in batch, we treat each (u, pos_item) as one positive,
  and average over counts per user (same for items).
  This is consistent with "each node has multiple positives" and avoids explicit neighbor loops.

RecBole compatibility:
- Inherit GeneralRecommender, InputType.PAIRWISE
- calculate_loss() returns BPR + lambda1*NL + lambda2*L2 (reg_weight)
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NLGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ===== dataset =====
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # ===== hyper-params =====
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]                      # L
        self.g_groups = config.get("g_groups", self.n_layers)   # G <= L
        self.g_groups = min(self.g_groups, self.n_layers)

        self.ssl_temp = config.get("ssl_temp", 0.2)             # tau
        self.ssl_weight = config.get("ssl_weight", 1e-3)        # lambda1
        self.reg_weight = config["reg_weight"]                  # lambda2 in paper (paired with EmbLoss)
        self.require_pow = config.get("require_pow", False)

        # scope: 'hetero' or 'entire'
        self.nl_scope = config.get("nl_scope", "hetero").lower()
        if self.nl_scope not in ("hetero", "entire"):
            raise ValueError(f"nl_scope must be in ['hetero', 'entire'], got {self.nl_scope}")

        # ===== parameters =====
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # losses
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # ===== graph =====
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # cache for full-sort eval
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # init
        self.apply(xavier_normal_initialization)

    # ---------- Graph utils (same as LightGCN) ----------
    def get_norm_adj_mat(self):
        """Construct D^{-1/2} A D^{-1/2} for bipartite user-item graph as a torch sparse tensor."""
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz))
        )
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D

        L = sp.coo_matrix(L)
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

    def get_ego_embeddings(self):
        """E^(0): concat([U, I])."""
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    # ---------- LightGCN encoder returning per-layer embeddings ----------
    def forward_layers(self):
        """
        Returns:
            user_layers: list length (L+1), each [n_users, dim]
            item_layers: list length (L+1), each [n_items, dim]
            user_final:  [n_users, dim] mean over layers
            item_final:  [n_items, dim] mean over layers
        """
        all_emb = self.get_ego_embeddings()
        layers_all = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            layers_all.append(all_emb)

        # split layers into user/item
        user_layers, item_layers = [], []
        for e in layers_all:
            u_e, i_e = torch.split(e, [self.n_users, self.n_items], dim=0)
            user_layers.append(u_e)
            item_layers.append(i_e)

        # final embeddings: mean over 0..L
        stacked = torch.stack(layers_all, dim=1)  # [n_nodes, L+1, dim]
        final_all = torch.mean(stacked, dim=1)
        user_final, item_final = torch.split(final_all, [self.n_users, self.n_items], dim=0)
        return user_layers, item_layers, user_final, item_final

    def forward(self):
        """Standard RecBole forward: returns final user/item embeddings."""
        _, _, user_final, item_final = self.forward_layers()
        return user_final, item_final

    # ---------- NLGCL loss (neighbor-layer CL) ----------
    @staticmethod
    def _group_mean(values: torch.Tensor, group_index: torch.Tensor, n_groups: int):
        """
        Compute mean of `values` per group id using scatter_add.
        Args:
            values: [B]
            group_index: [B] int64 in [0, n_groups)
            n_groups: int
        Returns:
            mean_per_group: [n_groups] (0 for empty groups)
            count_per_group: [n_groups]
        """
        device = values.device
        out = torch.zeros(n_groups, device=device, dtype=values.dtype)
        cnt = torch.zeros(n_groups, device=device, dtype=values.dtype)
        out.scatter_add_(0, group_index, values)
        ones = torch.ones_like(values, dtype=values.dtype)
        cnt.scatter_add_(0, group_index, ones)
        mean = out / (cnt + 1e-12)
        return mean, cnt

    def _nlgcl_loss_batch(self, user_layers, item_layers, batch_users, batch_pos_items):
        """
        Mini-batch approximation of Eq.(6)-(9) using:
          - user anchors: batch_users at layer g
          - positive items: batch_pos_items at layer g+1
          - item anchors: batch_pos_items at layer g (as item-side nodes)
          - positive users: batch_users at layer g+1
        Denominator uses global negatives (all items/users or all nodes) at layer g+1.

        Args:
            user_layers: list of [n_users, dim] len L+1
            item_layers: list of [n_items, dim] len L+1
            batch_users: LongTensor [B]
            batch_pos_items: LongTensor [B]
        Returns:
            nl_loss: scalar
        """
        tau = self.ssl_temp
        G = self.g_groups

        # We aggregate per unique anchor node (to simulate 1/|N_u| averaging and sum over all nodes)
        uniq_users, inv_u = torch.unique(batch_users, return_inverse=True)
        uniq_items, inv_i = torch.unique(batch_pos_items, return_inverse=True)

        nl_u_total = 0.0
        nl_i_total = 0.0

        # Precompute node pool sizes for normalization (paper uses |U|,|I|; we use #uniq in batch)
        # This is a standard mini-batch objective scaling.
        n_anchors_u = max(int(uniq_users.numel()), 1)
        n_anchors_i = max(int(uniq_items.numel()), 1)

        for g in range(G):
            # anchors at layer g
            u_g = user_layers[g][uniq_users]          # [Bu, d]
            i_g = item_layers[g][uniq_items]          # [Bi, d]

            # positives at layer g+1 (per interaction pair)
            pos_i_g1 = item_layers[g + 1][batch_pos_items]  # [B, d]
            pos_u_g1 = user_layers[g + 1][batch_users]      # [B, d]

            # optionally normalize (cosine); paper uses dot product, keep dot by default.
            # If you want cosine, uncomment:
            # u_g = F.normalize(u_g, dim=1)
            # i_g = F.normalize(i_g, dim=1)
            # pos_i_g1 = F.normalize(pos_i_g1, dim=1)
            # pos_u_g1 = F.normalize(pos_u_g1, dim=1)

            # ===== user-side: u^(g) vs items^(g+1) =====
            # numerator: sum_{i+ in N_u} (u·i+)/tau  (log product = sum logits)
            # Here each (u, pos_item) in batch is a positive sample for that u.
            # compute pair logits for each interaction, then group-sum by user.
            u_for_pairs = user_layers[g][batch_users]  # [B,d] (not uniq, aligned with pairs)
            pos_logits_u = (u_for_pairs * pos_i_g1).sum(dim=1) / tau  # [B]
            pos_sum_u, cnt_u = self._group_mean(pos_logits_u, inv_u, n_anchors_u)  # mean of logits
            # But paper uses (1/|N_u|) * log( exp(sum logits)/den ) =>
            # (1/|N_u|)*(sum logits - |N_u|*logsumexp(neg)) = mean_logits - logsumexp(neg)
            # So mean over positives is correct.

            # denominator: log sum_{neg in pool} exp(u·neg/tau)
            if self.nl_scope == "hetero":
                neg_pool_u = item_layers[g + 1]  # all items at layer g+1: [n_items, d]
            else:
                # entire: all nodes (users+items) at layer g+1
                neg_pool_u = torch.cat([user_layers[g + 1], item_layers[g + 1]], dim=0)

            # compute logsumexp for each unique anchor user
            # logits: [Bu, Nneg]
            neg_logits_u = torch.matmul(u_g, neg_pool_u.t()) / tau
            log_denom_u = torch.logsumexp(neg_logits_u, dim=1)  # [Bu]

            # loss per user: -(mean_pos_logit - log_denom)
            nl_u_layer = -(pos_sum_u - log_denom_u).mean()
            nl_u_total = nl_u_total + nl_u_layer

            # ===== item-side: i^(g) vs users^(g+1) =====
            i_for_pairs = item_layers[g][batch_pos_items]  # [B,d]
            pos_logits_i = (i_for_pairs * pos_u_g1).sum(dim=1) / tau  # [B]
            pos_sum_i, cnt_i = self._group_mean(pos_logits_i, inv_i, n_anchors_i)

            if self.nl_scope == "hetero":
                neg_pool_i = user_layers[g + 1]  # all users
            else:
                neg_pool_i = torch.cat([user_layers[g + 1], item_layers[g + 1]], dim=0)

            neg_logits_i = torch.matmul(i_g, neg_pool_i.t()) / tau
            log_denom_i = torch.logsumexp(neg_logits_i, dim=1)

            nl_i_layer = -(pos_sum_i - log_denom_i).mean()
            nl_i_total = nl_i_total + nl_i_layer

        nl_u_total = nl_u_total / G
        nl_i_total = nl_i_total / G
        return nl_u_total + nl_i_total

    # ---------- Training loss ----------
    def calculate_loss(self, interaction):
        # clear cache
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]         # [B]
        pos_item = interaction[self.ITEM_ID]     # [B]
        neg_item = interaction[self.NEG_ITEM_ID] # [B]

        user_layers, item_layers, user_final, item_final = self.forward_layers()

        # ===== BPR =====
        u_e = user_final[user]
        pos_e = item_final[pos_item]
        neg_e = item_final[neg_item]

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # ===== reg (as in LightGCN/SGL: on ego embeddings) =====
        u_ego = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego, pos_ego, neg_ego, require_pow=self.require_pow)

        # ===== NLGCL neighbor-layer loss =====
        if self.training and self.ssl_weight > 0:
            nl_loss = self._nlgcl_loss_batch(user_layers, item_layers, user, pos_item)
        else:
            nl_loss = torch.zeros((), device=self.device)

        return mf_loss + self.ssl_weight * nl_loss + self.reg_weight * reg_loss

    # ---------- Inference ----------
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_final, item_final = self.forward()
        scores = (user_final[user] * item_final[item]).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_e = self.restore_user_e[user]  # [B, d] or [1,d]
        scores = torch.matmul(u_e, self.restore_item_e.t())  # [B, n_items]
        return scores.view(-1)