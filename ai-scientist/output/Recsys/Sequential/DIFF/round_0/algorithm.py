# -*- coding: utf-8 -*-
# DIFF_2025
#
# Core ideas faithfully implemented:
# 1) Side-info (multi-attribute) sequential recommendation
# 2) Frequency-based noise filtering (FFT -> low/high split -> iFFT; learnable beta for high-freq)
# 3) Dual multi-sequence fusion per layer:
#    - ID-centric: fuse attention logits from ID and each attribute; value from ID
#    - Attribute-enriched: self-attention on early-fused (ID+attrs) embedding
#    - Final representation: alpha * Rv + (1-alpha) * Rva; use last position for prediction
# 4) Representation alignment loss between Ev and Ea (fused attrs) with learnable temperature
#
# Notes / minimal assumptions exposed as config switches:
# - fusion_type: how to fuse embeddings/logits (sum/concat/gate)
# - cutoff_mode: cutoff defined in rFFT frequency bins ('rfft') or as ratio of seq length ('ratio')
# - align_label_mode: label matrix Y uses attribute-id equality ('attr_id_equal') or embedding equality ('embed_equal')
# - align_loss_on_valid: compute alignment only on non-padding positions
#
# Safe degradation: if selected_features missing or invalid shapes, fall back to ID-only behavior.

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import FeatureSeqEmbLayer


class _FFN(nn.Module):
    """Transformer-style point-wise FFN with residual+LayerNorm kept outside (to match RecBole encoder style)."""

    def __init__(self, hidden_size: int, inner_size: int, hidden_dropout_prob: float, hidden_act: str, layer_norm_eps: float):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        if hidden_act.lower() == "gelu":
            self.act = F.gelu
        elif hidden_act.lower() == "relu":
            self.act = F.relu
        elif hidden_act.lower() == "swish":
            self.act = lambda x: x * torch.sigmoid(x)
        else:
            # fall back to gelu (common in RecBole configs)
            self.act = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        hidden = self.dense_1(x)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.dense_2(hidden)
        hidden = self.dropout(hidden)
        out = self.LayerNorm(hidden + x)
        return out


class _FreqFilter(nn.Module):
    """Frequency-based noise filtering along sequence length dimension.

    Implemented with rFFT/irFFT for real signals.
    Split in frequency domain: keep first c bins as low-frequency; rest as high-frequency.
    filtered = low + beta * high
    """

    def __init__(
        self,
        max_seq_length: int,
        cutoff: int,
        cutoff_mode: str = "rfft",  # 'rfft' or 'ratio'
        init_beta: float = 0.0,
        learnable_beta: bool = True,
    ):
        super().__init__()
        self.max_seq_length = int(max_seq_length)
        self.cutoff = int(cutoff)
        self.cutoff_mode = cutoff_mode
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(init_beta)))

        # cache masks per length to avoid rebuilding each batch (length varies but usually fixed to max_seq_length)
        self._mask_cache: Dict[int, torch.Tensor] = {}

    def _get_c(self, L: int) -> int:
        # rFFT length = L//2 + 1
        rlen = L // 2 + 1
        if self.cutoff_mode == "ratio":
            # cutoff is treated as ratio in [0,1], but config provides number -> interpret <=1.0 as ratio
            ratio = float(self.cutoff)
            if ratio > 1.0:
                # minimal assumption: if user passes integer >1 with ratio mode, treat as bins
                c = int(ratio)
            else:
                c = max(1, int(round(rlen * ratio)))
        else:
            c = int(self.cutoff)
        c = max(1, min(rlen, c))
        return c

    def _get_low_mask(self, L: int, device: torch.device) -> torch.Tensor:
        # mask: [1, rlen, 1] with 1 for low-freq bins
        if L in self._mask_cache and self._mask_cache[L].device == device:
            return self._mask_cache[L]
        rlen = L // 2 + 1
        c = self._get_c(L)
        mask = torch.zeros((1, rlen, 1), device=device, dtype=torch.float32)
        mask[:, :c, :] = 1.0
        self._mask_cache[L] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L,D]
        returns: [B,L,D]
        """
        B, L, D = x.shape
        # rfft on length dimension
        Xf = torch.fft.rfft(x, dim=1, norm="ortho")  # [B, rlen, D], complex
        low_mask = self._get_low_mask(L, x.device).to(dtype=Xf.real.dtype)
        low = Xf * low_mask
        high = Xf * (1.0 - low_mask)
        x_low = torch.fft.irfft(low, n=L, dim=1, norm="ortho")
        x_high = torch.fft.irfft(high, n=L, dim=1, norm="ortho")
        return x_low + self.beta * x_high


class _Fusion(nn.Module):
    """Fusion operator for embeddings and/or attention logits.

    Supports:
    - sum: elementwise sum
    - concat: concat on last dim then linear projection to hidden_size
    - gate: gated sum using a learned sigmoid gate per source
    """

    def __init__(self, hidden_size: int, num_sources: int, fusion_type: str = "sum", dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(dropout)

        if fusion_type == "concat":
            self.proj = nn.Linear(hidden_size * num_sources, hidden_size)
        elif fusion_type == "gate":
            # gate per source: g_s = sigmoid(W_s x_s), output = sum_s g_s * x_s
            self.gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_sources)])
        elif fusion_type == "sum":
            self.proj = None
        else:
            raise NotImplementedError(f"fusion_type must be in ['sum','concat','gate'], got {fusion_type}")

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # xs: list of tensors with same shape [..., D]
        if len(xs) == 1:
            return xs[0]
        if self.fusion_type == "sum":
            out = xs[0]
            for x in xs[1:]:
                out = out + x
            return self.dropout(out)
        elif self.fusion_type == "concat":
            out = torch.cat(xs, dim=-1)
            out = self.proj(out)
            return self.dropout(out)
        else:  # gate
            outs = []
            for i, x in enumerate(xs):
                g = torch.sigmoid(self.gate[i](x))
                outs.append(g * x)
            out = outs[0]
            for x in outs[1:]:
                out = out + x
            return self.dropout(out)


class _MultiHeadAttnLogitsFusion(nn.Module):
    """One layer of DIFF dual fusion (no external TransformerEncoder to keep behavior explicit).

    ID-centric:
      - compute logits A_v and A_a_k from Q/K projections
      - fuse logits across sources (ID + attributes) BEFORE softmax
      - apply causal mask once
      - softmax -> dropout -> matmul with V_v (ID values)
      - output projection -> dropout -> residual+LN
      - then FFN (residual+LN inside)
    Attribute-enriched:
      - standard self-attention on early-fused embeddings (va) with its own QKV
      - same mask
      - output projection -> dropout -> residual+LN
      - then FFN

    Returns updated sequences: Ev_seq, Eva_seq
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        num_attrs: int,
        attn_fusion_type: str = "sum",
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size({hidden_size}) must be divisible by n_heads({n_heads})")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_attrs = num_attrs

        # ID-centric projections: Q/K for ID and each attribute; V only for ID
        self.q_v = nn.Linear(hidden_size, hidden_size)
        self.k_v = nn.Linear(hidden_size, hidden_size)
        self.v_v = nn.Linear(hidden_size, hidden_size)

        self.q_a = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_attrs)])
        self.k_a = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_attrs)])

        # Fused (va) projections
        self.q_va = nn.Linear(hidden_size, hidden_size)
        self.k_va = nn.Linear(hidden_size, hidden_size)
        self.v_va = nn.Linear(hidden_size, hidden_size)

        # logits fusion (across sources), operates on [..., L, L]
        # We fuse per head; treat last dim not hidden, so implement simple sum/concat/gate with small paramization
        self.attn_fusion_type = attn_fusion_type
        if attn_fusion_type == "sum":
            self.logits_proj = None
        elif attn_fusion_type == "concat":
            # concat sources on last dim (source dim), then linear -> single logits
            # implement with 1x1 linear on source dimension
            self.logits_proj = nn.Linear(1 + num_attrs, 1, bias=False)
        elif attn_fusion_type == "gate":
            self.logits_gate = nn.Parameter(torch.zeros(1 + num_attrs))
        else:
            raise NotImplementedError("attn_fusion_type must be in ['sum','concat','gate']")

        # output projections
        self.o_v = nn.Linear(hidden_size, hidden_size)
        self.o_va = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)

        self.ln_v = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_va = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.ffn_v = _FFN(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.ffn_va = _FFN(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D] -> [B,H,L,dh]
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,L,dh] -> [B,L,D]
        B, H, L, dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * dh)
        return x

    def _fuse_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        # each: [B,H,L,L]
        if len(logits_list) == 1:
            return logits_list[0]
        if self.attn_fusion_type == "sum":
            out = logits_list[0]
            for t in logits_list[1:]:
                out = out + t
            return out
        elif self.attn_fusion_type == "gate":
            # scalar gate per source (broadcast)
            g = torch.sigmoid(self.logits_gate).view(-1)  # [S]
            out = logits_list[0] * g[0]
            for i, t in enumerate(logits_list[1:], start=1):
                out = out + t * g[i]
            return out
        else:  # concat then linear on source dimension
            # stack on a new source dim, then linear combine
            stacked = torch.stack(logits_list, dim=-1)  # [B,H,L,L,S]
            out = self.logits_proj(stacked).squeeze(-1)  # [B,H,L,L]
            return out

    def forward(
        self,
        e_v: torch.Tensor,                 # [B,L,D]
        e_attrs: List[torch.Tensor],       # m * [B,L,D]
        e_va: torch.Tensor,                # [B,L,D]
        attn_mask: torch.Tensor,           # RecBole mask: [B,1,L,L] float with -inf for masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ----- ID-centric -----
        qv = self._split_heads(self.q_v(e_v))  # [B,H,L,dh]
        kv = self._split_heads(self.k_v(e_v))
        vv = self._split_heads(self.v_v(e_v))

        logits_v = torch.matmul(qv, kv.transpose(-2, -1))  # [B,H,L,L]
        logits_list = [logits_v]

        for k in range(self.num_attrs):
            qa = self._split_heads(self.q_a[k](e_attrs[k]))
            ka = self._split_heads(self.k_a[k](e_attrs[k]))
            logits_a = torch.matmul(qa, ka.transpose(-2, -1))  # [B,H,L,L]
            logits_list.append(logits_a)

        fused_logits = self._fuse_logits(logits_list) * self.scale  # [B,H,L,L]
        fused_logits = fused_logits + attn_mask  # add mask once (contract)
        attn = torch.softmax(fused_logits, dim=-1)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, vv)  # [B,H,L,dh]
        ctx = self._merge_heads(ctx)  # [B,L,D]
        out_v = self.o_v(ctx)
        out_v = self.hidden_dropout(out_v)
        out_v = self.ln_v(out_v + e_v)  # residual + LN
        out_v = self.ffn_v(out_v)

        # ----- Attribute-enriched (early fused) -----
        qva = self._split_heads(self.q_va(e_va))
        kva = self._split_heads(self.k_va(e_va))
        vva = self._split_heads(self.v_va(e_va))
        logits_va = torch.matmul(qva, kva.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        logits_va = logits_va + attn_mask
        attn_va = torch.softmax(logits_va, dim=-1)
        attn_va = self.attn_dropout(attn_va)
        ctx_va = torch.matmul(attn_va, vva)
        ctx_va = self._merge_heads(ctx_va)
        out_va = self.o_va(ctx_va)
        out_va = self.hidden_dropout(out_va)
        out_va = self.ln_va(out_va + e_va)
        out_va = self.ffn_va(out_va)

        return out_v, out_va


class DIFF(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ----- core transformer params -----
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]  # 'CE' or 'BPR'

        # ----- side information -----
        self.selected_features: List[str] = config.get("selected_features", [])
        self.pooling_mode = config.get("pooling_mode", "mean")
        self.num_attrs = len(self.selected_features)

        # ----- DIFF-specific hyperparams -----
        self.fusion_type = config.get("fusion_type", "sum")  # for early fusion embeddings and Ea in align
        self.attn_fusion_type = config.get("attn_fusion_type", "sum")  # for attention logits fusion
        self.alpha = float(config.get("alpha", 0.5))
        self.lambda_align = float(config.get("lambda_align", 0.1))

        self.cutoff = int(config.get("cutoff", 4))  # c in paper (freq bins)
        self.cutoff_mode = config.get("cutoff_mode", "rfft")  # 'rfft' or 'ratio'
        self.init_beta = float(config.get("init_beta", 0.0))
        self.learnable_beta = bool(config.get("learnable_beta", True))

        self.align_label_mode = config.get("align_label_mode", "attr_id_equal")  # 'attr_id_equal' or 'embed_equal'
        self.align_loss_on_valid = bool(config.get("align_loss_on_valid", True))

        # ----- embeddings -----
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # attribute embedding via RecBole feature embed layer (robust to sparse/dense)
        # This layer returns per-feature embedding table aligned with item_seq.
        # If features missing, we will safely degrade at runtime.
        if self.num_attrs > 0:
            self.feature_embed_layer = FeatureSeqEmbLayer(
                dataset,
                self.hidden_size,
                self.selected_features,
                self.pooling_mode,
                self.device,
            )
            self.other_parameter_name = ["feature_embed_layer"]
        else:
            self.feature_embed_layer = None

        # early fusion module: fuse [Ev] + m*[Ea_k] into E_va, all are [B,L,D]
        self.early_fusion = _Fusion(self.hidden_size, 1 + max(0, self.num_attrs), self.fusion_type, dropout=self.hidden_dropout_prob)
        # for alignment Ea fusion, paper uses summation; keep configurable but default 'sum'
        self.attr_fusion_for_align = _Fusion(self.hidden_size, max(1, self.num_attrs), config.get("align_attr_fusion_type", "sum"), dropout=0.0)

        # ----- frequency filtering -----
        # beta scalars: beta_0 for ID, beta_k for each attr, beta_{m+1} for early fused
        self.filter_v = _FreqFilter(self.max_seq_length, self.cutoff, self.cutoff_mode, self.init_beta, self.learnable_beta)
        self.filter_va = _FreqFilter(self.max_seq_length, self.cutoff, self.cutoff_mode, self.init_beta, self.learnable_beta)
        self.filter_attrs = nn.ModuleList(
            [_FreqFilter(self.max_seq_length, self.cutoff, self.cutoff_mode, self.init_beta, self.learnable_beta) for _ in range(self.num_attrs)]
        )

        # ----- dual fusion layers -----
        self.layers = nn.ModuleList(
            [
                _MultiHeadAttnLogitsFusion(
                    hidden_size=self.hidden_size,
                    n_heads=self.n_heads,
                    inner_size=self.inner_size,
                    hidden_dropout_prob=self.hidden_dropout_prob,
                    attn_dropout_prob=self.attn_dropout_prob,
                    hidden_act=self.hidden_act,
                    layer_norm_eps=self.layer_norm_eps,
                    num_attrs=self.num_attrs,
                    attn_fusion_type=self.attn_fusion_type,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.input_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)

        # alignment temperature (learnable)
        # enforce positivity via softplus on forward
        init_tau = float(config.get("init_tau", 0.07))
        self._tau = nn.Parameter(torch.tensor(math.log(math.exp(init_tau) - 1.0)))  # inverse softplus

        # loss
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR','CE']!")

        self._warned_feature_once = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._tau) + 1e-8

    def _get_pos_emb(self, item_seq: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        position_ids = torch.arange(L, dtype=torch.long, device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        return self.position_embedding(position_ids)

    def _extract_attr_embeddings(self, item_seq: torch.Tensor) -> List[torch.Tensor]:
        """Return list of m tensors [B,L,D], aligned with selected_features order.

        Safe degradation: if missing/invalid, return empty list.
        """
        if self.feature_embed_layer is None or self.num_attrs == 0:
            return []

        try:
            sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
            # Both are dict keyed by 'item' in RecBole
            s = sparse_embedding.get("item", None) if sparse_embedding is not None else None
            d = dense_embedding.get("item", None) if dense_embedding is not None else None

            tables = []
            if s is not None:
                tables.append(s)  # [B,L,num_sparse,D] typically (with an extra feature dim)
            if d is not None:
                tables.append(d)  # [B,L,num_dense,D]
            if len(tables) == 0:
                return []

            feat_table = torch.cat(tables, dim=-2)  # [B,L,num_feat,D]
            if feat_table.dim() != 4 or feat_table.size(-1) != self.hidden_size:
                if not self._warned_feature_once:
                    warnings.warn(
                        f"[DIFF] Feature table shape invalid: got {tuple(feat_table.shape)}, expect [B,L,num_feat,{self.hidden_size}]. "
                        f"Fallback to ID-only."
                    )
                    self._warned_feature_once = True
                return []

            num_feat = feat_table.size(-2)
            if num_feat != self.num_attrs:
                # minimal assumption: selected_features count should match returned features
                if not self._warned_feature_once:
                    warnings.warn(
                        f"[DIFF] num features mismatch: selected_features={self.num_attrs}, returned={num_feat}. "
                        f"Fallback to first min(num_feat, selected_features)."
                    )
                    self._warned_feature_once = True
            num_use = min(num_feat, self.num_attrs)
            attrs = [feat_table[:, :, i, :] for i in range(num_use)]  # each [B,L,D]
            if num_use < self.num_attrs:
                # pad missing features with zeros (keeps model shape stable) but with warning once
                if not self._warned_feature_once:
                    warnings.warn(
                        f"[DIFF] Missing some selected features at runtime: use={num_use}/{self.num_attrs}. "
                        f"Pad rest with zeros."
                    )
                    self._warned_feature_once = True
                zero = torch.zeros_like(attrs[0])
                for _ in range(self.num_attrs - num_use):
                    attrs.append(zero)
            return attrs
        except Exception as e:
            if not self._warned_feature_once:
                warnings.warn(f"[DIFF] Feature extraction failed ({repr(e)}). Fallback to ID-only.")
                self._warned_feature_once = True
            return []

    def _early_fuse_embeddings(self, e_v: torch.Tensor, e_attrs: List[torch.Tensor]) -> torch.Tensor:
        # paper: Fusion can be sum/concat/gate; configurable
        if len(e_attrs) == 0:
            return e_v
        return self.early_fusion([e_v] + e_attrs)

    def _build_align_labels(self, attr_ids: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        attr_ids: [B,L] (fused attribute id, use the first attribute field by default if multiple;
                 for multi-attrs we use a hash-like combine: sum of ids with different primes)
        valid_mask: [B,L] bool, True for non-padding positions
        return Y: [B,L,L] float {0,1}
        """
        # equality in ID space
        eq = attr_ids.unsqueeze(-1).eq(attr_ids.unsqueeze(-2))  # [B,L,L]
        Y = eq.to(torch.float32)
        if valid_mask is not None:
            vm = valid_mask.to(torch.float32)
            Y = Y * (vm.unsqueeze(-1) * vm.unsqueeze(-2))
        return Y

    def _alignment_loss(
        self,
        e_v_raw: torch.Tensor,     # [B,L,D] before filtering (paper uses E_v and E_a)
        e_attrs_raw: List[torch.Tensor],  # m*[B,L,D]
        item_seq: torch.Tensor,    # [B,L] for padding mask
        attr_ids_for_label: Optional[torch.Tensor] = None,  # [B,L] fused attribute ids for label if available
    ) -> torch.Tensor:
        if self.lambda_align <= 0.0 or len(e_attrs_raw) == 0:
            return e_v_raw.new_zeros(())

        # fuse attributes for alignment (paper: summation)
        e_a_raw = self.attr_fusion_for_align(e_attrs_raw)  # [B,L,D]

        # normalize for stable training
        ev = F.normalize(e_v_raw, dim=-1)
        ea = F.normalize(e_a_raw, dim=-1)

        # similarity: [B,L,L]
        # softmax on last dim
        logits_va = torch.matmul(ev, ea.transpose(-2, -1)) / self.tau
        logits_av = torch.matmul(ea, ev.transpose(-2, -1)) / self.tau

        p_va = torch.softmax(logits_va, dim=-1)
        p_av = torch.softmax(logits_av, dim=-1)

        valid_mask = (item_seq > 0) if self.align_loss_on_valid else None

        # label Y
        if self.align_label_mode == "embed_equal":
            # strict float equality is brittle; keep as minimal option for fidelity, but default is attr_id_equal
            # here we consider equality after argmax over features is not defined, so use (ea_j == ea_k) exact
            eq = (e_a_raw.unsqueeze(-2) == e_a_raw.unsqueeze(-3)).all(dim=-1)  # [B,L,L] bool
            Y = eq.to(torch.float32)
            if valid_mask is not None:
                vm = valid_mask.to(torch.float32)
                Y = Y * (vm.unsqueeze(-1) * vm.unsqueeze(-2))
        else:
            # attr_id_equal: build labels by fused attribute IDs
            if attr_ids_for_label is None:
                # no attribute IDs available in interaction; degrade to diagonal positives only
                B, L, _ = e_v_raw.shape
                Y = torch.eye(L, device=e_v_raw.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
                if valid_mask is not None:
                    vm = valid_mask.to(torch.float32)
                    Y = Y * (vm.unsqueeze(-1) * vm.unsqueeze(-2))
            else:
                Y = self._build_align_labels(attr_ids_for_label, valid_mask)

        # loss: -(1/2b) sum_i sum(Y * log P_va + Y * log P_av)
        # Use epsilon for stability.
        eps = 1e-12
        log_p_va = torch.log(p_va.clamp_min(eps))
        log_p_av = torch.log(p_av.clamp_min(eps))

        # normalize by batch and by number of positive entries to keep scale stable across lengths
        pos_cnt = Y.sum(dim=(-1, -2)).clamp_min(1.0)  # [B]
        loss_i = -0.5 * ((Y * log_p_va).sum(dim=(-1, -2)) + (Y * log_p_av).sum(dim=(-1, -2))) / pos_cnt
        return loss_i.mean()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Return final user representation [B,D].
        Must be stable across train/infer and not return aux values (per requirement).
        """
        # embeddings
        pos_emb = self._get_pos_emb(item_seq)  # [B,L,D]
        e_v_raw = self.item_embedding(item_seq)  # [B,L,D]

        e_attrs_raw = self._extract_attr_embeddings(item_seq)  # list of [B,L,D], maybe empty
        # early fused embedding (raw)
        e_va_raw = self._early_fuse_embeddings(e_v_raw, e_attrs_raw)

        # add pos emb (applied to each stream consistently)
        e_v_in = self.input_dropout(self.input_ln(e_v_raw + pos_emb))
        if len(e_attrs_raw) > 0:
            e_attrs_in = [self.input_dropout(self.input_ln(ea + pos_emb)) for ea in e_attrs_raw]
        else:
            e_attrs_in = []
        e_va_in = self.input_dropout(self.input_ln(e_va_raw + pos_emb))

        # frequency-based filtering (independent per stream)
        e_v = self.filter_v(e_v_in)
        e_va = self.filter_va(e_va_in)
        if len(e_attrs_in) > 0:
            e_attrs = [self.filter_attrs[k](e_attrs_in[k]) for k in range(self.num_attrs)]
        else:
            e_attrs = []

        # attention mask (causal)
        attn_mask = self.get_attention_mask(item_seq)  # [B,1,L,L], float mask with -inf

        # L layers dual fusion
        for layer in self.layers:
            if len(e_attrs) == 0:
                # no attrs: ID-centric degenerates to self-attn on ID only; keep e_va as e_v for stability
                e_v, e_va = layer(e_v, [e_v for _ in range(self.num_attrs)], e_va, attn_mask) if self.num_attrs > 0 else layer(e_v, [], e_va, attn_mask)
            else:
                e_v, e_va = layer(e_v, e_attrs, e_va, attn_mask)

        # final aggregation (paper Eq.15)
        r_u = self.alpha * e_v + (1.0 - self.alpha) * e_va  # [B,L,D]
        seq_output = self.gather_indexes(r_u, item_seq_len - 1)  # [B,D]
        return seq_output

    def calculate_loss(self, interaction):
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
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight  # [n_items,D]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B,n_items]
            rec_loss = self.loss_fct(logits, pos_items)

        # alignment loss only in training mode (aux branch controlled by self.training)
        if not self.training or self.lambda_align <= 0.0 or self.num_attrs == 0 or self.feature_embed_layer is None:
            return rec_loss

        # build attr ids for labels if possible (best-effort)
        # RecBole feature fields for items are not guaranteed to be present in interaction in seq form;
        # FeatureSeqEmbLayer reads from dataset side, so here we can only approximate labels via item_seq->feature ids
        # However FeatureSeqEmbLayer doesn't expose ids, so minimal faithful option:
        # - If interaction contains feature sequences with same names, use them.
        attr_ids = None
        if self.align_label_mode == "attr_id_equal":
            # try to fetch feature sequences from interaction by field name + "_list" pattern is not guaranteed;
            # so check direct field name first.
            ids_list = []
            for f in self.selected_features:
                if f in interaction:
                    t = interaction[f]
                    # expect [B,L]; otherwise ignore
                    if isinstance(t, torch.Tensor) and t.dim() == 2 and t.size(1) == item_seq.size(1):
                        ids_list.append(t)
            if len(ids_list) > 0:
                # fuse multiple attr ids to a single id for equality test
                # hash-like combine: sum(ids * prime_k) mod large
                primes = [1000003, 1000033, 1000037, 1000039, 1000081, 1000099]
                fused = ids_list[0].clone()
                for k in range(1, len(ids_list)):
                    fused = fused + ids_list[k] * primes[k % len(primes)]
                attr_ids = fused

        # compute alignment on raw embeddings (before filtering) as in paper Eq.16
        e_v_raw = self.item_embedding(item_seq)
        e_attrs_raw = self._extract_attr_embeddings(item_seq)  # might still fail; then align_loss=0
        align_loss = self._alignment_loss(e_v_raw, e_attrs_raw, item_seq, attr_ids_for_label=attr_ids)

        return rec_loss + self.lambda_align * align_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=-1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores