# -*- coding: utf-8 -*-
# DIFF_2025: Dual Side-Information Filtering and Fusion (DIFF)
#
# Paper core ideas implemented:
# 1) Frequency-based noise filtering for ID / each attribute / early-fused embeddings via FFT:
#       E_tilde = IDFT(LFC) + beta * IDFT(HFC)
# 2) Dual multi-sequence fusion in each layer:
#    - ID-centric (intermediate fusion): fuse attention score matrices from ID and attributes, apply to ID value
#    - Attribute-enriched (early fusion): standard self-attention on early-fused embeddings
#    - Final user sequence representation: Ru = alpha * Rv + (1-alpha) * Rva; use last valid position
# 3) Representation alignment loss between ID embeddings and fused-attribute embeddings (sum fusion) with
#    temperature-scaled similarity matching.
#
# This implementation follows RecBole's SequentialRecommender interface style.

import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


def _gelu(x):
    return F.gelu(x)


class PositionwiseFFN(nn.Module):
    """Transformer-style FFN: Linear -> Act -> Dropout -> Linear -> Dropout, with residual+LayerNorm outside."""

    def __init__(self, hidden_size: int, inner_size: int, dropout: float, act: str = "gelu", layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act_name = act
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dense_1(x)
        if self.act_name == "gelu":
            x = _gelu(x)
        elif self.act_name == "relu":
            x = F.relu(x, inplace=False)
        elif self.act_name == "swish":
            x = x * torch.sigmoid(x)
        else:
            raise NotImplementedError(f"Unsupported hidden_act={self.act_name}")
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class DiffMHA(nn.Module):
    """
    Multi-head attention implemented explicitly for DIFF:
    - Uses causal/padding mask from RecBole (shape [B,1,L,L] with 0 or -10000)
    - Provides two modes:
        (a) attribute-enriched: standard attention on early-fused embeddings
        (b) ID-centric: fuse attention logits from ID and multiple attribute streams, then apply to ID value stream
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float,
        hidden_dropout: float,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads

        # Output projection (after concat heads)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, heads, L, d_head]
        b, l, _ = x.size()
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B, heads, L, d_head] -> [B, L, H]
        b, h, l, d = x.size()
        return x.transpose(1, 2).contiguous().view(b, l, h * d)

    def forward_enriched(
        self,
        x: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard multi-head self-attention over x.
        """
        residual = x
        q = self._shape(q_proj(x))
        k = self._shape(k_proj(x))
        v = self._shape(v_proj(x))

        # [B, heads, L, L]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_logits = attn_logits + attn_mask  # broadcast over heads

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v)  # [B, heads, L, d]
        ctx = self._merge(ctx)
        ctx = self.out_proj(ctx)
        ctx = self.hidden_dropout(ctx)
        out = self.layer_norm(ctx + residual)
        return out

    def forward_id_centric(
        self,
        e_v: torch.Tensor,
        e_attrs: List[torch.Tensor],
        qv_proj: nn.Linear,
        kv_proj: nn.Linear,
        vv_proj: nn.Linear,
        qa_projs: List[nn.Linear],
        ka_projs: List[nn.Linear],
        attn_mask: torch.Tensor,
        score_fusion: str = "sum",
        gate: Optional[nn.Parameter] = None,
    ) -> torch.Tensor:
        """
        ID-centric attention:
        - Queries/Keys from ID and each attribute stream -> attention score matrices
        - Fuse score matrices -> softmax -> apply to ID Value
        """
        residual = e_v

        qv = self._shape(qv_proj(e_v))
        kv = self._shape(kv_proj(e_v))
        vv = self._shape(vv_proj(e_v))

        score_list = []
        score_v = torch.matmul(qv, kv.transpose(-2, -1))  # [B, heads, L, L]
        score_list.append(score_v)

        for e_a, qa, ka in zip(e_attrs, qa_projs, ka_projs):
            q = self._shape(qa(e_a))
            k = self._shape(ka(e_a))
            score_a = torch.matmul(q, k.transpose(-2, -1))
            score_list.append(score_a)

        if score_fusion == "sum":
            fused_scores = torch.zeros_like(score_list[0])
            for s in score_list:
                fused_scores = fused_scores + s
        elif score_fusion == "mean":
            fused_scores = torch.stack(score_list, dim=0).mean(dim=0)
        elif score_fusion == "gate":
            # simple learnable scalar gate per source; gate shape [num_sources]
            # gate passed as nn.Parameter; apply sigmoid for stability
            if gate is None:
                raise ValueError("score_fusion='gate' requires gate parameter.")
            w = torch.sigmoid(gate)  # [num_sources]
            # normalize weights to sum=1
            w = w / (w.sum() + 1e-12)
            fused_scores = 0.0
            for idx, s in enumerate(score_list):
                fused_scores = fused_scores + w[idx] * s
        else:
            raise NotImplementedError(f"Unsupported score_fusion={score_fusion}")

        fused_scores = fused_scores / math.sqrt(self.d_head)
        fused_scores = fused_scores + attn_mask

        attn = torch.softmax(fused_scores, dim=-1)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, vv)  # [B, heads, L, d]
        ctx = self._merge(ctx)
        ctx = self.out_proj(ctx)
        ctx = self.hidden_dropout(ctx)
        out = self.layer_norm(ctx + residual)
        return out


class DiffBlock(nn.Module):
    """One DIFF layer: (1) Dual multi-sequence fusion attentions; (2) FFN for both branches."""

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        n_heads: int,
        hidden_dropout: float,
        attn_dropout: float,
        hidden_act: str,
        layer_norm_eps: float,
        n_attrs: int,
        score_fusion: str = "sum",
    ):
        super().__init__()
        self.n_attrs = n_attrs
        self.score_fusion = score_fusion

        self.mha_id = DiffMHA(hidden_size, n_heads, attn_dropout, hidden_dropout, layer_norm_eps)
        self.mha_va = DiffMHA(hidden_size, n_heads, attn_dropout, hidden_dropout, layer_norm_eps)

        # Projections for ID-centric branch
        self.qv = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, hidden_size)
        self.vv = nn.Linear(hidden_size, hidden_size)

        self.qa = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_attrs)])
        self.ka = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_attrs)])

        # Projections for VA (early-fused) branch
        self.qva = nn.Linear(hidden_size, hidden_size)
        self.kva = nn.Linear(hidden_size, hidden_size)
        self.vva = nn.Linear(hidden_size, hidden_size)

        # Optional gate weights for score fusion
        if self.score_fusion == "gate":
            # sources = ID + n_attrs
            self.score_gate = nn.Parameter(torch.zeros(1 + n_attrs))
        else:
            self.score_gate = None

        self.ffn_id = PositionwiseFFN(hidden_size, inner_size, hidden_dropout, hidden_act, layer_norm_eps)
        self.ffn_va = PositionwiseFFN(hidden_size, inner_size, hidden_dropout, hidden_act, layer_norm_eps)

    def forward(
        self,
        e_v: torch.Tensor,
        e_attrs: List[torch.Tensor],
        e_va: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        # ID-centric: fuse score matrices from ID and attrs, apply to ID value
        e_v = self.mha_id.forward_id_centric(
            e_v=e_v,
            e_attrs=e_attrs,
            qv_proj=self.qv,
            kv_proj=self.kv,
            vv_proj=self.vv,
            qa_projs=list(self.qa),
            ka_projs=list(self.ka),
            attn_mask=attn_mask,
            score_fusion=self.score_fusion,
            gate=self.score_gate,
        )
        e_v = self.ffn_id(e_v)

        # Attribute-enriched: standard attention over early-fused embeddings
        e_va = self.mha_va.forward_enriched(
            x=e_va,
            q_proj=self.qva,
            k_proj=self.kva,
            v_proj=self.vva,
            attn_mask=attn_mask,
        )
        e_va = self.ffn_va(e_va)

        return e_v, e_va


class DIFF(SequentialRecommender):
    # NOTE:
    # RecBole's Trainer expects config['clip_grad_norm'] to be a mapping (dict) when it calls:
    #   clip_grad_norm_(model.parameters(), **self.clip_grad_norm)
    # Some benchmarks provide a float (e.g., 1.0), which would crash with:
    #   TypeError: argument after ** must be a mapping, not float
    #
    # We defensively normalize it here to keep compatibility with such configs.
    @staticmethod
    def _clip_grad_norm_fix_hook(config):
        try:
            if "clip_grad_norm" in config and isinstance(config["clip_grad_norm"], (int, float)):
                config["clip_grad_norm"] = {"max_norm": float(config["clip_grad_norm"])}
        except Exception:
            # don't block model construction if config is a special object
            pass
    """
    RecBole-style DIFF model.

    Expected config keys:
        - n_layers, n_heads, hidden_size, inner_size
        - hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, initializer_range
        - loss_type: 'CE' or 'BPR'
        - attribute_fields: list of dataset feature field names for item attributes (categorical, int ids)
            e.g., ['brand_id', 'cate_id'] where interaction provides sequences for each field.
          For RecBole, item-side sequence field naming typically like: 'brand_id_list' etc.
          Here we assume the interaction contains each attribute sequence tensor directly with key in attribute_fields.
        - fusion_type: 'sum' | 'concat' | 'gate'   (for early fusion E_va)
        - score_fusion: 'sum' | 'mean' | 'gate'    (for intermediate fusion of attention scores)
        - alpha: float in [0,1]
        - freq_c: int cutoff c for low-frequency rows (along sequence length axis)
        - lambda_align: float
        - align_tau_init: float (temperature init)
    """

    def __init__(self, config, dataset):
        # fix potential RecBole Trainer config incompatibility before Trainer reads it
        self._clip_grad_norm_fix_hook(config)
        super().__init__(config, dataset)

        # ---- Transformer / general ----
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config.get("hidden_act", "gelu")
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        self.initializer_range = config.get("initializer_range", 0.02)

        self.loss_type = config.get("loss_type", "CE")
        if self.loss_type not in ["CE", "BPR"]:
            raise NotImplementedError("loss_type must be 'CE' or 'BPR'.")

        # ---- DIFF specific ----
        self.attribute_fields: List[str] = config["attribute_fields"]  # interaction keys for attribute sequences
        self.n_attrs = len(self.attribute_fields)

        self.fusion_type = config.get("fusion_type", "sum")  # for early fusion E_va
        self.score_fusion = config.get("score_fusion", "sum")  # for intermediate score fusion

        self.alpha = float(config.get("alpha", 0.5))
        self.freq_c = int(config.get("freq_c", max(1, self.max_seq_length // 4)))
        self.lambda_align = float(config.get("lambda_align", 0.1))

        # ---- Embeddings ----
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # Attribute vocab sizes: use dataset.field2token_id or num() depending on RecBole version.
        # Safer: dataset.num(field) exists for most RecBole datasets.
        self.attr_embeddings = nn.ModuleList()
        for f in self.attribute_fields:
            vocab_size = dataset.num(f)
            self.attr_embeddings.append(nn.Embedding(vocab_size, self.hidden_size, padding_idx=0))

        # ---- Early fusion module parameters ----
        if self.fusion_type == "concat":
            # concat (ID + m attrs) then project to d
            self.early_fuse_proj = nn.Linear(self.hidden_size * (1 + self.n_attrs), self.hidden_size)
        elif self.fusion_type == "gate":
            # gated sum: E_va = g * E_v + (1-g) * E_attr_sum
            self.early_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif self.fusion_type == "sum":
            self.early_fuse_proj = None
            self.early_gate = None
        else:
            raise NotImplementedError(f"Unsupported fusion_type={self.fusion_type}")

        # ---- Frequency filter parameters: beta_0..beta_{m+1} ----
        # beta[0]=ID, beta[1..m]=attrs, beta[m+1]=early-fused
        self.beta = nn.Parameter(torch.zeros(self.n_attrs + 2))

        # ---- DIFF blocks ----
        self.blocks = nn.ModuleList(
            [
                DiffBlock(
                    hidden_size=self.hidden_size,
                    inner_size=self.inner_size,
                    n_heads=self.n_heads,
                    hidden_dropout=self.hidden_dropout_prob,
                    attn_dropout=self.attn_dropout_prob,
                    hidden_act=self.hidden_act,
                    layer_norm_eps=self.layer_norm_eps,
                    n_attrs=self.n_attrs,
                    score_fusion=self.score_fusion,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.input_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)

        # ---- Alignment temperature (learnable) ----
        tau0 = float(config.get("align_tau_init", 0.2))
        # Keep positive via softplus during forward
        self._align_tau_unconstrained = nn.Parameter(torch.tensor(math.log(math.exp(tau0) - 1.0)))

        # ---- Loss ----
        if self.loss_type == "CE":
            self.rec_loss_fct = nn.CrossEntropyLoss()
        else:
            self.rec_loss_fct = BPRLoss()

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def align_tau(self) -> torch.Tensor:
        # positive temperature
        return F.softplus(self._align_tau_unconstrained) + 1e-8

    # ----------------------- Fusion / Filtering utilities -----------------------
    def _early_fusion(self, e_v: torch.Tensor, e_attrs: List[torch.Tensor]) -> torch.Tensor:
        """Fusion(E_v, E_a1..E_am) -> E_va."""
        if self.n_attrs == 0:
            return e_v

        if self.fusion_type == "sum":
            e = e_v
            for a in e_attrs:
                e = e + a
            return e
        elif self.fusion_type == "concat":
            x = torch.cat([e_v] + e_attrs, dim=-1)
            return self.early_fuse_proj(x)
        elif self.fusion_type == "gate":
            # attr summary uses summation
            e_a = 0.0
            for a in e_attrs:
                e_a = e_a + a
            g = torch.sigmoid(self.early_gate(torch.cat([e_v, e_a], dim=-1)))
            return g * e_v + (1.0 - g) * e_a
        else:
            raise NotImplementedError

    def _fft_filter(self, x: torch.Tensor, c: int, beta: torch.Tensor) -> torch.Tensor:
        """
        Frequency-based filtering for real sequence embeddings.
        x: [B, L, D]
        returns: [B, L, D] real
        """
        # FFT along sequence length dimension
        # Use orthonormal normalization to match paper's 1/sqrt(N)
        Xf = torch.fft.fft(x, dim=1, norm="ortho")  # complex [B,L,D]

        L = x.size(1)
        c_eff = max(1, min(int(c), L))  # ensure valid

        Xf_l = Xf[:, :c_eff, :]
        Xf_h = Xf[:, c_eff:, :]

        # IDFT back to time domain; keep same length by padding missing freqs with zeros
        # Approach: create masked spectrum and do ifft once for each component.
        mask_l = torch.zeros_like(Xf)
        mask_h = torch.zeros_like(Xf)
        mask_l[:, :c_eff, :] = Xf_l
        mask_h[:, c_eff:, :] = Xf_h

        xl = torch.fft.ifft(mask_l, dim=1, norm="ortho").real
        xh = torch.fft.ifft(mask_h, dim=1, norm="ortho").real

        return xl + beta * xh

    def _add_position_and_norm(self, x: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Standard SASRec-style input processing:
        - add position embedding
        - LayerNorm + Dropout
        """
        b, l = item_seq.size()
        pos_ids = torch.arange(l, device=item_seq.device, dtype=torch.long).unsqueeze(0).expand(b, l)
        pos_emb = self.position_embedding(pos_ids)
        x = x + pos_emb
        x = self.input_layer_norm(x)
        x = self.input_dropout(x)
        return x

    # ----------------------- Alignment loss -----------------------
    def _alignment_loss(
        self,
        e_v_raw: torch.Tensor,
        e_attrs_raw: List[torch.Tensor],
        attr_ids: List[torch.Tensor],
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements Eq (16-18) approximately in a practical way:
        - E_a is sum-fusion of attribute embeddings (paper: summation)
        - Normalize embeddings
        - Similarity matrices -> softmax with temperature
        - Y_{j,k}=1 if fused attribute IDs are identical; exclude padding positions.

        Inputs:
            e_v_raw: [B,L,D] (before FFT filtering, but after item embedding lookup)
            e_attrs_raw: list of [B,L,D]
            attr_ids: list of [B,L] original attribute ids aligned with sequences
            pad_mask: [B,L] True for valid positions (item_seq != 0)
        """
        if self.n_attrs == 0:
            return torch.zeros((), device=e_v_raw.device)

        # E_a by summation on embeddings, but Eq(18) uses equality of fused attribute *vectors*.
        # In practice, equality can be defined on fused attribute *IDs tuple*.
        e_a_raw = 0.0
        for a in e_attrs_raw:
            e_a_raw = e_a_raw + a

        # Normalize along embedding dim
        ev = F.normalize(e_v_raw, p=2, dim=-1)
        ea = F.normalize(e_a_raw, p=2, dim=-1)

        # Similarities: [B,L,L]
        sim_va = torch.matmul(ev, ea.transpose(1, 2)) / self.align_tau
        sim_av = torch.matmul(ea, ev.transpose(1, 2)) / self.align_tau

        # Softmax row-wise as in paper
        yhat_va = torch.softmax(sim_va, dim=-1)
        yhat_av = torch.softmax(sim_av, dim=-1)

        # Build Y based on equality of attribute id tuples at positions j,k
        # attr_ids: list of [B,L], create a combined code for tuple equality
        # Use a rolling hash-like combine (safe within 64-bit range for typical vocab sizes).
        # If only one attr, this is just that attr id.
        code = attr_ids[0].long()
        for t in attr_ids[1:]:
            code = code * 1000003 + t.long()  # large prime multiplier

        # Y[b,j,k] = 1 if code[b,j]==code[b,k] and both valid (not padding)
        # [B,L,1] == [B,1,L] -> [B,L,L]
        eq = (code.unsqueeze(-1) == code.unsqueeze(1)).float()

        # mask out padding rows/cols
        valid = pad_mask.float()
        mask_mat = valid.unsqueeze(-1) * valid.unsqueeze(1)  # [B,L,L]
        eq = eq * mask_mat

        # Cross-entropy with multi-positive targets:
        # L = -1/(2B) sum_i sum_{j,k} Y_{j,k} log yhat_{j,k} (both directions)
        eps = 1e-12
        loss_va = -(eq * torch.log(yhat_va + eps)).sum(dim=(1, 2))
        loss_av = -(eq * torch.log(yhat_av + eps)).sum(dim=(1, 2))

        # Normalize by number of valid pairs to stabilize (not specified, but practical)
        denom = mask_mat.sum(dim=(1, 2)).clamp_min(1.0)
        loss = 0.5 * ((loss_va / denom) + (loss_av / denom))
        return loss.mean()

    # ----------------------- Forward -----------------------
    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, attr_seq_list: Optional[List[torch.Tensor]] = None):
        """
        Returns seq_output: [B, H] representation at last valid position.
        If attr_seq_list is None, reads from interaction in calculate_loss/predict.
        """
        # item_seq: [B,L]
        pad_mask = item_seq.ne(0)  # [B,L]
        attn_mask = self.get_attention_mask(item_seq)  # [B,1,L,L], causal + padding

        # Embeddings
        e_v = self.item_embedding(item_seq)  # [B,L,D]
        e_attrs = []
        if attr_seq_list is None:
            # allow forward() used only by calculate_loss/predict which passes attr_seq_list
            attr_seq_list = []
        if len(attr_seq_list) != 0:
            for emb, ids in zip(self.attr_embeddings, attr_seq_list):
                e_attrs.append(emb(ids))
        else:
            # no attrs provided -> treat as zero
            e_attrs = [torch.zeros_like(e_v) for _ in range(self.n_attrs)]

        # Early fused embedding
        e_va = self._early_fusion(e_v, e_attrs)

        # Frequency-based filtering (independent for each stream)
        e_v_f = self._fft_filter(e_v, self.freq_c, self.beta[0])
        e_attrs_f = [self._fft_filter(e_a, self.freq_c, self.beta[i + 1]) for i, e_a in enumerate(e_attrs)]
        e_va_f = self._fft_filter(e_va, self.freq_c, self.beta[self.n_attrs + 1])

        # Add positional embeddings and normalize/dropout
        e_v_f = self._add_position_and_norm(e_v_f, item_seq)
        e_attrs_f = [self._add_position_and_norm(e_a, item_seq) for e_a in e_attrs_f]
        e_va_f = self._add_position_and_norm(e_va_f, item_seq)

        # L stacked DIFF blocks
        for blk in self.blocks:
            e_v_f, e_va_f = blk(e_v_f, e_attrs_f, e_va_f, attn_mask)

        # Final aggregated sequence representation Ru
        r_u = self.alpha * e_v_f + (1.0 - self.alpha) * e_va_f  # [B,L,D]
        out = self.gather_indexes(r_u, item_seq_len - 1)  # [B,D]
        return out

    # ----------------------- Loss / Predict -----------------------
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        # attribute sequences
        attr_seq_list = [interaction[f] for f in self.attribute_fields] if self.n_attrs > 0 else []

        seq_output = self.forward(item_seq, item_seq_len, attr_seq_list=attr_seq_list)

        # recommendation loss
        if self.loss_type == "CE":
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))  # [B, n_items]
            rec_loss = self.rec_loss_fct(logits, pos_items)
        else:
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = (seq_output * pos_emb).sum(dim=-1)
            neg_score = (seq_output * neg_emb).sum(dim=-1)
            rec_loss = self.rec_loss_fct(pos_score, neg_score)

        # alignment loss uses raw embeddings (no position, no FFT) per paper Eq(16-18)
        if self.lambda_align > 0 and self.n_attrs > 0:
            pad_mask = item_seq.ne(0)

            e_v_raw = self.item_embedding(item_seq)
            e_attrs_raw = [emb(interaction[f]) for emb, f in zip(self.attr_embeddings, self.attribute_fields)]
            attr_ids = [interaction[f] for f in self.attribute_fields]

            align_loss = self._alignment_loss(e_v_raw, e_attrs_raw, attr_ids, pad_mask)
            loss = rec_loss + self.lambda_align * align_loss
        else:
            loss = rec_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        attr_seq_list = [interaction[f] for f in self.attribute_fields] if self.n_attrs > 0 else []

        seq_output = self.forward(item_seq, item_seq_len, attr_seq_list=attr_seq_list)
        test_item_emb = self.item_embedding(test_item)
        scores = (seq_output * test_item_emb).sum(dim=-1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        attr_seq_list = [interaction[f] for f in self.attribute_fields] if self.n_attrs > 0 else []

        seq_output = self.forward(item_seq, item_seq_len, attr_seq_list=attr_seq_list)
        scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return scores