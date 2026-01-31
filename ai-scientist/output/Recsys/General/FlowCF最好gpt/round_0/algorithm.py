# -*- coding: utf-8 -*-
# @Time   : 2026/01/12
# @Author : OpenAI (Generated)
# @Email  : -
r"""
FlowCF
################################################
Paper:
    "FlowCF_2025" (KDD 2025)

Core ideas implemented:
1) Behavior-Guided Prior: X0 ~ Bernoulli(f), where f_i = (#interactions of item i) / |U|
2) Discrete flow training with discretized linear interpolation:
       Xt = Mt ⊙ X1 + (1-Mt) ⊙ X0,  Mt_ij ~ Bernoulli(t)
   And optimize simplified MSE:
       L = E[ || fθ(Xt, t) - X1 ||^2 ]
3) Inference: start from observed binary interactions X (as Xt),
   iterate a few Euler-like discrete updates using predicted vector field:
       v_t = (Xhat - Xt) / (1-t)
       Xt+1/N = 1{ Xt + v_t/N >= 0.5 }
   preserve observed positives: Xt = Xt OR X
   last step output Xhat = fθ(Xt, t_{N-1})

Notes:
- This is a LISTWISE generative model on full user-item vectors.
- We keep Xt binary during training/inference; fθ outputs real-valued scores (logits-like).
- For evaluation, full_sort_predict returns dense scores for all items.

Config keys expected (YAML):
- steps: int, discretization steps N (Eq.8), e.g., 1000
- sampling_steps: int, number of inference steps to run (<= steps). paper suggests 2.
- start_step: int or null, starting step index s. if null, we set s = steps - sampling_steps - 1
- embedding_size: int, timestep embedding dim
- dims_dnn: list[int], hidden dims for MLP
- mlp_act_func: str, activation in MLPLayers, e.g., 'tanh'/'relu'
- dropout: float, dropout on input before MLP
- norm: bool, whether to L2-normalize input vector before MLP
- prior_min: float, clamp lower bound for prior probabilities f (avoid all-zero), e.g., 1e-6
- prior_max: float, clamp upper bound for prior probabilities f, e.g., 1-1e-6
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils.enum_type import InputType


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    r"""Sinusoidal timestep embeddings (same as common diffusion impls).

    Args:
        timesteps: 1-D tensor of shape [B], can be int indices or floats.
        dim: embedding dimension.
        max_period: controls minimum frequency.
    Returns:
        Tensor of shape [B, dim]
    """
    # Ensure float
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )  # [half]
    args = timesteps[:, None] * freqs[None]  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FlowMLP(nn.Module):
    r"""Flow model f_theta(X_t, t) -> \hat{X}_1.

    Implementation follows the baseline DiffRec style:
    - sinusoidal timestep embedding -> linear projection
    - concatenate with input Xt
    - MLP outputs a dense vector of size n_items
    """

    def __init__(
        self,
        n_items: int,
        hidden_dims,
        time_emb_dim: int,
        act_func: str = "tanh",
        dropout: float = 0.1,
        norm: bool = False,
    ):
        super().__init__()
        self.n_items = n_items
        self.time_emb_dim = time_emb_dim
        self.norm = norm

        self.time_mlp = nn.Linear(time_emb_dim, time_emb_dim)
        mlp_layers = [n_items + time_emb_dim] + list(hidden_dims) + [n_items]
        self.mlp = MLPLayers(layers=mlp_layers, dropout=0.0, activation=act_func, last_activation=False)
        self.drop = nn.Dropout(dropout)

        self.apply(xavier_normal_initialization)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_items] float tensor (we pass binary 0/1 as float)
            timesteps: [B] tensor, can be integer step indices in [0, N-1]
        Returns:
            [B, n_items] prediction of target interactions \hat{X}_1
        """
        if self.norm:
            x = F.normalize(x, p=2, dim=-1)
        x = self.drop(x)

        t_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x, t_emb], dim=-1)
        out = self.mlp(h)
        return out


class FlowCF(GeneralRecommender, AutoEncoderMixin):
    r"""FlowCF (flow matching for collaborative filtering)"""

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Required config
        self.steps: int = config["steps"]  # N in paper
        self.sampling_steps: int = config["sampling_steps"]  # how many Euler updates during inference
        self.start_step = config.get("start_step", None)  # can be None

        self.embedding_size: int = config["embedding_size"]
        self.dims_dnn = config["dims_dnn"]
        self.mlp_act_func: str = config.get("mlp_act_func", "tanh")
        self.dropout: float = config.get("dropout", 0.1)
        self.norm: bool = config.get("norm", False)

        # Prior clamp for numerical stability / avoid trivial all-zeros prior
        self.prior_min: float = config.get("prior_min", 1e-6)
        self.prior_max: float = config.get("prior_max", 1.0 - 1e-6)

        assert self.steps >= 2, "steps (N) must be >= 2"
        assert 0 <= self.sampling_steps <= self.steps - 1, "sampling_steps must be in [0, steps-1]"

        self.build_histroy_items(dataset)

        # Build flow MLP
        self.flow = FlowMLP(
            n_items=self.n_items,
            hidden_dims=self.dims_dnn,
            time_emb_dim=self.embedding_size,
            act_func=self.mlp_act_func,
            dropout=self.dropout,
            norm=self.norm,
        ).to(self.device)

        # Behavior-guided prior probabilities f \in R^{|I|}
        # f_i = (# interactions for item i) / |U|
        self.register_buffer("item_freq", self._compute_item_frequency(dataset))  # [n_items]

        # Cache for evaluation
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        self.apply(xavier_normal_initialization)

    def _compute_item_frequency(self, dataset) -> torch.Tensor:
        """Compute global item frequency f as Eq.(9).

        Returns:
            f: float tensor [n_items] on self.device
        """
        inter_feat = dataset.inter_feat
        iid_field = dataset.iid_field
        item_ids = inter_feat[iid_field].to(torch.long)  # usually 1..n_items-1 in RecBole

        # Count interactions per item id
        counts = torch.bincount(item_ids, minlength=self.n_items).float()  # [n_items]
        # Divide by number of users |U| (paper Eq.9)
        f = counts / float(self.n_users)

        # Clamp probabilities into (0,1)
        f = torch.clamp(f, min=self.prior_min, max=self.prior_max)
        return f.to(self.device)

    # -------- Discrete flow utilities --------
    def _sample_t_indices(self, batch_size: int) -> torch.Tensor:
        """Sample discrete step indices i, corresponding to t_i=i/N (Eq.8).

        For training we avoid t=1 because (1-t) appears in vector field definition,
        and interpolation at t=1 trivially equals X1.
        """
        # sample i in [0, N-1]
        return torch.randint(low=0, high=self.steps, size=(batch_size,), device=self.device).long()

    def _t_from_indices(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Convert integer indices i to t=i/N in [0,1]."""
        return t_idx.float() / float(self.steps)

    def _sample_behavior_guided_prior(self, batch_size: int) -> torch.Tensor:
        """Sample X0 ~ Bernoulli(1 ⊗ f) (Eq.11) for a batch of users.

        Returns:
            X0: float tensor [B, n_items], entries in {0,1}
        """
        # Expand f to [B, n_items]
        probs = self.item_freq.unsqueeze(0).expand(batch_size, -1)
        # Bernoulli sampling
        x0 = torch.bernoulli(probs)
        return x0

    def _discrete_interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Discretized linear interpolation (Eq.13) using element-wise Bernoulli mask Mt~Bernoulli(t).

        Args:
            x0: [B, n_items] in {0,1}
            x1: [B, n_items] in {0,1}
            t:  [B] in [0,1]
        Returns:
            xt: [B, n_items] in {0,1}
        """
        # Sample mask Mt_ij ~ Bernoulli(t_i) (broadcast t to [B,1])
        probs = t.view(-1, 1).expand_as(x1)
        mt = torch.bernoulli(probs)
        xt = mt * x1 + (1.0 - mt) * x0
        return xt

    # -------- Core forward/loss/predict --------
    def forward(self, x_t: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Predict \hat{X}_1 = f_theta(X_t, t) (Eq.21)."""
        return self.flow(x_t, t_idx)

    def calculate_loss(self, interaction):
        """Algorithm 1: training loss (Eq.20): MSE between predicted and true interactions."""
        self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user).float()  # [B, n_items], 0/1

        batch_size = x1.size(0)

        # Sample discrete timestep indices and convert to scalar t (for interpolation only)
        t_idx = self._sample_t_indices(batch_size)  # [B]
        t = self._t_from_indices(t_idx)  # [B] in [0,1]

        # Sample behavior-guided prior and interpolate discretely
        x0 = self._sample_behavior_guided_prior(batch_size)  # [B, n_items] {0,1}
        x_t = self._discrete_interpolate(x0, x1, t)  # [B, n_items] {0,1}

        # Predict target interactions
        x1_hat = self.forward(x_t, t_idx)  # [B, n_items], real

        # Eq.(20): MSE to true interaction vector (x1 is binary)
        loss = F.mse_loss(x1_hat, x1, reduction="mean")
        return loss

    def _infer_start_step(self) -> int:
        """Select starting step index s for Algorithm 2."""
        if self.sampling_steps == 0:
            return self.steps - 1

        if self.start_step is not None:
            s = int(self.start_step)
        else:
            # If we plan to run K updates for i=s..N-2 inclusive, number of iterations is (N-1 - s).
            # Set s so that (N-1 - s) == sampling_steps  => s = N-1 - sampling_steps
            # But loop in Algorithm 2: i=s..N-2, iterations = (N-1 - s)
            s = (self.steps - 1) - self.sampling_steps

        s = max(0, min(s, self.steps - 2))  # must be <= N-2
        return s

    @torch.no_grad()
    def _flowcf_sample(self, x_obs: torch.Tensor) -> torch.Tensor:
        """Algorithm 2 sampling from observed interactions.

        Args:
            x_obs: [B, n_items] binary 0/1 float tensor (observed history)
        Returns:
            x_hat: [B, n_items] predicted scores (real-valued) in the last step
        """
        # Initialize X_t as observed interactions (paper: treat as noisy X_t)
        x_t = x_obs.clone()

        # Determine start step s
        s = self._infer_start_step()
        N = self.steps

        # Iterative updates until step N-2, then final predict at t_{N-1}
        for i in range(s, N - 1):  # i = s ... N-2
            t_idx = torch.full((x_t.size(0),), i, device=x_t.device, dtype=torch.long)
            t = float(i) / float(N)

            x_hat = self.forward(x_t, t_idx)  # \hat{X} in Algorithm 2

            # Vector field v_t = (\hat{X} - X_t)/(1-t) (Eq.18; using X_t as proxy for E[X_t])
            denom = max(1e-6, 1.0 - t)
            v_t = (x_hat - x_t) / denom

            # Discrete Euler-like update (Eq.22): X_{t+1/N} = 1{ X_t + v_t/N >= 0.5 }
            x_next = (x_t + v_t / float(N) >= 0.5).float()

            # Preserve observed interactions (Algorithm 2 line 8): X_t <- X_t OR X
            x_t = torch.logical_or(x_next.bool(), x_obs.bool()).float()

        # Last step prediction at t_{N-1}
        last_idx = torch.full((x_t.size(0),), N - 1, device=x_t.device, dtype=torch.long)
        x_hat = self.forward(x_t, last_idx)
        return x_hat

    def full_sort_predict(self, interaction):
        """Return scores for all items for each user (dense)."""
        user = interaction[self.USER_ID]
        x_obs = self.get_rating_matrix(user).float()  # [B, n_items]

        # Optionally cache in evaluation
        # (Note: caching is safe only if called once per user batch without parameter update)
        x_hat = self._flowcf_sample(x_obs)  # [B, n_items]
        return x_hat

    def predict(self, interaction):
        """Return scores for specific (user,item) pairs."""
        item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)  # [B, n_items]
        return scores[torch.arange(len(item), device=self.device), item]