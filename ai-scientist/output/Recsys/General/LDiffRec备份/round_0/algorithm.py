# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com

r"""
LDiffRec
################################################
Reference:
    Wenjie Wang et al. "Diffusion Recommender Model." in SIGIR 2023.
    
This implementation focuses on L-DiffRec (Latent Diffusion Recommender Model),
which clusters items for dimension compression via multiple VAEs and conducts 
diffusion processes in the latent space.
"""

import enum
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers
import typing


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    Takes noisy latent and timestep as input, predicts clean latent.
    """
    
    def __init__(
        self,
        dims: typing.List[int],
        emb_size: int,
        time_type: str = "cat",
        act_func: str = "tanh",
        norm: bool = False,
        dropout: float = 0.5,
    ):
        super(DNN, self).__init__()
        self.dims = dims.copy()
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        if self.time_type == "cat":
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError(f"Unimplemented timestep embedding type {self.time_type}")
        
        self.mlp_layers = MLPLayers(
            layers=self.dims, dropout=0, activation=act_func, last_activation=False
        )
        self.drop = nn.Dropout(dropout)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        h = self.mlp_layers(h)
        return h


class VAEEncoder(nn.Module):
    """Variational encoder for compressing item interactions to latent space."""
    
    def __init__(self, input_dim: int, hidden_dims: typing.List[int], latent_dim: int, dropout: float = 0.5):
        super(VAEEncoder, self).__init__()
        
        layers = [input_dim] + hidden_dims
        self.encoder = MLPLayers(layers=layers, dropout=dropout, activation='tanh', last_activation=True)
        
        self.fc_mu = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, latent_dim)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Variational decoder for reconstructing item interactions from latent space."""
    
    def __init__(self, latent_dim: int, hidden_dims: typing.List[int], output_dim: int, dropout: float = 0.5):
        super(VAEDecoder, self).__init__()
        
        layers = [latent_dim] + hidden_dims + [output_dim]
        self.decoder = MLPLayers(layers=layers, dropout=dropout, activation='tanh', last_activation=False)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, z):
        return self.decoder(z)


class LDiffRec(GeneralRecommender, AutoEncoderMixin):
    """
    L-DiffRec: Latent Diffusion Recommender Model
    
    This model clusters items for dimension compression via multiple VAEs 
    and conducts diffusion processes in the latent space.
    """
    
    input_type = InputType.LISTWISE
    
    def __init__(self, config, dataset):
        super(LDiffRec, self).__init__(config, dataset)
        
        # Mean type configuration
        if config["mean_type"] == "x0":
            self.mean_type = ModelMeanType.START_X
        elif config["mean_type"] == "eps":
            self.mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError(f"Unimplemented mean type {config['mean_type']}")
        
        # Time-aware reweighting configuration
        self.time_aware = config["time_aware"]
        self.w_max = config["w_max"]
        self.w_min = config["w_min"]
        
        # Build history items
        self.build_histroy_items(dataset)
        
        # Noise schedule configuration
        self.noise_schedule = config["noise_schedule"]
        self.noise_scale = config["noise_scale"]
        self.noise_min = config["noise_min"]
        self.noise_max = config["noise_max"]
        self.steps = config["steps"]
        self.beta_fixed = config["beta_fixed"]
        
        # Model architecture configuration
        self.emb_size = config["embedding_size"]
        self.norm = config["norm"]
        self.mlp_act_func = config["mlp_act_func"]
        self.dropout = config["dropout_prob"]
        
        # Latent space configuration
        self.latent_dim = config["latent_dimension"]
        self.n_cate = config["n_cate"]  # Number of item categories/clusters
        self.vae_hidden_dims = config["vae_hidden_dims"]
        self.dnn_dims = config["dnn_dims"]
        
        # Loss weighting
        self.reweight = config["reweight"]
        self.lamda = config["lamda"]  # Weight for diffusion loss
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        
        # Sampling configuration
        self.sampling_noise = config["sampling_noise"]
        self.sampling_steps = config["sampling_steps"]
        assert self.sampling_steps <= self.steps, "Too many steps in inference."
        
        # Importance sampling history
        self.history_num_per_term = config["history_num_per_term"]
        self.Lt_history = torch.zeros(
            self.steps, self.history_num_per_term, dtype=torch.float64
        ).to(self.device)
        self.Lt_count = torch.zeros(self.steps, dtype=int).to(self.device)
        
        # Training counters
        self.update_count = 0
        self.update_count_vae = 0
        
        # Initialize item clustering
        self._init_clustering(dataset, config)
        
        # Build VAE encoders and decoders for each category
        self._build_vae_components()
        
        # Build diffusion DNN
        total_latent_dim = self.latent_dim * self.n_cate
        dnn_dims = [total_latent_dim] + self.dnn_dims + [total_latent_dim]
        self.dnn = DNN(
            dims=dnn_dims,
            emb_size=self.emb_size,
            time_type="cat",
            norm=self.norm,
            act_func=self.mlp_act_func,
            dropout=self.dropout,
        ).to(self.device)
        
        # Initialize diffusion coefficients
        if self.noise_scale != 0.0:
            self.betas = torch.tensor(self._get_betas(), dtype=torch.float64).to(self.device)
            if self.beta_fixed:
                self.betas[0] = 0.00001
            self._calculate_diffusion_coefficients()
    
    def _init_clustering(self, dataset, config):
        """Initialize item clustering for latent diffusion."""
        if self.n_cate == 1:
            # No clustering, use all items as one category
            self.category_idx = [torch.arange(self.n_items).to(self.device)]
            self.category_map = torch.zeros(self.n_items, dtype=torch.long).to(self.device)
            self.category_sizes = [self.n_items]
        else:
            # Try to load item embeddings for clustering
            try:
                item_emb_path = config["item_emb_path"]
                item_emb = np.load(item_emb_path)
                item_emb = torch.tensor(item_emb, dtype=torch.float32)
                
                # K-means clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_cate, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(item_emb.numpy())
                
                self.category_idx = []
                self.category_map = torch.zeros(self.n_items, dtype=torch.long).to(self.device)
                self.category_sizes = []
                
                for c in range(self.n_cate):
                    idx = torch.where(torch.tensor(cluster_labels) == c)[0].to(self.device)
                    self.category_idx.append(idx)
                    self.category_map[idx] = c
                    self.category_sizes.append(len(idx))
            except Exception:
                # Fallback: equal split
                items_per_cate = self.n_items // self.n_cate
                self.category_idx = []
                self.category_map = torch.zeros(self.n_items, dtype=torch.long).to(self.device)
                self.category_sizes = []
                
                for c in range(self.n_cate):
                    start = c * items_per_cate
                    end = start + items_per_cate if c < self.n_cate - 1 else self.n_items
                    idx = torch.arange(start, end).to(self.device)
                    self.category_idx.append(idx)
                    self.category_map[idx] = c
                    self.category_sizes.append(len(idx))
    
    def _build_vae_components(self):
        """Build VAE encoders and decoders for each category."""
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for c in range(self.n_cate):
            input_dim = self.category_sizes[c]
            
            encoder = VAEEncoder(
                input_dim=input_dim,
                hidden_dims=self.vae_hidden_dims,
                latent_dim=self.latent_dim,
                dropout=self.dropout
            ).to(self.device)
            
            decoder = VAEDecoder(
                latent_dim=self.latent_dim,
                hidden_dims=self.vae_hidden_dims[::-1],
                output_dim=input_dim,
                dropout=self.dropout
            ).to(self.device)
            
            self.encoders.append(encoder)
            self.decoders.append(decoder)
    
    def build_histroy_items(self, dataset):
        """Build history items with optional time-aware reweighting."""
        if not self.time_aware:
            super().build_histroy_items(dataset)
        else:
            inter_feat = copy.deepcopy(dataset.inter_feat)
            inter_feat.sort(dataset.time_field)
            user_ids = inter_feat[dataset.uid_field].numpy()
            item_ids = inter_feat[dataset.iid_field].numpy()
            
            values = np.zeros(len(inter_feat))
            row_num = dataset.user_num
            
            for uid in range(1, row_num + 1):
                uindex = np.argwhere(user_ids == uid).flatten()
                int_num = len(uindex)
                if int_num > 1:
                    weight = np.linspace(self.w_min, self.w_max, int_num)
                else:
                    weight = np.array([self.w_max])
                values[uindex] = weight
            
            history_len = np.zeros(row_num, dtype=np.int64)
            for row_id in user_ids:
                history_len[row_id] += 1
            
            max_inter_num = np.max(history_len)
            col_num = max_inter_num
            
            history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
            history_value = np.zeros((row_num, col_num))
            history_len[:] = 0
            
            for row_id, value, col_id in zip(user_ids, values, item_ids):
                if history_len[row_id] >= col_num:
                    continue
                history_matrix[row_id, history_len[row_id]] = col_id
                history_value[row_id, history_len[row_id]] = value
                history_len[row_id] += 1
            
            self.history_item_id = torch.LongTensor(history_matrix).to(self.device)
            self.history_item_value = torch.FloatTensor(history_value).to(self.device)
    
    def _get_betas(self):
        """Get beta schedule for diffusion process."""
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                alpha_bar = 1 - np.linspace(start, end, self.steps, dtype=np.float64)
                betas = []
                betas.append(1 - alpha_bar[0])
                for i in range(1, self.steps):
                    betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
                return np.array(betas)
        elif self.noise_schedule == "cosine":
            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = []
            for i in range(self.steps):
                t1 = i / self.steps
                t2 = (i + 1) / self.steps
                betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
            return np.array(betas)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {self.noise_schedule}")
    
    def _calculate_diffusion_coefficients(self):
        """Calculate coefficients for diffusion process."""
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        )
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]
        )
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """Extract values from a 1-D tensor for a batch of indices."""
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def encode(self, x):
        """Encode full interaction vector to latent space using category VAEs."""
        z_list = []
        mu_list = []
        logvar_list = []
        
        for c in range(self.n_cate):
            idx = self.category_idx[c]
            x_c = x[:, idx]
            
            mu, logvar = self.encoders[c](x_c)
            
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            
            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        z = torch.cat(z_list, dim=1)
        return z, mu_list, logvar_list
    
    def decode(self, z):
        """Decode latent to full interaction vector using category VAEs."""
        x_recon = torch.zeros(z.size(0), self.n_items).to(self.device)
        
        for c in range(self.n_cate):
            start = c * self.latent_dim
            end = (c + 1) * self.latent_dim
            z_c = z[:, start:end]
            
            x_c = self.decoders[c](z_c)
            idx = self.category_idx[c]
            x_recon[:, idx] = x_c
        
        return x_recon
    
    def q_sample(self, z_start, t, noise=None):
        """Forward diffusion: add noise to latent."""
        if noise is None:
            noise = torch.randn_like(z_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, z_start.shape) * z_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z_start.shape) * noise
        )
    
    def p_mean_variance(self, z_t, t):
        """Compute mean and variance for reverse diffusion step."""
        model_output = self.dnn(z_t, t)
        
        model_variance = self._extract_into_tensor(self.posterior_variance, t, z_t.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, z_t.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_zstart = model_output
        else:
            pred_zstart = (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, z_t.shape) * z_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, z_t.shape) * model_output
            )
        
        model_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, z_t.shape) * pred_zstart
            + self._extract_into_tensor(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_zstart": pred_zstart,
        }
    
    def p_sample(self, z_start):
        """Reverse diffusion sampling."""
        steps = self.sampling_steps
        
        if steps == 0:
            z_t = z_start
        else:
            t = torch.tensor([steps - 1] * z_start.shape[0]).to(z_start.device)
            z_t = self.q_sample(z_start, t)
        
        if self.noise_scale == 0.0:
            for i in range(self.steps - 1, -1, -1):
                t = torch.tensor([i] * z_t.shape[0]).to(z_start.device)
                z_t = self.dnn(z_t, t)
            return z_t
        
        for i in range(self.steps - 1, -1, -1):
            t = torch.tensor([i] * z_t.shape[0]).to(z_start.device)
            out = self.p_mean_variance(z_t, t)
            
            if self.sampling_noise and i > 0:
                noise = torch.randn_like(z_t)
                z_t = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise
            else:
                z_t = out["mean"]
        
        return z_t
    
    def sample_timesteps(self, batch_size, device, method="uniform"):
        """Sample timesteps for training."""
        if method == "importance":
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all = pt_all * 0.999 + 0.001 / len(pt_all)
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        else:
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()
            return t, pt
    
    def SNR(self, t):
        """Compute signal-to-noise ratio."""
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def calculate_loss(self, interaction):
        """Calculate combined VAE and diffusion loss."""
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)
        batch_size = x_start.size(0)
        
        # Encode to latent space
        z_start, mu_list, logvar_list = self.encode(x_start)
        
        # VAE loss
        self.update_count_vae += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update_count_vae / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap
        
        # KL divergence for each category
        kl_loss = 0.0
        for c in range(self.n_cate):
            mu = mu_list[c]
            logvar = logvar_list[c]
            kl_loss += -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        kl_loss = kl_loss * anneal / self.n_cate
        
        # Reconstruction loss through diffusion
        if self.noise_scale != 0.0:
            # Sample timesteps
            ts, pt = self.sample_timesteps(batch_size, self.device, "importance" if self.reweight else "uniform")
            noise = torch.randn_like(z_start)
            z_t = self.q_sample(z_start, ts, noise)
        else:
            ts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            pt = torch.ones(batch_size, device=self.device)
            z_t = z_start
        
        # Diffusion model output
        model_output = self.dnn(z_t, ts)
        
        # Target based on mean type
        if self.mean_type == ModelMeanType.START_X:
            target = z_start
        else:
            target = noise
        
        # MSE loss
        mse = mean_flat((target - model_output) ** 2)
        
        # Reweight loss
        if self.reweight and self.noise_scale != 0.0:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where(ts == 0, torch.ones_like(weight), weight)
            else:
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = torch.where(ts == 0, torch.ones_like(weight), weight)
            diff_loss = (weight * mse / pt).mean()
        else:
            diff_loss = mse.mean()
        
        # Update importance sampling history
        for t, loss in zip(ts, mse):
            t_idx = t.item()
            if self.Lt_count[t_idx] == self.history_num_per_term:
                self.Lt_history[t_idx, :-1] = self.Lt_history[t_idx, 1:].clone()
                self.Lt_history[t_idx, -1] = loss.detach()
            else:
                self.Lt_history[t_idx, self.Lt_count[t_idx]] = loss.detach()
                self.Lt_count[t_idx] += 1
        
        # Decode for reconstruction loss
        z_pred = model_output if self.mean_type == ModelMeanType.START_X else self._predict_zstart_from_eps(z_t, ts, model_output)
        x_recon = self.decode(z_pred)
        
        # Multinomial reconstruction loss
        recon_loss = -(F.log_softmax(x_recon, dim=1) * x_start).sum(1).mean()
        
        # Combined loss
        total_loss = recon_loss + kl_loss + self.lamda * diff_loss
        
        return total_loss
    
    def _predict_zstart_from_eps(self, z_t, t, eps):
        """Predict z_0 from z_t and predicted epsilon."""
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, z_t.shape) * z_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, z_t.shape) * eps
        )
    
    def forward(self, x):
        """Forward pass for inference."""
        # Encode
        z, _, _ = self.encode(x)
        
        # Diffusion reverse process
        z_recon = self.p_sample(z)
        
        # Decode
        x_recon = self.decode(z_recon)
        
        return x_recon
    
    def predict(self, interaction):
        """Predict scores for specific user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        x_start = self.get_rating_matrix(user)
        scores = self.forward(x_start)
        
        return scores[torch.arange(len(item)).to(self.device), item]
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items."""
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)
        scores = self.forward(x_start)
        
        return scores.view(-1)