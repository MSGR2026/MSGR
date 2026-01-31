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


class ModelMeanType(enum.Enum):
    """Enum for model prediction target type"""
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element. (N,)
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    """Convert linear variance schedule to betas."""
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    Takes noisy latent and timestep embedding as input, predicts x_0.
    """
    
    def __init__(self, dims, emb_size, time_type="cat", act_func="tanh", norm=False, dropout=0.5):
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
        
        # Build MLP layers
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i < len(self.dims) - 2:
                if act_func == "tanh":
                    layers.append(nn.Tanh())
                elif act_func == "relu":
                    layers.append(nn.ReLU())
                elif act_func == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.Tanh())
        
        self.mlp_layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        h = self.mlp_layers(h)
        return h


class Encoder(nn.Module):
    """VAE Encoder for compressing interaction vectors to latent space."""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder for reconstructing interaction vectors from latent space."""
    
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(xavier_normal_initialization)
    
    def forward(self, z):
        z = self.dropout(z)
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
        
        # Model configuration
        self.mean_type = ModelMeanType.START_X if config["mean_type"] == "x0" else ModelMeanType.EPSILON
        
        # Time-aware reweighting
        self.time_aware = config["time_aware"]
        self.w_max = config["w_max"]
        self.w_min = config["w_min"]
        
        # Build history items
        self.build_histroy_items(dataset)
        
        # Noise schedule parameters
        self.noise_schedule = config["noise_schedule"]
        self.noise_scale = config["noise_scale"]
        self.noise_min = config["noise_min"]
        self.noise_max = config["noise_max"]
        self.steps = config["steps"]
        self.beta_fixed = config["beta_fixed"]
        
        # Model architecture parameters
        self.emb_size = config["embedding_size"]
        self.norm = config["norm"]
        self.reweight = config["reweight"]
        if self.noise_scale == 0.0:
            self.reweight = False
        
        # Inference parameters
        self.sampling_noise = config["sampling_noise"]
        self.sampling_steps = config["sampling_steps"]
        assert self.sampling_steps <= self.steps, "Too many steps in inference."
        
        # Importance sampling history
        self.history_num_per_term = config["history_num_per_term"]
        self.Lt_history = torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64).to(self.device)
        self.Lt_count = torch.zeros(self.steps, dtype=int).to(self.device)
        
        # VAE parameters
        self.n_cate = config["n_cate"]  # Number of item clusters
        self.latent_dim = config["latent_dimension"]
        self.encoder_dims = config["encoder_dims"]
        self.mlp_act_func = config["mlp_act_func"]
        self.dropout = config["dropout_prob"]
        
        # VAE annealing
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.update_count_vae = 0
        
        # Diffusion loss weight
        self.lamda = config["lamda"]
        
        # Initialize item clustering
        self._init_item_clusters(config, dataset)
        
        # Build VAE encoders and decoders for each cluster
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for c in range(self.n_cate):
            cluster_size = len(self.category_idx[c])
            encoder = Encoder(
                input_dim=cluster_size,
                hidden_dims=self.encoder_dims,
                latent_dim=self.latent_dim,
                dropout=self.dropout
            )
            decoder = Decoder(
                latent_dim=self.latent_dim,
                hidden_dims=self.encoder_dims,
                output_dim=cluster_size,
                dropout=self.dropout
            )
            self.encoders.append(encoder)
            self.decoders.append(decoder)
        
        # Build diffusion DNN
        total_latent_dim = self.n_cate * self.latent_dim
        dnn_dims = [total_latent_dim] + config["dims_dnn"] + [total_latent_dim]
        
        self.dnn = DNN(
            dims=dnn_dims,
            emb_size=self.emb_size,
            time_type="cat",
            norm=self.norm,
            act_func=self.mlp_act_func,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize diffusion coefficients
        if self.noise_scale != 0.0:
            self.betas = torch.tensor(self._get_betas(), dtype=torch.float64).to(self.device)
            if self.beta_fixed:
                self.betas[0] = 0.00001
            self._calculate_diffusion_coefficients()
    
    def _init_item_clusters(self, config, dataset):
        """Initialize item clusters using k-means or random assignment."""
        if self.n_cate == 1:
            # Single cluster: all items in one group
            self.category_idx = [list(range(self.n_items))]
            self.category_map = {i: (0, i) for i in range(self.n_items)}
        else:
            # Try to load item embeddings for clustering
            try:
                item_emb_path = config["item_emb_path"]
                item_emb = np.load(item_emb_path)
                
                # K-means clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_cate, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(item_emb)
                
                self.category_idx = [[] for _ in range(self.n_cate)]
                self.category_map = {}
                
                for item_id, cluster_id in enumerate(cluster_labels):
                    local_idx = len(self.category_idx[cluster_id])
                    self.category_idx[cluster_id].append(item_id)
                    self.category_map[item_id] = (cluster_id, local_idx)
                    
            except (KeyError, FileNotFoundError):
                # Random clustering if embeddings not available
                items_per_cluster = self.n_items // self.n_cate
                self.category_idx = []
                self.category_map = {}
                
                for c in range(self.n_cate):
                    start_idx = c * items_per_cluster
                    if c == self.n_cate - 1:
                        end_idx = self.n_items
                    else:
                        end_idx = (c + 1) * items_per_cluster
                    
                    cluster_items = list(range(start_idx, end_idx))
                    self.category_idx.append(cluster_items)
                    
                    for local_idx, item_id in enumerate(cluster_items):
                        self.category_map[item_id] = (c, local_idx)
    
    def build_histroy_items(self, dataset):
        """Build history items with optional time-aware reweighting."""
        if not self.time_aware:
            super().build_histroy_items(dataset)
        else:
            inter_feat = copy.deepcopy(dataset.inter_feat)
            inter_feat.sort(dataset.time_field)
            user_ids = inter_feat[dataset.uid_field].numpy()
            item_ids = inter_feat[dataset.iid_field].numpy()
            
            w_max = self.w_max
            w_min = self.w_min
            values = np.zeros(len(inter_feat))
            
            row_num = dataset.user_num
            
            for uid in range(1, row_num + 1):
                uindex = np.argwhere(user_ids == uid).flatten()
                int_num = len(uindex)
                if int_num > 1:
                    weight = np.linspace(w_min, w_max, int_num)
                else:
                    weight = np.array([w_max])
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
                return betas_from_linear_variance(
                    self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
                )
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
        """Calculate coefficients for the diffusion process."""
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        ).to(self.device)
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]
        ).to(self.device)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        
        # Coefficients for posterior mean (Eq. 10)
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
    
    def _split_by_category(self, x):
        """Split interaction vector by item categories."""
        x_split = []
        for c in range(self.n_cate):
            indices = torch.tensor(self.category_idx[c], device=x.device)
            x_c = x[:, indices]
            x_split.append(x_c)
        return x_split
    
    def _merge_by_category(self, x_split):
        """Merge split vectors back to full interaction vector."""
        batch_size = x_split[0].shape[0]
        x_full = torch.zeros(batch_size, self.n_items, device=x_split[0].device)
        
        for c in range(self.n_cate):
            indices = torch.tensor(self.category_idx[c], device=x_split[0].device)
            x_full[:, indices] = x_split[c]
        
        return x_full
    
    def encode(self, x):
        """Encode interaction vector to latent space."""
        x_split = self._split_by_category(x)
        z_list = []
        mu_list = []
        logvar_list = []
        
        for c in range(self.n_cate):
            mu, logvar = self.encoders[c](x_split[c])
            
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            
            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        z = torch.cat(z_list, dim=-1)
        return z, mu_list, logvar_list
    
    def decode(self, z):
        """Decode latent vector to interaction probabilities."""
        z_split = torch.split(z, self.latent_dim, dim=-1)
        x_split = []
        
        for c in range(self.n_cate):
            x_c = self.decoders[c](z_split[c])
            x_split.append(x_c)
        
        x = self._merge_by_category(x_split)
        return x
    
    def q_sample(self, z_start, t, noise=None):
        """Sample from q(z_t | z_0) - forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(z_start)
        
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, z_start.shape) * z_start +
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, z_start, z_t, t):
        """Compute the mean and variance of q(z_{t-1} | z_t, z_0)."""
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, z_t.shape) * z_start +
            self._extract_into_tensor(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, z_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, z_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, z_t, t):
        """Apply the model to get p(z_{t-1} | z_t)."""
        model_output = self.dnn(z_t, t)
        
        model_variance = self._extract_into_tensor(self.posterior_variance, t, z_t.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, z_t.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_zstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_zstart = (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, z_t.shape) * z_t -
                self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, z_t.shape) * model_output
            )
        
        model_mean, _, _ = self.q_posterior_mean_variance(z_start=pred_zstart, z_t=z_t, t=t)
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_zstart": pred_zstart,
        }
    
    def p_sample(self, z_start):
        """Generate latent vectors through reverse diffusion process."""
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
            
            if self.sampling_noise:
                noise = torch.randn_like(z_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(z_t.shape) - 1)))
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                z_t = out["mean"]
        
        return z_t
    
    def sample_timesteps(self, batch_size, device, method="importance", uniform_prob=0.001):
        """Sample timesteps for training."""
        if method == "importance":
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            
            return t, pt
        else:
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()
            return t, pt
    
    def update_Lt_history(self, ts, reloss):
        """Update loss history for importance sampling."""
        for t, loss in zip(ts, reloss):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                self.Lt_count[t] += 1
    
    def SNR(self, t):
        """Compute signal-to-noise ratio for timestep t."""
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def calculate_loss(self, interaction):
        """Calculate training loss."""
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)
        
        batch_size = x_start.size(0)
        device = x_start.device
        
        # Encode to latent space
        z_start, mu_list, logvar_list = self.encode(x_start)
        
        # VAE loss with annealing
        self.update_count_vae += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update_count_vae / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap
        
        # Reconstruction loss (decode from z_start without diffusion for VAE loss)
        x_recon = self.decode(z_start)
        recon_loss = -(F.log_softmax(x_recon, dim=1) * x_start).sum(1).mean()
        
        # KL divergence loss
        kl_loss = 0
        for c in range(self.n_cate):
            kl_loss += -0.5 * torch.mean(
                torch.sum(1 + logvar_list[c] - mu_list[c].pow(2) - logvar_list[c].exp(), dim=1)
            )
        kl_loss = kl_loss * anneal
        
        vae_loss = recon_loss + kl_loss
        
        # Diffusion loss
        if self.noise_scale != 0.0:
            ts, pt = self.sample_timesteps(batch_size, device, "importance")
            noise = torch.randn_like(z_start)
            z_t = self.q_sample(z_start, ts, noise)
            
            model_output = self.dnn(z_t, ts)
            
            if self.mean_type == ModelMeanType.START_X:
                target = z_start
            else:
                target = noise
            
            mse = mean_flat((target - model_output) ** 2)
            
            # Reweight loss
            if self.reweight:
                if self.mean_type == ModelMeanType.START_X:
                    weight = self.SNR(ts - 1) - self.SNR(ts)
                    weight = torch.where((ts == 0), 1.0, weight)
                else:
                    weight = torch.ones_like(mse)
                reloss = weight * mse
            else:
                reloss = mse
            
            self.update_Lt_history(ts, reloss)
            
            # Importance sampling weight
            reloss = reloss / pt
            diff_loss = reloss.mean()
        else:
            diff_loss = torch.tensor(0.0).to(device)
        
        # Combined loss
        total_loss = vae_loss + self.lamda * diff_loss
        
        return total_loss
    
    def forward(self, x):
        """Forward pass for inference."""
        # Encode
        z_start, _, _ = self.encode(x)
        
        # Diffusion reverse process
        z_recon = self.p_sample(z_start)
        
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
        
        return scores