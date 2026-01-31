import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from common.abstract_recommender import GeneralRecommender

class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.n_mm_layers = config['n_mm_layers']
        self.lambda_cl = config['lambda_cl']
        self.temperature = config['temperature']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        
        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        self.device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")
        
        # Modality features
        self.v_feat = dataset.v_feat.to(self.device) if hasattr(dataset, 'v_feat') else None
        self.t_feat = dataset.t_feat.to(self.device) if hasattr(dataset, 't_feat') else None
        
        # ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        
        # Modality projection MLPs
        self.modality_projections = nn.ModuleDict()
        if self.v_feat is not None:
            self.modality_projections['v'] = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        if self.t_feat is not None:
            self.modality_projections['t'] = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        
        # Frequency domain filters
        self.filters = nn.ModuleDict()
        for m in ['v', 't', 'f']:
            self.filters[m] = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.cfloat).to(self.device))
        
        # Graph structures
        self.S_mats = self.build_modality_graphs()
        self.fusion_graph = self.build_fusion_graph()
        
        # Preference modules
        self.W5_m = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.W6_m = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.W7_f = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        
        # Loss functions
        self.bpr_loss = BPRLoss()
        
    def build_modality_graphs(self):
        S_mats = {}
        for modality in ['v', 't']:
            if modality not in self.modality_projections:
                continue
            feats = self.modality_projections[modality](getattr(self, f"{modality}_feat"))
            similarity = feats @ feats.T
            similarity /= (torch.norm(feats, dim=1).unsqueeze(1) * torch.norm(feats, dim=1).unsqueeze(0)) + 1e-8
            knn_k = self.image_knn_k if modality == 'v' else self.text_knn_k
            knn_graph = self.knn_from_similarity(similarity, knn_k)
            S_mats[modality] = self.normalize_graph(knn_graph)
        return S_mats
    
    def build_fusion_graph(self):
        if len(self.S_mats) <2:
            return list(self.S_mats.values())[0]
        fusion_adj = torch.max(torch.stack([self.S_mats['v'], self.S_mats['t']]))
        return self.normalize_graph(fusion_adj)
    
    def knn_from_similarity(self, sim_matrix, k):
        topk_vals, topk_indices = sim_matrix.topk(k, dim=1)
        rows = torch.arange(sim_matrix.shape[0]).view(-1,1).repeat(1,k).flatten()
        cols = topk_indices.flatten()
        vals = topk_vals.flatten()
        return torch.sparse_coo_tensor([rows, cols], vals, sim_matrix.shape).to(self.device)
    
    def normalize_graph(self, adj):
        degree = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        deg_inv_sqrt = torch.pow(degree, -0.5)
        rows, cols = adj.coalesce().indices()
        vals = adj.coalesce().values() * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
        return torch.sparse_coo_tensor(adj.indices(), vals, adj.shape).to(self.device)
    
    def spectrum_fusion(self, H_v, H_t):
        H_v_freq = torch.fft.fft(H_v, dim=0)
        H_t_freq = torch.fft.fft(H_t, dim=0)
        filtered_v = H_v_freq * self.filters['v']
        filtered_t = H_t_freq * self.filters['t']
        fusion_freq = filtered_v * filtered_t
        filtered_fusion = fusion_freq * self.filters['f']
        return torch.fft.ifft(filtered_fusion, dim=0).real
    
    def forward(self):
        modality_features = {}
        for modality, proj in self.modality_projections.items():
            feats = proj(getattr(self, f"{modality}_feat"))
            modality_features[modality] = feats
        
        H_f = None
        if len(modality_features) ==2:
            H_f = self.spectrum_fusion(modality_features['v'], modality_features['t'])
        
        # Graph propagation
        item_embs = []
        for m in modality_features:
            feats = modality_features[m]
            for _ in range(self.n_mm_layers):
                feats = torch.sparse.mm(self.S_mats[m], feats)
            item_embs.append(feats)
        if H_f is not None:
            fusion_feats = H_f
            for _ in range(self.n_mm_layers):
                fusion_feats = torch.sparse.mm(self.fusion_graph, fusion_feats)
            item_embs.append(fusion_feats)
        item_embs = torch.cat(item_embs, dim=1)
        
        # User embeddings via aggregation
        user_embs = []
        inter_adj = self.inter_matrix().t()
        for emb in item_embs:
            agg = torch.sparse.mm(inter_adj, emb)
            user_embs.append(agg / (torch.norm(agg, dim=1, keepdim=True) + 1e-8))
        user_embs = torch.cat(user_embs, dim=1)
        
        return user_embs, item_embs
    
    def inter_matrix(self):
        return self.dataset.inter_matrix(form='coo').to(self.device)
    
    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        users = users.to(self.device)
        pos_items = pos_items.to(self.device)
        neg_items = neg_items.to(self.device)
        
        user_embs, item_embs = self.forward()
        u_e = user_embs[users]
        pos_i_e = item_embs[pos_items]
        neg_i_e = item_embs[neg_items]
        
        return self.bpr_loss(u_e, pos_i_e, neg_i_e)
    
    def full_sort_predict(self, interaction):
        user_id = interaction[0].to(self.device)
        user_embs, item_embs = self.forward()
        return user_embs[user_id] @ item_embs.t()