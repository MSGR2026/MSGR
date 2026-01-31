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
        self.n_layers = config['n_mm_layers']
        self.lambda_cl = config['lambda_cl']
        self.temperature = config['temperature']
        self.knn_k = config['image_knn_k']  # Assume config has knn parameters
        
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
        if self.v_feat is not None:
            self.mlp_v = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        if self.t_feat is not None:
            self.mlp_t = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        
        # Frequency domain filters
        if self.v_feat is not None:
            self.W2_v = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.cfloat).to(self.device))
        if self.t_feat is not None:
            self.W2_t = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.cfloat).to(self.device))
        self.W2_fusion = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.cfloat).to(self.device))
        
        # Graph structures
        self.S_mats = self.build_modality_graphs()
        self.fusion_graph = self.build_fusion_graph()
        
        # Preference modules
        self.W5_m = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.W6_m = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.W7_f = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        
        # Loss functions
        self.bpr_loss = BPRLoss()
        self.infonce_loss = nn.CrossEntropyLoss()
        
    def build_modality_graphs(self):
        S_mats = {}
        for modality in ['v', 't']:
            if modality == 'v' and self.v_feat is not None:
                feats = self.mlp_v(self.v_feat)
            elif modality == 't' and self.t_feat is not None:
                feats = self.mlp_t(self.t_feat)
            else:
                continue
            similarity = torch.matmul(feats, feats.T)
            similarity = F.normalize(similarity, p=2, dim=1)
            knn_graph = self.knn_from_similarity(similarity, self.knn_k)
            S_mats[modality] = self.normalize_graph(knn_graph)
        return S_mats
    
    def build_fusion_graph(self):
        # Max-pooling of modality graphs
        mods = []
        for m in ['v', 't']:
            if m in self.S_mats:
                mods.append(self.S_mats[m])
        fusion_adj = torch.max(*mods) if len(mods) >1 else mods[0]
        return self.normalize_graph(fusion_adj)
    
    def knn_from_similarity(self, sim_matrix, k):
        topk_values, topk_indices = torch.topk(sim_matrix, k, dim=1)
        rows = torch.arange(sim_matrix.shape[0]).view(-1,1).repeat(1,k).flatten()
        cols = topk_indices.flatten()
        vals = topk_values.flatten()
        return torch.sparse_coo_tensor([rows, cols], vals, sim_matrix.shape)
    
    def normalize_graph(self, adj):
        degree = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        deg_inv_sqrt = torch.pow(degree, -0.5)
        rows, cols = adj.coalesce().indices()
        values = adj.coalesce().values() * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
        return torch.sparse_coo_tensor(adj.indices(), values, adj.shape).to(self.device)
    
    def spectrum_fusion(self, H_v, H_t):
        # FFT transforms
        H_v_freq = torch.fft.fft(H_v, dim=0)
        H_t_freq = torch.fft.fft(H_t, dim=0)
        
        # Apply filters
        H_v_filtered = H_v_freq * self.W2_v
        H_t_filtered = H_t_freq * self.W2_t
        
        # Fusion in frequency domain
        fusion_freq = H_v_filtered * H_t_filtered  # Pointwise product
        fusion_filtered = fusion_freq * self.W2_fusion
        
        # Inverse FFT
        H_f = torch.fft.ifft(fusion_filtered, dim=0).real
        
        return H_f
    
    def forward(self):
        # Modality feature processing
        modality_features = {}
        if self.v_feat is not None:
            H_v = self.mlp_v(self.v_feat)
            modality_features['v'] = H_v
        if self.t_feat is not None:
            H_t = self.mlp_t(self.t_feat)
            modality_features['t'] = H_t
            
        # Fusion
        if len(modality_features) ==2:
            H_f = self.spectrum_fusion(H_v, H_t)
        else:
            H_f = list(modality_features.values())[0]
        
        # Graph propagation
        item_embs = []
        for m in modality_features:
            feats = modality_features[m]
            for _ in range(self.n_layers):
                feats = torch.sparse.mm(self.S_mats[m], feats)
            item_embs.append(feats)
        fusion_feats = H_f
        for _ in range(self.n_layers):
            fusion_feats = torch.sparse.mm(self.fusion_graph, fusion_feats)
        item_embs.append(fusion_feats)
        
        # User embeddings via aggregation
        user_embs = []
        for emb in item_embs:
            user_agg = torch.sparse.mm(self.interaction_matrix().t(), emb)
            user_embs.append(user_agg / torch.norm(user_agg, dim=1, keepdim=True))
        user_embs = torch.cat(user_embs, dim=1)
        
        item_embs = torch.cat(item_embs, dim=1)
        
        return user_embs, item_embs
    
    def inter_matrix(self):
        return self.dataset.inter_matrix形式='coo').to(self.device)
    
    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        
        user_embs, item_embs = self.forward()
        u_e = user_embs[users]
        pos_i_e = item_embs[pos_items]
        neg_i_e = item_embs[neg_items]
        
        bpr_loss = self.bpr_loss(u_e, pos_i_e, neg_i_e)
        
        # Contrastive loss
        cl_loss = 0
        for mod in ['v', 't']:
            if mod in self.S_mats:
                mod_feats = modality_features[mod]
                cl_loss += self.infonce_loss(mod_feats[users], u_e)
                cl_loss += self.infonce_loss(mod_feats[pos_items], pos_i_e)
        
        return bpr_loss + self.lambda_cl * cl_loss
    
    def full_sort_predict(self, interaction):
        user_id = interaction[0]
        all_user, all_item = self.forward()
        scores = torch.matmul(all_user[user_id], all_item.t())
        return scores