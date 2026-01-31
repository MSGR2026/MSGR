# -*- coding: utf-8 -*-
# NLGCL: A Contrastive Learning between Neighbor Layers for Graph Collaborative Filtering

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class LightGCNConv(nn.Module):
    """轻量级图卷积层，用于邻域聚合"""
    def __init__(self, dim):
        super(LightGCNConv, self).__init__()
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: 节点特征 [N, dim]
            edge_index: 边索引 [2, E] 或稀疏矩阵
            edge_weight: 边权重 [E] 或 None
        """
        if edge_weight is None:
            # 使用稀疏矩阵乘法
            return torch.sparse.mm(edge_index, x)
        else:
            # 手动消息传递
            row, col = edge_index
            out = torch.zeros_like(x)
            # 对每条边进行聚合
            out.index_add_(0, row, x[col] * edge_weight.view(-1, 1))
            return out


class NLGCL(GeneralRecommender):
    """NLGCL: 基于邻居层对比学习的图协同过滤模型"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # 加载数据集信息
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # 加载模型超参数
        self.latent_dim = config['embedding_size']  # 嵌入维度
        self.n_layers = config['n_layers']          # GCN层数
        self.reg_weight = config['reg_weight']      # L2正则化权重
        self.require_pow = config['require_pow']    # 是否使用幂次正则化
        
        # 对比学习超参数
        self.cl_temp = config['cl_temp']            # 对比学习温度
        self.cl_reg = config['cl_reg']              # 对比学习损失权重
        self.alpha = config['alpha']                # 用户/物品对比损失平衡系数

        # 初始化嵌入层
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        
        # 初始化图卷积层
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        
        # 初始化损失函数
        self.mf_loss = BPRLoss()    # BPR损失
        self.reg_loss = EmbLoss()   # 正则化损失

        # 用于全排序加速的缓存变量
        self.restore_user_e = None
        self.restore_item_e = None

        # 生成归一化的邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # 参数初始化
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        """获取归一化的用户-物品交互图邻接矩阵
        
        构建用户-物品二部图，并使用拉普拉斯归一化:
        A_hat = D^{-0.5} * A * D^{-0.5}
        
        Returns:
            归一化邻接矩阵的稀疏张量
        """
        # 提取用户-物品交互的coo矩阵
        inter_M = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items

        # 构建用户->物品的边: (user_id, item_id + n_users)
        row_user = inter_M.row
        col_item = inter_M.col + n_users
        data_user_item = np.ones(inter_M.nnz, dtype=np.float32)

        # 构建物品->用户的边: (item_id + n_users, user_id)
        row_item = inter_M.col + n_users
        col_user = inter_M.row
        data_item_user = np.ones(inter_M.nnz, dtype=np.float32)

        # 合并成对称的邻接矩阵
        row = np.concatenate([row_user, row_item])
        col = np.concatenate([col_item, col_user])
        data = np.concatenate([data_user_item, data_item_user])

        # 构建coo格式的邻接矩阵
        A = sp.coo_matrix(
            (data, (row, col)),
            shape=(n_users + n_items, n_users + n_items),
            dtype=np.float32
        )

        # 计算度矩阵 D
        sumArr = np.array(A.sum(axis=1)).flatten()
        diag = np.power(sumArr, -0.5)
        diag[np.isinf(diag)] = 0.0
        D = sp.diags(diag)

        # 归一化: D^{-0.5} * A * D^{-0.5}
        L = D @ A @ D
        L = L.tocoo()

        # 转换为PyTorch稀疏张量
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        shape = torch.Size(L.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_ego_embeddings(self):
        """获取初始嵌入（ego embeddings）"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return torch.cat([user_emb, item_emb], dim=0)

    def forward(self):
        """执行图卷积前向传播
        
        Returns:
            user_emb: 用户最终嵌入
            item_emb: 物品最终嵌入
            emb_list: 所有层的嵌入列表（包括第0层）
        """
        # 获取初始嵌入
        all_emb = self.get_ego_embeddings()
        emb_list = [all_emb]  # 存储所有层的嵌入
        
        # 多层图卷积
        for _ in range(self.n_layers):
            all_emb = self.gcn_conv(all_emb, self.norm_adj_matrix, None)
            emb_list.append(all_emb)
        
        # 使用均值池化聚合所有层的嵌入
        lightgcn_emb = torch.stack(emb_list, dim=1)
        lightgcn_emb = torch.mean(lightgcn_emb, dim=1)
        
        # 分离用户和物品嵌入
        user_emb, item_emb = torch.split(lightgcn_emb, [self.n_users, self.n_items])
        return user_emb, item_emb, emb_list

    def InfoNCE(self, anchor, positive, all_samples):
        """计算InfoNCE对比损失
        
        Args:
            anchor: 锚点样本嵌入
            positive: 正样本嵌入
            all_samples: 所有样本嵌入（用于负采样）
        """
        # 归一化嵌入以计算余弦相似度
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        all_samples = F.normalize(all_samples, dim=1)
        
        # 计算正样本相似度
        pos_score = torch.sum(anchor * positive, dim=1)
        pos_score = torch.exp(pos_score / self.cl_temp)
        
        # 计算所有样本相似度（包括负样本）
        ttl_score = torch.matmul(anchor, all_samples.t())
        ttl_score = torch.exp(ttl_score / self.cl_temp).sum(dim=1)
        
        # 计算最终对比损失
        return -torch.log(pos_score / ttl_score).sum()

    def neighbor_cl_loss(self, emb_list, users, pos_items, neg_items):
        """计算邻居层之间的对比学习损失
        
        这是NLGCL的核心创新：在相邻GCN层之间进行对比学习
        
        Args:
            emb_list: 所有层的嵌入列表
            users: 批次用户ID
            pos_items: 正样本物品ID
            neg_items: 负样本物品ID
        """
        # 获取初始嵌入（第0层）
        user_emb_0, item_emb_0 = torch.split(emb_list[0], [self.n_users, self.n_items])
        cl_user_loss = 0.0
        cl_item_loss = 0.0

        # 在每一层计算对比损失
        for layer_idx in range(1, self.n_layers + 1):
            # 获取当前层嵌入
            user_emb_k, item_emb_k = torch.split(emb_list[layer_idx], [self.n_users, self.n_items])
            
            # 用户侧对比损失：使用物品嵌入作为锚点，用户嵌入作为目标
            cl_user_loss = self.InfoNCE(
                anchor=item_emb_k[pos_items],
                positive=user_emb_0[users],
                all_samples=user_emb_0[users]
            ) + 1e-6

            # 物品侧对比损失：使用用户嵌入作为锚点，物品嵌入作为目标
            cl_item_loss = self.InfoNCE(
                anchor=user_emb_k[users],
                positive=item_emb_0[pos_items],
                all_samples=item_emb_0[pos_items]
            ) + 1e-6
            
            # 更新参考嵌入为当前层
            user_emb_0, item_emb_0 = user_emb_k, item_emb_k

        return cl_user_loss, cl_item_loss

    def calculate_loss(self, interaction):
        """计算训练损失
        
        总损失 = BPR损失 + 正则化损失 + 对比学习损失
        """
        # 清除缓存
        self.restore_user_e = None
        self.restore_item_e = None

        # 获取批次数据
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # 前向传播
        user_all_emb, item_all_emb, emb_list = self.forward()
        
        # 1. 计算BPR损失
        u_emb = user_all_emb[users]
        pos_emb = item_all_emb[pos_items]
        neg_emb = item_all_emb[neg_items]
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        bpr_loss = self.mf_loss(pos_scores, neg_scores)
        
        # 2. 计算正则化损失
        reg_loss = self.reg_loss(
            self.user_embedding(users),
            self.item_embedding(pos_items),
            self.item_embedding(neg_items),
            require_pow=self.require_pow
        )
        
        # 3. 计算对比学习损失
        cl_u, cl_i = self.neighbor_cl_loss(emb_list, users, pos_items, neg_items)
        cl_loss = self.alpha * cl_u + (1 - self.alpha) * cl_i
        
        # 返回加权的总损失
        return bpr_loss + self.reg_weight * reg_loss + self.cl_reg * cl_loss

    def predict(self, interaction):
        """预测用户-物品交互分数"""
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        # 获取嵌入并计算点积
        user_emb, item_emb, _ = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        return torch.mul(u_emb, i_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        """为给定用户预测所有物品的分数（全排序）"""
        users = interaction[self.USER_ID]
        
        # 使用缓存加速
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        
        # 通过矩阵乘法计算所有物品分数
        user_emb = self.restore_user_e[users]
        scores = torch.matmul(user_emb, self.restore_item_e.t())
        return scores.view(-1)
