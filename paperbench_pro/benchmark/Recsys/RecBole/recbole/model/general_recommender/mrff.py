# -*- coding: utf-8 -*-
# @Time   : 2025/01/06
# @Author : Migration from AAAI-25-MRFF
# @Email  : your_email@example.com

r"""
MRFF (Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach)
################################################

Reference:
    Zhang, Chunxu et al. "Multifaceted user modeling in recommendation: 
    A federated foundation models approach" in AAAI 2025.

Reference code:
    https://github.com/Zhangcx19/AAAI-25-MRFF

Note:
    This is a General Recommender version that inherits from GeneralRecommender.
    It processes user-item interactions without explicit sequence modeling.
"""

import torch
import torch.nn as nn
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class PointWiseFeedForward(nn.Module):
    """标准的点式前馈网络"""
    
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        outputs = self.dropout2(
            self.fc2(
                self.relu(
                    self.dropout1(
                        self.fc1(inputs)
                    )
                )
            )
        )
        outputs += inputs
        return outputs


class MOEPointWiseFeedForward(nn.Module):
    """多专家混合(MoE)前馈网络 - 通用推荐版本"""
    
    def __init__(self, hidden_units, switch_hidden_dims, 
                 dropout_rate, num_experts):
        super(MOEPointWiseFeedForward, self).__init__()
        
        self.n_experts = num_experts
        self.switch_hidden_dims = switch_hidden_dims
        
        # 用户专属专家网络
        self.uexpert = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate)
        )
        
        # 多个共享专家网络
        single_expert_net = lambda: nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate)
        )
        self.experts = nn.ModuleList([single_expert_net() for _ in range(num_experts)])
        
        # 门控网络(Switch/Router)
        self.switch = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(
            zip(self.switch_hidden_dims[:-1], self.switch_hidden_dims[1:])
        ):
            self.switch.append(nn.Linear(in_size, out_size))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, user_embedding):
        """
        Args:
            x: 当前层输入 [batch_size, hidden_units]
            user_embedding: 用户嵌入 [batch_size, hidden_units]
        """
        # 使用用户嵌入作为路由输入
        x_input = user_embedding
        
        # 通过门控网络选择专家
        for idx in range(len(self.switch)):
            x_input = self.switch[idx](x_input)
            if idx < len(self.switch) - 1:
                x_input = nn.ReLU()(x_input)
        
        route_prob = self.softmax(x_input)
        out_prob_max, routes = torch.max(route_prob, dim=-1)
        
        # 对batch中的每个样本选择专家
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            expert_idx = routes[b].item()
            adopted_expert = self.experts[expert_idx]
            output = adopted_expert(x[b:b+1])
            outputs.append(output)
        
        output = torch.cat(outputs, dim=0)
        
        # 用户专属专家输出
        uexpert_output = self.uexpert(x)
        
        # 组合输出
        final_output = output + uexpert_output
        
        return final_output


class MRFF(GeneralRecommender):
    r"""
    MRFF是一个基于多专家混合(MoE)的通用推荐模型。
    
    通用推荐版本说明:
    - 继承GeneralRecommender，处理(user, item)交互对
    - 使用用户和物品嵌入进行交互建模
    - 通过MoE机制捕获用户的多面兴趣
    """
    
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(MRFF, self).__init__(config, dataset)
        
        # 加载参数
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']
        
        # MRFF特有参数
        self.num_experts = config.get('num_experts', 4)
        self.use_moe = config.get('use_moe', True)
        
        # 计算switch网络的维度
        switch_hidden_dims = config.get('switch_hidden_dims', [])
        if isinstance(switch_hidden_dims, str):
            switch_hidden_dims = [int(x) for x in switch_hidden_dims.split(',') if x]
        if not switch_hidden_dims:
            switch_hidden_dims = [self.hidden_size, self.hidden_size // 2]
        switch_hidden_dims.append(self.num_experts)
        self.switch_hidden_dims = switch_hidden_dims
        
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        
        # 定义嵌入层
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)
        
        # Transformer风格的交互层
        self.interaction_layernorms = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            new_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.interaction_layernorms.append(new_layernorm)
            
            if self.use_moe:
                new_layer = MOEPointWiseFeedForward(
                    self.hidden_size,
                    self.switch_hidden_dims,
                    self.hidden_dropout_prob,
                    self.num_experts
                )
            else:
                new_layer = PointWiseFeedForward(
                    self.hidden_size,
                    self.hidden_dropout_prob
                )
            self.interaction_layers.append(new_layer)
        
        self.last_layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # 损失函数
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, user, item):
        """
        Args:
            user: [batch_size] 用户ID
            item: [batch_size] 物品ID
        
        Returns:
            scores: [batch_size] 预测分数
        """
        # 获取嵌入
        user_e = self.user_embedding(user)  # [batch_size, hidden_size]
        item_e = self.item_embedding(item)  # [batch_size, hidden_size]
        
        # 用户-物品交互表示
        interaction = user_e * item_e  # element-wise product
        interaction = self.dropout(interaction)
        
        # 通过多层交互层
        for i in range(len(self.interaction_layers)):
            interaction = self.interaction_layernorms[i](interaction)
            
            if self.use_moe:
                interaction = self.interaction_layers[i](interaction, user_e)
            else:
                interaction = self.interaction_layers[i](interaction)
        
        interaction = self.last_layernorm(interaction)
        
        # 计算最终分数
        scores = torch.sum(interaction, dim=-1)  # [batch_size]
        
        return scores
    
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        if self.loss_type == 'BPR':
            neg_item = interaction[self.NEG_ITEM_ID]
            pos_scores = self.forward(user, item)
            neg_scores = self.forward(user, neg_item)
            loss = self.loss_fct(pos_scores, neg_scores)
            return loss
        else:  # CE
            # 对于CE损失，需要计算所有物品的分数
            user_e = self.user_embedding(user)
            item_e = self.item_embedding.weight
            
            # 扩展用户嵌入以匹配所有物品
            user_e_expanded = user_e.unsqueeze(1)  # [batch_size, 1, hidden_size]
            item_e_expanded = item_e.unsqueeze(0)  # [1, n_items, hidden_size]
            
            # 计算交互
            interaction_emb = user_e_expanded * item_e_expanded  # [batch_size, n_items, hidden_size]
            
            # 通过交互层（简化版，不使用MoE以提高效率）
            batch_size = interaction_emb.size(0)
            n_items = interaction_emb.size(1)
            
            for i in range(len(self.interaction_layers)):
                interaction_flat = interaction_emb.view(-1, self.hidden_size)
                interaction_flat = self.interaction_layernorms[i](interaction_flat)
                
                if not self.use_moe:  # CE模式下建议关闭MoE以提高效率
                    interaction_flat = self.interaction_layers[i](interaction_flat)
                
                interaction_emb = interaction_flat.view(batch_size, n_items, self.hidden_size)
            
            interaction_emb = self.last_layernorm(interaction_emb.view(-1, self.hidden_size))
            interaction_emb = interaction_emb.view(batch_size, n_items, self.hidden_size)
            
            # 计算logits
            logits = torch.sum(interaction_emb, dim=-1)  # [batch_size, n_items]
            loss = self.loss_fct(logits, item)
            return loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        scores = self.forward(user, item)
        return scores
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)  # [batch_size, hidden_size]
        
        # 计算与所有物品的分数
        all_item_e = self.item_embedding.weight  # [n_items, hidden_size]
        
        # 简化版本：直接计算点积
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))  # [batch_size, n_items]
        
        return scores.view(-1)

