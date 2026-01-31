# -*- coding: utf-8 -*-
# @Time   : 2024/01/01
# @Author : MCCLK Implementation
# @Email  : mcclk@example.com

r"""
MCCLK
##################################################
Reference:
    Ding Zou et al. "Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System." 
    in SIGIR 2022.
    
Note: This implementation follows the CKE (Collaborative Knowledge Base Embedding) framework described
in the method section, which integrates structural, textual, and visual knowledge embeddings with 
collaborative filtering through Bayesian approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class TransRLayer(nn.Module):
    """
    TransR-based structural embedding layer.
    Maps entities and relations to distinct spaces via relation-specific projection matrices.
    """
    
    def __init__(self, n_entities, n_relations, embedding_size, relation_dim):
        super(TransRLayer, self).__init__()
        
        self.embedding_size = embedding_size
        self.relation_dim = relation_dim
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(n_entities, embedding_size)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        
        # Projection matrices for each relation: entity_dim -> relation_dim
        self.projection_matrices = nn.Embedding(n_relations, embedding_size * relation_dim)
        
    def forward(self, head, relation, tail):
        """
        Compute TransR score for triplets.
        
        Args:
            head: [batch_size] head entity IDs
            relation: [batch_size] relation IDs
            tail: [batch_size] tail entity IDs
            
        Returns:
            score: [batch_size] TransR scores (lower is better for valid triplets)
        """
        # Get embeddings
        h_emb = self.entity_embedding(head)  # [batch_size, embedding_size]
        r_emb = self.relation_embedding(relation)  # [batch_size, relation_dim]
        t_emb = self.entity_embedding(tail)  # [batch_size, embedding_size]
        
        # Get projection matrices
        proj_mat = self.projection_matrices(relation).view(
            -1, self.embedding_size, self.relation_dim
        )  # [batch_size, embedding_size, relation_dim]
        
        # Project entities to relation space
        h_projected = torch.bmm(h_emb.unsqueeze(1), proj_mat).squeeze(1)  # [batch_size, relation_dim]
        t_projected = torch.bmm(t_emb.unsqueeze(1), proj_mat).squeeze(1)  # [batch_size, relation_dim]
        
        # Compute score: ||h_r + r - t_r||^2
        score = torch.sum((h_projected + r_emb - t_projected) ** 2, dim=1)
        
        return score
    
    def get_entity_embedding(self, entity_ids):
        """Get entity embeddings."""
        return self.entity_embedding(entity_ids)


class StackedDenoisingAutoEncoder(nn.Module):
    """
    Stacked Denoising Auto-Encoder (SDAE) for textual embedding.
    Learns latent representations by reconstructing clean input from corrupted input.
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.3):
        super(StackedDenoisingAutoEncoder, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, add_noise=True):
        """
        Forward pass with optional noise injection.
        
        Args:
            x: [batch_size, input_dim] input features
            add_noise: whether to add noise during training
            
        Returns:
            latent: [batch_size, latent_dim] latent representation
            reconstructed: [batch_size, input_dim] reconstructed input
        """
        # Add noise (corruption)
        if add_noise and self.training:
            x_corrupted = self.dropout(x)
        else:
            x_corrupted = x
        
        # Encode
        latent = self.encoder(x_corrupted)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return latent, reconstructed
    
    def get_latent(self, x):
        """Get latent representation without noise."""
        return self.encoder(x)


class KGAggregator(nn.Module):
    """
    Knowledge Graph Aggregator for structural view.
    Aggregates neighbor information using relation-aware attention.
    """
    
    def __init__(self, embedding_size):
        super(KGAggregator, self).__init__()
        self.embedding_size = embedding_size
        
    def forward(self, entity_emb, edge_index, edge_type, relation_emb, n_entities):
        """
        Aggregate neighbor information on KG.
        
        Args:
            entity_emb: [n_entities, embedding_size] entity embeddings
            edge_index: [2, n_edges] edge indices (head, tail)
            edge_type: [n_edges] relation types
            relation_emb: [n_relations, embedding_size] relation embeddings
            n_entities: number of entities
            
        Returns:
            aggregated: [n_entities, embedding_size] aggregated entity embeddings
        """
        from torch_scatter import scatter_mean
        
        head, tail = edge_index
        
        # Get relation embeddings for each edge
        edge_relation_emb = relation_emb[edge_type]  # [n_edges, embedding_size]
        
        # Neighbor message: tail entity * relation
        neighbor_msg = entity_emb[tail] * edge_relation_emb  # [n_edges, embedding_size]
        
        # Aggregate messages to head entities
        aggregated = scatter_mean(
            src=neighbor_msg,
            index=head,
            dim=0,
            dim_size=n_entities
        )  # [n_entities, embedding_size]
        
        return aggregated


class LightGCNLayer(nn.Module):
    """
    LightGCN layer for collaborative view.
    Simple message passing without feature transformation.
    """
    
    def __init__(self):
        super(LightGCNLayer, self).__init__()
        
    def forward(self, user_emb, item_emb, norm_adj):
        """
        Perform one layer of LightGCN propagation.
        
        Args:
            user_emb: [n_users, embedding_size]
            item_emb: [n_items, embedding_size]
            norm_adj: normalized user-item adjacency matrix
            
        Returns:
            new_user_emb: [n_users, embedding_size]
            new_item_emb: [n_items, embedding_size]
        """
        # Concatenate user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Propagate
        all_emb_new = torch.sparse.mm(norm_adj, all_emb)
        
        # Split back
        n_users = user_emb.shape[0]
        new_user_emb = all_emb_new[:n_users]
        new_item_emb = all_emb_new[n_users:]
        
        return new_user_emb, new_item_emb


class MCCLK(KnowledgeRecommender):
    r"""MCCLK (Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System)
    
    This implementation follows the CKE framework which integrates:
    1. Structural embedding via TransR on knowledge graph
    2. Collaborative filtering with LightGCN
    3. Multi-level contrastive learning across views
    
    The item latent vector combines:
    - CF offset vector (η_j)
    - Structural representation (v_j) from TransR
    - Optional textual/visual representations
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MCCLK, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config.get("kg_embedding_size", self.embedding_size)
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.kg_weight = config.get("kg_weight", 1.0)
        self.cl_weight = config.get("cl_weight", 0.1)
        self.temperature = config.get("temperature", 0.2)
        self.mess_dropout_rate = config.get("mess_dropout", 0.1)
        
        # Build graphs
        self._build_graphs(dataset)
        
        # User embeddings (for CF)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        
        # Item offset vector (η_j in the paper)
        self.item_offset = nn.Embedding(self.n_items, self.embedding_size)
        
        # TransR layer for structural embedding
        self.transr = TransRLayer(
            self.n_entities, 
            self.n_relations, 
            self.embedding_size,
            self.kg_embedding_size
        )
        
        # KG aggregator for multi-hop propagation
        self.kg_aggregators = nn.ModuleList([
            KGAggregator(self.embedding_size) for _ in range(self.n_layers)
        ])
        
        # LightGCN layers for collaborative view
        self.lightgcn_layers = nn.ModuleList([
            LightGCNLayer() for _ in range(self.n_layers)
        ])
        
        # Projection heads for contrastive learning
        self.fc_kg = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
        
        self.fc_cf = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
        
        # Dropout
        self.mess_dropout = nn.Dropout(self.mess_dropout_rate)
        
        # Loss functions
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Cache for full sort prediction
        self.restore_user_e = None
        self.restore_item_e = None
        
        # Initialize parameters
        self.apply(xavier_uniform_initialization)

    def _build_graphs(self, dataset):
        """Build necessary graph structures."""
        
        # Get interaction matrix
        inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Build normalized user-item adjacency for LightGCN
        self.norm_adj = self._build_lightgcn_adj(inter_matrix)
        
        # Get KG graph
        kg_graph = dataset.kg_graph(form='coo', value_field='relation_id')
        
        # Convert to edge index and edge type
        self.kg_edge_index = torch.LongTensor(
            np.array([kg_graph.row, kg_graph.col])
        ).to(self.device)
        self.kg_edge_type = torch.LongTensor(kg_graph.data).to(self.device)
        
    def _build_lightgcn_adj(self, inter_matrix):
        """Build normalized adjacency matrix for LightGCN."""
        
        n_nodes = self.n_users + self.n_items
        
        # Build adjacency matrix
        users = inter_matrix.row
        items = inter_matrix.col + self.n_users
        
        # Create symmetric adjacency
        row = np.concatenate([users, items])
        col = np.concatenate([items, users])
        data = np.ones(len(row))
        
        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        
        # Normalize: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        
        # Convert to sparse tensor
        indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def _kg_propagate(self, entity_emb):
        """
        Multi-hop propagation on knowledge graph.
        
        Args:
            entity_emb: [n_entities, embedding_size] initial entity embeddings
            
        Returns:
            entity_emb_list: list of entity embeddings at each layer
        """
        entity_emb_list = [entity_emb]
        relation_emb = self.transr.relation_embedding.weight
        
        for layer in range(self.n_layers):
            # Aggregate neighbors
            aggregated = self.kg_aggregators[layer](
                entity_emb_list[-1],
                self.kg_edge_index,
                self.kg_edge_type,
                relation_emb,
                self.n_entities
            )
            
            # Apply dropout
            if self.training:
                aggregated = self.mess_dropout(aggregated)
            
            # Normalize
            aggregated = F.normalize(aggregated, p=2, dim=1)
            
            entity_emb_list.append(aggregated)
        
        return entity_emb_list

    def _cf_propagate(self, user_emb, item_emb):
        """
        Multi-hop propagation on user-item graph (LightGCN).
        
        Args:
            user_emb: [n_users, embedding_size] initial user embeddings
            item_emb: [n_items, embedding_size] initial item embeddings
            
        Returns:
            user_emb_list: list of user embeddings at each layer
            item_emb_list: list of item embeddings at each layer
        """
        user_emb_list = [user_emb]
        item_emb_list = [item_emb]
        
        for layer in range(self.n_layers):
            new_user_emb, new_item_emb = self.lightgcn_layers[layer](
                user_emb_list[-1],
                item_emb_list[-1],
                self.norm_adj
            )
            
            # Apply dropout
            if self.training:
                new_user_emb = self.mess_dropout(new_user_emb)
                new_item_emb = self.mess_dropout(new_item_emb)
            
            user_emb_list.append(new_user_emb)
            item_emb_list.append(new_item_emb)
        
        return user_emb_list, item_emb_list

    def forward(self):
        """
        Forward pass to compute user and item representations.
        
        Returns:
            user_emb: [n_users, embedding_size] final user embeddings
            item_emb: [n_items, embedding_size] final item embeddings
            kg_item_emb: [n_items, embedding_size] KG-based item embeddings
            cf_item_emb: [n_items, embedding_size] CF-based item embeddings
        """
        # Get initial embeddings
        user_emb_init = self.user_embedding.weight
        item_offset = self.item_offset.weight
        entity_emb_init = self.transr.entity_embedding.weight
        
        # KG propagation (structural view)
        entity_emb_list = self._kg_propagate(entity_emb_init)
        
        # Average across layers for KG embeddings
        kg_entity_emb = torch.stack(entity_emb_list, dim=0).mean(dim=0)
        kg_item_emb = kg_entity_emb[:self.n_items]  # Items are first n_items entities
        
        # CF propagation (collaborative view)
        # Use item offset + KG item embedding as initial item embedding for CF
        item_emb_for_cf = item_offset + kg_item_emb
        user_emb_list, item_emb_list = self._cf_propagate(user_emb_init, item_emb_for_cf)
        
        # Average across layers for CF embeddings
        cf_user_emb = torch.stack(user_emb_list, dim=0).mean(dim=0)
        cf_item_emb = torch.stack(item_emb_list, dim=0).mean(dim=0)
        
        # Final item embedding: e_j = η_j + v_j (combining CF and KG)
        # Following the paper: item latent vector combines knowledge base embeddings and CF offset
        final_item_emb = cf_item_emb + kg_item_emb
        final_user_emb = cf_user_emb
        
        return final_user_emb, final_item_emb, kg_item_emb, cf_item_emb

    def _compute_contrastive_loss(self, kg_emb, cf_emb, batch_indices):
        """
        Compute contrastive loss between KG and CF views.
        
        Args:
            kg_emb: [n_items, embedding_size] KG item embeddings
            cf_emb: [n_items, embedding_size] CF item embeddings
            batch_indices: [batch_size] indices of items in batch
            
        Returns:
            cl_loss: contrastive loss
        """
        # Get batch embeddings
        kg_batch = kg_emb[batch_indices]
        cf_batch = cf_emb[batch_indices]
        
        # Project to contrastive space
        kg_proj = F.normalize(self.fc_kg(kg_batch), p=2, dim=1)
        cf_proj = F.normalize(self.fc_cf(cf_batch), p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(kg_proj, cf_proj.t()) / self.temperature
        
        # Labels: positive pairs are on diagonal
        batch_size = batch_indices.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # InfoNCE loss (both directions)
        loss_kg_cf = F.cross_entropy(sim_matrix, labels)
        loss_cf_kg = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_kg_cf + loss_cf_kg) / 2

    def calculate_loss(self, interaction):
        """
        Calculate total loss.
        
        Args:
            interaction: interaction data
            
        Returns:
            loss: total loss
        """
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        # Forward pass
        user_emb, item_emb, kg_item_emb, cf_item_emb = self.forward()
        
        # Get batch embeddings
        u_emb = user_emb[user]
        pos_emb = item_emb[pos_item]
        neg_emb = item_emb[neg_item]
        
        # BPR loss for recommendation
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        rec_loss = self.bpr_loss(pos_scores, neg_scores)
        
        # Regularization loss
        u_emb_init = self.user_embedding(user)
        pos_offset = self.item_offset(pos_item)
        neg_offset = self.item_offset(neg_item)
        reg_loss = self.reg_loss(u_emb_init, pos_offset, neg_offset)
        
        # Contrastive loss
        unique_items = torch.unique(torch.cat([pos_item, neg_item]))
        cl_loss = self._compute_contrastive_loss(kg_item_emb, cf_item_emb, unique_items)
        
        # Total loss
        loss = rec_loss + self.reg_weight * reg_loss + self.cl_weight * cl_loss
        
        return loss

    def calculate_kg_loss(self, interaction):
        """
        Calculate KG embedding loss (TransR).
        
        Args:
            interaction: KG interaction data
            
        Returns:
            kg_loss: TransR loss
        """
        head = interaction[self.HEAD_ENTITY_ID]
        relation = interaction[self.RELATION_ID]
        pos_tail = interaction[self.TAIL_ENTITY_ID]
        neg_tail = interaction[self.NEG_TAIL_ENTITY_ID]
        
        # Compute TransR scores
        pos_score = self.transr(head, relation, pos_tail)
        neg_score = self.transr(head, relation, neg_tail)
        
        # Margin-based ranking loss (lower score is better)
        kg_loss = -torch.mean(F.logsigmoid(neg_score - pos_score))
        
        # Regularization for TransR parameters
        h_emb = self.transr.entity_embedding(head)
        pos_t_emb = self.transr.entity_embedding(pos_tail)
        neg_t_emb = self.transr.entity_embedding(neg_tail)
        r_emb = self.transr.relation_embedding(relation)
        
        reg_loss = self.reg_loss(h_emb, pos_t_emb, neg_t_emb, r_emb)
        
        return self.kg_weight * (kg_loss + self.reg_weight * reg_loss)

    def predict(self, interaction):
        """
        Predict scores for user-item pairs.
        
        Args:
            interaction: interaction data
            
        Returns:
            scores: predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_emb, item_emb, _, _ = self.forward()
        
        u_emb = user_emb[user]
        i_emb = item_emb[item]
        
        scores = torch.sum(u_emb * i_emb, dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        
        Args:
            interaction: interaction data with user IDs
            
        Returns:
            scores: [batch_size * n_items] predicted scores
        """
        user = interaction[self.USER_ID]
        
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _, _ = self.forward()
        
        u_emb = self.restore_user_e[user]
        
        # Score all items
        scores = torch.matmul(u_emb, self.restore_item_e.t())
        
        return scores.view(-1)