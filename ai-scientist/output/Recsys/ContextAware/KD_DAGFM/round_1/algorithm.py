# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com
# @File   : kd_dagfm.py

r"""
FFM (Field-aware Factorization Machines)
################################################
Reference:
    Yuchin Juan et al. "Field-aware Factorization Machines for CTR Prediction" in RecSys 2016.

Field-aware Factorization Machines (FFM) is a variant of FM that utilizes field information.
In FFMs, each feature has several latent vectors. Depending on the field of other features,
one of them is used to compute the inner product.

The model is expressed as:
phi_FFM(w, x) = sum_{j1=1}^{n} sum_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}

where f1 and f2 are the fields of j1 and j2 respectively.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender


class KD_DAGFM(ContextRecommender):
    """Field-aware Factorization Machines for CTR Prediction.
    
    FFM learns field-aware latent vectors for each feature. When computing the interaction
    between two features, FFM uses different latent vectors depending on the field of the
    other feature, allowing for more expressive feature interactions.
    
    The model is expressed as:
    phi_FFM(w, x) = sum_{j1=1}^{n} sum_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
    
    where f1 and f2 are the fields of j1 and j2 respectively.
    """

    def __init__(self, config, dataset):
        super(KD_DAGFM, self).__init__(config, dataset)

        # Load parameters info
        self.num_fields = self.num_feature_field
        
        # In FFM, each feature in each field needs a latent vector for each target field
        # We implement this using a 3D embedding structure
        # For each source field, we create num_fields embedding tables (one for each target field)
        
        # Build field-aware embedding layers for token features
        # ffm_embeddings[source_field][target_field] gives embeddings when source_field interacts with target_field
        
        self.token_field_dims = []
        self.token_field_names_list = []
        
        if self.token_field_names:
            for field_name in self.token_field_names:
                vocab_size = dataset.num(field_name)
                self.token_field_dims.append(vocab_size)
                self.token_field_names_list.append(field_name)
        
        self.num_token_fields = len(self.token_field_dims)
        
        # Float field handling
        self.float_field_names_list = []
        if self.float_field_names:
            self.float_field_names_list = list(self.float_field_names)
        self.num_float_fields = len(self.float_field_names_list)
        
        # Create field-aware embeddings for token features
        # For each (source_field, target_field) pair, we need an embedding table
        if self.num_token_fields > 0:
            self.ffm_token_embeddings = nn.ModuleList()
            for src_idx in range(self.num_token_fields):
                target_embeddings = nn.ModuleList()
                for tgt_idx in range(self.num_fields):
                    emb = nn.Embedding(self.token_field_dims[src_idx], self.embedding_size)
                    target_embeddings.append(emb)
                self.ffm_token_embeddings.append(target_embeddings)
        
        # Create field-aware embeddings for float features
        # Each float feature needs a weight vector for each target field
        if self.num_float_fields > 0:
            self.ffm_float_embeddings = nn.ModuleList()
            for src_idx in range(self.num_float_fields):
                target_embeddings = nn.ModuleList()
                for tgt_idx in range(self.num_fields):
                    linear = nn.Linear(1, self.embedding_size, bias=False)
                    target_embeddings.append(linear)
                self.ffm_float_embeddings.append(target_embeddings)
        
        # Output layers
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # Parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # Initialize with uniform distribution [0, 1/sqrt(k)] as per paper
            nn.init.uniform_(module.weight.data, 0, 1.0 / (self.embedding_size ** 0.5))
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def _get_ffm_embedding(self, interaction, source_field_idx, target_field_idx):
        """Get the embedding for a source field when interacting with a target field.
        
        Args:
            interaction: The interaction data
            source_field_idx: Index of the source field (0 to num_fields-1)
            target_field_idx: Index of the target field (0 to num_fields-1)
            
        Returns:
            embedding: Tensor of shape [batch_size, embedding_size]
        """
        # Determine if source field is token or float
        if source_field_idx < self.num_token_fields:
            # Token field
            field_name = self.token_field_names_list[source_field_idx]
            token_ids = interaction[field_name]  # [batch_size] or [batch_size, seq_len]
            
            emb_layer = self.ffm_token_embeddings[source_field_idx][target_field_idx]
            
            if len(token_ids.shape) > 1:
                # Sequence field - mean pooling
                emb = emb_layer(token_ids)  # [batch_size, seq_len, embed_dim]
                emb = emb.mean(dim=1)  # [batch_size, embed_dim]
            else:
                emb = emb_layer(token_ids)  # [batch_size, embed_dim]
            
            return emb
        else:
            # Float field
            float_idx = source_field_idx - self.num_token_fields
            field_name = self.float_field_names_list[float_idx]
            float_values = interaction[field_name]  # [batch_size]
            
            # Reshape to [batch_size, 1]
            float_values = float_values.unsqueeze(-1).float()
            
            linear_layer = self.ffm_float_embeddings[float_idx][target_field_idx]
            emb = linear_layer(float_values)  # [batch_size, embed_dim]
            
            return emb

    def _compute_ffm_interaction(self, interaction):
        """Compute the FFM second-order interaction term.
        
        phi_FFM(w, x) = sum_{j1=1}^{n} sum_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
        
        For categorical features with one-hot encoding, x_{j1} * x_{j2} = 1 when both are active.
        In our case, each field has exactly one active feature, so we compute:
        sum_{f1=1}^{F} sum_{f2=f1+1}^{F} (w_{f1,f2} · w_{f2,f1})
        
        where w_{f1,f2} is the embedding of the active feature in field f1 for target field f2.
        """
        # Get batch size from first available field
        if self.num_token_fields > 0:
            first_field = self.token_field_names_list[0]
            batch_size = interaction[first_field].shape[0]
        else:
            first_field = self.float_field_names_list[0]
            batch_size = interaction[first_field].shape[0]
        
        device = interaction[first_field].device
        interaction_sum = torch.zeros(batch_size, device=device)
        
        # Compute pairwise field interactions
        for f1 in range(self.num_fields):
            for f2 in range(f1 + 1, self.num_fields):
                # w_{f1, f2}: embedding of feature in field f1 for target field f2
                w_f1_f2 = self._get_ffm_embedding(interaction, f1, f2)  # [batch_size, embed_dim]
                
                # w_{f2, f1}: embedding of feature in field f2 for target field f1
                w_f2_f1 = self._get_ffm_embedding(interaction, f2, f1)  # [batch_size, embed_dim]
                
                # Dot product: (w_{f1,f2} · w_{f2,f1})
                interaction_term = (w_f1_f2 * w_f2_f1).sum(dim=-1)  # [batch_size]
                interaction_sum = interaction_sum + interaction_term
        
        return interaction_sum  # [batch_size]

    def forward(self, interaction):
        """Forward pass of FFM.
        
        Args:
            interaction: The interaction data containing features
            
        Returns:
            output: Logits of shape [batch_size]
        """
        # First-order term (using standard embeddings from parent class)
        first_order = self.first_order_linear(interaction)  # [batch_size, 1]
        
        # Second-order FFM interaction term
        second_order = self._compute_ffm_interaction(interaction)  # [batch_size]
        
        # Combine
        output = first_order.squeeze(-1) + second_order
        
        return output

    def calculate_loss(self, interaction):
        """Calculate the training loss.
        
        Args:
            interaction: The interaction data
            
        Returns:
            loss: The BCE loss value
        """
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        """Predict the click probability.
        
        Args:
            interaction: The interaction data
            
        Returns:
            prob: Click probabilities of shape [batch_size]
        """
        return self.sigmoid(self.forward(interaction))