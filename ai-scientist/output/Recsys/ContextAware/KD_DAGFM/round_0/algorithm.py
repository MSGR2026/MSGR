# -*- coding: utf-8 -*-
# @Time   : 2024
# @Author : Anonymous
# @Email  : anonymous@example.com
# @File   : ffm.py

r"""
FFM
################################################
Reference:
    Yuchin Juan et al. "Field-aware Factorization Machines for CTR Prediction" in RecSys 2016.

Field-aware Factorization Machines (FFM) is a variant of FM that utilizes field information.
In FFMs, each feature has several latent vectors. Depending on the field of other features,
one of them is used to compute the inner product.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender


class FFM(ContextRecommender):
    """Field-aware Factorization Machines for CTR Prediction.
    
    FFM learns field-aware latent vectors for each feature. When computing the interaction
    between two features, FFM uses different latent vectors depending on the field of the
    other feature, allowing for more expressive feature interactions.
    
    The model is expressed as:
    phi_FFM(w, x) = sum_{j1=1}^{n} sum_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
    
    where f1 and f2 are the fields of j1 and j2 respectively.
    """

    input_type = None  # Uses default POINTWISE

    def __init__(self, config, dataset):
        super(FFM, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.num_fields = self.num_feature_field
        
        # In FFM, each feature needs a latent vector for each field
        # We create field-aware embeddings: for each field, we have embeddings for all features
        # Shape: (num_fields, num_features, embedding_size)
        # This is implemented as a list of embedding layers, one per target field
        
        # Get the total number of features (tokens + float fields)
        # For categorical features, we use the token embedding
        # For numerical features, we use separate embeddings
        
        # Build field-aware embedding layers
        # We need to create embeddings for each (feature, target_field) pair
        # For simplicity, we create num_fields embedding tables, each containing embeddings for all features
        
        # Calculate total feature vocabulary size
        self.token_field_offsets = self._get_token_field_offsets(dataset)
        self.num_token_features = self._get_total_token_features(dataset)
        self.num_float_features = len(self.float_field_names) if self.float_field_names else 0
        
        # Field-aware embeddings for token features
        # For each target field, we have an embedding table for all token features
        if self.num_token_features > 0:
            self.ffm_token_embeddings = nn.ModuleList([
                nn.Embedding(self.num_token_features, self.embedding_size)
                for _ in range(self.num_fields)
            ])
        
        # Field-aware embeddings for float features
        # For each target field, we have linear layers for float features
        if self.num_float_features > 0:
            self.ffm_float_embeddings = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(1, self.embedding_size, bias=False)
                    for _ in range(self.num_float_features)
                ])
                for _ in range(self.num_fields)
            ])
        
        # First-order weights (optional, for better performance)
        self.first_order_linear_layer = nn.Linear(self.num_fields * self.embedding_size, 1, bias=True)
        
        # Output layers
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # Parameters initialization
        self.apply(self._init_weights)

    def _get_token_field_offsets(self, dataset):
        """Get the offset for each token field in the concatenated embedding space."""
        offsets = {}
        current_offset = 0
        if self.token_field_names:
            for field_name in self.token_field_names:
                offsets[field_name] = current_offset
                # Get vocabulary size for this field
                vocab_size = dataset.num(field_name)
                current_offset += vocab_size
        return offsets

    def _get_total_token_features(self, dataset):
        """Get the total number of token features across all token fields."""
        total = 0
        if self.token_field_names:
            for field_name in self.token_field_names:
                total += dataset.num(field_name)
        return total

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # Initialize with uniform distribution [0, 1/sqrt(k)] as per paper
            nn.init.uniform_(module.weight.data, 0, 1.0 / (self.embedding_size ** 0.5))
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def _get_field_aware_embeddings(self, interaction, target_field_idx):
        """Get embeddings for all features when interacting with a specific target field.
        
        Args:
            interaction: The interaction data
            target_field_idx: The index of the target field
            
        Returns:
            embeddings: Tensor of shape [batch_size, num_fields, embedding_size]
        """
        batch_size = None
        embeddings_list = []
        field_idx = 0
        
        # Process token fields
        if self.token_field_names:
            for field_name in self.token_field_names:
                token_ids = interaction[field_name]  # [batch_size] or [batch_size, seq_len]
                if batch_size is None:
                    batch_size = token_ids.shape[0]
                
                # Add offset to get global feature index
                offset = self.token_field_offsets[field_name]
                global_ids = token_ids + offset
                
                # Handle sequence fields (take mean)
                if len(global_ids.shape) > 1:
                    # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
                    emb = self.ffm_token_embeddings[target_field_idx](global_ids)
                    # Mean pooling over sequence
                    emb = emb.mean(dim=1)  # [batch_size, embed_dim]
                else:
                    emb = self.ffm_token_embeddings[target_field_idx](global_ids)  # [batch_size, embed_dim]
                
                embeddings_list.append(emb)
                field_idx += 1
        
        # Process float fields
        if self.float_field_names:
            for i, field_name in enumerate(self.float_field_names):
                float_values = interaction[field_name]  # [batch_size]
                if batch_size is None:
                    batch_size = float_values.shape[0]
                
                # Reshape to [batch_size, 1] for linear layer
                float_values = float_values.unsqueeze(-1).float()
                emb = self.ffm_float_embeddings[target_field_idx][i](float_values)  # [batch_size, embed_dim]
                
                embeddings_list.append(emb)
                field_idx += 1
        
        # Stack all field embeddings
        embeddings = torch.stack(embeddings_list, dim=1)  # [batch_size, num_fields, embed_dim]
        return embeddings

    def _compute_ffm_interaction(self, interaction):
        """Compute the FFM second-order interaction term.
        
        phi_FFM(w, x) = sum_{j1=1}^{n} sum_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
        
        For categorical features with one-hot encoding, x_{j1} * x_{j2} = 1 when both are active.
        """
        batch_size = None
        
        # Get all field-aware embeddings
        # all_embeddings[target_field_idx] = embeddings for all features when target field is target_field_idx
        all_embeddings = []
        for target_field_idx in range(self.num_fields):
            emb = self._get_field_aware_embeddings(interaction, target_field_idx)
            all_embeddings.append(emb)
            if batch_size is None:
                batch_size = emb.shape[0]
        
        # all_embeddings: list of num_fields tensors, each [batch_size, num_fields, embed_dim]
        # Stack to [num_fields, batch_size, num_fields, embed_dim]
        all_embeddings = torch.stack(all_embeddings, dim=0)  # [num_target_fields, batch_size, num_source_fields, embed_dim]
        
        # Compute pairwise interactions
        # For fields f1 and f2 (f1 < f2):
        # interaction = (w_{f1, f2} · w_{f2, f1})
        # w_{f1, f2} = embedding of feature in field f1 when interacting with field f2
        # w_{f2, f1} = embedding of feature in field f2 when interacting with field f1
        
        interaction_sum = torch.zeros(batch_size, device=all_embeddings.device)
        
        for f1 in range(self.num_fields):
            for f2 in range(f1 + 1, self.num_fields):
                # w_{f1, f2}: embedding of field f1's feature for target field f2
                # This is all_embeddings[f2, :, f1, :]
                w_f1_f2 = all_embeddings[f2, :, f1, :]  # [batch_size, embed_dim]
                
                # w_{f2, f1}: embedding of field f2's feature for target field f1
                # This is all_embeddings[f1, :, f2, :]
                w_f2_f1 = all_embeddings[f1, :, f2, :]  # [batch_size, embed_dim]
                
                # Dot product
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