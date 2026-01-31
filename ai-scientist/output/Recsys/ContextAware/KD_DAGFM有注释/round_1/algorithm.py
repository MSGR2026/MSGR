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


class KD_DAGFM(ContextRecommender):
    """Field-aware Factorization Machines for CTR Prediction.
    
    FFM learns a dedicated latent vector for each feature-field combination.
    The interaction between two features uses the latent vectors corresponding
    to each other's fields.
    
    The model equation is:
    φ_FFM(w, x) = Σ_{j1=1}^{n} Σ_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
    
    where f1 and f2 are the fields of j1 and j2 respectively.
    """

    input_type = None  # Use default POINTWISE

    def __init__(self, config, dataset):
        super(KD_DAGFM, self).__init__(config, dataset)

        # Load parameters info
        self.embedding_size = config["embedding_size"]
        self.num_fields = self.num_feature_field
        
        # In FFM, each feature needs a latent vector for each field
        # We create field-aware embeddings: for each field, we have embeddings for all features
        # Shape: [num_fields, num_features, embedding_size]
        # But since we use separate embeddings per token field, we need to handle this differently
        
        # Get the total number of features (tokens) across all fields
        # We'll create field-aware embedding tables
        
        # For FFM, we need to know the field index for each feature
        # In the context-aware setting, each token field corresponds to one field
        
        # Create field-aware embeddings
        # For each original embedding table, we create num_fields copies
        self.ffm_embeddings = nn.ModuleDict()
        
        # Get token field information from parent class
        # token_field_names contains the names of categorical feature fields
        # float_field_names contains the names of numerical feature fields
        
        # For token fields, create field-aware embeddings
        if self.token_field_names is not None:
            for field_idx, field_name in enumerate(self.token_field_names):
                # Get the number of unique values for this field
                num_tokens = self.token_field_dims[field_idx]
                # Create num_fields embedding tables for this field
                field_embeddings = nn.ModuleList([
                    nn.Embedding(num_tokens, self.embedding_size, padding_idx=0)
                    for _ in range(self.num_fields)
                ])
                self.ffm_embeddings[field_name] = field_embeddings
        
        # For float fields, create field-aware linear transformations
        self.ffm_float_embeddings = nn.ModuleDict()
        if self.float_field_names is not None:
            for field_name in self.float_field_names:
                # Create num_fields linear layers for this field
                float_embeddings = nn.ModuleList([
                    nn.Linear(1, self.embedding_size, bias=False)
                    for _ in range(self.num_fields)
                ])
                self.ffm_float_embeddings[field_name] = float_embeddings
        
        # Build field index mapping
        self._build_field_index_mapping()
        
        # First order linear term (optional, can be included for better performance)
        # Already provided by parent class through first_order_linear()
        
        # Output layers
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # Parameters initialization
        self.apply(self._init_weights)

    def _build_field_index_mapping(self):
        """Build mapping from field name to field index."""
        self.field_name_to_idx = {}
        idx = 0
        
        if self.token_field_names is not None:
            for field_name in self.token_field_names:
                self.field_name_to_idx[field_name] = idx
                idx += 1
        
        if self.float_field_names is not None:
            for field_name in self.float_field_names:
                self.field_name_to_idx[field_name] = idx
                idx += 1
        
        self.field_names = list(self.field_name_to_idx.keys())

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_ffm_embeddings(self, interaction, target_field_idx):
        """Get embeddings for all features, using the latent vectors for target_field_idx.
        
        Args:
            interaction: Input interaction data
            target_field_idx: The target field index to use for selecting latent vectors
            
        Returns:
            embeddings: [batch_size, num_fields, embedding_size]
        """
        embeddings_list = []
        
        # Process token fields
        if self.token_field_names is not None:
            for field_name in self.token_field_names:
                # Get the input token indices for this field
                token_input = interaction[field_name]  # [batch_size] or [batch_size, seq_len]
                
                # Get the embedding table for this field targeting target_field_idx
                embedding_table = self.ffm_embeddings[field_name][target_field_idx]
                
                # Get embeddings
                if len(token_input.shape) == 1:
                    emb = embedding_table(token_input)  # [batch_size, embedding_size]
                else:
                    # Handle sequence case by averaging
                    emb = embedding_table(token_input).mean(dim=1)  # [batch_size, embedding_size]
                
                embeddings_list.append(emb)
        
        # Process float fields
        if self.float_field_names is not None:
            for field_name in self.float_field_names:
                # Get the input float values for this field
                float_input = interaction[field_name]  # [batch_size] or [batch_size, 1]
                
                if len(float_input.shape) == 1:
                    float_input = float_input.unsqueeze(-1)  # [batch_size, 1]
                
                # Get the linear transformation for this field targeting target_field_idx
                linear_layer = self.ffm_float_embeddings[field_name][target_field_idx]
                
                # Get embeddings
                emb = linear_layer(float_input)  # [batch_size, embedding_size]
                
                embeddings_list.append(emb)
        
        # Stack all embeddings: [batch_size, num_fields, embedding_size]
        embeddings = torch.stack(embeddings_list, dim=1)
        
        return embeddings

    def ffm_interaction(self, interaction):
        """Compute FFM second-order feature interactions.
        
        φ_FFM(w, x) = Σ_{j1=1}^{n} Σ_{j2=j1+1}^{n} (w_{j1,f2} · w_{j2,f1}) x_{j1} x_{j2}
        
        In our setting, each field has one feature (the token/float value), so:
        φ_FFM = Σ_{f1=1}^{F} Σ_{f2=f1+1}^{F} (w_{f1,f2} · w_{f2,f1})
        
        where w_{f1,f2} is the embedding of field f1 for interacting with field f2.
        
        Returns:
            interaction_output: [batch_size, 1]
        """
        batch_size = interaction[self.field_names[0]].shape[0]
        
        # Compute pairwise interactions
        interaction_sum = torch.zeros(batch_size, device=self.device)
        
        for f1 in range(self.num_fields):
            for f2 in range(f1 + 1, self.num_fields):
                # Get embedding of field f1 for interacting with field f2
                # w_{f1, f2}: embedding of feature in field f1, using latent vector for field f2
                emb_f1_for_f2 = self._get_single_field_embedding(interaction, f1, f2)
                
                # Get embedding of field f2 for interacting with field f1
                # w_{f2, f1}: embedding of feature in field f2, using latent vector for field f1
                emb_f2_for_f1 = self._get_single_field_embedding(interaction, f2, f1)
                
                # Compute inner product: (w_{f1,f2} · w_{f2,f1})
                # [batch_size, embedding_size] * [batch_size, embedding_size] -> [batch_size]
                inner_product = (emb_f1_for_f2 * emb_f2_for_f1).sum(dim=-1)
                
                interaction_sum = interaction_sum + inner_product
        
        return interaction_sum.unsqueeze(-1)  # [batch_size, 1]

    def _get_single_field_embedding(self, interaction, source_field_idx, target_field_idx):
        """Get the embedding of a single field for interacting with another field.
        
        Args:
            interaction: Input interaction data
            source_field_idx: The field index of the source feature
            target_field_idx: The target field index (determines which latent vector to use)
            
        Returns:
            embedding: [batch_size, embedding_size]
        """
        field_name = self.field_names[source_field_idx]
        
        # Check if it's a token field or float field
        if self.token_field_names is not None and field_name in self.token_field_names:
            token_input = interaction[field_name]
            embedding_table = self.ffm_embeddings[field_name][target_field_idx]
            
            if len(token_input.shape) == 1:
                emb = embedding_table(token_input)
            else:
                emb = embedding_table(token_input).mean(dim=1)
            return emb
        
        elif self.float_field_names is not None and field_name in self.float_field_names:
            float_input = interaction[field_name]
            if len(float_input.shape) == 1:
                float_input = float_input.unsqueeze(-1)
            
            linear_layer = self.ffm_float_embeddings[field_name][target_field_idx]
            emb = linear_layer(float_input)
            return emb
        
        else:
            raise ValueError(f"Unknown field name: {field_name}")

    def forward(self, interaction):
        """Forward pass of FFM.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            output: [batch_size] logits
        """
        # First order term (linear)
        first_order = self.first_order_linear(interaction)  # [batch_size, 1]
        
        # Second order FFM interactions
        second_order = self.ffm_interaction(interaction)  # [batch_size, 1]
        
        # Combine
        output = first_order + second_order
        
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        """Calculate the training loss.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            loss: Scalar loss value
        """
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        """Predict the click probability.
        
        Args:
            interaction: Input interaction data
            
        Returns:
            prediction: [batch_size] click probabilities
        """
        return self.sigmoid(self.forward(interaction))