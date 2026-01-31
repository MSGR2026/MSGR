# -*- coding: utf-8 -*-
# @Time   : 2020/7/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# @File   : nfm.py

r"""
NFM
################################################
Reference:
    He X, Chua T S. "Neural factorization machines for sparse predictive analytics" in SIGIR 2017
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

# DIFFERENT: KD_DAGFM uses xavier_normal_initialization from recbole.model.init, no FM or MLP layers
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers


class NFM(ContextRecommender):
    """NFM replace the fm part as a mlp to model the feature interaction."""
    # DIFFERENT: KD_DAGFM uses teacher-student architecture with knowledge distillation phases

    def __init__(self, config, dataset):
        super(NFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # DIFFERENT: KD_DAGFM uses DAGFM student network with DAG-based feature interaction via einsum
        # define layers and loss
        size_list = [self.embedding_size] + self.mlp_hidden_size
        self.fm = BaseFactorizationMachine(reduce_sum=False)
        self.bn = nn.BatchNorm1d(num_features=self.embedding_size)
        self.mlp_layers = MLPLayers(
            size_list, self.dropout_prob, activation="sigmoid", bn=True
        )
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # DIFFERENT: KD_DAGFM uses BCELoss and combines CTR loss with KD loss during distillation
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        # REUSE: Both use concat_embed_input_fields to get embeddings [batch_size, num_field, embed_dim]
        nfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        # DIFFERENT: KD_DAGFM uses DAG-based FeatureInteraction instead of FM+MLP
        bn_nfm_all_embeddings = self.bn(self.fm(nfm_all_embeddings))

        output = self.predict_layer(
            self.mlp_layers(bn_nfm_all_embeddings)
        ) + self.first_order_linear(interaction)
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        # DIFFERENT: KD_DAGFM has phase-dependent loss: teacher_training/finetuning use BCE, distillation adds KD loss
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
