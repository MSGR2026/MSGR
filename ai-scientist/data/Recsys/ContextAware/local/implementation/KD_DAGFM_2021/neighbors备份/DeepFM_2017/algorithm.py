# -*- coding: utf-8 -*-
# @Time   : 2020/7/8
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : deepfm.py

# UPDATE:
# @Time   : 2020/8/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
DeepFM
################################################
Reference:
    Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers

# DIFFERENT: KD_DAGFM uses knowledge distillation with teacher/student networks; DeepFM is a single model
class DeepFM(ContextRecommender):
    """DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
    Also DeepFM can be seen as a combination of FNN and FM.

    """

    def __init__(self, config, dataset):
        super(DeepFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        # DIFFERENT: KD_DAGFM uses DAG-based feature interaction with depth parameter; DeepFM uses FM + MLP
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size
        # DIFFERENT: KD_DAGFM student uses einsum-based graph aggregation; DeepFM uses standard MLP layers
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1
        )  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()
        # SIMILAR: Both use BCE-based loss functions
        self.loss = nn.BCEWithLogitsLoss()

        # REUSE: Both apply weight initialization via self.apply()
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
        deepfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        batch_size = deepfm_all_embeddings.shape[0]
        # DIFFERENT: KD_DAGFM routes through FeatureInteraction based on phase; DeepFM combines FM + deep
        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
        )
        y = y_fm + y_deep
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        # DIFFERENT: KD_DAGFM has phase-dependent loss with KD loss (alpha*ctr + beta*kd); DeepFM uses simple BCE
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
