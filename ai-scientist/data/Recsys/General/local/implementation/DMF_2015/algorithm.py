# _*_ coding: utf-8 _*_
# @Time : 2020/8/21
# @Author : Kaizhou Zhang
# @Email  : kaizhou361@163.com

# UPDATE
# @Time   : 2020/08/31  2020/09/18
# @Author : Kaiyuan Li  Zihan Lin
# @email  : tsotfsk@outlook.com  linzihan.super@foxmail.con

r"""
DMF
################################################
Reference:
    Hong-Jian Xue et al. "Deep Matrix Factorization Models for Recommender Systems." in IJCAI 2017.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender

from recbole.utils import InputType


# ===== Inlined Components =====
class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score
# ===== End Inlined Components =====


class DMF(GeneralRecommender):
    r"""DMF is an neural network enhanced matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}` is set as model input,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    We just implement the model following the original author with a pointwise training mode.

    Note:

        Our implementation is a improved version which is different from the original paper.
        For a better performance and stability, we replace cosine similarity to inner-product when calculate
        final score of user's and item's embedding.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config["RATING_FIELD"]

        # load parameters info
        self.user_embedding_size = config["user_embedding_size"]
        self.item_embedding_size = config["item_embedding_size"]
        self.user_hidden_size_list = config["user_hidden_size_list"]
        self.item_hidden_size_list = config["item_hidden_size_list"]
        # The dimensions of the last layer of users and items must be the same
        assert self.user_hidden_size_list[-1] == self.item_hidden_size_list[-1]
        self.inter_matrix_type = config["inter_matrix_type"]

        # generate intermediate data
        if self.inter_matrix_type == "01":
            (
                self.history_user_id,
                self.history_user_value,
                _,
            ) = dataset.history_user_matrix()
            (
                self.history_item_id,
                self.history_item_value,
                _,
            ) = dataset.history_item_matrix()
            self.interaction_matrix = dataset.inter_matrix(form="csr").astype(
                np.float32
            )
        elif self.inter_matrix_type == "rating":
            (
                self.history_user_id,
                self.history_user_value,
                _,
            ) = dataset.history_user_matrix(value_field=self.RATING)
            (
                self.history_item_id,
                self.history_item_value,
                _,
            ) = dataset.history_item_matrix(value_field=self.RATING)
            self.interaction_matrix = dataset.inter_matrix(
                form="csr", value_field=self.RATING
            ).astype(np.float32)
        else:
            raise ValueError(
                "The inter_matrix_type must in ['01', 'rating'] but get {}".format(
                    self.inter_matrix_type
                )
            )
        self.max_rating = self.history_user_value.max()
        # tensor of shape [n_items, H] where H is max length of history interaction.
        self.history_user_id = self.history_user_id.to(self.device)
        self.history_user_value = self.history_user_value.to(self.device)
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        # define layers
        self.user_linear = nn.Linear(
            in_features=self.n_items, out_features=self.user_embedding_size, bias=False
        )
        self.item_linear = nn.Linear(
            in_features=self.n_users, out_features=self.item_embedding_size, bias=False
        )
        self.user_fc_layers = MLPLayers(
            [self.user_embedding_size] + self.user_hidden_size_list
        )
        self.item_fc_layers = MLPLayers(
            [self.item_embedding_size] + self.item_hidden_size_list
        )
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Save the item embedding before dot product layer to speed up evaluation
        self.i_embedding = None

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ["i_embedding"]

    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, user, item):
        user = self.get_user_embedding(user)

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        col_indices = self.history_user_id[item].flatten()
        row_indices = (
            torch.arange(item.shape[0])
            .to(self.device)
            .repeat_interleave(self.history_user_id.shape[1], dim=0)
        )
        matrix_01 = torch.zeros(1).to(self.device).repeat(item.shape[0], self.n_users)
        matrix_01.index_put_(
            (row_indices, col_indices), self.history_user_value[item].flatten()
        )
        item = self.item_linear(matrix_01)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        # cosine distance is replaced by dot product according the result of our experiments.
        vector = torch.mul(user, item).sum(dim=1)

        return vector

    def calculate_loss(self, interaction):
        # when starting a new epoch, the item embedding we saved must be cleared.
        if self.training:
            self.i_embedding = None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == "01":
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == "rating":
            label = interaction[self.RATING] * interaction[self.LABEL]
        output = self.forward(user, item)

        label = label / self.max_rating  # normalize the label to calculate BCE loss.
        loss = self.bce_loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        predict = self.sigmoid(self.forward(user, item))
        return predict

    def get_user_embedding(self, user):
        r"""Get a batch of user's embedding with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        matrix_01 = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        matrix_01.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        user = self.user_linear(matrix_01)

        return user

    def get_item_embedding(self):
        r"""Get all item's embedding with history interaction matrix.

        Considering the RAM of device, we use matrix multiply on sparse tensor for generalization.

        Returns:
            torch.FloatTensor: The embedding tensor of all item, shape: [n_items, embedding_size]
        """
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = (
            torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape))
            .to(self.device)
            .transpose(0, 1)
        )
        item = torch.sparse.mm(item_matrix, self.item_linear.weight.t())

        item = self.item_fc_layers(item)
        return item

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.get_user_embedding(user)
        u_embedding = self.user_fc_layers(u_embedding)

        if self.i_embedding is None:
            self.i_embedding = self.get_item_embedding()

        similarity = torch.mm(u_embedding, self.i_embedding.t())
        similarity = self.sigmoid(similarity)
        return similarity.view(-1)
