# -*- coding: utf-8 -*-
# @Time   : 2025/01/10
# @Author : Integrated from IDEA implementation

r"""
IDEAGRU
################################################

Reference:
    IDEA: Invariant Defense for Graph Adversarial Robustness (SIGIR 2025 Submission)

"""

import torch
from torch import nn
import random

from recbole.model.sequential_recommender.gru4rec import GRU4Rec


class EnvGenerator(nn.Module):
    """Environment generator for creating diverse training environments."""
    
    def __init__(self, config, device):
        super(EnvGenerator, self).__init__()
        self.device = device
        self.num_subseq = config["num_subseq"]
        self.len_seq = config["MAX_ITEM_LIST_LENGTH"]
        self.temp = config["temp"]
        self.latent_dim = config["embedding_size"]
        
        self.linear_in = nn.ModuleList([
            nn.Linear(self.len_seq, 128) for _ in range(self.num_subseq)
        ])
        self.linear_out = nn.ModuleList([
            nn.Linear(128, self.len_seq) for _ in range(self.num_subseq)
        ])
        self.activate = nn.ReLU()

    def get_env_mask(self, seq_item_emb):
        """Generate environment-specific masks for the sequence embeddings."""
        seq_emb = torch.mean(seq_item_emb, dim=-1)
        all_masked_seq_emb = []
        
        for k in range(self.num_subseq):
            tmp_logit = self.activate(self.linear_in[k](seq_emb))
            logit = self.linear_out[k](tmp_logit)
            eps = torch.rand(logit.shape).to(self.device)
            eps = torch.log(eps) - torch.log(1 - eps)
            mask = (logit + eps) / self.temp
            mask = torch.sigmoid(mask)
            mask_expand = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim)
            masked_seq_emb = mask_expand * seq_item_emb
            all_masked_seq_emb.append(masked_seq_emb)
            
        return all_masked_seq_emb


class IDEAGRU(GRU4Rec):
    r"""IDEAGRU extends GRU4Rec with Invariant Defense mechanism.
    
    This model introduces environment-based data augmentation and invariant learning
    to improve the robustness and generalization of sequential recommendation.
    """

    def __init__(self, config, dataset):
        super(IDEAGRU, self).__init__(config, dataset)
        
        # Load IDEA-specific parameters
        self.fake_ratio = config["fake_ratio"]
        self.num_env = config["num_env"]
        self.alpha = config["alpha"]
        self.latent_dim = config["embedding_size"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.var_hyper = config["var_hyper"]
        self.num_subseq = config["num_subseq"]
        self.temp = config["temp"]
        
        # Reinitialize item embedding with extra space for fake items
        self.item_embedding = nn.Embedding(
            self.n_items + 1000, 
            self.embedding_size, 
            padding_idx=0
        )
        
        # Initialize environment generator
        self.env = EnvGenerator(config, self.device).to(self.device)
        
        # Apply initialization
        self.apply(self._init_weights)

    def get_fake_seq(self, item_seq):
        """Inject fake items into the sequence for data augmentation."""
        if self.fake_ratio == 0:
            return item_seq
            
        batch_size, _ = item_seq.size()
        fake_seq = item_seq.clone()
        
        for i in range(batch_size):
            seq_len = (item_seq[i] > 0).sum().item()
            if seq_len == 0:
                continue
                
            num_fake = max(1, int(self.fake_ratio * seq_len))
            fake_idxs = random.sample(range(seq_len), num_fake)
            fake_items = torch.randint(
                self.n_items, 
                self.n_items + 1000, 
                (num_fake,), 
                device=self.device
            )
            fake_seq[i, fake_idxs] = fake_items
            
        return fake_seq

    def env_mixup(self, all_masked_seq_emb):
        """Mix up embeddings from different environments."""
        emb1, emb2 = random.sample(all_masked_seq_emb, 2)
        batch_size, _, _ = emb1.shape
        
        lambda_val = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size, 1, 1])
        lambda_val = lambda_val.expand(-1, self.seq_len, self.latent_dim).to(self.device)
        mixed_emb = lambda_val * emb1 + (1 - lambda_val) * emb2
        
        return mixed_emb

    def train_forward(self, item_seq_emb, item_seq_len):
        """Forward pass for training with custom embeddings."""
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def get_env_loss(self, seq_output, pos_items):
        """Calculate environment-specific loss."""
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def calculate_all_loss(self, interaction):
        """Calculate losses across all environments."""
        seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        
        # Apply fake item injection
        processed_item_seq = self.get_fake_seq(item_seq)
        seq_item_emb = self.item_embedding(processed_item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Generate environment-specific masked embeddings
        all_erm_loss = []
        all_masked_seq_emb = self.env.get_env_mask(seq_item_emb)
        
        for _ in range(self.num_env):
            mixed_seq_emb = self.env_mixup(all_masked_seq_emb).to(self.device)
            mixed_seq_output = self.train_forward(mixed_seq_emb, seq_len)
            erm_loss = self.get_env_loss(mixed_seq_output, pos_items)
            all_erm_loss.append(erm_loss)
            
        var_erm = torch.var(torch.stack(all_erm_loss))
        mean_erm = torch.mean(torch.stack(all_erm_loss))
        
        return mean_erm, var_erm

    def calculate_loss(self, interaction):
        """Standard loss calculation (used in pretraining phase)."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def calculate_loss_env(self, interaction):
        """Calculate environment loss for adversarial training."""
        _, var_erm = self.calculate_all_loss(interaction)
        return -var_erm

    def calculate_loss_il(self, interaction):
        """Calculate invariant learning loss."""
        mean_erm, var_erm = self.calculate_all_loss(interaction)
        return mean_erm, self.var_hyper * var_erm
