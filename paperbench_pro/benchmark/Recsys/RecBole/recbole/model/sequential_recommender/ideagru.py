import torch
from torch import nn
from recbole.model.sequential_recommender import GRU4Rec
import random

class env_generator(nn.Module):
    def __init__(self, config):
        super(env_generator, self).__init__()
        self.device = config["device"]
        self.num_subseq = config["num_subseq"]
        self.len_seq = config["MAX_ITEM_LIST_LENGTH"]
        self.temp = config["temp"]
        # 修改: 统一使用 embedding_size
        self.latent_dim = config["embedding_size"]
        self.linear_in = nn.ModuleList([nn.Linear(self.len_seq, 128) for _ in range(self.num_subseq)])
        self.linear_out = nn.ModuleList([nn.Linear(128, self.len_seq) for _ in range(self.num_subseq)])
        self.activate = torch.nn.ReLU()


    def get_env_mask(self, seq_item_emb):
        seq_emb = torch.mean(seq_item_emb, dim=-1)
        all_masked_seq_emb = []
        for k in range(self.num_subseq):
            tmp_logit = self.activate(self.linear_in[k](seq_emb))
            logit = self.linear_out[k](tmp_logit)
            eps = torch.rand(logit.shape).to(self.device)
            eps = torch.log(eps) - torch.log(1-eps)
            mask = (logit + eps) / self.temp
            mask = torch.sigmoid(mask)
            mask_expand = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim)
            masked_seq_emb = mask_expand * seq_item_emb
            all_masked_seq_emb.append(masked_seq_emb)
        return all_masked_seq_emb
    

class IDEAGRU(GRU4Rec):
    def __init__(self, config, dataset):
        super(IDEAGRU, self).__init__(config, dataset)
        self.fake_ratio = config["fake_ratio"]
        self.num_env = config["num_env"]
        self.alpha = config["alpha"]
        # 修改: 统一使用 embedding_size
        self.latent_dim = config["embedding_size"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.var_hyper = config["var_hyper"]
        self.item_embedding = nn.Embedding(self.n_items + 1000, self.embedding_size, padding_idx=0)  # add 1000 fake items
        self.env = env_generator(config).to(self.device)


    def calculate_all_loss(self, interaction):
        seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        processed_item_seq = self.get_fake_seq(item_seq)
        seq_item_emb = self.item_embedding(processed_item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
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
    
    def get_env_loss(self, seq_output, pos_items):
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss
    
    def train_forward(self, item_seq_emb, item_seq_len):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output
    
    def env_mixup(self, all_masked_seq_emb):
        emb1, emb2 = random.sample(all_masked_seq_emb, 2)
        batch_size, _, _  = emb1.shape
        lambda_val = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size, 1, 1])
        lambda_val = lambda_val.expand(-1, self.seq_len, self.latent_dim).to(self.device)        
        mixed_emb = lambda_val * emb1 + (1 - lambda_val) * emb2
        return mixed_emb

    def get_fake_seq(self, item_seq):
        if self.fake_ratio == 0:
            return item_seq
        batch_size, _  = item_seq.size()
        fake_seq = item_seq.clone()
        for i in range(batch_size):
            seq_len = (item_seq[i] > 0).sum().item()            
            num_fake = max(1, int(self.fake_ratio * seq_len)) 
            fake_idxs = random.sample(range(seq_len), num_fake)
            fake_items = torch.randint(self.n_items, self.n_items + 1000, (num_fake,), device=self.device)
            fake_seq[i, fake_idxs] = fake_items
        return fake_seq
    
    def calculate_loss_env(self, interaction):
        _, var_erm = self.calculate_all_loss(interaction)
        return -var_erm
    
    def calculate_loss_il(self, interaction):
        mean_erm, var_erm = self.calculate_all_loss(interaction)
        return (mean_erm, self.var_hyper * var_erm)
