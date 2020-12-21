import torch
import torch.nn as nn

from .BaseModel import BaseModel


class Generator(nn.Module):
    def __init__(self, _inp_dim):
        super().__init__()
        self.in_dim = _inp_dim
        self.up_one = self.in_dim * 2
        self.up_two = self.up_one * 2
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.up_one),
            nn.ReLU(),
            nn.Linear(self.up_one, self.up_two),
            nn.ReLU(),
            nn.Linear(self.up_two, self.in_dim),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator(nn.Module):
    def __init__(self,  _inp_dim, _drop):
        self.in_dim = _inp_dim
        self.up_one = self.in_dim * 2
        self.out_dim = 1  # representing a probability
        self.dropout = _drop
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.up_one),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.up_one, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class BoilerplateGAN(BaseModel):

    def __init__(self, _sent_weights, _entity_weights, _relation_weights, _concat, _normalize_vecs, _beta, _tune):
        super().__init__()
        # torch.manual_seed(17)
        self.concat_method = _concat
        self.sent_weights = _sent_weights
        self.ent_weights = _entity_weights
        self.rel_weights = _relation_weights
        self.normalize_vecs = _normalize_vecs
        self.normalize_embeddings()
        self.beta = _beta
        self.update_embeddings = _tune

        self.sent_embedding, self.sent_dim = self.create_emb_layer(self.sent_weights, _tune)
        self.entity_embedding, self.ent_dim = self.create_emb_layer(self.ent_weights, _tune)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(self.rel_weights, _tune)
        self.kg_dim = self.compute_kg_dim()
        self.mapping = nn.Linear(self.kg_dim, self.sent_dim, bias=False)
        torch.nn.init.orthogonal_(self.mapping.weight)

        self.generator_sent = Generator(self.sent_dim)
        self.discrim_sent = Discriminator(self.sent_dim)
        self.generator_kg = Generator(self.kg_dim)
        self.discrim_kg = Discriminator(self.kg_dim)

