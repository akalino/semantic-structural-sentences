import sys
import torch
import torch.nn as nn


class GraphGemModel(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def compute_kg_dim(self):
        if self.concat_method == 'gem':
            kg_dim = self.triple_dim
        else:
            print('Unrecognized concatenation method, please see documentation.')
            sys.exit()
        return kg_dim

    def represent_triple(self, _triple):
        _norm_triple = nn.functional.normalize(self.triple_weights[_triple], p=2, dim=1)
        return _norm_triple

    def get_norm_method(self):
        return self.normalize_vecs

    def normalize_embeddings(self):
        if self.normalize_vecs == 'whiten':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._center_embeddings()
            self._whiten_embeddings()
        elif self.normalize_vecs == 'mean':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            self._scale_embeddings()
        elif self.normalize_vecs == 'both':
            print("Normalizing embeddings with {}".format(self.normalize_vecs))
            # Artexte - robust self learning
            self._scale_embeddings()
            self._center_embeddings()
            self._scale_embeddings()
        else:
            print('Warning: no normalization performed')

    def _center_embeddings(self):
        # this should mean center each dimension
        sent_mean = self.sent_weights.mean(axis=0)
        self.sent_weights.data = self.sent_weights - sent_mean
        triple_mean = self.triple_weights.mean(axis=0)
        self.triple_weights.data = self.triple_weights - triple_mean
        self.centered = True

    def _scale_embeddings(self):
        sent_norm = self.sent_weights.norm(p=2, dim=1, keepdim=True)
        self.sent_weights.data = self.sent_weights.div(sent_norm)
        triple_norm = self.triple_weights.norm(p=2, dim=1, keepdim=True)
        self.triple_weights.data = self.triple_weights.div(triple_norm)

    def _whiten_embeddings(self):
        if not self.centered:
            raise ValueError("Whitening needs centering to be done in advance")
        sent_cov = self.cov(self.sent_weights.T)
        u_sent, s_sent, v_sent = torch.svd(sent_cov)
        w_sent = (v_sent.T / torch.sqrt(s_sent)).T
        self.sent_weights.data = self.sent_weights.data @ w_sent.T
        triple_cov = self.cov(self.triple_weights.T)
        u_ent, s_ent, v_ent = torch.svd(ent_cov)
        w_ent = (v_ent.T / torch.sqrt(s_ent)).T
        self.triple_weights.data = self.triple_weights.data @ w_ent.T

    @staticmethod
    def cov(m, rowvar=True, inplace=False):
        """
        Estimate a covariance matrix given data.
        """
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        if inplace:
            m -= torch.mean(m, dim=1, keepdim=True)
        else:
            m = m - torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()

    @staticmethod
    def create_emb_layer(weights_matrix, trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if trainable:
            emb_layer.weight.requires_grad = True
        else:
            emb_layer.weight.requires_grad = False

        return emb_layer, embedding_dim

    def forward(self):
        pass
