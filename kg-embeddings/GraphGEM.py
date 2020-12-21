import torch
import torch.nn as nn

from tqdm import tqdm


class GeometricGraphEmbedder:

    def __init__(self,
                 triples,
                 entity_matrix,
                 relation_matrix,
                 k, h, s,
                 _tune):
        if len(triples) == 1:
            self.triples = triples[0]
        else:
            self.triples = triples
        self.entity_embedding, self.ent_dim = self.create_emb_layer(entity_matrix, _tune)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(relation_matrix, _tune)
        self.singular_values = None
        self.k = k
        self.h = h
        self.sigma = s
        self.triple_embeddings, self.triple_idx = self.gem()
        # self.gem_weights, self.triple_idx = self.gem()
        # self.triple_embeddings = self.create_emb_layer(self.gem_weights, _tune)

    def gem(self):
        X = torch.zeros((self.ent_dim, len(self.triples)))
        stacked_triples = []
        triple_idx = []
        print('Processing {n} triples'.format(n=len(self.triples)))
        for i, triple in tqdm(enumerate(self.triples)):
            _triple_mat = torch.stack([self.entity_embedding(triple[0]),
                                       self.entity_embedding(triple[2]),
                                       self.rel_embedding(triple[1])])
            triple_idx.append(_triple_mat.tolist())
            stacked_triples.append(_triple_mat.T)
            U, s, Vh = torch.svd(_triple_mat.T)
            p = U @ (s ** self.sigma)
            X[:, i] = p
        # https://github.com/pytorch/pytorch/issues/24900
        D, s, _ = torch.svd(X.T)  # Transpose for memory efficiencies in pytorch
        self.singular_values = s.T
        D = D.T[:, :self.k]
        s = s.T[:self.k]
        C = torch.zeros((self.ent_dim, len(self.triples)))
        for j, sent in tqdm(enumerate(self.triples)):
            embedded_triple = stacked_triples[j]
            order = s * torch.norm(torch.mm(embedded_triple.T, D))
            toph = order.argsort()[:self.h]
            alpha = torch.zeros(embedded_triple.shape[1])
            for i in range(embedded_triple.shape[1]):
                Q, R = self.modified_gram_schmidt_qr(embedded_triple)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = torch.exp(r[-1] / (torch.norm(r, dim=0)) + 1e-18)
                alpha_s = r[-1] / embedded_triple.shape[1]
                alpha_u = torch.exp(-torch.norm(s[toph] * (q.T @ D[:, toph])) / self.h)
                alpha[i] = alpha_n + alpha_s + alpha_u
            C[:, j] = embedded_triple @ alpha
            C[:, j] = C[:, j] - D @ (D.T @ C[:, j])
        triple_embeddings = C.T
        return triple_embeddings, triple_idx

    def lookup_triple(self, _h, _r, _t):
        h_emb = self.entity_embedding(_h)
        t_emb = self.entity_embedding(_t)
        r_emb = self.rel_embedding(_r)
        trip_rep = torch.stack([h_emb, t_emb, r_emb])
        idx = self.triple_idx.index(trip_rep.tolist())
        return self.triple_embeddings[idx]

    def lookup_batched_triple(self, _batch):
        out = []
        for _triple in tqdm(_batch):
            t = self.lookup_triple(_triple[0], _triple[1], _triple[2])
            out.append(t)
        return torch.stack(out)

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

    @staticmethod
    def modified_gram_schmidt_qr(_a):
        n_rows, n_cols = _a.shape
        q = torch.zeros((n_rows, n_cols))
        r = torch.zeros((n_rows, n_cols))
        for j in range(n_cols):
            u = torch.clone(_a[:, j])
            for i in range(j):
                proj1 = torch.dot(u, q[:, i])
                proj = proj1 * q[:, i]
                u -= proj
            u_norm = torch.norm(u, dim=0)
            if u_norm != 0:
                u /= u_norm
            q[:, j] = u
        for j in range(n_cols):
            for i in range(j + 1):
                r[i, j] = torch.dot(_a[:, j], q[:, i])
        return q, r
