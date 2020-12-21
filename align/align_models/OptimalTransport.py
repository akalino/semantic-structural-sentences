import pdb
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg

import ot

try:  # test if cudamat installed
    from ot.gpu import bregman
    from gw_optim_gpu import gromov_wass_solver
except ImportError:
    from .gw_optim import gromov_wass_solver

from .gw_optim import orth_procrustes
from .BaseModel import BaseModel


def zipf_init(lang, n):
    # See (Piantadosi, 2014)
    if lang == 'en':
        alpha, beta = 1.40, 1.88 #Other sources give: 1.13, 2.73
    elif lang == 'fi':
        alpha, beta = 1.17, 0.60
    elif lang == 'fr':
        alpha, beta = 1.71, 2.09
    elif lang == 'de':
        alpha, beta = 1.10, 0.40
    elif lang == 'es':
        alpha, beta = 1.84, 3.81
    else: # Deafult to EN
        alpha, beta = 1.40, 1.88
    p = np.array([1/((i+1)+beta)**alpha for i in range(n)])
    return p/p.sum()


def csls(scores, knn=5):
        """
            Adapted from Conneau et al.

            rt = [1/k *  sum_{zt in knn(xs)} score(xs, zt)
            rs = [1/k *  sum_{zs in knn(xt)} score(zs, xt)
            csls(x_s, x_t) = 2*score(xs, xt) - rt - rs

        """

        def mean_similarity(scores, knn, axis=1):
            nghbs = np.argpartition(scores, -knn,
                                    axis=axis)  # for rows #[-k:] # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
            # TODO: There must be a faster way to do this slicing
            if axis == 1:
                nghbs = nghbs[:, -knn:]
                nghbs_score = np.concatenate([row[indices] for row, indices in zip(scores, nghbs)]).reshape(nghbs.shape)
            else:
                nghbs = nghbs[-knn:, :].T
                nghbs_score = np.concatenate([col[indices] for col, indices in zip(scores.T, nghbs)]).reshape(
                    nghbs.shape)

            return nghbs_score.mean(axis=1)

        # 1. Compute mean similarity return_scores
        src_ms = mean_similarity(scores, knn, axis=1)
        trg_ms = mean_similarity(scores, knn, axis=0)
        # 2. Compute updated scores
        normalized_scores = ((2 * scores - trg_ms).T - src_ms).T
        return normalized_scores


class GWTransport(BaseModel):

    def __init__(self, _sent_weights, _entity_weights,
                 _relation_weights, _concat, _normalize_vecs,
                 _beta, _tune, _batch_sz,
                 _distribs, _adjust, _tol, _score_type):
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
        self.adjust = _adjust
        self.ns = _batch_sz
        self.nt = _batch_sz
        self.tol = _tol
        self.metric = 'euclidean'

        self.sent_embedding, self.sent_dim = self.create_emb_layer(self.sent_weights, _tune)
        self.entity_embedding, self.ent_dim = self.create_emb_layer(self.ent_weights, _tune)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(self.rel_weights, _tune)
        self.kg_dim = self.compute_kg_dim()
        self.mapping = np.eye(self.sent_embedding.weight.shape[1])

        if _distribs == 'uniform':
            self.p = ot.unif(self.ns)
            self.q = ot.unif(self.nt)
        elif _distribs == 'zipf':
            self.p = zipf_init(self.src_lang, self.ns)
            self.q = zipf_init(self.trg_lang, self.nt)
        elif _distribs == 'file':
            pdb.set_trace()
        else:
            raise ValueError()

        self.coupling = None
        self.test_dict = None
        self.score_type = _score_type
        self.scores = None
        self.init_optimizer(self.tol, use_gpu=False)
        if self.solver == 'both':
            self.solver.normalized = True

    def init_optimizer(self, *args, **kwargs):
        print('Initializing Gromov-Wasserstein optimizer')
        self.solver = gromov_wass_solver(**kwargs)
        if self.test_dict is not None:
            self.solver.accuracy_function = self._test_accuracy
        else:
            self.solver.compute_accuracy = False

    def compute_scores(self, xs, xt, score_type, adjust=None, verbose=False):
        """ Exlcusive to OT methods. Given coupling matrix, compute scores that will
            be used to determine word translations. Options:
                - coupling: use directly the GW coupling
                - barycentric: using barycenter transported samples to target domain,
                     compute distaces there and use as scores
                - adjust (str): refinement method [None|csls|isf]

        """
        if score_type == 'coupling':
            scores = self.coupling
        elif score_type == 'barycentric':
            ot_emd = ot.da.EMDTransport()
            ot_emd.xs_ = xs
            ot_emd.xt_ = xt
            ot_emd.coupling_ = self.coupling
            xt_s = ot_emd.inverse_transform(Xt=xt)  # Maps target to source space
            scores = -sp.spatial.distance.cdist(xs, xt_s, metric=self.metric)  # FIXME: should this be - dist?
        elif score_type == 'distance':
            # For baselines that only use distances without OT
            scores = -sp.spatial.distance.cdist(xs, xt, metric=self.metric)
        elif score_type == 'projected':
            # Uses projection mapping, computes distance in projected space
            scores = -sp.spatial.distance.cdist(xs, xt @ self.mapping.T, metric=self.metric)
        if adjust == 'csls':
            scores = csls(scores, knn=10)
            # print('here')
        elif adjust == 'isf':
            raise NotImplementedError('Inverted Softmax not implemented yet')

        self.scores = scores
        if verbose:
            plt.figure()
            plt.imshow(scores, cmap='jet')
            plt.colorbar()
            plt.show()

    def test_accuracy(self, score_type=None, adjust=None,
                      verbose=False):
        """
            Same as above, but uses the object's attribute G.

            Seems uncessary to have both this and compute_scores. Consider merging.
        """
        if score_type is None:
            score_type = self.score_type
        if adjust is None:
            adjust = self.adjust
        if self.coupling is None:
            raise ValueError('Optimal coupling G has not been computed yet')
        self.compute_scores(score_type, adjust, verbose=verbose > 1)
        accs = self.score_translations(self.test_dict, verbose=verbose > 1)
        if verbose > 0:
            for k, v in accs.items():
                print('Accuracy @{:2}: {:8.2f}%'.format(k, 100*v))
        return accs

    def find_mutual_nn(self):
        best_match_src = self.scores.argmax(1)
        best_match_trg = self.scores.argmax(0)
        paired = []
        for i in range(self.ns):
            m = best_match_src[i]
            if best_match_trg[m] == i:
                paired.append((i, m))
        return paired

    def get_mapping(self, xs, xt,
                    type='orthogonal',
                    anchor_method='mutual_nn', max_anchors=None):
        """
            Infer mapping given optimal coupling
        """
        # Method 1: Orthogonal projection that best macthes NN
        xs = xs.detach().numpy()
        xt = xt.detach().numpy()
        self.compute_scores(xs, xt, score_type=self.score_type) # TO refresh
        if anchor_method == 'mutual_nn':
            pseudo = self.find_mutual_nn()#[:100]
        elif anchor_method == 'all':
            translations, oov = self.generate_translations()
            pseudo = [(k, v[0]) for k, v in translations.items()]
        if max_anchors:
            pseudo = pseudo[:max_anchors]
        print('Finding orthogonal mapping with {} anchor points via {}'.format(len(pseudo), anchor_method))
        if anchor_method in ['mutual_nn', 'all']:
            idx_src = [ws for ws, _ in pseudo]
            idx_trg = [wt for _, wt in pseudo]
            xs_nn = xs[idx_src]
            xt_nn = xt[idx_trg]
            p = orth_procrustes(xs_nn, xt_nn)
        elif anchor_method == 'barycenter':
            ot_emd = ot.da.EMDTransport()
            ot_emd.xs_ = xs
            ot_emd.xt_ = xt
            ot_emd.coupling_ = self.coupling
            xt_hat = ot_emd.inverse_transform(Xt=xt) # Maps target to source space
            p = orth_procrustes(xt_hat, xt)
        return p

    def fit(self, xs, xt, tol=1e-9,
            maxiter=300, print_every=None,
            plot_every=None, verbose=False, save_plots=None):
        """
            Wrapper function that computes all necessary steps to estimate
            bilingual mapping using Gromov Wasserstein.
        """
        print('Fitting bilingual mapping with Gromov Wasserstein')

        # 0. Pre-processing
        self.normalize_embeddings()

        # 1. Solve Gromov Wasserstein problem
        print('Solving optimization problem...')
        g = self.solver.solve(xs.numpy(),
                              xt.numpy(),
                              self.p, self.q,
                              maxiter=maxiter,
                              plot_every=plot_every,
                              print_every=print_every,
                              verbose=verbose,
                              save_plots=save_plots)
        self.coupling = g

        # 2. From Couplings to Translation Score
        print('Computing translation scores...')
        print(self.score_type, self.adjust)
        self.compute_scores(xs.numpy(), xt.numpy(),
                            self.score_type,
                            adjust=self.adjust,
                            verbose=True)
        self.mapping = self.get_mapping(xs, xt,
                                        type='orthogonal',
                                        anchor_method='mutual_nn')
