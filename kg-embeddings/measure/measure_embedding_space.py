import argparse
import os
import pandas as pd
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

from PrinComp import PCA


def load_spaces(_dim, _mt, _ds):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    out_path = os.path.join(wd, "kg-embeddings", "data")
    _entities = torch.load(os.path.join(out_path,
                                        'entities_{m}_{d}_{s}.pt'.format(m=str(_mt), d=_dim, s=_ds)))
    _relations = torch.load(os.path.join(out_path,
                                         'relations_{m}_{d}_{s}.pt'.format(m=str(_mt), d=_dim, s=_ds)))

    return _entities, _relations


def compute_metrics(_dim, _entities, _relations):
    """

    :param _dim:
    :param _entities:
    :param _relations:
    :return:
    """
    norms = []
    means = np.zeros(_dim)
    for j in range(_entities.shape[0]):
        vec = torch.norm(_entities[j]).detach().numpy()
        norms.append(vec)
        means += _entities[j].detach().numpy()
    _ent_norm = np.mean(norms)
    _ent_mu = means / _entities.shape[0]
    _ent_mean = np.linalg.norm(_ent_mu)
    norms = []
    means = np.zeros(_dim)
    for j in range(_relations.shape[0]):
        vec = torch.norm(_relations[j]).detach().numpy()
        norms.append(vec)
        means += _entities[j].detach().numpy()
    _rel_norm = np.mean(vec)
    _rel_mu = means / _relations.shape[0]
    _rel_mean = np.linalg.norm(_rel_mu)
    return _ent_norm, _ent_mean, _rel_norm, _rel_mean


def compute_isotropy(_vectors):
    _vectors = _vectors.detach().numpy()
    # getting isotropy
    w, v = np.linalg.eig(np.matmul(_vectors.T, _vectors))
    _isotropy = np.sum(np.exp(np.matmul(_vectors, v)), axis=0)
    _max_iso = _isotropy.max()
    _min_iso = _isotropy.min()
    _iso_ratio = _isotropy.min() / _isotropy.max()
    return _max_iso, _min_iso, _iso_ratio


def compute_variances(_dim, _entities, _relations):
    ent_pca = PCA(_entities)
    ent_pca.decomposition(_entities, _dim)
    ent_exp_var = ent_pca.explained_variance()
    rel_pca = PCA(_relations)
    rel_pca.decomposition(_relations, _dim)
    rel_exp_var = rel_pca.explained_variance()
    return ent_exp_var, rel_exp_var


def plot_variances(_vars, _type, _model, _dim):
    plt.clf()
    plt.bar(range(len(_vars)), _vars, alpha=0.5,
            align='center', label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('{t} embedding from {m} model, d={d}'.format(t=_type,
                                                       m=_model,
                                                       d=_dim))
    plt.legend(loc='best')
    plt.savefig('{}_{}_{}_var.png'.format(_type, _model, _dim))
    plt.close()


def compute_pca_frequency_degree(_dim, _entities, _relations, _dp, _mt):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    ent_freq = pd.read_csv(os.path.join(_dp, 'entities_frequencies.csv'))
    rel_freq = pd.read_csv(os.path.join(_dp, 'relation_frequencies.csv'))
    ent_pca = PCA(_entities)
    ent_pca.decomposition(_entities, 2)
    plt.clf()
    ent_freq_bins = np.digitize(ent_freq['freq'].tolist(),
                                bins=[1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 10000])
    plt.scatter(ent_pca.transformed.T[0].detach().numpy(),
                ent_pca.transformed.T[1].detach().numpy(),
                cmap='RdPu',
                #c=ent_freq['freq'].tolist())
                c=ent_freq_bins)
    plt.colorbar()
    plt.xlabel(r'$ \alpha_1 (w)$')
    plt.ylabel(r'$ \alpha_2 (w)$')
    plt.title('Entity embedding PCA')
    plt.savefig('{}_{}_entity_pca.png'.format(_mt, _dim))
    plt.close()
    plt.clf()
    rel_pca = PCA(_relations)
    rel_pca.decomposition(_relations, 2)
    rel_freq_bins = np.digitize(rel_freq['freq'].tolist(),
                                bins=[1, 2, 3, 4, 5, 10, 100,
                                      1000, 10000, 100000, 500000])
    plt.scatter(rel_pca.transformed.T[0].detach().numpy(),
                rel_pca.transformed.T[1].detach().numpy(),
                cmap='RdPu',
                #c=rel_freq['freq'].tolist())
                c=rel_freq_bins)
    plt.colorbar()
    plt.xlabel(r'$ \alpha_1 (w)$')
    plt.ylabel(r'$ \alpha_2 (w)$')
    plt.title('Relation embedding PCA')
    plt.savefig('{}_{}_relation_pca.png'.format(_mt, _dim))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_node_space",
                                     description="Builds vector spaces of kg embeddings")
    parser.add_argument('-d', '--dimension', required=False, type=int, default=200,
                        help='Embedding dimension for h, r and t',
                        dest='dim')
    parser.add_argument('-m', '--model', required=True, type=str, default='transe',
                        help='Knowledge graph embedding model type',
                        dest='mt')
    parser.add_argument('-s', '--set', required=True, type=str, default='reside',
                        help='Dataset type type',
                        dest='ds')
    args = parser.parse_args()
    entities, relations = load_spaces(args.dim, args.mt, args.ds)
    ent_norm, ent_mean, rel_norm, rel_mean = compute_metrics(args.dim, entities, relations)
    print('Entities norm {} and mean {}'.format(ent_norm, ent_mean))
    print('Relations norm {} and mean {}'.format(rel_norm, rel_mean))
    ent_var, rel_var = compute_variances(args.dim, entities, relations)
    plot_variances(ent_var, 'Entity', args.mt, args.dim)
    plot_variances(rel_var, 'Relation', args.mt, args.dim)
    compute_pca_frequency_degree(args.dim, entities, relations,
                                 "benchmarks\FB15K237")