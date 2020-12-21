import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy as sp
import torch
import tqdm
import umap

from collections import Counter
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tests.clusterability as clus
import tests.clustering_validation as clus_val
import tests.dissimilarity_vis as dsv
from tests.hopkins import hopkins


def load_process_embedding_matrix(_path):
    """
    Loads and applies PCA to embedding matrix.
    :param _path: Path to embeddings as .pt file.
    :return: 2D PCA projected embedding matrix.
    """
    embs = torch.load(_path).numpy()
    print('Embeddings loaded')
    print('Running PCA')
    embs_pca = PCA(n_components=2).fit_transform(embs)  #, svd_solver='arpack')
    print('Complete PCA')
    print('Running UMAP')
    #embs_tsne = TSNE(n_components=2, verbose=1, n_jobs=-1,
    #                 n_iter=250, n_iter_without_progress=10).fit_transform(embs)
    um_reducer = umap.UMAP(verbose=True, low_memory=True)
    embs_tsne = um_reducer.fit_transform(embs)
    print('Complete UMAP')
    #embs_tsne = 0
    print('Embeddings projected')
    return embs, embs_pca, embs_tsne


def apply_spatial_historgram(_embs, _bins, _path):
    """
    Applied the spatial histogram clusterability metric.
    :param _embs: A set of PCA projected embeddings.
    :return: Spatial histogram information.
    """
    if 'random' in _path:
        _embs = np.random.rand(len(_embs), 2)
    kls_embs = clus.spaHist(_embs, bins=_bins, n=50)
    mu_kls = kls_embs.mean()
    std_kls = kls_embs.std()
    print('Spatial histogram:  Mu: {m}, sigma: {s}'.format(m=mu_kls, s=std_kls))
    return kls_embs, mu_kls, std_kls


def plot_spatial_histograms(_sent_emb_list, _kls_list, _bins):
    """
    Plots Spatial Histograms for a set of sentence embeddings.
    :param _sent_emb_list: List of sentence embedding names.
    :param _kls_list: List of spatial histogram data, same order as sentence embedding names.
    :return: None, plots.
    """
    plt.figure(figsize=(12, 8))
    for j in range(len(_sent_emb_list)):
        sns.distplot(_kls_list[j], hist=False, label=str(_sent_emb_list[j]))
    plt.legend()
    plt.title('Sentence Embedding Space Spatial Histograms')
    plt.savefig('sentence_embedding_clusterability_{b}.png'.format(b=_bins))


def calculate_hopkins(_embs, _hbins):
    embs_hopkins = []
    for i in tqdm.tqdm(range(50)):
        embs_hopkins.append(hopkins(_embs, _hbins))
    return embs_hopkins, np.mean(embs_hopkins), np.std(embs_hopkins)


def plot_hopkins(_sent_emb_list, _kls_list, _bins):
    """
    Plots Spatial Histograms for a set of sentence embeddings.
    :param _sent_emb_list: List of sentence embedding names.
    :param _kls_list: List of spatial histogram data, same order as sentence embedding names.
    :return: None, plots.
    """
    plt.figure(figsize=(12, 8))
    for j in range(len(_sent_emb_list)):
        sns.distplot(_kls_list[j], hist=False, label=str(_sent_emb_list[j]))
    plt.legend()
    plt.title('Sentence Embedding Space Hopkins Statistic')
    plt.savefig('sentence_embedding_hopkins_{b}.png'.format(b=_bins))


def plot_embedding_tsne(_paths, _names):
    fig, axs = plt.subplots(3, 3)
    embs, proj0, tsne0 = load_process_embedding_matrix(_paths[0])
    axs[0, 0].scatter(x=tsne0[:, 0], y=tsne0[:, 1])
    axs[0, 0].set_title('{a}'.format(a=str(_names[0])))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    embs, proj1, tsne1 = load_process_embedding_matrix(_paths[1])
    axs[0, 1].scatter(x=tsne1[:, 0], y=tsne1[:, 1])
    axs[0, 1].set_title('{a}'.format(a=str(_names[1])))
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    embs, proj2, tsne2 = load_process_embedding_matrix(_paths[2])
    axs[0, 2].scatter(x=tsne2[:, 0], y=tsne2[:, 1])
    axs[0, 2].set_title('{a}'.format(a=str(_names[2])))
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    embs, proj3, tsne3 = load_process_embedding_matrix(_paths[3])
    axs[1, 0].scatter(x=tsne3[:, 0], y=tsne3[:, 1])
    axs[1, 0].set_title('{a}'.format(a=str(_names[3])))
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    embs, proj4, tsne4 = load_process_embedding_matrix(_paths[4])
    axs[1, 1].scatter(x=tsne4[:, 0], y=tsne4[:, 1])
    axs[1, 1].set_title('{a}'.format(a=str(_names[4])))
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    embs, proj5, tsne5 = load_process_embedding_matrix(_paths[5])
    axs[1, 2].scatter(x=tsne5[:, 0], y=tsne5[:, 1])
    axs[1, 2].set_title('{a}'.format(a=str(_names[5])))
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    embs, proj6, tsne6 = load_process_embedding_matrix(_paths[6])
    axs[2, 0].scatter(x=tsne6[:, 0], y=tsne6[:, 1])
    axs[2, 0].set_title('{a}'.format(a=str(_names[6])))
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])

    embs, proj7, tsne7 = load_process_embedding_matrix(_paths[7])
    axs[2, 1].scatter(x=tsne7[:, 0], y=tsne7[:, 1])
    axs[2, 1].set_title('{a}'.format(a=str(_names[7])))
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    embs, proj8, tsne8 = load_process_embedding_matrix(_paths[8])
    axs[2, 2].scatter(x=tsne8[:, 0], y=tsne8[:, 1])
    axs[2, 2].set_title('{a}'.format(a=str(_names[8])))
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])

    fig.suptitle('TSNE Projections of Sentence Embedding Spaces')
    # plt.show()
    fig.savefig('sentence_tsne_comp.png')


def plot_embedding_umap_labeled(_paths, _names, _labels):
    group = _labels['rel_idx'].tolist()
    group = list(filter((35).__ne__, group))
    group = list(filter((1000).__ne__, group))
    print(Counter(group))
    fig, axs = plt.subplots(3, 3)
    embs, proj0, tsne0 = load_process_embedding_matrix(_paths[0])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 0].scatter(x=tsne0[i, 0], y=tsne0[i, 1], label=g)
    axs[0, 0].set_title('{a}'.format(a=str(_names[0])))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    embs, proj1, tsne1 = load_process_embedding_matrix(_paths[1])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 1].scatter(x=tsne1[i, 0], y=tsne1[i, 1], label=g)
    axs[0, 1].set_title('{a}'.format(a=str(_names[1])))
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    embs, proj2, tsne2 = load_process_embedding_matrix(_paths[2])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 2].scatter(x=tsne2[i, 0], y=tsne2[i, 1], label=g)
    axs[0, 2].set_title('{a}'.format(a=str(_names[2])))
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    embs, proj3, tsne3 = load_process_embedding_matrix(_paths[3])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 0].scatter(x=tsne3[i, 0], y=tsne3[i, 1], label=g)
    axs[1, 0].set_title('{a}'.format(a=str(_names[3])))
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    embs, proj4, tsne4 = load_process_embedding_matrix(_paths[4])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 1].scatter(x=tsne4[i, 0], y=tsne4[i, 1], label=g)
    axs[1, 1].set_title('{a}'.format(a=str(_names[4])))
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    embs, proj5, tsne5 = load_process_embedding_matrix(_paths[5])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 2].scatter(x=tsne5[i, 0], y=tsne5[i, 1], label=g)
    axs[1, 2].set_title('{a}'.format(a=str(_names[5])))
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    embs, proj6, tsne6 = load_process_embedding_matrix(_paths[6])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 0].scatter(x=tsne6[i, 0], y=tsne6[i, 1], label=g)
    axs[2, 0].set_title('{a}'.format(a=str(_names[6])))
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])

    embs, proj7, tsne7 = load_process_embedding_matrix(_paths[7])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 1].scatter(x=tsne7[i, 0], y=tsne7[i, 1], label=g)
    axs[2, 1].set_title('{a}'.format(a=str(_names[7])))
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    embs, proj8, tsne8 = load_process_embedding_matrix(_paths[8])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 2].scatter(x=tsne8[i, 0], y=tsne8[i, 1], label=g)
    axs[2, 2].set_title('{a}'.format(a=str(_names[8])))
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])

    fig.suptitle('Labeled UMAP Projections of Sentence Embedding Spaces')
    # plt.show()
    fig.savefig('sentence_umap_comp.png')


def plot_embedding_pca_labeled(_paths, _names, _labels):
    group = _labels['rel_idx'].tolist()
    group = list(filter((35).__ne__, group))
    group = list(filter((1000).__ne__, group))
    print(Counter(group))
    fig, axs = plt.subplots(3, 3)
    embs, proj0, tsne0 = load_process_embedding_matrix(_paths[0])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 0].scatter(x=proj0[i, 0], y=proj0[i, 1], label=g)
    axs[0, 0].set_title('{a}'.format(a=str(_names[0])))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    embs, proj1, tsne1 = load_process_embedding_matrix(_paths[1])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 1].scatter(x=proj1[i, 0], y=proj1[i, 1], label=g)
    axs[0, 1].set_title('{a}'.format(a=str(_names[1])))
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    embs, proj2, tsne2 = load_process_embedding_matrix(_paths[2])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[0, 2].scatter(x=proj2[i, 0], y=proj2[i, 1], label=g)
    axs[0, 2].set_title('{a}'.format(a=str(_names[2])))
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    embs, proj3, tsne3 = load_process_embedding_matrix(_paths[3])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 0].scatter(x=proj3[i, 0], y=proj3[i, 1], label=g)
    axs[1, 0].set_title('{a}'.format(a=str(_names[3])))
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    embs, proj4, tsne4 = load_process_embedding_matrix(_paths[4])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 1].scatter(x=proj4[i, 0], y=proj4[i, 1], label=g)
    axs[1, 1].set_title('{a}'.format(a=str(_names[4])))
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    embs, proj5, tsne5 = load_process_embedding_matrix(_paths[5])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[1, 2].scatter(x=proj5[i, 0], y=proj5[i, 1], label=g)
    axs[1, 2].set_title('{a}'.format(a=str(_names[5])))
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    embs, proj6, tsne6 = load_process_embedding_matrix(_paths[6])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 0].scatter(x=proj6[i, 0], y=proj6[i, 1], label=g)
    axs[2, 0].set_title('{a}'.format(a=str(_names[6])))
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])

    embs, proj7, tsne7 = load_process_embedding_matrix(_paths[7])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 1].scatter(x=proj7[i, 0], y=proj7[i, 1], label=g)
    axs[2, 1].set_title('{a}'.format(a=str(_names[7])))
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    embs, proj8, tsne8 = load_process_embedding_matrix(_paths[8])
    for g in np.unique(group):
        i = np.where(group == g)
        axs[2, 2].scatter(x=proj8[i, 0], y=proj8[i, 1], label=g)
    axs[2, 2].set_title('{a}'.format(a=str(_names[8])))
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])

    fig.suptitle('Labeled PCA Projections of Sentence Embedding Spaces')
    # plt.show()
    fig.savefig('sentence_pca_comp.png')


def plot_embedding_spaces(_paths, _names):
    fig, axs = plt.subplots(3, 3)
    embs, proj0, t0 = load_process_embedding_matrix(_paths[0])
    axs[0, 0].scatter(x=proj0[:, 0], y=proj0[:, 1])
    axs[0, 0].set_title('{a}'.format(a=str(_names[0])))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    embs, proj1, t1 = load_process_embedding_matrix(_paths[1])
    axs[0, 1].scatter(x=proj1[:, 0], y=proj1[:, 1])
    axs[0, 1].set_title('{a}'.format(a=str(_names[1])))
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    embs, proj2, t2 = load_process_embedding_matrix(_paths[2])
    axs[0, 2].scatter(x=proj2[:, 0], y=proj2[:, 1])
    axs[0, 2].set_title('{a}'.format(a=str(_names[2])))
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    embs, proj3, t3 = load_process_embedding_matrix(_paths[3])
    axs[1, 0].scatter(x=proj3[:, 0], y=proj3[:, 1])
    axs[1, 0].set_title('{a}'.format(a=str(_names[3])))
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    embs, proj4, t4 = load_process_embedding_matrix(_paths[4])
    axs[1, 1].scatter(x=proj4[:, 0], y=proj4[:, 1])
    axs[1, 1].set_title('{a}'.format(a=str(_names[4])))
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    embs, proj5, t5 = load_process_embedding_matrix(_paths[5])
    axs[1, 2].scatter(x=proj5[:, 0], y=proj5[:, 1])
    axs[1, 2].set_title('{a}'.format(a=str(_names[5])))
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    embs, proj6, t6 = load_process_embedding_matrix(_paths[6])
    axs[2, 0].scatter(x=proj6[:, 0], y=proj6[:, 1])
    axs[2, 0].set_title('{a}'.format(a=str(_names[6])))
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])

    embs, proj7, t7 = load_process_embedding_matrix(_paths[7])
    axs[2, 1].scatter(x=proj7[:, 0], y=proj7[:, 1])
    axs[2, 1].set_title('{a}'.format(a=str(_names[7])))
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    embs, proj8, t8 = load_process_embedding_matrix(_paths[8])
    axs[2, 2].scatter(x=proj8[:, 0], y=proj8[:, 1])
    axs[2, 2].set_title('{a}'.format(a=str(_names[8])))
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])

    fig.suptitle('PCA Projections of Sentence Embedding Spaces')
    # plt.show()
    fig.savefig('sentence_proj_comp.png')


if __name__ == "__main__":
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "sentence_mapper.json")
    with open(sentence_path, 'rb') as f:
        sentences = json.load(f)
    sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "validation_data.csv")
    train_df = pd.read_csv(sentence_path)
    try:
        emb_labels = pd.read_csv('embedding_labels_valid.csv')
    except FileNotFoundError:
        rel_labels = []
        i = 0
        for sent_blob in tqdm.tqdm(sentences):
            i += 1
            try:
                rel_labels.append(train_df[train_df['sentence'] == sent_blob]['rel_idx'].values[0])
            except IndexError:
                rel_labels.append(1000)
        print(len(rel_labels))
        emb_labels = pd.DataFrame({'sentence_emb': list(sentences), 'rel_idx': rel_labels})
        emb_labels.to_csv('embedding_labels_valid.csv', index=False)
    paths = ['sentence-embeddings/random/riedel_random_space.pt',
             'sentence-embeddings/gem/riedel_gem_space.pt',
             'sentence-embeddings/glove/riedel_glove_space.pt',
             'sentence-embeddings/infersent/riedel_infersent1glove_space.pt',
             'sentence-embeddings/infersent/riedel_infersent2ft_space.pt',
             'sentence-embeddings/laser/riedel_laser_space.pt',
             'sentence-embeddings/quickthought/riedel_quickthought_space.pt',
             'sentence-embeddings/sentbert/riedel_sentbert_space.pt',
             'sentence-embeddings/skipthought/riedel_skipthought_space.pt',
             'sentence-embeddings/dct/riedel_dct_1_space.pt',
             'sentence-embeddings/dct/riedel_dct_3_space.pt',
             'sentence-embeddings/dct/riedel_dct_5_space.pt']
    paths = [os.path.join(wd, p) for p in paths]
    names = ['random', 'gem', 'glove',
             'inferv1', 'inferv2',
             'laser', 'quickthought',
             'sentbert', 'skipthought',
             'dct1', 'dct3', 'dct5']
    plot_embedding_pca_labeled(paths, names, emb_labels)
    plot_embedding_umap_labeled(paths, names, emb_labels)
    sp = []
    mus = []
    sigs = []
    bins = 20
    for pt in tqdm.tqdm(paths):
        print(pt)
        emb, des, tsn = load_process_embedding_matrix(pt)
        hbins = 400
        h, m, s = apply_spatial_historgram(des, bins, pt)
        sp.append(h)
        mus.append(m)
        sigs.append(s)
    plot_spatial_histograms(names, sp, bins)
    outs = pd.DataFrame({'model': names, 'mean': mus, 'std': sigs})
    print(outs)
    with open('spatial-tsne.tex', 'w') as tf:
        tf.write(outs.to_latex(index=False))
