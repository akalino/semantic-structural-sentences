import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from time import sleep
from time import time
import matplotlib.pylab as plt


from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from annoy import AnnoyIndex

from align_models.OptimalTransport import GWTransport
from align_models.CLR import CyclicLR
from align_models.SupervisedDataLoader import SupervisedDataLoader
from align_models.EigenvectorSimilarity import EigenvectorSimilarity
from align_models.BottleneckSimilarity import BottleneckSimilarity

from eval_utilities import find_precision, find_similarities
try:
    from text_complete import text_results
except ImportError:
    pass


def evaluate_nn(_model, _xs, _ys, _tx, _ty, _ann):
    hits10 = 0
    for src_idx in range(len(_xs)):
        #print(_model.scores[src_idx, :])
        knn = np.argpartition(_model.scores[src_idx, :], -_ann)[-_ann:]
        # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
        knn_sort = knn[np.argsort(-_model.scores[src_idx, knn])]
        _x_act = [_ty[z] for z in knn_sort]
        # With - to get descending order
        target = _ty[src_idx]
        print(_x_act)
        print(target)
        if target in knn_sort:
            hits10 += 1
    print(hits10)


def evaluate(_model, _batch_s, _batch_t, _map, _n1, _n2):
    _xs = _xs.detach().numpy()
    _ys = _ys.detach().numpy()
    f = _ys.shape[1]
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
    for i in range(len(_ys)):
        v = _ys[i]
        t.add_item(i, v)
    print('Building Approximate Nearest Neighbors')
    t.build(500)
    print('Oh Yeah!')
    ind_vecs_5 = []
    ind_vecs_10 = []
    for pv_idx in tqdm(range(len(_xs))):
        pred_vec = _xs[pv_idx]@_map
        ind_vecs_5.append(t.get_nns_by_vector(pred_vec, _n1, search_k=-1, include_distances=False))
        ind_vecs_10.append(t.get_nns_by_vector(pred_vec, _n2, search_k=-1, include_distances=False))
    return ind_vecs_5, ind_vecs_10


def run(_sent_space, _ent_space, _rel_space,
        _agg, _norm,
        _train, _valid, _batch_sz,
        _init_lr, _epochs, _es_tol, _beta, _tune):
    _model = GWTransport(_sent_space, _ent_space,
                         _rel_space, _agg, _norm,
                         _beta, _tune, _batch_sz,
                         'uniform', 'csls', 1e-10, 'distance')
    data = SupervisedDataLoader(_train, _valid, _batch_sz)
    _valid_x, _valid_y = data.get_validation()
    _train_x, _train_y = data.create_all_batches()
    _train_x = _train_x[0]
    _train_y = _train_y[0]
    xs = _model.represent_triple(_train_x[:, 0], _train_x[:, 1], _train_x[:, 2])
    print(xs.detach().numpy().shape)
    ys = _model.sent_embedding(_train_y).squeeze()
    print(ys.detach().numpy().shape)
    start = time()
    _model.fit(xs.detach(),
               ys.detach(),
               maxiter=500,
               tol=1e-12,
               print_every=1,
               plot_every=1,
               verbose=True,
               save_plots='glove_gw')
    _map = _model.get_mapping(xs, ys,
                              type='orthogonal',
                              anchor_method='mutual_nn',
                              max_anchors=None)
    print(_map.shape)
    plt.close('all')
    print('Total elapsed time: {}s'.format(time() - start))
    #acc_dict = {}
    #print('Results on test dictionary for fitting vectors: (via coupling)')
    #acc_dict['coupling'] = _model.test_accuracy(verbose=True, score_type='coupling')
    #print('Results on test dictionary for fitting vectors: (via coupling + csls)')
    #acc_dict['coupling_csls'] = _model.test_accuracy(verbose=True, score_type='coupling', adjust='csls')
    return _model, xs, ys, _valid_x, _valid_y, _map


def experiment(_config_name):
    print(_config_name)
    config_label = _config_name.split('/')[-1].split('.')[0]
    with open(_config_name, 'rb') as json_file:
        config = json.load(json_file)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if config['language_model'] == 'dct':
        sent_emb_path = os.path.join(wd,
                                     'sentence-embeddings/{lm}/{d}_{lm}_{k}_space.pt'.format(
                                         lm=config['language_model'],
                                         k=config['komp'],
                                         d=config['data_source']))
    elif config['language_model'] == 'infersent':
        sent_emb_path = os.path.join(wd,
                                     'sentence-embeddings/{lm}/{d}_{mn}_space.pt'.format(
                                         lm=config['language_model'],
                                         mn=config['sentname'],
                                         d=config['data_source']))
    else:
        sent_emb_path = os.path.join(wd,
                                     'sentence-embeddings/{lm}/{d}_{lm}_space.pt'.format(lm=config['language_model'],
                                                                                         d=config['data_source']))
    ent_emb_path = os.path.join(wd, 'kg-embeddings/data/entities_{e}_{d}_{a}.pt'.format(e=config['kg_embedding'],
                                                                                        d=config['kg_dimension'],
                                                                                        a=config['data_source']))
    rel_emb_path = os.path.join(wd, 'kg-embeddings/data/relations_{e}_{d}_{a}.pt'.format(e=config['kg_embedding'],
                                                                                         d=config['kg_dimension'],
                                                                                         a=config['data_source']))
    sent_emb = torch.load(sent_emb_path)
    ent_emb = torch.load(ent_emb_path)
    rel_emb = torch.load(rel_emb_path)
    train_path = os.path.join(wd, 'data/RESIDE/{d}_data/training_data.csv'.format(d=config['data_source']))
    valid_path = os.path.join(wd, 'data/RESIDE/{d}_data/validation_data.csv'.format(d=config['data_source']))
    model, xs, ys, tx, ty, gw_map = run(sent_emb, ent_emb, rel_emb,
                                        config['kg_combination'],
                                        config['normalization'],
                                        train_path, valid_path,
                                        config['batch_size'],
                                        config['learning_rate'],
                                        config['epochs'],
                                        config['patience'],
                                        config['beta'],
                                        True)
    #ind_5, ind_10 = evaluate(xs, ys, gw_map, 5, 10)
    #print(ind_10)
    #try:
    #    text_results('Experiments for config {l} done: H5 {r} H10 {s}, '
    #                 'AS {c}'.format(l=_config_name, r=h5, s=h10, c=avg_sim))
    #except NameError:
    #    print('Experiments for config {l} done: H5 {r} H10 {s}, '
    #          'AS {c}'.format(l=_config_name, r=h5, s=h10, c=avg_sim))
    #return _config_name, h5, h10, avg_sim


if __name__ == "__main__":
    config_name = str(sys.argv[1])
    experiment(config_name)
    #cn, hits5, hits10, avgsim =
