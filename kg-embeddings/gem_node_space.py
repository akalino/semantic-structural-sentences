import argparse
import os
import pandas as pd
import sys
sys.path.append('../')
import torch

from OpenKE.openke.module.model import TransE, TransR, TransH, SimplE
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

from GraphGEM import GeometricGraphEmbedder
from align.align_models.GEMDataLoader import GEMDataLoader


def load_embeddings(_fb_path, _model_path, _dim, _mt):
    train_dataloader = TrainDataLoader(in_path=_fb_path,
                                       nbatches=100,
                                       threads=1,
                                       sampling_mode="normal",
                                       bern_flag=1,
                                       filter_flag=1,
                                       neg_ent=25,
                                       neg_rel=0)
    if _mt == 'transe':
        transe = TransE(ent_tot=train_dataloader.get_ent_tot(),
                        rel_tot=train_dataloader.get_rel_tot(),
                        dim=_dim,
                        p_norm=1,
                        norm_flag=True)
        transe.load_checkpoint(_model_path)
        _ent_embds = transe.ent_embeddings.weight
        _rel_embds = transe.rel_embeddings.weight
    elif _mt == 'transr':
        transr = TransR(ent_tot=train_dataloader.get_ent_tot(),
                        rel_tot=train_dataloader.get_rel_tot(),
                        dim_e=_dim,
                        dim_r=_dim,
                        p_norm=1,
                        norm_flag=True)
        transr.load_checkpoint(_model_path)
        _ent_embds = transr.ent_embeddings.weight
        _rel_embds = transr.rel_embeddings.weight
    elif _mt == 'transh':
        transh = TransH(ent_tot=train_dataloader.get_ent_tot(),
                        rel_tot=train_dataloader.get_rel_tot(),
                        dim=_dim,
                        p_norm=1,
                        norm_flag=True)
        transh.load_checkpoint(_model_path)
        _ent_embds = transh.ent_embeddings.weight
        _rel_embds = transh.rel_embeddings.weight
    elif _mt == 'simple':
        simple = SimplE(ent_tot=train_dataloader.get_ent_tot(),
                        rel_tot=train_dataloader.get_rel_tot(),
                        dim=_dim)
        simple.load_checkpoint(_model_path)
        _ent_embds = simple.ent_embeddings.weight
        _rel_embds = simple.rel_embeddings.weight
    return _ent_embds, _rel_embds


def run_and_save(_dim, _mt, _ds, _hyper):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if _ds == 'riedel':
        fb_path = os.path.join(wd, "data", "RESIDE_KG/")
    elif _ds == 'gids':
        fb_path = os.path.join(wd, "data", "GIDS_KG/")
    else:
        print('Invalid data set, please run the preprocessing steps.')
        sys.exit()
    if _mt == 'transr':
        model_path = os.path.join(wd, "kg-embeddings", "checkpoints", "{m}_{d}_{s}.ckpt".format(m=_mt,
                                                                                                    d=_dim,
                                                                                                    s=_ds))
    else:
        model_path = os.path.join(wd, "kg-embeddings", "checkpoints", "{m}_{d}_{s}.ckpt".format(m=_mt,
                                                                                                d=_dim,
                                                                                                s=_ds))
    out_path = os.path.join(wd, "kg-embeddings", "data")
    entity_emb, rel_emb = load_embeddings(fb_path, model_path, _dim, _mt)
    train_path = os.path.join(wd, 'data/RESIDE/{d}_data/gem_training_set.csv'.format(d=_ds))
    valid_path = os.path.join(wd, 'data/RESIDE/{d}_data/gem_validation_set.csv'.format(d=_ds))
    test_path = os.path.join(wd, 'data/RESIDE/{d}_data/gem_testing_set.csv'.format(d=_ds))
    data = GEMDataLoader(train_path, valid_path, test_path, -1)
    all_triples = data.all_triples
    batch_x, batch_y = data.create_all_batches()
    valid_batch_x, valid_batch_y = data.get_validation_batches()
    test_batch_x, test_batch_y = data.get_test_batches()
    hyper = [str(x) for x in _hyper]
    hyper = '_'.join(hyper)
    gge = GeometricGraphEmbedder(all_triples, entity_emb, rel_emb,
                                 _hyper[0], _hyper[1], _hyper[2],
                                 False)
    train_vecs = gge.lookup_batched_triple(data.train_x)
    print(train_vecs.shape)
    torch.save(train_vecs,
               os.path.join(out_path, 'train_gem_triples_{m}_{d}_{s}_{h}.pt'.format(m=str(_mt), d=_dim, s=_ds, h=hyper)))
    valid_vecs = gge.lookup_batched_triple(data.valid_x)
    print(valid_vecs.shape)
    torch.save(valid_vecs,
               os.path.join(out_path, 'valid_gem_triples_{m}_{d}_{s}_{h}.pt'.format(m=str(_mt), d=_dim, s=_ds, h=hyper)))
    test_vecs = gge.lookup_batched_triple(data.test_x)
    print(test_vecs.shape)
    torch.save(test_vecs,
               os.path.join(out_path, 'test_gem_triples_{m}_{d}_{s}_{h}.pt'.format(m=str(_mt), d=_dim, s=_ds, h=hyper)))


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
    parser.add_argument('-p', '--param', required=True, type=str, default='default',
                        help='GEM hyperparameters',
                        dest='hyp')
    args = parser.parse_args()
    param_dict = {'default': [45, 17, 3],
                  'smaller': [30, 5, 3]}
    hyper = param_dict[args.hyp]
    run_and_save(args.dim, args.mt, args.ds, hyper)
