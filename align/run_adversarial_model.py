import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from time import sleep

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from align_models.BoilerplateGAN import BoilerplateGAN
from align_models.CLR import CyclicLR
from align_models.SupervisedDataLoader import SupervisedDataLoader
from align_models.EigenvectorSimilarity import EigenvectorSimilarity
from align_models.BottleneckSimilarity import BottleneckSimilarity

from eval_utilities import find_precision, find_similarities
try:
    from text_complete import text_results
except ImportError:
    pass


def run(_sent_space, _ent_space, _rel_space,
        _agg, _norm,
        _train, _valid, _batch_sz,
        _init_lr, _epochs, _es_tol, _beta, _tune):
    _model = BoilerplateGAN(_sent_space, _ent_space, _rel_space,
                            _agg, _norm, _beta, _tune)
    lm_criterion = nn.MSELoss()
    adv_criterion = nn.BCELoss()
    lm_optimizer = torch.optim.Adam(_model.mapping.parameters(), lr=_init_lr)
    disc_sent_optim = torch.optim.Adam(_model.discrim_sent.parameters(), lr=_init_lr)
    disc_kg_optim = torch.optim.Adam(_model.discrim_kg.parameters(), lr=_init_lr)
    gen_sent_optim = torch.optim.Adam(_model.generator_sent.parameters(), lr=_init_lr)
    gen_kg_optim = torch.optim.Adam(_model.generator_kg.parameters(), lr=_init_lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    if torch.cuda.is_available():
        _model.cuda()
    val_prior = 10
    es_tol = 0
    data = SupervisedDataLoader(_train, _valid, _batch_sz)
    _valid_x, _valid_y = data.get_validation()
    for e in tqdm(range(_epochs)):
        losses = []
        batch_x, batch_y = data.create_all_batches()
        for idx in range(len(batch_x)):
            _model.generator_kg.zero_grad()
            _model.generator_sent.zero_grad()
            _model.disrcim_kg.zero_grad()
            _model.discrim_sent.zero_grad()
            _model.mapping.zero_grad()
            if torch.cuda.is_available():
                inputs = Variable(batch_x[idx].cuda())
                labels = Variable(_model.sent_embedding(batch_y[idx].cuda()).squeeze())
            # kg gan first
            # generate triple representation
            b_triples = _model.represent_triple(inputs[:, 0], inputs[:, 1], inputs[:, 2])
            # generate fake triples
            b_trip_fake = _model.generator_kg(b_triples)
            # stack this batch
            batch_triple_fakes = ()
            # sent gan second
            b_sent = labels
            b_sent_fake = _model.generator_sent(b_sent)
            _model.zero_grad()
            if torch.cuda.is_available():
                inputs = Variable(batch_x[idx].cuda())
                labels = Variable(_model.sent_embedding(batch_y[idx].cuda()).squeeze())
            outputs = _model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())
        _model.eval()
        val_losses = []
        valid_batch_x, valid_batch_y = data.get_validation_batches()
        for v_idx in range(len(valid_batch_x)):
            if torch.cuda.is_available():
                inputs = Variable(valid_batch_x[v_idx].cuda())
                labels = Variable(_model.sent_embedding(valid_batch_y[v_idx].cuda()).squeeze())
            outputs = _model(inputs)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss)
            torch.cuda.empty_cache()
        val_loss = sum(val_losses)/_batch_sz
        scheduler.step(val_loss)
        if val_loss >= val_prior:
            es_tol += 1
        else:
            val_prior = val_loss
        if es_tol >= _es_tol:
            print('Reached early stopping criteria')
            break
        if e % 1 == 0 and e != 0:
            print('Epoch number: {n}'.format(n=e))
            print('Mean epoch training loss: {l}'.format(l=np.mean(losses)))
            print('Validation loss: {l}'.format(l=val_loss))
            print('Patience threshold: {p}'.format(p=es_tol))
    del data
    torch.cuda.empty_cache()
    return _model, _valid_x, _valid_y