import argparse
import os
import sys

from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import SimplE
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.loss import SoftplusLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

from text_complete import text_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_hole",
                                     description="Trains the HolE algorithm for kg embeddings")
    parser.add_argument('-d', '--dimension', required=True, type=int, default=200,
                        help='Embedding dimension for h, r and t',
                        dest='dim')
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if args.data == 'riedel':
        fb_path = os.path.join(wd, "data", "RESIDE_KG/")
        out_path = os.path.join(wd, "kg-embeddings", "checkpoints", "simple_{d}_riedel.ckpt".format(d=args.dim))
    elif args.data == 'gids':
        fb_path = os.path.join(wd, "data", "GIDS_KG/")
        out_path = os.path.join(wd, "kg-embeddings", "checkpoints", "simple_{d}_gids.ckpt".format(d=args.dim))
    else:
        print('Unknown dataset, please run pre-processing steps.')
        sys.exit()

    train_dataloader = TrainDataLoader(in_path=fb_path,
                                       nbatches=100,
                                       threads=8,
                                       sampling_mode="normal",
                                       bern_flag=1,
                                       filter_flag=1,
                                       neg_ent=25,
                                       neg_rel=0)
    test_dataloader = TestDataLoader(fb_path, "link")

    simple = SimplE(ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim=args.dim)

    model = NegativeSampling(model=simple, loss=SoftplusLoss(),
                             batch_size=train_dataloader.get_batch_size(),
                             regul_rate=1.0)

    trainer = Trainer(model=model, data_loader=train_dataloader,
                      train_times=5000, alpha=0.5, use_gpu=True,
                      opt_method="adagrad")
    trainer.run()
    simple.save_checkpoint(out_path)
    simple.load_checkpoint(out_path)
    tester = Tester(model=simple, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=True)
    text_results("Finished training SimplE with {n} dimensional vectors on {d} data.".format(n=args.dim, d=args.data))
