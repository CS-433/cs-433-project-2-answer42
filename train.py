import os
import random
import time

import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models
from torch import nn
from logger import Logger
import myexman
import numpy as np
import utils
import os, sys
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import itertools
import matplotlib.pyplot as plt


warnings.simplefilter("ignore")


def add_learner_params(parser):
    parser.add_argument('--name', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--eval_only', type=bool, default=False)
    # trainer params
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--val_iters', type=int, default=1)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_seed', default=0, type=int)


def main():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    add_learner_params(parser)
    args, _ = parser.parse_known_args()

    if args.model not in models.REGISTERED_MODELS:
        raise NotImplementedError(f'There is no registered model named {args.model}!')

    # add model specific parameters
    models.REGISTERED_MODELS[args.model].add_model_hparams(parser)

    # parse all args
    args = parser.parse_args(namespace=args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create logger
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create model
    model = models.REGISTERED_MODELS[args.model](args)

    # Data loading code
    # set seed for data preprocessing
    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)
    train_loader, val_loader = model.get_dataloaders()

    # set seed for the resto of the training
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define optimizer
    epoch = 0
    model.compile(device=device)

    # optionally resume from a checkpoint
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        # TODO: move this logic to model class
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            model.optimizer.load_state_dict(ckpt['optimizer'])

    cudnn.benchmark = True

    data_time, it_time = 0, 0
    for epoch in range(1, args.epochs + 1):
        train_logs = []
        model.train()

        start_time = dtime_start = time.time()
        for _, batch in enumerate(train_loader):
            batch = utils.put_on_device(batch, device)
            data_time += time.time() - dtime_start

            # forward pass, compute loss, optimizer step
            logs = {}
            if not args.eval_only:
                logs = model.train_step(batch)

            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})

            dtime_start = time.time()

        if epoch % args.save_freq == 0:
            model.save(f'{args.root}/ckpt.torch')

        if epoch % args.eval_freq == 0 or epoch >= args.epochs:
            test_logs = []
            model.eval()
            for batch in val_loader:
                batch = utils.put_on_device(batch, device)
                with torch.no_grad():
                    # forward pass
                    logs = model.test_step(batch)
                # save logs for the batch
                test_logs.append(logs)
            model.train()

            test_logs = utils.agg_all_metrics(test_logs)
            logger.add_logs(epoch, test_logs, pref='test_')

        it_time += time.time() - start_time

        train_logs = utils.agg_all_metrics(train_logs)

        logger.add_logs(epoch, train_logs, pref='train_')
        logger.add_scalar(epoch, 'lr', model.optimizer.param_groups[0]['lr'])
        logger.add_scalar(epoch, 'data_time', data_time)
        logger.add_scalar(epoch, 'epoch_time', it_time)
        logger.iter_info()
        logger.save()

        data_time, it_time = 0, 0
        train_logs = []
        start_time = time.time()

    model.save(f'{args.root}/ckpt.torch')

if __name__ == '__main__':
    main()
