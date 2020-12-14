import argparse
import json
import math
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

from logger import Logger
import myexman
import models
import utils
import pruner


def init_config():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    parser.add_argument('--name', type=str, default='')

    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--lr_schedule', type=str, default='original')

    parser.add_argument('--network', type=str, default='vgg')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exception', type=int, nargs='*', default=[])
    
    parser.add_argument('--ratio', type=float, default=0.9)
    parser.add_argument('--prune_last', type=bool, default=True)

    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--pruning', type=str, default='grasp')

    # Grasp params
    parser.add_argument('--samples_per_class', type=int, default=10)
    args = parser.parse_args()
    return args

def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    print('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        print('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        count += 1


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, logger):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    logger.add_scalar(epoch, 'train_loss', train_loss / (batch_idx + 1))
    logger.add_scalar(epoch, 'train_acc', 100. * correct / total)


def test(net, loader, criterion, epoch, logger):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    logger.add_scalar(epoch, 'test_loss', test_loss / (batch_idx + 1))
    logger.add_scalar(epoch, 'test_acc', acc)

    return acc


def main(args):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200
    }
    logger = Logger('logs', base=args.root)

    # build model
    if args.network == 'vgg':
        net = models.base.VGG(depth=19, dataset=args.dataset, batchnorm=True)
        depth = 19
    elif args.network == 'resnet':
        net = models.base.resnet(depth=32, dataset=args.dataset)
        depth = 33
    else:
        raise NotImplementedError('Network unsupported')

    mb = models.model_base.ModelBase(args.network, depth, args.dataset, net)
    mb.cuda()

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = utils.data_utils.get_dataloader(args.dataset, args.batch_size, 1024, 4)

    # ====================================== start pruning ======================================
    mb.model.apply(models.base.init_utils.weights_init)

    if args.pruning == 'grasp':
        masks = pruner.GraSP(
            mb.model, args.ratio, trainloader, 'cuda', 
            num_classes=classes[args.dataset],
            samples_per_class=args.samples_per_class,
            num_iters=1)
    elif args.pruning == 'random':
        masks = pruner.random_ticket(mb.model, depth, args.ratio, vgg_scaling=('vgg' in args.network))
    elif args.pruning == 'random-uniform':
        masks = pruner.random_ticket(mb.model, depth, args.ratio, vgg_scaling=('vgg' in args.network), uniform=True)
    else:
        raise NotImplementedError

    if not args.prune_last:
        for k in masks.keys():
            if isinstance(k, nn.Linear):
                masks[k] = torch.ones_like(masks[k])

    mb.register_mask(masks)

    # ========== save pruned network ============
    state = {
        'net': mb.model.state_dict(),
        'acc': -1,
        'epoch': -1,
        'args': args,
        'mask': mb.masks,
        'ratio': mb.get_ratio_at_each_layer()
    }
    torch.save(state, os.path.join(args.root, 'ckpt_init.tar'))

    # ========== print pruning details ============
    print_mask_information(mb, logger)

    # ========== finetuning =======================
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9, weight_decay=args.wd)
    lr_schedule = {0: learning_rate,
                   int(args.epochs * 0.5): learning_rate * 0.1,
                   int(args.epochs * 0.75): learning_rate * 0.01}
    if args.lr_schedule == 'original':
        lr_scheduler = utils.common_utils.PresetLRScheduler(lr_schedule)
    elif args.lr_schedule == 'linear':
        lr_scheduler = utils.common_utils.LinearLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    best_acc = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, logger)
        test_acc = test(net, testloader, criterion, epoch, logger)

        logger.add_scalar(epoch, 'lr', lr_scheduler.get_lr(optimizer))
        logger.add_scalar(epoch, 'epoch_time', time.time() - t0)
        t0 = time.time()

        logger.iter_info()
        logger.save()

        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'args': config,
            'mask': mb.masks,
            'ratio': mb.get_ratio_at_each_layer()
        }
        torch.save(state, os.path.join(args.root, 'ckpt.tar'))
        if test_acc > best_acc:
            torch.save(state, os.path.join(args.root, 'ckpt_best.tar'))
            best_acc = test_acc
            best_epoch = epoch

    print('===> Best acc: %.4f, epoch: %d' % (best_acc, best_epoch))

if __name__ == '__main__':
    config = init_config()
    main(config)
