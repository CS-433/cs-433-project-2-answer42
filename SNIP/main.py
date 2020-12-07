import argparse
import torch
import torch.nn as nn

from schedulers import PresetLRScheduler
from models import resnet32, vgg19
from train import train, eval_net
from snip import SNIP
from pruning_utils import apply_masks
from argparse_utils import check_normalized, check_positive
from dataloading_utils import prepare_cifar10


def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network and apply SNIP')
    parser.add_argument('--epochs', type=check_positive, help='Number of epochs (must be >= 0)', 
                        required=True)
    parser.add_argument('--dataset', type=str, help='Dataset (supported: cifar10, cifar100, tinyimagenet)', 
                        required=True, choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--architecture', type=str, help='Neural network architecture (supported: resnet32, vgg19)',
                        required=True, choices=['resnet32', 'vgg19'])
    parser.add_argument('--pruning_ratio', type=check_normalized, help='Percent of weights to prune (in range [0, 1])',
                        required=False, default=0)
    args = parser.parse_args()
    return args


def main(config):
    supported_architectures = {
        'resnet32': resnet32,
        'vgg19': vgg19
    }
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    datasets = {
        'cifar10': lambda: prepare_cifar10(training_batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    }

    net = supported_architectures[config.architecture](config.dataset)
    loss = nn.CrossEntropyLoss()
    learning_rate = 1e-1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_schedule = {0: learning_rate,
                   int(config.epochs * 0.5): learning_rate * 0.1,
                   int(config.epochs * 0.75): learning_rate * 0.01}
    scheduler = PresetLRScheduler(lr_schedule)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Used device:', device)

    train_loader, test_loader = datasets[config.dataset]()
    keep_ratio = 1. - config.pruning_ratio
    net.to(device)
    masks = SNIP(net, keep_ratio, train_loader, device, loss)
    print(f'Mask calcualted (keeping {keep_ratio * 100:.1f}% weights)')
    apply_masks(net, masks)

    for i in range(config.epochs):
        print(f'Epoch {i}', end='')
        scheduler(optimizer, i)
        train_loss, train_acc = train(net, train_loader, optimizer, loss, device)    
        test_acc = eval_net(net, test_loader, device)
        print(f': loss {train_loss:.6f}, train_acc {train_acc * 100:.1f}%, test_acc {test_acc * 100:.1f}%')


if __name__ == "__main__":
    main(parse_args())
