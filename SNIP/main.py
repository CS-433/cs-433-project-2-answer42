import argparse
import torch
import torch.nn as nn

from schedulers import PresetLRScheduler
from models import resnet32, vgg19
from train import train, eval_net
from snip import SNIP
from pruning_utils import apply_masks
from argparse_utils import check_normalized, check_positive
from dataloading_utils import load_cifar10, create_loader, split_features_and_labels

import sanitychecks


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
    parser.add_argument('-sc', '--sanity_checks', type=str, nargs='*', choices=[
        'random_labels', 
        'random_pixels',
        'half_dataset',
        'layerwise_rearrange',
        'layerwise_weights_shuffling'])
    args = parser.parse_args()
    return args


def main(config):
    supported_architectures = {
        'resnet32': resnet32,
        'vgg19': vgg19
    }
    TEST_BATCH_SIZE=1024
    TRAINING_BATCH_SIZE = 64
    NUM_WORKERS = 4
    datasets = {
        'cifar10': {
            'data': load_cifar10,
            'num_classes': 10
        }
    }
        
    config.sanity_checks = set(config.sanity_checks if config.sanity_checks else [])

    net = supported_architectures[config.architecture](config.dataset)
    loss = nn.CrossEntropyLoss()
    learning_rate = 1e-1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_schedule = {0: learning_rate,
                   int(config.epochs * 0.5): learning_rate * 0.1,
                   int(config.epochs * 0.75): learning_rate * 0.01}
    scheduler = PresetLRScheduler(lr_schedule)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(' *** Used device:', device)

    dataset = datasets[config.dataset]
    train_data, test_data = dataset['data']()
    train_loader = create_loader(train_data, TRAINING_BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS)
    test_loader = create_loader(test_data, TEST_BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS)

    pruning_data = train_data
    pruning_features, pruning_labels = split_features_and_labels(pruning_data)

    if 'random_labels' in config.sanity_checks:
        pruning_labels = sanitychecks.random_labels(
            len(pruning_labels), dataset['num_classes'])
        print(' *** Random labels done')
    
    if 'random_pixels' in config.sanity_checks:
        pruning_features = [
            sanitychecks.randomize_pixels(image)\
                for image in pruning_features
        ]
        print(' *** Random pixels done')

    pruning_dataloader = create_loader(list(zip(pruning_features, pruning_labels)), 
        TRAINING_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    keep_ratio = 1. - config.pruning_ratio
    net.to(device)
    masks = SNIP(net, keep_ratio, pruning_dataloader, device, loss)
    print(f' *** Mask calcualted (keeping {keep_ratio * 100:.1f}% weights)')
    if 'layerwise_rearrange' in config.sanity_checks:
        masks = sanitychecks.layerwise_rearrange(masks)
        print(' *** Layerwise rearrange done')
    apply_masks(net, masks)

    for i in range(config.epochs):
        print(f'Epoch {i}', end='')
        scheduler(optimizer, i)
        train_loss, train_acc = train(net, train_loader, optimizer, loss, device)    
        test_acc = eval_net(net, test_loader, device)
        print(f': loss {train_loss:.7f}, train_acc {train_acc * 100:.2f}%, test_acc {test_acc * 100:.2f}%')


if __name__ == "__main__":
    main(parse_args())
