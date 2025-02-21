import argparse
import torch
import torch.nn as nn
import numpy as np

from schedulers import PresetLRScheduler
from train import train, eval_net
from pruning import SNIP
from utils.argparse_utils import check_normalized, check_positive
from pipeline_config import supported_architectures, NUM_WORKERS, TEST_BATCH_SIZE, TRAINING_BATCH_SIZE, datasets
from utils.dataloading_utils import create_loader, split_features_and_labels
from utils.pruning_utils import calculate_network_sparsity_ratio, apply_masks

import sanitychecks


def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network and apply SNIP')
    parser.add_argument('--epochs', type=check_positive, help='Number of epochs (must be >= 0)', 
                        required=True)
    supported_datasets_list = list(datasets.keys())
    supported_datasets_str = ', '.join(supported_datasets_list)
    parser.add_argument('--dataset', type=str, help=f'Dataset (supported: {supported_datasets_str})', 
                        required=True, choices=supported_datasets_list)
    supported_architectures_list = list(supported_architectures.keys())
    supported_architectures_str = ', '.join(supported_architectures_list)
    parser.add_argument('--architecture', type=str, help=f'Neural network architecture (supported: {supported_architectures_str})',
                        required=True, choices=supported_architectures_list)
    parser.add_argument('--pruning_ratio', type=check_normalized, help='Percent of weights to prune (in range [0, 1])',
                        required=False, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-sc', '--sanity_checks', type=str, nargs='*', choices=[
        'random_labels', 
        'random_pixels',
        'layerwise_rearrange'
    ])
    args = parser.parse_args()
    return args


def main(config):
    if config.seed is not None:
        print(f' => Using seed {config.seed}')
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

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

    net_sparsity_ratio = calculate_network_sparsity_ratio(net)
    print(f' => Sparsity ratio before training {net_sparsity_ratio}')

    for i in range(config.epochs):
        print(f'Epoch {i}', end='')
        scheduler(optimizer, i)
        train_loss, train_acc = train(net, train_loader, optimizer, loss, device)    
        test_acc = eval_net(net, test_loader, device)
        print(f': loss {train_loss:.7f}, train_acc {train_acc * 100:.2f}%, test_acc {test_acc * 100:.2f}%')

    net_sparsity_ratio = calculate_network_sparsity_ratio(net)
    print(f' => Sparsity ratio after training {net_sparsity_ratio}')


if __name__ == "__main__":
    main(parse_args())
