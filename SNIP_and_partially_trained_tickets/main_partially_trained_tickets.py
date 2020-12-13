import argparse
import torch
import torch.nn as nn

from schedulers import PresetLRScheduler
from models import resnet32, vgg19
from train import train, eval_net
from pruning import SNIP
from pruning_utils import apply_masks
from argparse_utils import check_normalized, check_positive
from dataloading_utils import load_cifar10, load_cifar100, create_loader, split_features_and_labels

import sanitychecks


def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network and apply partially trained ticket methods')
    parser.add_argument('--epochs', type=check_positive, help='Number of epochs (must be >= 0)', 
                        required=True)
    parser.add_argument('--dataset', type=str, help='Dataset (supported: cifar10, cifar100, tinyimagenet)', 
                        required=True, choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--architecture', type=str, help='Neural network architecture (supported: resnet32, vgg19)',
                        required=True, choices=['resnet32', 'vgg19'])
    parser.add_argument('--pruning_ratio', type=check_normalized, help='Percent of weights to prune (in range [0, 1])',
                        required=False, default=0)
    parser.add_argument('--only_train',  action='store_true')
    parser.add_argument('--rewind_epoch', type=check_positive, help='Epoch to rewind values to', required=True)
    parser.add_argument('--rewinding_type', type=str, choices=['weights', 'learning_rate'], required=True)
    
    parser.add_argument('-sc', '--sanity_checks', type=str, nargs='*', choices=[
        'half_dataset',
        'layerwise_weights_shuffling'
    ])
    args = parser.parse_args()
    return args


def main(config):
    supported_architectures = {
        'resnet32': resnet32,
        'vgg19': vgg19
    }
    TEST_BATCH_SIZE = 1024
    TRAINING_BATCH_SIZE = 64
    NUM_WORKERS = 4
    datasets = {
        'cifar10': {
            'data': load_cifar10,
            'num_classes': 10
        },
        'cifar100': {
            'data': load_cifar100,
            'num_classes': 100
        }
    }
        
    config.sanity_checks = set(config.sanity_checks if config.sanity_checks else [])

    print(config.__dict__)


if __name__ == "__main__":
    main(parse_args())
