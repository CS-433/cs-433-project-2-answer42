import argparse
import torch
import torch.nn as nn
import numpy as np
import copy
import os

from schedulers import PresetLRScheduler
from train import train, eval_net
from pruning import magnitude_pruning, hybrid_tickets
from pruning_utils import apply_masks
from argparse_utils import check_normalized, check_positive
from pipeline_config import supported_architectures, NUM_WORKERS, TEST_BATCH_SIZE, TRAINING_BATCH_SIZE, datasets, SMART_RATIO_CONFIG
from dataloading_utils import create_loader

import sanitychecks

def save_model(net, net_at_rewind_epoch, rewind_epoch, total_epochs, filename):
    """Saves model and its rewind parameters into the given file"""
    if not os.path.exists(filename):
        state = {
            'full_net': net.state_dict(),
            'rewind_net': net_at_rewind_epoch.state_dict(),
            'rewind_epoch': rewind_epoch,
            'total_epochs': total_epochs
        }
        print(f' => Saving model at \'{filename}\'')
        torch.save(state, filename)
    else:
        print(' *** A file on save path already exists!')


def load_model(filename, model_factory):
    """Loads model from the given path in the format like it's saved by save_model function"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f' *** No model at \'{filename}')
    print(f' => Loading model \'{filename}\'')
    state = torch.load(filename)
    net = model_factory()
    net.load_state_dict(state['full_net'])
    net_at_rewind_epoch = model_factory()
    net_at_rewind_epoch.load_state_dict(state['rewind_net'])
    rewind_epoch = state['rewind_epoch']
    total_epochs = state['total_epochs']
    return net, net_at_rewind_epoch, rewind_epoch, total_epochs


def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network and apply partially trained ticket methods')
    parser.add_argument('--epochs', type=check_positive, help='Number of epochs (must be >= 0)', default=None)
    supported_datasets_list = list(datasets.keys())
    supported_datasets_str = ', '.join(supported_datasets_list)
    parser.add_argument('--dataset', type=str, help=f'Dataset (supported: {supported_datasets_str})', 
                        required=True, choices=supported_datasets_list)
    supported_architectures_list = list(supported_architectures.keys())
    supported_architectures_str = ', '.join(supported_architectures_list)
    parser.add_argument('--architecture', type=str, help=f'Neural network architecture (supported: {supported_architectures_str})', 
                        choices=supported_architectures_list, required=True)
    parser.add_argument('--pruning_ratio', type=check_normalized, help='Percent of weights to prune (in range [0, 1])',
                        required=False, default=0)
    parser.add_argument('--hybrid_tickets', action='store_true')
    parser.add_argument('--rewind_epoch', type=check_positive, help='Epoch to rewind values to', default=None)
    parser.add_argument('--rewinding_type', type=str, choices=['weights', 'learning_rate'], default=None)
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Load model on specified path instead of training a new one')
    parser.add_argument('--fine_tuning_epochs', type=check_positive, help='Number of fine tuning epochs', 
                        default=0)
    parser.add_argument('--save_model', type=str, default=None, 
                        help='Save a trained model on specified path after training')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-sc', '--sanity_checks', type=str, nargs='*', choices=[
        'half_dataset',
        'layerwise_weights_shuffling'
    ])
    args = parser.parse_args()
    return args


def main(config):
    if config.seed is not None:
        print(f' => Using seed {config.seed}')
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    config.sanity_checks = set(config.sanity_checks if config.sanity_checks else [])

    # Prepare training parameters
    loss = nn.CrossEntropyLoss()
    learning_rate = 1e-1
    create_optimizer = lambda net: torch.optim.SGD(net.parameters(), 
        lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    create_model = lambda: supported_architectures[config.architecture](config.dataset)
    
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(' => Using device:', device)

    # Load Data
    dataset = datasets[config.dataset]
    train_data, test_data = dataset['data']()
    train_loader = create_loader(train_data, TRAINING_BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS)
    test_loader = create_loader(test_data, TEST_BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS)

    # Load or train model
    if config.load_model is not None:
        net, rewind_net, config.rewind_epoch, config.epochs = load_model(config.load_model, create_model)
    else:
        if config.epochs is None or config.rewind_epoch is None or config.rewind_epoch >= config.epochs:
            raise ValueError('Paramaters epoch and rewind epoch must be provided when ' + 
                    'training a model and rewind epoch must be smaller than epochs')
        lr_schedule = {0: learning_rate,
                   int(config.epochs * 0.5): learning_rate * 0.1,
                   int(config.epochs * 0.75): learning_rate * 0.01}
        scheduler = PresetLRScheduler(lr_schedule)
        net = create_model()
        rewind_net = None
        optimizer = create_optimizer(net)
        pruning_train_loader = train_loader
        if 'half_dataset' in config.sanity_checks:
            print(' => Using half dataset for pretraining')
            pruning_train_loader = create_loader(train_data, TRAINING_BATCH_SIZE, 
                shuffle=True, num_workers=NUM_WORKERS, half_dataset=True)
        # Train
        for i in range(config.epochs):
            print(f'Pretraining epoch {i}', end='')
            scheduler(optimizer, i)
            train_loss, train_acc = train(net, pruning_train_loader, optimizer, loss, device)    
            test_acc = eval_net(net, test_loader, device)
            print(f': loss {train_loss:.7f}, train_acc {train_acc * 100:.2f}%, test_acc {test_acc * 100:.2f}%')
            # Make a copy of model at rewind epoch
            if i == config.rewind_epoch:
                rewind_net = copy.deepcopy(net)
        # Save model if save path specified
        if config.save_model is not None:
            save_model(net, rewind_net, config.rewind_epoch, config.epochs, config.save_model)

    # Pruning and fine tuning
    if config.fine_tuning_epochs > 0:
        if config.rewinding_type is None:
            raise ValueError('Rewind type must be provided when doing pruning') 
        # Set seed again as these pretraining and fine tuning parts might be run independently
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        keep_ratio = 1. - config.pruning_ratio     
        print(f' => Pruning (keeping {keep_ratio * 100:.1f}% weights)')
        net.to(device)
        if config.hybrid_tickets:
            ratio_config = SMART_RATIO_CONFIG[config.architecture]
            masks = hybrid_tickets(net, keep_ratio, ratio_config['L'], ratio_config['vgg_scaling'])
        else:
            masks = magnitude_pruning(net, keep_ratio=keep_ratio)
        pruned_network = rewind_net if 'weights' == config.rewinding_type else net
        pruned_network.to(device)
        # Order of whether we apply weight shuffling or apply mask doesn't matter,
        #   but it's safer to do weight shuffling first if we want to make a copy
        #   as it's questionable whether mask would work if we apply it before shuffling weights
        if 'layerwise_weights_shuffling' in config.sanity_checks:
            print(' => Doing layerwise weights shuffling')
            pruned_network = sanitychecks.layerwise_weights_shuffling(pruned_network, make_copy=False)
        apply_masks(pruned_network, masks)

        # Fine tune
        lr_schedule = {0: learning_rate,
                   int(config.epochs * 0.5): learning_rate * 0.1,
                   int(config.epochs * 0.75): learning_rate * 0.01}
        scheduler = PresetLRScheduler(lr_schedule)
        optimizer = create_optimizer(pruned_network)
        for i in range(config.fine_tuning_epochs):
            print(f'Fine tuning epoch {i} (scheduling like it\'s {i + config.rewind_epoch})', end='')
            scheduler(optimizer, i + config.rewind_epoch)
            train_loss, train_acc = train(pruned_network, train_loader, optimizer, loss, device)    
            test_acc = eval_net(pruned_network, test_loader, device)
            print(f': loss {train_loss:.7f}, train_acc {train_acc * 100:.2f}%, test_acc {test_acc * 100:.2f}%')
    

if __name__ == "__main__":
    main(parse_args())
