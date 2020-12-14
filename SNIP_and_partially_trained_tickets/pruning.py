import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import types

from pruning_utils import get_fc_and_conv_layers


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, loss_fn):

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs = [torch.abs(layer.weight_mask.grad) for layer in get_fc_and_conv_layers(net)]

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    #print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)


def magnitude_pruning(net, keep_ratio):
    """Prunes the given network such that only keep_ratio percent of
    weights with the highest magnitude remains unpruned. The 
    result is a list of masks, one of each "prunable" layer in the network
    in the order these layers appear given by nn.Module.modules() method

    Parameters
    ----------
    net : nn.Module
        Neural network
    keep_ratio : float
        A number between 0 and 1 that indicates a percent of weight to keep
    """
    weights_abs = [torch.abs(layer.weight.data) for layer in get_fc_and_conv_layers(net)]

    all_weights = torch.cat([torch.flatten(x) for x in weights_abs])

    num_weights_to_keep = int(len(all_weights) * keep_ratio)
    threshold, _ = torch.topk(all_weights, num_weights_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for weight_magnitude in weights_abs:
        keep_masks.append((weight_magnitude >= acceptable_score).float())

    return keep_masks


def calculate_smart_ratios(net, L, ratio, vgg_scaling=False, last_ratio=0.3, uniform=False):
    """Calculate smart ratio for each layer of the given network with respect to the passed parameters

    Parameters
    ----------
    net : nn.Module
        Neural network
    L : int

    ratio : float
        Percentage of weights to prune (sparsity ratio)
    vgg_scaling : bool
        Indicate whether the architecture of the network is VGG as it uses special scaling
    last_ratio : float
        Keep ratio of the last, linear, layer
    uniform : bool
    """
    prunable_layers = get_fc_and_conv_layers(net)

    n_params = np.array([m.weight.shape.numel() for m in prunable_layers])

    p = np.arange(1, n_params.shape[0] + 1).astype(float)
    if vgg_scaling:
        p = ((L - p + 1)**2 + (L - p + 1)) / p**2
    else:
        p = (L - p + 1)**2 + (L - p + 1)
    
    if uniform:
        p = p*0 + 1

    p /= ((p * n_params) / ((1 - ratio) * n_params.sum() - last_ratio * n_params[-1])).sum()
    p[-1] = last_ratio
    print(f'===> smart ratios before rearanging:', ', '.join([f"{x:.3f}" for x in p]))
    for i in range(len(p)):
        if p[i] > 1.:
            p[i + 1] = p[i+1] + (p[i] - 1) * n_params[i] / n_params[i+1]
            p[i] = 1
    print(f'===> smart ratios:', ', '.join([f"{x:.3f}" for x in p]))
    print(f'===> total keep ratio: {(n_params * p).sum()/n_params.sum()}')
    return p.tolist()


def hybrid_tickets(net, keep_ratio, L, vgg_scaling=False):
    """Calculates pruning mask for the given network using hybrid tickets approach

    Parameters
    ----------
    net : nn.Module
        Neural network
    keep_ratio : float
        Percent of weights to keep
    L : int

    vgg_scaling : bool
        Indicate whether the architecture of the network is VGG as it uses special scaling
    """
    keep_masks = []
    smart_keep_ratios = calculate_smart_ratios(
        net, L, 1 - keep_ratio, vgg_scaling=vgg_scaling)
    for layer, layer_keep_ratio in zip(get_fc_and_conv_layers(net), smart_keep_ratios):
        # magnitude pruning returns a single array mask for a layer passed as a neural net
        current_layer_mask = magnitude_pruning(layer, layer_keep_ratio)[0]
        keep_masks.append(current_layer_mask)
    return keep_masks