import torch
import copy
from pruning_utils import get_fc_and_conv_layers

def shuffle(tensor):
    """Shuffles tensor
    
    Parameters
    ----------
    tensor : torch.Tensor
    """
    return tensor[torch.randperm(tensor.size(0))]
    

def flat_shuffle(tensor):
    """Shuffles elements of multidimension tensor
    
    Parameters
    ----------
    tensor : torch.Tensor
    """
    shape_ = tensor.size()
    flat_tensor = tensor.view(-1)
    shuffled_flat_tensor = shuffle(flat_tensor)
    return shuffled_flat_tensor.view(shape_)

    
def randomize_pixels(image):
    """Shuffles pixels of image in the input tensor
    
    Parameters
    ----------
    image : torch.Tensor
        3d Tensor (list of images where last dimension is number of color channels)
    """
    shape_ = image.size()
    image_flat = image.view(-1, image.size(-1))
    shuffled_image = shuffle(image_flat)
    return shuffled_image.view(shape_)


def random_labels(size, num_classes):
    """Returns randomly generated labels for the given size from a uniform distribution"""
    return torch.randint(high=num_classes, size=(size,)).int().tolist()


def layerwise_rearrange(masks):
    """Performs layerwise rearrange
    
    Parameters
    ----------
    masks: List[torch.Tensor]
        List of pruning masks
    """
    return [flat_shuffle(mask) for mask in masks]


def layerwise_weights_shuffling(net, make_copy=False):
    """Shuffle weights of each prunable layer of the given
    network

    Parameters
    ----------
    net : nn.Module
        Neural network
    make_coopy : bool
        Specifies whether a sanity check should be done directly on a network or a copy of it
    """
    if make_copy:
        net = copy.deepcopy(net)
    for layer in get_fc_and_conv_layers(net):
        layer.weight.data = flat_shuffle(layer.weight.data)
    return net