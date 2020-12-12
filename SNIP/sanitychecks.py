import torch

def shuffle(tensor):
    """Shuffles tensor
    
    Parameters
    ----------
    tensor : torch.Tensor
    """
    return tensor[torch.randperm(tensor.size(0))]
    
    
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
    def shuffle_layer_mask(mask):
        shape_ = mask.size()
        flat_mask = mask.view(-1)
        shuffled_flat_mask = shuffle(flat_mask)
        return shuffled_flat_mask.view(shape_)
    return [shuffle_layer_mask(mask) for mask in masks]