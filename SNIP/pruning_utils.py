def get_fc_and_conv_layers(net):
    """Returns list of convolution and fully connected layers from the passed
    neural network

    Parameters
    ----------
    net: nn.Module
        Neural network
    """
    def is_fc_or_conv_layer(layer):
        """Checks whether the passed layer is convolution or fully connected and
        returns true if it is either of those
        
        Parameters
        ----------
        layer: nn.Module
            Pytorch module
        """
        return isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
    return list(filter(is_fc_or_conv_layer, net.modules()))


def apply_masks(net, pruning_masks):
    """Prunes the weights of the passed network based on the pruning_mask

    Parameters
    ----------
    net: nn.Module
        Neural network
    pruning_masks: List[torch.Tensor]
        A list of masks for each fully connected and convolutional layer of 
        the neural network. It assumes masks are in the same order as the 
        specified layers in the neural network
    """
    def hook_factory(pruning_mask):
        """Makes hook that prevents backpropagation from affecting
        pruned weights in the network
        """
        def hook(grads):
            return grads * pruning_mask
        return hook

    # Get specified layers
    layers_to_prune = get_fc_and_conv_layers(net)

    for layer, mask in zip(layers_to_prune, pruning_masks):
        assert layer.weight.shape == mask.shape
        # Apply mask on the layer
        layer.weight.data = layer.weight.data * mask
        layer.weight.register_hook(hook_factory(mask))
        

        
