import torch

def rearrange_tensor(tensor):
    tensor_shape = tensor.shape
    tensor = tensor.view(-1)
    perm = torch.randperm(tensor.shape[0])
    tensor = tensor[perm]
    tensor = tensor.view(tensor_shape)
    return tensor