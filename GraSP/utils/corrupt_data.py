import torch

def get_num_classes(dataset):
    num_classes = None
    if dataset == 'cifar10' or dataset == 'cinic-10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'tiny_imagenet':
        return 200
    
def corrupt_label(batch, num_classes):
    pic, target = batch
    corr_target = torch.randint(low=0, high=num_classes, size=target.size())
    return [pic, corr_target]

def corrupt_pixels(batch):
    pic, target = batch
    bsize, xsize, ysize = pic.shape[0], pic.shape[2], pic.shape[3]
    for b in range(bsize):
        xperm, yperm = torch.randperm(xsize), torch.randperm(ysize)
        pic[b, :, :, :] = pic[b, :, xperm, :]
        pic[b, :, :, :] = pic[b, :, :, yperm]
    return [pic, target]

def corrupt_batch(batch, num_classes):
    batch = corrupt_label(batch, num_classes)
    batch = corrupt_pixels(batch)
    return batch

def corrupt_dataloader(dataloader, num_classes):
    batches = []
    for batch in dataloader:
        batches.append(corrupt_batch(batch, num_classes))
    return batches