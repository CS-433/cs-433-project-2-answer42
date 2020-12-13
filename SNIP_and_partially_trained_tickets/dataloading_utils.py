import torch
import torchvision
from torchvision.transforms import transforms

def load_cifar10():
    additional_train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]
    train_data = torchvision.datasets.CIFAR10('./cifar10', True, 
        transform=transforms.Compose(additional_train_transforms + test_transform_list), download=True)
    test_data = torchvision.datasets.CIFAR10('./cifar10', False, 
        transform=transforms.Compose(test_transform_list), download=True)
    return train_data, test_data


def load_cifar100():
    additional_train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
    ]
    train_data = torchvision.datasets.CIFAR100('./cifar100', True, 
        transform=transforms.Compose(additional_train_transforms + test_transform_list), download=True)
    test_data = torchvision.datasets.CIFAR100('./cifar100', False, 
        transform=transforms.Compose(test_transform_list), download=True)
    return train_data, test_data


def split_features_and_labels(dataset):
    features = [entry[0] for entry in dataset]
    labels = [entry[1] for entry in dataset]
    return features, labels


def create_loader(data, batch_size, shuffle=True, num_workers=2, half_dataset=False):
    """Creates loader for the given dataset and the batch sampling parameters.
    Also takes an additional parameter that makes data loader that satisfies
    Half Dataset sanity check."""
    if half_dataset:
        indices_to_keep = torch.randperm(len(data))[:len(data) // 2]
        sampler = torch.utils.data.SubsetRandomSampler(indices=indices_to_keep)
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
            num_workers=num_workers, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=shuffle)
    return dataloader