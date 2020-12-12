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


def split_features_and_labels(dataset):
    features = [entry[0] for entry in dataset]
    labels = [entry[1] for entry in dataset]
    return features, labels


def create_loader(data, batch_size, shuffle=True, num_workers=2):
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers)
    return dataloader