import torch
import torchvision
from torchvision.transforms import transforms

def prepare_cifar10(training_batch_size, num_workers=2):
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

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=training_batch_size, 
        shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader