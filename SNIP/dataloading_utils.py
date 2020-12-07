import torch
import torchvision
from torchvision.transforms import transforms

def prepare_cifar10(training_batch_size, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_data = torchvision.datasets.CIFAR10('./cifar10', True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10('./cifar10', False, transform=transform, download=True)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=training_batch_size, 
        shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader