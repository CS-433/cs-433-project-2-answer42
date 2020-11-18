import os
from models import base, resnet, schedulers
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms


class BasicSupervised(base.TrainableModel):
    DATA_ROOT = os.environ.get('DATA_ROOT', './data')

    @classmethod
    @base.AbstractModel.add_parent_hparams
    def add_model_hparams(cls, parser):
        # Model params
        parser.add_argument('--net_arch', default='resnet32', type=str)
        # Data params
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--workers', default=2, type=int)

    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)

        if self.hparams.net_arch == 'resnet32':
            self.net = resnet.resnet(32)
        else:
            raise NotImplementedError

    def step(self, batch):
        x, y = batch
        logs = {}

        p = self.net(x)

        logs['acc'] = (y == p.argmax(1)).float()
        logs['loss'] = F.cross_entropy(p, y)

        return logs

    def get_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = datasets.CIFAR10(root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
        self.testset = datasets.CIFAR10(root=self.DATA_ROOT, train=False, download=True, transform=test_transform)

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_size=512,
        )

        return self.train_loader, self.test_loader

    def compile(self, device=None):
        super().compile(device)
        if self.hparams.lr_schedule == 'linear':
            self.scheduler = schedulers.LinearLR(self.optimizer, self.hparams.epochs * len(self.train_loader))
        elif self.hparams.lr_schedule == 'from-paper':
            self.scheduler = schedulers.PaperScheduler(self.optimizer, self.hparams.epochs * len(self.train_loader))
        elif self.hparams.lr_schedule == 'const':
            pass
        else:
            raise NotImplementedError
