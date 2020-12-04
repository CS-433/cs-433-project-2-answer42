import torch
import numpy as np
import warnings

class PaperScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            if self.last_epoch >= int(0.5 * self.num_epochs):
                lr *= 0.1
            if self.last_epoch >= int(0.75 * self.num_epochs):
                lr *= 0.1
            res.append(lr)
        return res