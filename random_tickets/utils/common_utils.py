import os
import torch
import numpy as np


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def try_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class StairCaseLRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (start_at >= 0) \
                and (iteration >= start_at) \
                and (iteration + 1) % interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                print('[%d]Decay lr to %f' % (iteration, param_group['lr']))

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        print(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self, *args):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res
    
    def __call__(self, *args):
        self.step()


