
import os, sys, random, yaml, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from argparse import Namespace
from models import schedulers


class AbstractModel(nn.Module):
    def __init__(self, hparams, device=None):
        super().__init__()
        self.compiled = False
        self.hparams = hparams
        self.device = device

    @classmethod
    def add_model_hparams(cls, parser):
        pass

    @staticmethod
    def add_parent_hparams(add_model_hparams):
        def foo(cls, parser):
            for base in cls.__bases__:
                base.add_model_hparams(parser)
            add_model_hparams(cls, parser)
        return foo

    @classmethod
    def load(cls, ckpt, device=None, strict=True):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=ckpt['hparams'])

        res = cls(hparams, device=device)
        res.load_state_dict(ckpt['state_dict'], strict=strict)

        if ckpt['optimizer'] is not None:
            res.compile()
            res.optimizer.load_state_dict(ckpt['optimizer'])

        return res.to(device)

    @classmethod
    def default(cls, device=None, **kwargs):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=Namespace(**kwargs))
        res = cls(hparams, device=device)
        return res

    def trainable_parameters(self):
        for p in super().parameters():
            if p.requires_grad:
                yield p

    def save(self, weights_file):
        torch.save(self.get_state_dict(), weights_file)

    def get_state_dict(self):
        return {
            'hparams': self.hparams,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.compiled else None
        }

""" Model that implements training and prediction on generator objects, with
the ability to print train and validation metrics.
"""
class TrainableModel(AbstractModel):
    @classmethod
    @AbstractModel.add_parent_hparams
    def add_model_hparams(cls, parser):
        # Optimizer Params
        parser.add_argument('--opt', default='adam', type=str)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--lr_schedule', default='linear', type=str)
        parser.add_argument('--weight_decay', default=0., type=float)

    def train_step(self, batch):
        assert self.compile

        res = self.step(batch)

        self.optimizer.zero_grad()
        res['loss'].backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return res

    def test_step(self, batch):
        with torch.no_grad():
            return self.step(batch)

    def predict(self, datagen):
        preds = [self.predict_on_batch(x) for x in datagen]
        preds = torch.cat(preds, dim=0)
        return preds

    def compile(self, device=None):
        if self.hparams.opt == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.opt == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        else:
            NotImplementedError

        self.scheduler = None

        self.compiled = True
        self.to(device)
