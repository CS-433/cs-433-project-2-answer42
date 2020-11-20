# ML Reproducibility Challenge (answer42)

## Paper to reproduce

[Jingtong Su, Yihang Chen, Tianle Cai, Tianhao Wu, Ruiqi Gao, Liwei Wang,
and Jason D. Lee. Sanity-checking pruning methods: Random tickets can
win the jackpot, 2020.](https://arxiv.org/pdf/2009.11094v1.pdf)

## Usage

Reproduce ResNet32 on CIFAR10 w/o prunning (work in progress):

```python source/train.py --model basic-supervised --epochs 160 --workers 4 --batch_size 64 --lr_schedule from-paper --lr 0.1 --opt sgd --weight_decay 1e-4 --name reproduce-nopruning-baseline --save_freq 10 --eval_freq 5```
