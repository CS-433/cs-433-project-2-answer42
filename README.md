# ML Reproducibility Challenge (answer42)

# Paper to reproduce

[Sanity-checking pruning methods: Random tickets canwin the jackpot](https://arxiv.org/pdf/2009.11094v1.pdf) 
Jingtong Su, Yihang Chen, Tianle Cai, Tianhao Wu, Ruiqi Gao, Liwei Wang,and Jason D. Lee.

# Usage

## Reproduce ResNet32/VGG19 on CIFAR10 with GraSP pruning:

Run from GraSP directory:

```
cd GraSP
```

Several examples:

* <b>CIFAR-10, VGG19, Pruning ratio = 98%</b>

```
python main.py --config configs/cifar10/vgg19/GraSP_98.json --seed 0
```

* <b>CIFAR-10, VGG19, Pruning ratio = 98%, corrupted data</b>

```
python main.py --config configs/cifar10/vgg19/GraSP_98_corrupt.json
```

* <b>CIFAR-10, VGG19, Pruning ratio = 98%, rearranged layers</b>

```
python main.py --config configs/cifar10/vgg19/GraSP_98_rearrange.json
```

Checkout experiments logs: `GraSP/runs/pruning/cifar10/vgg19/...`

Code is based on the initial [GraSP repo](https://github.com/alecwangcq/GraSP)

GraSP paper: [Picking Winning Tickets Before Training by Preserving Gradient Flow](https://openreview.net/forum?id=SkgsACVKPH)


## Random Tickets

To reproduce results for Random Tickets move to the `cd random_tickets` directory and run the following command:
```(bash)
python train.py --dataset <cifar10/cifar100> --network <vgg/resnet> --pruning random --ratio <0.9/0.95/0.98> --wd 0.0005 --seed 0
```

For example, for VGG19 on CIFAR10 with 90% sparsity run the following command:
```(bash)
python train.py --dataset cifar10 --network vgg --pruning random --ratio 0.9 --wd 0.0005 --seed 0
```

You can then find the logs at `./logs/myexman-train.py/runs/<id>/logs.csv`
