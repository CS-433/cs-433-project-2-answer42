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


## SNIP

It is recommended to run SNIP code from its directory:

```(bash)
cd SNIP_and_partially_trained_tickets
```

In the given directory is a file `main_snip.py` which does pruning according to SNIP and trains network afterwards based on used parameters. It can be run using the following command:

```(bash)
python main_snip.py --dataset <cifar10/cifar100> --architecture <vgg19/resnet32> --epochs [number of training epochs] --pruning_ratio [percent of weights to prune in range between 0 and 1 (inclusive)] --seed [value of seed] -sc [names of sanity checks divided by whitespace]
```

Supported sanity checks for SNIP are: **random_labels**, **random_pixels**, **layerwise_rearrange**.

##### Examples of usage:

* **CIFAR-10, VGG19, Pruning ratio = 98%, training for 160 epochs after pruning**

```(bash)
python main_snip.py --dataset cifar10 --architecture vgg19 --epochs 160 --pruning_ratio 0.98 --seed 2020
```

* **CIFAR-10, VGG19, Pruning ratio = 98%, training for 160 epochs after pruning, corrupted data**

```(bash)
python main_snip.py --dataset cifar10 --architecture vgg19 --epochs 160 --pruning_ratio 0.98 --seed 2020 -sc random_pixels random_labels
```

* **CIFAR-10, VGG19, Pruning ratio = 98%, training for 160 epochs after pruning, layerwise rearrange**

```(bash)
python main_snip.py --dataset cifar10 --architecture vgg19 --epochs 160 --pruning_ratio 0.98 --seed 2020 -sc layerwise_rearrange
```

## Random Tickets


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow" colspan="3">CIFAR-10</th>
    <th class="tg-c3ow" colspan="3">CIFAR-100</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-btxf"><span style="text-decoration:none">Network\Sparsity</span></td>
    <td class="tg-abip">90%</td>
    <td class="tg-abip">95%</td>
    <td class="tg-abip">98%</td>
    <td class="tg-abip">90%</td>
    <td class="tg-abip">95%</td>
    <td class="tg-abip">98%</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:normal;font-style:normal;text-decoration:none">VGG19</span></td>
    <td class="tg-c3ow">93.65 (-0.12)</td>
    <td class="tg-c3ow">93.22 (-0.20)</td>
    <td class="tg-c3ow">92.41 (-0.04)</td>
    <td class="tg-c3ow">72.75 (0.20)</td>
    <td class="tg-c3ow">72.00 (0.63)</td>
    <td class="tg-c3ow">68.77 (-0.21)</td>
  </tr>
  <tr>
    <td class="tg-btxf"><span style="font-weight:normal;font-style:normal;text-decoration:none">ResNet32</span></td>
    <td class="tg-abip"><span style="font-weight:normal;font-style:normal;text-decoration:none">92.84 (-0.13)</span></td>
    <td class="tg-abip">91.64 (0.04)</td>
    <td class="tg-abip">88.68 (-0.42)</td>
    <td class="tg-mxj2"><span style="font-weight:normal;text-decoration:none">68.50 (-1.20)</span></td>
    <td class="tg-abip">65.99 (-0.83)</td>
    <td class="tg-abip">59.67 (-0.44)</td>
  </tr>
</tbody>
</table>

To reproduce results for Random Tickets move to the `cd random_tickets` directory and run the following command:
```(bash)
python train.py --dataset <cifar10/cifar100> --network <vgg/resnet> --pruning random --ratio <0.9/0.95/0.98> --wd 0.0005 --seed 0
```

For example, for VGG19 on CIFAR10 with 90% sparsity run the following command:
```(bash)
python train.py --dataset cifar10 --network vgg --pruning random --ratio 0.9 --wd 0.0005 --seed 0
```

You can then find the logs at `./logs/myexman-train.py/runs/<id>/logs.csv`

## Partially-trained Tickets

It is recommended to run code for partially-trained tickets from its directory:

```(bash)
cd SNIP_and_partially_trained_tickets
```

In the given directory is a file `main_partially_trained_tickets.py` which supports partially-trained ticket pruning methods used in the original paper and their corresponding sanity checks. It can be run using the following command:

```(bash)
python main_partially_trained_tickets.py --dataset <cifar10/cifar100> --architecture <vgg19/resnet32> --epochs [number of training epochs] --fine_tuning_epochs [number of fine-tuning epochs] --pruning_ratio [percent of weights to prune in range between 0 and 1 (inclusive)] --seed [value of seed] --rewinding_type <weights/learning_rate> --rewind_epoch [number of epoch to which scheduler or weights should be rewinded after pruning] -sc [names of sanity checks divided by whitespace]
```

The given command also accepts an optinal argument `--hybrid_tickets` which specifies that hybrid tickets method should be used while pruning weights. According to the original paper `learning_rate` should be used as a `rewinding_type` with hybrid tickets, but the code will also work with `weights` as `rewinding_type`.

Supported sanity checks for partially-trained tickets are: **half_dataset**, **layerwise_weights_shuffling**.

All the results for partially-trained tickets are obtained using seed **2020**.

##### Examples of usage:

* **CIFAR-10, VGG19, Pre-trained for 160 epochs on half dataset, pruned 98% of weights, rewinded to the 40th epoch using weight rewinding and retrained for additional 160 epochs**

```(bash)
python main_partially_trained_tickets.py --dataset cifar10 --architecture vgg19 --epochs 160 --rewind_epoch 40 --fine_tuning_epochs 160 --pruning_ratio 0.98 --seed 2020 --rewinding_type weights -sc half_dataset
```

* **CIFAR-100, ResNet32, Pre-trained for 160 epochs, pruned using hybrid tickets 95% of weights and rewinded to the 40th epoch using learning rate rewinding and retrained for additional 160 epochs**

```(bash)
python main_partially_trained_tickets.py --dataset cifar100 --architecture resnet32 --epochs 160 --rewind_epoch 40 --fine_tuning_epochs 160 --pruning_ratio 0.95 --hybrid_tickets --seed 2020 --rewinding_type learning_rate
```

* **CIFAR-100, ResNet32, Pre-trained for 160 epochs, pruned 90% of weights, performed layerwise weight shuffling and rewinded to the 40th epoch using learning rate rewinding and retrained for additional 160 epochs**

```(bash)
python main_partially_trained_tickets.py --dataset cifar100 --architecture resnet32 --epochs 160 --rewind_epoch 40 --fine_tuning_epochs 160 --pruning_ratio 0.9 --seed 2020 --rewinding_type learning_rate -sc layerwise_weights_shuffling
```

## Authors

* Andrei Atanov: andrei.atanov@epfl.ch
* Valentina Shumovskaia: valentina.shumovskaia@epfl.ch
* Miloš Vujasinović: milos.vujasinovic@epfl.ch