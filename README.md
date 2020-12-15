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


<style type="text/css">
.tg  {border:none;border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-mxj2{background-color:#f9f9f9;border-color:inherit;font-style:italic;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-btxf{background-color:#f9f9f9;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:top}
</style>
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
