import torch
import sanitychecks

a = torch.Tensor([
    [
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[4,4,4], [5,5,5], [6,6,6]]
    ],
    [
        [[7,7,7],[8,8,8],[9,9,9]],
        [[10,10,10], [11,11,11], [12,12,12]]
    ]
])

masks = [
    torch.Tensor([
        [0, 1, 1],
        [1, 1, 0]
    ]),
    torch.Tensor([
        [0, 0, 1],
        [1, 1, 0]
    ])
]

print(sanitychecks.layerwise_rearrange(masks))

print(sanitychecks.shuffle(torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 6, 8]
])))

print(sanitychecks.randomize_pixels(a[0]))