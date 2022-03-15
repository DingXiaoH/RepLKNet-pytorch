# RepLKNet-pytorch (CVPR 2022)

This is the official PyTorch implementation of **RepLKNet**, from the following CVPR-2022 paper:

Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs.

The paper is released on arXiv: https://arxiv.org/abs/2203.06717.

## Other implementations

| framework | link |
|:---:|:---:|
|MegEngine (official)|https://github.com/megvii-research/RepLKNet|
|PyTorch (official)|https://github.com/DingXiaoH/RepLKNet-pytorch|
|Tensorflow| re-implementations are welcomed |
|PaddlePaddle  | re-implementations are welcomed |
| ... | |

More re-implementations and efficient conv kernel optimizations are welcomed.

## Catalog
- [x] Model code
- [x] PyTorch pretrained models
- [ ] PyTorch training code
- [ ] PyTorch downstream models
- [ ] PyTorch downstream code

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K Models

| name | resolution |acc | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:|:---:|
|:RepLKNet-31B:|:224x224:|:83.5:|:---:| :---:|:[Google Drive](https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view?usp=sharing) | Baidu Cloud:|
|:RepLKNet-31B:|:384x384:|:84.8:|:---:| :---:|:---:|

| name | resolution |acc | #params | FLOPs | model |

| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |


### ImageNet-22K Models

| name | resolution |acc | #params | FLOPs | 22K model | 1K model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|



### MegData-73M Models
| name | resolution |acc@1 | #params | FLOPs | MegData-73M model | 1K model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|



## Evaluation


## Training


## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.




Accepted by CVPR 2022!

The paper is released on arXiv: https://arxiv.org/abs/2203.06717.

Model code released.

Uploading model weights. Almost completed.

Working on the training code.


