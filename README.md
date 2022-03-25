# RepLKNet-pytorch (CVPR 2022)

This is the official PyTorch implementation of **RepLKNet**, from the following CVPR-2022 paper:

Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs.

The paper is released on arXiv: https://arxiv.org/abs/2203.06717.

Update: will upload the models and update the training commands in two days.

## Other implementations

| framework | link |
|:---:|:---:|
|MegEngine (official)|https://github.com/megvii-research/RepLKNet|
|PyTorch (official)|https://github.com/DingXiaoH/RepLKNet-pytorch|
|Tensorflow|https://github.com/shkarupa-alex/tfreplknet|
| ... | |

More re-implementations are welcomed.

## Use our efficient large-kernel convolution with PyTorch

We have released an example for **PyTorch**. Please check ```setup.py``` and ```depthwise_conv2d_implicit_gemm.py``` (**a replacement of torch.nn.Conv2d**) in https://github.com/MegEngine/cutlass/tree/master/examples/19_large_depthwise_conv2d_torch_extension.

1. Clone ```cutlass``` (https://github.com/MegEngine/cutlass), enter the directory.
2. ```cd examples/19_large_depthwise_conv2d_torch_extension```
3. ```./setup.py install --user```. If you get errors, check your ```CUDA_HOME```.
4. A quick check: ```python depthwise_conv2d_implicit_gemm.py```
5. Add ```WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` into your ```PYTHONPATH``` so that you can ```from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM``` anywhere. Then you may use ```DepthWiseConv2dImplicitGEMM``` as a replacement of ```nn.Conv2d```.
6. ```export LARGE_KERNEL_CONV_IMPL=WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` so that RepLKNet will use the efficient implementation. Or you may simply modify the related code (```get_conv2d```) in ```replknet.py```.

Our implementation mentioned in the paper has been integrated into MegEngine. The engine will automatically use it. If you would like to use it in other frameworks like Tensorflow, you may need to compile our released cuda sources (the ```*.cu``` files in the above example should work with other frameworks) and use some tools to load them, just like ```cutlass``` and ```torch.utils.cpp_extension``` in the PyTorch example. Would be appreciated if you could share with us your experience.

You may refer to the MegEngine source code: https://github.com/MegEngine/MegEngine/tree/8a2e92bd6c5ac02807b27d174dce090ee391000b/dnn/src/cuda/conv_bias/chanwise. . 

Pull requests (e.g., better or other implementations or implementations on other frameworks) are welcomed.

## Catalog
- [x] Model code
- [x] PyTorch pretrained models
- [x] PyTorch large-kernel conv impl
- [x] PyTorch training code
- [ ] PyTorch downstream models
- [ ] PyTorch downstream code

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K Models

| name | resolution |ImageNet-1K acc | #params | FLOPs | ImageNet-1K pretrained model |
|:---:|:---:|:---:|:---:| :---:|:---:|
|RepLKNet-31B|224x224|83.5| 79M   |  15.3G   |[Google Drive](https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1gspbbfqooMtegt_DO1TUeA?pwd=lknt)|
|RepLKNet-31B|384x384|84.8| 79M   |  45.1G   |[Google Drive](https://drive.google.com/file/d/1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1WhLaCKKv4NuKc3qMYECOIQ?pwd=lknt)|



### ImageNet-22K Models

| name | resolution |ImageNet-1K acc | #params | FLOPs | 22K pretrained model | 1K finetuned model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
|RepLKNet-31B|224x224|  85.2  |  79M  |  15.3G  |[Google Drive](https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1YiQSn7VJDiNWX1IWg19O6g?pwd=lknt)|[Google Drive](https://drive.google.com/file/d/1DslZ2voXZQR1QoFY9KnbsHAeF84hzS0s/view?usp=sharing), [Baidu](https://pan.baidu.com/s/169wDunCdop-jQM8K-AX27g?pwd=lknt)|
|RepLKNet-31B|384x384|  86.0  |  79M  | 45.1G   | - |[Google Drive](https://drive.google.com/file/d/1Sc46BWdXXm2fVP-K_hKKU_W8vAB-0duX/view?usp=sharing), [Baidu](https://pan.baidu.com/s/11-F3JIKEzSOU7KUhebUWEQ?pwd=lknt)|
|RepLKNet-31L|384x384|  86.6  |  172M  |  96.0G  |[Google Drive](https://drive.google.com/file/d/16jcPsPwo5rko7ojWS9k_W-svHX-iFknY/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1KWazk_cOVYoLuuVJc_9HxA?pwd=lknt)|[Google Drive](https://drive.google.com/file/d/1JYXoNHuRvC33QV1pmpzMTKEni1hpWfBl/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1MWIsPXJJV4mTEtgcKv3F3w?pwd=lknt)|


### MegData-73M Models
(uploading)
| name | resolution |ImageNet-1K acc | #params | FLOPs | MegData-73M pretrained model | 1K finetuned model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
|RepLKNet-XL| 320x320 | 87.8 | 335M | 128.7G | 



## Evaluation
For RepLKNet-31B/L with 224x224 or 384x384, we use the "IMAGENET_DEFAULT_MEAN/STD" for preprocessing (see [here](https://github.com/rwightman/pytorch-image-models/blob/73ffade1f8203a611c9cdd6df437b436b780daca/timm/data/constants.py#L2)). For examples,
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31B --batch_size 32 --eval True --resume RepLKNet-31B_ImageNet-1K_224.pth --input_size 224
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31L --batch_size 32 --eval True --resume RepLKNet-31L_ImageNet-22K-to-1K_384.pth --input_size 384
```
For RepLKNet-XL, please note that we used mean=\[0.5,0.5,0.5\] and std=\[0.5,0.5,0.5\] for preprocessing on MegData73M dataset as well as finetuning on ImageNet-1K. This mean/std setting is also referred to as "IMAGENET_INCEPTION_MEAN/STD" in timm, see [here](https://github.com/rwightman/pytorch-image-models/blob/73ffade1f8203a611c9cdd6df437b436b780daca/timm/data/constants.py#L4). Add ```--imagenet_default_mean_and_std false``` to use this mean/std setting (see [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/datasets.py#L58)). As noted in the paper, we did not use small kernels for re-parameterization.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-XL --batch_size 32 --eval true --resume RepLKNet-XL_MegData73M_ImageNet1K.pth --imagenet_default_mean_and_std false --input_size 320
```

To verify the equivalency of Structural Re-parameterization (i.e., the outputs before and after ```structural_reparam```), add ```--with_small_kernel_merged true```.


## Training

You may use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit). Please install:
```
pip install submitit
```
If you have limited GPU memory (e.g., 2080Ti), use ```--use_checkpoint True``` to save GPU memory.

### Pretrain RepLKNet-31B on ImageNet-1K
Single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31B --drop_path 0.5 --batch_size 64 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 10 --epochs 300 --use_checkpoint True --output_dir your_training_dir
```
Four machines:
```
python run_with_submitit.py --nodes 4 --ngpus 8 --model RepLKNet-31B --drop_path 0.5 --batch_size 64 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 10 --epochs 300 --use_checkpoint True --job_dir your_training_dir
```

### Finetune the ImageNet-1K-pretrained (224x224) RepLKNet-31B with 384x384
Single machine:

### Pretrain RepLKNet-31B on ImageNet-22K

### Finetune 22K-pretrained RepLKNet-31B on ImageNet-1K (224x224)

### Finetune 22K-pretrained RepLKNet-31B on ImageNet-1K (384x384)

### Pretrain RepLKNet-31L on ImageNet-22K

### Finetune 22K-pretrained RepLKNet-31L on ImageNet-1K (224x224)

### Finetune 22K-pretrained RepLKNet-31L on ImageNet-1K (384x384)

## Semantic Segmentation

We use MMSegmentation framework. Just clone MMSegmentation, and

1. Put ```segmentation/replknet.py``` into ```mmsegmentation/mmseg/models/backbones/```. The only difference between segmentation/replknet.py and replknet.py is the ```@BACKBONES.register_module```.
2. Add RepLKNet into ```mmsegmentation/mmseg/models/backbones/\_\_init\_\_.py```. That is
  ```
  ...
  from .replknet import RepLKNet
  \_\_all\_\_ = \['ResNet', ..., 'RepLKNet'\]
  ```
3. Put ```segmentation/configs/*.py``` into ```mmsegmentation/configs/replknet/```.
4. Use our released weights.

  Cityscapes:
  ```
  python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=-29500 tools/test.py configs/replknet/RepLKNet-31B_1Kpretrain_upernet_80k_cityscapes_769.py /data/dingxiaohan/replknet_release/RepLKNet-31B_ImageNet-1K_UperNet_Cityscapes.pth --launcher pytorch --eval mIoU
  ```
  
  ADE20K:


## Object Detection

We use MMDetection framework. Just clone MMDetection, and

1. Put ```segmentation/replknet.py``` into ```mmdetection/mmdet/models/backbones/```. The only difference between segmentation/replknet.py and replknet.py is the ```@BACKBONES.register_module```.
2. Add RepLKNet into ```mmdetection/mmdet/models/backbones/\_\_init\_\_.py```. That is
  ```
  ...
  from .replknet import RepLKNet
  \_\_all\_\_ = \['ResNet', ..., 'RepLKNet'\]
  ```
3. Put ```detection/configs/*.py``` into ```mmdetection/configs/replknet/```.
4. Use our released weights.

| backbone | pretraining | method | train schedule | AP_box | AP_mask | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|:---:|:---:|
|RepLKNet-31B | ImageNet-1K | FCOS | 2x | 47.0 | - | 87M | 437G | x |
|RepLKNet-31B | ImageNet-1K | CascMaskRCNN | 3x | 52.2 | 45.2 | 137M | 965G | x |
|RepLKNet-31B | ImageNet-22K | CascMaskRCNN | 3x | 53.0 | 46.0 | 137M | 965G | x |
|RepLKNet-31L | ImageNet-22K | CascMaskRCNN | 3x | 53.9 | 46.5 | 229M | 1321G | x |
|RepLKNet-XL | MegData73M | CascMaskRCNN | 3x | 55.5 | 48.0 | 392M | 1958G | x |


## Acknowledgement
The released PyTorch training script is based on the code of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), which was built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories. 

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.




