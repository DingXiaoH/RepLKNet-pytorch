# RepLKNet-pytorch (CVPR 2022)

This is the official PyTorch implementation of **RepLKNet**, from the following CVPR-2022 paper:

Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs.

The paper is now released on arXiv: https://arxiv.org/abs/2203.06717.

Update: **all the pretrained models, ImageNet-1K models, and Cityscapes/ADE20K/COCO models have been released**. 

Update: **released a script to visualize the ERF. To get the ERF of your own model, you only need to add a few lines of code!**

Update: **released the training commands and more examples**. 

If you find the paper or this repository helpful, please consider citing

        @article{replknet,
        title={Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs},
        author={Ding, Xiaohan and Zhang, Xiangyu and Zhou, Yizhuang and Han, Jungong and Ding, Guiguang and Sun, Jian},
        journal={arXiv preprint arXiv:2203.06717},
        year={2022}
        }

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

You may refer to the MegEngine source code: https://github.com/MegEngine/MegEngine/tree/8a2e92bd6c5ac02807b27d174dce090ee391000b/dnn/src/cuda/conv_bias/chanwise.

Pull requests (e.g., better or other implementations or implementations on other frameworks) are welcomed.

## Catalog
- [x] Model code
- [x] PyTorch pretrained models
- [x] PyTorch large-kernel conv impl
- [x] PyTorch training code
- [x] PyTorch downstream models
- [x] PyTorch downstream code
- [x] A script to visualize the ERF
- [x] How to obtain the shape bias

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

| name | resolution |ImageNet-1K acc | #params | FLOPs | MegData-73M pretrained model | 1K finetuned model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
|RepLKNet-XL| 320x320 | 87.8 | 335M | 128.7G | [Google Drive](https://drive.google.com/file/d/1CBHAEUlCzoHfiAQmMIjZhDMAIyHUmAAj/view?usp=sharing), [Baidu](https://pan.baidu.com/s/168Wb2P3rSp23-DCpNG3aCQ?pwd=lknt) | [Google Drive](https://drive.google.com/file/d/1tPC60El34GntXByIRHb-z-Apm4Y5LX1T/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1C1bJJLz9O28ChZ4NcxjUsw?pwd=lknt)|



## Evaluation

For RepLKNet-31B/L with 224x224 or 384x384, we use the "IMAGENET_DEFAULT_MEAN/STD" for preprocessing (see [here](https://github.com/rwightman/pytorch-image-models/blob/73ffade1f8203a611c9cdd6df437b436b780daca/timm/data/constants.py#L2)). For examples,
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31B --batch_size 32 --eval True --resume RepLKNet-31B_ImageNet-1K_224.pth --input_size 224
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31L --batch_size 32 --eval True --resume RepLKNet-31L_ImageNet-22K-to-1K_384.pth --input_size 384
```
For RepLKNet-XL, please note that we used ```mean=[0.5,0.5,0.5]``` and ```std=[0.5,0.5,0.5]``` for preprocessing on MegData73M dataset as well as finetuning on ImageNet-1K. This mean/std setting is also referred to as "IMAGENET_INCEPTION_MEAN/STD" in timm, see [here](https://github.com/rwightman/pytorch-image-models/blob/73ffade1f8203a611c9cdd6df437b436b780daca/timm/data/constants.py#L4). Add ```--imagenet_default_mean_and_std false``` to use this mean/std setting (see [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/datasets.py#L58)). As noted in the paper, we did not use small kernels for re-parameterization.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-XL --batch_size 32 --eval true --resume RepLKNet-XL_MegData73M_ImageNet1K.pth --imagenet_default_mean_and_std false --input_size 320
```

To verify the equivalency of Structural Re-parameterization (i.e., the outputs before and after ```structural_reparam```), add ```--with_small_kernel_merged true```.


## Training

You may use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit). Please install:
```
pip install submitit
```
If you have limited GPU memory (e.g., 2080Ti), use ```--use_checkpoint true``` to save GPU memory.

### Pretrain RepLKNet-31B on ImageNet-1K
Single machine (note ```--update_freq 4```):
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --model RepLKNet-31B --drop_path 0.5 --batch_size 64 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 10 --epochs 300 --output_dir your_training_dir
```
Four machines (note ```--update_freq 1```):
```
python run_with_submitit.py --nodes 4 --ngpus 8 --model RepLKNet-31B --drop_path 0.5 --batch_size 64 --lr 4e-3 --update_freq 1 --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 10 --epochs 300 --job_dir your_training_dir
```
In the following, we only present multi-machine commands. You may train with a single machine in a similar way.

### Finetune the ImageNet-1K-pretrained (224x224) RepLKNet-31B with 384x384
```
python run_with_submitit.py --nodes 4 --ngpus 8 --model RepLKNet-31B --drop_path 0.8 --input_size 384 --batch_size 32 --lr 4e-4 --epochs 30 --weight_decay 1e-8 --update_freq 1 --cutmix 0 --mixup 0 --finetune RepLKNet-31B_ImageNet-1K_224.pth --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 1 --job_dir your_training_dir --layer_decay 0.7
```
### Pretrain RepLKNet-31B on ImageNet-22K
```
python run_with_submitit.py --nodes 16 --ngpus 8 --model RepLKNet-31B --drop_path 0.1 --batch_size 32 --lr 4e-3 --update_freq 1 --warmup_epochs 5 --epochs 90 --data_set image_folder --nb_classes 21841 --disable_eval true --data_path /path/to/imagenet-22k --job_dir /path/to/save_results
```
### Finetune 22K-pretrained RepLKNet-31B on ImageNet-1K (224x224)
```
python run_with_submitit.py --nodes 2 --ngpus 8 --model RepLKNet-31B --drop_path 0.2 --input_size 224 --batch_size 32 --lr 4e-4 --epochs 30 --weight_decay 1e-8 --update_freq 1 --cutmix 0 --mixup 0 --finetune RepLKNet-31B_ImageNet-22K.pth --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 1 --job_dir your_training_dir --layer_decay 0.7
```
### Finetune 22K-pretrained RepLKNet-31B on ImageNet-1K (384x384)
```
python run_with_submitit.py --nodes 4 --ngpus 8 --model RepLKNet-31B --drop_path 0.3 --input_size 384 --batch_size 16 --lr 4e-4 --epochs 30 --weight_decay 1e-8 --update_freq 1 --cutmix 0 --mixup 0 --finetune RepLKNet-31B_ImageNet-22K.pth --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 1 --job_dir your_training_dir --layer_decay 0.7 --min_lr 3e-4
```
### Pretrain RepLKNet-31L on ImageNet-22K
```
python run_with_submitit.py --nodes 16 --ngpus 8 --model RepLKNet-31L --drop_path 0.1 --batch_size 32 --lr 4e-3 --update_freq 1 --warmup_epochs 5 --epochs 90 --data_set image_folder --nb_classes 21841 --disable_eval true --data_path /path/to/imagenet-22k --job_dir /path/to/save_results
```
### Finetune 22K-pretrained RepLKNet-31L on ImageNet-1K (384x384)
```
python run_with_submitit.py --nodes 4 --ngpus 8 --model RepLKNet-31L --drop_path 0.3 --input_size 384 --batch_size 16 --lr 4e-4 --epochs 30 --weight_decay 1e-8 --update_freq 1 --cutmix 0 --mixup 0 --finetune RepLKNet-31L_ImageNet-22K.pth --model_ema true --model_ema_eval true --data_path /path/to/imagenet-1k --warmup_epochs 1 --job_dir your_training_dir --layer_decay 0.7 --min_lr 3e-4
```

## Semantic Segmentation and Object Detection

We use MMSegmentation and MMDetection frameworks. Just clone MMSegmentation or MMDetection, and

1. Put ```segmentation/replknet.py``` into ```mmsegmentation/mmseg/models/backbones/``` or ```mmdetection/mmdet/models/backbones/```. The only difference between ```segmentation/replknet.py``` and ```replknet.py``` is the ```@BACKBONES.register_module```.
2. Add RepLKNet into ```mmsegmentation/mmseg/models/backbones/__init__.py``` or ```mmdetection/mmdet/models/backbones/__init__.py```. That is
  ```
  ...
  from .replknet import RepLKNet
  __all__ = ['ResNet', ..., 'RepLKNet']
  ```
3. Put ```segmentation/configs/*.py``` into ```mmsegmentation/configs/replknet/``` or ```detection/configs/*.py``` into ```mmdetection/configs/replknet/```
4. Download and use our weights. For examples, to evaluate RepLKNet-31B + UperNet on Cityscapes
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/test.py configs/replknet/RepLKNet-31B_1Kpretrain_upernet_80k_cityscapes_769.py RepLKNet-31B_ImageNet-1K_UperNet_Cityscapes.pth --launcher pytorch --eval mIoU
  ```
  or RepLKNet-31B + Cascade Mask R-CNN on COCO:
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/test.py configs/replknet/RepLKNet-31B_22Kpretrain_cascade_mask_rcnn_3x_coco.py RepLKNet-31B_ImageNet-22K_CascMaskRCNN_COCO.pth --eval bbox --launcher pytorch
  ```
5. Or you may finetune our released pretrained weights. For example,
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/replknet/some_config.py --launcher pytorch --options model.backbone.pretrained=some_pretrained_weights.pth
  ```
  
We have released all the Cityscapes/ADE20K/COCO model weights.
 
Single-scale (ss) and multi-scale (ms) mIoU tested with UperNet (FLOPs is computed with 2048×512 for the ImageNet-1K pretrained models and 2560×640 for the 22K and MegData73M pretrained models, following Swin): 
  
| backbone | pretraining | dataset | train schedule | mIoU (ss) | mIoU (ms) | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|:---:|:---:|
|RepLKNet-31B | ImageNet-1K | Cityscapes | 80k  | 83.1 | 83.5 | 110M | 2315G | [Google Drive](https://drive.google.com/file/d/1j3YwToRqTHHi7ocln0iBz1tE6hItD_JO/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1lqrecK4KQUFt0KobFTFKTQ?pwd=lknt) |
|RepLKNet-31B | ImageNet-1K | ADE20K     | 160k | 49.9 | 50.6 | 112M | 1170G | [Google Drive](https://drive.google.com/file/d/1ZV1CP1KzeSdH6_wKw4ytVe9BlCGHf89s/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1xjNzR7Z82iqsrocRBFGLLw?pwd=lknt) |
|RepLKNet-31B | ImageNet-22K| ADE20K     | 160k | 51.5 | 52.3 | 112M | 1829G | [Google Drive](https://drive.google.com/file/d/1W2W4nD2HzTsG_yP9ppLYAqMo3T3JHBNW/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1fJF1FffgbFoRvzBOT3a_gA?pwd=lknt) |
|RepLKNet-31L | ImageNet-22K| ADE20K     | 160k | 52.4 | 52.7 | 207M | 2404G | [Google Drive](https://drive.google.com/file/d/1nrZ723LC3QYjcVHJm8jpOOcecfWMsjxL/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1W4hc2iuUyfB3UH7OWul60g?pwd=lknt) |
|RepLKNet-XL  | MegData73M  | ADE20K     | 160k | 55.2 | 56.0 | 374M | 3431G | [Google Drive](https://drive.google.com/file/d/14GbBI8tdeEl_ECytDCdrfAfNm1McqMv4/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1TYSxbj2Zh_Rfq9-aGM3lyA?pwd=lknt) |

Cascade Mask R-CNN on COCO (FLOPs is computed with 1280x800):

| backbone | pretraining | method | train schedule | AP_box | AP_mask | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|:---:|:---:|
|RepLKNet-31B | ImageNet-1K | FCOS | 2x | 47.0 | - | 87M | 437G | [Google Drive](https://drive.google.com/file/d/1g0jSpONBF2wJDXYhB7P3mbqAVE587Myi/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1VsQC073t8xEw7nzvJ2VHqQ?pwd=lknt) |
|RepLKNet-31B | ImageNet-1K | Cascade Mask RCNN | 3x | 52.2 | 45.2 | 137M | 965G | [Google Drive](https://drive.google.com/file/d/1XqWSkQZSLMIhyaQvaIhGbuFhdDoRJQ4Z/view?usp=sharing), [Baidu](https://pan.baidu.com/s/10uM9ypVwzxhOodOoZzhekQ?pwd=lknt) |
|RepLKNet-31B | ImageNet-22K | Cascade Mask RCNN | 3x | 53.0 | 46.0 | 137M | 965G | [Google Drive](https://drive.google.com/file/d/1faI-MiNuPidum6dC6dGADfYWe6J-bTAz/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1sO55h6GD9x8CxrCxxnFOHg?pwd=lknt) |
|RepLKNet-31L | ImageNet-22K | Cascade Mask RCNN | 3x | 53.9 | 46.5 | 229M | 1321G | [Google Drive](https://drive.google.com/file/d/1qLQONhIjCEuykhdy-wHx1Ah9lT3v2OXh/view?usp=sharing), [Baidu](https://pan.baidu.com/s/10VSqeiKowQlccZaQ_RB9nQ?pwd=lknt) |
|RepLKNet-XL | MegData73M | Cascade Mask RCNN | 3x | 55.5 | 48.0 | 392M | 1958G | [Google Drive](https://drive.google.com/file/d/1i0TqfwQJQUVHWdB5oyDgxlUUqzPZmIVr/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1AVCX86XVBpznstU5E88cyw?pwd=lknt) |

## Tips on the pretraining or finetuning

1. The mean/std values on MegData73M are different from ImageNet. So we used ```mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]``` for pretraining on MegData73M and finetuning on ImageNet-1K. Accordingly, we should let ```img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)``` in MMSegmentation and MMDetection. Please check [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/segmentation/configs/RepLKNet-XL_MegData73M_upernet_160k_ade20k_640.py#L31) and [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/detection/configs/RepLKNet-XL_MegData73Mpretrain_cascade_mask_rcnn_3x_coco.py#L22). For other models, we use the default ImageNet mean/std.
2. For RepLKNet-XL on ADE20K and COCO, we batch-normalize the intermediate feature maps before feeding them into the heads. Just use ```RepLKNet(..., norm_intermediate_features=True)```. For other models, there is no need to do so.
3. For RepLKNet-31B/L on Cityscapes and ADE20K, we used 4 or 8 2080Ti nodes each with 8 GPUs, the batch size per GPU was smaller than the default (the default is 4 per GPU, see [here](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/datasets/ade20k.py#L35)), but the global batch size was larger. Accordingly, we reduced the number of iterations to ensure the same total training examples. Please check the comments in the config files. If you wish to train with our config files, please set the batch size and number of iterations according to your own situation.
4. Lowering the learning rate for lower-level layers may improve the performance when finetuning on ImageNet-1K or downstream tasks, just like ConvNeXt and BeiT. We are not sure if the improvements would be significant. For ImageNet, [our implementation](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/optim_factory.py#L34) simply follows ConvNeXt and BeiT. For MMSegmentation and MMDetection, please raise an issue if you need a showcase, 
5. Tips on the drop_path_rate: bigger model, higher drop_path; bigger pretraining data, lower drop_path.

## Visualizing the Effective Receptive Field

We have released our script to visualize and analyze the Effective Receptive Field (ERF). For example, to automatically download the ResNet-101 from torchvision and obtain the aggregated contribution score matrix,
```
python erf/visualize_erf.py --model resnet101 --data_path /path/to/imagenet-1k --save_path resnet101_erf_matrix.npy
```
Then calculate the high-contribution area ratio and visualize the ERF by
```
python erf/analyze_erf.py --source resnet101_erf_matrix.npy --heatmap_save resnet101_heatmap.png
```
Note this plotting script works with matplotlib 3.3. If you use a higher version of matplotlib, see the comments [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/erf/analyze_erf.py#L40).

To visualize your own model, first define a model that outputs the last feature map rather than the logits (following [this example](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/erf/resnet_for_erf.py#L25)), add the code for building model and loading weights [here](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/erf/visualize_erf.py#L74), then
```
python erf/visualize_erf.py --model your_model --weights /path/to/your/weights --data_path /path/to/imagenet-1k --save_path your_model_erf_matrix.npy
```

## How to obtain the shape bias

1. Install https://github.com/bethgelab/model-vs-human
2. Add your code for building model and loading weights in [this file](https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/pytorch/model_zoo.py). For example
```
@register_model("pytorch")
def replknet(model_name, *args):
    model = ...
    model.load_state_dict(...)
    return model
```
3. Modify examples/evaluate.py (```models = ['replknet']```) and examples/plotting_definition.py (```decision_makers.append(DecisionMaker(name_pattern="replknet", ...))```), following its examples. 


## Acknowledgement
The released PyTorch training script is based on the code of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), which was built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories. 

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.




