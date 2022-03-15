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

| name | resolution |ImageNet-1K acc | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:|:---:|
|RepLKNet-31B|224x224|83.5| 79M   |  15.3G   |[Google Drive](https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1gspbbfqooMtegt_DO1TUeA?pwd=lknt)|
|RepLKNet-31B|384x384|84.8| 79M   |  45.1G   |[Google Drive](https://drive.google.com/file/d/1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1WhLaCKKv4NuKc3qMYECOIQ?pwd=lknt)|



### ImageNet-22K Models

| name | resolution |ImageNet-1K acc | #params | FLOPs | 22K model | 1K model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
|RepLKNet-31B|224x224|  85.2  |  79M  |  15.3G  |[Google Drive](https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1YiQSn7VJDiNWX1IWg19O6g?pwd=lknt)|[Google Drive](https://drive.google.com/file/d/1DslZ2voXZQR1QoFY9KnbsHAeF84hzS0s/view?usp=sharing), [Baidu](https://pan.baidu.com/s/169wDunCdop-jQM8K-AX27g?pwd=lknt)|
|RepLKNet-31B|384x384|  86.0  |  79M  | 45.1G   | - |[Google Drive](https://drive.google.com/file/d/1Sc46BWdXXm2fVP-K_hKKU_W8vAB-0duX/view?usp=sharing), [Baidu](https://pan.baidu.com/s/11-F3JIKEzSOU7KUhebUWEQ?pwd=lknt)|
|RepLKNet-31L|384x384|  86.6  |  172M  |  96.0G  |[Google Drive](https://drive.google.com/file/d/16jcPsPwo5rko7ojWS9k_W-svHX-iFknY/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1KWazk_cOVYoLuuVJc_9HxA?pwd=lknt)|[Google Drive](https://drive.google.com/file/d/1JYXoNHuRvC33QV1pmpzMTKEni1hpWfBl/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1MWIsPXJJV4mTEtgcKv3F3w?pwd=lknt)|


### MegData-73M Models
(uploading)
| name | resolution |ImageNet-1K acc | #params | FLOPs | MegData-73M model | 1K model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
|RepLKNet-XL| 320x320 | 87.8 | 335M | 128.7G | 



## Evaluation


## Training


## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.




