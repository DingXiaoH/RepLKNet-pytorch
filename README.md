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
|RepLKNet-31B|224x224|83.5|    |     |[Google Drive](https://drive.google.com/file/d/1azQUiCxK9feYVkkrPqwVPBtNsTzDrX7S/view?usp=sharing) | Baidu Cloud|
|RepLKNet-31B|384x384|84.8|    |     |[Google Drive](https://drive.google.com/file/d/1vo-P3XB6mRLUeDzmgv90dOu73uCeLfZN/view?usp=sharing) | Baidu Cloud|



### ImageNet-22K Models

| name | resolution |acc | #params | FLOPs | 22K model | 1K model |
|RepLKNet-31B|224x224|    |    |    |[Google Drive](https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1DslZ2voXZQR1QoFY9KnbsHAeF84hzS0s/view?usp=sharing)|
|RepLKNet-31B|384x384|    |    |    | - |[Google Drive](https://drive.google.com/file/d/1Sc46BWdXXm2fVP-K_hKKU_W8vAB-0duX/view?usp=sharing)|




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


