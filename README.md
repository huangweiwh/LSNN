# LSNN
The code of "LSNN: A Nonlinear Multi-Scale Pooling Method with a Lifting Scheme-based Unified Representation of Linear and Nonlinear Layers in Deep Networks".

## Requirements
- Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
- Python >= 3.8
- PyTorch >= 1.13.0
- torchvision
- [NVIDIA Apex](https://github.com/NVIDIA/apex)
  
```python
pip install -r requirements.txt
```
## Usage
- train

```sh
python train.py --config ./configs/cifar10/LSML/resnet_ls_fix_attention.yaml
python train.py --config ./configs/cifar10/LSFL/resnet_one_fusion_HL.yaml
```
- test

Before you try to test, you should finish training.

Then you will get the checkpoint, add the checkpoint path to 'test' part of 'config.yaml'.
```sh
python evaluate.py --config ./experiments/cifar10/LSML/resnet_ls_fix_attention/exp00/config.yaml
```
