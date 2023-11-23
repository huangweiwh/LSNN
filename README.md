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
```python
python train.py --config ./configs/cifar10/LSML/resnet_ls_fix_attention.yaml
```
- test
```python
python evaluate.py --config ./experiments/cifar10/LSML/resnet_ls_fix_attention/exp00/config.yaml
```
