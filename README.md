![python](https://img.shields.io/badge/python-3.6+-blue.svg)
![pytorch](https://img.shields.io/badge/pytorch-1.0%2B-brightgreen)
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2004.05571)

# CoCosNet
Pytorch Implementation of the paper ["Cross-domain Correspondence Learning for Exemplar-based Image Translation"](https://panzhang0212.github.io/CoCosNet) (CVPR 2020 oral).

This repo will be updated frequently these days, please check up the progress in the following TODO list, and stay tuned!

![teaser](https://panzhang0212.github.io/CoCosNet/images/teaser.png)

# Environment
- Ubuntu/CentOS
- Pytorch 1.0+
- opencv-python
- tqdm

# TODO list
- [ ] Prepare dataset
- [ ] Implement the network
- [ ] Implement the trainer
- [ ] Training on DeepFashion
- [ ] Training for other tasks

# Dataset Preparation
### DeepFashion

# Pretrained Model
The pretrained model for human pose transfer task: [TO BE RELEASED](https://github.com/Lotayou)

run `sh test_HPT.sh`.

# Training 
run `sh train_HPT.sh`.

# Citations
If you find this repo useful for your research, don't forget to cite the original paper:
```
@article{Zhang2020CrossdomainCL,
  title={Cross-domain Correspondence Learning for Exemplar-based Image Translation},
  author={Pan Zhang and Bo Zhang and Dong Chen and Lu Yuan and Fang Wen},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.05571}
}
```

# Acknowledgement
TODO.
