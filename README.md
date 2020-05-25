![python](https://img.shields.io/badge/python-3.6+-blue.svg)
![pytorch](https://img.shields.io/badge/pytorch-1.0%2B-brightgreen)
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2004.05571)

# CoCosNet
Pytorch Implementation of the paper ["Cross-domain Correspondence Learning for Exemplar-based Image Translation"](https://panzhang0212.github.io/CoCosNet) (CVPR 2020 oral).


![teaser](https://panzhang0212.github.io/CoCosNet/images/teaser.png)

### Update:
20200525: Training code for deepfashion complete. Due to the memory limitations, I employed the following conversions:
- Disable the non-local layer, as the memory cost is infeasible on common hardware. If the original paper is telling the truth that the non-lacal layer works on (128-128-256) tensors, then each attention matrix would contain 128^4 elements (which takes 1GB).
- Shrink the correspondence map size from 64 to 32, leading to 4x memory save on dense correspondence matrices.
- Shrink the base number of filters from 64 to 16.

The truncated model barely fits in a 12GB GTX Titan X card, but the performance would not be the same.

# Environment
- Ubuntu/CentOS
- Pytorch 1.0+
- opencv-python
- tqdm

# TODO list
- [x] Prepare dataset
- [x] Implement the network
- [x] Implement the loss functions
- [x] Implement the trainer
- [x] Training on DeepFashion
- [ ] Adjust network architecture to satisfy a single 16 GB GPU.
- [ ] Training for other tasks

# Dataset Preparation
### DeepFashion
Just follow the routine in [the PATN repo](https://github.com/Lotayou/Pose-Transfer)

# Pretrained Model
The pretrained model for human pose transfer task: [TO BE RELEASED](https://github.com/Lotayou)

# Training 
run `python train.py`.

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
