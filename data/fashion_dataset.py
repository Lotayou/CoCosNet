"""
    fashion dataset: load deepfashion models
    Requires skeleton input as stick figures.
"""

import random
import numpy as np
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset

class FashionDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        
        # TODO: copy from RATE implementation
        
    def __getitem__(self, opt):
        pass
        
    def __len__(self):
        pass
    