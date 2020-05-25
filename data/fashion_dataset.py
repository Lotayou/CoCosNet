"""
    fashion dataset: load deepfashion models
    Requires skeleton input as stick figures.
"""

import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
from tqdm import tqdm
import os
from data.base_dataset import BaseDataset
from data.pose_utils import draw_pose_from_cords, load_pose_cords_from_strings

class FashionDataset(BaseDataset):
    # Beware, the pose annotation is fitted for 256*176 images, need additional resizing
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.h = opt.image_size
        self.w = opt.image_size - 2 * opt.padding
        self.size = (self.h, self.w)
        self.pd = opt.padding

        self.white = torch.ones((3, self.h, self.h), dtype=torch.float32)
        self.black = -1 * self.white

        self.dir_Img = os.path.join(opt.dataroot, opt.phase)  # person images (exemplar)
        self.dir_Anno = os.path.join(opt.dataroot, opt.phase + '_pose_rgb')  # rgb pose images

        pairLst = os.path.join(opt.dataroot, 'fasion-resize-pairs-%s.csv' % opt.phase)
        self.init_categories(pairLst)

        if not os.path.isdir(self.dir_Anno):
            print('Folder %s not found or annotation incomplete...' % self.dir_Anno)
            annotation_csv = os.path.join(opt.dataroot, 'fasion-resize-annotation-%s.csv' % opt.phase)
            if os.path.isfile(annotation_csv):
                print('Found backup annotation file, start generating required pose images...')
                self.draw_stick_figures(annotation_csv, self.dir_Anno)


    def trans(self, x, bg='black'):
        x = torch.from_numpy(x / 127.5 - 1).permute(2, 0, 1).float()
        full = torch.ones((3, self.h, self.h), dtype=torch.float32)
        if bg == 'black':
            full = -1 * full

        full[:,:,self.pd:self.pd+self.w] = x
        return full

    def draw_stick_figures(self, annotation, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        with open(annotation, 'r') as f:
            lines = [l.strip() for l in f][1:]

        for l in tqdm(lines):
            name, str_y, str_x = l.split(':')
            target_name = os.path.join(target_dir, name)
            cords = load_pose_cords_from_strings(str_y, str_x)
            target_im, _ = draw_pose_from_cords(cords, self.size)
            cv2.imwrite(target_name, target_im)


    def init_categories(self, pairLst):
        '''
        Using pandas is too f**king slow...

        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)
        '''
        with open(pairLst, 'r') as f:
            lines = [l for l in f][1:self.opt.max_dataset_size+1]
        self.pairs = [l.strip().split(',') for l in lines]
        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        P1_name, P2_name = self.pairs[index]

        P1 = self.trans(cv2.imread(os.path.join(self.dir_Img, P1_name)), bg='white') # person 1
        BP1 = self.trans(cv2.imread(os.path.join(self.dir_Anno, P1_name)), bg='black')  # bone of person 1
        P2 = self.trans(cv2.imread(os.path.join(self.dir_Img, P2_name)), bg='white')  # person 2
        BP2 = self.trans(cv2.imread(os.path.join(self.dir_Anno, P2_name)), bg='black')  # bone of person 2
        # domain x: posemap 
        # domain y: exemplar
        return {'a': BP2, 'b_gt': P2, 'a_exemplar': BP1, 'b_exemplar': P1}

        
    def __len__(self):
        return len(self.pairs)
    
